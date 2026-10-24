"""No-device checks for the U16 path of ``scripts/axera/linear_emit.py``
(``emit_linear_u16``, ``linear_params_u16``) against native Pulsar2 U16 builds.

The builds are local scratch (not committed; up to 28 MB each), from
``AXERA_LINEAR_U16_BUILDS`` (default: the pulsar-free scratch tree). Tests whose
build is missing are skipped. ``tpl/k{K}n{N}_{tpl,held}`` are single-MatMul U16
builds (``layer_configs`` FullyConnected U16) with uniform +-0.1 weights from
``default_rng(1)`` / ``default_rng(2024)`` (``w.npy``) and the same input
calibration; ``{gate,up,down}_u16_fullyconnected`` are SmolLM2-135M layer-0 MLP
linears calibrated on real activations, so their ``zp_x`` differs from the
templates'.
"""

import json
import os
import sys

import numpy as np
import onnx
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "axera"))

import graph_stitch as gs  # noqa: E402
import linear_emit as le  # noqa: E402

BUILDS = os.environ.get(
    "AXERA_LINEAR_U16_BUILDS",
    "/mnt/data/cache/claude-work/tmp/llm-pulsar-free/u16_linears",
)
# npu_params sizes of the native U16 builds
SIZES = {
    (576, 192): 112168,
    (576, 576): 336424,
    (576, 1536): 897064,
    (1536, 576): 946984,
    (576, 49152): 28704868,
    (576, 1728): 1009192,
    (576, 3072): 1794088,
    (768, 768): 648232,
    (768, 3840): 3241000,
    (768, 6144): 5185576,
    (768, 32000): 27008040,
    (3072, 768): 2525224,
}
HELD_SHAPES = [
    (576, 192),
    (576, 576),
    (576, 1536),
    (1536, 576),
    (576, 1728),
    (576, 3072),
    (768, 768),
    (768, 3840),
    (768, 6144),
    (768, 32000),
    (3072, 768),
]
# real-calibration builds -> template shape
REAL = {"gate": (576, 1536), "up": (576, 1536), "down": (1536, 576)}
# compare_models differences that are not content (slot-table order shows up as stream bytes)
NOISE = {"segment stream layout", "blob tail bytes"}


def _path(directory):
    path = os.path.join(directory, "out", "compiled.axmodel")
    if not os.path.exists(path):
        pytest.skip(f"native build not present: {directory}")
    return path


def _tpl(shape, role="tpl"):
    return os.path.join(BUILDS, "tpl", f"k{shape[0]}n{shape[1]}_{role}")


def _params(model) -> bytes:
    return bytes(
        next(i for i in model.graph.initializer if i.name == "npu_params").raw_data
    )


def _quant(directory):
    quant = json.load(
        open(os.path.join(directory, "out", "quant", "quant_axmodel.json"))
    )
    key = next(k for k, v in quant["tensor_configs"].items() if "x" in v)
    cfg, values = quant["tensor_configs"][key], quant["values"]
    v = {t: values[str(cfg[t]["hash"])] for t in ("x", "y")}
    return (
        {t: float(v[t]["scale"][0]) for t in v},
        {t: int(v[t]["zero_point"][0]) for t in v},
    )


def _weights(directory, shape):
    if os.path.exists(os.path.join(directory, "w.npy")):
        return np.load(os.path.join(directory, "w.npy"))
    model = onnx.load(os.path.join(directory, "model.onnx"))
    w = next(
        a
        for a in map(onnx.numpy_helper.to_array, model.graph.initializer)
        if a.ndim == 2
    )
    return w if w.shape == shape else w.T


@pytest.mark.parametrize("shape", list(SIZES))
def test_table_size_matches_the_native_build(shape):
    model = gs.load_model(_path(_tpl(shape)))
    params = _params(model)
    assert len(params) == SIZES[shape]
    assert len(params) - le.u16_table_bytes(*shape) in le.U16_TAILS


@pytest.mark.parametrize("shape", list(SIZES))
def test_linear_params_u16_reproduces_the_template(shape):
    """The template's own weights and calibration give its npu_params back
    (the 49152-wide layer has no held-out build)."""
    directory = _tpl(shape)
    params = _params(gs.load_model(_path(directory)))
    scales, zps = _quant(directory)
    got = le.linear_params_u16(
        params, _weights(directory, shape), scales["x"], zps["x"], scales["y"], zps["y"]
    )
    assert got == params


@pytest.mark.parametrize("shape", HELD_SHAPES)
def test_emit_linear_u16_equals_the_held_build(shape):
    """Other weights and output calibration: npu_params byte-equal, every
    decoded record equal (segment 0's slot table compared as a set)."""
    held = _tpl(shape, "held")
    want = gs.load_model(_path(held))  # skips when the build is absent (CI)
    scales, zps = _quant(held)
    out = le.emit_linear_u16(
        gs.load_model(_path(_tpl(shape))), _weights(held, shape), scales, zps
    )
    assert _params(out) == _params(want)
    assert set(gs.compare_models(out, want)) <= NOISE


@pytest.mark.parametrize("name", list(REAL))
def test_emit_linear_u16_with_another_input_calibration(name):
    """Real-activation calibration (another ``zp_x`` than the template's): the
    core's ``0x03d0`` copy of ``zp_x`` is rewritten too."""
    shape = REAL[name]
    directory = os.path.join(BUILDS, f"{name}_u16_fullyconnected")
    want = gs.load_model(_path(directory))
    scales, zps = _quant(directory)
    template = gs.load_model(_path(_tpl(shape)))
    assert zps["x"] != _quant(_tpl(shape))[1]["x"]
    out = le.emit_linear_u16(template, _weights(directory, shape), scales, zps)
    assert _params(out) == _params(want)
    assert set(gs.compare_models(out, want)) <= NOISE


def test_segments():
    """The general rule at every input width it was checked on (the native
    builds' group bytes and byte-equal emits above)."""
    assert le.u16_segments(576) == [288]
    assert le.u16_segments(768) == [112, 272]
    assert le.u16_segments(1536) == [224, 272, 272]
    assert le.u16_segments(3072) == [176] + [272] * 5
    assert le.u16_segments(1024) == [240, 272]  # Qwen3-0.6B
    assert le.u16_segments(2048) == [208] + [272] * 3
    for k in le.U16_K:
        segs = le.u16_segments(k)
        assert sum(segs) == k // 2 and max(segs) <= le.U16_FIRST_MAX
    with pytest.raises(ValueError, match="positive even"):
        le.u16_segments(577)
    with pytest.raises(ValueError, match="multiple of 32"):
        le.u16_table_bytes(576, 100)


def test_every_weight_code_has_its_own_byte():
    """Each (output, input pair, nibble plane) lands on a distinct table byte,
    and the 256-byte requantization blocks hold the bias and multiplier."""
    for k, n in [(576, 64), (1536, 32)]:
        codes = np.full((n, k), 128, np.uint8)
        bias = np.zeros(n, np.float32)
        m = np.zeros(n, np.float32)
        base = le._u16_table(codes, bias, m)
        # 128 = 0x80: nibble planes give 0x00 (low) and 0x88 (high) for a pair
        assert set(np.unique(base)) == {0x00, 0x88}
        hits = np.zeros(base.size, np.int64)
        for o in range(0, n, 7):
            for i in range(0, k, 13):
                c = codes.copy()
                c[o, i] = 129  # low nibble 1 instead of 0
                diff = np.nonzero(le._u16_table(c, bias, m) != base)[0]
                assert diff.size == 1
                hits[diff] += 1
        assert hits.max() == 1


def test_zero_points_are_checked_before_any_build_bytes():
    # a parsed IO-only model: the zero-point guard runs before the MCode is read
    model = onnx.parser.parse_model(
        '<ir_version: 9, opset_import: ["" : 17]> '
        "g (float[1,576] x) => (float[1,192] y) { y = MatMul(x, w) }"
    )
    w = np.random.default_rng(3).uniform(-0.1, 0.1, (576, 192)).astype(np.float32)
    for zp in (0, 65536):
        with pytest.raises(ValueError, match="1..65535"):
            le.emit_linear_u16(model, w, {"x": 1e-4, "y": 1e-4}, {"x": zp, "y": 100})
    with pytest.raises(ValueError, match="the template is 576x192"):
        le.emit_linear_u16(model, w.T, {"x": 1e-4, "y": 1e-4}, {"x": 1, "y": 1})


def test_gptq_weight_stays_on_the_card_grid():
    """GPTQ weights for ``emit_linear_u16``: ``weight_codes`` recomputes the
    same scales, each channel's largest element is untouched, every other
    element is a code within +-127 times its channel's scale, and the layer's
    output error is below plain rounding's."""
    pytest.importorskip("onnxsim")
    rng = np.random.default_rng(0)
    x = rng.standard_normal((400, 12)) @ rng.standard_normal((12, 256))
    x += 0.05 * rng.standard_normal(x.shape)
    w = (rng.standard_normal((256, 64)) * 0.05).astype(np.float32)
    g = le.gptq_weight(w, x)
    c0, s0 = le.weight_codes(w)
    c, s = le.weight_codes(g)
    assert np.array_equal(s, s0)
    keep = (np.abs(w) == np.abs(w).max(axis=0, keepdims=True)).T
    assert np.array_equal(g.T[keep], w.T[keep])
    assert np.array_equal(c[keep], c0[keep])
    codes = c.astype(np.int64) - 128
    assert (np.abs(codes[~keep]) <= 127).all()
    assert np.allclose(g.T[~keep], (codes * s[:, None])[~keep], rtol=0, atol=1e-7)

    def err(cc):
        q = ((cc.astype(np.float64) - 128) * s0[:, None]).T
        return np.linalg.norm(x @ (q - w)) / np.linalg.norm(x @ w)

    assert err(c) < 0.8 * err(c0)
