"""No-device checks for the non-square families of ``scripts/axera/linear_emit.py``
(``wide``, ``tall576``, ``plain128``) against native Pulsar2 builds.

The builds are local scratch (not committed; a build is up to 165 MB). Their
directory comes from ``AXERA_LINEAR_BUILDS`` (default: the pulsar-free scratch
tree); the held-out wide builds from ``AXERA_LINEAR_HELDOUT``. Tests whose
build is missing are skipped. Each held build is ``out/compiled.axmodel``,
``out/quant/quant_axmodel.json`` (scales) and ``w.npy`` (float weights
``[in, out]``, uniform +-0.1 from ``default_rng(2024)``); a template is a native
build of the same shape from ``default_rng(1)`` weights.
"""

import json
import os
import sys

import numpy as np
import onnx
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
AXERA = os.path.join(HERE, "..", "scripts", "axera")
sys.path.insert(0, AXERA)

import graph_stitch as gs  # noqa: E402
import linear_emit as le  # noqa: E402

PULSAR = os.environ.get(
    "AXERA_LINEAR_BUILDS", "/mnt/data/cache/claude-work/tmp/llm-pulsar-free"
)
HELDOUT = os.environ.get(
    "AXERA_LINEAR_HELDOUT", "/mnt/data/cache/claude-work/tmp/llm-heldout"
)

# held-out shape -> (build directory, template directory or None)
HELD = {
    (576, 192): (f"{PULSAR}/k576n192_held", f"{PULSAR}/k576n192_tpl"),
    (576, 1536): (f"{PULSAR}/k576n1536_held", f"{PULSAR}/k576n1536_tpl"),
    (576, 4096): (f"{PULSAR}/k576n4096_held", f"{PULSAR}/k576n4096_tpl"),
    (576, 49152): (f"{PULSAR}/k576n49152_held", None),
    (768, 768): (f"{PULSAR}/k768n768_held", f"{PULSAR}/k768n768_tpl"),
    (768, 3072): (f"{PULSAR}/k768n3072_held", f"{PULSAR}/k768n3072_tpl"),
    (768, 32000): (f"{PULSAR}/k768n32000_held", None),
    (1536, 576): (f"{PULSAR}/k1536n576_held", f"{PULSAR}/k1536n576_tpl"),
    (3072, 768): (f"{PULSAR}/k3072n768_held", f"{PULSAR}/k3072n768_tpl"),
    (1024, 4096): (f"{PULSAR}/k1024n4096_held", f"{PULSAR}/k1024n4096_tpl"),
    (1024, 151936): (f"{PULSAR}/k1024n151936_held", f"{PULSAR}/k1024n151936_tpl"),
    (128, 128): (f"{PULSAR}/k128n128_held", f"{PULSAR}/k128n128_tpl"),
    (1024, 1024): (f"{HELDOUT}/k1024n1024", None),
    (1024, 2048): (f"{HELDOUT}/k1024n2048", None),
    (1024, 3072): (f"{HELDOUT}/k1024n3072", None),
    (2048, 1024): (f"{HELDOUT}/k2048n1024", None),
    (3072, 1024): (f"{HELDOUT}/k3072n1024", None),
}
# npu_params sizes of the native builds (all 17 held-out shapes)
SIZES = {
    (576, 192): 112168,
    (576, 1536): 897064,
    (576, 4096): 2392104,
    (576, 49152): 28704808,
    (768, 768): 614440,
    (768, 3072): 2457640,
    (768, 32000): 25600040,
    (1536, 576): 921640,
    (3072, 768): 2451496,
    (1024, 4096): 4456488,
    (1024, 151936): 165306468,
    (128, 128): 19496,
    (1024, 1024): 1114152,
    (1024, 2048): 2228264,
    (1024, 3072): 3342376,
    (2048, 1024): 2154536,
    (3072, 1024): 3268648,
}
FAMILY = {
    **{s: "tall576" for s in SIZES if s[0] == 576},
    (128, 128): "plain128",
    **{s: "wide" for s in SIZES if s[0] != 576 and s != (128, 128)},
}
# 1024 x 151936 is refused by emit_linear (four DEQUANT jobs): npu_params only
TEMPLATED = [s for s, (_, t) in HELD.items() if t is not None and s != (1024, 151936)]


def _need(directory):
    path = os.path.join(directory, "out", "compiled.axmodel")
    if not os.path.exists(path):
        pytest.skip(f"native build not present: {directory}")
    return path


def _params(model) -> bytes:
    return bytes(
        next(i for i in model.graph.initializer if i.name == "npu_params").raw_data
    )


def _held(shape):
    """``(params, w, scales, zero points)`` of a held-out build."""
    directory, _ = HELD[shape]
    model = gs.load_model(_need(directory))
    quant = json.load(
        open(os.path.join(directory, "out", "quant", "quant_axmodel.json"))
    )
    key = next(k for k, v in quant["tensor_configs"].items() if "x" in v)
    cfg, values = quant["tensor_configs"][key], quant["values"]
    scales = {t: values[str(cfg[t]["hash"])] for t in ("x", "y")}
    return (
        _params(model),
        np.load(os.path.join(directory, "w.npy")),
        {t: float(scales[t]["scale"][0]) for t in ("x", "y")},
        {t: int(scales[t]["zero_point"][0]) for t in ("x", "y")},
    )


def _template_params(shape) -> bytes:
    _, template = HELD[shape]
    if template is None:
        pytest.skip(f"no template build for {shape}")
    return _params(gs.load_model(_need(template)))


def _tail(shape) -> int:
    return le.layer_params_bytes(*shape) - le._table_bytes(*shape)


@pytest.mark.parametrize("shape", list(SIZES))
def test_npu_params_size_and_family(shape):
    assert le.layer_params_bytes(*shape) == SIZES[shape]
    assert le.family(*shape) == FAMILY[shape]


@pytest.mark.parametrize("shape", list(SIZES))
def test_held_build_has_the_expected_tail(shape):
    # 40 zero bytes; 100 non-zero bytes for 1024 x 151936 (its template's)
    params = _held(shape)[0]
    tail = 100 if shape == (1024, 151936) else 40
    assert _tail(shape) == tail
    assert bool(np.frombuffer(params[-tail:], np.uint8).any()) == (tail == 100)


@pytest.mark.parametrize("shape", list(SIZES))
def test_linear_params_equals_the_held_build(shape):
    """The ``npu_params`` of every held-out build from its own weights; the IO
    tail comes from the template where there is one (else the held build's,
    which is zeros there)."""
    params, w, scales, zps = _held(shape)
    if HELD[shape][1] is not None:
        template = _template_params(shape)
    else:
        template = bytes(le._table_bytes(*shape)) + params[-_tail(shape) :]
    got = le.linear_params(template, w, scales["x"], zps["x"], scales["y"], zps["y"])
    assert len(got) == len(params)
    assert got == params


@pytest.mark.parametrize("shape", TEMPLATED)
def test_emit_linear_from_a_template_equals_the_held_build(shape):
    """Full ``emit_linear`` for the templated shapes: the template's main-engine
    records are rewritten for the held build's scales. Segment 0's LZ77 stream
    length can differ by a byte from the slot-table order (as in the template
    itself); every decoded record matches."""
    params, w, scales, zps = _held(shape)
    template = gs.load_model(_need(HELD[shape][1]))
    out = le.emit_linear(template, w, scales, zps)
    assert _params(out) == params
    want = gs.load_model(_need(HELD[shape][0]))
    issues = gs.compare_models(out, want)
    assert set(issues) <= {"segment stream layout", "blob tail bytes"}, issues


def test_1024_151936_is_refused_by_emit_linear():
    # four DEQUANT jobs (the output is split); npu_params alone is built
    shape = (1024, 151936)
    if HELD[shape][1] is None:
        pytest.skip("no template build")
    template = gs.load_model(_need(HELD[shape][1]))
    _, w, scales, zps = _held(shape)
    with pytest.raises(ValueError, match="main-engine jobs"):
        le.emit_linear(template, w, scales, zps)


def test_unbuilt_shapes_are_refused():
    with pytest.raises(ValueError, match="no layout built"):
        le.family(512, 512)
    with pytest.raises(ValueError, match="no layout built"):
        le.family(576, 100)  # not a multiple of 64
    with pytest.raises(ValueError, match="no layout built"):
        le.family(1024, 100)  # not a multiple of 32


def test_a_template_of_another_shape_is_refused():
    # a parsed IO-only model of width 512: no layout, refused before any bytes
    model = onnx.parser.parse_model(
        '<ir_version: 9, opset_import: ["" : 17]> '
        "g (float[1,512] x) => (float[1,512] y) { y = MatMul(x, w) }"
    )
    with pytest.raises(ValueError, match="no layout built"):
        le.template_shape(model)


@pytest.mark.parametrize("shape", [(128, 128), (768, 768), (576, 192)])
def test_zero_channel_refused_for_families(shape):
    k, n = shape
    w = np.random.default_rng(7).uniform(-0.1, 0.1, (k, n)).astype(np.float32)
    w[:, 3] = 0
    with pytest.raises(ValueError, match="all zero"):
        le.weight_codes(w)
    # a random weight set without a template: the table has the family's size
    w[:, 3] = 1e-3
    codes, s_w = le.weight_codes(w)
    table = le._family_table(codes, s_w, 0.0078, 128, 0.03, 128)
    assert table.size == le._table_bytes(k, n)
