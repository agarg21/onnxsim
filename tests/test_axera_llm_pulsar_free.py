"""Host-only checks of ``scripts/axera/llm_pulsar_free`` (the Pulsar-free U16
decode for the AX650N): the plan layout, the fake-quant evaluator, and the
whole plan wiring on synthetic models in ``FakeAX`` against the float model.

No card, no Pulsar2, no checkpoint: the models are
``llm_layer_decomposition.make_synthetic``'s random tiny llama / qwen3
layouts, and the programs float stand-ins (``pf_plan.write_float_set``).
"""

import os
import sys

import numpy as np
import pytest
from onnx.reference import ReferenceEvaluator

HERE = os.path.dirname(os.path.abspath(__file__))
PF = os.path.join(HERE, "..", "scripts", "axera", "llm_pulsar_free")
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "axera"))
sys.path.insert(0, PF)

import llm_layer_decomposition as ld  # noqa: E402
import pf_estimate  # noqa: E402
import pf_fake_quant as fqm  # noqa: E402
import pf_layout as lay  # noqa: E402
import pf_model  # noqa: E402
import pf_plan  # noqa: E402
import pf_templates as tpl  # noqa: E402

F32 = np.float32


def _softmax(a):
    e = np.exp(a - a.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


@pytest.mark.parametrize("p", [0, 1, 5, 127])
def test_plan_mask_attends_to_the_same_rows(p):
    """The plan layout (new row in slot p) and layer_graph's (new row at S)
    give the same attention output."""
    rng = np.random.default_rng(p)
    S, d = lay.S, 8
    q = rng.standard_normal((1, d)).astype(F32)
    rows = rng.standard_normal((p + 1, 2, d)).astype(F32)  # (k, v) of positions 0..p
    Kp, Vp = np.zeros((2, S + 1, d), F32)
    Kp[: p + 1], Vp[: p + 1] = rows[:, 0], rows[:, 1]
    Kr, Vr = np.zeros((2, S + 1, d), F32)
    Kr[:p], Vr[:p] = rows[:p, 0], rows[:p, 1]
    Kr[S], Vr[S] = rows[p, 0], rows[p, 1]
    yp = _softmax(q @ Kp.T + lay.mask_plan(p)) @ Vp
    yr = _softmax(q @ Kr.T + lay.mask_ref(p)) @ Vr
    np.testing.assert_allclose(yp, yr, rtol=1e-5, atol=1e-6)


def test_qkv_layout():
    """With a per-head k norm, k_proj and k_rot are one contiguous slice."""
    parts = lay.qkv_parts(16, 128, 1024, qk_norm=True)
    o = lay.offsets(parts)
    assert o["k_rot"] == o["k_proj"] + 1024
    assert sum(n for _, n in parts) == 2 * 2048 + 3 * 1024
    assert (
        lay.layer_programs(True).index("kc") == lay.layer_programs(True).index("rk") - 1
    )
    assert "kc" not in lay.layer_programs(False)


DIMS = tpl.Dims(H=32, I=64, nh=4, nkv=2, hd=16, eps=1e-6, vocab=64)


@pytest.mark.parametrize(
    "name",
    [
        "crms32",
        "rms32",
        "rope64",
        "qkrope64",
        "qkrope32t",
        "hcrms64",
        "attn64",
        "silu64",
        "nrm32",
        "anorm32",
        "qkf64",
        "lanorm64",
        "slanorm64",
    ],
)
def test_fused_eval_float_matches_reference(name):
    """With no calibration FusedEval is the float graph (every output of a
    merged program, in graph order); with one it rounds every tensor to its
    U16 grid (close to float, not equal)."""
    _, _, ins, _, _, gens = tpl.graph(name, DIMS)
    m = tpl.float_graph(name, DIMS)
    feeds = gens[0](np.random.default_rng(0))
    wants = ReferenceEvaluator(m).run(None, feeds)
    gots = fqm.FusedEval(m, {})(**feeds)
    gots = gots if isinstance(gots, tuple) else (gots,)
    assert len(gots) == len(wants) == len(m.graph.output)
    for got, want in zip(gots, wants):
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)
    want = wants[-1]
    res = ReferenceEvaluator(m).run(None, feeds, intermediate=True)
    cal = {}
    for k, v in res.items():
        a = np.asarray(v)
        if a.dtype.kind == "f" and k not in {i.name for i in m.graph.initializer}:
            lo, hi = min(float(a.min()), 0.0), max(float(a.max()), 0.0)
            s = (hi - lo) / 65535 or 1.0
            cal[k] = (s, int(np.rint(-lo / s)))
    got = fqm.FusedEval(m, cal)(**feeds)
    got = got[-1] if isinstance(got, tuple) else got
    assert lay.rel_err(got, want) < 1e-2


def test_fq_grid():
    a = np.array([-1.0, 0.0, 0.3, 2.0])
    np.testing.assert_allclose(fqm.fq(a, (0.5, 2)), [-1.0, 0.0, 0.5, 2.0])
    assert fqm.fq(np.array([1e9]), (1.0, 0), signed=True)[0] == 32767


def test_split_weights_sum_to_w():
    w = np.random.default_rng(3).standard_normal((48, 32)).astype(F32)
    q, r = pf_model.Config.split_weights(w)
    np.testing.assert_allclose(q + r, w, atol=1e-6)
    assert np.abs(r).max() < np.abs(w).max() / 127
    codes = q / (np.abs(w).max(axis=0) / 127.5)
    np.testing.assert_allclose(codes, np.rint(codes), atol=1e-3)


def test_fake_ax_plan_lines(tmp_path):
    """FakeAX runs W / X / C / R lines over its buffers as the card's runner does."""
    import onnx

    ax = lay.FakeAX(str(tmp_path))
    path = str(tmp_path / "add.onnx")
    onnx.save(tpl.add_model(4), path)
    m = ax.load(path, lambda f: f["a"] + f["b"])
    assert [i.name for i in m.inputs] == ["a", "b"] and m.outputs[0].nbytes == 16
    b = int(ax._cmd("BUF 48").split()[1])
    y = int(ax._cmd("BUF 16").split()[1])
    np.arange(12, dtype=F32).tofile(tmp_path / "t" / "in.bin")
    for line in (f"BIND 0 I 0 {b} 0", f"BIND 0 I 1 {b} 16", f"BIND 0 O 0 {y} 0"):
        assert ax._cmd(line) == "OK"
    plan = tmp_path / "t" / "p.txt"
    plan.write_text(
        f"W {b} 0 48 {tmp_path}/t/in.bin\nX 0\nC {b} 32 {y} 0 16\n"
        f"R {b} 0 48 {tmp_path}/t/out.bin\n"
    )
    assert ax._cmd(f"PLAN {plan}").startswith("OK")
    out = np.fromfile(tmp_path / "t" / "out.bin", F32)
    np.testing.assert_array_equal(out[8:], [4, 6, 8, 10])


def _config(name, out, fuse=0):
    model = ld.make_synthetic(name)
    if (
        model.qk_norm
    ):  # Qwen3-like: position 0 first, shared linears, split MLP at layer 0
        return pf_model.Config(
            model,
            out=out,
            first_pos=0,
            share_linears=True,
            split_first=dict(layers=[0], keys=["gu", "down_proj"]),
            fuse=fuse,
        )
    return pf_model.Config(model, out=out, first_pos=1, fuse=fuse)


WIRING = [("qwen3_like", 0), ("llama_gqa", 0), ("llama_mha", 0)]
WIRING += [("qwen3_like", f) for f in (1, 2, 3, 4)]


@pytest.mark.parametrize("name,fuse", WIRING)
def test_plan_wiring_matches_float_model(name, fuse, tmp_path):
    """The plan (buffers, binds, merged-linear slices, K row rebinding, V row
    copies, the two program sets, sharing and split linears; with ``fuse``
    the merged programs and their several outputs) with float programs
    reproduces the float model step by step; so does the host chain
    pf_estimate.Chain with every program exact."""
    cfg = _config(name, str(tmp_path), fuse)
    root = str(tmp_path / "float_set")
    pf_plan.write_float_set(cfg, root)
    plan = pf_plan.Plan(lay.FakeAX(str(tmp_path / "fake")), cfg, fake=True, root=root)
    ids = [int(t) for t in np.random.default_rng(5).integers(1, cfg.vocab, 6)]
    rows = pf_plan.decode(plan, cfg, ids)
    assert len(rows) == 6
    assert max(r["rel"] for r in rows) < 1e-4
    assert all(r["top1"] == r["float_top1"] for r in rows)
    chains = {
        st: pf_estimate.Chain(cfg, st, exact=True, root=root)
        for st in ("first", "rest")
    }
    est = pf_estimate.run(cfg, ids, chains)
    assert max(r["rel"] for r in est) < 1e-4


def test_template_names():
    cfg = _config("qwen3_like", ".")
    names = tpl.needed(cfg)
    assert {
        "crms32",
        "rms32",
        "qkrope64",
        "qkrope32",
        "qkrope64t",
        "hcrms64",
        "attn64",
    } <= set(names)
    assert "add128" in names and "k64n32" in names  # split gu Add; down_proj
    assert tpl.template_dir("r", "k32n64").endswith(os.path.join("linear", "k32n64"))
    assert tpl.template_dir("r", "add32").endswith(os.path.join("single", "add32"))


@pytest.mark.parametrize("fuse,runs", [(0, 15), (1, 9), (2, 7), (3, 6), (4, 4)])
def test_programs_per_layer(fuse, runs):
    """Executes per layer of the rest set; layer 0 also runs the first norm,
    and a split first-set ``down_proj`` keeps its own programs."""
    cfg = _config("qwen3_like", ".", fuse)
    assert len(cfg.programs("rest", "L1")) == runs
    if fuse == 4:
        assert cfg.programs("rest", "L0")[0] == "n1q"
        assert cfg.programs("first", "L0")[-5:] == [
            "on2",
            "gu",
            "silu_f",
            "down_proj",
            "dn",
        ]  # split
        assert cfg.programs("first", "L1")[0] == "qkv5"
        assert cfg.programs("rest", f"L{cfg.L - 1}")[-1] == "dn"
    elif fuse >= 2:
        assert cfg.programs("rest", "L0")[0] == "n1"
        assert "n1" not in cfg.programs("rest", "L1")
        first = cfg.programs("first", "L0")
        assert first.index("down_proj") == first.index("dn") - 1
        assert "silu_f" in first
        assert ("silu_f" in cfg.programs("rest", "L0")) == (fuse == 2)
        assert cfg.programs("rest", "post") == ["lm_head"]


def test_merged_template_names():
    for fuse, want in ((1, {"nrm32", "anorm32", "qkf64"}), (3, {"slanorm64"})):
        names = set(tpl.needed(_config("qwen3_like", ".", fuse)))
        assert want <= names and "crms32" not in names
    assert {"lanorm64"} <= set(tpl.needed(_config("qwen3_like", ".", 2)))


def test_signed_inputs_skip_a_constant_weight():
    """A MatMul by a constant is a FullyConnected: its input stays unsigned."""
    import pf_generate

    assert pf_generate.signed_inputs(tpl.float_graph("lanorm64", DIMS)) == set()
    assert pf_generate.signed_inputs(tpl.float_graph("attn64", DIMS)) >= {"qh", "kt"}


def test_tied_ranges_follow_the_template():
    """Tensors at one template calibration share one range in the target;
    a tie that would widen a range past TIE_MAX refuses the template."""
    import pf_generate

    g = pf_generate.Generator.__new__(pf_generate.Generator)
    m = tpl.float_graph("nrm32", DIMS)
    rg = {"u": [-2.0, 3.0], "d": [-1.0, 1.5], "x": [-50.0, 60.0]}
    tq = {"u": (1e-4, 30000), "d": (1e-4, 30000), "x": (2e-3, 31000)}
    out = g._tied(rg, m, tq)
    assert out["u"] == out["d"] == [-2.0, 3.0]
    tq["x"] = (1e-4, 30000)
    with pytest.raises(ValueError, match="widen"):
        g._tied(rg, m, tq)
