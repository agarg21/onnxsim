"""`scripts/axera/mixed_precision.py`: choosing ops to promote to U16 from a U8
and a U16 Pulsar2 build, checked without a build, Docker or a card.

The tests write the two files a real `pulsar2 build` output directory holds
(`quant/quant_axmodel.json`, `quant/quant_axmodel.onnx`) for a small graph:

    a = Mul(x, k)            op M
    r = Reshape(a)           (data movement)
    b = Abs(r)               op B   reads a (through the Reshape)
    c = Sigmoid(a)           op C   reads a too
    y = Add(b, c)            op Y

so `a` has two readers and only widens when both are promoted, the rule the
card showed (a promoted op's operand that an 8-bit op also reads stays
8-bit).
"""

import json
import os
import sys

import numpy as np
import onnx
import pytest
from onnx import numpy_helper, parser

_AXERA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "axera"
)
if _AXERA_DIR not in sys.path:
    sys.path.insert(0, _AXERA_DIR)

import mixed_precision as mp  # noqa: E402

TENSORS = ("x", "a", "r", "b", "c", "y")
# op -> tensors listed under it in quant_axmodel.json (inputs and outputs)
LISTED = {
    "M": ("x", "a"),
    "B": ("r", "b"),
    "C": ("a", "c"),
    "Y": ("b", "c", "y"),
}


def _float_model():
    m = parser.parse_model(
        """
        <ir_version: 8, opset_import: ["" : 17]>
        g (float[1, 8] x) => (float[1, 8] y)
        <float k = {3.0}, int64[2] shape = {1, 8}>
        {
            a = Mul(x, k)
            r = Reshape(a, shape)
            b = Abs(r)
            c = Sigmoid(a)
            y = Add(b, c)
        }
        """
    )
    onnx.checker.check_model(m)
    return m


def _named_quant_graph():
    # The compiler's lowered graph: same tensor names, Ax op types (no
    # schema, so never checked), named as quant_axmodel.json names its ops.
    g = parser.parse_model(
        """
        <ir_version: 8, opset_import: ["" : 17]>
        g (float[1, 8] x) => (float[1, 8] y)
        <int64[2] shape = {1, 8}>
        {
            a = AxQuantizedMul(x)
            r = AxReshape(a, shape)
            b = AxQuantizedAbs(r)
            c = AxQuantizedSigmoid(a)
            y = AxQuantizedAdd(b, c)
        }
        """
    )
    for n in g.graph.node:
        n.name = {"a": "M", "r": "R", "b": "B", "c": "C", "y": "Y"}[n.output[0]]
    return g


def _write_build(tmp_path, scales, bits):
    """`scales`: {tensor: scale}; `bits`: 8, 16, or {tensor: bits}; zero
    point at mid-range."""
    quant = tmp_path / "quant"
    quant.mkdir(parents=True)
    values, configs = {}, {}
    width = bits if isinstance(bits, dict) else {t: bits for t in TENSORS}
    for op, tensors in LISTED.items():
        configs[op] = {}
        for t in tensors:
            bits = width[t]
            h = str(len(values) + 1)
            values[h] = {"scale": [scales[t]], "zero_point": [2 ** (bits - 1)]}
            configs[op][t] = {
                "bit_width": bits,
                "policy": {"PER_CHANNEL": False, "SYMMETRICAL": False},
                "state": "ACTIVATED",
                "hash": int(h),
                "quant_min": 0,
                "quant_max": 2**bits - 1,
            }
    (quant / "quant_axmodel.json").write_text(
        json.dumps({"tensor_configs": configs, "values": values})
    )
    onnx.save(_named_quant_graph(), str(quant / "quant_axmodel.onnx"))
    return str(tmp_path)


def _builds(tmp_path):
    # U8: `b` (|3x|, up to ~9) gets a scale 40x too coarse -- the outlier the
    # search must find; everything else is fine at 8 bits.
    lo = {t: 0.05 for t in TENSORS}
    lo["b"] = 2.0
    lo["r"] = lo["a"] = lo["y"] = 0.1
    hi = {t: 0.0005 for t in TENSORS}
    return (
        _write_build(tmp_path / "u8", lo, 8),
        _write_build(tmp_path / "u16", hi, 16),
    )


def test_build_reads_ops_readers_and_layers(tmp_path):
    lo_dir, _ = _builds(tmp_path)
    b = mp.Build(lo_dir)
    assert [o.name for o in b.ops] == ["M", "B", "C", "Y"]
    reads = {o.name: o.tensors for o in b.ops}
    # B reads `a` through the Reshape: the real tensor, not the movement alias
    assert reads == {"M": ["x"], "B": ["a"], "C": ["a"], "Y": ["b", "c"]}
    assert b.consumers["a"] == {"B", "C"}
    assert b.aliases["a"] == {"a", "r"}


def test_a_shared_operand_widens_only_with_every_reader(tmp_path):
    lo_dir, _ = _builds(tmp_path)
    b = mp.Build(lo_dir)
    op = {o.name: o for o in b.ops}
    assert b.widened([op["B"]]) == set()  # C still reads `a` at 8 bits
    assert b.widened([op["B"], op["C"]]) == {"a", "r"}
    assert b.widened([op["Y"]]) == {"b", "c"}  # Y's output stays 8-bit


def test_tensor_groups_and_layer_configs(tmp_path):
    lo_dir, _ = _builds(tmp_path)
    b = mp.Build(lo_dir)
    groups = mp.group(b, "tensor")
    assert {t: [o.name for o in ops] for t, ops in groups.items()} == {
        "x": ["M"],
        "a": ["B", "C"],
        "b": ["Y"],
        "c": ["Y"],
    }
    assert mp.layer_configs(groups["a"]) == [
        {"layer_names": ["B", "C"], "data_type": "U16"}
    ]
    assert mp.layer_configs([]) == []


def test_rank_and_greedy_find_the_outlier(tmp_path):
    pytest.importorskip("onnxruntime")
    from replay import load_scales

    lo_dir, hi_dir = _builds(tmp_path)
    b = mp.Build(lo_dir)
    lo, hi = (
        load_scales(lo_dir, fused_aware=False),
        load_scales(hi_dir, fused_aware=False),
    )
    rng = np.random.default_rng(0)
    feeds = [{"x": rng.uniform(-3, 3, (1, 8)).astype(np.float32)} for _ in range(8)]
    rp = mp.Replayer(_float_model(), feeds)

    ranked = mp.rank(rp, b, lo, hi, mp.group(b, "tensor"))
    assert ranked[0].group == "b"
    assert ranked[0].gain > 10

    sel = mp.greedy(rp, b, lo, hi, ranked, target=rp.score(lo) + 10)
    assert sel.groups == ["b"]
    assert [o.name for o in sel.ops] == ["Y"]
    assert mp.layer_configs(sel.ops) == [{"layer_names": ["Y"], "data_type": "U16"}]

    # a budget of zero groups selects nothing
    none = mp.greedy(rp, b, lo, hi, ranked, target=1e9, max_groups=0)
    assert none.ops == [] and none.score == pytest.approx(rp.score(lo))


def test_spliced_scales_only_replaces_named_tensors():
    lo = {"a": (8, 1.0), "b": (8, 2.0)}
    hi = {"a": (16, 0.1), "b": (16, 0.2)}
    assert mp.spliced_scales(lo, hi, {"a", "zz"}) == {"a": (16, 0.1), "b": (8, 2.0)}
    assert lo == {"a": (8, 1.0), "b": (8, 2.0)}


def test_float_model_is_valid():
    m = _float_model()
    assert [n.op_type for n in m.graph.node] == [
        "Mul",
        "Reshape",
        "Abs",
        "Sigmoid",
        "Add",
    ]
    assert numpy_helper.to_array(m.graph.initializer[0]).item() == 3.0


def test_search_adds_the_best_op_type_group(tmp_path):
    pytest.importorskip("onnxruntime")
    from replay import load_scales

    lo_dir, hi_dir = _builds(tmp_path)
    b = mp.Build(lo_dir)
    lo = load_scales(lo_dir, fused_aware=False)
    hi = load_scales(hi_dir, fused_aware=False)
    rng = np.random.default_rng(0)
    feeds = [{"x": rng.uniform(-3, 3, (1, 8)).astype(np.float32)} for _ in range(8)]
    rp = mp.Replayer(_float_model(), feeds)

    sel = mp.search(rp, b, lo, hi, mp.group(b, "op_type"), slack_db=3.0)
    assert sel.groups[0] == "AxQuantizedAdd"  # Y reads the coarse `b`
    assert sel.score >= rp.score(mp.spliced_scales(lo, hi, b.widened(b.ops))) - 3.0
    assert sel.trace[0][0] == "<U8>"


def test_dev_pairs_never_contain_the_eval_pairs():
    import encoder_classifier as ec

    for held_out in (ec.EVAL_PAIRS, ec.NLI_PAIRS):
        dev = ec.dev_pairs(held_out, n=500)
        assert len(dev) == len(set(dev)) > 64
        assert not {(a, b) for a, b, *_ in held_out} & set(dev)
    assert ec.dev_pairs(ec.NLI_PAIRS) == ec.dev_pairs(ec.NLI_PAIRS)


# --------------------------------------------------------------------------
# auto(): the end-to-end loop with a fake compiler and a fake card. The fake
# card replays each build's own table (what replay.py shows the real card
# does); the fake compiler widens exactly what `Build.widened()` says.
# --------------------------------------------------------------------------

LO = {**{t: 0.05 for t in TENSORS}, "b": 2.0, "r": 0.1, "a": 0.1, "y": 0.1}
# the graph input and output keep their 8-bit grids at 16 bits too (as a
# compiler that widens only interior tensors would); the rest are fine
HI = {**{t: 0.0005 for t in TENSORS}, "x": LO["x"], "y": LO["y"]}


def _fake_builder(tmp_path, honest=True):
    calls = []

    def build(root, cfgs):
        calls.append(cfgs)
        out = os.path.join(root, "out")
        if os.path.exists(out):
            return mp.BuildOut(out, 0.0, True)
        from pathlib import Path

        if cfgs is None:
            _write_build(Path(out), LO, 8)
        elif "op_types" in cfgs[0]:
            _write_build(Path(out), HI, 16)
        else:
            u8 = mp.Build(os.path.join(os.path.dirname(root), "U8", "out"))
            names = set(cfgs[0]["layer_names"])
            wide = (
                u8.widened([o for o in u8.ops if o.name in names]) if honest else set()
            )
            _write_build(
                Path(out),
                {t: HI[t] if t in wide else LO[t] for t in TENSORS},
                {t: 16 if t in wide else 8 for t in TENSORS},
            )
        open(os.path.join(out, "compiled.axmodel"), "wb").close()
        return mp.BuildOut(out, 1.0, False)

    return build, calls


class _FakeCard:
    """AXSession's surface card_run() uses; latency 1 ms + 0.1 ms per 16-bit
    tensor."""

    def __init__(self, model, feeds):
        self.rp = mp.Replayer(model, feeds)
        self.exec_us = self.runs = 0
        self.inputs = [type("IO", (), {"name": "x"})()]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def load(self, axmodel):
        from replay import load_scales

        d = os.path.dirname(axmodel)
        self.table = load_scales(d, fused_aware=False)
        self.us = 1000 + 100 * mp.count_16bit(d)
        return self

    def unload(self, m):
        pass

    def run(self, m, inputs):
        self.exec_us += self.us
        self.runs += 1
        from replay import insert_qdq

        q, _, _ = insert_qdq(self.rp.model, self.table)
        return [self.rp._run_one(q, {"x": inputs[0]})]


def _auto(tmp_path, honest=True, **kw):
    pytest.importorskip("onnxruntime")
    model = _float_model()
    path = str(tmp_path / "m.onnx")
    onnx.save(model, path)
    rng = np.random.default_rng(0)
    feeds = [{"x": rng.uniform(-3, 3, (1, 8)).astype(np.float32)} for _ in range(6)]
    builder, calls = _fake_builder(tmp_path, honest)
    rep = mp.auto(
        path,
        feeds,
        feeds,
        str(tmp_path / "work"),
        builder=lambda root, cfgs: builder(root, cfgs),
        session_factory=lambda: _FakeCard(model, feeds),
        runs=2,
        log=lambda *_: None,
        **kw,
    )
    return rep, calls


def test_auto_picks_the_mix_and_writes_the_report(tmp_path):
    rep, calls = _auto(tmp_path)
    labels = [c["label"] for c in rep["candidates"]]
    assert labels[:2] == ["U8", "U16"]
    assert rep["chosen"].startswith("mix:Add")  # Add holds the outlier `b`
    assert "Y" in rep["layer_configs"][0]["layer_names"]
    chosen = rep["candidates"][-1]
    u16 = rep["candidates"][1]
    assert chosen["accepted"] and chosen["label"] == rep["chosen"]
    assert chosen["tensors_16bit"] < u16["tensors_16bit"]
    assert chosen["latency_ms"] < u16["latency_ms"]
    assert chosen["card_db"] >= rep["budget"]["snr_floor_db"]
    assert abs(chosen["card_db"] - chosen["predicted_db"]) < 0.5
    md = open(tmp_path / "work" / "auto_report.md").read()
    assert "**chosen**" in md and rep["chosen"] in md
    saved = json.load(open(tmp_path / "work" / "auto_report.json"))
    assert saved["chosen"] == rep["chosen"]


def test_auto_rerun_reuses_the_builds(tmp_path):
    _auto(tmp_path)
    rep, _ = _auto(tmp_path)
    assert all(c["cached"] for c in rep["candidates"])


def test_auto_falls_back_when_the_card_misses(tmp_path):
    # a compiler that ignores the promotion: every mix measures like U8, so
    # each candidate misses and the flow ends on whole-model U16
    rep, _ = _auto(tmp_path, honest=False, max_candidates=2)
    tried = [c for c in rep["candidates"] if c["label"].startswith("mix:")]
    assert tried and all(not c["accepted"] for c in tried)
    assert all("next prefix" in c["note"] for c in tried)
    assert rep["chosen"] == "U16"


def test_auto_respects_the_latency_budget(tmp_path):
    # U16 (6 tensors -> 1.6 ms) and the mix (2 -> 1.2 ms) both over 1.1 ms
    rep, _ = _auto(tmp_path, max_latency_ms=1.1)
    assert rep["chosen"] is None
    mix = [c for c in rep["candidates"] if c["label"].startswith("mix:")]
    assert mix and "latency" in mix[0]["note"]


def test_auto_accepts_u8_when_it_already_meets_the_budget(tmp_path):
    rep, calls = _auto(tmp_path, max_err=10.0)
    assert rep["chosen"] == "U8"
    assert len(calls) == 2  # U8 and U16 only
