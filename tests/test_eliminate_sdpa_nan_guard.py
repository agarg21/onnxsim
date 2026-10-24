"""Tests for the ``eliminate_sdpa_nan_guard`` C++ pass
(onnxsim/passes/eliminate_sdpa_nan_guard.h).

PyTorch's ``scaled_dot_product_attention`` export follows the attention
Softmax with ``Where(IsNaN(P), 0, P)``, which zeroes the NaN row a fully -inf
masked row produces. Backends without ``IsNaN`` (Axera's Pulsar2) reject it.
The pass, opt-in via ``extra_optimizers=["eliminate_sdpa_nan_guard"]``,
removes it when the mask is provably finite (the dynamo exporter's finfo.min
mask), and for the legacy exporter's ``Where(c, 0, -inf)`` mask rewrites -inf
to finfo.min plus an explicit row-validity multiply, so an all-padding row
still comes out as zeros. Each test compares onnxruntime's outputs before and
after, padded and all-padding rows included.
"""

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

ort = pytest.importorskip("onnxruntime")

PASS = "eliminate_sdpa_nan_guard"
B, H, S, D = 3, 2, 6, 4
FINFO_MIN = float(np.finfo(np.float32).min)


def _model(body, inputs, opset=17, out=f"float[{B},{H},{S},{D}] out"):
    model = parser.parse_model(
        f"""
        <ir_version: 8, opset_import: ["" : {opset}]>
        sdpa ({inputs}) => ({out})
        {{
            {body}
        }}
        """
    )
    onnx.checker.check_model(model)
    return model


def _guarded(mask_expr, softmax_axis=-1, scores_first=True):
    """Q@K^T * scale (+ mask) -> Softmax -> Where(IsNaN(P), 0, P) -> @V."""
    add = "Add(scores, mask)" if scores_first else "Add(mask, scores)"
    if mask_expr is None:  # unmasked
        mask_expr, add = "", "Identity(scores)"
    return f"""
        kt = Transpose <perm=[0,1,3,2]> (k)
        qk = MatMul(q, kt)
        scale = Constant <value = float {{0.5}}> ()
        scores = Mul(qk, scale)
        {mask_expr}
        logits = {add}
        p = Softmax <axis={softmax_axis}> (logits)
        nan = IsNaN(p)
        zero = Constant <value = float {{0.0}}> ()
        guarded = Where(nan, zero, p)
        out = MatMul(guarded, v)
    """


QKV = f"float[{B},{H},{S},{D}] q, float[{B},{H},{S},{D}] k, float[{B},{H},{S},{D}] v"
# Padding mask over the keys, [B, 1, 1, S]: row 0 unpadded, row 1 padded from
# position 3, row 2 all padding (the case the guard exists for).
KEEP = np.ones((B, 1, 1, S), dtype=bool)
KEEP[1, ..., 3:] = False
KEEP[2] = False


def _feeds(seed=0, **extra):
    rng = np.random.default_rng(seed)
    f = {n: rng.standard_normal((B, H, S, D)).astype(np.float32) for n in "qkv"}
    f.update(extra)
    return f


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    names = {i.name for i in sess.get_inputs()}
    return sess.run(None, {k: v for k, v in feeds.items() if k in names})[0]


def _simplify(model, feeds, **kw):
    out, ok = onnxsim.simplify(
        model,
        check_n=1,
        input_data=feeds,
        attention_fusion_backend="none",
        extra_optimizers=[PASS],
        **kw,
    )
    assert ok
    return out


def _ops(model):
    return [n.op_type for n in model.graph.node]


def _has_value(model, pred):
    for t in model.graph.initializer:
        a = onnx.numpy_helper.to_array(t)
        if a.dtype == np.float32 and pred(a).any():
            return True
    for n in model.graph.node:
        if n.op_type == "Constant":
            a = onnx.numpy_helper.to_array(n.attribute[0].t)
            if a.dtype == np.float32 and pred(a).any():
                return True
    return False


def _assert_matches_guarded(got, want):
    # Rows with a kept key must match bit for bit. The all-padding batch row
    # (2) must be the guard's zeros; it is checked against 0 rather than the
    # reference because onnxruntime's Softmax over an all -inf row is
    # platform-dependent (NaN on x86-64, not NaN on aarch64, where the guard
    # then lets it through).
    np.testing.assert_array_equal(got[:2], want[:2])
    np.testing.assert_array_equal(got[2], 0)


@pytest.mark.parametrize("scores_first", [True, False])
def test_finite_finfo_min_mask_drops_guard(scores_first):
    # The dynamo exporter's form: Where(keep, 0, finfo.min).
    mask = f"""
        mzero = Constant <value = float {{0.0}}> ()
        mmin = Constant <value = float {{{FINFO_MIN!r}}}> ()
        mask = Where(keep, mzero, mmin)
    """
    model = _model(
        _guarded(mask, scores_first=scores_first), f"{QKV}, bool[{B},1,1,{S}] keep"
    )
    feeds = _feeds(keep=KEEP)
    simp = _simplify(model, feeds)
    assert "IsNaN" not in _ops(simp)
    assert "Mul" not in _ops(simp)[_ops(simp).index("Softmax") :]
    # finfo.min mask: an all-padding row is uniform both before and after.
    np.testing.assert_array_equal(_run(simp, feeds), _run(model, feeds))


def test_unmasked_attention_drops_guard():
    model = _model(_guarded(None), QKV)
    feeds = _feeds()
    simp = _simplify(model, feeds)
    assert "IsNaN" not in _ops(simp)
    np.testing.assert_array_equal(_run(simp, feeds), _run(model, feeds))


def test_int_mask_cast_to_float_is_finite():
    # Expand/Cast of an integer mask: provably finite.
    mask = f"""
        mf = Cast <to = 1> (keep_i)
        shape = Constant <value = int64[4] {{{B}, 1, {S}, {S}}}> ()
        mask = Expand(mf, shape)
    """
    model = _model(_guarded(mask), f"{QKV}, int64[{B},1,1,{S}] keep_i")
    feeds = _feeds(keep_i=KEEP.astype(np.int64))
    simp = _simplify(model, feeds)
    assert "IsNaN" not in _ops(simp)
    np.testing.assert_array_equal(_run(simp, feeds), _run(model, feeds))


@pytest.mark.parametrize("opset", [17, 18])
@pytest.mark.parametrize("inf_when_true", [False, True])
def test_neg_inf_mask_rewritten_exactly(opset, inf_when_true):
    # The legacy exporter's form: Where(keep, 0, -inf) (or the mirrored
    # Where(pad, -inf, 0)). An all-padding row is NaN before the guard and
    # zeros after it; the rewrite must keep the zeros.
    branches = "minf, mzero" if inf_when_true else "mzero, minf"
    cond = "pad" if inf_when_true else "keep"
    mask = f"""
        mzero = Constant <value = float[1] {{0.0}}> ()
        minf = Constant <value = float[1] {{-inf}}> ()
        mask = Where({cond}, {branches})
    """
    model = _model(_guarded(mask), f"{QKV}, bool[{B},1,1,{S}] {cond}", opset=opset)
    flag = ~KEEP if inf_when_true else KEEP
    feeds = _feeds(**{cond: flag})
    simp = _simplify(model, feeds)
    ops = _ops(simp)
    assert "IsNaN" not in ops
    assert ("Not" in ops) == inf_when_true
    assert "ReduceMax" in ops
    assert not _has_value(simp, np.isinf)
    _assert_matches_guarded(_run(simp, feeds), _run(model, feeds))


def test_neg_inf_mask_shared_across_layers():
    # Two attention blocks reading the same -inf mask, as every layer of an
    # encoder does.
    mask = """
        mzero = Constant <value = float[1] {0.0}> ()
        minf = Constant <value = float[1] {-inf}> ()
        mask = Where(keep, mzero, minf)
    """
    one = _guarded(mask)
    two = (
        one.replace("out = MatMul", "h = MatMul")
        + """
        kt2 = Transpose <perm=[0,1,3,2]> (h)
        qk2 = MatMul(q, kt2)
        scores2 = Mul(qk2, scale)
        logits2 = Add(scores2, mask)
        p2 = Softmax <axis=-1> (logits2)
        nan2 = IsNaN(p2)
        guarded2 = Where(nan2, zero, p2)
        out = MatMul(guarded2, v)
        """
    )
    model = _model(two, f"{QKV}, bool[{B},1,1,{S}] keep")
    feeds = _feeds(keep=KEEP)
    simp = _simplify(model, feeds)
    assert "IsNaN" not in _ops(simp)
    assert not _has_value(simp, np.isinf)
    _assert_matches_guarded(_run(simp, feeds), _run(model, feeds))


def test_not_on_by_default():
    mask = """
        mzero = Constant <value = float[1] {0.0}> ()
        minf = Constant <value = float[1] {-inf}> ()
        mask = Where(keep, mzero, minf)
    """
    model = _model(_guarded(mask), f"{QKV}, bool[{B},1,1,{S}] keep")
    simp, ok = onnxsim.simplify(model, attention_fusion_backend="none")
    assert ok
    assert "IsNaN" in _ops(simp)


def test_float_mask_input_declined():
    # A float mask from a graph input may hold -inf: nothing is provable.
    model = _model(_guarded(""), f"{QKV}, float[{B},1,1,{S}] mask")
    m = np.where(KEEP, 0.0, -np.inf).astype(np.float32)
    feeds = _feeds(mask=m)
    simp = _simplify(model, feeds)
    assert "IsNaN" in _ops(simp)
    np.testing.assert_array_equal(_run(simp, feeds), _run(model, feeds))


def test_neg_inf_mask_softmax_not_over_keys_declined():
    # Softmax over the query axis: the row-validity reduction over the key
    # axis would be wrong, so the -inf rewrite declines.
    mask = """
        mzero = Constant <value = float[1] {0.0}> ()
        minf = Constant <value = float[1] {-inf}> ()
        mask = Where(keep, mzero, minf)
    """
    model = _model(_guarded(mask, softmax_axis=2), f"{QKV}, bool[{B},1,1,{S}] keep")
    feeds = _feeds(keep=KEEP)
    simp = _simplify(model, feeds)
    assert "IsNaN" in _ops(simp)
    np.testing.assert_array_equal(_run(simp, feeds), _run(model, feeds))


def test_non_scores_operand_declined():
    # Softmax(a + mask) where `a` is a graph input, not a Q@K^T product: `a`
    # may be non-finite, so the guard is kept.
    body = """
        mzero = Constant <value = float {0.0}> ()
        mmin = Constant <value = float {-1.0}> ()
        mask = Where(keep, mzero, mmin)
        logits = Add(a, mask)
        p = Softmax <axis=-1> (logits)
        nan = IsNaN(p)
        zero = Constant <value = float {0.0}> ()
        out = Where(nan, zero, p)
    """
    model = _model(
        body,
        f"float[{B},{H},{S},{S}] a, bool[{B},1,1,{S}] keep",
        out=f"float[{B},{H},{S},{S}] out",
    )
    rng = np.random.default_rng(0)
    feeds = {"a": rng.standard_normal((B, H, S, S)).astype(np.float32), "keep": KEEP}
    simp = _simplify(model, feeds)
    assert "IsNaN" in _ops(simp)


def test_neg_inf_mask_with_other_softmax_consumer_declined():
    # P is also a graph output: rewriting the -inf mask would turn its
    # all-padding NaN row finite there, so the guard is kept.
    mask = """
        mzero = Constant <value = float[1] {0.0}> ()
        minf = Constant <value = float[1] {-inf}> ()
        mask = Where(keep, mzero, minf)
    """
    model = _model(
        _guarded(mask),
        f"{QKV}, bool[{B},1,1,{S}] keep",
        out=f"float[{B},{H},{S},{D}] out, float[{B},{H},{S},{S}] p",
    )
    feeds = _feeds(keep=KEEP)
    simp = _simplify(model, feeds)
    assert "IsNaN" in _ops(simp)
    assert not _has_value(simp, lambda a: a == FINFO_MIN)
