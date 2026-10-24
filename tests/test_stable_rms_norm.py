"""Tests for the ``stable_rms_norm`` C++ pass (onnxsim/passes/stable_rms_norm.h).

A decomposed RMSNorm ``X / Sqrt(ReduceMean(X * X) + eps)`` becomes its two-pass
form ``u = X / r0``, ``y = u / Sqrt(ReduceMean(u * u) + eps / r0^2)`` -- equal
in real arithmetic, and range-safe for a backend that quantizes every op's
output: the second pass's ``u * u`` is bounded by the normalized width
whatever the token, where the first pass's ``X * X`` range is set by rare
outlier tokens. Opt-in via ``extra_optimizers=["stable_rms_norm"]``.

Models are built with ``onnx.parser`` (AGENTS.md); numeric equivalence comes
from ``onnxsim.simplify``'s own ``check_n`` plus an explicit comparison.
"""

import numpy as np
import pytest
from onnx import numpy_helper, parser
from onnx.reference import ReferenceEvaluator

import onnxsim

H = 64
EPS = "{1e-5}"


def _model(body, opset=17, gain=True):
    model = parser.parse_model(
        f"""
        <ir_version: 9, opset_import: ["": {opset}]>
        {body}
        """
    )
    if gain:
        # a non-trivial per-channel gain: a numpy initializer, not a text literal
        g = np.random.RandomState(3).uniform(0.5, 1.5, (H,)).astype(np.float32)
        model.graph.initializer.append(numpy_helper.from_array(g, "g"))
    return model


def _x(rows=8, seed=0, outliers=True):
    x = np.random.RandomState(seed).randn(rows, H).astype(np.float32)
    if outliers:  # an LLM residual stream's few huge channels, on some rows only
        x[1, 5] = 900.0
        x[4, 17] = -2500.0
    return x


def _ops(model):
    return [n.op_type for n in model.graph.node]


def _simplify(model, x, **kw):
    sim, ok = onnxsim.simplify(
        model,
        check_n=1,
        input_data={"x": x},
        extra_optimizers=["stable_rms_norm"],
        **kw,
    )
    assert ok
    return sim


def _same(a, b, x):
    ya = ReferenceEvaluator(a).run(None, {"x": x})[0]
    yb = ReferenceEvaluator(b).run(None, {"x": x})[0]
    np.testing.assert_allclose(yb, ya, rtol=2e-5, atol=2e-6)


SPELLINGS = {
    "div": "n = Div(x, r)",
    "reciprocal": "inv = Reciprocal(r)\n n = Mul(x, inv)",
    "one_over": "inv = Div(one, r)\n n = Mul(inv, x)",
}


@pytest.mark.parametrize("spelling", list(SPELLINGS))
@pytest.mark.parametrize("square", ["Mul(x, x)", "Pow(x, two)"])
def test_each_spelling_becomes_two_pass_and_computes_the_same(spelling, square):
    model = _model(
        f"""
        g (float[8,{H}] x) => (float[8,{H}] y)
        <float eps = {EPS}, float one = {{1.0}}, float two = {{2.0}}, float[{H}] g>
        {{
          sq = {square}
          m = ReduceMean<axes = [-1], keepdims = 1>(sq)
          e = Add(m, eps)
          r = Sqrt(e)
          {SPELLINGS[spelling]}
          y = Mul(n, g)
        }}
        """
    )
    x = _x()
    sim = _simplify(model, x)
    ops = _ops(sim)
    assert ops.count("ReduceMean") == 2 and ops.count("Sqrt") == 2
    assert "Reciprocal" not in ops
    _same(model, sim, x)


def test_gainless_norm_at_opset_18_with_axes_as_an_input():
    model = _model(
        f"""
        g (float[8,{H}] x) => (float[8,{H}] y)
        <float eps = {EPS}, int64[1] axes = {{-1}}>
        {{
          sq = Mul(x, x)
          m = ReduceMean<keepdims = 1>(sq, axes)
          e = Add(eps, m)
          r = Sqrt(e)
          y = Div(x, r)
        }}
        """,
        opset=18,
        gain=False,
    )
    x = _x()
    sim = _simplify(model, x)
    assert _ops(sim).count("ReduceMean") == 2
    _same(model, sim, x)


def test_rewritten_graph_is_not_rewritten_again():
    """The second pass's eps is not a constant and the first Sqrt has several
    uses, so running the pass twice gives the same graph."""
    model = _model(
        f"""
        g (float[8,{H}] x) => (float[8,{H}] y)
        <float eps = {EPS}, float[{H}] g>
        {{
          sq = Mul(x, x)
          m = ReduceMean<axes = [-1], keepdims = 1>(sq)
          e = Add(m, eps)
          r = Sqrt(e)
          n = Div(x, r)
          y = Mul(n, g)
        }}
        """
    )
    x = _x()
    once = _simplify(model, x)
    twice = _simplify(once, x)
    assert _ops(once) == _ops(twice)
    assert _ops(twice).count("ReduceMean") == 2


def test_declines_when_an_intermediate_has_another_use():
    model = _model(
        f"""
        g (float[8,{H}] x) => (float[8,{H}] y, float[8,1] m)
        <float eps = {EPS}, float[{H}] g>
        {{
          sq = Mul(x, x)
          m = ReduceMean<axes = [-1], keepdims = 1>(sq)
          e = Add(m, eps)
          r = Sqrt(e)
          n = Div(x, r)
          y = Mul(n, g)
        }}
        """
    )
    sim = _simplify(model, _x())
    assert _ops(sim).count("ReduceMean") == 1


def test_declines_a_reduction_over_an_inner_axis():
    model = _model(
        f"""
        g (float[8,{H}] x) => (float[8,{H}] y)
        <float eps = {EPS}>
        {{
          sq = Mul(x, x)
          m = ReduceMean<axes = [0], keepdims = 1>(sq)
          e = Add(m, eps)
          r = Sqrt(e)
          y = Div(x, r)
        }}
        """,
        gain=False,
    )
    sim = _simplify(model, _x())
    assert _ops(sim).count("ReduceMean") == 1


def test_not_run_by_default():
    model = _model(
        f"""
        g (float[8,{H}] x) => (float[8,{H}] y)
        <float eps = {EPS}>
        {{
          sq = Mul(x, x)
          m = ReduceMean<axes = [-1], keepdims = 1>(sq)
          e = Add(m, eps)
          r = Sqrt(e)
          y = Div(x, r)
        }}
        """,
        gain=False,
    )
    sim, ok = onnxsim.simplify(model)
    assert ok
    assert _ops(sim).count("ReduceMean") == 1


def _fake_quant_u16(a, lo, hi):
    """Quantize to 16-bit unsigned codes over [min(lo, 0), max(hi, 0)], as a
    per-tensor MinMax-calibrated backend stores an op's output."""
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    s = (hi - lo) / 65535.0
    zp = np.rint(-lo / s)
    return (np.clip(np.rint(a / s) + zp, 0, 65535) - zp) * s


def test_two_pass_rms_survives_16_bit_squares_with_rare_outliers():
    """Model of the backend that motivated the pass: every op output is
    quantized to 16 bits at its own calibrated range. At SmolLM2-135M's
    proportions (width 576, a typical mean square of ~12, one calibration
    token with a ~1600 outlier channel) the one-pass x^2 step is ~40, so an
    ordinary token's squares mostly round to 0 or 40 and its RMS comes out
    several percent off; the second pass quantizes u^2 instead, whose range is
    bounded by the width up to the first pass's error."""
    width = 576
    rng = np.random.RandomState(7)
    calib = rng.randn(512, width) * 3.5
    calib[17, 3] = 1600.0  # one outlier token in calibration
    test = rng.randn(128, width) * 3.5  # ordinary tokens
    eps = 1e-5
    rms_true = np.sqrt((test**2).mean(-1) + eps)

    sq_hi = (calib**2).max()
    m_hi = (calib**2).mean(-1).max()

    def one_pass(x):
        sq = _fake_quant_u16(x * x, 0.0, sq_hi)
        m = _fake_quant_u16(sq.mean(-1), 0.0, m_hi)
        return np.sqrt(m + eps)

    r0_cal = one_pass(calib)
    u_cal = calib / r0_cal[:, None]
    uu_hi = (u_cal**2).max()
    mu_hi = (u_cal**2).mean(-1).max() * 2

    def two_pass(x):
        r0 = one_pass(x)
        u = x / r0[:, None]
        uu = _fake_quant_u16(u * u, 0.0, uu_hi)
        mu = _fake_quant_u16(uu.mean(-1), 0.0, mu_hi)
        return r0 * np.sqrt(mu + eps / r0**2)

    err_one = np.abs(one_pass(test) / rms_true - 1).max()
    err_two = np.abs(two_pass(test) / rms_true - 1).max()
    assert uu_hi <= 2 * width  # bounded by the width, not by the outlier
    assert err_one > 0.02  # the one-pass RMS is visibly off
    assert err_two < 1e-3  # the second pass recovers it
