# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Residual Add -> LayerNormalization outlined into a model-local function,
gated by the per-kernel traffic model."""

import numpy as np
import onnx
import pytest
from onnx import parser

from onnxsim.local_function_fusion import LOCAL_DOMAIN, outline_residual_add_norm
from onnxsim.model_info import kernel_traffic_breakdown

ort = pytest.importorskip("onnxruntime")

B, S, C = 1, 64, 256
FLOAT = 4  # bytes
SIZE = B * S * C * FLOAT  # one [B, S, C] float tensor
PARAMS = 2 * C * FLOAT  # scale + bias


def _model(with_bias=True):
    bias_in = ", float[256] bias" if with_bias else ""
    bias_arg = ", bias" if with_bias else ""
    return parser.parse_model(
        f"""
        <ir_version: 9, opset_import: ["" : 17]>
        g (float[{B}, {S}, {C}] x, float[{B}, {S}, {C}] skip,
           float[{C}] scale{bias_in}) => (float[{B}, {S}, {C}] y) {{
          s = Add(x, skip)
          y = LayerNormalization<axis=-1, epsilon=0.00001>(s, scale{bias_arg})
        }}
        """
    )


def _two_pairs_model():
    return parser.parse_model(
        f"""
        <ir_version: 9, opset_import: ["" : 17]>
        g (float[{B}, {S}, {C}] x, float[{B}, {S}, {C}] skip,
           float[{C}] scale, float[{C}] bias)
            => (float[{B}, {S}, {C}] y, float[{B}, {S}, {C}] z) {{
          s1 = Add(x, skip)
          y = LayerNormalization<axis=-1, epsilon=0.00001>(s1, scale, bias)
          s2 = Add(y, skip)
          z = LayerNormalization<axis=-1, epsilon=0.00001>(s2, scale, bias)
        }}
        """
    )


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)[0]


def _feeds(with_bias=True):
    rng = np.random.default_rng(0)
    feeds = {
        "x": rng.standard_normal((B, S, C)).astype(np.float32),
        "skip": rng.standard_normal((B, S, C)).astype(np.float32),
        "scale": rng.standard_normal(C).astype(np.float32),
    }
    if with_bias:
        feeds["bias"] = rng.standard_normal(C).astype(np.float32)
    return feeds


def _total(model, decompose=False):
    return sum(kernel_traffic_breakdown(model, decompose_functions=decompose).values())


@pytest.mark.parametrize("with_bias", [True, False])
def test_outlined_model_matches_original_numerics(with_bias):
    model = _model(with_bias)
    result = outline_residual_add_norm(model)
    assert result.count == 1
    feeds = _feeds(with_bias)
    np.testing.assert_allclose(
        _run(result.model, feeds), _run(model, feeds), rtol=1e-5, atol=1e-5
    )


def test_structure_is_one_local_call_and_one_function():
    result = outline_residual_add_norm(_model())
    outlined = result.model
    assert result.count == 1
    assert [n.op_type for n in outlined.graph.node] == ["ResidualAddNorm_0"]
    call = outlined.graph.node[0]
    assert call.domain == LOCAL_DOMAIN
    assert list(call.input) == ["x", "skip", "scale", "bias"]
    assert list(call.output) == ["y"]
    assert len(outlined.functions) == 1
    func = outlined.functions[0]
    assert [n.op_type for n in func.node] == ["Add", "LayerNormalization"]
    assert any(o.domain == LOCAL_DOMAIN for o in outlined.opset_import)
    onnx.checker.check_model(outlined, full_check=False)


def test_measured_saving_is_the_sum_and_the_norms_reread():
    # Original, per kernel: Add reads x, skip and writes sum (3S). The norm
    # reads sum, scale, bias, writes y and re-reads sum for its statistics
    # (3S + params). Total 6S + params.
    # Outlined, as one kernel: x, skip, scale, bias in and y out = 3S + params.
    model = _model()
    result = outline_residual_add_norm(model)
    assert _total(model) == 6 * SIZE + PARAMS
    assert _total(result.model) == 3 * SIZE + PARAMS
    assert result.saved_bytes == (3 * SIZE,)


def test_decomposed_function_costs_the_same_as_the_original():
    # Inlining the function restores the two primitive kernels, so the model
    # cannot claim the saving unless the runtime keeps the function as one.
    model = _model()
    result = outline_residual_add_norm(model)
    assert _total(result.model, decompose=True) == _total(model, decompose=True)


def test_gate_is_the_measured_saving():
    model = _model()
    # The traffic model measures 3S for this pair: a threshold of exactly 3S
    # keeps it, one byte more drops it.
    assert outline_residual_add_norm(model, min_bytes_saved=3 * SIZE).count == 1
    dropped = outline_residual_add_norm(model, min_bytes_saved=3 * SIZE + 1)
    assert dropped.count == 0
    assert [n.op_type for n in dropped.model.graph.node] == [
        "Add",
        "LayerNormalization",
    ]
    assert len(dropped.model.functions) == 0


def test_per_pair_savings_add_up_to_the_combined_saving():
    model = _two_pairs_model()
    result = outline_residual_add_norm(model)
    assert result.count == 2
    assert sum(result.saved_bytes) == _total(model) - _total(result.model)
    feeds = _feeds()
    np.testing.assert_allclose(
        _run(result.model, feeds), _run(model, feeds), rtol=1e-5, atol=1e-5
    )


def test_shared_sum_is_not_outlined():
    model = parser.parse_model(
        """
        <ir_version: 9, opset_import: ["" : 17]>
        g (float[1, 64, 256] x, float[1, 64, 256] skip, float[256] scale)
            => (float[1, 64, 256] y, float[1, 64, 256] z) {
          s = Add(x, skip)
          y = LayerNormalization<axis=-1, epsilon=0.00001>(s, scale)
          z = Mul(s, s)
        }
        """
    )
    assert outline_residual_add_norm(model).count == 0


def test_advisor_estimate_matches_the_measured_saving():
    # fusion_advisor predicts the saving for a sole-consumer residual pair from
    # static shapes alone; the outliner measures the same pair in the traffic
    # model. They must agree, or one of them is wrong.
    from onnxsim.fusion_advisor import fusion_candidates

    model = _model()
    (candidate,) = [
        c for c in fusion_candidates(model) if c.kind == "residual_add_norm"
    ]
    assert candidate.bytes_saved == outline_residual_add_norm(model).saved_bytes[0]


def test_model_info_reports_the_outlined_kernel_traffic_when_asked():
    from onnxsim.model_info import ModelInfo

    model = _model()
    default = ModelInfo(model, kernel_model=True).mem_access
    outlined = ModelInfo(model, kernel_model=True, outline_min_bytes=0).mem_access
    assert default == 6 * SIZE + PARAMS
    assert outlined == 3 * SIZE + PARAMS
    # The default is unchanged: outlining is opt-in.
    assert (
        ModelInfo(model, kernel_model=True, outline_min_bytes=None).mem_access
        == default
    )


def test_advisor_masked_softmax_estimate_matches_the_measured_call():
    # The Where -> Softmax pair measured as-is, then as one local-function call
    # with the same boundary tensors. The difference is the advisor's estimate.
    from onnx import helper

    from onnxsim.fusion_advisor import fusion_candidates

    unfused = parser.parse_model(
        f"""
        <ir_version: 9, opset_import: ["" : 17]>
        g (float[{B}, 4, 8, 8] s, bool[{B}, 1, 8, 8] cond, float[1, 1, 1, 1] neg)
            => (float[{B}, 4, 8, 8] p) {{
          w = Where(cond, s, neg)
          p = Softmax<axis=-1>(w)
        }}
        """
    )
    (candidate,) = [c for c in fusion_candidates(unfused) if c.kind == "masked_softmax"]
    body = [
        helper.make_node("Where", ["cond", "s", "neg"], ["w"]),
        helper.make_node("Softmax", ["w"], ["p"], axis=-1),
    ]
    func = helper.make_function(
        LOCAL_DOMAIN,
        "WhereSoftmax",
        ["s", "cond", "neg"],
        ["p"],
        body,
        [helper.make_opsetid("", 17)],
    )
    call = parser.parse_model(
        f"""
        <ir_version: 9, opset_import: ["" : 17, "{LOCAL_DOMAIN}" : 1]>
        g (float[{B}, 4, 8, 8] s, bool[{B}, 1, 8, 8] cond, float[1, 1, 1, 1] neg)
            => (float[{B}, 4, 8, 8] p) {{
          p = {LOCAL_DOMAIN}.WhereSoftmax(s, cond, neg)
        }}
        """
    )
    call.functions.append(func)
    measured = _total(unfused) - _total(call)
    small = 1 * 4 * 8 * 8 * FLOAT  # the [1, 4, 8, 8] intermediate
    assert candidate.bytes_saved == measured == 3 * small


def test_simplify_outlining_is_opt_in():
    from onnxsim import simplify

    model = _model()
    default, ok = simplify(model)
    assert ok
    assert len(default.functions) == 0
    assert [n.op_type for n in default.graph.node] == ["Add", "LayerNormalization"]


def test_simplify_outlining_when_requested_keeps_numerics():
    from onnxsim import simplify

    model = _model()
    outlined, ok = simplify(model, outline_residual_min_bytes=0)
    assert ok
    assert [n.op_type for n in outlined.graph.node] == ["ResidualAddNorm_0"]
    feeds = _feeds()
    np.testing.assert_allclose(
        _run(outlined, feeds), _run(model, feeds), rtol=1e-5, atol=1e-5
    )
