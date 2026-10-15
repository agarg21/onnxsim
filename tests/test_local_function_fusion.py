# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Residual Add -> LayerNormalization outlined into a model-local function."""

import numpy as np
import onnx
import pytest
from onnx import parser

from onnxsim.local_function_fusion import LOCAL_DOMAIN, outline_residual_add_norm
from onnxsim.model_info import kernel_traffic_breakdown

ort = pytest.importorskip("onnxruntime")

B, S, C = 1, 64, 256
FLOAT = 4  # bytes


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


@pytest.mark.parametrize("with_bias", [True, False])
def test_outlined_model_matches_original_numerics(with_bias):
    model = _model(with_bias)
    outlined, count = outline_residual_add_norm(model)
    assert count == 1
    feeds = _feeds(with_bias)
    np.testing.assert_allclose(
        _run(outlined, feeds), _run(model, feeds), rtol=1e-5, atol=1e-5
    )


def test_structure_is_one_local_call_and_one_function():
    outlined, count = outline_residual_add_norm(_model())
    assert count == 1
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


def test_one_kernel_call_saves_the_sum_and_its_norm_reread():
    # Original, per kernel: Add reads x, skip and writes sum (3S). The norm
    # reads sum, scale, bias, writes y and re-reads sum for its statistics
    # (S + S + S + sc + b + S = 3S + sc + b). Total 6S + sc + b.
    # Outlined, as one kernel: x, skip, scale, bias in and y out = 3S + sc + b.
    model = _model()
    outlined, _ = outline_residual_add_norm(model)
    size = B * S * C * FLOAT
    params = 2 * C * FLOAT  # scale + bias

    before = sum(kernel_traffic_breakdown(model).values())
    after = sum(kernel_traffic_breakdown(outlined).values())
    assert before == 6 * size + params
    assert after == 3 * size + params
    assert before - after == 3 * size


def test_decomposed_function_costs_the_same_as_the_original():
    # Inlining the function restores the two primitive kernels, so the model
    # cannot claim the saving unless the runtime keeps the function as one.
    model = _model()
    outlined, _ = outline_residual_add_norm(model)
    assert sum(
        kernel_traffic_breakdown(outlined, decompose_functions=True).values()
    ) == sum(kernel_traffic_breakdown(model, decompose_functions=True).values())


def test_gate_skips_pairs_below_the_threshold():
    model = _model()
    size = B * S * C * FLOAT
    outlined, count = outline_residual_add_norm(model, min_bytes_saved=2 * size + 1)
    assert count == 0
    assert len(outlined.functions) == 0
    assert [n.op_type for n in outlined.graph.node] == ["Add", "LayerNormalization"]


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
    _, count = outline_residual_add_norm(model)
    assert count == 0
