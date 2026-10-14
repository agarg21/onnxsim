# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Per-kernel traffic model: fused elementwise chains, views and multi-pass ops."""

from onnx import parser

from onnxsim.model_info import ModelInfo, kernel_traffic_breakdown

N = 64
BYTES = 4  # float32


def _model(body, inputs, outputs):
    return parser.parse_model(
        f"""
        <ir_version: 9, opset_import: ["" : 20]>
        g ({inputs}) => ({outputs}) {{
          {body}
        }}
        """
    )


def test_elementwise_chain_is_one_kernel_with_external_io_only():
    # Add -> Mul -> Sigmoid: the two intermediates stay inside one kernel. It
    # reads a and b once and writes y once: 3 * N elements. Per node, the same
    # chain would move 8 * N elements.
    model = _model(
        "t1 = Add(a, b)\nt2 = Mul(t1, a)\ny = Sigmoid(t2)",
        f"float[{N}] a, float[{N}] b",
        f"float[{N}] y",
    )
    breakdown = kernel_traffic_breakdown(model)
    assert breakdown == {"Add+Mul+Sigmoid": 3 * N * BYTES}
    per_node = ModelInfo(model).mem_access
    assert per_node == 8 * N * BYTES
    assert ModelInfo(model, kernel_model=True).mem_access == 3 * N * BYTES


def test_shared_intermediate_is_not_fused():
    # t1 feeds two consumers, so it is a materialized tensor: the chain splits.
    model = _model(
        "t1 = Add(a, b)\nu = Mul(t1, a)\nv = Sigmoid(t1)\ny = Add(u, v)",
        f"float[{N}] a, float[{N}] b",
        f"float[{N}] y",
    )
    breakdown = kernel_traffic_breakdown(model)
    assert "Add+Mul+Sigmoid" not in breakdown
    assert sum(breakdown.values()) == ModelInfo(model, kernel_model=True).mem_access
    # t1 is written by its kernel and read by two others (separate kernels).
    assert len(breakdown) > 1


def test_reshape_moves_no_bytes():
    model = _model(
        "shp = Constant<value_ints = [4, 16]>()\nr = Reshape(a, shp)\ny = Relu(r)",
        f"float[{N}] a",
        "float[4, 16] y",
    )
    breakdown = kernel_traffic_breakdown(model)
    assert breakdown.get("Reshape", 0) == 0
    assert breakdown["Relu"] == 2 * N * BYTES  # read r once, write y once


def test_softmax_reads_its_input_twice():
    # Statistics pass plus the normalizing pass: input read twice, output once.
    model = _model("y = Softmax<axis = -1>(a)", f"float[{N}] a", f"float[{N}] y")
    breakdown = kernel_traffic_breakdown(model)
    assert breakdown["Softmax"] == 3 * N * BYTES


def test_default_model_keeps_node_level_numbers():
    model = _model(
        "t1 = Add(a, b)\ny = Sigmoid(t1)",
        f"float[{N}] a, float[{N}] b",
        f"float[{N}] y",
    )
    default = ModelInfo(model)
    assert default.kernel_model is False
    assert default.mem_access == 5 * N * BYTES  # Add: 3 streams, Sigmoid: 2
    assert ModelInfo(model, kernel_model=True).mem_access == 3 * N * BYTES
