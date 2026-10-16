# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Device placement and partitioning of :mod:`onnxsim.graph_partition`."""

import onnx
from onnx import helper, parser

from onnxsim.graph_partition import CPU, partition_graph


def _model(body, inputs, outputs, opset=20):
    return parser.parse_model(
        f"""
        <ir_version: 9, opset_import: ["" : {opset}]>
        g ({inputs}) => ({outputs}) {{
          {body}
        }}
        """
    )


CONV_IN = "float[1, 8, 16, 16] x, float[8, 8, 3, 3] w, float[1, 8, 16, 16] r"
CONV_OUT = "float[1, 8, 16, 16] y"
TENSOR = 1 * 8 * 16 * 16 * 4  # one float32 activation


def test_conv_and_its_activation_stay_on_the_primary_device():
    # The Relu and Add read the conv output in a chain, so they follow it onto
    # WebNN: no boundary, one partition.
    model = _model(
        "c = Conv<pads=[1, 1, 1, 1]>(x, w)\n  a = Relu(c)\n  b = Add(a, r)\n  y = Sigmoid(b)",
        CONV_IN,
        CONV_OUT,
    )
    plan = partition_graph(model, "webnn")
    assert plan.counts() == {"webnn": 4}
    assert plan.heavy_on_primary == plan.heavy_total == 1
    assert plan.boundary_tensors == []
    assert plan.transfer_bytes == 0


def test_unsupported_op_falls_back_and_its_input_crosses_once():
    # Mod is not in the WebNN list, so it runs on CPU. Its input, the conv
    # activation, is the one tensor that crosses the device boundary.
    model = _model(
        "c = Conv<pads=[1, 1, 1, 1]>(x, w)\n  a = Relu(c)\n  m = Mod<fmod=1>(a, two)\n  y = Cast<to=1>(m)",
        CONV_IN + ", float[1] two = {2.0}",
        CONV_OUT,
    )
    plan = partition_graph(model, "webnn")
    assert plan.counts() == {"webnn": 2, CPU: 2}
    assert plan.boundary_tensors == [("a", "webnn", CPU)]
    assert plan.transfer_bytes == TENSOR


def test_webgpu_attention_mask_is_a_hazard_placed_on_the_fallback():
    # A com.microsoft Attention with mask_index is flagged for WebGPU. The plan
    # keeps it off WebGPU and reports it, since ORT commits such nodes at
    # partition time and would fail at run time.
    attention = helper.make_node(
        "Attention",
        ["x", "w", "b", "mask"],
        ["y"],
        name="attn",
        domain="com.microsoft",
        num_heads=2,
    )
    graph = helper.make_graph(
        [attention],
        "g",
        [
            helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 4, 8]),
            helper.make_tensor_value_info("w", onnx.TensorProto.FLOAT, [8, 24]),
            helper.make_tensor_value_info("b", onnx.TensorProto.FLOAT, [24]),
            helper.make_tensor_value_info("mask", onnx.TensorProto.INT32, [1, 4]),
        ],
        [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 4, 8])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 20),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )
    plan = partition_graph(model, "webgpu")
    assert plan.hazards == [0]
    assert plan.assignment == {0: CPU}


def test_split_by_a_fallback_node_is_reported_as_nonconvex():
    # Conv1 -> Mod (CPU) -> Add, and Add also reads Conv1 directly. The WebNN
    # region {Conv1, Add} is entered again after leaving through Mod, so it is
    # not convex and an ORT-style scheduler could not run it as one partition.
    model = _model(
        "c = Conv<pads=[1, 1, 1, 1]>(x, w)\n  m = Mod<fmod=1>(c, two)\n  y = Add(c, m)",
        CONV_IN + ", float[1] two = {2.0}",
        CONV_OUT,
    )
    plan = partition_graph(model, "webnn")
    assert plan.counts() == {"webnn": 2, CPU: 1}
    assert plan.nonconvex_partitions == 1
