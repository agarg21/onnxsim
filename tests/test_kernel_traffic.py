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


def test_reduction_reads_its_input_once():
    model = _model("y = ReduceSum<keepdims = 1>(a)", f"float[{N}] a", "float[1] y")
    breakdown = kernel_traffic_breakdown(model)
    assert breakdown["ReduceSum"] == N * BYTES + BYTES  # one read, one-element out


def test_decomposed_softmax_costs_more_than_native_kernel():
    # Lowered to ReduceMax, Sub+Exp, ReduceSum, Div: each primitive is its own
    # kernel (or a fused elementwise chain), and the intermediates are written.
    model = _model("y = Softmax<axis = -1>(a)", f"float[{N}] a", f"float[{N}] y")
    native = sum(kernel_traffic_breakdown(model).values())
    decomposed = kernel_traffic_breakdown(model, decompose_functions=True)
    assert {"ReduceMax", "Sub+Exp", "ReduceSum", "Div"} <= set(decomposed)
    assert sum(decomposed.values()) > native
    assert native == 3 * N * BYTES


def test_gather_moves_the_selected_rows_not_the_whole_table():
    # A 1000-row float table, 4 int64 indices, output [4, 64]. The kernel reads
    # the 4 selected rows (the output size) and the indices, and writes the
    # output: 2 * (4 * 64) floats + 4 int64s. The table is not read in full.
    model = _model(
        "y = Gather<axis=0>(table, idx)",
        "float[1000, 64] table, int64[4] idx",
        "float[4, 64] y",
    )
    rows = 4 * 64 * BYTES
    assert sum(kernel_traffic_breakdown(model).values()) == 2 * rows + 4 * 8


def test_gather_of_an_unknown_output_charges_no_rows_but_keeps_the_indices():
    model = _model(
        "y = Gather<axis=0>(table, idx)",
        "float[1000, 64] table, int64[4] idx",
        "float[4, 64] y",
    )
    assert ModelInfo(model, kernel_model=True).mem_access == 2 * 4 * 64 * BYTES + 4 * 8


def test_gather_nd_moves_the_selected_rows_not_the_whole_table():
    # Decomposed grid sample gathers corners with GatherND: a 1000x64 table,
    # 4 index tuples, output [4, 64]. Same accounting as Gather: indices plus
    # the selected rows (the output size) read, and the output written.
    model = _model(
        "y = GatherND(table, idx)",
        "float[1000, 64] table, int64[4, 1] idx",
        "float[4, 64] y",
    )
    rows = 4 * 64 * BYTES
    assert sum(kernel_traffic_breakdown(model).values()) == 2 * rows + 4 * 8


def test_scatter_nd_is_charged_as_a_copy_plus_updates():
    # BEV pooling scatter-adds into a [C, H, W] grid. ScatterND copies the grid
    # to its output, so the grid is read once and written once, and the updates
    # and indices are read. The model keeps this copy convention.
    grid = 64 * 128 * 128 * BYTES
    k = 1000
    model = _model(
        'y = ScatterND<reduction="add">(grid, idx, upd)',
        f"float[64, 128, 128] grid, int64[{k}, 3] idx, float[{k}] upd",
        "float[64, 128, 128] y",
    )
    indices = k * 3 * 8
    updates = k * BYTES
    assert sum(kernel_traffic_breakdown(model).values()) == 2 * grid + indices + updates


def test_slice_moves_the_selected_elements_not_the_whole_input():
    # A 1024x64 float tensor sliced to its first 4 rows. The kernel reads the 4
    # rows (the output size) and the three int64 bounds, and writes the output:
    # 2 * (4 * 64) floats + 3 int64s. The 1024-row input is not read in full.
    model = _model(
        "y = Slice(x, s, e, a)",
        "float[1024, 64] x, int64[1] s, int64[1] e, int64[1] a",
        "float[4, 64] y",
    )
    rows = 4 * 64 * BYTES
    assert sum(kernel_traffic_breakdown(model).values()) == 2 * rows + 3 * 8


def test_activation_after_matmul_is_its_epilogue():
    # MatMul [4,8] x [8,16] then Relu. Unfused, the MatMul writes its output and
    # Relu reads it back: A + B + out, then out + out. As one kernel the output
    # stays in registers: A + B + the activated output only.
    model = _model(
        "m = MatMul(a, b)\n  y = Relu(m)",
        "float[4, 8] a, float[8, 16] b",
        "float[4, 16] y",
    )
    a, b, out = 4 * 8 * BYTES, 8 * 16 * BYTES, 4 * 16 * BYTES
    breakdown = kernel_traffic_breakdown(model)
    assert breakdown == {"MatMul+Relu": a + b + out}


def test_gemm_and_conv_take_an_activation_epilogue():
    gemm = _model(
        "m = Gemm<alpha=1.0, beta=1.0>(a, b, c)\n  y = Sigmoid(m)",
        "float[4, 8] a, float[8, 16] b, float[16] c",
        "float[4, 16] y",
    )
    assert kernel_traffic_breakdown(gemm) == {
        "Gemm+Sigmoid": (4 * 8 + 8 * 16 + 16 + 4 * 16) * BYTES
    }
    conv = _model(
        "m = Conv<kernel_shape=[3, 3]>(x, w)\n  y = Relu(m)",
        "float[1, 2, 8, 8] x, float[4, 2, 3, 3] w",
        "float[1, 4, 6, 6] y",
    )
    assert kernel_traffic_breakdown(conv) == {
        "Conv+Relu": (1 * 2 * 8 * 8 + 4 * 2 * 3 * 3 + 1 * 4 * 6 * 6) * BYTES
    }


def test_matmul_output_with_a_second_consumer_is_not_fused():
    # The MatMul result is also a graph output, so it must be written: no epilogue.
    model = _model(
        "m = MatMul(a, b)\n  y = Relu(m)",
        "float[4, 8] a, float[8, 16] b",
        "float[4, 16] y, float[4, 16] m",
    )
    a, b, out = 4 * 8 * BYTES, 8 * 16 * BYTES, 4 * 16 * BYTES
    assert sum(kernel_traffic_breakdown(model).values()) == (a + b + out) + (out + out)


def test_matmul_output_shared_by_elementwise_consumers_is_one_kernel():
    # A GELU-like fan-out: the MatMul output feeds both a Mul and a Div, and an
    # Add joins them. All consumers are elementwise and in one kernel, so the
    # output is shared in registers: inputs a, b, c and the final output only.
    model = _model(
        "m = MatMul(a, b)\n  p = Mul(m, c)\n  q = Div(m, c)\n  y = Add(p, q)",
        "float[4, 8] a, float[8, 16] b, float[1] c",
        "float[4, 16] y",
    )
    a, b, c, out = 4 * 8 * BYTES, 8 * 16 * BYTES, BYTES, 4 * 16 * BYTES
    assert kernel_traffic_breakdown(model) == {"MatMul+Mul+Div+Add": a + b + c + out}


def test_two_matmuls_never_share_a_kernel_epilogue():
    # Each MatMul output feeds an Add. The Add can absorb only one of the two
    # matmuls: a kernel takes at most one matmul.
    model = _model(
        "m1 = MatMul(a, b)\n  m2 = MatMul(a2, b2)\n  y = Add(m1, m2)",
        "float[4, 8] a, float[8, 16] b, float[4, 8] a2, float[8, 16] b2",
        "float[4, 16] y",
    )
    breakdown = kernel_traffic_breakdown(model)
    assert set(breakdown) == {"MatMul+Add", "MatMul"}


def test_view_mode_slice_is_read_in_place_by_its_elementwise_reader():
    # Default: the Slice is its own kernel and the Relu another, each reading and
    # writing its tensors. View mode: the Slice joins the Relu and reads its
    # selected elements in place, so the intermediate is never written.
    model = _model(
        "y = Slice(x, s, e, a)\n  z = Relu(y)",
        "float[4, 8] x, int64[1] s = {0}, int64[1] e = {2}, int64[1] a = {0}",
        "float[2, 8] z",
    )
    x_out = 2 * 8 * BYTES
    indices = 3 * 8
    assert (
        sum(kernel_traffic_breakdown(model).values())
        == (2 * x_out + indices) + 2 * x_out
    )
    assert kernel_traffic_breakdown(model, view_mode=True) == {
        "Slice+Relu": x_out + indices + x_out
    }


def test_view_mode_transpose_feeds_an_elementwise_kernel_without_a_copy():
    model = _model(
        "t = Transpose<perm=[2, 0, 1]>(x)\n  z = Relu(t)",
        "float[2, 3, 4] x",
        "float[4, 2, 3] z",
    )
    size = 2 * 3 * 4 * BYTES
    assert sum(kernel_traffic_breakdown(model).values()) == (2 * size) + (2 * size)
    assert kernel_traffic_breakdown(model, view_mode=True) == {
        "Transpose+Relu": 2 * size
    }


def test_view_mode_keeps_a_view_whose_output_is_a_graph_output():
    model = _model(
        "t = Transpose<perm=[2, 0, 1]>(x)\n  z = Relu(t)",
        "float[2, 3, 4] x",
        "float[4, 2, 3] z, float[4, 2, 3] t",
    )
    assert kernel_traffic_breakdown(model, view_mode=True) == kernel_traffic_breakdown(
        model
    )


def test_view_mode_transpose_is_read_in_place_by_its_matmul():
    # Transpose [4,8] -> [8,4] feeding a MatMul with [4,16]. Default: the copy is
    # written and read back. View mode: the MatMul reads the transposed input in
    # place, so only a, b and the output move.
    model = _model(
        "t = Transpose<perm=[1, 0]>(a)\n  y = MatMul(t, b)",
        "float[4, 8] a, float[4, 16] b",
        "float[8, 16] y",
    )
    a, b, out = 4 * 8 * BYTES, 4 * 16 * BYTES, 8 * 16 * BYTES
    assert sum(kernel_traffic_breakdown(model).values()) == (a + a) + (a + b + out)
    assert kernel_traffic_breakdown(model, view_mode=True) == {
        "Transpose+MatMul": a + b + out
    }


def test_view_mode_never_puts_two_matmuls_in_one_kernel():
    # Each MatMul reads a transposed input, and both feed one Add. The Add can
    # take only one of the two matmuls, so no kernel label holds two MatMuls.
    model = _model(
        "t1 = Transpose<perm=[1, 0]>(a)\n  m1 = MatMul(t1, b)\n"
        "  t2 = Transpose<perm=[1, 0]>(a2)\n  m2 = MatMul(t2, b2)\n  y = Add(m1, m2)",
        "float[4, 8] a, float[4, 16] b, float[4, 8] a2, float[4, 16] b2",
        "float[8, 16] y",
    )
    breakdown = kernel_traffic_breakdown(model, view_mode=True)
    assert all(label.count("MatMul") <= 1 for label in breakdown)


def test_view_mode_keeps_a_transpose_whose_output_is_a_graph_output():
    model = _model(
        "t = Transpose<perm=[1, 0]>(a)\n  y = MatMul(t, b)",
        "float[4, 8] a, float[4, 16] b",
        "float[8, 16] y, float[8, 4] t",
    )
    assert kernel_traffic_breakdown(model, view_mode=True) == kernel_traffic_breakdown(
        model
    )


_ATTENTION = "s = MatMul(q, kt)\n  m = Add(s, mask)\n  p = Softmax<axis=-1>(m)\n  y = MatMul(p, v)"
_ATTENTION_INPUTS = "float[4, 8] q, float[8, 4] kt, float[4, 4] mask, float[4, 8] v"


def test_flash_attention_is_one_kernel_that_reads_qkv_and_mask_only():
    # Decomposed attention: scores = Q.K^T, mask add, softmax, probs.V. With flash
    # attention the scores and probabilities never reach memory: the kernel reads
    # Q, K, the mask and V, and writes the output.
    model = _model(_ATTENTION, _ATTENTION_INPUTS, "float[4, 8] y")
    q, kt, mask, v, out = (4 * 8, 8 * 4, 4 * 4, 4 * 8, 4 * 8)
    assert kernel_traffic_breakdown(model, flash_attention=True) == {
        "MatMul+Add+Softmax+MatMul": (q + kt + mask + v + out) * BYTES
    }


def test_flash_attention_needs_a_single_consumer_softmax():
    # The probabilities also leave the kernel, so they must be written: no fusion.
    model = _model(
        _ATTENTION + "\n  z = Relu(p)",
        _ATTENTION_INPUTS,
        "float[4, 8] y, float[4, 4] z",
    )
    assert kernel_traffic_breakdown(
        model, flash_attention=True
    ) == kernel_traffic_breakdown(model)


def test_reduction_takes_its_elementwise_input_chain_into_its_kernel():
    # Mul(x, x) feeds ReduceMean over axis 1. Default: the product is written and
    # read back. With reduction fusion the reduction reads x and the axes inside
    # one kernel and writes only the [2, 1] result.
    model = _model(
        "y = Mul(x, x)\n  r = ReduceMean<keepdims=1>(y, ax)",
        "float[2, 4] x, int64[1] ax = {1}",
        "float[2, 1] r",
    )
    x, y, out = 2 * 4 * BYTES, 2 * 4 * BYTES, 2 * 1 * BYTES
    axes = 8
    assert sum(kernel_traffic_breakdown(model).values()) == (x + y) + (y + axes + out)
    assert kernel_traffic_breakdown(model, reduction_fusion=True) == {
        "Mul+ReduceMean": x + axes + out
    }


def test_elementwise_reader_of_a_reduction_joins_it():
    model = _model(
        "r = ReduceSum<keepdims=1>(x, ax)\n  y = Add(r, b)",
        "float[2, 4] x, int64[1] ax = {1}, float[2, 1] b",
        "float[2, 1] y",
    )
    x, axes, b, out = 2 * 4 * BYTES, 8, 2 * 1 * BYTES, 2 * 1 * BYTES
    assert kernel_traffic_breakdown(model, reduction_fusion=True) == {
        "ReduceSum+Add": x + axes + b + out
    }


def test_reduction_chain_that_is_a_graph_output_is_not_absorbed():
    model = _model(
        "y = Mul(x, x)\n  r = ReduceMean<keepdims=1>(y, ax)",
        "float[2, 4] x, int64[1] ax = {1}",
        "float[2, 1] r, float[2, 4] y",
    )
    assert kernel_traffic_breakdown(
        model, reduction_fusion=True
    ) == kernel_traffic_breakdown(model)


def test_view_mode_split_is_a_view_its_consumers_read_in_place():
    # Split [4, 8] into two [4, 4] slices that an Add reads. Default: the Split
    # reads its input and writes both slices. View mode: the slices are views, so
    # the Split costs nothing and the Add reads them directly.
    model = _model(
        "a, b = Split<axis=1>(x, sp)\n  y = Add(a, b)",
        "float[4, 8] x, int64[2] sp = {4, 4}",
        "float[4, 4] y",
    )
    x, sp, half, out = 4 * 8 * BYTES, 2 * 8, 4 * 4 * BYTES, 4 * 4 * BYTES
    assert sum(kernel_traffic_breakdown(model).values()) == (x + sp + 2 * half) + (
        2 * half + out
    )
    assert kernel_traffic_breakdown(model, view_mode=True) == {
        "Split": 0,
        "Add": 2 * half + out,
    }


def test_view_mode_split_output_that_is_a_graph_output_is_materialized():
    model = _model(
        "a, b = Split<axis=1>(x, sp)",
        "float[4, 8] x, int64[2] sp = {4, 4}",
        "float[4, 4] a, float[4, 4] b",
    )
    half = 4 * 4 * BYTES
    assert kernel_traffic_breakdown(model, view_mode=True) == {"Split": 2 * half}
