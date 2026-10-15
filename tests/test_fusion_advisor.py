# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Fusion advisor: candidate traffic savings and the should_fuse gate."""

import dataclasses

from onnx import parser

from onnxsim.fusion_advisor import FusionCandidate, fusion_candidates, should_fuse

N = 64
BYTES = 4  # float32
TENSOR = N * BYTES  # 256 bytes per [N] float32 tensor


def _model(body, inputs, outputs, opset=20):
    return parser.parse_model(
        f"""
        <ir_version: 9, opset_import: ["" : {opset}]>
        g ({inputs}) => ({outputs}) {{
          {body}
        }}
        """
    )


def _chain(kind_candidates):
    return [c for c in kind_candidates if c.kind == "elementwise_chain"]


def test_three_node_chain_is_one_candidate_with_exact_bytes_saved():
    # Add -> Mul -> Sigmoid. t1 and t2 are each written by one node and read by
    # the next; a fused kernel keeps both in registers, saving 2 * size each.
    model = _model(
        "t1 = Add(a, b)\nt2 = Mul(t1, a)\ny = Sigmoid(t2)",
        f"float[{N}] a, float[{N}] b",
        f"float[{N}] y",
    )
    candidates = fusion_candidates(model)
    assert len(candidates) == 1
    (chain,) = candidates
    assert chain.kind == "elementwise_chain"
    assert chain.op_types == ("Add", "Mul", "Sigmoid")
    assert chain.tensors == ("t1", "t2")
    assert chain.bytes_saved == 2 * TENSOR + 2 * TENSOR == 1024


def test_shared_intermediate_yields_no_chain_candidate():
    # t1 feeds two consumers (Relu and Neg), so it must be materialized: no
    # chain may include it, and the Relu->Sigmoid chain sees t1 only as an
    # external input.
    model = _model(
        "t1 = Add(a, b)\nu = Relu(t1)\ny = Sigmoid(u)\nz = Neg(t1)",
        f"float[{N}] a, float[{N}] b",
        f"float[{N}] y, float[{N}] z",
    )
    chains = _chain(fusion_candidates(model))
    assert [c.op_types for c in chains] == [("Relu", "Sigmoid")]
    assert all("t1" not in c.tensors for c in chains)


def test_fully_split_shared_intermediate_yields_no_chain():
    # t1 is shared by Relu and Sigmoid, which are both graph outputs and so
    # cannot join it: no multi-node chain exists at all.
    model = _model(
        "t1 = Add(a, b)\nu = Relu(t1)\nv = Sigmoid(t1)",
        f"float[{N}] a, float[{N}] b",
        f"float[{N}] u, float[{N}] v",
    )
    assert _chain(fusion_candidates(model)) == []


def test_residual_add_feeding_layernorm_saves_sum_write_and_read():
    # The sum is consumed only by the norm: written and re-read are both saved.
    model = _model(
        "s = Add(a, b)\ny = LayerNormalization<axis = -1>(s, g, bt)",
        f"float[1, {N}] a, float[1, {N}] b, float[{N}] g, float[{N}] bt",
        f"float[1, {N}] y",
    )
    (candidate,) = fusion_candidates(model)
    assert candidate.kind == "residual_add_norm"
    assert candidate.op_types == ("Add", "LayerNormalization")
    assert candidate.tensors == ("s",)
    # Written by the Add, then read twice by the multi-pass norm: 3 * TENSOR.
    assert candidate.bytes_saved == 3 * TENSOR


def test_residual_stream_still_materialized_saves_only_the_reread():
    # The sum is also a graph output (the residual stream), so it is written
    # anyway; the norm's two reads of it are avoided: 2 * TENSOR.
    model = _model(
        "s = Add(a, b)\ny = LayerNormalization<axis = -1>(s, g, bt)",
        f"float[1, {N}] a, float[1, {N}] b, float[{N}] g, float[{N}] bt",
        f"float[1, {N}] y, float[1, {N}] s",
    )
    (candidate,) = fusion_candidates(model)
    assert candidate.bytes_saved == 2 * TENSOR


def test_where_feeding_softmax_is_a_masked_softmax_candidate():
    model = _model(
        "w = Where(c, a, a)\ny = Softmax<axis = -1>(w)",
        f"bool[1, {N}] c, float[1, {N}] a",
        f"float[1, {N}] y",
    )
    (candidate,) = fusion_candidates(model)
    assert candidate.kind == "masked_softmax"
    assert candidate.op_types == ("Where", "Softmax")
    assert candidate.bytes_saved == 2 * TENSOR


def test_gate_flips_at_threshold_on_real_candidate():
    # Add -> Relu over float[512]: the intermediate is 2048 bytes, saving 4096.
    def chain_of(width):
        model = _model(
            "t = Add(a, b)\ny = Relu(t)",
            f"float[{width}] a, float[{width}] b",
            f"float[{width}] y",
        )
        (candidate,) = fusion_candidates(model)
        return candidate

    assert should_fuse(chain_of(512)) is True  # exactly 4096: at threshold
    assert should_fuse(chain_of(256)) is False  # 2048 bytes: launch-bound


def test_gate_boundaries_and_custom_threshold():
    def cand(bytes_saved):
        return FusionCandidate("elementwise_chain", (), (), (), bytes_saved)

    assert should_fuse(cand(4095)) is False
    assert should_fuse(cand(4096)) is True
    assert should_fuse(cand(8192)) is True
    assert should_fuse(cand(0)) is False
    assert should_fuse(cand(4095), min_bytes_saved=4095) is True
    assert should_fuse(cand(4096), min_bytes_saved=8192) is False


def test_symbolic_size_is_undecided():
    # batch is a dim_param: the saving is 2 * 2 * batch * 64 * 4 bytes, a formula.
    model = _model(
        "t1 = Add(a, b)\nt2 = Mul(t1, a)\ny = Sigmoid(t2)",
        f"float[batch, {N}] a, float[batch, {N}] b",
        f"float[batch, {N}] y",
    )
    (candidate,) = fusion_candidates(model)
    assert candidate.bytes_saved is not None
    assert candidate.bytes_saved.free_symbols
    assert should_fuse(candidate) is None
    assert should_fuse(dataclasses.replace(candidate, bytes_saved=None)) is None


def test_unknown_size_is_undecided():
    candidate = FusionCandidate("elementwise_chain", (), (), (), None)
    assert should_fuse(candidate) is None
