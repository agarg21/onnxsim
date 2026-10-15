"""Tests for the tiled vs. materializing attention traffic model in
``onnxsim.model_info``.

A tiled (flash-style) attention kernel moves only its inputs and outputs. A
materializing kernel also writes and reads the S x S score matrix, which is
counted as extra ``mem_access`` and as transient workspace in
``memory_footprint``. Unfused graphs spell that matrix out as a tensor, so they
are unaffected by the flag. Models are built with ``onnx.parser.parse_model``.
"""

import itertools

import pytest
from onnx import parser

from onnxsim import model_info
from onnxsim.model_info import METADATA_PREFIX, ModelInfo, annotate_metadata

B, H, S, D = 2, 8, 16, 64
# Score matrix elements: batch * heads * Sq * Skv.
SCORE_ELEMS = B * H * S * S


def _parse(body, ms_domain=False):
    domains = '"": 23' + (', "com.microsoft": 1' if ms_domain else "")
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: [{domains}]
        >
        {body}
        """
    )
    return model


def _fused_attention(dtype="float", hq=H, hkv=H):
    return f"""
    g ({dtype}[{B},{hq},{S},{D}] q, {dtype}[{B},{hkv},{S},{D}] k,
       {dtype}[{B},{hkv},{S},{D}] v) => ({dtype}[{B},{hq},{S},{D}] y)
    {{
      y = Attention(q, k, v)
    }}
    """


def _unfused_attention():
    # Same computation spelled out: the S x S scores are an explicit tensor.
    return f"""
    g (float[{B},{H},{S},{D}] q, float[{B},{H},{D},{S}] kt,
       float[{B},{H},{S},{D}] v) => (float[{B},{H},{S},{D}] y)
    {{
      s = MatMul(q, kt)
      p = Softmax<axis=-1>(s)
      y = MatMul(p, v)
    }}
    """


def _info(body, **kwargs):
    return ModelInfo(_parse(body), **kwargs)


# --------------------------------------------------------------------------- #
# Fused attention: tiled vs. materializing
# --------------------------------------------------------------------------- #
def test_default_is_tiled_and_unchanged():
    # Four float32 tensors of B*H*S*D elements each: q, k, v read; y written.
    body = _fused_attention()
    expected = 4 * B * H * S * D * 4
    assert _info(body).mem_access == expected
    assert _info(body, attention_tiled=True).mem_access == expected


def test_materialized_adds_score_write_and_read():
    # The S x S matrix is written once and read once, at the float32 size.
    body = _fused_attention()
    tiled = _info(body).mem_access
    materialized = _info(body, attention_tiled=False).mem_access
    assert materialized - tiled == 2 * SCORE_ELEMS * 4


def test_fused_tiled_traffic_is_below_fused_materialized():
    body = _fused_attention()
    assert _info(body).mem_access < _info(body, attention_tiled=False).mem_access


def test_fused_tiled_density_is_above_materialized():
    body = _fused_attention()
    tiled = _info(body)
    materialized = _info(body, attention_tiled=False)
    assert tiled.flops == materialized.flops
    assert tiled.compute_density > materialized.compute_density


def test_materialized_footprint_includes_score_workspace():
    # Single node: inputs q, k, v and output y are live at the node; the score
    # matrix is transient workspace on top of them while the node runs.
    body = _fused_attention()
    tiled = _info(body).memory_footprint
    materialized = _info(body, attention_tiled=False).memory_footprint
    assert tiled == 4 * B * H * S * D * 4
    assert materialized == tiled + SCORE_ELEMS * 4


def test_score_bytes_follow_activation_dtype():
    # float16 halves the score matrix bytes (and every other tensor's bytes).
    body = _fused_attention(dtype="float16")
    tiled = _info(body).mem_access
    materialized = _info(body, attention_tiled=False).mem_access
    assert tiled == 4 * B * H * S * D * 2
    assert materialized - tiled == 2 * SCORE_ELEMS * 2


def test_gqa_score_uses_query_heads():
    # kv heads < q heads: all query heads materialize their score matrix.
    body = _fused_attention(hq=H, hkv=2)
    tiled = _info(body).mem_access
    materialized = _info(body, attention_tiled=False).mem_access
    assert materialized - tiled == 2 * SCORE_ELEMS * 4


# --------------------------------------------------------------------------- #
# Unfused graphs are unaffected by the flag
# --------------------------------------------------------------------------- #
def test_unfused_traffic_is_unchanged():
    body = _unfused_attention()
    # MatMul: q + kt + s; Softmax: s + p; MatMul: p + v + y (float32).
    expected = (
        (B * H * S * D + B * H * D * S + SCORE_ELEMS)
        + (SCORE_ELEMS + SCORE_ELEMS)
        + (SCORE_ELEMS + B * H * S * D + B * H * S * D)
    ) * 4
    assert _info(body).mem_access == expected
    assert _info(body, attention_tiled=False).mem_access == expected


def test_unfused_footprint_is_unchanged():
    body = _unfused_attention()
    assert (
        _info(body, attention_tiled=False).memory_footprint
        == _info(body).memory_footprint
    )


def test_unfused_fused_tiled_totals_differ_only_by_the_score_matrix():
    # The fused tiled kernel drops the explicit S x S tensor (written and read
    # twice in the unfused graph), which is the credit the flag keeps honest.
    unfused = _info(_unfused_attention()).mem_access
    fused = _info(_fused_attention()).mem_access
    assert unfused - fused == 4 * SCORE_ELEMS * 4


# --------------------------------------------------------------------------- #
# Helper: which nodes count as materialized attention, and their score size
# --------------------------------------------------------------------------- #
def _node_and_maps(body, ms_domain=False):
    model = _parse(body, ms_domain=ms_domain)
    graph = model.graph
    shapes = model_info._collect_shapes(graph, {})
    dtypes = model_info._collect_dtypes(graph, {})
    return graph.node[0], shapes, dtypes


def test_score_bytes_none_for_non_attention_op():
    body = f"""
    g (float[{B},{S},{D}] x) => (float[{B},{S},{D}] y)
    {{
      y = Relu(x)
    }}
    """
    node, shapes, dtypes = _node_and_maps(body)
    assert model_info._attention_score_bytes(node, shapes, dtypes) is None
    assert model_info._attention_score_traffic(node, shapes, dtypes) == 0


def test_score_bytes_unknown_shapes_contribute_nothing():
    node, _, _ = _node_and_maps(_fused_attention())
    assert model_info._attention_score_bytes(node, {}, {}) is None


def test_score_bytes_multihead_attention_3d():
    # com.microsoft MultiHeadAttention: (batch, seq, hidden) query with the head
    # count in num_heads; the key is (batch, kv_seq, hidden).
    hidden = 512
    body = f"""
    g (float[{B},{S},{hidden}] q, float[{B},{S},{hidden}] k,
       float[{B},{S},{hidden}] v) => (float[{B},{S},{hidden}] y)
    {{
      y = com.microsoft.MultiHeadAttention<num_heads={H}>(q, k, v)
    }}
    """
    node, shapes, dtypes = _node_and_maps(body, ms_domain=True)
    assert model_info._attention_score_bytes(node, shapes, dtypes) == SCORE_ELEMS * 4


def test_score_bytes_com_microsoft_attention_3d():
    # com.microsoft Attention takes (batch, seq, input_dim) plus a weight matrix;
    # self-attention, so Skv equals the sequence length.
    hidden = 512
    body = f"""
    g (float[{B},{S},{hidden}] x, float[{hidden},{3 * hidden}] w)
      => (float[{B},{S},{hidden}] y)
    {{
      y = com.microsoft.Attention<num_heads={H}>(x, w)
    }}
    """
    node, shapes, dtypes = _node_and_maps(body, ms_domain=True)
    assert model_info._attention_score_bytes(node, shapes, dtypes) == SCORE_ELEMS * 4


def test_score_bytes_zero_head_count_is_unknown():
    # A 3D ai.onnx Attention with no q_num_heads has no knowable head split.
    body = f"""
    g (float[{B},{S},{H * D}] q, float[{B},{S},{H * D}] k,
       float[{B},{S},{H * D}] v) => (float[{B},{S},{H * D}] y)
    {{
      y = Attention(q, k, v)
    }}
    """
    node, shapes, dtypes = _node_and_maps(body)
    assert model_info._attention_score_bytes(node, shapes, dtypes) is None


# --------------------------------------------------------------------------- #
# annotate_metadata matches ModelInfo for every model-flag combination
# --------------------------------------------------------------------------- #
def _annotated_model_props(model, **flags):
    meta = {
        entry.key: entry.value
        for entry in annotate_metadata(model, **flags).metadata_props
    }
    return meta


def _attention_relu_model():
    # Attention (a score-matrix op) feeding an elementwise op, so the kernel
    # model fuses the Relu separately and decompose_functions has a schema
    # function op to expand.
    return _parse(
        f"""
        g (float[{B},{H},{S},{D}] q, float[{B},{H},{S},{D}] k,
           float[{B},{H},{S},{D}] v) => (float[{B},{H},{S},{D}] y)
        {{
          a = Attention(q, k, v)
          y = Relu(a)
        }}
        """
    )


FLAG_COMBINATIONS = list(itertools.product([True, False], repeat=2))


def test_annotate_metadata_defaults_unchanged():
    model = _attention_relu_model()
    default = annotate_metadata(model)
    explicit = annotate_metadata(
        model, attention_tiled=True, kernel_model=False, decompose_functions=False
    )
    assert default.SerializeToString() == explicit.SerializeToString()


@pytest.mark.parametrize("attention_tiled, kernel_model", FLAG_COMBINATIONS)
@pytest.mark.parametrize("decompose_functions", [False, True])
def test_annotate_metadata_model_values_match_model_info(
    attention_tiled, kernel_model, decompose_functions
):
    model = _attention_relu_model()
    info = ModelInfo(
        model,
        attention_tiled=attention_tiled,
        kernel_model=kernel_model,
        decompose_functions=decompose_functions,
    )
    meta = _annotated_model_props(
        model,
        attention_tiled=attention_tiled,
        kernel_model=kernel_model,
        decompose_functions=decompose_functions,
    )
    p = METADATA_PREFIX
    assert meta[p + "macs"] == str(info.macs)
    assert meta[p + "flops"] == str(info.flops)
    assert meta[p + "mem_access"] == str(info.mem_access)
    assert meta[p + "memory_footprint"] == str(info.memory_footprint)
    assert meta[p + "model_size"] == str(info.model_size)
    assert meta[p + "compute_density"] == str(info.compute_density)


def test_annotate_metadata_flags_change_the_annotated_totals():
    model = _attention_relu_model()
    tiled = _annotated_model_props(model)
    materialized = _annotated_model_props(model, attention_tiled=False)
    p = METADATA_PREFIX
    assert int(materialized[p + "mem_access"]) > int(tiled[p + "mem_access"])
    assert int(materialized[p + "memory_footprint"]) == (
        int(tiled[p + "memory_footprint"]) + SCORE_ELEMS * 4
    )


def test_annotate_metadata_attention_node_carries_score_traffic():
    model = _attention_relu_model()
    p = METADATA_PREFIX

    def attention_mem(**flags):
        annotated = annotate_metadata(model, **flags)
        node = next(n for n in annotated.graph.node if n.op_type == "Attention")
        return int(
            next(e.value for e in node.metadata_props if e.key == p + "mem_access")
        )

    assert attention_mem(attention_tiled=False) == (
        attention_mem(attention_tiled=True) + 2 * SCORE_ELEMS * 4
    )
