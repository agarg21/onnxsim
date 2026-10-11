"""One decoder layer and the post model as the compiled op sequence
(scripts/axera/llm_layer_decomposition.py), checked against llm_reference.

Three synthetic configs (qwen3_like: q/k RMSNorm and a q width that is not the
hidden size; llama_gqa: grouped-query attention with a tied head; llama_mha: one
kv head per query head with an untied lm_head). The reference's cached cache rows
come from llm_reference's own steps, so the layer sees real rotated keys.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import onnx
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "axera"))

import llm_layer_decomposition as ld  # noqa: E402

PROMPT = [3, 17, 42, 7, 29, 11]
CONFIGS = sorted(ld.SYNTHETIC)
ALLOWED_OPS = {
    "Mul",
    "ReduceMean",
    "Add",
    "Sqrt",
    "Div",
    "MatMul",
    "Reshape",
    "Concat",
    "Transpose",
    "Softmax",
    "Sigmoid",
}


@pytest.fixture(scope="module")
def models():
    return {name: ld.make_synthetic(name) for name in CONFIGS}


def _rel(a, b):
    return float(np.abs(np.asarray(a) - np.asarray(b)).max() / np.abs(b).max())


@pytest.mark.parametrize("name", CONFIGS)
def test_layer_graph_uses_only_emitter_ops(models, name):
    m = models[name]
    model = ld.layer_graph(m, 0, len(PROMPT) - 1)
    onnx.checker.check_model(model)
    ops = {n.op_type for n in model.graph.node}
    assert ops <= ALLOWED_OPS, ops - ALLOWED_OPS
    # no Slice / Neg / Gather: the rotate-half is the MatMul with the permutation
    assert "Neg" not in ops and "Slice" not in ops
    assert [o.name for o in model.graph.output] == [
        "output",
        "K_cache_out",
        "V_cache_out",
    ]
    init = {i.name for i in model.graph.initializer}
    assert {"rot_perm", "wq_t", "wk_t", "wv_t", "wo_t", "wg_t", "wu_t", "wd_t"} <= init
    runtime = [i.name for i in model.graph.input]
    assert runtime == [
        "input",
        "K_cache",
        "V_cache",
        "mask",
        "cos_q",
        "sinq",
        "cos_k",
        "sink",
    ]
    assert ("qn" in init) == m.qk_norm


@pytest.mark.parametrize("name", CONFIGS)
@pytest.mark.parametrize("li", [0, 1])
@pytest.mark.parametrize("masked", [(), (1,)])
def test_layer_matches_reference(models, name, li, masked):
    m = models[name]
    x, K, V, _, pos = ld.decode_context(m, PROMPT, li)
    mask = ld.mask_for(K.shape[0], masked).reshape(-1)
    y_ref, k_ref, v_ref = m.layer_step(li, x, pos, K, V, mask)
    y, k_out, v_out = ld.layer_outputs(m, li, x, K, V, mask, pos)
    assert np.abs(y - y_ref).max() < 1e-5
    assert _rel(y, y_ref) < 1e-5
    # the cache rows the host would store
    assert np.abs(k_out[-1] - k_ref).max() < 1e-5
    assert np.abs(v_out[-1] - v_ref).max() < 1e-5
    # the earlier rows pass through the concat unchanged
    assert np.array_equal(k_out[:-1], K)
    assert np.array_equal(v_out[:-1], V)


@pytest.mark.parametrize("name", CONFIGS)
def test_mask_is_applied(models, name):
    m = models[name]
    x, K, V, _, pos = ld.decode_context(m, PROMPT, 0)
    free = ld.layer_outputs(m, 0, x, K, V, ld.mask_for(K.shape[0]), pos)[0]
    hidden = ld.layer_outputs(m, 0, x, K, V, ld.mask_for(K.shape[0], (1, 2)), pos)[0]
    assert np.abs(free - hidden).max() > 1e-3


@pytest.mark.parametrize("name", CONFIGS)
def test_post_matches_reference(models, name):
    m = models[name]
    x = m.embed([5])[0]
    # bitwise: the head enters as the transposed view, as llm_reference multiplies it
    assert np.array_equal(ld.post_outputs(m, x), m.post(x))
    graph = ld.post_graph(m)
    onnx.checker.check_model(graph)
    assert [i.name for i in graph.graph.input] == ["input", "head_t"]
    assert {n.op_type for n in graph.graph.node} <= ALLOWED_OPS


@pytest.mark.parametrize("name", ["qwen3_like", "llama_gqa"])
def test_tied_head_is_the_embedding(models, name):
    m = models[name]
    assert m._head == "model.embed_tokens.weight"
    x = m.embed([9])[0]
    # the head is the embedding table itself: logits of the embedding rows
    got = ld.post_outputs(m, x)
    assert np.array_equal(got, m.post(x))
    assert got.shape == (m.vocab,)
