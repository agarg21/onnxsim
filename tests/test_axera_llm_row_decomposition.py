"""The decoder layer as one-row programs (scripts/axera/llm_row_decomposition.py).

Checks: the row layer agrees with llm_reference (relative error on y about 1e-7,
k and v rows bit-exact) on the three synthetic configs in both split modes; its
program ledger equals the formula and does not depend on the sequence length; at
the real SmolLM2-135M and Qwen3-0.6B dimensions (one layer, random weights) the
ledger gives 81 / 177 rowwise and 71 / 111 with the native multi-row split.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "axera"))

import llm_layer_decomposition as ld  # noqa: E402
import llm_row_decomposition as rd  # noqa: E402

PROMPT = [3, 17, 42, 7, 29, 11]
CONFIGS = sorted(ld.SYNTHETIC)
MODES = [False, True]  # native (multi-row) split off / on

SMOLLM2 = dict(
    model_type="llama",
    hidden=576,
    heads=9,
    kv_heads=3,
    head_dim=64,
    intermediate=1536,
    vocab=64,
    layers=1,
    qk_norm=False,
    tied=True,
    theta=1e5,
    seed=5,
)
QWEN3 = dict(
    model_type="qwen3",
    hidden=1024,
    heads=16,
    kv_heads=8,
    head_dim=128,
    intermediate=3072,
    vocab=64,
    layers=1,
    qk_norm=True,
    tied=True,
    theta=1e6,
    seed=6,
)


@pytest.fixture(scope="module")
def models():
    return {name: ld.make_synthetic(name) for name in CONFIGS}


@pytest.mark.parametrize("native", MODES)
@pytest.mark.parametrize("name", CONFIGS)
@pytest.mark.parametrize("li", [0, 1])
def test_row_layer_matches_reference(models, name, li, native):
    m = models[name]
    x, K, V, _, pos = ld.decode_context(m, PROMPT, li)
    mask = ld.mask_for(K.shape[0], (1,)).reshape(-1)
    y_ref, k_ref, v_ref = m.layer_step(li, x, pos, K, V, mask)
    y, K_out, V_out, _ = rd.row_layer(m, li, x, K, V, mask, pos, native=native)
    rel = np.abs(y - y_ref).max() / np.abs(y_ref).max()
    assert rel < 1e-6, rel
    assert np.array_equal(K_out[-1], k_ref)
    assert np.array_equal(V_out[-1], v_ref)
    assert np.array_equal(K_out[:-1], K) and np.array_equal(V_out[:-1], V)


@pytest.mark.parametrize("name", CONFIGS)
def test_row_layer_agrees_with_the_layer_graph(models, name):
    m = models[name]
    x, K, V, _, pos = ld.decode_context(m, PROMPT, 0)
    mask = ld.mask_for(K.shape[0]).reshape(-1)
    y_graph, _, _ = ld.layer_outputs(m, 0, x, K, V, mask, pos)
    y_rows, _, _, _ = rd.row_layer(m, 0, x, K, V, mask, pos)
    assert np.abs(y_rows - y_graph).max() < 1e-5


@pytest.mark.parametrize("name", CONFIGS)
def test_ledger_matches_formula(models, name):
    m = models[name]
    x, K, V, _, pos = ld.decode_context(m, PROMPT, 0)
    mask = ld.mask_for(K.shape[0]).reshape(-1)
    for native in MODES:
        _, _, _, L = rd.row_layer(m, 0, x, K, V, mask, pos, native=native)
        assert L.total == rd.expected_programs(m.nh, m.nkv, m.qk_norm, native)
        # four attention programs per query head, one of each kind
        for kind in ("attn_score", "attn_mask", "attn_softmax", "attn_pv"):
            assert L.counts[kind] == m.nh
        assert L.counts["cache_concat"] == 2 and L.counts["head_concat"] == 1


@pytest.mark.parametrize("name", CONFIGS)
def test_program_count_does_not_depend_on_sequence_length(models, name):
    m = models[name]
    totals = set()
    for S in (1, 5):
        ids = list(range(1, S + 2))
        x, K, V, _, pos = ld.decode_context(m, ids, 0)
        mask = ld.mask_for(K.shape[0]).reshape(-1)
        _, _, _, L = rd.row_layer(m, 0, x, K, V, mask, pos)
        totals.add(L.total)
    assert len(totals) == 1


def test_expected_counts_for_the_real_models():
    # SmolLM2-135M: 9 query heads, 3 kv heads, no q/k norm
    assert rd.expected_programs(9, 3, False, native=False) == 81
    assert rd.expected_programs(9, 3, False, native=True) == 71
    # Qwen3-0.6B: 16 query heads, 8 kv heads, q/k RMSNorm
    assert rd.expected_programs(16, 8, True, native=False) == 177
    assert rd.expected_programs(16, 8, True, native=True) == 111


@pytest.mark.parametrize("cfg,expected", [(SMOLLM2, (81, 71)), (QWEN3, (177, 111))])
def test_real_dimensions_ledger(cfg, expected):
    m = ld.make_synthetic(cfg)
    x, K, V, _, pos = ld.decode_context(m, PROMPT, 0)
    mask = ld.mask_for(K.shape[0]).reshape(-1)
    y_ref, k_ref, v_ref = m.layer_step(0, x, pos, K, V, mask)
    y, K_out, V_out, L = rd.row_layer(m, 0, x, K, V, mask, pos, native=False)
    assert L.total == expected[0]
    assert np.abs(y - y_ref).max() / np.abs(y_ref).max() < 1e-6
    assert np.array_equal(K_out[-1], k_ref) and np.array_equal(V_out[-1], v_ref)
    _, _, _, Ln = rd.row_layer(m, 0, x, K, V, mask, pos, native=True)
    assert Ln.total == expected[1]


def test_native_flag_selects_the_native_split(models, monkeypatch):
    m = models["llama_mha"]
    x, K, V, _, pos = ld.decode_context(m, PROMPT, 0)
    mask = ld.mask_for(K.shape[0]).reshape(-1)
    monkeypatch.setattr(rd, "NATIVE_MULTIROW", True)
    _, _, _, L = rd.row_layer(m, 0, x, K, V, mask, pos)
    assert L.total == rd.expected_programs(m.nh, m.nkv, m.qk_norm, True)
    assert L.counts["reduce_mean"] == 2 and L.counts["rope_matmul"] == 2
    monkeypatch.setattr(rd, "NATIVE_MULTIROW", False)
    _, _, _, L = rd.row_layer(m, 0, x, K, V, mask, pos)
    assert L.counts["rope_matmul"] == m.nh + m.nkv
