"""RoPE rotate-half as MatMul + Mul + Add (scripts/axera/rope_rotate_half.py).

The decomposition is checked bitwise against llm_reference's RoPE (`Model._rot`
in float32) for the head dims and rope thetas of the real checkpoints (Qwen3-0.6B:
hd 128, theta 1e6; SmolLM2-135M: hd 64, theta 1e5; llama-160m: hd 64, theta 1e4),
at several positions and input scales. No device, no checkpoint.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import onnx
import pytest
from onnx.reference import ReferenceEvaluator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "axera"))

import llm_reference as ref  # noqa: E402
import rope_rotate_half as rr  # noqa: E402

POSITIONS = [0, 1, 7, 63, 128, 255, 1000]
INPUT_SCALES = [0.1, 1.0, 8.0]
# (name, head_dim, rope_theta, heads): the checkpoints' per-head RoPE
CONFIGS = [
    ("qwen3", 128, 1e6, 16),
    ("smollm2", 64, 1e5, 9),
    ("llama", 64, 1e4, 4),
]


def _ref_rot(x, cos, sin):
    return ref.Model._rot(x, cos, sin)


def _run(model: onnx.ModelProto, x, cos, sin_s):
    (y,) = ReferenceEvaluator(model).run(None, {"x": x, "cos": cos, "sin_signed": sin_s})
    return y


@pytest.mark.parametrize("hd", [2, 4, 8, 64, 128])
def test_permutation_matrix(hd):
    p = rr.perm_matrix(hd)
    h = hd // 2
    # a 0/1 permutation matrix: one 1 per row and per column, symmetric
    assert set(np.unique(p)) <= {0.0, 1.0}
    assert (p.sum(axis=0) == 1).all() and (p.sum(axis=1) == 1).all()
    assert (p == p.T).all()
    rng = np.random.default_rng(hd)
    x = rng.standard_normal((5, hd)).astype(np.float32)
    want = np.concatenate([x[:, h:], x[:, :h]], axis=-1)
    assert np.array_equal(rr.rotate_half_matmul(x), want)
    assert np.array_equal(x @ p, want)
    # the sign vector: -1 on the first half, +1 on the second
    s = rr.sign_vector(hd)
    assert (s[:h] == -1).all() and (s[h:] == 1).all()


@pytest.mark.parametrize("hd", [2, 4, 8, 64, 128])
def test_numpy_form_is_the_reference_rope(hd):
    rng = np.random.default_rng(100 + hd)
    x = rng.standard_normal((3, hd)).astype(np.float32)
    cos, sin = rr.rope_tables(hd, 1e4, np.array([[5], [17], [900]]))
    cos, sin = cos.reshape(3, hd), sin.reshape(3, hd)
    assert np.array_equal(rr.apply(x, cos, sin), _ref_rot(x, cos, sin))


def test_calibration_weight_dequantizes_to_the_permutation():
    for hd in (2, 4, 8, 64, 128):
        w = rr.calibration_weight(hd)
        codes, s = rr.weight_codes(w)
        assert set(np.unique(codes)) <= {0, 127}
        assert (s == np.float32(1 / 127)).all()
        dequant = codes.astype(np.float32) * s
        assert np.array_equal(dequant, rr.perm_matrix(hd))


def test_perm_initializer_is_the_exact_permutation():
    init = rr.perm_initializer(64, "rot_perm")
    from onnx import numpy_helper

    assert init.name == "rot_perm"
    assert np.array_equal(numpy_helper.to_array(init), rr.perm_matrix(64))


@pytest.mark.parametrize("name,hd,theta,heads", CONFIGS)
@pytest.mark.parametrize("scale", INPUT_SCALES)
def test_graph_is_bitwise_the_reference_rope(name, hd, theta, heads, scale):
    rng = np.random.default_rng(hd + int(scale * 10))
    per_head = rr.rope_rotate_half_graph(hd, rows=1)
    all_heads = rr.rope_rotate_half_graph(hd, rows=heads)
    for pos in POSITIONS:
        cos, sin = rr.rope_tables(hd, theta, pos)
        sin_s = rr.sin_signed(sin)
        x = (rng.standard_normal((heads, hd)) * scale).astype(np.float32)
        want = _ref_rot(x, cos, sin)
        # per-head program: one [1, hd] row
        for h in range(heads):
            y = _run(
                per_head,
                x[h : h + 1],
                cos[None],
                sin_s[None],
            )
            assert np.array_equal(y[0], want[h]), (name, pos, scale, h)
        # the same graph over every head at once
        y = _run(
            all_heads,
            x,
            np.broadcast_to(cos, (heads, hd)).copy(),
            np.broadcast_to(sin_s, (heads, hd)).copy(),
        )
        assert np.array_equal(y, want), (name, pos, scale)


def test_graph_structure_has_no_slice_concat_or_neg():
    model = rr.rope_rotate_half_graph(128, rows=16)
    onnx.checker.check_model(model)
    ops = [n.op_type for n in model.graph.node]
    assert ops == ["MatMul", "Mul", "Mul", "Add"]
    assert not {"Slice", "Concat", "Neg", "Gather", "Split"} & set(ops)
    assert [i.name for i in model.graph.initializer] == ["rot_perm"]
    assert [i.name for i in model.graph.input] == ["x", "cos", "sin_signed"]
    assert [o.name for o in model.graph.output] == ["y"]
