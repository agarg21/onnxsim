"""Numerical tests for ``fuse_attention``'s causal-mask and KV-cache coverage.

``fuse_attention.h`` originally only matched unmasked self-attention. These
tests build hand-written attention blocks with an additive mask between the
scaled ``Q @ K^T`` and the ``Softmax`` (the shape a from-scratch causal
attention emits), optionally with a KV cache (``past_key``/``past_value``
concatenated in front of the new K/V), and check two things per case:

  * which case is fused into a standard ``ai.onnx`` ``Attention`` (opset 23)
    node versus declined (left unfused), and
  * the simplified model's output matches the original on random inputs.

Declining is always an acceptable outcome for the numerical checks; the
per-case fusion assertions document the intended coverage.
"""

import collections

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

ort = pytest.importorskip("onnxruntime")


def _model(body, initializer=(), opset=23):
    model = parser.parse_model(
        f"""
        <
          ir_version: 10,
          opset_import: ["": {opset}]
        >
        {body}
        """
    )
    model.graph.initializer.extend(initializer)
    return model


def _f32(array, name):
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


def _i64(array, name):
    return onnx.numpy_helper.from_array(np.asarray(array, dtype=np.int64), name)


def _linear_body(rng, x_name, in_dim, out_dim, prefix):
    # MatMul(x, W) + Add(B, .) -- the plain ``nn.Linear`` MatMul export shape.
    w = _f32(rng.standard_normal((in_dim, out_dim)) * 0.3, f"{prefix}_w")
    b = _f32(rng.standard_normal(out_dim) * 0.1, f"{prefix}_b")
    text = (
        f"{prefix}_mm = MatMul({x_name}, {prefix}_w)\n"
        f"{prefix}_out = Add({prefix}_b, {prefix}_mm)\n"
    )
    return text, [w, b], f"{prefix}_out"


def _head_split_body(x_name, shape_name, perm, prefix):
    perm_str = ", ".join(str(p) for p in perm)
    text = (
        f"{prefix}_reshape = Reshape({x_name}, {shape_name})\n"
        f"{prefix}_transpose = Transpose<perm = [{perm_str}]>({prefix}_reshape)\n"
    )
    return text, f"{prefix}_transpose"


def _range_causal_mask_text(seq_len_name):
    # Causal additive mask [1,1,S,S] built at runtime from a sequence length
    # value (Range/Greater/Where/Unsqueeze), as a from-scratch implementation
    # with `torch.tril`-free code would. `seq_len_name` may be a constant
    # (static S, folded by constant propagation) or a Shape-derived value.
    return (
        f"cm_r = Range(zero_i, {seq_len_name}, one_i)\n"
        "cm_col = Unsqueeze(cm_r, axes0)\n"
        "cm_row = Unsqueeze(cm_r, axes1)\n"
        "cm_future = Greater(cm_col, cm_row)\n"
        "cm_neg = Where(cm_future, neg_inf, zero_f)\n"
        "mask4 = Unsqueeze(cm_neg, axes01)\n"
    )


def _range_causal_mask_inits(S):
    return [
        _i64(0, "zero_i"),
        _i64(1, "one_i"),
        _i64(S, "seq_len_static"),
        _f32(0.0, "zero_f"),
        _f32(-np.inf, "neg_inf"),
        _i64([0], "axes0"),
        _i64([1], "axes1"),
        _i64([0, 1], "axes01"),
    ]


def _mask_array(kind, Sq, Skv, P):
    i = np.arange(Sq)[:, None]
    j = np.arange(Skv)[None, :]
    if kind == "static_causal":  # j <= i, square.
        m = np.where(j <= i, 0.0, -np.inf)
    elif kind == "decode_bottom_right":  # j <= i + P: cache-aware causal.
        m = np.where(j <= i + P, 0.0, -np.inf)
    elif kind == "near_causal":  # j <= i + 1: superdiagonal also visible.
        m = np.where(j <= i + 1, 0.0, -np.inf)
    elif kind == "constant_padding":  # last key column padded out.
        m = np.zeros((Sq, Skv))
        m[:, Skv - 1] = -1e4
    else:
        raise ValueError(kind)
    return m.reshape(1, 1, Sq, Skv).astype(np.float32)


def _causal_kv_model(
    *,
    B=2,
    Sq=5,
    Skv=5,
    H=32,
    NH=4,
    mask="static_causal",
    opset=23,
    seed=0,
):
    """Builds ``y = Identity(ctx)`` for a masked self/cached attention block.

    ``mask`` selects how the additive mask feeding the ``Add`` is produced:
      static_causal        constant lower-triangular [1,1,S,S]
      static_range_causal  causal mask built by Range/Greater/Where, static S
      dynamic_causal       same construction but S read from Shape(x) at runtime
      decode_bottom_right  constant [1,1,Sq,Skv] mask, KV cache (Skv > Sq)
      near_causal          constant mask that is NOT exactly causal
      constant_padding     constant non-causal key-padding mask
      runtime_padding      padding mask supplied as a graph input [B,1,1,Skv]
    Skv > Sq implies a KV cache: past_key/past_value graph inputs are
    concatenated in front of the new K/V (the unfused decode shape).
    """
    rng = np.random.default_rng(seed)
    D = H // NH
    P = Skv - Sq
    dynamic = mask == "dynamic_causal"
    s_dim = "S" if dynamic else str(Sq)
    shape_q = [B, -1, NH, D] if dynamic else [B, Sq, NH, D]
    shape_ctx = [B, -1, H] if dynamic else [B, Sq, H]
    inits = [_i64(shape_q, "shape_q"), _i64(shape_ctx, "shape_ctx")]
    body = ""

    q_text, q_inits, q_out = _linear_body(rng, "x", H, H, "q")
    k_text, k_inits, k_out = _linear_body(rng, "x", H, H, "k")
    v_text, v_inits, v_out = _linear_body(rng, "x", H, H, "v")
    body += q_text + k_text + v_text
    inits += q_inits + k_inits + v_inits

    q_head_text, q_t = _head_split_body(q_out, "shape_q", [0, 2, 1, 3], "q")
    body += q_head_text
    inputs = [f"float[{B},{s_dim},{H}] x"]
    if P == 0:
        # Direct K^T head split, as in fuse_attention's own MHA tests.
        k_head_text, k_t = _head_split_body(k_out, "shape_q", [0, 2, 3, 1], "k")
        v_head_text, v_t = _head_split_body(v_out, "shape_q", [0, 2, 1, 3], "v")
        body += k_head_text + v_head_text
    else:
        # KV-cache decode: new K/V head-split to [B,NH,Sq,D], concatenated
        # after the cached past along the sequence axis, K then transposed
        # for the QK^T matmul.
        k_head_text, k_new = _head_split_body(k_out, "shape_q", [0, 2, 1, 3], "k")
        v_head_text, v_new = _head_split_body(v_out, "shape_q", [0, 2, 1, 3], "v")
        body += k_head_text + v_head_text
        body += f"k_all = Concat<axis = 2>(past_k, {k_new})\n"
        body += f"v_all = Concat<axis = 2>(past_v, {v_new})\n"
        body += "k_t = Transpose<perm = [0, 1, 3, 2]>(k_all)\n"
        k_t = "k_t"
        v_t = "v_all"
        inputs += [
            f"float[{B},{NH},{P},{D}] past_k",
            f"float[{B},{NH},{P},{D}] past_v",
        ]

    body += f"qk = MatMul({q_t}, {k_t})\n"
    inits.append(_f32(float(D) ** 0.5, "divisor"))
    body += "scores = Div(qk, divisor)\n"

    if mask in ("static_causal", "decode_bottom_right", "near_causal",
                "constant_padding"):
        inits.append(_f32(_mask_array(mask, Sq, Skv, P), "mask"))
        body += "masked = Add(scores, mask)\n"
    elif mask == "static_range_causal":
        inits += _range_causal_mask_inits(Sq)
        inits.append(_i64(Sq, "seq_len"))
        body += _range_causal_mask_text("seq_len_static")
        body += "masked = Add(scores, mask4)\n"
    elif mask == "dynamic_causal":
        inits += _range_causal_mask_inits(Sq)
        body += "x_shape = Shape(x)\n"
        body += "seq_len_dyn = Gather<axis = 0>(x_shape, one_i)\n"
        body += _range_causal_mask_text("seq_len_dyn")
        body += "masked = Add(scores, mask4)\n"
    elif mask == "runtime_padding":
        inputs.append(f"float[{B},1,1,{Skv}] pad")
        body += "masked = Add(scores, pad)\n"
    else:
        raise ValueError(mask)

    body += "attn = Softmax<axis = -1>(masked)\n"
    body += f"ctx0 = MatMul(attn, {v_t})\n"
    body += "ctx1 = Transpose<perm = [0, 2, 1, 3]>(ctx0)\n"
    body += "ctx2 = Reshape(ctx1, shape_ctx)\n"
    body += "y = Identity(ctx2)\n"

    return _model(
        f"""
        g ({", ".join(inputs)}) => (float[{B},{s_dim},{H}] y)
        {{
          {body}
        }}
        """,
        initializer=inits,
        opset=opset,
    )


def _feeds(seed, B=2, Sq=5, Skv=5, H=32, NH=4, mask="static_causal", x_seq=None):
    rng = np.random.default_rng(seed)
    S = Sq if x_seq is None else x_seq
    feeds = {"x": rng.standard_normal((B, S, H)).astype(np.float32)}
    P = Skv - Sq
    D = H // NH
    if P > 0:
        feeds["past_k"] = rng.standard_normal((B, NH, P, D)).astype(np.float32)
        feeds["past_v"] = rng.standard_normal((B, NH, P, D)).astype(np.float32)
    if mask == "runtime_padding":
        pad = np.zeros((B, 1, 1, Skv), dtype=np.float32)
        pad[:, :, :, Skv - 2 :] = -1e4  # two padded trailing keys, per row.
        feeds["pad"] = pad
    return feeds


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _assert_close(float_outputs, fused_outputs):
    for f, q in zip(float_outputs, fused_outputs):
        f = np.asarray(f, dtype=np.float64).ravel()
        q = np.asarray(q, dtype=np.float64).ravel()
        assert np.all(np.isfinite(q))
        rel_l2 = np.linalg.norm(f - q) / max(np.linalg.norm(f), 1e-6)
        assert rel_l2 < 1e-4, f"relative L2 error too large: {rel_l2:.6f}"


def _std_attention_nodes(model):
    # Standard ai.onnx Attention (domain "" or "ai.onnx"), not com.microsoft.
    return [
        n
        for n in model.graph.node
        if n.op_type == "Attention" and n.domain in ("", "ai.onnx")
    ]


def _attr_int(node, name, default=0):
    for a in node.attribute:
        if a.name == name:
            return a.i
    return default


def _check_equal(model, simplified, feeds):
    onnx.checker.check_model(simplified)
    _assert_close(_run(model, feeds), _run(simplified, feeds))


# --------------------------------------------------------------------------- #
# (a) causal mask, static sequence length
# --------------------------------------------------------------------------- #
def test_static_causal_constant_mask_fuses_with_is_causal():
    model = _causal_kv_model(mask="static_causal")
    simplified, ok = onnxsim.simplify(model)
    assert ok
    nodes = _std_attention_nodes(simplified)
    assert len(nodes) == 1
    attn = nodes[0]
    # Square, exactly lower-triangular mask: is_causal=1 and no mask input.
    assert _attr_int(attn, "is_causal") == 1
    assert len(attn.input) == 3
    assert not any(n.op_type == "Softmax" for n in simplified.graph.node)
    _check_equal(model, simplified, _feeds(100))


def test_static_causal_mask_built_from_range_fuses_with_is_causal():
    # The same causal mask, but built by a Range/Greater/Where subgraph with a
    # static S. Constant propagation should reduce it to the constant case.
    model = _causal_kv_model(mask="static_range_causal")
    simplified, ok = onnxsim.simplify(model)
    assert ok
    nodes = _std_attention_nodes(simplified)
    assert len(nodes) == 1
    assert _attr_int(nodes[0], "is_causal") == 1
    _check_equal(model, simplified, _feeds(101))


# --------------------------------------------------------------------------- #
# (b) causal mask built from a dynamic-length subgraph
# --------------------------------------------------------------------------- #
def test_dynamic_length_causal_mask_declines_but_stays_correct():
    # S comes from Shape(x) at runtime, so the mask shape is not statically
    # provable. The pass must decline (not guess a static S) and the graph
    # must stay correct for every sequence length it is run with.
    model = _causal_kv_model(mask="dynamic_causal")
    simplified, ok = onnxsim.simplify(model)
    assert ok
    assert _std_attention_nodes(simplified) == []
    for seq in (3, 5, 7):
        _check_equal(model, simplified, _feeds(102 + seq, Sq=seq, Skv=seq, x_seq=seq, mask="dynamic_causal"))


# --------------------------------------------------------------------------- #
# (c) KV-cache decode, Sq != Skv
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "Sq,Skv,n_inputs",
    [
        # Sq=1: the bottom-right mask is all zeros, so constant folding removes
        # the Add entirely and the unmasked KV-cache path fires (Q, K, V only).
        (1, 6, 3),
        # Sq=3: a real mask remains and is passed through as attn_mask.
        (3, 7, 4),
    ],
)
def test_kv_cache_decode_bottom_right_mask_fuses_without_is_causal(Sq, Skv, n_inputs):
    # Sq != Skv: is_causal alignment (bottom-right, offset by past length) is
    # not something this pass can prove matches the unfused mask, so the
    # exact mask is passed as the attn_mask input instead, with is_causal
    # left at 0.
    model = _causal_kv_model(Sq=Sq, Skv=Skv, mask="decode_bottom_right")
    simplified, ok = onnxsim.simplify(model)
    assert ok
    nodes = _std_attention_nodes(simplified)
    assert len(nodes) == 1
    attn = nodes[0]
    assert _attr_int(attn, "is_causal") == 0
    assert len(attn.input) == n_inputs
    _check_equal(
        model, simplified, _feeds(110 + Sq, Sq=Sq, Skv=Skv, mask="decode_bottom_right")
    )


# --------------------------------------------------------------------------- #
# (d) padding masks
# --------------------------------------------------------------------------- #
def test_runtime_padding_mask_declines_but_stays_correct():
    # Padding mask is a runtime graph input, not a constant: nothing to verify
    # against, so the block must stay unfused and correct.
    model = _causal_kv_model(mask="runtime_padding")
    simplified, ok = onnxsim.simplify(model)
    assert ok
    assert _std_attention_nodes(simplified) == []
    _check_equal(model, simplified, _feeds(120, mask="runtime_padding"))


def test_constant_padding_mask_fuses_as_attn_mask_not_is_causal():
    # A constant, non-causal additive mask is passed through exactly as the
    # attn_mask input (no causal semantics are assumed for it).
    model = _causal_kv_model(mask="constant_padding")
    simplified, ok = onnxsim.simplify(model)
    assert ok
    nodes = _std_attention_nodes(simplified)
    assert len(nodes) == 1
    assert _attr_int(nodes[0], "is_causal") == 0
    assert len(nodes[0].input) == 4
    _check_equal(model, simplified, _feeds(121))


def test_near_causal_mask_is_not_misfused_as_is_causal():
    # Superdiagonal visible (j <= i + 1): square but NOT lower-triangular.
    # Converting it to is_causal=1 would silently change the output, so it
    # must go through attn_mask.
    model = _causal_kv_model(mask="near_causal")
    simplified, ok = onnxsim.simplify(model)
    assert ok
    nodes = _std_attention_nodes(simplified)
    assert len(nodes) == 1
    assert _attr_int(nodes[0], "is_causal") == 0
    _check_equal(model, simplified, _feeds(122))


# --------------------------------------------------------------------------- #
# Opset gate: ai.onnx Attention exists only from opset 23.
# --------------------------------------------------------------------------- #
def test_causal_mask_declines_below_opset_23():
    model = _causal_kv_model(mask="static_causal", opset=17)
    simplified, ok = onnxsim.simplify(model)
    assert ok
    assert _std_attention_nodes(simplified) == []
    _check_equal(model, simplified, _feeds(130))
