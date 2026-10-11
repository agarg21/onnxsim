"""One decoder layer and the post model as the op sequence the emitters compile.

The graphs are what `pulsar2 llm_build` would be given for one decode step: a
single token, the KV cache rows of earlier tokens as inputs, the caches extended
with the new row inside the graph. Only ops the Pulsar-free emitters have
(MatMul, Add, Mul, Div, Sqrt, ReduceMean, Softmax, Sigmoid, Concat, Reshape,
Transpose) appear, and the decomposition is:

* RMSNorm: Mul, ReduceMean, Add (eps), Sqrt, Div, Mul (gain);
* linear: MatMul with the weight transposed to [in, out], plus Add when the model
  has a bias;
* RoPE: rope_rotate_half (MatMul + Mul + Add; sign folded into the sine table);
  the 1/sqrt(head_dim) softmax scale is folded into the q RoPE constants, so the
  scores need no scale op;
* grouped-query attention: q is reshaped to [nkv, group, hd], the caches are
  concatenated with the new row and batched-MatMul'd per kv head; Softmax over
  the S + 1 scores;
* SiLU as Mul(g, Sigmoid(g)), then the gate Mul with up.

Inputs of the layer graph (float32): input [1, H], K_cache [S, nkv*hd] (rotated
keys of the earlier tokens), V_cache [S, nkv*hd], mask [1, S+1] (additive),
cos_q / sinq [1, hd] (already scaled by 1/sqrt(hd)), cos_k / sink [1, hd] (sink
= sin_signed). Outputs: output [1, H], K_cache_out and V_cache_out [S+1, nkv*hd].
Everything else (weights, eps, the permutation, shape constants) is an
initializer.

`llm_reference.Model.layer_step` is the reference: `layer_outputs` returns its
(y, k_row, v_row) with k_row = K_cache_out[-1], v_row = V_cache_out[-1].
"""

from __future__ import annotations

import os
import sys

import numpy as np
import onnx
from onnx import parser
from onnx.reference import ReferenceEvaluator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import llm_reference as ref  # noqa: E402
import rope_rotate_half as rr  # noqa: E402

# Synthetic configs of the tests (the three shapes the earlier tests used).
# qwen3_like: q/k RMSNorm, head_dim 16 with q width 4*16 != hidden 32, tied head.
# llama_gqa: four query heads share two kv heads, tied head.
# llama_mha: one kv head per query head, untied lm_head.
SYNTHETIC = {
    "qwen3_like": dict(
        model_type="qwen3",
        hidden=32,
        heads=4,
        kv_heads=2,
        head_dim=16,
        intermediate=64,
        vocab=64,
        layers=2,
        qk_norm=True,
        tied=True,
        theta=1e6,
        seed=1,
    ),
    "llama_gqa": dict(
        model_type="llama",
        hidden=32,
        heads=4,
        kv_heads=2,
        head_dim=8,
        intermediate=64,
        vocab=64,
        layers=2,
        qk_norm=False,
        tied=True,
        theta=1e4,
        seed=2,
    ),
    "llama_mha": dict(
        model_type="llama",
        hidden=32,
        heads=4,
        kv_heads=4,
        head_dim=8,
        intermediate=64,
        vocab=64,
        layers=2,
        qk_norm=False,
        tied=False,
        theta=1e4,
        seed=3,
    ),
}
MASK_NEG = -65536.0


def make_synthetic(name_or_config) -> ref.Model:
    """A random llama/qwen3-layout model of one SYNTHETIC config (or of a config dict
    with the same keys), as an in-memory llm_reference.Model (weights are a dict)."""
    c = SYNTHETIC[name_or_config] if isinstance(name_or_config, str) else name_or_config
    hd, nh, nkv = c["head_dim"], c["heads"], c["kv_heads"]
    H, inter, V = c["hidden"], c["intermediate"], c["vocab"]
    rng = np.random.default_rng(c["seed"])

    def w(*shape, scale):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    cfg = {
        "model_type": c["model_type"],
        "hidden_size": H,
        "num_hidden_layers": c["layers"],
        "num_attention_heads": nh,
        "num_key_value_heads": nkv,
        "head_dim": hd,
        "intermediate_size": inter,
        "vocab_size": V,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": c["tied"],
        "rope_theta": c["theta"],
        "eos_token_id": 0,
    }
    t = {"model.embed_tokens.weight": w(V, H, scale=1.0)}
    for i in range(c["layers"]):
        p = f"model.layers.{i}."
        t[p + "input_layernorm.weight"] = 1 + w(H, scale=0.1)
        t[p + "self_attn.q_proj.weight"] = w(nh * hd, H, scale=0.6)
        t[p + "self_attn.k_proj.weight"] = w(nkv * hd, H, scale=0.6)
        t[p + "self_attn.v_proj.weight"] = w(nkv * hd, H, scale=0.4)
        t[p + "self_attn.o_proj.weight"] = w(H, nh * hd, scale=0.4)
        if c["qk_norm"]:
            t[p + "self_attn.q_norm.weight"] = 2 + w(hd, scale=0.5)
            t[p + "self_attn.k_norm.weight"] = 2 + w(hd, scale=0.5)
        t[p + "post_attention_layernorm.weight"] = 1 + w(H, scale=0.1)
        t[p + "mlp.gate_proj.weight"] = w(inter, H, scale=0.2)
        t[p + "mlp.up_proj.weight"] = w(inter, H, scale=0.2)
        t[p + "mlp.down_proj.weight"] = w(H, inter, scale=0.2)
    t["model.norm.weight"] = 1 + w(H, scale=0.1)
    if not c["tied"]:
        t["lm_head.weight"] = w(V, H, scale=1.0)
    return ref.Model(cfg, t)


# ---- graph construction ---------------------------------------------------


def _shape(a: np.ndarray) -> str:
    return ",".join(str(d) for d in a.shape)


def _decl(dtype: str, name: str, shape) -> str:
    return f"{dtype}[{','.join(str(d) for d in shape)}] {name}"


def make_graph(
    name: str,
    runtime: dict,
    consts: dict,
    body: list,
    outputs: list,
) -> onnx.ModelProto:
    """A one-graph ONNX model. `runtime` maps input name -> shape (float32); `consts`
    maps name -> array, declared as inputs in the text and attached as initializers.
    `body` is a list of node lines; `outputs` the output names (float32, shape free)."""
    decls = [_decl("float", n, s) for n, s in runtime.items()]
    for n, a in consts.items():
        dt = "int64" if np.asarray(a).dtype == np.int64 else "float"
        decls.append(_decl(dt, n, np.asarray(a).shape))
    text = (
        '<ir_version: 8, opset_import: ["" : 17]>\n'
        f"{name} ({', '.join(decls)})\n"
        f"    => ({', '.join('float ' + o for o in outputs)})\n"
        "{\n    " + "\n    ".join(body) + "\n}\n"
    )
    model = parser.parse_model(text)
    return rr.attach_initializers(model, consts)


def rmsnorm_lines(p: str, src: str, gain: str, out: str, eps: str = "eps") -> list:
    """RMSNorm over the last axis: Mul, ReduceMean, Add, Sqrt, Div, Mul (gain)."""
    return [
        f"{p}_sq = Mul({src}, {src})",
        f"{p}_m = ReduceMean<axes=[-1], keepdims=1>({p}_sq)",
        f"{p}_e = Add({p}_m, {eps})",
        f"{p}_r = Sqrt({p}_e)",
        f"{p}_d = Div({src}, {p}_r)",
        f"{out} = Mul({p}_d, {gain})",
    ]


def rope_lines(p: str, src: str, cos: str, sin: str, out: str, perm: str) -> list:
    """RoPE with the sign folded into `sin` (sin_signed): MatMul, Mul, Mul, Add."""
    return [
        f"{p}_rot = MatMul({src}, {perm})",
        f"{p}_a = Mul({src}, {cos})",
        f"{p}_b = Mul({p}_rot, {sin})",
        f"{out} = Add({p}_a, {p}_b)",
    ]


def _layer_consts(model: ref.Model, li: int) -> dict:
    lw = model.layer_weights(li)
    hd = model.hd
    consts = {
        "eps": np.array([model.eps], np.float32),
        "ln1": lw["n1"].astype(np.float32),
        "ln2": lw["n2"].astype(np.float32),
        "wq_t": np.ascontiguousarray(lw["q"].T),
        "wk_t": np.ascontiguousarray(lw["k"].T),
        "wv_t": np.ascontiguousarray(lw["v"].T),
        "wo_t": np.ascontiguousarray(lw["o"].T),
        "wg_t": np.ascontiguousarray(lw["gate"].T),
        "wu_t": np.ascontiguousarray(lw["up"].T),
        "wd_t": np.ascontiguousarray(lw["down"].T),
        "rot_perm": rr.perm_matrix(hd),
        "shape_q": np.array([model.nh, hd], np.int64),
        "shape_k": np.array([model.nkv, hd], np.int64),
        "shape_kv_rows": np.array([1, model.kvdim], np.int64),
        "shape_att": np.array([1, model.qdim], np.int64),
    }
    for key in ("qb", "kb", "vb", "ob", "gateb", "upb", "downb"):
        if lw[key] is not None:
            raise ValueError("the decomposition does not model linear biases yet")
    if lw["qn"] is not None:
        consts["qn"] = lw["qn"].astype(np.float32)
        consts["kn"] = lw["kn"].astype(np.float32)
    return consts


def layer_graph(model: ref.Model, li: int, S: int) -> onnx.ModelProto:
    """The decode-step graph of layer `li` with a cache of S earlier tokens."""
    H, hd, nh, nkv = model.H, model.hd, model.nh, model.nkv
    g = nh // nkv
    kvd = model.kvdim
    consts = _layer_consts(model, li)
    body = []
    body += rmsnorm_lines("n1", "input", "ln1", "a")
    body += [
        "qp = MatMul(a, wq_t)",
        "q0 = Reshape(qp, shape_q)",
        "kp = MatMul(a, wk_t)",
        "k0 = Reshape(kp, shape_k)",
        "vp = MatMul(a, wv_t)",
        "v_row = Reshape(vp, shape_kv_rows)",
    ]
    q_src, k_src = "q0", "k0"
    if model.qk_norm:
        body += rmsnorm_lines("qnorm", "q0", "qn", "qg_")
        body += rmsnorm_lines("knorm", "k0", "kn", "kg_")
        q_src, k_src = "qg_", "kg_"
    body += rope_lines("rq", q_src, "cos_q", "sinq", "qr", "rot_perm")
    body += rope_lines("rk", k_src, "cos_k", "sink", "kr", "rot_perm")
    body += [
        "k_row = Reshape(kr, shape_kv_rows)",
        "K_cache_out = Concat<axis=0>(K_cache, k_row)",
        "V_cache_out = Concat<axis=0>(V_cache, v_row)",
        "Kh = Reshape(K_cache_out, shape_kv_t)",
        "Kt = Transpose<perm=[1,2,0]>(Kh)",
        "Vh = Reshape(V_cache_out, shape_kv_t)",
        "Vt = Transpose<perm=[1,0,2]>(Vh)",
        "qgrp = Reshape(qr, shape_grp)",
        "sc = MatMul(qgrp, Kt)",
        "sm = Add(sc, mask)",
        "p = Softmax<axis=-1>(sm)",
        "pv = MatMul(p, Vt)",
        "att = Reshape(pv, shape_att)",
        "op = MatMul(att, wo_t)",
        "x1 = Add(input, op)",
    ]
    body += rmsnorm_lines("n2", "x1", "ln2", "m")
    body += [
        "gate = MatMul(m, wg_t)",
        "up = MatMul(m, wu_t)",
        "sig = Sigmoid(gate)",
        "silu = Mul(gate, sig)",
        "act = Mul(silu, up)",
        "down = MatMul(act, wd_t)",
        "output = Add(x1, down)",
    ]
    consts["shape_kv_t"] = np.array([S + 1, nkv, hd], np.int64)
    consts["shape_grp"] = np.array([nkv, g, hd], np.int64)
    runtime = {
        "input": (1, H),
        "K_cache": (S, kvd),
        "V_cache": (S, kvd),
        "mask": (1, S + 1),
        "cos_q": (1, hd),
        "sinq": (1, hd),
        "cos_k": (1, hd),
        "sink": (1, hd),
    }
    return make_graph(
        f"layer_{li}",
        runtime,
        consts,
        body,
        ["output", "K_cache_out", "V_cache_out"],
    )


def post_graph(model: ref.Model) -> onnx.ModelProto:
    """RMSNorm with model.norm, then the head: logits [1, V] = norm(x) @ head_t.

    head_t [H, V] is the head weight transposed. It is a graph input fed with the
    transposed view of the stored [V, H] matrix (the embedding for a tied head),
    because the view is what makes the float32 product bitwise llm_reference's
    `x @ W.T`; a contiguous copy gives the same values to ~1e-7 only."""
    consts = {
        "eps": np.array([model.eps], np.float32),
        "ln_f": model.weight("model.norm.weight").astype(np.float32),
    }
    body = rmsnorm_lines("nf", "input", "ln_f", "xn")
    body += ["logits = MatMul(xn, head_t)"]
    return make_graph(
        "post",
        {"input": (1, model.H), "head_t": (model.H, model.vocab)},
        consts,
        body,
        ["logits"],
    )


def rope_inputs(model: ref.Model, pos: int) -> dict:
    """The per-position constants of one token: cos/sin for k, and for q with the
    1/sqrt(head_dim) softmax scale folded in. sin tables carry the rotate-half sign."""
    cos, sin = model.rope_cos_sin(pos)
    scale = np.float32(1.0) / np.float32(np.sqrt(model.hd))
    sin_s = rr.sin_signed(sin)
    return {
        "cos_q": (cos * scale).reshape(1, -1).astype(np.float32),
        "sinq": (sin_s * scale).reshape(1, -1).astype(np.float32),
        "cos_k": cos.reshape(1, -1).astype(np.float32),
        "sink": sin_s.reshape(1, -1).astype(np.float32),
    }


def mask_for(S: int, masked=()) -> np.ndarray:
    """Additive mask [1, S+1]: 0, with MASK_NEG at the cache rows listed in `masked`."""
    m = np.zeros((1, S + 1), np.float32)
    for j in masked:
        m[0, j] = MASK_NEG
    return m


def run(model: onnx.ModelProto, feeds: dict):
    return ReferenceEvaluator(model).run(None, feeds)


def layer_outputs(model: ref.Model, li: int, x, K, V, mask, pos: int):
    """The layer graph run on one token. Returns (y [H], K_cache_out, V_cache_out)."""
    S = K.shape[0]
    feeds = {
        "input": np.asarray(x, np.float32).reshape(1, model.H),
        "K_cache": np.asarray(K, np.float32).reshape(S, model.kvdim),
        "V_cache": np.asarray(V, np.float32).reshape(S, model.kvdim),
        "mask": np.asarray(mask, np.float32).reshape(1, S + 1),
        **rope_inputs(model, pos),
    }
    y, kout, vout = run(layer_graph(model, li, S), feeds)
    return y.reshape(model.H), kout, vout


def post_outputs(model: ref.Model, x) -> np.ndarray:
    feeds = {
        "input": np.asarray(x, np.float32).reshape(1, model.H),
        "head_t": model.weight(model._head).T,
    }
    (logits,) = run(post_graph(model), feeds)
    return logits.reshape(-1)


def decode_context(model: ref.Model, ids, li: int):
    """The inputs of layer `li` for the last token of `ids`, from the reference: the
    cache holds the rotated K and V rows of ids[:-1] (computed by llm_reference's
    own steps), and x is the new token's hidden state entering layer `li`.
    Returns (x, K, V, mask, pos) with K, V [S, nkv*hd] and mask [S + 1]."""
    state = model.new_state()
    for t in ids[:-1]:
        model.step(state, int(t))
    pos = len(ids) - 1
    x = model.embed([int(ids[-1])])[0]
    for lj in range(li):
        K, V = state["K"][lj], state["V"][lj]
        x, _, _ = model.layer_step(
            lj, x, pos, K, V, np.zeros(K.shape[0] + 1, np.float32)
        )
    K, V = state["K"][li], state["V"][li]
    return x, K, V, np.zeros(K.shape[0] + 1, np.float32), pos
