"""The decoder layer split into one-row programs, with a program ledger.

Each program is a small ONNX graph (built with llm_layer_decomposition.make_graph)
run by onnx.reference. What a program may hold follows the row rule:

* an elementwise op over [R, n] is one flat program over [1, R*n];
* a per-row reduction (ReduceMean over head_dim) or a per-row broadcast (Div by
  the [R, 1] inverse rms, the RoPE rotate-half MatMul) is one program per row;
* attention takes four one-row programs per query head: the score column
  MatMul [S+1, hd] x [hd, 1], the mask Add, Softmax over [1, S+1], and the PV
  MatMul [1, S+1] x [S+1, hd]; the score column is read as a row between the
  mask and the softmax (a layout change, not a program);
* the K and V cache concatenations and the head-merge concatenation are one
  program each.

Rows are concatenated into the flat buffers by layout, which costs no program.

With NATIVE_MULTIROW set (a multi-row build of the compiler), each per-row
reduction, per-row broadcast and rotate-half MatMul over [R, n] is a single
program. Elementwise and attention counts do not change, so a layer costs:

    SmolLM2-135M (9 q heads, 3 kv heads, hd 64):   81 rowwise, 71 native
    Qwen3-0.6B   (16 q heads, 8 kv heads, hd 128): 177 rowwise, 111 native
    (`expected_programs` is the formula; the 3 concatenation programs are the K and
    V cache concats and the head merge.)

and it is independent of the sequence length S.

`row_layer` returns the same (output, K_cache_out, V_cache_out) as
llm_layer_decomposition.layer_outputs; the ledger counts the programs it ran.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from typing import Optional

import numpy as np
from onnx.reference import ReferenceEvaluator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import llm_layer_decomposition as ld  # noqa: E402
import rope_rotate_half as rr  # noqa: E402

NATIVE_MULTIROW = False


class Ledger:
    """Counts the programs run, by kind."""

    def __init__(self):
        self.counts: Counter = Counter()

    def add(self, kind: str, n: int = 1) -> None:
        self.counts[kind] += n

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def _norm_programs(R: int, native: bool) -> int:
    return 6 if native else 4 + 2 * R


def _rope_programs(R: int, native: bool) -> int:
    return 4 if native else R + 3


def expected_programs(nh: int, nkv: int, qk_norm: bool, native: bool = False) -> int:
    """Programs of one decoder layer for one token, independent of the sequence length."""
    total = _norm_programs(1, native)  # input RMSNorm over [1, H]
    total += 3  # q, k, v projections
    if qk_norm:
        total += _norm_programs(nh, native) + _norm_programs(nkv, native)
    total += _rope_programs(nh, native) + _rope_programs(nkv, native)
    total += 4 * nh  # attention: score, mask, softmax, PV per query head
    total += 3  # K cache concat, V cache concat, head merge
    total += 1  # o projection
    total += 1  # residual
    total += _norm_programs(1, native)  # post-attention RMSNorm
    total += 2  # gate, up
    total += 3  # SiLU: Sigmoid, Mul, gate Mul
    total += 1  # down
    total += 1  # residual
    return total


# ---- program runner -------------------------------------------------------


def _program(L: Optional[Ledger], kind: str, feeds: dict, body: list, consts=None):
    """One program: `feeds` are its runtime inputs (float32), `consts` its initializers."""
    model = ld.make_graph(
        "prog",
        {k: np.asarray(v).shape for k, v in feeds.items()},
        consts or {},
        body,
        ["y"],
    )
    (y,) = ReferenceEvaluator(model).run(None, feeds)
    if L is not None:
        L.add(kind)
    return y


def _elementwise(L, op, *arrays):
    """One flat elementwise program over [1, n]: y = op(i0, i1, ...)."""
    feeds = {f"i{k}": np.asarray(a, np.float32) for k, a in enumerate(arrays)}
    args = ", ".join(feeds)
    return _program(L, "elementwise", feeds, [f"y = {op}({args})"])


def _add_eps(L, x, eps):
    return _program(
        L,
        "elementwise",
        {"i0": x},
        ["y = Add(i0, eps)"],
        {"eps": np.array([eps], np.float32)},
    )


def _linear(L, x, w_view):
    """MatMul [1, in] x [in, out]. The weight is fed as the transposed view of the
    checkpoint's [out, in] matrix, which keeps the BLAS path of llm_reference."""
    return _program(L, "linear", {"x": x, "w": w_view}, ["y = MatMul(x, w)"])


def _rmsnorm(L, x, gain, R, n, eps, native):
    """RMSNorm over R rows of n values held in the flat buffers x, gain: [1, R*n]."""
    sq = _elementwise(L, "Mul", x, x)
    if native:
        mean = _program(
            L,
            "reduce_mean",
            {"x": sq.reshape(R, n)},
            ["y = ReduceMean<axes=[-1], keepdims=1>(x)"],
        ).reshape(1, R)
    else:
        mean = np.concatenate(
            [
                _program(
                    L,
                    "reduce_mean",
                    {"x": sq[:, r * n : (r + 1) * n]},
                    ["y = ReduceMean<axes=[-1], keepdims=1>(x)"],
                )
                for r in range(R)
            ],
            axis=1,
        )
    rms = _elementwise(L, "Sqrt", _add_eps(L, mean, eps))  # [1, R]
    if native:
        d = _program(
            L,
            "div_row",
            {"x": x.reshape(R, n), "d": rms.reshape(R, 1)},
            ["y = Div(x, d)"],
        ).reshape(1, R * n)
    else:
        d = np.concatenate(
            [
                _program(
                    L,
                    "div_row",
                    {"x": x[:, r * n : (r + 1) * n], "d": rms[:, r : r + 1]},
                    ["y = Div(x, d)"],
                )
                for r in range(R)
            ],
            axis=1,
        )
    return _elementwise(L, "Mul", d, gain)


def _rope(L, flat, cos, sin_s, R, hd, native):
    """RoPE over R head rows of the flat [1, R*hd] buffer; cos, sin_s are [1, R*hd]
    (the per-position table tiled over the rows, sin_s with the rotate-half sign)."""
    P = rr.perm_matrix(hd)
    a = _elementwise(L, "Mul", flat, cos)
    if native:
        rot = _program(
            L, "rope_matmul", {"x": flat.reshape(R, hd), "p": P}, ["y = MatMul(x, p)"]
        ).reshape(1, R * hd)
    else:
        rot = np.concatenate(
            [
                _program(
                    L,
                    "rope_matmul",
                    {"x": flat[:, r * hd : (r + 1) * hd], "p": P},
                    ["y = MatMul(x, p)"],
                )
                for r in range(R)
            ],
            axis=1,
        )
    b = _elementwise(L, "Mul", rot, sin_s)
    return _elementwise(L, "Add", a, b)


def row_layer(
    model: ld.ref.Model,
    li: int,
    x,
    K,
    V,
    mask,
    pos: int,
    ledger: Optional[Ledger] = None,
    native: Optional[bool] = None,
):
    """Layer `li` for one token, as one-row programs.

    Same contract as llm_layer_decomposition.layer_outputs: returns
    (output [H], K_cache_out [S+1, nkv*hd], V_cache_out [S+1, nkv*hd]); the program
    counts go to `ledger` (a new one when None)."""
    native = NATIVE_MULTIROW if native is None else native
    L = ledger if ledger is not None else Ledger()
    H, hd, nh, nkv, eps = model.H, model.hd, model.nh, model.nkv, model.eps
    g = nh // nkv
    S = K.shape[0]
    lw = model.layer_weights(li)
    rope = ld.rope_inputs(model, pos)
    xin = np.asarray(x, np.float32).reshape(1, H)
    K = np.asarray(K, np.float32).reshape(S, model.kvdim)
    V = np.asarray(V, np.float32).reshape(S, model.kvdim)
    mask = np.asarray(mask, np.float32).reshape(S + 1)

    a = _rmsnorm(L, xin, lw["n1"].reshape(1, H), 1, H, eps, native)
    q = _linear(L, a, lw["q"].T)  # [1, nh*hd]
    k = _linear(L, a, lw["k"].T)
    v = _linear(L, a, lw["v"].T)
    if model.qk_norm:
        q = _rmsnorm(L, q, np.tile(lw["qn"], nh)[None], nh, hd, eps, native)
        k = _rmsnorm(L, k, np.tile(lw["kn"], nkv)[None], nkv, hd, eps, native)
    q = _rope(
        L,
        q,
        np.tile(rope["cos_q"], (1, nh)),
        np.tile(rope["sinq"], (1, nh)),
        nh,
        hd,
        native,
    )
    k = _rope(
        L,
        k,
        np.tile(rope["cos_k"], (1, nkv)),
        np.tile(rope["sink"], (1, nkv)),
        nkv,
        hd,
        native,
    )

    K_out = _program(L, "cache_concat", {"c": K, "r": k}, ["y = Concat<axis=0>(c, r)"])
    V_out = _program(L, "cache_concat", {"c": V, "r": v}, ["y = Concat<axis=0>(c, r)"])
    mask_col = mask.reshape(S + 1, 1)
    heads = []
    for h in range(nh):
        kv = h // g
        qcol = q[0, h * hd : (h + 1) * hd].reshape(hd, 1)
        Kh = K_out[:, kv * hd : (kv + 1) * hd]
        Vh = V_out[:, kv * hd : (kv + 1) * hd]
        s = _program(
            L, "attn_score", {"k": Kh, "q": qcol}, ["y = MatMul(k, q)"]
        )  # [S+1, 1]
        s = _program(L, "attn_mask", {"s": s, "m": mask_col}, ["y = Add(s, m)"])
        p = _program(
            L, "attn_softmax", {"x": s.reshape(1, S + 1)}, ["y = Softmax<axis=-1>(x)"]
        )  # [1, S+1]
        heads.append(_program(L, "attn_pv", {"p": p, "v": Vh}, ["y = MatMul(p, v)"]))
    names = [f"h{i}" for i in range(nh)]
    att = _program(
        L,
        "head_concat",
        dict(zip(names, heads)),
        [f"y = Concat<axis=1>({', '.join(names)})"],
    )  # [1, nh*hd]
    o = _linear(L, att, lw["o"].T)
    x1 = _elementwise(L, "Add", xin, o)
    m = _rmsnorm(L, x1, lw["n2"].reshape(1, H), 1, H, eps, native)
    gate = _linear(L, m, lw["gate"].T)
    up = _linear(L, m, lw["up"].T)
    sig = _elementwise(L, "Sigmoid", gate)
    act = _elementwise(L, "Mul", _elementwise(L, "Mul", gate, sig), up)
    down = _linear(L, act, lw["down"].T)
    y = _elementwise(L, "Add", x1, down)
    return y.reshape(H), K_out, V_out, L
