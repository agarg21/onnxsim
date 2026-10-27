"""Fake-quant evaluation of a Pulsar-free program the way the card runs it:
every input, every inner tensor and the output rounded at its own U16
calibration (MatMul operands signed: s16 codes, zero point 0), float math in
between; linears with the card's s8 per-output-channel weights.

Fused graphs are evaluated node by node (Mul, Add, Div, Sqrt, ReduceMean,
MatMul, Transpose, Softmax, Sigmoid, Reshape); a tensor without a calibration
(a Reshape's output, or every tensor with an empty calibration) passes through
unrounded, so ``FusedEval(model, {})`` is the float function of the program.

``card_add=True`` evaluates each U16 Add of two calibrated tensors as the card
does (``card_add``) instead of in float: the scale ratios are u16 fixed point,
and Pulsar2's offset word takes the unrounded ratios, so the sum carries a
constant bias of ``zp_a * (A - r_a) + zp_b * (B - r_b)`` output codes (up to
~1.5 codes when the zero points are large and a ratio is just above 1).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import onnx

_AXERA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AXERA not in sys.path:
    sys.path.insert(0, _AXERA)

F32 = np.float32


def fq(a, sz, signed: bool = False) -> np.ndarray:
    """``a`` rounded to the U16 grid ``(scale, zero point)`` (s16 when ``signed``)."""
    s, zp = sz
    if s == 0:
        return np.asarray(a, np.float64)
    lo, hi = (-32768, 32767) if signed else (0, 65535)
    return (np.clip(np.rint(np.asarray(a, np.float64) / s) + zp, lo, hi) - zp) * s


def _softmax(a):
    e = np.exp(a - a.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def _node(op, x, at):
    if op == "Mul":
        return x[0] * x[1]
    if op == "Add":
        return x[0] + x[1]
    if op == "Div":
        return x[0] / x[1]
    if op == "Sqrt":
        return np.sqrt(x[0])
    if op == "ReduceMean":
        return x[0].mean(
            axis=tuple(at.get("axes", [-1])), keepdims=bool(at.get("keepdims", 1))
        )
    if op == "MatMul":
        return np.matmul(x[0], x[1])
    if op == "Transpose":
        return np.transpose(x[0], at.get("perm"))
    if op == "Softmax":
        return _softmax(x[0])
    if op == "Sigmoid":
        return 1 / (1 + np.exp(-x[0]))
    if op == "Reshape":
        return x[0].reshape([int(d) for d in x[1]])
    raise NotImplementedError(op)


def card_add(a, b, ca, cb, cy, exact_offset: bool = False) -> np.ndarray:
    """``a + b`` the way the card's U16 Add computes it, ``a`` and ``b`` on
    their grids ``ca`` / ``cb`` ``(scale, zero point)``, the result on ``cy``'s:
    ``(qa * A + qb * B + off) / 2^n`` rounded half up, with the ratios
    ``A, B = round(s_a / s_y * 2^n), round(s_b / s_y * 2^n)`` (``n = 15 - k``,
    ``u16_retarget.add_shift``) and Pulsar2's offset word
    ``off = int((zp_y - zp_a * f32(s_a / s_y) - zp_b * f32(s_b / s_y)) * 2^n)``
    (``u16_retarget.candidates``), which takes the unrounded ratios.
    ``exact_offset``: the program carries ``u16_retarget.exact_add_offset``
    instead (``pf_add_offset.py``)."""
    import u16_retarget as ur

    (sa, za), (sb, zb), (sy, zy) = ca, cb, cy
    ra, rb = sa / sy, sb / sy
    n = 15 - ur.add_shift(ra, rb)
    big_a, big_b = round(ra * 2.0**n), round(rb * 2.0**n)
    off = int((zy - za * float(F32(ra)) - zb * float(F32(rb))) * 2**n)
    if exact_offset:
        off = zy * 2**n - za * big_a - zb * big_b
    qa = np.rint(np.asarray(a, np.float64) / sa).astype(np.int64) + za
    qb = np.rint(np.asarray(b, np.float64) / sb).astype(np.int64) + zb
    acc = qa * big_a + qb * big_b + off
    q = np.clip((acc + 2 ** (n - 1)) >> n, 0, 65535)
    return (q - zy) * float(sy)


def add_bias(ca, cb, cy) -> float:
    """``card_add``'s systematic error in output codes: with the ratios in
    fixed point and the offset word from the unrounded ratios, the card's sum
    is off by ``(zp_a * A + zp_b * B + off) / 2^n - zp_y`` codes (plus the
    negligible relative ratio error on the inputs' own values)."""
    import u16_retarget as ur

    (sa, za), (sb, zb), (sy, zy) = ca, cb, cy
    ra, rb = sa / sy, sb / sy
    n = 15 - ur.add_shift(ra, rb)
    off = int((zy - za * float(F32(ra)) - zb * float(F32(rb))) * 2**n)
    return (za * round(ra * 2.0**n) + zb * round(rb * 2.0**n) + off) / 2.0**n - zy


def debias_add(ca, cb, lo, hi, widen=0.02, tol=0.02) -> tuple[float, int]:
    """The U16 MinMax calibration of an Add's output range ``[lo, hi]``,
    widened by the least relative amount (at most ``widen``) that brings
    ``add_bias`` within ``tol`` codes (else the widening with the smallest
    bias): the fixed-point ratios' rounding moves with the output scale. A
    ratio of a few hundred units (an input range much narrower than the
    output's) needs a widening of up to ~1 / that to sweep one unit."""
    import u16_retarget as ur

    best = None
    for t in np.arange(0.0, widen, 2e-6):
        cy = ur.unsigned_minmax(lo * (1 + t), hi * (1 + t))
        b = abs(add_bias(ca, cb, cy))
        if b <= tol:
            return cy
        if best is None or b < best[0]:
            best = (b, cy)
    return best[1]


class FusedEval:
    """A fused program's float graph at calibration ``calib`` (tensor ->
    ``(scale, zero point)``); ``signed`` names the s16 tensors. ``card_add``:
    Adds of two calibrated tensors as the card computes them (``card_add``),
    with ``exact_offset`` their offset words as ``pf_add_offset.py`` writes them."""

    def __init__(
        self,
        model,
        calib,
        signed=(),
        card_add: bool = False,
        exact_offset: bool = False,
    ):
        m = model if isinstance(model, onnx.ModelProto) else onnx.load(model)
        self.consts = {
            i.name: onnx.numpy_helper.to_array(i) for i in m.graph.initializer
        }
        self.nodes = [
            (
                n.op_type,
                list(n.input),
                n.output[0],
                {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute},
            )
            for n in m.graph.node
        ]
        self.outs = [o.name for o in m.graph.output]
        self.out = self.outs[0]
        self.cal = {k: tuple(v) for k, v in calib.items()}
        self.signed = set(signed)
        self.card_add, self.exact_offset = card_add, exact_offset

    def _card_add_ok(self, ins, out) -> bool:
        """Both operands and the output on U16 grids (no constant, no s16)."""
        ts = (*ins, out)
        return all(
            t in self.cal
            and t not in self.consts
            and t not in self.signed
            and self.cal[t][0] > 0
            for t in ts
        )

    def _r(self, name, a):
        if name in self.cal and name not in self.consts:
            return fq(a, self.cal[name], name in self.signed)
        return np.asarray(a, np.float64)

    def __call__(self, **feeds):
        """The output (a tuple of the outputs, in graph order, for several)."""
        v = self.trace(**feeds)
        if len(self.outs) > 1:
            return tuple(v[o].astype(F32) for o in self.outs)
        return v[self.out].astype(F32)

    def trace(self, **feeds) -> dict:
        """Every tensor of the graph (float64), inputs and constants included."""
        v = {
            k: (a.astype(np.float64) if a.dtype.kind == "f" else a)
            for k, a in self.consts.items()
        }
        for k, a in feeds.items():
            v[k] = self._r(k, a)
        for op, ins, o, at in self.nodes:
            if op == "Add" and self.card_add and self._card_add_ok(ins, o):
                v[o] = card_add(
                    *(v[i] for i in ins),
                    *(self.cal[t] for t in (*ins, o)),
                    exact_offset=self.exact_offset,
                )
                continue
            # an operand quantized for a consumer MatMul (signed) is rounded where it is produced
            v[o] = self._r(o, _node(op, [v[i] for i in ins], at))
        return v


def s8_weights(w) -> np.ndarray:
    """The card's s8 per-output-channel weights of ``w [in, out]``, dequantized."""
    import linear_emit as le

    codes, s_w = le.weight_codes(np.asarray(w, F32))
    return ((codes.astype(np.float64) - 128) * s_w[:, None].astype(np.float64)).T


class LinearEval:
    """``y = fq(fq(x) @ W8, y calibration)``, ``W8`` the card's s8 weights
    (``w8=False``: the float weights, activations still rounded)."""

    def __init__(self, w, cx, cy, w8: bool = True):
        w = s8_weights(w) if w8 else np.asarray(w, np.float64)
        self.w, self.cx, self.cy = np.asarray(w, np.float64), tuple(cx), tuple(cy)

    def __call__(self, x) -> np.ndarray:
        return fq(fq(x, self.cx) @ self.w, self.cy).astype(F32)


class AddEval:
    """A residual Add at its calibration (``a``, ``b``, ``y``); ``card_add``:
    the card's fixed-point Add (``card_add``) instead of a float sum."""

    def __init__(self, cal, card_add: bool = False, exact_offset: bool = False):
        self.cal = {k: tuple(v) for k, v in cal.items()}
        self.card_add, self.exact_offset = card_add, exact_offset

    def __call__(self, a, b) -> np.ndarray:
        a, b = fq(a, self.cal["a"]), fq(b, self.cal["b"])
        if self.card_add:
            c = self.cal
            return card_add(
                a, b, c["a"], c["b"], c["y"], exact_offset=self.exact_offset
            ).astype(F32)
        return fq(a + b, self.cal["y"]).astype(F32)
