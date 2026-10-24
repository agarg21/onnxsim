"""Fake-quant evaluation of a Pulsar-free program the way the card runs it:
every input, every inner tensor and the output rounded at its own U16
calibration (MatMul operands signed: s16 codes, zero point 0), float math in
between; linears with the card's s8 per-output-channel weights.

Fused graphs are evaluated node by node (Mul, Add, Div, Sqrt, ReduceMean,
MatMul, Transpose, Softmax, Sigmoid, Reshape); a tensor without a calibration
(a Reshape's output, or every tensor with an empty calibration) passes through
unrounded, so ``FusedEval(model, {})`` is the float function of the program.
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


class FusedEval:
    """A fused program's float graph at calibration ``calib`` (tensor ->
    ``(scale, zero point)``); ``signed`` names the s16 tensors."""

    def __init__(self, model, calib, signed=()):
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
        self.out = m.graph.output[0].name
        self.cal = {k: tuple(v) for k, v in calib.items()}
        self.signed = set(signed)

    def _r(self, name, a):
        if name in self.cal and name not in self.consts:
            return fq(a, self.cal[name], name in self.signed)
        return np.asarray(a, np.float64)

    def __call__(self, **feeds) -> np.ndarray:
        v = {
            k: (a.astype(np.float64) if a.dtype.kind == "f" else a)
            for k, a in self.consts.items()
        }
        for k, a in feeds.items():
            v[k] = self._r(k, a)
        for op, ins, o, at in self.nodes:
            # an operand quantized for a consumer MatMul (signed) is rounded where it is produced
            v[o] = self._r(o, _node(op, [v[i] for i in ins], at))
        return v[self.out].astype(F32)


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
    """A residual Add at its calibration (``a``, ``b``, ``y``)."""

    def __init__(self, cal):
        self.cal = {k: tuple(v) for k, v in cal.items()}

    def __call__(self, a, b) -> np.ndarray:
        return fq(fq(a, self.cal["a"]) + fq(b, self.cal["b"]), self.cal["y"]).astype(
            F32
        )
