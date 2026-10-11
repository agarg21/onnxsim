"""RoPE rotate-half as MatMul + Mul + Add: no Slice, Concat or Neg.

llm_reference's RoPE is ``y = x*cos + rotate_half(x)*sin`` with
``rotate_half(x) = concat(-x[h:], x[:h])`` and ``h = hd/2``. The compiled graph
has no Slice, Concat or Neg, so the rotation is split in two:

* the permutation ``x @ P`` with the 0/1 matrix ``P`` (``P[k+h,k] = P[k,k+h] = 1``
  for ``k < h``): ``(x @ P) = concat(x[h:], x[:h])``, no signs;
* the sign folded into the sine table: ``sin_signed = sin * s`` with
  ``s = [-1]*h ++ [+1]*h``.

Then ``y = x*cos + (x @ P) * sin_signed``, which is bitwise the reference's
``_rot`` in float32 (the products are the same, only the zeros of the
permutation are extra terms that add exactly 0).

The graph runs on one head row ``[1, hd]`` (or ``[R, hd]`` for a multi-row
build). ``P`` is an initializer; the calibration weight that the quantizer sees
is ``P * 127.5/127``, which makes ``weight_codes`` produce codes of 0 and 127
whose dequantized values are exactly 0 and 1.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import numpy_helper, parser

# The emitter convention (linear_emit.weight_codes): per output channel
# s = max|w| / 127.5, code = clip(rint(w / s), -128, 127), dequant = code * s.
CALIBRATION_GAIN = np.float32(127.5 / 127)


def _half(hd: int) -> int:
    if hd < 2 or hd % 2:
        raise ValueError(f"head_dim must be even and >= 2, got {hd}")
    return hd // 2


def perm_matrix(hd: int) -> np.ndarray:
    """P [hd, hd] float32 with (x @ P)[k] = x[k+h] and (x @ P)[k+h] = x[k]."""
    h = _half(hd)
    p = np.zeros((hd, hd), np.float32)
    k = np.arange(h)
    p[k + h, k] = 1.0
    p[k, k + h] = 1.0
    return p


def sign_vector(hd: int) -> np.ndarray:
    """s = [-1]*h ++ [+1]*h float32."""
    h = _half(hd)
    return np.concatenate([np.full(h, -1.0, np.float32), np.ones(h, np.float32)])


def sin_signed(sin: np.ndarray) -> np.ndarray:
    """The sine table with the rotate-half sign folded in (exact: a sign flip)."""
    sin = np.asarray(sin, np.float32)
    return (sin * sign_vector(sin.shape[-1])).astype(np.float32)


def calibration_weight(hd: int) -> np.ndarray:
    """P * 127.5/127: the matrix the quantizer calibrates, so that its codes are
    exactly 0 / 127 with scale 1/127 and dequantize to P."""
    return (perm_matrix(hd) * CALIBRATION_GAIN).astype(np.float32)


def weight_codes(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(codes int8 [in, out], s_w [out] float32)`` of a float weight ``w [in, out]``,
    with the emitter convention (scale per output channel, codes clipped to int8)."""
    w = np.asarray(w, np.float32)
    if not np.isfinite(w).all():
        raise ValueError("the weight matrix holds a non-finite value")
    s = (np.abs(w).max(axis=0) / np.float32(127.5)).astype(np.float32)
    if not s.all():
        raise ValueError("an output channel of the weight matrix is all zero")
    codes = np.clip(np.rint(w / s), -128, 127).astype(np.int8)
    return codes, s


def perm_initializer(hd: int, name: str = "rot_perm") -> onnx.TensorProto:
    """The float permutation P as an initializer (exact 0/1 values)."""
    return numpy_helper.from_array(perm_matrix(hd), name)


def rotate_half_matmul(x: np.ndarray, hd: int | None = None) -> np.ndarray:
    """x @ P for x [..., hd] (numpy form of the MatMul in the graph)."""
    x = np.asarray(x, np.float32)
    return (x @ perm_matrix(x.shape[-1] if hd is None else hd)).astype(np.float32)


def apply(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """The decomposition in numpy: x*cos + (x @ P) * sin_signed, float32."""
    x = np.asarray(x, np.float32)
    return (x * cos + rotate_half_matmul(x) * sin_signed(sin)).astype(np.float32)


def attach_initializers(model: onnx.ModelProto, consts: dict) -> onnx.ModelProto:
    """Turn the graph inputs named in `consts` into initializers holding those arrays.

    The text form declares every constant as an input (the text format cannot hold
    big literals); this moves them out of the input list, as the emitter stores them."""
    graph = model.graph
    keep = [i for i in graph.input if i.name not in consts]
    del graph.input[:]
    graph.input.extend(keep)
    graph.initializer.extend(
        numpy_helper.from_array(np.asarray(v), k) for k, v in consts.items()
    )
    return model


def rope_rotate_half_graph(hd: int, rows: int = 1) -> onnx.ModelProto:
    """y = x*cos + MatMul(x, P)*sin_signed over [rows, hd]: MatMul, Mul and Add only.

    Inputs x, cos and sin_signed ([rows, hd] each; sin_signed = sin_signed(sin));
    initializer ``rot_perm`` (P)."""
    text = f"""
    <ir_version: 8, opset_import: ["" : 17]>
    rope_rotate_half (float[{rows},{hd}] x, float[{rows},{hd}] cos,
                      float[{rows},{hd}] sin_signed, float[{hd},{hd}] rot_perm)
        => (float[{rows},{hd}] y)
    {{
        rot = MatMul(x, rot_perm)
        xc = Mul(x, cos)
        rs = Mul(rot, sin_signed)
        y = Add(xc, rs)
    }}
    """
    model = parser.parse_model(text)
    return attach_initializers(model, {"rot_perm": perm_matrix(hd)})


def rope_tables(hd: int, theta: float, pos) -> tuple[np.ndarray, np.ndarray]:
    """cos/sin [..., hd] of llm_reference (HF layout, float64 angles, float32 result)."""
    inv_freq = 1.0 / (theta ** (np.arange(0, hd, 2, dtype=np.float64) / hd))
    ang = np.asarray(pos, dtype=np.float64)[..., None] * inv_freq
    ang = np.concatenate([ang, ang], axis=-1)
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
