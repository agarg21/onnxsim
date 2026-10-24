"""BEVFormer-tiny's encoder in tinygrad, the tinygrad counterpart of the hand path in ../msda_hvx/.

The encoder is the torch `model.Encoder` (reference=False path) with every op re-expressed in tinygrad
Tensors. The weights are the same tensors (`Encoder.state_dict()`), so the only difference is the
execution. The multi-scale deformable attention is one function, `msda`, with the hand kernel's contract
(../../msda_hvx/msda_ref.py `msda_reference`, mode "pix"):

  out[q, h*D:(h+1)*D] = 1 / max(1, #visible maps) * sum over visible maps v, points p, bilinear taps
                        attw[q, h, o, p] * bilinear(value[v, :, h*D:(h+1)*D], ref[v, q, p % R] * (W, H) - 0.5 + off)

Every map, point and tap is one vectorized gather of (NV, Q, M, P, D) rows, so tinygrad sees 4 gathers and
one reduction per level instead of NV * P * 4 scalar chains. The per-tap gathers use the same tensor
indexing the fork's oracle (test/external/dsp/hand/msda/tg_msda.py) relies on.

usage: from bevformer_tg import TinyEncoder; enc = TinyEncoder.from_torch(torch_encoder); q = enc(*enc_in)
"""
from __future__ import annotations

import functools
import pathlib
import sys

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import Ops

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import model as M  # noqa: E402  (the torch reference: constants, ref_2d, the Encoder for weights)

E, HEADS = M.EMBED, M.HEADS
D = E // HEADS
NQ, BEV_H, BEV_W = M.NQ, M.BEV_H, M.BEV_W
FH, FW = M.FH, M.FW
TAPS = ((0, 0), (1, 0), (0, 1), (1, 1))
BRANCH_FREE = __import__("os").environ.get("MSDA_BRANCH_FREE", "0") == "1"   # experiment switch


def gather_rows(flat: Tensor, idx: Tensor, d: int) -> Tensor:
  """flat (R*d,) realized, idx int (...) -> (..., d): row idx of flat viewed as (R, d), as a direct indexed load.

  Tensor's own indexing (__getitem__, gather) is a one-hot compare + sum over all R rows, which tinygrad realizes
  per tap. A UOp INDEX with the computed element index instead loads the d values directly (needs the rangeify
  fix in tinygrad's schedule/indexing.py: the index args of INDEX take the output's ranges, the buffer does not)."""
  elems = idx.reshape(-1, 1) * d + Tensor.arange(d, dtype=dtypes.int32).reshape(1, d)
  return Tensor(flat.uop.index(elems.reshape(-1).uop)).reshape(*idx.shape, d)


def _sample_coords(ref: Tensor, off: Tensor, attw: Tensor, vis: Tensor | None, hw: tuple[int, int]):
  """The per-sample math shared by the taps: -> (shared (5, NV, Q, M, P) realized: floor x, floor y, fx, fy and the
  attention weight zeroed where the map does not see the query; inv (Q,) = 1 / max(1, #visible maps))."""
  H, W = hw
  NV, Q, P = ref.shape[0], off.shape[0], off.shape[3]
  R = ref.shape[2]
  NO = off.shape[2]
  refp = Tensor.stack(*[ref[:, :, p % R] for p in range(P)], dim=2)
  wh = Tensor([float(W), float(H)], dtype=dtypes.float32)
  off_v = off.permute(2, 0, 1, 3, 4)
  att_v = attw.permute(2, 0, 1, 3)
  if NO == 1 and NV > 1:
    off_v, att_v = off_v.expand(NV, -1, -1, -1, -1), att_v.expand(NV, -1, -1, -1)
  # the contract's order: loc = ref + off / (W, H), then pixel = loc * (W, H) - 0.5 (see msda's note)
  px = (refp.reshape(NV, Q, 1, P, 2) + off_v / wh) * wh - 0.5
  x, y = px[..., 0], px[..., 1]
  x0f, y0f = x.floor(), y.floor()
  fx, fy = x - x0f, y - y0f
  if vis is None:
    count = Tensor.full((Q,), float(NV), dtype=dtypes.float32)
    att_s = att_v.expand(NV, Q, HEADS, P)
  else:
    count = vis.sum(0)
    att_s = att_v.expand(NV, Q, HEADS, P) * vis.reshape(NV, Q, 1, 1)
  inv = (1.0 / count.maximum(1.0)).realize()
  shared = Tensor.stack(x0f, y0f, fx, fy, att_s, dim=0).realize()
  return shared, inv


def msda(value: Tensor, hw: tuple[int, int], ref: Tensor, off: Tensor, attw: Tensor, vis: Tensor | None) -> Tensor:
  """One level of the hand kernel's contract, vectorized.

  value (NV, H*W, E) float32: channels-last value maps (NV = 2 queue frames for TSA, 6 cameras for SCA).
  hw (H, W); ref (NV, Q, R, 2) normalized reference points, point p uses anchor p % R.
  off (Q, M, NO, P, 2) raw sampling offsets in pixels; attw (Q, M, NO, P) softmaxed; NO is 1 (shared) or NV.
  vis (NV, Q) float {0, 1} or None (all visible). Returns (Q, E).
  """
  H, W = hw
  NV, S = ref.shape[0], H * W                    # one level: S = H * W rows per map
  Q, P = off.shape[0], off.shape[3]
  R = ref.shape[2]
  NO = off.shape[2]

  # per-point reference: (NV, Q, P, 2) in pixels, then (NV, Q, 1, P, 2) to broadcast over heads
  refp = Tensor.stack(*[ref[:, :, p % R] for p in range(P)], dim=2)
  wh = Tensor([float(W), float(H)], dtype=dtypes.float32)
  off_v = off.permute(2, 0, 1, 3, 4)                      # (NO, Q, M, P, 2)
  att_v = attw.permute(2, 0, 1, 3)                        # (NO, Q, M, P)
  if NO == 1 and NV > 1:
    off_v, att_v = off_v.expand(NV, -1, -1, -1, -1), att_v.expand(NV, -1, -1, -1)
  # the contract's order: loc = ref + off / (W, H) (normalized), then pixel = loc * (W, H) - 0.5. The other
  # order, (ref * W - 0.5) + off, rounds differently and flips floor() at pixel boundaries (one wrong sample
  # of 0.1 on the real TSA frame, msda_io/l0_tsa)
  px = (refp.reshape(NV, Q, 1, P, 2) + off_v / wh) * wh - 0.5  # (NV, Q, M, P, 2)
  x, y = px[..., 0], px[..., 1]
  x0f, y0f = x.floor(), y.floor()
  fx, fy = x - x0f, y - y0f
  # the taps share the sample coordinates and attention weights: one realized tensor (one kernel), read by all taps.
  # Left lazy, each tap recomputes the coordinate math (float divides, floor) and tinygrad realizes it per tap
  shared = Tensor.stack(x0f, y0f, fx, fy, att_v.expand(NV, Q, HEADS, P), dim=0).realize()
  x0f, y0f, fx, fy, att_s = shared[0], shared[1], shared[2], shared[3], shared[4]

  if vis is None:
    count = Tensor.full((Q,), float(NV), dtype=dtypes.float32)
    vmask = None
  else:
    count = vis.sum(0)
    vmask = vis.reshape(NV, Q, 1, 1) != 0
  inv = 1.0 / count.maximum(1.0)                          # (Q,)

  # row of (value map v, pixel s, head m) in value.reshape(NV * S * M, D)
  head = Tensor.arange(HEADS, dtype=dtypes.int32).reshape(1, 1, HEADS, 1)
  vofs = (Tensor.arange(NV, dtype=dtypes.int32) * S).reshape(NV, 1, 1, 1)
  rows = (value if value.ndim == 1 else value.reshape(-1)).realize()   # flat: a reshape of a buffer keeps a RESHAPE view
  out = None
  for dx, dy in TAPS:
    xt, yt = x0f + dx, y0f + dy
    ok = (xt >= 0) & (xt < W) & (yt >= 0) & (yt < H)
    if vmask is not None:
      ok = ok & vmask
    w = ((fx if dx else 1.0 - fx) * (fy if dy else 1.0 - fy)) * att_s
    if BRANCH_FREE:
      # no per-element select: clamp the pixel into the map and multiply by the 0/1 mask (vectorizes; 0 * finite = 0)
      xc, yc = xt.clip(0.0, float(W - 1)), yt.clip(0.0, float(H - 1))
      idx = ((vofs + yc.cast(dtypes.int32) * W + xc.cast(dtypes.int32)) * HEADS + head)
      val = gather_rows(rows, idx, D)
      term = (val * ok.where(w, 0.0).unsqueeze(-1)).sum(axis=(0, 3))
    else:
      idx = ((vofs + yt.cast(dtypes.int32) * W + xt.cast(dtypes.int32)) * HEADS + head)   # (NV, Q, M, P)
      idx = ok.where(idx, 0)
      val = gather_rows(rows, idx, D)                        # (NV, Q, M, P, D)
      term = (val * ok.where(w, 0.0).unsqueeze(-1)).sum(axis=(0, 3))   # (Q, M, D)
    out = term if out is None else out + term
  return (out * inv.reshape(Q, 1, 1)).reshape(Q, E)


def layer_norm(x: Tensor, g: Tensor, b: Tensor, eps: float = 1e-5) -> Tensor:
  mu = x.mean(-1, keepdim=True)
  var = ((x - mu) * (x - mu)).mean(-1, keepdim=True)
  return (x - mu) * (var + eps).rsqrt() * g + b


def linear(x: Tensor, w: Tensor, b: Tensor) -> Tensor:
  return x @ w.transpose(0, 1) + b


class TinyEncoder:
  """The 3-layer encoder. `w` maps torch's Encoder.state_dict() names to float32 Tensors."""

  def __init__(self, w: dict[str, Tensor], ref_2d: np.ndarray, layers: int):
    self.w, self.layers = w, layers
    self.ref_2d = Tensor(ref_2d.astype(np.float32))

  @classmethod
  def from_torch(cls, enc) -> "TinyEncoder":
    w = {k: Tensor(v.detach().cpu().numpy().astype(np.float32)).realize() for k, v in enc.state_dict().items()}
    return cls(w, M.ref_2d().numpy(), len(enc.layers))

  def _lin(self, name: str, x: Tensor) -> Tensor:
    return linear(x, self.w[f"{name}.weight"], self.w[f"{name}.bias"])

  def _tsa(self, i: int, q: Tensor, pos: Tensor, v_prev: Tensor, v_cur: Tensor, ref_prev: Tensor) -> Tensor:
    p = f"layers.{i}.tsa"
    qcat = Tensor.cat(v_prev, q + pos, dim=-1)                       # (NQ, 2E)
    value = Tensor.stack(self._lin(f"{p}.value_proj", v_prev), self._lin(f"{p}.value_proj", v_cur), dim=0)
    off = self._lin(f"{p}.sampling_offsets", qcat).reshape(NQ, HEADS, 2, 4, 2)   # (Q, M, NO, P, 2)
    att = self._lin(f"{p}.attention_weights", qcat).reshape(NQ, HEADS, 2, 4).softmax(-1)
    ref = Tensor.stack(ref_prev, self.ref_2d, dim=0).reshape(2, NQ, 1, 2)
    out = msda(value, (BEV_H, BEV_W), ref, off, att, None)         # mean over the 2-frame queue
    return self._lin(f"{p}.output_proj", out) + q

  def _sca(self, i: int, q: Tensor, img_value: Tensor, ref_cam: Tensor, vis: Tensor) -> Tensor:
    p = f"layers.{i}.sca"
    value = self._lin(f"{p}.value_proj", img_value)                # (C, FH*FW, E)
    off = self._lin(f"{p}.sampling_offsets", q).reshape(NQ, HEADS, 1, 8, 2)
    att = self._lin(f"{p}.attention_weights", q).reshape(NQ, HEADS, 1, 8).softmax(-1)
    slots = msda(value, (FH, FW), ref_cam, off, att, vis)
    return self._lin(f"{p}.output_proj", slots) + q

  def _ffn_norm(self, i: int, x: Tensor, norm: int) -> Tensor:
    n = f"layers.{i}.norms.{norm}"
    return layer_norm(x, self.w[f"{n}.weight"], self.w[f"{n}.bias"])

  def __call__(self, feats: Tensor, prev_bev: Tensor, has_prev: Tensor, shift: Tensor, can_bus: Tensor,
               ref_cam: Tensor, bev_mask: Tensor) -> Tensor:
    w = self.w
    # bev_embedding + can_bus_mlp(can_bus), and the image value input (cams + level embeddings)
    cb = self._lin("can_bus_mlp.0", can_bus.reshape(1, 18)).relu()
    cb = self._lin("can_bus_mlp.2", cb).relu()
    cb = layer_norm(cb, w["can_bus_mlp.4.weight"], w["can_bus_mlp.4.bias"])
    q0 = w["bev_embedding"] + cb
    img_value = feats.reshape(NUM_CAMS, E, FH * FW).transpose(1, 2) + w["cams_embeds"].reshape(NUM_CAMS, 1, E) \
        + w["level_embeds"][0]
    col = w["col_embed"].reshape(1, BEV_W, E // 2).expand(BEV_H, BEV_W, E // 2)
    row = w["row_embed"].reshape(BEV_H, 1, E // 2).expand(BEV_H, BEV_W, E // 2)
    pos = Tensor.cat(col, row, dim=-1).reshape(NQ, E)
    ref_prev = self.ref_2d + shift * has_prev
    vis = (bev_mask.sum(-1) > 0).cast(dtypes.float32)               # (C, Q)
    q = q0
    for i in range(self.layers):
      v_prev = prev_bev * has_prev + q * (1 - has_prev)
      v_cur = q0 * has_prev + q * (1 - has_prev)
      q = self._ffn_norm(i, self._tsa(i, q, pos, v_prev, v_cur, ref_prev), 0)
      q = self._ffn_norm(i, self._sca(i, q, img_value, ref_cam, vis), 1)
      f = self._lin(f"layers.{i}.ffn.fc2", self._lin(f"layers.{i}.ffn.fc1", q).relu())
      q = self._ffn_norm(i, q + f, 2)
    return q


NUM_CAMS = M.NUM_CAMS


# ---- custom kernel (Tensor.custom_kernel: a kernel written in UOps, inside the tinygrad graph) ----------------------
from tinygrad.uop.ops import UOp, KernelInfo, AxisType  # noqa: E402

def _custom_msda_taps(out: UOp, value: UOp, shared: UOp, inv: UOp, *, NV: int, S: int, Q: int, P: int, W: int, H: int) -> UOp:
  """out[q, m, d] = inv[q] * sum over (v, p, tap) of ok * weight * value[v, pixel, m, d].

  value: flat (NV*S*E,). shared: (5, NV, Q, M, P) = floor x, floor y, fx, fy, attention weight (zeroed where the
  map does not see the query). Each output element is one thread's reduction over (v, p, tap); the coordinates are
  read from `shared`, not recomputed."""
  i = UOp.range(Q * HEADS * D, 0)
  q, m, d = i // (HEADS * D), (i // D) % HEADS, i % D
  k = UOp.range(NV * P * 4, 1, AxisType.REDUCE)
  v, p, t = k // (P * 4), (k // 4) % P, k % 4
  dx, dy = t % 2, t // 2
  x0, y0 = shared[0, v, q, m, p], shared[1, v, q, m, p]
  fx, fy, a = shared[2, v, q, m, p], shared[3, v, q, m, p], shared[4, v, q, m, p]
  xt, yt = x0 + dx.cast(dtypes.float), y0 + dy.cast(dtypes.float)
  ok = (xt >= 0.0) & (xt < float(W)) & (yt >= 0.0) & (yt < float(H))
  wx = dx.cast(dtypes.float) * fx + (1.0 - dx.cast(dtypes.float)) * (1.0 - fx)
  wy = dy.cast(dtypes.float) * fy + (1.0 - dy.cast(dtypes.float)) * (1.0 - fy)
  pix = v * S + yt.cast(dtypes.int32) * W + xt.cast(dtypes.int32)
  row = ok.where(pix * HEADS + m, 0)
  val = value[row * D + d]
  contrib = ok.where(val * (wx * wy * a), 0.0)
  red = contrib.reduce(k, arg=Ops.ADD)
  store = out[i].store((red * inv[q]).cast(out.dtype))
  return store.end(i).sink(arg=KernelInfo(f"msda_taps_{Q}_{NV}_{P}"))

def msda_custom(value: Tensor, hw: tuple[int, int], ref: Tensor, off: Tensor, attw: Tensor, vis: Tensor | None) -> Tensor:
  """msda() with the tap-and-reduce stage as a custom UOp kernel (Tensor.custom_kernel). Same contract."""
  H, W = hw
  NV, Q, P = ref.shape[0], off.shape[0], off.shape[3]
  rows = (value if value.ndim == 1 else value.reshape(-1)).realize()
  shared, inv = _sample_coords(ref, off, attw, vis, hw)
  out = Tensor.empty(Q * E, dtype=dtypes.float32)
  fxn = functools.partial(_custom_msda_taps, NV=NV, S=H * W, Q=Q, P=P, W=W, H=H)
  return Tensor.custom_kernel(out, rows, shared, inv, fxn=fxn)[0].reshape(Q, E)
