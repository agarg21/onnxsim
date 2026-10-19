"""Spatial cross-attention over only the visible (camera, query) pairs, as tinygrad Tensors.

The dense msda (bevformer_tg.msda) samples every (camera, query) pair, NV * Q of them, and multiplies the invisible ones
by zero. Here each camera gets K slots holding its visible queries (compact()), so the sampling runs over N = NV * K
entries, one per (camera, slot). The entries are reduced back to queries through the inverse table: a query's result is
the sum over the cameras that see it, divided by that count. Same numbers as msda with a visibility mask; the work is
N / (NV * Q) of the dense work (~0.2 at the real frame's visibility).
"""
from __future__ import annotations

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.helpers import getenv

import bevformer_tg as B


def compact(vis: np.ndarray, K: int):
  """vis (NV, Q) {0, 1} -> (pidx (NV, K) int32 query per slot, pad 0; pmask (NV, K) float {0, 1};
  inv (NV, Q) int32 entry per (camera, query), NV * K where the camera does not see the query (a zero row);
  count (Q,) float, cameras that see each query). Raises if a camera sees more than K queries."""
  NV, Q = vis.shape
  N = NV * K
  pidx = np.zeros((NV, K), np.int32)
  pmask = np.zeros((NV, K), np.float32)
  inv = np.full((NV, Q), N, np.int32)
  for v in range(NV):
    qs = np.nonzero(vis[v])[0]
    if len(qs) > K: raise ValueError(f"camera {v} sees {len(qs)} queries > K={K}; raise K")
    pidx[v, :len(qs)] = qs
    pmask[v, :len(qs)] = 1.0
    inv[v, qs] = v * K + np.arange(len(qs), dtype=np.int32)
  return pidx, pmask, inv, vis.sum(0).astype(np.float32)


def sample_entries(value: Tensor, hw: tuple[int, int], ref: Tensor, off: Tensor, attw: Tensor, mapid: Tensor) -> Tensor:
  """One value map per entry. value (NV, H*W, E); ref (N, R, 2) normalized; off (N, M, P, 2) raw pixels;
  attw (N, M, P) with the entry's weight already applied; mapid (N,) int32 the map each entry samples. -> (N, E).
  Same order of float operations as msda, so it matches it to rounding."""
  H, W = hw
  N, R, P = ref.shape[0], ref.shape[1], off.shape[2]
  S = H * W
  refp = Tensor.stack(*[ref[:, p % R] for p in range(P)], dim=1)                       # (N, P, 2)
  wh = Tensor([float(W), float(H)], dtype=dtypes.float32)
  px = (refp.reshape(N, 1, P, 2) + off / wh) * wh - 0.5                                   # (N, M, P, 2)
  x, y = px[..., 0], px[..., 1]
  x0f, y0f = x.floor(), y.floor()
  fx, fy = x - x0f, y - y0f
  # SP_FUSE=1: keep the coordinates lazy, so the sample kernel computes them in registers instead of reading a realized
  # (N, M, P) table back from memory (the profile: its misses are the streamed coordinate table)
  shared = Tensor.stack(x0f, y0f, fx, fy, attw, dim=0)
  if not getenv("SP_FUSE", 0): shared = shared.realize()
  x0f, y0f, fx, fy, att_s = shared[0], shared[1], shared[2], shared[3], shared[4]

  head = Tensor.arange(B.HEADS, dtype=dtypes.int32).reshape(1, B.HEADS, 1)
  base = (mapid.cast(dtypes.int32) * S).reshape(N, 1, 1)
  rows = value.reshape(-1).realize()
  out = None
  for dx, dy in B.TAPS:
    xt, yt = x0f + dx, y0f + dy
    ok = (xt >= 0) & (xt < W) & (yt >= 0) & (yt < H)
    w = ((fx if dx else 1.0 - fx) * (fy if dy else 1.0 - fy)) * att_s
    idx = ((base + yt.cast(dtypes.int32) * W + xt.cast(dtypes.int32)) * B.HEADS + head)  # (N, M, P)
    idx = ok.where(idx, 0)
    val = B.gather_rows(rows, idx, B.D)                                                  # (N, M, P, D)
    term = (val * ok.where(w, 0.0).unsqueeze(-1)).sum(axis=2)                           # (N, M, D)
    out = term if out is None else out + term
  return out.reshape(N, B.E)


def _flat(t: Tensor) -> Tensor:
  """A realized 1-D copy: gathering through reshape(-1) of a realized tensor keeps a wrapping view, and every element
  load then computes a % and a / of the flat index (the reduce kernel's 35 ms was mostly that)."""
  return t.reshape(-1).contiguous().realize()


def sca_sparse(value: Tensor, hw: tuple[int, int], ref_cam: Tensor, off: Tensor, attw: Tensor, pidx: Tensor,
               pmask: Tensor, inv: Tensor, count: Tensor) -> Tensor:
  """value (NV, S, E) projected camera maps; ref_cam (NV, Q, Z, 2); off (Q, M, 1, P, 2) raw pixel offsets shared by
  the cameras; attw (Q, M, 1, P) softmaxed; pidx (NV, K) int32 / pmask (NV, K) / inv (NV, Q) int32 / count (Q,) from
  compact(). -> (Q, E), equal to bevformer_tg.msda(value, hw, ref_cam, off, attw, vis)."""
  NV, Q, Z = ref_cam.shape[0], ref_cam.shape[1], ref_cam.shape[2]
  K = pidx.shape[1]
  N = NV * K
  M, P = B.HEADS, off.shape[3]
  mapid = Tensor.arange(NV, dtype=dtypes.int32).reshape(NV, 1).expand(NV, K).reshape(N)
  qe = pidx.cast(dtypes.int32).reshape(N)
  # the entry's query row in each table: references per (camera, query), offsets/weights per query (shared)
  row = mapid * Q + qe
  ref_e = B.gather_rows(_flat(ref_cam), row, Z * 2).reshape(N, Z, 2)
  off_e = B.gather_rows(_flat(off), qe, M * P * 2).reshape(N, M, P, 2)
  att_e = B.gather_rows(_flat(attw), qe, M * P).reshape(N, M, P) * pmask.reshape(N, 1, 1)
  # realized: a lazy INDEX gather feeding the reference stack fails the rangeify verifier (weakint RANGE)
  if not getenv("SP_LAZY_GATHER", 0):   # the realize keeps the INDEX gathers out of the sampler's stack (see above)
    ref_e, off_e, att_e = ref_e.realize(), off_e.realize(), att_e.realize()
  out_e = sample_entries(value, hw, ref_e, off_e, att_e, mapid)                          # (N, E)
  # reduce: each (camera, query) reads its slot's entry, or the zero row N where the camera does not see it
  zero = Tensor.zeros(1, B.E, dtype=dtypes.float32)
  out_pad = _flat(Tensor.cat(out_e, zero, dim=0))
  per_cam = B.gather_rows(out_pad, inv.cast(dtypes.int32).reshape(-1), B.E).reshape(NV, Q, B.E)
  return per_cam.sum(0) * (1.0 / count.maximum(1.0)).reshape(Q, 1)
