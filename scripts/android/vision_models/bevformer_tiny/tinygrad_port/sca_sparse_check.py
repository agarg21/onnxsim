#!/usr/bin/env python3
"""Check sca_sparse_tg.sca_sparse against the dense bevformer_tg.msda with a visibility mask (CPU or DSP).

usage: [DEV=CPU] python3 sca_sparse_check.py [--density 0.19] [--K 640]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from tinygrad import Tensor

import bevformer_tg as B
import sca_sparse_tg as S


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--density", type=float, default=0.19)
  ap.add_argument("--K", type=int, default=640)
  ap.add_argument("--seed", type=int, default=0)
  a = ap.parse_args()
  rng = np.random.default_rng(a.seed)
  NV, Q, P, Z = 6, B.NQ, 8, 4
  H, W = B.FH, B.FW
  value = rng.standard_normal((NV, H * W, B.E)).astype(np.float32)
  ref = (rng.random((NV, Q, Z, 2)) * 1.2 - 0.1).astype(np.float32)
  ref[:, :3] = 0.0                                                   # corners: zero-padding taps
  off = (rng.standard_normal((Q, B.HEADS, 1, P, 2)) * 2.0).astype(np.float32)
  attw = rng.random((Q, B.HEADS, 1, P)).astype(np.float32)
  attw /= attw.sum(-1, keepdims=True)
  vis = (rng.random((NV, Q)) < a.density).astype(np.float32)
  pidx, pmask, inv, count = S.compact(vis, a.K)

  want = B.msda(Tensor(value), (H, W), Tensor(ref), Tensor(off), Tensor(attw), Tensor(vis)).numpy()
  t0 = time.time()
  got = S.sca_sparse(Tensor(value), (H, W), Tensor(ref), Tensor(off), Tensor(attw), Tensor(pidx), Tensor(pmask),
                     Tensor(inv), Tensor(count)).numpy()
  dt = time.time() - t0
  d = float(np.abs(got - want).max())
  print(f"visible {int(vis.sum())}/{NV * Q} (K={a.K}): sparse vs dense max abs {d:.2e} ({dt:.2f}s incl. compile)")
  assert d < 1e-4, d


if __name__ == "__main__":
  main()
