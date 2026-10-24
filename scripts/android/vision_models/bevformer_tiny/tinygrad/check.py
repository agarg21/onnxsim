#!/usr/bin/env python3
"""Check the tinygrad encoder against the torch reference.

  unit:    msda() vs model.msda_rank5 (torch, grid_sample) on random TSA-shaped and SCA-shaped calls
  encoder: TinyEncoder vs model.Encoder (torch) on the saved frames (<work>/frames/*.pt)

usage: [DEV=CPU|DSP MOCKDSP=1] python3 check.py unit
       [DEV=CPU] python3 check.py encoder --ckpt <pth> --work <work> [--frames 0,1,2]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bevformer_tg as T  # noqa: E402
from tinygrad import Tensor  # noqa: E402
import model as M  # noqa: E402


def _cos(a: np.ndarray, b: np.ndarray) -> float:
  a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
  return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def unit(_a) -> None:
  rng = np.random.default_rng(0)
  cases = {
    # TSA: 2 queue frames, per-frame offsets (NO = NV = 2), 4 points, one reference per frame
    "tsa": dict(nv=2, hw=(M.BEV_H, M.BEV_W), no=2, p=4, r=1, vis=False),
    # SCA: 6 cameras, shared offsets (NO = 1), 8 points over 4 pillar anchors, a visibility mask
    "sca": dict(nv=6, hw=(M.FH, M.FW), no=1, p=8, r=4, vis=True),
  }
  for name, c in cases.items():
    H, W = c["hw"]
    nv, no, p, r, nq = c["nv"], c["no"], c["p"], c["r"], M.NQ
    value = rng.standard_normal((nv, H * W, M.EMBED)).astype(np.float32)
    ref = rng.random((nv, nq, r, 2)).astype(np.float32) * 1.2 - 0.1
    off = (rng.standard_normal((nq, M.HEADS, no, p, 2)) * 2.0).astype(np.float32)
    attw = rng.random((nq, M.HEADS, no, p)).astype(np.float32)
    attw /= attw.sum(-1, keepdims=True)
    vis = (rng.random((nv, nq)) > 0.3).astype(np.float32) if c["vis"] else None
    ref[:, :3] = 0.0                                          # corners: zero-padding taps
    got = T.msda(Tensor(value), (H, W), Tensor(ref), Tensor(off), Tensor(attw),
                 None if vis is None else Tensor(vis)).numpy()
    # torch reference, one value map at a time: loc = ref + off / (W, H) (normalized), point p uses anchor p % r,
    # and the NO = 1 offsets/weights are shared by every map
    cols = []
    for v in range(nv):
      o = v if no > 1 else 0
      refp = np.stack([ref[v, :, q % r] for q in range(p)], 1)                  # (nq, p, 2)
      loc = refp[:, None] + off[:, :, o] / np.array([W, H], np.float32)           # (nq, M, p, 2)
      lv = torch.tensor(loc).reshape(1, nq, M.HEADS, p, 2)
      ww = torch.tensor(attw[:, :, o]).reshape(1, nq, M.HEADS, p)
      vv = torch.tensor(value[v]).reshape(1, H * W, M.HEADS, M.EMBED // M.HEADS)
      out = M.msda_rank5(vv, (H, W), lv, ww).reshape(nq, M.EMBED).numpy()
      cols.append(out * (1.0 if vis is None else vis[v][:, None]))
    cnt = np.full((nq, 1), float(nv)) if vis is None else np.maximum(vis.sum(0)[:, None], 1.0)
    want = sum(cols) / cnt
    d = float(np.abs(got - want).max())
    print(f"unit {name}: max abs {d:.2e}  cos {_cos(got, want):.9f}")
    assert d < 1e-4, (name, d)


def encoder(a) -> None:
  _, enc, _ = M.load_official(a.ckpt)
  tiny = T.TinyEncoder.from_torch(enc)
  for k in a.frames:
    f = torch.load(Path(a.work) / "frames" / f"{k}.pt", weights_only=False)
    with torch.no_grad():
      ref = enc(*f["enc_in"]).numpy()
    t0 = time.time()
    got = tiny(*[Tensor(x.numpy()).realize() for x in f["enc_in"]]).numpy()
    dt = time.time() - t0
    print(f"frame {k}: tinygrad vs torch encoder max abs {np.abs(got - ref).max():.2e}  "
          f"cos {_cos(got, ref):.9f}  ({dt:.1f}s incl. compile)")


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("what", choices=["unit", "encoder"])
  ap.add_argument("--ckpt", default=str(Path.home() / ".cache/onnxsim-bevformer/bevformer_tiny_epoch_24.pth"))
  ap.add_argument("--work", default=str(Path.home() / ".cache/onnxsim-bevformer/work"))
  ap.add_argument("--frames", type=lambda s: [int(x) for x in s.split(",")], default=[0, 1, 2])
  a = ap.parse_args()
  unit(a) if a.what == "unit" else encoder(a)


if __name__ == "__main__":
  main()
