#!/usr/bin/env python3
"""BEVFormer-tiny end to end in tinygrad: images -> backbone -> encoder -> decoder -> host decode, carried across frames
(prev_bev rotated and shifted on the host, as validate.py does). Per frame: each stage vs the torch reference
saved by validate.py (<work>/frames/<i>.pt), GT matches, and wall time per stage (tinygrad, this host's backend).

usage: [DEV=CPU] python3 e2e_tg.py --ckpt <pth> --work <work> [--frames 0,1,2]
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
sys.path.insert(0, str(HERE.parent))
import model as M  # noqa: E402
from nuscenes import match  # noqa: E402
from tinygrad import Tensor  # noqa: E402
from backbone_tg import TinyBackbone  # noqa: E402
from bevformer_tg import TinyEncoder  # noqa: E402
from decoder_tg import TinyDecoder  # noqa: E402


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt", required=True)
  ap.add_argument("--work", required=True)
  ap.add_argument("--frames", type=lambda s: [int(x) for x in s.split(",")], default=[0, 1, 2])
  a = ap.parse_args()
  torch.set_grad_enabled(False)
  bb_t, enc_t, dec_t = M.load_official(a.ckpt)
  t0 = time.time()
  bb, enc, dec = TinyBackbone.from_torch(bb_t), TinyEncoder.from_torch(enc_t), TinyDecoder.from_torch(dec_t)
  print(f"weights to tinygrad: {time.time() - t0:.1f}s")
  prev = None
  for k in a.frames:
    ref = torch.load(Path(a.work) / "frames" / f"{k}.pt", weights_only=False)
    feats_in, prev_ref_in, has_prev, shift, can_bus, ref_cam, bev_mask = ref["enc_in"]
    times = {}
    t = time.time()
    feats = bb(Tensor(ref["img"].numpy())).realize()
    times["backbone"] = time.time() - t
    if prev is None:
      prev_in = Tensor(np.zeros((M.NQ, M.EMBED), np.float32))
    else:
      prev_in = Tensor(M.rotate_prev_bev(torch.from_numpy(prev.numpy()), can_bus).numpy())
    t = time.time()
    bev = enc(feats.reshape(6, M.EMBED, M.FH, M.FW), prev_in.realize(), Tensor(has_prev.numpy()),
              Tensor(shift.numpy()), Tensor(can_bus.numpy()), Tensor(ref_cam.numpy()), Tensor(bev_mask.numpy())).realize()
    times["encoder"] = time.time() - t
    t = time.time()
    cls, bbox = dec(bev)
    cls, bbox = cls.realize().numpy(), bbox.realize().numpy()
    times["decoder"] = time.time() - t
    cls_t, bbox_t = torch.from_numpy(cls), torch.from_numpy(bbox)
    boxes, scores, labels = M.decode(cls_t, bbox_t)
    tp, npred, ngt = match(boxes, scores, labels, ref["gt"])
    tp_ref, _, _ = match(*M.decode(ref["cls"], ref["bbox"]), ref["gt"])
    e_feats = np.abs(feats.numpy() - feats_in.numpy()).max()
    e_bev = np.abs(bev.numpy() - ref["bev"].numpy()).max()
    print(f"frame {k}: feats max abs {e_feats:.2e} | bev max abs {e_bev:.2e} | cls {np.abs(cls - ref['cls'].numpy()).max():.2e}"
          f" bbox {np.abs(bbox - ref['bbox'].numpy()).max():.2e} | GT matched {tp} (torch {tp_ref}) of {ngt}, dets {npred}")
    print("  ms: " + "  ".join(f"{n} {v * 1e3:.0f}" for n, v in times.items()) + f"  total {sum(times.values()) * 1e3:.0f}")
    prev = bev


if __name__ == "__main__":
  main()
