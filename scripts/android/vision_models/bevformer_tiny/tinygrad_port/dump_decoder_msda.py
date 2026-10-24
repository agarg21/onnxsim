#!/usr/bin/env python3
"""Dump the decoder's deformable cross-attention calls (900 queries over the 50x50 BEV map) of real frame data as msda_io
case directories (the phone harness's format), with the torch reference output. Layers come from the torch Decoder run
on the frame's BEV.   usage: dump_decoder_msda.py --ckpt <pth> --work <work> --out <dir> [--frame 0] [--layers 0,5]"""
import argparse, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import model as M  # noqa: E402

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt", required=True); ap.add_argument("--work", required=True); ap.add_argument("--out", required=True)
  ap.add_argument("--frame", type=int, default=0); ap.add_argument("--layers", type=lambda s: [int(x) for x in s.split(",")], default=[0, 5])
  a = ap.parse_args()
  torch.set_grad_enabled(False)
  _, enc, dec = M.load_official(a.ckpt)
  fr = torch.load(Path(a.work) / "frames" / f"{a.frame}.pt", weights_only=False)
  bev = enc(*fr["enc_in"])
  caps = {}
  for i in a.layers:
    def pre(mod, args, i=i): caps[i] = args
    dec.layers[i].cross_attn.register_forward_pre_hook(pre)
  dec(bev)
  out = Path(a.out)
  for i in a.layers:
    query, qpos, value, ref = caps[i][0], caps[i][1], caps[i][2], caps[i][3]   # value = bev (NQ, E); ref (Q, 2)
    m = dec.layers[i].cross_attn
    with torch.no_grad():
      q = query + qpos
      v = m.value_proj(value).reshape(1, M.NQ, M.HEADS, M.EMBED // M.HEADS)
      off = m.sampling_offsets(q).reshape(1, 900, M.HEADS, 4, 2)
      w = torch.softmax(m.attention_weights(q).reshape(1, 900, M.HEADS, 4), -1)
      loc = ref.reshape(1, 900, 1, 1, 2) + off / torch.tensor([M.BEV_W, M.BEV_H], dtype=off.dtype)
      ref_out = M.msda_rank5(v, (M.BEV_H, M.BEV_W), loc, w)[0].numpy()        # (900, 256), pre output_proj
    d = out / f"dec_l{i}"; d.mkdir(parents=True, exist_ok=True)
    (d / "meta.txt").write_text(f"1 1 {M.NQ} {M.HEADS} {M.EMBED // M.HEADS} 4 900 1 1 1 1 1 2 0 0\n{M.BEV_H} {M.BEV_W} 0\n")
    v[0].numpy().astype(np.float32).tofile(d / "value.f32")
    off.reshape(900, M.HEADS, 1, 4, 2).numpy().astype(np.float32).tofile(d / "loc.f32")
    w.reshape(900, M.HEADS, 1, 4).numpy().astype(np.float32).tofile(d / "attw.f32")
    ref.reshape(1, 900, 1, 2).numpy().astype(np.float32).tofile(d / "ref.f32")
    ref_out.astype(np.float32).tofile(d / "ref_out.f32")
    print(d, "ref_out", ref_out.shape, float(np.abs(ref_out).max()))

if __name__ == "__main__":
  main()
