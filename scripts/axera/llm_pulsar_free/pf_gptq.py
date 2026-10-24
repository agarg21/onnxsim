"""GPTQ rounding of every linear of a model on the card's own s8 grid
(``linear_emit.gptq_weight``: per output channel, ``s = max|w[:, c]| / 127.5``;
each channel's largest element keeps its float value, so ``emit_linear_u16``
recomputes the same scale and emits GPTQ's codes), with the input statistics
of ``pf_record.py``'s activations (both calibration sets). Pulsar-free.

Writes ``<out>/gptq/L<l>.npz`` (qkv5, o_proj, gu, down_proj; merged layouts as
``Config.QKV`` / ``Config.GU``) and ``post.npz`` (lm_head); ``GPTQ=1`` makes
``pf_generate.py`` / ``pf_estimate.py`` / ``pf_plan.py`` use them. Split
first-set linears keep the original weights (the split carries the precision).
Run it on CUDA (``--device cuda``, the default when torch sees a GPU): the
lm_head alone is a 1024 x 151936 GPTQ for Qwen3-0.6B. Without a GPU it runs
in onnxsim's C++ column loop on 4 threads.

usage: MODEL=qwen3_0.6b pf_gptq.py [--damp 0.01] [--device cuda]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AXERA = os.path.dirname(HERE)
for _p in (HERE, AXERA):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import linear_emit as le  # noqa: E402
import pf_model  # noqa: E402

F32 = np.float32
#: linear -> its input in the recorded activations
INPUTS = {"qkv5": "a", "o_proj": "att", "gu": "m", "down_proj": "act"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument(
        "--device", default=None, help="torch device (default: cuda if available)"
    )
    a = ap.parse_args()
    device = a.device
    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else None
    cfg = pf_model.load()
    os.makedirs(os.path.join(cfg.out, "gptq"), exist_ok=True)
    t0 = time.time()
    for wh in [f"L{li}" for li in range(cfg.L)] + ["post"]:
        acts = [
            np.load(os.path.join(cfg.out, "acts", st, f"{wh}.npz"))
            for st in ("rest", "first")
        ]

        def X(key, acts=acts):
            return np.concatenate(
                [np.asarray(z[key], F32).reshape(len(z[key]), -1) for z in acts]
            )

        if wh == "post":
            res = {"lm_head": le.gptq_weight(cfg.head_t, X("xn"), a.damp, device)}
        else:
            li = int(wh[1:])
            res = {
                key: le.gptq_weight(
                    cfg.merged_weight(key, li, gptq=False), X(xk), a.damp, device
                )
                for key, xk in INPUTS.items()
            }
        np.savez(os.path.join(cfg.out, "gptq", f"{wh}.npz"), **res)
        print(wh, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
