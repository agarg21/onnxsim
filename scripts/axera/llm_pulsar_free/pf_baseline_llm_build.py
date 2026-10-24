"""The reference point: Pulsar2's own ``llm_build`` output (s8 weights, bf16
activations, one program per layer + post) run on the card through
``llm_layer_loop`` on the same held-out passages and steps as ``pf_plan.py``,
teacher-forced, against the same float reference (``Config.float_step``).

usage: MODEL=llama160m pf_baseline_llm_build.py --build-dir DIR [--prefix P]
       [--docs 0,1] [--prompt 48]   -> <out>/baseline_llm_build.json
(DIR: llm_build's output directory with <prefix>_p128_l<i>_together.axmodel and
<prefix>_post.axmodel)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AXERA = os.path.dirname(HERE)
for _p in (HERE, AXERA):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import llm_layer_loop as lll  # noqa: E402
import pf_layout as lay  # noqa: E402
import pf_model  # noqa: E402
from axcl_session import AXSession, health_check  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build-dir", required=True)
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--docs", default="0,1")
    ap.add_argument("--prompt", type=int, default=48)
    a = ap.parse_args()
    cfg = pf_model.load()
    res = {}
    with AXSession() as s:
        res["health_before"] = health_check(s)

        def load(path):
            m = s.load(open(path, "rb").read())
            for sp in m.outputs:  # bf16 comes back as raw 16-bit patterns
                sp.dtype = np.uint16
            return m

        def run(m, arrays):
            feeds = []
            for x, sp in zip(arrays, m.inputs):
                x = np.ascontiguousarray(x)
                if sp.dtype == np.uint8:
                    feeds.append(x.view(np.uint8).ravel())
                else:
                    feeds.append(x.astype(sp.dtype).ravel())
            return [np.asarray(o) for o in s.run(m, feeds)]

        backend = lll.DeviceBackend(
            load, run, a.build_dir, prefix=a.prefix, unload=s.unload
        )
        host = lll.Host.from_model(cfg.model, backend.spec)
        for d in map(int, a.docs.split(",")):
            ids = cfg.passage_ids(lay.EVAL_DOCS[d])[: a.prompt]
            host.reset()
            fK, fV = cfg.new_cache()
            rows = []
            for p, tok in enumerate(ids):
                t0 = time.time()
                lg_bf16, _ = host.step(tok, backend)
                wall = time.time() - t0
                flg, _ = cfg.float_step(fK, fV, tok, p)
                if p == 0 and cfg.zero_pos0:
                    continue  # the same steps as pf_plan.py
                lg = lll.bf16_to_f32(lg_bf16)
                rows.append(
                    dict(
                        p=p,
                        top1=int(lg.argmax()),
                        float_top1=int(np.argmax(flg)),
                        wall_s=wall,
                        rel=lay.rel_err(lg, flg),
                    )
                )
            r = dict(
                lay.summarize(rows),
                s_per_token=float(np.mean([r["wall_s"] for r in rows])),
            )
            res[lay.EVAL_DOCS[d]] = dict(r, steps=rows)
            print(lay.EVAL_DOCS[d], r, flush=True)
        backend.close()
        res["health_after"] = health_check(s)
    print("health", res["health_before"], res["health_after"])
    json.dump(
        res, open(os.path.join(cfg.out, "baseline_llm_build.json"), "w"), indent=1
    )


if __name__ == "__main__":
    main()
