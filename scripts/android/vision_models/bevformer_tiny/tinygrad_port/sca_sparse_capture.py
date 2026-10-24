#!/usr/bin/env python3
"""Capture the sparse SCA (sca_sparse_tg) as a phone kernel set, with the same tag interface as capture_msda.py.

The compacted slot arrays (pidx, pmask, inv, count) are computed on the phone in prep from the visibility mask, so the
generated run.c keeps the dense SCA's signature and the dispatch in phone/tg_dispatch.c needs no change.

usage: [DEV=DSP MOCKDSP=1] python3 sca_sparse_capture.py <case dir> --out <gen dir> [--K 640]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import capture_msda as C
import sca_sparse_tg as S
from tinygrad import Tensor


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("case", type=Path)
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--K", type=int, default=640)
  a = ap.parse_args()
  c = C.read_case(a.case)
  pidx, pmask, inv, count = S.compact(c["vis"], a.K)
  c["compact"] = (pidx, pmask, inv, count)

  def graph(c: dict) -> Tensor:
    return S.sca_sparse(Tensor(c["value"].reshape(-1)), (c["H"], c["W"]), Tensor(c["ref"]), Tensor(c["off"]),
                        Tensor(c["attw"]), Tensor(pidx), Tensor(pmask), Tensor(inv), Tensor(count))

  got, t = C.trace(c, graph)
  d = float(np.abs(got - c["ref_out"]).max())
  print(f"{a.case.name}: sparse SCA (DSP mock) vs torch reference max abs {d:.2e}; "
        f"{len(t['launches'])} launches of {len(set(t['src_of_lib']))} distinct kernels")
  C.generate(c, t, a.case.name, a.out)


if __name__ == "__main__":
  main()
