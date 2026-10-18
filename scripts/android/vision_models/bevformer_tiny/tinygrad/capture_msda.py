#!/usr/bin/env python3
"""Trace tinygrad's msda() on one msda_hvx case directory (msda_io/l0_tsa, l0_sca: a real frame's call, with the
torch reference output the hand kernel is checked against) and generate the Hexagon C for it.

The DSP backend (DEV=DSP, MOCKDSP=1 by default) is traced, not re-rendered: every program launch records the
buffers it binds, every host upload records its bytes, and every program's C source is taken from the compiler.
From that the script writes, into --out:
  kernels.c   the distinct kernels of the call, boilerplate removed, each renamed tg_<tag>_k<i> (one body per source)
  run.c       tg_<tag>_run(value, ref, off, attw, vis_u8, out): the launch sequence as straight-line C. Inputs are
              the caller's buffers (value/ref/off/attw are used in place; vis is converted u8 -> float into a
              scratch buffer), the output is written straight into `out`, other buffers are scratch (uploaded
              constants are copied from their recorded bytes).
  trace.json  the launch list, for the record

usage: capture_msda.py <case_dir> --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

os.environ.setdefault("DEV", "DSP")
os.environ.setdefault("MOCKDSP", "1")

import numpy as np  # noqa: E402
from tinygrad import Tensor  # noqa: E402
from tinygrad.runtime import ops_dsp as dsp  # noqa: E402

import bevformer_tg as T  # noqa: E402

BOILERPLATE = "/* DSP boilerplate */"


def read_case(d: Path) -> dict:
  meta = (d / "meta.txt").read_text().split("\n")
  nums = [int(x) for x in meta[0].split()]
  NV, L, S, M, D, P, Q, NO, mode, NVR, RL, R, RD, has_vis, vdtype = nums
  H, W, start = (int(x) for x in meta[1].split())
  assert (L, M, D, mode, RL, vdtype, RD) == (1, T.HEADS, T.D, 1, 1, 0, 2), nums
  f32 = lambda n, shape: np.fromfile(d / n, np.float32).reshape(shape)
  c = dict(NV=NV, S=S, Q=Q, P=P, NO=NO, R=R, H=H, W=W, has_vis=has_vis)
  c["value"] = f32("value.f32", (NV, S, M * D))
  c["ref"] = f32("ref.f32", (NVR, Q, R, RD))
  c["off"] = f32("loc.f32", (Q, M, NO, P, 2))   # raw offsets (mode 1), (Q, M, NO, L=1, P, 2)
  c["attw"] = f32("attw.f32", (Q, M, NO, P))
  c["vis"] = np.fromfile(d / "vis.u8", np.uint8).reshape(NV, Q) if has_vis else None
  c["ref_out"] = f32("ref_out.f32", (Q, M * D))
  return c


def trace(c: dict) -> tuple[np.ndarray, dict]:
  """Run msda() on DEV=DSP (MOCKDSP) with launch/compile hooks. Returns (output, trace).

  The mock program writes back every buffer it binds, so the outputs of earlier launches are the buffers
  that are written; the first launch that reads a buffer nothing produced yet is where it was uploaded, and
  its bytes are recorded there (the inputs and the uploaded constants). Kernels write their output as data0."""
  src_of_lib: dict[bytes, str] = {}
  launches: list[dict] = []
  uploads: dict[int, bytes] = {}
  produced: set[int] = set()
  orig_compile = dsp.DSPCompiler.compile
  orig_call = dsp.MockDSPProgram.__call__

  def _compile(self, src):
    lib = orig_compile(self, src)
    src_of_lib[lib] = src
    return lib

  def _call(self, *bufs, **kw):
    for x in bufs[1:]:
      if x.va_addr not in produced and x.va_addr not in uploads:
        uploads[x.va_addr] = bytes(dsp.to_mv(x.va_addr, x.size))
    launches.append({"lib": self.lib, "bufs": [x.va_addr for x in bufs], "sizes": [x.size for x in bufs]})
    try:
      r = orig_call(self, *bufs, **kw)
    except subprocess.CalledProcessError:
      # TRACE_ONLY: qemu can't run this arch's code (v69 qfloat), so the values are not meaningful. The launches, buffers
      # and sources are still recorded, which is all the phone build needs; the phone check is the correctness test then.
      if not os.environ.get("TRACE_ONLY"): raise
      r = 0.0
    produced.add(bufs[0].va_addr)
    return r

  dsp.DSPCompiler.compile, dsp.MockDSPProgram.__call__ = _compile, _call
  try:
    vis = None if c["vis"] is None else Tensor(c["vis"].astype(np.float32))
    out = ({"custom": T.msda_custom}.get(os.environ.get("MSDA_IMPL", ""), T.msda))(Tensor(c["value"].reshape(-1)), (c["H"], c["W"]), Tensor(c["ref"]), Tensor(c["off"]), Tensor(c["attw"]), vis)
    out = out.realize()
    out_va = out.uop.buffer._buf.va_addr
    got = out.numpy()
  finally:
    dsp.DSPCompiler.compile, dsp.MockDSPProgram.__call__ = orig_compile, orig_call
  sizes = {x: s for L in launches for x, s in zip(L["bufs"], L["sizes"])}
  return got, dict(launches=launches, uploads=uploads, sizes=sizes, src_of_lib=src_of_lib, out_va=out_va)


def _strip(src: str) -> tuple[list[str], str, str, str]:
  """-> (typedef lines, helper text before the kernel function (qfloat helpers, deduped by the caller), the E_* function
  text, its name)."""
  head, _, _ = src.partition(BOILERPLATE)
  types = [ln for ln in head.splitlines() if ln.startswith("typedef")]
  lines = [ln for ln in head.splitlines() if not ln.startswith("typedef")]
  start = next((i for i, ln in enumerate(lines) if "noinline" in ln or re.match(r"\s*void \w+\(", ln)), None)
  assert start is not None, src[:200]
  helpers = "\n".join(lines[:start]).strip()
  body = "\n".join(lines[start:]).strip()
  m = re.search(r"void (\w+)\(", body)
  return types, helpers, body, m.group(1)


def generate(c: dict, t: dict, tag: str, out: Path) -> None:
  out.mkdir(parents=True, exist_ok=True)
  # one C function per distinct kernel source
  names: dict[bytes, str] = {}
  types: list[str] = []
  bodies: list[str] = []
  helper_blocks: list[str] = []
  for lib, src in t["src_of_lib"].items():
    if lib in names:
      continue
    ty, helpers, body, fn = _strip(src)
    types += [x for x in ty if x not in types]
    for ln in helpers.splitlines():  # the helper preludes repeat the same one-line definitions: keep each once
      if ln.strip() and ln not in helper_blocks: helper_blocks.append(ln)
    names[lib] = f"tg_{tag}_k{len(names)}"
    bodies.append(re.sub(rf"\b{fn}\(", f"{names[lib]}(", body, count=1))

  # buffer roles: the inputs by content (uploads), the output by identity, the rest scratch or uploaded constants
  inputs = {"value": c["value"].tobytes(), "ref": c["ref"].tobytes(), "off": c["off"].tobytes(),
            "attw": c["attw"].tobytes()}
  if c["vis"] is not None:
    inputs["vis"] = c["vis"].astype(np.float32).tobytes()
  role: dict[int, str] = {t["out_va"]: "out"}
  for va, data in t["uploads"].items():
    for k, v in inputs.items():
      if data == v and va not in role:
        role[va] = k
  for k in inputs:
    assert k in role.values(), (k, "input not found among the uploads")

  scratch: dict[int, str] = {}
  size_of: dict[int, int] = {}
  for L in t["launches"]:
    for va, sz in zip(L["bufs"], L["sizes"]):
      size_of.setdefault(va, sz)
      if va not in role and va not in scratch:
        scratch[va] = f"s{len(scratch)}"
  pname = {"value": "value", "ref": "ref", "off": "off", "attw": "attw", "vis": "vis_f", "out": "out"}

  decl, body = [], []
  if "vis" in role.values():
    decl.append("static float vis_f[NV_Q] __attribute__((aligned(128)));")
    body.append("  for (int i = 0; i < NV_Q; i++) vis_f[i] = vis_u8[i] ? 1.0f : 0.0f;")
  for va, name in scratch.items():
    sz = size_of[va]
    decl.append(f"static unsigned char {name}[{sz}] __attribute__((aligned(128)));")
    if va in t["uploads"]:
      arr = ",".join(str(b) for b in t["uploads"][va])
      decl.append(f"static const unsigned char {name}_init[{sz}] = {{{arr}}};")
      body.append(f"  memcpy({name}, {name}_init, {sz});")
  for i, L in enumerate(t["launches"]):
    args = [f"(void*){pname[role[va]] if va in role else scratch[va]}" for va in L["bufs"]]
    body.append(f"  TG_PROF_BEGIN({i});")
    body.append(f"  {names[L['lib']]}({', '.join(args)});")

  # __TG_RECIP differs per kernel (a float or a float32 association); one definition serves all of them. float32 and
  # __hvx_f are the same type, so the vector case is one branch
  helper_blocks = [ln for ln in helper_blocks if not ln.startswith("#define __TG_RECIP")]
  if not any("__tg_recip_f32(" in ln for ln in helper_blocks):
    helper_blocks.append("static inline float32 __tg_recip_f32(float32 x) { return (float32)__tg_recip_v(x); }")
  helper_blocks.append("#define __TG_RECIP(...) _Generic((__VA_ARGS__), float: __tg_recip_s, default: __tg_recip_f32)(__VA_ARGS__)")
  (out / "kernels.c").write_text(
      f"/* generated by capture_msda.py from tinygrad's DSP renderer ({tag}) */\n"
      + "\n".join(types) + "\n\n" + "\n".join(helper_blocks) + "\n\n" + "\n\n".join(bodies) + "\n")
  prof = (
      "/* TG_PROF builds (profiling): launch i runs only while i < tg_stop (tg_stop < 0: all launches), so timing\n"
      " * prefixes of the sequence gives each launch's cost as a difference */\n"
      "#ifdef TG_PROF\n"
      "extern int tg_stop;\n"
      "#define TG_PROF_BEGIN(i) if (tg_stop >= 0 && i >= tg_stop) return;\n"
      "#else\n#define TG_PROF_BEGIN(i)\n#endif\n\n")
  run = ("/* generated by capture_msda.py: tinygrad's msda launch sequence for " + tag + f" (NV={c['NV']}, Q={c['Q']}) */\n"
         "#include <string.h>\n#include \"kernels.c\"\n\n" + prof +
         f"#define NV_Q {c['NV'] * c['Q']}\n\n" + "\n".join(decl) + "\n\n"
         f"void tg_{tag}_run(const float* value, const float* ref, const float* off, const float* attw,\n"
         "                 const unsigned char* vis_u8, float* out) {\n"
         "  (void)vis_u8;\n" + "\n".join(body) + "\n}\n")
  (out / "run.c").write_text(run)
  (out / "trace.json").write_text(json.dumps({
      "tag": tag, "launches": [{"kernel": names[L["lib"]],
                                "bufs": [role.get(va, scratch.get(va)) for va in L["bufs"]]} for L in t["launches"]],
      "distinct_kernels": len(names)}, indent=1))


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("case", type=Path)
  ap.add_argument("--out", type=Path, required=True)
  a = ap.parse_args()
  c = read_case(a.case)
  got, t = trace(c)
  d = np.abs(got - c["ref_out"]).max()
  print(f"{a.case.name}: tinygrad (DSP mock) vs torch reference max abs {d:.2e}; "
        f"{len(t['launches'])} launches of {len(set(t['src_of_lib']))} distinct kernels")
  if not os.environ.get("TRACE_ONLY"): assert d < 1e-4, d
  generate(c, t, a.case.name, a.out)


if __name__ == "__main__":
  main()
