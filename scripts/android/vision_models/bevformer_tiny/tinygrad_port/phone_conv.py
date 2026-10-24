#!/usr/bin/env python3
"""One float 3x3 conv (64 -> 64, 30 x 50, + bias + relu) as a tinygrad DSP graph on the phone (runtime/support/dsp_graph:
capture -> emit -> build_skel -> client). Checks the phone's output against torch and times the inferences.

usage: [HEXAGON_SDK_ROOT=.. HEXAGON_TOOLCHAIN=..] python3 phone_conv.py <outdir>      (then the run line it prints)"""
import os, sys, pathlib
os.environ.setdefault("DEV", "DSP"); os.environ.setdefault("MOCKDSP", "1")
import numpy as np, torch
from tinygrad import Tensor
from tinygrad.runtime.support import dsp_graph as G

out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
C, H, W = 64, 30, 50
w = Tensor((rng.standard_normal((C, C, 3, 3)) * 0.05).astype(np.float32)).realize()
b = Tensor((rng.standard_normal((C,)) * 0.1).astype(np.float32)).realize()
x_np = rng.standard_normal((1, C, H, W)).astype(np.float32)
x = Tensor(x_np).realize()
y = x.conv2d(w, b, padding=1).relu()
calls, bufs = G.capture(lambda: y.realize())
info = G.emit(out, calls, bufs, x.uop.buffer, y.uop.buffer)
print("emitted", len(calls), "calls;", info if isinstance(info, dict) and len(str(info)) < 400 else "")
ref = torch.relu(torch.nn.functional.conv2d(torch.from_numpy(x_np), torch.from_numpy(w.numpy()), torch.from_numpy(b.numpy()), padding=1)).numpy()
case = out / "case"; case.mkdir(exist_ok=True)
(case / "a.bin").write_bytes(x_np.tobytes())
(case / "ref.bin").write_bytes(ref.astype(np.float32).tobytes())
(case / "b.bin").write_bytes((out / "blob.bin").read_bytes())
so, client = G.build_skel(out)
print("skel:", so, "client:", client)
print("run on the phone: client 'file:///tg_hmx_rpc.so?tg_hmx_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp' <case> 10 prof")
