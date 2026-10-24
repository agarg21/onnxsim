#!/usr/bin/env python3
"""The repo's int8 backbone (backbone1.q8.onnx, one camera) as a tinygrad DSP graph on the HMX (QDQ_HMX lowering in
nn/onnx_qdq.py): capture under DEV=DSP MOCKDSP=1 (sources only; qemu can't run HMX) -> emit -> build_skel -> case dir.
ORT's output is the reference (uint8 feats).

usage: [HEXAGON_SDK_ROOT, HEXAGON_TOOLCHAIN, HMX_VTCM_KB=4096] python3 phone_qdq_backbone.py <outdir>
"""
import os, sys, pathlib
os.environ.setdefault("DEV", "DSP"); os.environ.setdefault("MOCKDSP", "1"); os.environ.setdefault("QDQ_HMX", "1")
os.environ.setdefault("HMX_VTCM_KB", "4096")
os.environ.setdefault("HMX", "1")   # HMX tile tensor cores; without it the matmuls fall back to scalar vrmpy (~359 s)
import numpy as np, onnxruntime as ort, torch
from tinygrad import Tensor
from tinygrad.nn.onnx import OnnxRunner
from tinygrad.runtime.support import dsp_graph as G

W = pathlib.Path.home() / ".cache/onnxsim-bevformer/work"
out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
fr = torch.load(W / "frames/0.pt", weights_only=False)
img = fr["img"][:1].numpy()
q_in = np.clip(np.round(img.transpose(0, 2, 3, 1) / 0.01865844801068306) + 114, 0, 255).astype(np.uint8)
ref = ort.InferenceSession(str(W / "backbone1.q8.onnx"), providers=["CPUExecutionProvider"]).run(None, {"img": q_in})[0]

run = OnnxRunner(str(W / "backbone1.q8.onnx"))
# constants first: the lowering reads scales with .numpy(), and inside capture no kernel runs (zeros otherwise)
for v in run.graph_values.values():
  if isinstance(v, Tensor): v.realize()
net = run._qdq_grid()   # the QDQ lowering is built on the first call; build it here, outside the capture (it reads scales)
assert net is not None, "QDQ lowering not used for this graph"
x = Tensor(q_in).realize()
y_holder = {}
def fn():
  r = run({"img": x})
  y = r if isinstance(r, Tensor) else list(r.values())[0]
  y_holder["y"] = y.realize()
calls, bufs = G.capture(fn)
y = y_holder["y"]
info = G.emit(out, calls, bufs, x.uop.buffer, y.uop.buffer)
print("emitted", len(calls), "calls", {k: v for k, v in info.items() if k != "kernels"}, "kernels", len(info["kernels"]))
case = out / "case"; case.mkdir(exist_ok=True)
(case / "a.bin").write_bytes(q_in.tobytes())
(case / "ref.bin").write_bytes(ref.tobytes())
(case / "b.bin").write_bytes((out / "blob.bin").read_bytes())
so, client = G.build_skel(out)
print("skel", so, "client", client)
