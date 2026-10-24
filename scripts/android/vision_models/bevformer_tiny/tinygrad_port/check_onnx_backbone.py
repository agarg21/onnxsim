"""The repo's exported backbone (export.py -> backbone6.sim.onnx) run by tinygrad's OnnxRunner, vs the torch feats."""
import sys, time, numpy as np, torch
from pathlib import Path
from tinygrad import Tensor
from tinygrad.nn.onnx import OnnxRunner
work = Path(sys.argv[1])
run = OnnxRunner(str(work / sys.argv[2]))
print('inputs', {k: v for k, v in run.graph_inputs.items()}, 'outputs', run.graph_outputs)
fr = torch.load(work / 'frames' / '0.pt', weights_only=False)
img = fr['img'].numpy().astype(np.float32)
name = next(iter(run.graph_inputs))
t0 = time.time(); out = run({name: Tensor(img)}); got = next(iter(out.values())).numpy() if isinstance(out, dict) else out.numpy()
print('onnx(tinygrad) feats', got.shape, 'time', round(time.time()-t0, 1))
print('max abs vs torch feats', np.abs(got.reshape(6, 256, 15, 25) - fr['enc_in'][0].numpy()).max())
