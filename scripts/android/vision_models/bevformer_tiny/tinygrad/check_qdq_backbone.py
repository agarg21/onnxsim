"""The repo's int8 backbone (backbone1.q8.onnx: uint8 NHWC image in, uint8 feats out) for camera 0 of frame 0.
ORT CPU is the reference; tinygrad's OnnxRunner runs the same graph on the current device (DEV) and is compared with it.
Also reports the dequantized feats against the fp32 torch features (the accuracy the phone path has to keep)."""
import sys, time, numpy as np, torch, onnxruntime as ort
from pathlib import Path
from tinygrad import Tensor
from tinygrad.nn.onnx import OnnxRunner
W = Path.home() / '.cache/onnxsim-bevformer/work'
fr = torch.load(W / 'frames/0.pt', weights_only=False)
img = fr['img'][:1].numpy()                                  # (1, 3, 480, 800) normalized, camera 0
q_in = np.clip(np.round(img.transpose(0, 2, 3, 1) / 0.01865844801068306) + 114, 0, 255).astype(np.uint8)  # NHWC
sess = ort.InferenceSession(str(W / 'backbone1.q8.onnx'), providers=['CPUExecutionProvider'])
t0 = time.time(); ref_q = sess.run(None, {'img': q_in})[0]; print('ORT', round(time.time() - t0, 2), 's')
run = OnnxRunner(str(W / 'backbone1.q8.onnx'))
t0 = time.time(); got_q = run({'img': Tensor(q_in)}); got_q = (got_q if isinstance(got_q, Tensor) else list(got_q.values())[0]).numpy()
print('tinygrad', round(time.time() - t0, 2), 's', 'uint8 off-by vs ORT: max', int(np.abs(got_q.astype(int) - ref_q.astype(int)).max()), 'count', int((got_q != ref_q).sum()), '/', ref_q.size)
S, Z = 0.16442053020000458, 107
f_ref = fr['enc_in'][0].numpy()[0:1]
print('ORT feats vs torch fp32 (dequant): max abs', np.abs((ref_q.astype(np.float32) - Z) * S - f_ref).max(), 'max |feat|', np.abs(f_ref).max())
