"""One ONNX Conv (3x3, 64->64, 30x50, stride 1, pad 1, + bias + relu) through tinygrad's OnnxRunner on the DSP backend."""
import time, numpy as np, onnx, torch
from onnx import helper, TensorProto, numpy_helper
from tinygrad import Tensor
from tinygrad.nn.onnx import OnnxRunner
import os
rng = np.random.default_rng(0)
C, H, W = 64, 30, 50
w = rng.standard_normal((C, C, 3, 3)).astype(np.float32) * 0.05
b = rng.standard_normal((C,)).astype(np.float32) * 0.1
x = rng.standard_normal((1, C, H, W)).astype(np.float32)
g = helper.make_graph(
  [helper.make_node('Conv', ['x', 'w', 'b'], ['c'], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]),
   helper.make_node('Relu', ['c'], ['y'])],
  'conv', [helper.make_tensor_value_info('x', TensorProto.FLOAT, x.shape)],
  [helper.make_tensor_value_info('y', TensorProto.FLOAT, None)],
  [numpy_helper.from_array(w, 'w'), numpy_helper.from_array(b, 'b')])
m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 17)])
onnx.save(m, '/mnt/data/cache/claude-work/bevtg/conv3x3.onnx')
ref = torch.relu(torch.nn.functional.conv2d(torch.from_numpy(x), torch.from_numpy(w), torch.from_numpy(b), padding=1)).numpy()
run = OnnxRunner('/mnt/data/cache/claude-work/bevtg/conv3x3.onnx')
t0 = time.time(); got = list(run({'x': Tensor(x)}).values())[0].numpy() if isinstance(run({'x': Tensor(x)}), dict) else run({'x': Tensor(x)}).numpy()
print('DSP conv3x3 max abs', np.abs(got - ref).max(), 'max |ref|', np.abs(ref).max(), 'time', round(time.time() - t0, 1))
