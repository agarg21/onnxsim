import sys, time, numpy as np, torch
sys.path.insert(0, '..')
import model as M
from tinygrad import Tensor
from backbone_tg import TinyBackbone
_, enc, _ = M.load_official(sys.argv[1])
bb = M.Backbone().eval()
bb.load_state_dict(M.load_official(sys.argv[1])[0].state_dict())
tiny = TinyBackbone.from_torch(bb)
x = np.random.default_rng(0).standard_normal((1, 3, 480, 800)).astype(np.float32)
with torch.no_grad(): ref = bb(torch.from_numpy(x)).numpy()
t0 = time.time(); got = tiny(Tensor(x)).numpy(); dt = time.time() - t0
print('backbone max abs', np.abs(got - ref).max(), 'max |ref|', np.abs(ref).max(), 'time', round(dt, 2))
