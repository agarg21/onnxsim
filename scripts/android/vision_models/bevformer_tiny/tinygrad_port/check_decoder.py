import sys, time, numpy as np, torch
sys.path.insert(0, '..')
import model as M
from tinygrad import Tensor
from decoder_tg import TinyDecoder
_, enc, dec = M.load_official(sys.argv[1])
frame = torch.load(sys.argv[2], weights_only=False)
with torch.no_grad():
  bev = enc(*frame['enc_in'])
  cls_r, bb_r = dec(bev)
tiny = TinyDecoder.from_torch(dec)
t0 = time.time()
cls, bb = tiny(Tensor(bev.numpy()).realize())
cls, bb = cls.numpy(), bb.numpy()
print('decoder cls max abs', np.abs(cls - cls_r.numpy()).max(), 'bbox max abs', np.abs(bb - bb_r.numpy()).max(), 'time', round(time.time()-t0, 2))
