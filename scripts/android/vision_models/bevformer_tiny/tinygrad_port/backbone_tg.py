"""BEVFormer-tiny's backbone (ResNet-50 + 1-level FPN, model.Backbone) in tinygrad.

BatchNorm is frozen in inference, so each conv+BN pair folds to one conv with a bias at load time. The
layout is torchvision's ResNet v1.5 (the stride of a stage's first block sits on its 3x3 conv).
  TinyBackbone.from_torch(model.Backbone()) ; bb(img) with img (N, 3, 480, 800) normalized -> (N, 256, 15, 25)
"""
from __future__ import annotations

import numpy as np
from tinygrad import Tensor

BLOCKS = [3, 4, 6, 3]


def fold(conv_w: np.ndarray, bn: dict, eps: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
  s = bn["weight"] / np.sqrt(bn["running_var"] + eps)
  return conv_w * s[:, None, None, None], bn["bias"] - bn["running_mean"] * s


class TinyBackbone:
  def __init__(self, layers: list[list[dict]], stem: dict, lateral: dict, fpn: dict):
    self.stem, self.layers, self.lateral, self.fpn = stem, layers, lateral, fpn

  @staticmethod
  def _conv(sd: dict, name: str) -> dict:
    return {"w": sd[f"{name}.weight"].numpy().astype(np.float32)}

  @classmethod
  def from_torch(cls, bb) -> "TinyBackbone":
    sd = {k: v.detach() for k, v in bb.state_dict().items()}
    t = lambda k: sd[k].numpy().astype(np.float32)
    def bn_of(prefix: str) -> dict:
      return {k: t(f"{prefix}.{k}") for k in ("weight", "bias", "running_mean", "running_var")}
    def cbn(conv: str, bn: str, stride: int, pad: int) -> dict:
      w, b = fold(t(f"{conv}.weight"), bn_of(bn))
      return {"w": Tensor(w).realize(), "b": Tensor(b).realize(), "s": stride, "p": pad}
    stem = cbn("body.0", "body.1", 2, 3)
    layers = []
    for li, n in enumerate(BLOCKS):
      blocks = []
      for bi in range(n):
        pre = f"body.{4 + li}.{bi}"
        stride = 2 if (bi == 0 and li > 0) else 1
        blk = {"c1": cbn(f"{pre}.conv1", f"{pre}.bn1", 1, 0),
               "c2": cbn(f"{pre}.conv2", f"{pre}.bn2", stride, 1),
               "c3": cbn(f"{pre}.conv3", f"{pre}.bn3", 1, 0)}
        if bi == 0:
          blk["down"] = cbn(f"{pre}.downsample.0", f"{pre}.downsample.1", stride, 0)
        blocks.append(blk)
      layers.append(blocks)
    lat = {"w": Tensor(t("lateral.weight")).realize(), "b": Tensor(t("lateral.bias")).realize(), "s": 1, "p": 0}
    fpn = {"w": Tensor(t("fpn.weight")).realize(), "b": Tensor(t("fpn.bias")).realize(), "s": 1, "p": 1}
    return cls(layers, stem, lat, fpn)

  @staticmethod
  def _cb(x: Tensor, c: dict, relu: bool) -> Tensor:
    y = x.conv2d(c["w"], c["b"], stride=c["s"], padding=c["p"])
    return y.relu() if relu else y

  def __call__(self, img: Tensor) -> Tensor:
    x = self._cb(img, self.stem, True)
    x = x.max_pool2d(kernel_size=3, stride=2, padding=1)
    for blocks in self.layers:
      for blk in blocks:
        idt = self._cb(x, blk["down"], False) if "down" in blk else x
        y = self._cb(x, blk["c1"], True)
        y = self._cb(y, blk["c2"], True)
        y = self._cb(y, blk["c3"], False)
        x = (y + idt).relu()
    return self._cb(self._cb(x, self.lateral, False), self.fpn, False)
