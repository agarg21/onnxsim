"""BEVFormer-tiny's decoder + head (model.Decoder) in tinygrad: 6 layers of self-attention, deformable cross-attention
over the BEV map (bevformer_tg.msda: one value map, 2-d reference points), FFN, and the per-layer box refinement;
the last layer's cls/bbox heads. Weights are model.Decoder.state_dict()."""
from __future__ import annotations

import math

import numpy as np
from tinygrad import Tensor

import bevformer_tg as B

E, HEADS = B.E, B.HEADS
DH = E // HEADS
PC_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]


def linear(x: Tensor, w: Tensor, b: Tensor) -> Tensor:
  return x @ w.transpose(0, 1) + b


def inv_sig(x: Tensor, eps: float = 1e-5) -> Tensor:
  x = x.clip(0.0, 1.0)
  return (x.clip(eps, None) / (1 - x).clip(eps, None)).log()


class TinyDecoder:
  def __init__(self, w: dict[str, Tensor]):
    self.w = w

  @classmethod
  def from_torch(cls, dec) -> "TinyDecoder":
    return cls({k: Tensor(v.detach().cpu().numpy().astype(np.float32)).realize() for k, v in dec.state_dict().items()})

  def _lin(self, name: str, x: Tensor) -> Tensor:
    return linear(x, self.w[f"{name}.weight"], self.w[f"{name}.bias"])

  def _ln(self, name: str, x: Tensor) -> Tensor:
    return B.layer_norm(x, self.w[f"{name}.weight"], self.w[f"{name}.bias"])

  def _mha(self, i: int, qk: Tensor, v_in: Tensor) -> Tensor:
    p = f"layers.{i}.self_attn"
    W, b = self.w[f"{p}.in_proj_weight"], self.w[f"{p}.in_proj_bias"]
    q = linear(qk, W[:E], b[:E]).reshape(-1, HEADS, DH).transpose(0, 1)
    k = linear(qk, W[E:2 * E], b[E:2 * E]).reshape(-1, HEADS, DH).transpose(0, 1)
    v = linear(v_in, W[2 * E:], b[2 * E:]).reshape(-1, HEADS, DH).transpose(0, 1)
    att = ((q @ k.transpose(1, 2)) / math.sqrt(DH)).softmax(-1)
    out = (att @ v).transpose(0, 1).reshape(-1, E)
    return self._lin(f"{p}.out_proj", out)

  def _cross(self, i: int, q: Tensor, qpos: Tensor, bev: Tensor, ref: Tensor) -> Tensor:
    p = f"layers.{i}.cross_attn"
    value = self._lin(f"{p}.value_proj", bev)                       # (NQ, E) BEV value map
    qq = q + qpos
    off = self._lin(f"{p}.sampling_offsets", qq).reshape(-1, HEADS, 1, 4, 2)
    att = self._lin(f"{p}.attention_weights", qq).reshape(-1, HEADS, 1, 4).softmax(-1)
    out = B.msda(value.reshape(1, B.NQ, E), (B.BEV_H, B.BEV_W), ref.reshape(1, -1, 1, 2), off, att, None)
    return self._lin(f"{p}.output_proj", out) + q

  def _layer(self, i: int, q: Tensor, qpos: Tensor, bev: Tensor, ref: Tensor) -> Tensor:
    qk = q + qpos
    sa = self._mha(i, qk, q)
    q = self._ln(f"layers.{i}.norms.0", q + sa)
    q = self._ln(f"layers.{i}.norms.1", self._cross(i, q, qpos, bev, ref))
    f = self._lin(f"layers.{i}.ffn.fc2", self._lin(f"layers.{i}.ffn.fc1", q).relu())
    return self._ln(f"layers.{i}.norms.2", q + f)

  def _reg(self, i: int, x: Tensor) -> Tensor:
    p = f"reg_branches.{i}"
    x = self._lin(f"{p}.0", x).relu()
    x = self._lin(f"{p}.2", x).relu()
    return self._lin(f"{p}.4", x)

  def _cls(self, x: Tensor) -> Tensor:
    p = "cls_last"
    x = self._lin(f"{p}.0", x)
    x = self._ln(f"{p}.1", x).relu()
    x = self._lin(f"{p}.3", x)
    x = self._ln(f"{p}.4", x).relu()
    return self._lin(f"{p}.6", x)

  def __call__(self, bev: Tensor) -> tuple[Tensor, Tensor]:
    w = self.w
    qpos, q = w["query_embedding"][:, :E], w["query_embedding"][:, E:]
    refp = self._lin("reference_points", qpos).sigmoid()           # (Q, 3)
    prev_ref = refp
    tmp = None
    for i in range(6):
      q = self._layer(i, q, qpos, bev, refp[:, :2])
      tmp = self._reg(i, q)
      prev_ref = refp
      xy = (tmp[:, 0:2] + inv_sig(refp[:, :2])).sigmoid()
      z = (tmp[:, 4:5] + inv_sig(refp[:, 2:3])).sigmoid()
      refp = xy.cat(z, dim=-1)
    cls = self._cls(q)
    r = inv_sig(prev_ref)
    xy = (tmp[:, 0:2] + r[:, 0:2]).sigmoid()
    z = (tmp[:, 4:5] + r[:, 2:3]).sigmoid()
    x = xy[:, 0:1] * (PC_RANGE[3] - PC_RANGE[0]) + PC_RANGE[0]
    y = xy[:, 1:2] * (PC_RANGE[4] - PC_RANGE[1]) + PC_RANGE[1]
    zz = z * (PC_RANGE[5] - PC_RANGE[2]) + PC_RANGE[2]
    bbox = x.cat(y, tmp[:, 2:4], zz, tmp[:, 5:], dim=-1)
    return cls, bbox
