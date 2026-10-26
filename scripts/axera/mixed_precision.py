#!/usr/bin/env python3
"""Pick which ops to run at 16 bits on the AX650N, from two builds and no card.

Whole-model U16 often fixes a model that int8 breaks, but costs latency on
every op. Usually a handful of ops carry most of the int8 error. This module
finds them from two ordinary `pulsar2 build` outputs of the same ONNX model
-- one at U8, one at U16 -- and turns the choice into Pulsar2
`quant.layer_configs`.

How it works:

* `replay.py` already reproduces the card from a build's own quantisation
  table: QuantizeLinear/DequantizeLinear pairs at the scales Pulsar2 wrote
  into `quant/quant_axmodel.json`, run in onnxruntime. Splicing the two
  tables -- U8 everywhere, the U16 build's scales for the tensors a mix
  widens -- emulates a mixed build without building it.
* `Build` reads the compiler's quantised ops (post-fusion names from
  `quant_axmodel.json`, which `layer_configs.layer_names` accepts) and which
  tensors a set of promoted ops actually widens (`widened()`): `layer_configs`
  promotes an op's *inputs*, and a tensor goes to 16 bits only when every op
  reading it is promoted -- otherwise Pulsar2 keeps it 8-bit and requantises.
  This rule is what made the splice agree with real mixed builds (DistilBERT:
  splice 0.93 / 0.91 / 0.39 logit error, real build replayed 0.93 / 0.94 /
  0.40, card 0.94 / 0.91 / 0.41).
* `search()` (default, coarse groups such as op types) adds, at each step,
  the group whose addition scores best, until within `slack_db` of every op
  promoted. It finds groups that only pay together -- on BERT-style encoders
  the residual stream (`LayerNorm` + residual `Add`).
* `rank()` + `greedy()` do the same for fine groups (one per tensor, made of
  the ops that read it): rank once by gain alone, then add in that order.
* `layer_configs()` writes the result as Pulsar2 config.

Pick on a dev set that is not the evaluation set, then build the mix and run
it on the card to confirm.

    python mixed_precision.py model.onnx u8_build/out u16_build/out \\
        --feeds dev.npz --json mix.json   # then: encoder_classifier.py --mix
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import onnx

Feeds = Dict[str, np.ndarray]

#: data-movement ops: they carry codes and are never worth promoting alone.
_MOVEMENT = ("AxReshape", "AxTranspose", "AxSlice", "AxTile", "AxPad", "AxConcat")
_LAYER = re.compile(r"(?:^|/)((?:layer|layers|block|blocks|h)[._]\d+)(?:/|$)")


@dataclass
class Op:
    """One quantised op in a Pulsar2 build."""

    name: str  # post-fusion name, as quant_axmodel.json / layer_names use it
    op_type: str  # AxQuantized* type from quant_axmodel.onnx ("" if absent)
    layer: str  # enclosing block ("layer.3"), "" outside any block
    tensors: List[str] = field(default_factory=list)  # inputs U16 widens


def _doc(build_dir: str) -> dict:
    with open(os.path.join(build_dir, "quant", "quant_axmodel.json")) as fh:
        return json.load(fh)


class Build:
    """A `pulsar2 build` output read for mixed-precision planning.

    What `layer_configs` promotes is an op's **inputs**, and only where the
    op is their sole reader. Confirmed on the card: a U16 `Add` makes its
    producer write 16 bits, but an operand that another, 8-bit op also reads
    stays 8-bit and is requantised into 16 (no precision gained), and the
    `Add`'s own output stays 8-bit unless its consumer is promoted too. So a
    tensor is 16-bit in a mix exactly when every quantised op reading it
    (looking through data-movement ops) is promoted -- `widened()`.

    `ops` are the quantised ops in graph order (post-fusion names, which
    `layer_configs.layer_names` accepts), each with the tensors it reads and
    the encoder layer it sits in -- from its own name, or else the nearest
    preceding named op (fused ops get generated names such as `Gemm_15`).
    """

    def __init__(self, build_dir: str):
        tc = _doc(build_dir)["tensor_configs"]
        graph = onnx.load(
            os.path.join(build_dir, "quant", "quant_axmodel.onnx"),
            load_external_data=False,
        ).graph
        producer = {o: n for n in graph.node for o in n.output}
        weights = {i.name for i in graph.initializer}

        def root(t: str) -> str:  # through Reshape/Transpose/... to the real tensor
            seen = set()
            while t in producer and producer[t].op_type.startswith(_MOVEMENT):
                if t in seen:
                    break
                seen.add(t)
                t = producer[t].input[0]
            return t

        self.aliases: Dict[str, set] = {}
        for n in graph.node:
            for o in n.output:
                self.aliases.setdefault(root(o), set()).add(o)
        self.consumers: Dict[str, set] = {}
        self.ops: List[Op] = []
        layer = ""
        for n in graph.node:
            m = _LAYER.search(n.name)
            if m:
                layer = m.group(1)
            elif "/" in n.name:  # a named op outside the blocks (head, embeddings)
                layer = ""
            if n.name not in tc or n.op_type.startswith(_MOVEMENT):
                continue
            reads = [
                root(i)
                for i in n.input
                if i
                and i not in weights
                and tc[n.name].get(root(i), tc[n.name].get(i, {})).get("bit_width")
                in (8, 16)
            ]
            if not reads:
                continue
            for r in reads:
                self.consumers.setdefault(r, set()).add(n.name)
            self.ops.append(Op(n.name, n.op_type, layer, reads))

    def widened(self, selected: Sequence[Op]) -> set:
        """Tensor names (every alias) that are 16-bit when `selected` are."""
        names = {o.name for o in selected}
        out = set()
        for o in selected:
            for t in o.tensors:
                if self.consumers.get(t, set()) <= names:
                    out |= {t} | self.aliases.get(t, set())
        return out


def group(build: "Build", by: str = "tensor") -> Dict[str, List[Op]]:
    """`{group: [ops]}`. `by="tensor"` (default): one group per activation,
    holding every op that reads it -- the unit that actually widens one
    tensor. Or "op", "op_type", "layer"."""
    if by == "tensor":
        named = {o.name: o for o in build.ops}
        return {
            t: [named[n] for n in sorted(readers)]
            for t, readers in build.consumers.items()
        }
    g: Dict[str, List[Op]] = {}
    for o in build.ops:
        key = {"op": o.name, "op_type": o.op_type or "?", "layer": o.layer or "-"}[by]
        g.setdefault(key, []).append(o)
    return g


def spliced_scales(lo: dict, hi: dict, tensors) -> dict:
    """`lo`'s table with `tensors` taken from `hi` (the U16 build)."""
    out = dict(lo)
    for t in tensors:
        if t in hi:
            out[t] = hi[t]
    return out


def snr_db(ref: Sequence[np.ndarray], got: Sequence[np.ndarray]) -> float:
    r = np.concatenate([np.ravel(x) for x in ref]).astype(np.float64)
    g = np.concatenate([np.ravel(x) for x in got]).astype(np.float64)
    return float(10 * np.log10((r**2).sum() / max(((r - g) ** 2).sum(), 1e-30)))


class Replayer:
    """Runs `model` under any scale table on fixed feeds; scores by SNR of
    the first output against the float model."""

    def __init__(
        self,
        model: onnx.ModelProto,
        feeds: Sequence[Feeds],
        metric: Optional[Callable] = None,
    ):
        import onnxruntime as ort

        self.ort, self.model, self.feeds = ort, model, list(feeds)
        self.metric = metric or snr_db
        self.ref = self._run(model)

    def _run(self, model):
        o = self.ort.SessionOptions()
        o.graph_optimization_level = self.ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        s = self.ort.InferenceSession(
            model.SerializeToString(), o, providers=["CPUExecutionProvider"]
        )
        return [s.run(None, f)[0] for f in self.feeds]

    def outputs(self, scales: dict):
        from replay import insert_qdq

        q, _, _ = insert_qdq(self.model, scales)
        return self._run(q)

    def score(self, scales: dict) -> float:
        return self.metric(self.ref, self.outputs(scales))


@dataclass
class Ranked:
    group: str
    ops: List[Op]
    score: float  # metric with only this group promoted
    gain: float  # score - all-lo score


def rank(
    rp: Replayer, build: Build, lo: dict, hi: dict, groups: Dict[str, List[Op]]
) -> List[Ranked]:
    """Each group promoted on its own, best gain first."""
    base = rp.score(lo)
    rows = []
    for g, members in groups.items():
        s = rp.score(spliced_scales(lo, hi, build.widened(members)))
        rows.append(Ranked(g, members, s, s - base))
    return sorted(rows, key=lambda r: -r.gain)


@dataclass
class Selection:
    groups: List[str]
    ops: List[Op]
    score: float
    trace: List[tuple]  # (group, score after adding it)


def greedy(
    rp: Replayer,
    build: Build,
    lo: dict,
    hi: dict,
    ranked: Sequence[Ranked],
    target: float,
    max_groups: Optional[int] = None,
    min_gain: float = 0.0,
) -> Selection:
    """Promote groups in `ranked` order while the metric is below `target`;
    a group whose addition does not raise the score by more than `min_gain`
    is skipped."""
    chosen: List[Ranked] = []
    score = rp.score(lo)
    trace = [("<U8>", score)]
    for r in ranked:
        if score >= target or (max_groups is not None and len(chosen) >= max_groups):
            break
        trial = [o for c in chosen + [r] for o in c.ops]
        s = rp.score(spliced_scales(lo, hi, build.widened(trial)))
        if s - score > min_gain:
            chosen.append(r)
            score = s
            trace.append((r.group, s))
    ops_ = list({o.name: o for c in chosen for o in c.ops}.values())
    return Selection([c.group for c in chosen], ops_, score, trace)


def search(
    rp: Replayer,
    build: Build,
    lo: dict,
    hi: dict,
    groups: Dict[str, List[Op]],
    target: Optional[float] = None,
    slack_db: float = 1.0,
    max_steps: Optional[int] = None,
) -> Selection:
    """Forward selection with re-ranking: at each step try adding every
    remaining group and keep the one that scores best (ties: fewer widened
    tensors), until the score reaches `target` -- by default `slack_db`
    below every op promoted.

    Unlike `greedy` over one-at-a-time ranks, this finds groups that only
    pay together. On BERT-style encoders the residual stream is one: each
    residual `Add` / `LayerNorm` tensor widened alone is worth little, all of
    them together recover the whole-model U16 accuracy (see
    docs/axera-encoder-classifiers.md). Costs one replay per remaining group
    per step, so use it with coarse groups (`by="op_type"` or `"layer"`).
    """
    if target is None:
        target = rp.score(spliced_scales(lo, hi, build.widened(build.ops))) - slack_db
    chosen: List[str] = []
    score = rp.score(lo)
    trace = [("<U8>", score)]

    def ops_of(names):
        return list({o.name: o for g in names for o in groups[g]}.values())

    while score < target and (max_steps is None or len(chosen) < max_steps):
        best = None
        for g in groups:
            if g in chosen:
                continue
            trial = ops_of(chosen + [g])
            s = rp.score(spliced_scales(lo, hi, build.widened(trial)))
            key = (s, -len(build.widened(trial)))
            if best is None or key > best[0]:
                best = (key, g, s)
        if best is None:
            break
        chosen.append(best[1])
        score = best[2]
        trace.append((best[1], score))
    return Selection(chosen, ops_of(chosen), score, trace)


def layer_configs(selected: Sequence[Op], data_type: str = "U16") -> List[dict]:
    """Pulsar2 `quant.layer_configs` promoting exactly `selected` (by their
    post-fusion names, the only name form `layer_names` reliably accepts --
    pre-fusion ONNX names fail loudly, fused op types are silently ignored;
    see scripts/axera/README.md "Mixed precision on the AX650")."""
    names = sorted({o.name for o in selected})
    return [{"layer_names": names, "data_type": data_type}] if names else []


def main(argv=None) -> int:
    from replay import load_scales

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("model", help="the float ONNX both builds were made from")
    p.add_argument("lo_build", help="U8 build output dir")
    p.add_argument("hi_build", help="U16 build output dir")
    p.add_argument(
        "--feeds", required=True, help=".npz of stacked inputs, axis 0 = sample"
    )
    p.add_argument(
        "--by", default="op_type", choices=["op_type", "layer", "tensor", "op"]
    )
    p.add_argument(
        "--slack-db",
        type=float,
        default=1.0,
        help="stop within this many dB of every op promoted",
    )
    p.add_argument("--max-steps", type=int)
    p.add_argument("--json", help="write the selection and layer_configs here")
    a = p.parse_args(argv)

    z = np.load(a.feeds)
    feeds = [{k: z[k][i : i + 1] for k in z.files} for i in range(len(z[z.files[0]]))]
    rp = Replayer(onnx.load(a.model), feeds)
    lo, hi = load_scales(a.lo_build), load_scales(a.hi_build)
    b = Build(a.lo_build)
    groups = group(b, a.by)
    if a.by in ("tensor", "op"):  # fine groups: rank once, then add in order
        ranked = rank(rp, b, lo, hi, groups)
        target = rp.score(spliced_scales(lo, hi, b.widened(b.ops))) - a.slack_db
        sel = greedy(rp, b, lo, hi, ranked, target, a.max_steps)
    else:
        sel = search(rp, b, lo, hi, groups, slack_db=a.slack_db, max_steps=a.max_steps)
    for g, s in sel.trace:
        print(f"{s:7.2f} dB  + {g}")
    print(
        f"{len(sel.ops)} ops promoted, {len(b.widened(sel.ops))} tensors widened "
        f"(U8 {rp.score(lo):.2f} dB, all-U16 {rp.score(hi):.2f} dB)"
    )
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(
                {
                    "selection": sel.groups,
                    "score_db": sel.score,
                    "trace": sel.trace,
                    "layer_configs": layer_configs(sel.ops),
                },
                fh,
                indent=1,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
