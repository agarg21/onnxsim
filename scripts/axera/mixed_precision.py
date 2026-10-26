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
_LAYER = re.compile(
    r"/([A-Za-z_]+[._]\d+)(?:/|$)"
)  # ".../layer.3/...", ".../features.12/..."


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

    def _session(self, model):
        o = self.ort.SessionOptions()
        o.graph_optimization_level = self.ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        return self.ort.InferenceSession(
            model.SerializeToString(), o, providers=["CPUExecutionProvider"]
        )

    def _run(self, model):
        s = self._session(model)
        return [s.run(None, f)[0] for f in self.feeds]

    def _run_one(self, model, feed):
        return self._session(model).run(None, feed)[0]

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


# ---------------------------------------------------------------------------
# End to end: build, search, build the mix, check it on the card, fall back
# ---------------------------------------------------------------------------

#: Pulsar2 image the automatic flow builds with.
IMAGE = "pulsar2:7.0-lite"


def _digest(*parts) -> str:
    import hashlib

    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else json.dumps(p, sort_keys=True).encode())
    return h.hexdigest()[:16]


def pulsar2_config(
    input_names: Sequence[str],
    n_calib: int,
    layer_configs_: Optional[List[dict]] = None,
    method: str = "MinMax",
) -> dict:
    """A `pulsar2 build` config for raw (Numpy) inputs: one calibration tar
    per input at `./dataset/<name>.tar`, optional `layer_configs`."""
    quant = {
        "input_configs": [
            {
                "tensor_name": n,
                "calibration_dataset": f"./dataset/{n}.tar",
                "calibration_format": "Numpy",
                "calibration_size": n_calib,
            }
            for n in input_names
        ],
        "calibration_method": method,
        "precision_analysis": False,
    }
    if layer_configs_:
        quant["layer_configs"] = layer_configs_
    return {
        "model_type": "ONNX",
        "npu_mode": "NPU1",
        "quant": quant,
        "compiler": {"check": 0},
    }


def u16_layer_configs(model: onnx.ModelProto) -> List[dict]:
    """Whole-model U16: every op type in the float graph."""
    return [
        {"op_types": sorted({n.op_type for n in model.graph.node}), "data_type": "U16"}
    ]


@dataclass
class BuildOut:
    build_dir: Optional[str]  # <root>/out, or None when the build failed
    seconds: float
    cached: bool
    error: str = ""

    @property
    def axmodel(self) -> Optional[str]:
        return (
            os.path.join(self.build_dir, "compiled.axmodel") if self.build_dir else None
        )


def pulsar2_build(
    root: str,
    onnx_path: str,
    calib: Sequence[Feeds],
    layer_configs_: Optional[List[dict]] = None,
    method: str = "MinMax",
    image: str = IMAGE,
    timeout: int = 3600,
) -> BuildOut:
    """`pulsar2 build` of `onnx_path` into `<root>/out`, cached: a rerun with
    the same model bytes, calibration data and config reuses the build (the
    key is written to `<root>/build_key`)."""
    import time

    import pulsar2_docker as pd

    names = list(calib[0])
    cfg = pulsar2_config(names, len(calib), layer_configs_, method)
    with open(onnx_path, "rb") as fh:
        model_bytes = fh.read()
    key = _digest(
        model_bytes,
        cfg,
        image,
        *[np.ascontiguousarray(c[n]).tobytes() for c in calib for n in names],
    )
    out = os.path.join(root, "out")
    key_file = os.path.join(root, "build_key")
    if (
        os.path.exists(os.path.join(out, "compiled.axmodel"))
        and os.path.exists(key_file)
        and open(key_file).read() == key
    ):
        return BuildOut(out, 0.0, True)
    os.makedirs(os.path.join(root, "dataset"), exist_ok=True)
    os.makedirs(os.path.join(root, "config"), exist_ok=True)
    with open(os.path.join(root, "model.onnx"), "wb") as fh:
        fh.write(model_bytes)
    for n in names:
        pd.make_numpy_calibration_tar(
            os.path.join(root, "dataset", f"{n}.tar"), [c[n] for c in calib]
        )
    with open(os.path.join(root, "config", "c.json"), "w") as fh:
        json.dump(cfg, fh, indent=1)
    t0 = time.time()
    r = pd.build(
        root,
        "model.onnx",
        "out",
        config_path="config/c.json",
        image=image,
        timeout=timeout,
    )
    secs = time.time() - t0
    if not r.success:
        return BuildOut(None, secs, False, (r.error or r.stdout_tail)[-1500:])
    with open(key_file, "w") as fh:
        fh.write(key)
    return BuildOut(out, secs, False)


def count_16bit(build_dir: str) -> int:
    """Activation tensors a build quantised at 16 bits."""
    tc = _doc(build_dir)["tensor_configs"]
    return len(
        {t for cfg in tc.values() for t, c in cfg.items() if c.get("bit_width") == 16}
    )


def card_run(session, axmodel: str, feeds: Sequence[Feeds], runs: int = 20):
    """Run `axmodel` on every feed (inputs cast to what the compiled model
    takes) and time `runs` extra calls. Returns (first outputs, NPU ms/call)."""
    m = session.load(axmodel)
    try:
        order = [i.name for i in m.inputs]
        outs = [session.run(m, [f[n] for n in order])[0] for f in feeds]
        us0, n0 = session.exec_us, session.runs
        for _ in range(runs):
            session.run(m, [feeds[0][n] for n in order])
        ms = (session.exec_us - us0) / max(session.runs - n0, 1) / 1000
    finally:
        session.unload(m)
    return outs, ms


def _default_session():
    from axcl_session import AXSession, health_check

    class _Checked:
        """One AXSession per card visit, health-checked on entry and exit, so
        the device lock is not held across builds."""

        def __enter__(self):
            self.s = AXSession().__enter__()
            health_check(self.s)
            return self.s

        def __exit__(self, *exc):
            try:
                if exc[0] is None:
                    health_check(self.s)
            finally:
                self.s.__exit__(*exc)

    return _Checked()


def _err(snr: float) -> float:
    """Relative error (||got-ref|| / ||ref||) of an SNR in dB."""
    return float(10 ** (-snr / 20))


@dataclass
class Candidate:
    label: str
    groups: List[str]
    n_ops: int
    predicted_db: Optional[float]  # replay of the spliced tables
    card_db: Optional[float] = None
    latency_ms: Optional[float] = None
    tensors_16bit: Optional[int] = None
    build_s: float = 0.0
    cached: bool = False
    accepted: bool = False
    note: str = ""
    axmodel: Optional[str] = None
    layer_configs: List[dict] = field(default_factory=list)


def auto(
    model_path: str,
    calib: Sequence[Feeds],
    dev: Sequence[Feeds],
    work: str,
    slack_db: float = 1.0,
    max_err: Optional[float] = None,
    max_latency_ms: Optional[float] = None,
    by: str = "auto",
    max_candidates: int = 4,
    margin_db: float = 3.0,
    image: str = IMAGE,
    session_factory: Optional[Callable] = None,
    builder: Optional[Callable] = None,
    runs: int = 20,
    log: Callable = print,
) -> dict:
    """Mixed precision end to end, for any ONNX model with raw inputs.

    1. Build U8 and whole-model U16 (cached in `work`), run both on the card
       on `dev`: SNR of the first output against onnxruntime fp32, latency.
    2. The accuracy budget: `max_err` (relative error, as an SNR floor) if
       given, else `slack_db` below the card's U16 SNR. U8 meeting both
       budgets ends here.
    3. `search()` over `by` groups on the dev set orders the groups (`by=
       "auto"`: op types and layers both searched, the one reaching the
       budget with fewer widened tensors kept); every
       prefix of that order is a candidate, with a replay-predicted SNR. The
       first candidate whose prediction comes within `margin_db` of the
       budget (moved into replay terms by the U16 build's card-vs-replay
       offset) is built, run on the card, and accepted
       if the card meets both budgets; otherwise the next prefix is tried (up
       to `max_candidates`). Past the latency budget, longer prefixes only
       get slower, so the search stops.
    4. Nothing accepted: U16 if it meets the latency budget, else no choice.

    Returns the report (also written to `<work>/auto_report.json` and `.md`).
    `session_factory`/`builder` are injection points for tests.
    """
    from replay import load_scales

    builder = builder or (
        lambda root, layer_cfgs: pulsar2_build(
            root, model_path, calib, layer_cfgs, image=image
        )
    )
    session_factory = session_factory or _default_session
    model = onnx.load(model_path)
    rp = Replayer(model, dev)

    def visit(axmodel):
        with session_factory() as s:
            outs, ms = card_run(s, axmodel, dev, runs)
        return snr_db(rp.ref, outs), ms

    cands: List[Candidate] = []

    def record(c: Candidate, b: BuildOut) -> Candidate:
        c.build_s, c.cached = round(b.seconds, 1), b.cached
        if b.build_dir is None:
            c.note = "build failed: " + b.error[-300:]
            log(f"[auto] {c.label}: build failed")
            return c
        c.axmodel, c.tensors_16bit = b.axmodel, count_16bit(b.build_dir)
        c.card_db, c.latency_ms = visit(b.axmodel)
        log(
            f"[auto] {c.label}: predicted {c.predicted_db if c.predicted_db is None else round(c.predicted_db, 2)} dB, "
            f"card {c.card_db:.2f} dB, {c.latency_ms:.3f} ms, {c.tensors_16bit} tensors 16-bit"
            + (" (cached build)" if c.cached else "")
        )
        cands.append(c)
        return c

    u8b = builder(os.path.join(work, "U8"), None)
    u16b = builder(os.path.join(work, "U16"), u16_layer_configs(model))
    if u8b.build_dir is None or u16b.build_dir is None:
        raise RuntimeError(f"U8/U16 build failed: {u8b.error or u16b.error}")
    lo, hi = load_scales(u8b.build_dir), load_scales(u16b.build_dir)
    u8 = record(Candidate("U8", [], 0, rp.score(lo)), u8b)
    u16 = record(Candidate("U16", ["*"], -1, rp.score(hi)), u16b)
    floor = -20 * np.log10(max_err) if max_err is not None else u16.card_db - slack_db

    def ok(c: Candidate) -> bool:
        return (
            c.card_db is not None
            and c.card_db >= floor
            and (max_latency_ms is None or c.latency_ms <= max_latency_ms)
        )

    chosen = None
    if ok(u8):
        u8.accepted, chosen = True, u8
        u8.note = "U8 already meets the budget"
    else:
        b = Build(u8b.build_dir)
        # the card budget in replay terms: shifted by how far the replay of
        # the U16 build is from the card's U16
        pred_floor = floor - (u16.card_db - u16.predicted_db)

        def plan(how):
            groups = group(b, how)
            order = search(rp, b, lo, hi, groups, target=float("inf"))
            prefixes = [
                (order.groups[: k + 1], order.trace[k + 1][1])
                for k in range(len(order.groups))
            ]
            reach = next(
                (gs for gs, db in prefixes if db >= pred_floor - margin_db), None
            )
            cost = (
                len(b.widened([o for g in reach for o in groups[g]]))
                if reach
                else float("inf")
            )
            return groups, order, prefixes, cost

        if by == "auto":
            # whichever grouping's first candidate (the first prefix within
            # `margin_db` of the budget, as below) widens the fewest tensors:
            # op types win on transformers (the residual stream), blocks on
            # CNNs
            plans = {how: plan(how) for how in ("op_type", "layer")}
            by = min(plans, key=lambda h: (plans[h][3], h != "op_type"))
            log(
                "[auto] grouping: "
                + ", ".join(
                    f"{h} reaches it widening {p[3]} tensors" for h, p in plans.items()
                )
                + f" -> {by}"
            )
            groups, order, prefixes, _ = plans[by]
        else:
            groups, order, prefixes, _ = plan(by)
        # the replay of a mix runs pessimistic (DistilBERT LN+Add: 17.0 dB
        # replayed, 18.4 dB on the card), so a prefix up to `margin_db` short
        # of the budget is still worth a build; the card decides
        start = next(
            (i for i, (_, db) in enumerate(prefixes) if db >= pred_floor - margin_db),
            len(prefixes) - 1,
        )
        log(
            f"[auto] search order {order.groups}; budget {floor:.2f} dB on the card "
            f"({pred_floor:.2f} dB replayed); starting at prefix {start + 1}"
        )
        for gs, db in prefixes[start : start + max_candidates]:
            sel = list({o.name: o for g in gs for o in groups[g]}.values())
            if len(b.widened(sel)) >= len(b.widened(b.ops)):
                break  # this prefix is whole-model U16 in all but name
            label = "mix:" + "+".join(g.replace("AxQuantized", "") for g in gs)
            cfg = layer_configs(sel)
            c = Candidate(label, list(gs), len(sel), db, layer_configs=cfg)
            record(c, builder(os.path.join(work, "mix_" + _digest(cfg)), cfg))
            if ok(c):
                c.accepted, chosen = True, c
                break
            if (
                c.latency_ms is not None
                and max_latency_ms is not None
                and c.latency_ms > max_latency_ms
            ):
                c.note = "over the latency budget; longer prefixes are slower"
                break
            c.note = "card missed the budget; next prefix"
        if chosen is None and ok(u16):
            u16.accepted, chosen = True, u16
            u16.note = "fallback: no mix met the budget"
    report = {
        "model": model_path,
        "budget": {
            "snr_floor_db": round(float(floor), 3),
            "max_rel_err": _err(floor),
            "slack_db": slack_db if max_err is None else None,
            "max_latency_ms": max_latency_ms,
        },
        "grouping": by,
        "chosen": chosen.label if chosen else None,
        "layer_configs": chosen.layer_configs if chosen else None,
        "axmodel": chosen.axmodel if chosen else None,
        "candidates": [
            {k: v for k, v in vars(c).items() if k not in ("layer_configs",)}
            | {"card_rel_err": None if c.card_db is None else _err(c.card_db)}
            for c in cands
        ],
    }
    os.makedirs(work, exist_ok=True)
    with open(os.path.join(work, "auto_report.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    with open(os.path.join(work, "auto_report.md"), "w") as fh:
        fh.write(report_markdown(report))
    return report


def report_markdown(report: dict) -> str:
    rows = [
        "| candidate | ops promoted | 16-bit tensors | predicted dB | card dB | card rel. err | latency | build | result |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in report["candidates"]:
        fmt = lambda v, f: "-" if v is None else f.format(v)  # noqa: E731
        res = "**chosen**" if c["accepted"] else (c["note"] or "")
        rows.append(
            f"| {c['label']} | {'all' if c['n_ops'] < 0 else c['n_ops']} | "
            f"{fmt(c['tensors_16bit'], '{}')} | {fmt(c['predicted_db'], '{:.2f}')} | "
            f"{fmt(c['card_db'], '{:.2f}')} | {fmt(c['card_rel_err'], '{:.1%}')} | "
            f"{fmt(c['latency_ms'], '{:.3f} ms')} | "
            f"{'cached' if c['cached'] else str(c['build_s']) + ' s'} | {res} |"
        )
    b = report["budget"]
    head = (
        f"Budget: card SNR >= {b['snr_floor_db']:.2f} dB (rel. err <= {b['max_rel_err']:.1%})"
        + (f", latency <= {b['max_latency_ms']} ms" if b["max_latency_ms"] else "")
        + f". Chosen: {report['chosen']}.\n\n"
    )
    return head + "\n".join(rows) + "\n"


def _load_feeds(path: str) -> List[Feeds]:
    z = np.load(path)
    return [{k: z[k][i : i + 1] for k in z.files} for i in range(len(z[z.files[0]]))]


def auto_main(argv) -> int:
    p = argparse.ArgumentParser(
        prog="mixed_precision.py auto",
        description="build U8/U16, search, build and check the mix on the card",
    )
    p.add_argument("model", help="float ONNX with static input shapes")
    p.add_argument("--calib", required=True, help=".npz, axis 0 = sample")
    p.add_argument("--dev", required=True, help=".npz, axis 0 = sample")
    p.add_argument("--work", required=True, help="build cache + report dir")
    p.add_argument("--slack-db", type=float, default=1.0)
    p.add_argument("--max-err", type=float, help="relative-error budget instead")
    p.add_argument("--max-latency-ms", type=float)
    p.add_argument("--by", default="auto", choices=["auto", "op_type", "layer"])
    p.add_argument("--max-candidates", type=int, default=4)
    p.add_argument("--margin-db", type=float, default=3.0)
    a = p.parse_args(argv)
    rep = auto(
        os.path.abspath(a.model),
        _load_feeds(a.calib),
        _load_feeds(a.dev),
        os.path.abspath(a.work),
        a.slack_db,
        a.max_err,
        a.max_latency_ms,
        a.by,
        a.max_candidates,
        a.margin_db,
    )
    print(report_markdown(rep))
    return 0 if rep["chosen"] else 1


def main(argv=None) -> int:
    import sys

    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["auto"]:
        return auto_main(argv[1:])
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
