"""Move a compiled single-op U16 program (AX650, float32 IO) to another calibration.

``retarget(template, op, template_q, target_q, inputs)`` returns the program of
the same op and shape at the target calibration, from one native Pulsar2
build at any other calibration; ``calibrate(directory, op)`` computes that
calibration from the program's samples, so no Pulsar2 build is needed per
calibration. Ops: Mul, Add (also with a constant operand, and the
attention-mask Add), Div, ReduceMean, Sqrt, Sigmoid, Softmax and dynamic
(two-input) MatMul, built with
``layer_configs: [{op_types: [<op>], data_type: U16}]``. A constant-weight
MatMul is a FullyConnected (``linear_emit.emit_linear_u16``).

A fused program (several ops in one native build: an RMSNorm, a SiLU block,
a RoPE apply) goes through ``retarget_graph(template, model, template_q,
target_q)`` with ``calibrate_graph(directory)``: the same rules per graph
node, node i's formulas only in the i-th CORE job of the main engine.

How a record is found
---------------------

Every A1 record of the template and every 2-byte-aligned ``npu_params`` word
is compared with candidate formulas evaluated on the template's calibration:
per tensor ``zp``, ``float32(1 / s)`` and ``float32(s)``, and the op's own
formulas (``candidates``). A match is rewritten with the same formula on the
target calibration. In the main engine a QUANT job may only hold an input's
``1 / s`` and ``zp``, a DEQUANT job only the output's ``s`` and ``zp``;
formulas tied to one register are written ``name@0xREG``. A record that
matches two formulas with different target values is refused (pick a
template whose tensors differ).

The formulas are the U8 rules of ``graph_stitch`` at 16 bits, plus:

* Add: ``0x1ea0 = 15 - k``, with ``k`` the shift that brings both scale
  ratios below 1 (``graph_stitch.add_const``), and the zero-point offset in
  ``Q(15 - k)``; with a constant operand the U16 constant comes first in
  ``npu_params`` and the ratio constant follows at byte 2.
* dynamic MatMul: ``npu_params`` holds lane blocks ``zp_y``, ``M``, ``zp_y``,
  ``256 M`` (``M = s_a s_b / s_y``).
* Sigmoid: a 256-entry u16 table, one entry per 256 input codes
  (``0x1050..0x1840``, two per register; entry 256 in ``0x1850``), in
  single precision.

Write elision
-------------

Pulsar2 leaves out a write of the value a register already holds (the first
write of each register in a segment is always there: the PARAM and QUANT jobs
reset them). After the rewrite, a write that became redundant is dropped and
the segment is re-padded with zero records to a multiple of four. A record
the template already left out cannot be restored, so a template must have
zero points above 255 (smaller ones are not located) and distinct zero points
and scales; ``retarget`` refuses otherwise.

Checked offline against native builds (``tests/test_axera_u16_retarget.py``):
every pair of builds of one op and shape, the SmolLM2-135M layer-0 programs
and calibration variants, gives records equal to the native build of the
target calibration. ``calibrate`` reproduces Pulsar2's U16 MinMax exactly on
all but one of those builds (a ReduceMean output scale two ulps off).
"""

from __future__ import annotations

import io
import json
import os
import struct
import sys
import tarfile
from collections.abc import Mapping, Sequence

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import graph_stitch as gs  # noqa: E402
import linear_emit as le  # noqa: E402
import misc_op_record_emit as mre  # noqa: E402
import short_unit_codec as suc  # noqa: E402
import width_retarget as wr  # noqa: E402

F = np.float32
OPS = ("Mul", "Add", "Div", "ReduceMean", "Sqrt", "Sigmoid", "Softmax", "MatMul")
SIGMOID_TABLE = 0x1050  # 128 registers, two u16 entries each
SIGMOID_TAIL = 0x1850
ADD_SHIFT = 0x1EA0


def _fbits(v) -> int:
    return int(np.asarray(F(v)).view(np.uint32))


def quant_of(directory: str) -> dict[str, tuple[float, int]]:
    """``{tensor: (scale, zero point)}`` of a build's ``quant_axmodel.json``
    (per-tensor entries only)."""
    q = json.load(open(os.path.join(directory, "out", "quant", "quant_axmodel.json")))
    out = {}
    for tc in q["tensor_configs"].values():
        for t, c in tc.items():
            v = q["values"].get(
                str(c["hash"])
            )  # absent: an inner tensor's consumer side
            if v is not None and len(v["scale"]) == 1:
                out[t] = (float(v["scale"][0]), int(v["zero_point"][0]))
    return out


# ---- calibration --------------------------------------------------------------------
def unsigned_minmax(lo: float, hi: float) -> tuple[float, int]:
    """U16 MinMax of a range folded with 0: ``zp = rint(-lo / s)``, except a
    symmetric range (``lo == -hi``, e.g. RoPE's sin table), where Pulsar2
    gives 32767 although ``-lo / s`` rounds to 32768."""
    lo, hi = min(float(lo), 0.0), max(float(hi), 0.0)
    s = F((hi - lo) / 65535.0)
    if lo == -hi and hi > 0:
        return float(s), 32767
    return float(s), int(np.clip(np.rint(-lo / float(s)), 0, 65535))


def signed_absmax(absmax: float) -> tuple[float, int]:
    """A dynamic MatMul operand: signed, ``s = max|x| / 32767.5``, zero point 0."""
    return float(F(F(absmax) / F(32767.5))), 0


def _samples(path: str) -> list[np.ndarray]:
    with tarfile.open(path) as t:
        members = sorted(
            (m for m in t.getmembers() if m.isfile()),
            key=lambda m: int(os.path.basename(m.name).split(".")[0]),
        )
        return [np.load(io.BytesIO(t.extractfile(m).read())) for m in members]


def calibrate(directory: str, op: str) -> dict[str, tuple[float, int]]:
    """The U16 calibration of a single-op build directory (``model.onnx``,
    ``dataset/<input>.tar``) from its samples: MinMax of each input, of each
    constant operand, and of the float op's output over every sample."""
    m = onnx.load(os.path.join(directory, "model.onnx"))
    consts = {i.name for i in m.graph.initializer}
    ins = [i.name for i in m.graph.input if i.name not in consts]
    data = {n: _samples(os.path.join(directory, "dataset", f"{n}.tar")) for n in ins}
    ref = ReferenceEvaluator(m)
    ys = [
        ref.run(None, {n: data[n][j].astype(F) for n in ins})[0]
        for j in range(len(data[ins[0]]))
    ]
    q = {}
    for n in ins:
        a = np.concatenate([x.reshape(-1) for x in data[n]])
        q[n] = (
            signed_absmax(np.abs(a).max())
            if op == "MatMul"
            else unsigned_minmax(a.min(), a.max())
        )
    if op != "MatMul":
        for t in m.graph.initializer:
            a = onnx.numpy_helper.to_array(t)
            q[t.name] = unsigned_minmax(a.min(), a.max())
    y = np.concatenate([v.reshape(-1) for v in ys])
    q[m.graph.output[0].name] = unsigned_minmax(y.min(), y.max())
    return q


# ---- record formulas ----------------------------------------------------------------
def candidates(
    op: str,
    q: Mapping[str, tuple[float, int]],
    inputs: Sequence[str],
    out: str = "y",
    attrs: Mapping | None = None,
) -> dict[str, int]:
    """Formula name -> 32-bit record value on calibration ``q``."""
    if op not in OPS:
        raise ValueError(f"{op}: not a retargetable U16 op ({', '.join(OPS)})")
    s = {t: v[0] for t, v in q.items()}
    z = {t: v[1] for t, v in q.items()}
    c = {}
    for t in q:
        c[f"zp:{t}"] = z[t]
        c[f"inv:{t}"] = _fbits(1 / F(s[t]))
        c[f"s:{t}"] = _fbits(s[t])
    if op == "Mul":
        a, b = inputs
        c["mul_lane"] = _fbits(s[out] / (s[a] * s[b]))
    elif op == "Add":
        a, b = inputs
        k = 0
        while max(s[a] / s[out], s[b] / s[out]) * 2.0**-k >= 1.0:
            k += 1
        c[f"add_shift@{ADD_SHIFT:#06x}"] = 15 - k
        c["add_offset"] = int(
            (z[out] - z[a] * gs._f32(s[a] / s[out]) - z[b] * gs._f32(s[b] / s[out]))
            * 2 ** (15 - k)
        )
        c["add_const"] = struct.unpack("<I", gs.add_const(s[a], s[b], s[out]))[0]
    elif op == "Div":
        a, b = inputs
        c["div_lane"] = _fbits(s[a] / (s[b] * s[out]))
    elif op == "ReduceMean":
        (x,) = inputs
        n = int((attrs or {})["count"])
        c["rm_lane"] = gs.reducemean_lane(s[x], s[out], n)
        c["rm_zp"] = z[x] * min(n, gs.REDUCE_WINDOW)
    elif op == "MatMul":
        a, b = inputs
        m = F(s[a] * s[b] / s[out])
        c["mm_scale"] = _fbits(m)
        c["mm_scale256"] = _fbits(F(m * F(256)))
        c["mm_zp_f"] = _fbits(F(z[out]))
    elif op == "Sigmoid":
        (x,) = inputs
        xf = ((np.arange(257) * 256 - z[x]).astype(F) * F(s[x])).astype(F)
        sg = (F(1) / (F(1) + np.exp(-xf).astype(F))).astype(F)
        t = np.clip(np.rint((sg / F(s[out])).astype(F)) + z[out], 0, 65535).astype(int)
        for k in range(128):
            c[f"sig_t{k}@{SIGMOID_TABLE + 16 * k:#06x}"] = int(t[2 * k]) | (
                int(t[2 * k + 1]) << 16
            )
        c[f"sig_tail@{SIGMOID_TAIL:#06x}"] = int(t[256])
    return {k: v & 0xFFFFFFFF for k, v in c.items()}


def _allowed(role: str, name: str, inputs, out: str, reg: int | None) -> bool:
    if "@" in name and int(name.split("@")[1], 16) != reg:
        return False
    if role == "QUANT":
        return name in {f"inv:{t}" for t in inputs} | {f"zp:{t}" for t in inputs}
    if role == "DEQUANT":
        return name in (f"s:{out}", f"zp:{out}")
    return True


def _segs(model) -> list[list[bytes]]:
    mc = bytes(mre.mcode_initializer(model).raw_data)
    return [gs.records(r) for r in suc.decode_segments(mc)]


def _base(name: str) -> str:
    """A candidate's formula name without its node prefix (``n<i>|``)."""
    return name.split("|", 1)[-1]


def _rewrite(template, ct, cn, allowed, n_core=None) -> onnx.ModelProto:
    """Rewrite every A1 record and ``npu_params`` word of ``template`` that
    matches one candidate of ``ct`` (allowed by ``allowed(tag, name, reg)``,
    ``tag`` one of QUANT, DEQUANT, PARAM, CORE<i> (the i-th CORE job of the
    main engine) or OTHER (other segments)) with its ``cn`` value, then
    apply Pulsar2's write elision and re-pad the segments."""
    model = gs.load_model(template)
    by_val: dict[int, list[str]] = {}
    for k, v in sorted(ct.items()):
        if _base(k).startswith("zp:") and v < 256:
            if cn[k] != v:
                raise ValueError(
                    f"template {k} = {v}: below 256 it cannot be located in the records, "
                    f"and the target needs {cn[k]}; use a template whose zero points are > 255"
                )
            continue
        by_val.setdefault(v, []).append(k)
    segs = _segs(model)
    tag_at, core = {}, 0
    for role, a, b, _ in le._jobs(segs[gs.MAIN]):
        tag = role
        if role == "CORE":
            tag, core = f"CORE{core}", core + 1
        tag_at.update({i: tag for i in range(a, b)})
    if n_core is not None and core != n_core:
        raise ValueError(f"the template has {core} CORE jobs for {n_core} graph nodes")
    touched = [set() for _ in segs]
    for si, seg in enumerate(segs):
        for i, r in enumerate(seg):
            if r[0] != gs.A1:
                continue
            reg, val = gs._reg(r), gs._val(r)
            tag = tag_at.get(i, "OTHER") if si == gs.MAIN else "OTHER"
            names = [n for n in by_val.get(val, []) if allowed(tag, n, reg)]
            if len({cn[n] for n in names}) > 1:
                raise ValueError(
                    f"segment {si} record {i} ({reg:#06x} = {val:#x}) matches {names}: "
                    "ambiguous template"
                )
            if names:
                seg[i] = gs._rec(gs.A1, reg, cn[names[0]])
                touched[si].add(reg)
    # write elision on the calibration registers
    tsegs = _segs(model)
    for si, seg in enumerate(segs):
        if not touched[si]:
            continue
        st_t, st_n, keep = {}, {}, []
        for r, rt in zip(seg, tsegs[si]):
            if r[0] == gs.A1 and gs._reg(r) in touched[si]:
                reg = gs._reg(r)
                if st_t.get(reg) == gs._val(rt):
                    raise ValueError(
                        f"the template repeats a write to {reg:#06x}; a template with "
                        "distinct zero points and scales is needed"
                    )
                st_t[reg] = gs._val(rt)
                if st_n.get(reg) == gs._val(r):
                    continue
                st_n[reg] = gs._val(r)
            keep.append(r)
        while keep and keep[-1] == bytes(8):
            keep.pop()
        segs[si] = keep + [bytes(8)] * (-len(keep) % 4)
    result = wr.with_segments(model, segs)
    params = next((t for t in result.graph.initializer if t.name == "npu_params"), None)
    if params is not None:
        p = bytearray(params.raw_data)
        for off in range(0, len(p) - 3, 2):
            val = struct.unpack_from("<I", p, off)[0]
            names = [
                n
                for n in by_val.get(val, [])
                if not _base(n).startswith(("zp:", "inv:", "s:"))
            ]
            if len(names) == 1:
                struct.pack_into("<I", p, off, cn[names[0]])
        params.raw_data = bytes(p)
    return result


def retarget(
    template,
    op: str,
    template_q: Mapping[str, tuple[float, int]],
    target_q: Mapping[str, tuple[float, int]],
    inputs: Sequence[str],
    out: str = "y",
    attrs: Mapping | None = None,
) -> onnx.ModelProto:
    """The template program at calibration ``target_q``. Both calibrations
    are keyed by the template's tensor names (``inputs`` and ``out``); the
    result keeps the template's IO names."""
    ct = candidates(op, template_q, inputs, out, attrs)
    cn = candidates(op, target_q, inputs, out, attrs)

    def allowed(tag, name, reg):
        return _allowed(tag, name, inputs, out, reg)

    return _rewrite(template, ct, cn, allowed)


# ---- fused (multi-op) programs ---------------------------------------------------------
UNARY = ("Sqrt", "Sigmoid", "Softmax", "ReduceMean")


def graph_nodes(model) -> list[tuple[str, tuple, str, dict]]:
    """``(op, inputs, output, attrs)`` per node of a fused program's float
    graph (``model.onnx``); ReduceMean's ``count`` is the reduced width."""
    m = onnx.shape_inference.infer_shapes(
        model if isinstance(model, onnx.ModelProto) else onnx.load(model)
    )
    dims = {
        v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
        for v in [*m.graph.input, *m.graph.value_info, *m.graph.output]
    }
    out = []
    for n in m.graph.node:
        attrs = {}
        if n.op_type == "ReduceMean":
            sh = dims[n.input[0]]
            axes = next((list(a.ints) for a in n.attribute if a.name == "axes"), [-1])
            attrs["count"] = int(np.prod([sh[a] for a in axes]))
        out.append((n.op_type, tuple(n.input), n.output[0], attrs))
    return out


def _graph_candidates(nodes, q) -> dict[str, int]:
    c = {}
    for i, (op, ins, out, attrs) in enumerate(nodes):
        sub = {t: q[t] for t in {*ins, out}}
        used = ins[:1] if op in UNARY else ins
        for k, v in candidates(op, sub, used, out, attrs).items():
            c[f"n{i}|{k}"] = v
    return c


def retarget_graph(
    template,
    model,
    template_q: Mapping[str, tuple[float, int]],
    target_q: Mapping[str, tuple[float, int]],
    out: str | None = None,
) -> onnx.ModelProto:
    """``retarget`` for a fused program (several ops in one native build).
    ``model`` is its float graph; the calibrations cover every tensor, inner
    ones included (``quant_of`` / ``calibrate_graph``). Node i's formulas may
    only sit in the i-th CORE job of the main engine (CORE jobs follow the
    node order); QUANT jobs hold a graph input's ``1 / s`` and zero point,
    the DEQUANT job the output's ``s`` and zero point; PARAM jobs and the
    other segments any node's."""
    m = model if isinstance(model, onnx.ModelProto) else onnx.load(model)
    nodes = graph_nodes(m)
    consts = {i.name for i in m.graph.initializer}
    graph_ins = [i.name for i in m.graph.input if i.name not in consts]
    out = out or m.graph.output[0].name
    ct, cn = _graph_candidates(nodes, template_q), _graph_candidates(nodes, target_q)

    def allowed(tag, name, reg):
        node, base = name.split("|", 1)
        if "@" in base and int(base.split("@")[1], 16) != reg:
            return False
        if tag == "QUANT":
            return base in {f"inv:{t}" for t in graph_ins} | {
                f"zp:{t}" for t in graph_ins
            }
        if tag == "DEQUANT":
            return base in (f"s:{out}", f"zp:{out}")
        if tag.startswith("CORE"):
            return node == f"n{tag[4:]}"
        return True

    return _rewrite(template, ct, cn, allowed, n_core=len(nodes))


def calibrate_graph(directory: str) -> dict[str, tuple[float, int]]:
    """``calibrate`` for a fused program: U16 MinMax of every tensor (inputs,
    constants, inner tensors, output) over the float forward of the whole
    graph. Pulsar2's inner scales can be a float32 ulp off this (seen on a
    fused RMSNorm); the zero points agree."""
    m = onnx.load(os.path.join(directory, "model.onnx"))
    consts = {i.name: onnx.numpy_helper.to_array(i) for i in m.graph.initializer}
    ins = [i.name for i in m.graph.input if i.name not in consts]
    data = {n: _samples(os.path.join(directory, "dataset", f"{n}.tar")) for n in ins}
    ref = ReferenceEvaluator(m)
    lo, hi = {}, {}
    for j in range(len(data[ins[0]])):
        res = ref.run(None, {n: data[n][j].astype(F) for n in ins}, intermediate=True)
        for k, v in res.items():
            a = np.asarray(v)
            if k in consts or a.dtype.kind != "f" or a.size == 0:
                continue
            lo[k] = min(lo.get(k, np.inf), float(a.min()))
            hi[k] = max(hi.get(k, -np.inf), float(a.max()))
    q = {k: unsigned_minmax(lo[k], hi[k]) for k in lo}
    for k, a in consts.items():
        q[k] = unsigned_minmax(a.min(), a.max())
    return q


def compare(got, want) -> list[str]:
    """Differences that carry calibration between two compiled single-op
    programs: ``npu_params`` and, per segment, every record but the A8 ones in
    order (segment 0's slot table as a set) and the A8 records as a set. The
    IO names and their order (which flips between builds, with the A8 slot
    records that follow it) are not compared."""
    diffs = []
    pa = next(i.raw_data for i in got.graph.initializer if i.name == "npu_params")
    pb = next(i.raw_data for i in want.graph.initializer if i.name == "npu_params")
    if pa != pb:
        diffs.append("npu_params")
    for si, (ra, rb) in enumerate(zip(_segs(got), _segs(want))):
        if len(ra) != len(rb):
            diffs.append(f"segment {si}: {len(ra)} records vs {len(rb)}")
            continue
        lo, hi = gs.slot_table(rb) if si == 0 else (0, 0)
        if sorted(ra[lo:hi]) != sorted(rb[lo:hi]):
            diffs.append("segment 0 slot table")
        if sorted(r for r in ra if r[0] == gs.A8) != sorted(
            r for r in rb if r[0] == gs.A8
        ):
            diffs.append(f"segment {si}: A8 records")
        diffs += [
            f"segment {si} record {i}: {x.hex()} vs {y.hex()}"
            for i, (x, y) in enumerate(zip(ra, rb))
            if x != y and not lo <= i < hi and x[0] != gs.A8
        ]
    return diffs
