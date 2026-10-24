"""Host-only pieces of the Pulsar-free decode: the plan layout (mask, merged
linear column offsets, program order), the evaluation text, the error metric,
and ``FakeAX``, a numpy stand-in for ``axcl_session.AXSession``'s plan mode.

Nothing here loads a checkpoint or touches the card.

Plan layout of one decode step at position ``p`` (0..S-1): per layer a
token-major K / V cache ``[S + 1, nkv * hd]``; the step's own K / V row is
written into slot ``p``, rows ``0..p`` are valid in the mask (0), rows
``p + 1..S`` masked (``MASK``; row ``S`` always masked). In float this equals
``llm_layer_decomposition.layer_graph``'s "S slots + the new row at S"
(``mask_ref``): the softmax runs over the same set of rows.
"""

from __future__ import annotations

import os
import re

import numpy as np

F32 = np.float32
S = 128  # cache slots: positions 0..S-1
MASK = -32.0  # additive mask of an invalid row (exp(-32) is below U16 resolution)

#: programs of one layer, in execution order (``kc`` is inserted before ``rk``
#: for a model with per-head q / k RMSNorm)
LAYER = [
    "n1_c",
    "n1_rms",
    "qkv5",
    "rq",
    "rk",
    "attn",
    "o_proj",
    "res1",
    "n2_c",
    "n2_rms",
    "gu",
    "silu_f",
    "down_proj",
    "res2",
]
POST = ["nf_c", "nf_rms", "lm_head"]
LINEARS = ("qkv5", "o_proj", "gu", "down_proj", "lm_head")


def layer_programs(qk_norm: bool) -> list[str]:
    progs = list(LAYER)
    if qk_norm:
        progs.insert(progs.index("rk"), "kc")
    return progs


def mask_ref(p: int, s: int = S) -> np.ndarray:
    """``layer_graph``'s mask: slots ``p..s-1`` masked, row ``s`` (the new token) valid."""
    m = np.zeros((1, s + 1), F32)
    m[0, p:s] = MASK
    return m


def mask_plan(p: int, s: int = S) -> np.ndarray:
    """The plan layout's mask: the new row is in slot ``p``; rows ``0..p`` valid,
    ``p + 1..s`` masked."""
    m = np.full((1, s + 1), MASK, F32)
    m[0, : p + 1] = 0.0
    return m


def qkv_parts(nh: int, hd: int, kvd: int, qk_norm: bool) -> list[tuple[str, int]]:
    """Columns of the merged ``qkv5`` linear, in order. With a per-head k norm,
    ``k_proj`` sits next to ``k_rot``: the coarse k norm (``kc``) reads
    ``[k | k_rot]`` as one slice."""
    if qk_norm:
        return [
            ("q_proj", nh * hd),
            ("q_rot", nh * hd),
            ("k_proj", kvd),
            ("k_rot", kvd),
            ("v_proj", kvd),
        ]
    return [
        ("q_proj", nh * hd),
        ("k_proj", kvd),
        ("v_proj", kvd),
        ("q_rot", nh * hd),
        ("k_rot", kvd),
    ]


def gu_parts(inter: int) -> list[tuple[str, int]]:
    return [("gate_proj", inter), ("up_proj", inter)]


def offsets(parts) -> dict[str, int]:
    """Column offset of each ``(name, width)`` part."""
    o, out = 0, {}
    for k, n in parts:
        out[k] = o
        o += n
    return out


def rel_err(got, want) -> float:
    """RMS error of ``got`` over the RMS of ``want`` (the logits metric)."""
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    return float(np.sqrt(((got - want) ** 2).mean()) / np.sqrt((want**2).mean()))


def summarize(rows: list[dict]) -> dict:
    """``n``, top-1 agreement, median / p90 relative error of per-step rows
    (``top1``, ``float_top1``, ``rel``)."""
    rel = [r["rel"] for r in rows]
    return dict(
        n=len(rows),
        top1=sum(r["top1"] == r["float_top1"] for r in rows),
        median=float(np.median(rel)),
        p90=float(np.percentile(rel, 90)),
    )


# ---- text ---------------------------------------------------------------------------
#: calibration passages (repo docs); the held-out evaluation passages
CAL_DOCS = [
    "README.md",
    "docs/qat.md",
    "docs/accuracy-drop.md",
    "docs/attention-quantization.md",
]
EVAL_DOCS = ["docs/adaquant.md", "docs/activation-memory-planning.md"]


def prose(path: str, max_chars: int = 4000) -> str:
    """The running text of a Markdown file: code blocks, tables, headings,
    HTML and quotes dropped, link targets and emphasis marks removed."""
    out, code = [], False
    for line in open(path, encoding="utf-8"):
        if line.lstrip().startswith("```"):
            code = not code
            continue
        s = line.strip()
        if code or not s or s.startswith(("|", "#", "<", "!", ">")):
            continue
        s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)
        s = re.sub(r"[`*_]", "", s)
        out.append(s)
    return " ".join(out)[:max_chars]


# ---- a numpy stand-in for the runner's plan mode -----------------------------------
class _IO:
    def __init__(self, name: str, shape):
        self.name, self.shape = name, tuple(int(d) for d in shape)
        self.nbytes = 4 * int(np.prod(self.shape))


class _Model:
    def __init__(self, mid: int, inputs, outputs):
        self.id, self.inputs, self.outputs, self.bind = mid, inputs, outputs, {}


class FakeAX:
    """``BUF`` / ``BIND`` / ``PLAN`` and ``load`` of ``axcl_session.AXSession``
    over numpy float32 buffers. ``load(path, fn)`` takes a program's IO from
    its ONNX graph (initializers are not inputs) and runs ``fn(feeds)`` (input
    name -> array) for its ``X`` line. Plan lines: ``B mid I|O idx buf off``,
    ``X mid``, ``C dst doff src soff nbytes``, ``W buf off nbytes file``,
    ``R buf off nbytes file``."""

    def __init__(self, directory: str):
        self.host_dir = self.guest_dir = directory
        os.makedirs(os.path.join(directory, "t"), exist_ok=True)
        self.bufs: list[np.ndarray] = []
        self.models: list[_Model] = []
        self.fns: list = []

    def load(self, path: str, fn) -> _Model:
        import onnx

        g = onnx.load(path, load_external_data=False).graph
        inits = {i.name for i in g.initializer}

        def io(v):
            return _IO(v.name, [d.dim_value for d in v.type.tensor_type.shape.dim])

        m = _Model(
            len(self.models),
            [io(i) for i in g.input if i.name not in inits],
            [io(o) for o in g.output],
        )
        self.models.append(m)
        self.fns.append(fn)
        return m

    def _bind(self, args: str) -> None:
        mid, io_, idx, b, off = args.split()
        self.models[int(mid)].bind[(io_, int(idx))] = (int(b), int(off))

    def _cmd(self, line: str) -> str:
        op, _, args = line.partition(" ")
        if op == "BUF":
            self.bufs.append(np.zeros(int(args) // 4, F32))
            return f"OK {len(self.bufs) - 1}"
        if op == "BIND":
            self._bind(args)
            return "OK"
        if op == "PLAN":
            for ln in open(args).read().splitlines():
                if ln:
                    self._op(ln)
            return "OK 0 0"
        raise ValueError(f"FakeAX: unknown command {line!r}")

    def _op(self, ln: str) -> None:
        c, args = ln[0], ln[2:]
        if c == "B":
            self._bind(args)
        elif c == "X":
            m = self.models[int(args)]
            feeds = {}
            for i, spec in enumerate(m.inputs):
                b, off = m.bind[("I", i)]
                n = spec.nbytes // 4
                a = self.bufs[b][off // 4 : off // 4 + n]
                if a.size != n:
                    raise ValueError(
                        f"model {m.id} input {spec.name}: buffer too short"
                    )
                feeds[spec.name] = a.reshape(spec.shape).copy()
            y = np.asarray(self.fns[m.id](feeds), F32).ravel()
            b, off = m.bind[("O", 0)]
            self.bufs[b][off // 4 : off // 4 + y.size] = y
        elif c == "C":
            db, doff, sb, soff, n = map(int, args.split())
            self.bufs[db][doff // 4 : (doff + n) // 4] = self.bufs[sb][
                soff // 4 : (soff + n) // 4
            ]
        elif c in "WR":
            b, off, n, f = args.split()
            b, off, n = int(b), int(off), int(n)
            if c == "W":
                self.bufs[b][off // 4 : (off + n) // 4] = np.fromfile(f, F32)
            else:
                self.bufs[b][off // 4 : (off + n) // 4].tofile(f)
        else:
            raise ValueError(f"FakeAX: unknown plan line {ln!r}")
