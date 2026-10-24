"""Native U16 template builds (Pulsar2) for every program shape of a model,
each at four calibration variants, for ``pf_generate.py`` to retarget. This
is the only step that runs Pulsar2 (``pulsar2_docker``, image
``pulsar2:7.0-lite``), once per shape; the programs themselves are emitted
without it.

Template directory (``--root``, default ``$LLM_PF_TEMPLATES`` or
``<LLM_PF_OUT>/../templates``):

* ``fused/<name>/v<i>``: fused multi-op programs
  (``layer_configs: [{op_types: [...], data_type: U16}]``);
* ``single/add<n>/v<i>``: the single-op residual Add;
* ``linear/k<K>n<N>``: a ``[1, K] @ [K, N]`` FullyConnected at U16
  activations (weights ``default_rng(1).uniform(-0.1, 0.1)``).

Fused graphs (``graph(name, dims)``; the generated programs use the same
graph, so ``FusedEval`` / ``float_graph`` evaluate exactly what the card runs):

* ``crms<n>``: coarse norm ``u = x / sqrt(mean(x^2) + eps)`` (first pass of
  the two-pass RMSNorm), ``rms<n>``: RMSNorm with the gain ``g`` as an input;
* ``rope<n>``: ``y = q * c + r * sn`` (``r`` = rotate_half(q), from the merged
  linear's ``q_rot`` / ``k_rot`` columns);
* ``qkrope<n>`` (Qwen3): per-head q / k RMSNorm fused with RoPE,
  ``y = (q*c + r*sn) / sqrt(mean_head(q^2) + eps)`` with the norm gain folded
  into the host's tables; ``qkrope<n>t`` the same graph with calibrations whose
  two Adds shift differently (an elided ``0x1ea0`` write cannot be restored);
* ``hcrms<n>`` (Qwen3): per-head coarse norm of ``[k | k_rot]``;
* ``attn<n>``: the whole layer's attention, ``q [1, nh*hd]`` as
  ``[nkv, G, hd]``, token-major K / V ``[S + 1, nkv*hd]`` (the plan's cache
  layout), mask ``[1, S + 1]`` (Reshape / Transpose inside);
* ``silu<n>``: ``y = gate * sigmoid(gate) * up``.

usage: MODEL=llama160m pf_templates.py [--root DIR] [--list] [NAME...]
(no NAME: every template the model needs that is not built yet; serial builds)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass

import numpy as np
import onnx
from onnx.parser import parse_model

HERE = os.path.dirname(os.path.abspath(__file__))
AXERA = os.path.dirname(HERE)
for _p in (HERE, AXERA):
    if _p not in sys.path:
        sys.path.insert(0, _p)

F = np.float32
OPSET = 'ir_version: 9, opset_import: ["" : 17]'
IMAGE = "pulsar2:7.0-lite"


@dataclass(frozen=True)
class Dims:
    H: int
    I: int  # noqa: E741
    nh: int
    nkv: int
    hd: int
    eps: float
    vocab: int
    S: int = 128

    @classmethod
    def of(cls, cfg) -> "Dims":
        return cls(cfg.H, cfg.I, cfg.nh, cfg.nkv, cfg.hd, cfg.eps, cfg.vocab, cfg.S)


CRMS = """
  sq = Mul(x, x)
  m = ReduceMean<axes=[-1], keepdims=1>(sq)
  e = Add(m, eps)
  r = Sqrt(e)
  y = Div(x, r)
"""
RMS = """
  sq = Mul(x, x)
  m = ReduceMean<axes=[-1], keepdims=1>(sq)
  e = Add(m, eps)
  r = Sqrt(e)
  d = Div(x, r)
  y = Mul(d, g)
"""
HCRMS = """
  xh = Reshape(x, shh)
  sq = Mul(xh, xh)
  m = ReduceMean<axes=[-1], keepdims=1>(sq)
  e = Add(m, eps)
  r = Sqrt(e)
  uh = Div(xh, r)
  y = Reshape(uh, shy)
"""
ROPE = """
  qa = Mul(q, c)
  qb = Mul(r, sn)
  y = Add(qa, qb)
"""
QKROPE = """
  qa = Mul(q, c)
  qb = Mul(r, sn)
  t = Add(qa, qb)
  qh = Reshape(q, shh)
  sq = Mul(qh, qh)
  m = ReduceMean<axes=[-1], keepdims=1>(sq)
  e = Add(m, eps)
  rr = Sqrt(e)
  th = Reshape(t, shh)
  yh = Div(th, rr)
  y = Reshape(yh, shy)
"""
SILU = """
  sg = Sigmoid(gate)
  sl = Mul(gate, sg)
  y = Mul(sl, up)
"""
ATTN = """
  qh = Reshape(q, shq)
  kh = Reshape(k, shk)
  kt = Transpose<perm=[1,2,0]>(kh)
  s = MatMul(qh, kt)
  sm = Add(s, mask)
  p = Softmax<axis=-1>(sm)
  vh = Reshape(v, shk)
  vt = Transpose<perm=[1,0,2]>(vh)
  o = MatMul(p, vt)
  y = Reshape(o, sho)
"""
NORM_OPS = ["Mul", "ReduceMean", "Add", "Sqrt", "Div"]


# ---- calibration generators of the four variants (Pulsar2 MinMax over 18 draws) ------
def _norm_gen(n, xs, off, g=None):
    def gen(r):
        d = {"x": (r.standard_normal((1, n)) * xs + off).astype(F)}
        if g is not None:
            d["g"] = r.uniform(g[0], g[1], (1, n)).astype(F)
        return d

    return gen


def _rope_gen(n, qs, qoff, cs):
    def gen(r):
        return {
            "q": (r.standard_normal((1, n)) * qs + qoff).astype(F),
            "r": (r.standard_normal((1, n)) * qs - qoff).astype(F),
            "c": r.uniform(-0.6 * cs, cs, (1, n)).astype(F),
            "sn": r.uniform(-cs, 0.8 * cs, (1, n)).astype(F),
        }

    return gen


def _rope_shift_gen(n, qs, cs, k):
    """r ~ -q and sn ~ c: q*c + r*sn mostly cancels, so its Add's scale ratios
    reach past 1 (shift < 15) while the eps Add stays at 15."""

    def gen(r):
        q = (r.standard_normal((1, n)) * qs).astype(F)
        c = r.uniform(-0.6 * cs, cs, (1, n)).astype(
            F
        )  # straddles 0: nonzero zero points
        return {
            "q": q,
            "r": (-q * k + r.standard_normal((1, n)) * qs * 0.05).astype(F),
            "c": c,
            "sn": (c * r.uniform(0.9, 1.1, (1, n))).astype(F),
        }

    return gen


def _silu_gen(n, gs, goff, us, uoff):
    def gen(r):
        return {
            "gate": (r.standard_normal((1, n)) * gs + goff).astype(F),
            "up": (r.standard_normal((1, n)) * us + uoff).astype(F),
        }

    return gen


def _attn_gen(d: Dims, qs, ks, vs, voff):
    def gen(r):
        valid = int(r.integers(1, d.S))
        m = np.zeros((1, d.S + 1), F)
        m[0, valid:] = -32.0
        return {
            "q": (r.standard_normal((1, d.nh * d.hd)) * qs).astype(F),
            "k": (r.standard_normal((d.S + 1, d.nkv * d.hd)) * ks).astype(F),
            "v": (r.standard_normal((d.S + 1, d.nkv * d.hd)) * vs + voff).astype(F),
            "mask": m,
        }

    return gen


NORM_X = ((1.0, 0.1), (3.0, -0.4), (0.5, 0.2), (8.0, 1.0))
NORM_G = ((-0.6, 0.6), (-0.3, 1.2), (0.1, 2.0), (-2.0, 0.5))
#: the single-op Add's input ranges per variant
ADD_RANGES = [
    {"a": (-2.0, 3.0), "b": (-1.0, 1.0)},
    {"a": (-1.0, 1.0), "b": (-0.5, 4.0)},
    {"a": (-3.0, 0.5), "b": (-2.0, 3.0)},
    {"a": (-8.0, 6.0), "b": (-1.0, 0.5)},
]


def graph(name: str, d: Dims):
    """``(op types, body, inputs {name: shape}, output shape, initializers,
    variant generators)`` of a fused template ``name``."""
    m = re.fullmatch(r"(crms|rms|hcrms|rope|qkrope|silu|attn)(\d+)(t?)", name)
    if not m:
        raise ValueError(f"{name}: not a fused template name")
    kind, n, alt = m.group(1), int(m.group(2)), m.group(3)
    eps = {"eps": np.array([d.eps], F)}
    heads = {
        "shh": np.array([n // d.hd, d.hd], np.int64),
        "shy": np.array([1, n], np.int64),
    }
    four = {"q": [1, n], "r": [1, n], "c": [1, n], "sn": [1, n]}
    if kind == "crms":
        gens = [_norm_gen(n, s, o) for s, o in NORM_X]
        return NORM_OPS, CRMS, {"x": [1, n]}, [1, n], eps, gens
    if kind == "rms":
        gens = [_norm_gen(n, s, o, g) for (s, o), g in zip(NORM_X, NORM_G)]
        return NORM_OPS, RMS, {"x": [1, n], "g": [1, n]}, [1, n], eps, gens
    if kind == "hcrms":
        gens = [_norm_gen(n, s, o) for s, o in NORM_X]
        ops = NORM_OPS + ["Reshape"]
        return ops, HCRMS, {"x": [1, n]}, [1, n], {**eps, **heads}, gens
    if kind == "rope":
        p = ((3.0, 0.5, 0.125), (1.0, -0.3, 1.0), (6.0, 1.0, 0.2), (2.0, 0.2, 0.5))
        return ["Mul", "Add"], ROPE, four, [1, n], {}, [_rope_gen(n, *a) for a in p]
    if kind == "qkrope":
        if alt:
            p = ((3.0, 1.0, 0.95), (1.0, 0.5, 1.0), (6.0, 0.3, 0.9), (2.0, 0.8, 0.98))
            gens = [_rope_shift_gen(n, *a) for a in p]
        else:
            p = ((3.0, 0.5, 0.15), (1.0, -0.3, 1.0), (6.0, 1.0, 0.3), (2.0, 0.2, 0.6))
            gens = [_rope_gen(n, *a) for a in p]
        ops = NORM_OPS + ["Reshape"]
        return ops, QKROPE, four, [1, n], {**eps, **heads}, gens
    if kind == "silu":
        p = (
            (1.0, 0.0, 1.0, 0.1),
            (2.0, -0.5, 0.7, -0.2),
            (0.8, 0.3, 1.5, 0.0),
            (1.5, -0.2, 2.0, 0.4),
        )
        gens = [_silu_gen(n, *a) for a in p]
        return (
            ["Sigmoid", "Mul"],
            SILU,
            {"gate": [1, n], "up": [1, n]},
            [1, n],
            {},
            gens,
        )
    # attention
    if n != d.nh * d.hd:
        raise ValueError(f"{name}: the model's attention is attn{d.nh * d.hd}")
    kv = d.nkv * d.hd
    shapes = {
        "shq": np.array([d.nkv, d.nh // d.nkv, d.hd], np.int64),
        "shk": np.array([d.S + 1, d.nkv, d.hd], np.int64),
        "sho": np.array([1, n], np.int64),
    }
    ins = {"q": [1, n], "k": [d.S + 1, kv], "v": [d.S + 1, kv], "mask": [1, d.S + 1]}
    p = (
        (0.3, 2.0, 0.5, 0.05),
        (0.15, 3.0, 1.0, -0.1),
        (0.5, 1.0, 0.3, 0.02),
        (0.2, 4.0, 0.8, 0.1),
    )
    gens = [_attn_gen(d, *a) for a in p]
    ops = ["MatMul", "Add", "Softmax", "Transpose", "Reshape"]
    return ops, ATTN, ins, [1, n], shapes, gens


def model_of(body: str, ins: dict, out, inits: dict) -> onnx.ModelProto:
    sig = ", ".join(f"float[{', '.join(map(str, s))}] {n}" for n, s in ins.items())
    m = parse_model(
        f"<{OPSET}> g ({sig}) => (float[{', '.join(map(str, out))}] y)\n{{{body}}}\n"
    )
    for n, a in inits.items():
        m.graph.initializer.append(onnx.numpy_helper.from_array(a, n))
    onnx.checker.check_model(m)
    return m


def float_graph(name: str, d: Dims) -> onnx.ModelProto:
    """The float graph of fused template ``name`` (what the program computes)."""
    _, body, ins, out, inits, _ = graph(name, d)
    return model_of(body, ins, out, inits)


def add_model(n: int) -> onnx.ModelProto:
    return parse_model(
        f"<{OPSET}> g (float[1,{n}] a, float[1,{n}] b) => (float[1,{n}] y) "
        "{ y = Add(a, b) }"
    )


def linear_model(w: np.ndarray) -> onnx.ModelProto:
    k, n = w.shape
    m = parse_model(
        f"<{OPSET}> lin (float[1,{k}] x) => (float[1,{n}] y) {{ y = MatMul(x, w) }}"
    )
    m.graph.initializer.append(onnx.numpy_helper.from_array(w, "w"))
    return m


def needed(cfg) -> list[str]:
    """Template names (``fused`` / ``add<n>`` / ``k<K>n<N>``) of a model's programs."""
    H, inter, qd, kvd = cfg.H, cfg.I, cfg.nh * cfg.hd, cfg.kvd
    names = [f"crms{H}", f"rms{H}", f"silu{inter}", f"attn{qd}", f"add{H}"]
    if cfg.qk_norm:
        names += [f"qkrope{qd}", f"qkrope{kvd}", f"qkrope{qd}t", f"qkrope{kvd}t"]
        names += [f"hcrms{2 * kvd}"]
    else:
        names += [f"rope{qd}", f"rope{kvd}"]
    widths = {"gu": 2 * inter, "down_proj": H}
    names += [
        f"add{widths[k]}" for k in cfg.split_first["keys"] if cfg.split_first["layers"]
    ]
    qkv = sum(n for _, n in cfg.QKV)
    names += [f"k{H}n{qkv}", f"k{qd}n{H}", f"k{H}n{2 * inter}", f"k{inter}n{H}"]
    names += [f"k{H}n{cfg.vocab}"]
    return list(dict.fromkeys(names))


def default_root(cfg=None) -> str:
    if os.environ.get("LLM_PF_TEMPLATES"):
        return os.environ["LLM_PF_TEMPLATES"]
    out = cfg.out if cfg is not None else os.environ.get("LLM_PF_OUT", ".")
    return os.path.join(os.path.dirname(os.path.abspath(out)), "templates")


def template_dir(root: str, name: str) -> str:
    if re.fullmatch(r"k\d+n\d+", name):
        return os.path.join(root, "linear", name)
    if name.startswith("add"):
        return os.path.join(root, "single", name)
    return os.path.join(root, "fused", name)


# ---- Pulsar2 builds ---------------------------------------------------------------------
def _pulsar2_build(W, ins, samples, ops):
    import pulsar2_docker as P

    os.makedirs(os.path.join(W, "dataset"), exist_ok=True)
    cfgs = []
    for n in ins:
        P.make_numpy_calibration_tar(os.path.join(W, "dataset", f"{n}.tar"), samples[n])
        cfgs.append(
            {
                "tensor_name": n,
                "calibration_dataset": f"./dataset/{n}.tar",
                "calibration_format": "Numpy",
                "calibration_size": len(samples[n]),
            }
        )
    cfg = {
        "model_type": "ONNX",
        "npu_mode": "NPU1",
        "compiler": {"check": 0},
        "quant": {
            "input_configs": cfgs,
            "calibration_method": "MinMax",
            "precision_analysis": False,
            "layer_configs": [{"op_types": ops, "data_type": "U16"}],
        },
    }
    json.dump(cfg, open(os.path.join(W, "config.json"), "w"), indent=1)
    t0 = time.time()
    res = P.build(
        W, "model.onnx", "out", config_path="config.json", image=IMAGE, timeout=1800
    )
    ok = bool(res.success) and os.path.exists(
        os.path.join(W, "out", "compiled.axmodel")
    )
    err = (res.error or "")[-500:]
    json.dump(
        dict(ok=ok, error=err, seconds=time.time() - t0),
        open(os.path.join(W, "result.json"), "w"),
        indent=1,
    )
    return ok, err


def build_fused(root: str, name: str, d: Dims) -> None:
    """Four variants; samples: 18 draws of the variant's generator (seed 300 + i)."""
    ops, body, ins, out, inits, gens = graph(name, d)
    for i, gen in enumerate(gens):
        W = os.path.join(template_dir(root, name), f"v{i}")
        if os.path.exists(os.path.join(W, "result.json")):
            continue
        os.makedirs(W, exist_ok=True)
        onnx.save(model_of(body, ins, out, inits), os.path.join(W, "model.onnx"))
        r = np.random.default_rng(300 + i)
        draws = [gen(r) for _ in range(18)]
        ok, err = _pulsar2_build(W, ins, {n: [x[n] for x in draws] for n in ins}, ops)
        print(name, i, ok, "" if ok else err[-200:], flush=True)


def build_add(root: str, name: str) -> None:
    """Four variants; per input 16 uniform samples over the variant's range plus
    the two range ends as constant samples (seed 100 + i)."""
    n = int(name[3:])
    for i, rng in enumerate(ADD_RANGES):
        W = os.path.join(template_dir(root, name), f"v{i}")
        if os.path.exists(os.path.join(W, "result.json")):
            continue
        os.makedirs(W, exist_ok=True)
        onnx.save(add_model(n), os.path.join(W, "model.onnx"))
        r = np.random.default_rng(100 + i)
        samples = {}
        for t in ("a", "b"):
            lo, hi = rng[t]
            s = [r.uniform(lo, hi, (1, n)).astype(F) for _ in range(16)]
            samples[t] = s + [np.full((1, n), lo, F), np.full((1, n), hi, F)]
        ok, err = _pulsar2_build(W, ("a", "b"), samples, ["Add"])
        print(name, i, ok, "" if ok else err[-200:], flush=True)


def build_linear(root: str, name: str) -> None:
    """One build; calibration: 16 samples of default_rng(0).uniform(-1, 1)."""
    import pulsar2_docker as P

    k, n = map(int, re.fullmatch(r"k(\d+)n(\d+)", name).groups())
    W = template_dir(root, name)
    if os.path.exists(os.path.join(W, "result.json")):
        return
    os.makedirs(os.path.join(W, "dataset"), exist_ok=True)
    w = np.random.default_rng(1).uniform(-0.1, 0.1, (k, n)).astype(F)
    onnx.save(linear_model(w), os.path.join(W, "model.onnx"))
    r = np.random.default_rng(0)
    xs = [r.uniform(-1, 1, (1, k)).astype(F) for _ in range(16)]
    P.make_numpy_calibration_tar(os.path.join(W, "dataset", "x.tar"), xs)
    cfg = P._llm_quant_config({"x": "./dataset/x.tar"}, 16)
    cfg["quant"]["layer_configs"] = [
        {"op_types": ["FullyConnected"], "data_type": "U16"}
    ]
    json.dump(cfg, open(os.path.join(W, "config.json"), "w"), indent=2)
    t0 = time.time()
    res = P.build(
        W, "model.onnx", "out", config_path="config.json", image=IMAGE, timeout=1800
    )
    ok = bool(res.success) and os.path.exists(
        os.path.join(W, "out", "compiled.axmodel")
    )
    json.dump(
        dict(ok=ok, error=(res.error or "")[-500:], seconds=time.time() - t0),
        open(os.path.join(W, "result.json"), "w"),
        indent=1,
    )
    print(name, ok, f"{time.time() - t0:.0f}s", flush=True)


def build(root: str, name: str, d: Dims) -> None:
    if re.fullmatch(r"k\d+n\d+", name):
        build_linear(root, name)
    elif name.startswith("add"):
        build_add(root, name)
    else:
        build_fused(root, name, d)


def main():
    import pf_model

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--root", default=None)
    ap.add_argument("--list", action="store_true", help="print the names and exit")
    a = ap.parse_args()
    cfg = pf_model.load()
    names = a.names or needed(cfg)
    if a.list:
        print("\n".join(names))
        return
    root = a.root or default_root(cfg)
    for nm in names:
        build(root, nm, Dims.of(cfg))


if __name__ == "__main__":
    main()
