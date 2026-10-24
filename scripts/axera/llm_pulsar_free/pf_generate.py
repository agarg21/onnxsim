"""Emit a model's Pulsar-free U16 program set (one calibration set at a time)
by retargeting the native templates of ``pf_templates.py`` -- no Pulsar2 run.

Per layer (``pf_layout.LAYER``; ``kc`` only with a per-head k norm):

  n1_c    fused coarse norm  x -> u = x / sqrt(mean(x^2) + eps)       crms<H>
  n1_rms  fused RMSNorm      u, g -> a  (second pass of the two-pass norm)
  qkv5    merged linear      a -> the ``QKV`` parts                   k<H>n<..>
  rq, rk  fused RoPE apply   q, r, c, sn -> y      rope<n> (qkrope<n> with q / k norm)
  kc      per-head coarse k norm of [k | k_rot] (Qwen3)              hcrms<2 kvd>
  attn    the whole layer's attention over the token-major K / V cache
  o_proj  linear, res1 Add, n2_c, n2_rms, gu merged linear (gate | up),
          silu_f fused SiLU, down_proj linear, res2 Add
  post:   nf_c, nf_rms, lm_head

Calibration (Pulsar-free, from ``pf_record.py``'s activations): fused /
single-op programs from their samples (real + x2 copies for norms and
residual Adds, x sqrt(2) copies of the activations for products and
attention; tables and the mask as they are), every tensor's U16 MinMax
(MatMul operands signed; ``u16_retarget.calibrate_graph``); the second norm
pass sees the card's u (the coarse program in fake-quant); linears at the real
range x2 (merged: the union of the parts). Each program is retargeted from
the first template variant that accepts the target
(``u16_retarget.retarget_graph`` / ``retarget``, ``linear_emit.emit_linear_u16``).

With ``share_linears`` (Qwen3) the ``rest`` set's linear serves both sets
when the first set's ranges are within ``SHARE_MAX`` (calibrated on the
union); run ``rest`` before ``first``. With ``first_pos == 0`` the first set
also shares the rest's ``rq`` / ``rk`` / ``kc`` (sin = 0 at position 0
degenerates their own calibration), which are then calibrated on both sets.

Writes ``<out>/<set>/L<l>/<program>.axmodel``, ``<out>/cal/<set>/...`` (the
programs' float graphs and calibration samples) and
``<out>/manifest_<set>.json``.

usage: MODEL=llama160m pf_generate.py SET [first_layer last_layer]   (SET: first | rest)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import onnx

HERE = os.path.dirname(os.path.abspath(__file__))
AXERA = os.path.dirname(HERE)
for _p in (HERE, AXERA):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import linear_emit as le  # noqa: E402
import pf_model  # noqa: E402
import pf_templates as tpl  # noqa: E402
import u16_retarget as ur  # noqa: E402
from pf_fake_quant import FusedEval  # noqa: E402

F32 = np.float32
K2 = F32(np.sqrt(2.0))


def signed_inputs(model) -> set:
    """Tensors Pulsar2 quantizes signed: both operands of every MatMul."""
    m = model if isinstance(model, onnx.ModelProto) else onnx.load(model)
    return {t for n in m.graph.node if n.op_type == "MatMul" for t in n.input}


def rng2(*arrs):
    """U16 MinMax of the arrays' range x2 (folded with 0)."""
    lo = min(float(np.min(a)) for a in arrs)
    hi = max(float(np.max(a)) for a in arrs)
    return ur.unsigned_minmax(2 * min(lo, 0), 2 * max(hi, 0))


class Generator:
    def __init__(self, cfg: pf_model.Config, root: str):
        self.cfg, self.root, self.out = cfg, root, cfg.out

    def _rel(self, p, base=None):
        return os.path.relpath(p, base or self.out)

    def _acts(self, st, where):
        return np.load(os.path.join(self.out, "acts", st, f"{where}.npz"))

    def _manifest(self, st):
        return json.load(open(os.path.join(self.out, f"manifest_{st}.json")))

    @staticmethod
    def _tar(d, name, samples):
        import pulsar2_docker as P

        os.makedirs(os.path.join(d, "dataset"), exist_ok=True)
        P.make_numpy_calibration_tar(
            os.path.join(d, "dataset", f"{name}.tar"),
            [np.asarray(a, F32) for a in samples],
        )

    # ---- one program ------------------------------------------------------------------
    def fused(self, d, dst, name, data):
        os.makedirs(d, exist_ok=True)
        tdir0 = tpl.template_dir(self.root, name)
        shutil.copy(
            os.path.join(tdir0, "v0", "model.onnx"), os.path.join(d, "model.onnx")
        )
        for n, v in data.items():
            self._tar(d, n, v)
        model = os.path.join(d, "model.onnx")
        q = ur.propagate(model, ur.calibrate_graph(d))
        refused = []
        tdirs = [
            os.path.join(tpl.template_dir(self.root, nm), f"v{i}")
            for nm in (name, name + "t")
            for i in range(4)
        ]
        for tdir in tdirs:
            axm = os.path.join(tdir, "out", "compiled.axmodel")
            if not os.path.exists(axm):
                continue
            try:
                m = ur.retarget_graph(axm, model, ur.quant_of(tdir), q)
            except ValueError as e:
                refused.append(f"{self._rel(tdir, self.root)}: {str(e)[:90]}")
                continue
            open(dst, "wb").write(m.SerializeToString())
            return dict(
                kind="fused",
                op=name,
                template=self._rel(tdir, self.root),
                model=self._rel(model),
                calibration=q,
                signed=sorted(signed_inputs(model)),
                bytes=os.path.getsize(dst),
                refused=refused,
            )
        raise RuntimeError(f"{dst}: no usable template: {refused}")

    def add(self, d, dst, a, b, n=None):
        """A residual Add (single-op) from the ``add<n>`` templates."""
        n = n or self.cfg.H
        os.makedirs(d, exist_ok=True)
        onnx.save(tpl.add_model(n), os.path.join(d, "model.onnx"))
        self._tar(d, "a", list(a) + list(a * F32(2)))
        self._tar(d, "b", list(b) + list(b * F32(2)))
        q = ur.calibrate(d, "Add")
        refused = []
        for i in range(4):
            tdir = os.path.join(tpl.template_dir(self.root, f"add{n}"), f"v{i}")
            try:
                m = ur.retarget(
                    os.path.join(tdir, "out", "compiled.axmodel"),
                    "Add",
                    ur.quant_of(tdir),
                    q,
                    ("a", "b"),
                )
            except (ValueError, OSError) as e:
                refused.append(f"v{i}: {str(e)[:90]}")
                continue
            open(dst, "wb").write(m.SerializeToString())
            return dict(
                kind="add",
                op="Add",
                template=self._rel(tdir, self.root),
                calibration=q,
                bytes=os.path.getsize(dst),
                refused=refused,
            )
        raise RuntimeError(f"{dst}: no usable template: {refused}")

    def linear(self, dst, w, x, ys):
        k, n = w.shape
        tdir = tpl.template_dir(self.root, f"k{k}n{n}")
        sx, sy = rng2(x), rng2(*ys)
        m = le.emit_linear_u16(
            os.path.join(tdir, "out", "compiled.axmodel"),
            w,
            {"x": sx[0], "y": sy[0]},
            {"x": sx[1], "y": sy[1]},
        )
        open(dst, "wb").write(m.SerializeToString())
        return dict(
            kind="linear",
            op="Linear",
            template=self._rel(tdir, self.root),
            calibration={"x": list(sx), "y": list(sy)},
            shape=[k, n],
            bytes=os.path.getsize(dst),
        )

    def split(self, st, li, key, dst, xk, yks):
        """A first-set linear with ~16-bit weights: hi = s8(w), lo = s8(w - Q(w)),
        y = Add(hi, lo). The split carries the precision itself: original weights."""
        where = f"L{li}"
        A = self._acts(st, where)
        x = A[xk].reshape(len(A[xk]), -1).astype(F32)
        w = self.cfg.merged_weight(key, li, gptq=False)
        q, r = self.cfg.split_weights(w)
        C = os.path.join(self.out, "cal", st, where)
        ys = [A[k].reshape(len(A[k]), -1) for k in yks]
        hi = self.linear(dst.replace(".axmodel", "_hi.axmodel"), w, x, ys)
        lo = self.linear(dst.replace(".axmodel", "_lo.axmodel"), r, x, [x @ r])
        ad = self.add(
            os.path.join(C, f"{key}_add"),
            dst.replace(".axmodel", "_add.axmodel"),
            x @ q,
            x @ r,
            w.shape[1],
        )
        return dict(
            kind="split",
            op="SplitLinear",
            split=dict(hi=hi, lo=lo, add=ad),
            bytes=hi["bytes"] + lo["bytes"] + ad["bytes"],
        )

    def lin(self, st, li, key, dst, w, xk, yks):
        """A linear program; with ``share_linears`` the rest set's program serves
        both sets (calibrated on the union) and the first set's manifest points at it."""
        cfg = self.cfg
        where = "post" if li is None else f"L{li}"
        if cfg.is_split(st, li, key):
            return self.split(st, li, key, dst, xk, yks)
        if cfg.share_linears and st == "first":
            e = self._manifest("rest")[where][key]
            if e.get("union_of"):
                return dict(e, shared="rest")
        sets = (st,)
        if cfg.share_linears and st == "rest":
            R, Fs = self._acts("rest", where), self._acts("first", where)
            ratio = max(
                float(np.abs(Fs[k]).max() / max(np.abs(R[k]).max(), 1e-30))
                for k in [xk] + yks
            )
            if ratio <= pf_model.SHARE_MAX:
                sets = ("rest", "first")
        acts = [self._acts(s_, where) for s_ in sets]
        x = np.concatenate([a[xk].reshape(len(a[xk]), -1) for a in acts])
        ys = [np.concatenate([a[k].reshape(len(a[k]), -1) for a in acts]) for k in yks]
        e = self.linear(dst, w, x, ys)
        if len(sets) > 1:
            e["union_of"] = list(sets)
        return e

    def two_pass(self, C, D, p, x, g, ml):
        """n_c then n_rms; the second pass is calibrated on the card's u (n_c in fake-quant)."""
        H = self.cfg.H
        ml[f"{p}_c"] = e = self.fused(
            f"{C}/{p}_c",
            f"{D}/{p}_c.axmodel",
            f"crms{H}",
            {"x": list(x) + list(x * F32(2))},
        )
        ev = FusedEval(
            os.path.join(self.out, e["model"]), e["calibration"], e["signed"]
        )
        u = np.stack([ev(x=xi) for xi in x])
        ml[f"{p}_rms"] = self.fused(
            f"{C}/{p}_rms",
            f"{D}/{p}_rms.axmodel",
            f"rms{H}",
            {"x": list(u) + list(u * F32(2)), "g": [g] * (2 * len(u))},
        )
        ml[f"{p}_rms"]["u_absmax"] = float(np.abs(u).max())

    # ---- one layer / the post programs --------------------------------------------------
    def layer(self, st, li, man):
        cfg = self.cfg
        nh, nkv, hd, kvd = cfg.nh, cfg.nkv, cfg.hd, cfg.kvd
        w = f"L{li}"
        A = self._acts(st, w)
        D = os.path.join(self.out, st, w)
        C = os.path.join(self.out, "cal", st, w)
        os.makedirs(D, exist_ok=True)
        g1, g2 = cfg.gains(li)
        shared_first = st == "first" and cfg.first_pos == 0
        ml = {}
        x = A["x"].astype(F32)
        self.two_pass(C, D, "n1", x, g1, ml)
        wq = cfg.merged_weight("qkv5", li)
        ml["qkv5"] = self.lin(
            st, li, "qkv5", f"{D}/qkv5.axmodel", wq, "a", ["qp", "kp", "vp"]
        )
        offs = pf_model.lay.offsets(cfg.QKV)
        ml["qkv5"]["parts"] = {k: dict(col=offs[k], n=n) for k, n in cfg.QKV}
        # with qk_norm the layer's q / k gains are folded into the tables
        tiles = [cfg.rope_tiles(int(p), li) for p in A["p"]]
        rope_t = "qkrope" if cfg.qk_norm else "rope"
        if (
            cfg.qk_norm
        ):  # k's per-head norm needs the two-pass form: kc gives u = k / r0_head
            kk = np.concatenate(
                [A["kp"].astype(F32), cfg.rot(A["kp"].astype(F32), nkv)], axis=-1
            )
            if shared_first:
                # shared with the rest set (calibrated on both)
                e = ml["kc"] = dict(self._manifest("rest")[w]["kc"], shared="rest")
            else:
                cal = kk
                if cfg.first_pos == 0:
                    Fk = self._acts("first", w)["kp"].astype(F32)
                    cal = np.concatenate(
                        [kk, np.concatenate([Fk, cfg.rot(Fk, nkv)], axis=-1)]
                    )
                ml["kc"] = e = self.fused(
                    f"{C}/kc",
                    f"{D}/kc.axmodel",
                    f"hcrms{2 * kvd}",
                    {"x": list(cal) + list(cal * F32(2))},
                )
            ev = FusedEval(
                os.path.join(self.out, e["model"]), e["calibration"], e["signed"]
            )
            uk = np.stack([ev(x=xi) for xi in kk])
            ml["kc"]["u_absmax"] = float(np.abs(uk).max())
        for key, src, heads, c, sn in (
            ("rq", "qp", nh, "cos_q", "sin_q"),
            ("rk", "kp", nkv, "cos_k", "sin_k"),
        ):
            q = A[src].astype(F32)
            r = cfg.rot(q, heads)
            if cfg.qk_norm and key == "rk":
                q, r = uk[..., :kvd], uk[..., kvd:]
            if shared_first:
                # sin = 0 at position 0: r * sn has a zero range, the calibration
                # degenerates; the rest set's program (calibrated on both) covers it
                ml[key] = dict(self._manifest("rest")[w][key], shared="rest")
                continue
            tl = [t[c] for t in tiles], [t[sn] for t in tiles]
            if st == "rest" and cfg.first_pos == 0:
                # position 0 has cos = 1 on every dimension: its output reaches past the
                # rest positions' range (Qwen3 rk: 30.7 vs 21.6 at L18 -> K row 0, the
                # attention sink, clipped). Calibrate on both sets.
                qf = self._acts("first", w)[src].astype(F32)
                rf = cfg.rot(qf, heads)
                if cfg.qk_norm and key == "rk":
                    ukf = np.stack(
                        [ev(x=xi) for xi in np.concatenate([qf, rf], axis=-1)]
                    )
                    qf, rf = ukf[..., :kvd], ukf[..., kvd:]
                t0 = cfg.rope_tiles(0, li)
                q, r = np.concatenate([q, qf]), np.concatenate([r, rf])
                tl = (tl[0] + [t0[c]] * len(qf), tl[1] + [t0[sn]] * len(qf))
            ml[key] = self.fused(
                f"{C}/{key}",
                f"{D}/{key}.axmodel",
                f"{rope_t}{heads * hd}",
                {
                    "q": list(q) + list(q * K2),
                    "r": list(r) + list(r * K2),
                    "c": tl[0] * 2,
                    "sn": tl[1] * 2,
                },
            )
        q, K, V = A["qr"].astype(F32), A["K"].astype(F32), A["V"].astype(F32)
        mask = [cfg.mask_plan(int(p)) for p in A["p_attn"]]
        ml["attn"] = self.fused(
            f"{C}/attn",
            f"{D}/attn.axmodel",
            f"attn{nh * hd}",
            {
                "q": list(q) + list(q * K2),
                "k": list(K) + list(K * K2),
                "v": list(V) + list(V * K2),
                "mask": mask * 2,
            },
        )
        ml["o_proj"] = self.lin(
            st,
            li,
            "o_proj",
            f"{D}/o_proj.axmodel",
            cfg.merged_weight("o_proj", li),
            "att",
            ["op"],
        )
        ml["res1"] = self.add(f"{C}/res1", f"{D}/res1.axmodel", x, A["op"].astype(F32))
        x1 = A["x1"].astype(F32)
        self.two_pass(C, D, "n2", x1, g2, ml)
        wg = cfg.merged_weight("gu", li)
        ml["gu"] = self.lin(st, li, "gu", f"{D}/gu.axmodel", wg, "m", ["gate", "up"])
        og = pf_model.lay.offsets(cfg.GU)
        ml["gu"]["parts"] = {k: dict(col=og[k], n=n) for k, n in cfg.GU}
        gate, up = A["gate"].astype(F32), A["up"].astype(F32)
        ml["silu_f"] = self.fused(
            f"{C}/silu_f",
            f"{D}/silu_f.axmodel",
            f"silu{cfg.I}",
            {"gate": list(gate) + list(gate * K2), "up": list(up) + list(up * K2)},
        )
        ml["down_proj"] = self.lin(
            st,
            li,
            "down_proj",
            f"{D}/down_proj.axmodel",
            cfg.merged_weight("down_proj", li),
            "act",
            ["down"],
        )
        ml["res2"] = self.add(
            f"{C}/res2", f"{D}/res2.axmodel", x1, A["down"].astype(F32)
        )
        man[w] = ml

    def post(self, st, man):
        cfg = self.cfg
        A = self._acts(st, "post")
        D = os.path.join(self.out, st, "post")
        C = os.path.join(self.out, "cal", st, "post")
        os.makedirs(D, exist_ok=True)
        ml = {}
        self.two_pass(C, D, "nf", A["x"].astype(F32), cfg.ln_f, ml)
        ml["lm_head"] = self.lin(
            st,
            None,
            "lm_head",
            f"{D}/lm_head.axmodel",
            cfg.merged_weight("lm_head"),
            "xn",
            ["logits"],
        )
        man["post"] = ml


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("set", choices=("first", "rest"))
    ap.add_argument("layers", nargs="*", type=int, help="first_layer last_layer")
    ap.add_argument("--templates", default=None, help="template root (pf_templates.py)")
    a = ap.parse_args()
    cfg = pf_model.load()
    gen = Generator(cfg, a.templates or tpl.default_root(cfg))
    lo_l, hi_l = (a.layers[0], a.layers[1]) if len(a.layers) == 2 else (0, cfg.L - 1)
    st = a.set
    mpath = os.path.join(cfg.out, f"manifest_{st}.json")
    man = json.load(open(mpath)) if os.path.exists(mpath) else {}
    t0 = time.time()
    for li in range(lo_l, hi_l + 1):
        gen.layer(st, li, man)
        json.dump(man, open(mpath, "w"), indent=1)
        ml = man[f"L{li}"]
        tnames = sorted(
            {
                os.path.basename(os.path.dirname(e.get("template", "split/x")))
                for e in ml.values()
            }
        )
        print(
            f"{st} L{li}: {len(ml)} programs, templates {tnames}, u max "
            f"{ml['n1_rms']['u_absmax']:.1f}/{ml['n2_rms']['u_absmax']:.1f}, "
            f"{time.time() - t0:.0f}s",
            flush=True,
        )
    if hi_l == cfg.L - 1:
        gen.post(st, man)
        json.dump(man, open(mpath, "w"), indent=1)
        print(f"{st} post done, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
