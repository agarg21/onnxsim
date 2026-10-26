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
        return self._retarget_fused(dst, name, model, q)

    def _retarget_fused(self, dst, name, model, q):
        """The program of fused graph ``model`` at calibration ``q``, from the
        first template variant (``<name>/v*``, then ``<name>t/v*``) that accepts it."""
        refused = []
        tdirs = []
        for nm in (name, name + "t"):
            base = tpl.template_dir(self.root, nm)
            if os.path.isdir(base):
                vs = sorted(
                    v for v in os.listdir(base) if v[:1] == "v" and v[1:].isdigit()
                )
                tdirs += [
                    os.path.join(base, v) for v in sorted(vs, key=lambda v: int(v[1:]))
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

    # ---- merged programs (``fuse=1``) ------------------------------------------------------
    def _stage(self, body, ins, outs, feeds, inits):
        """Ranges ``{tensor: [lo, hi]}`` of a part of a merged graph (its text
        ``body``) over ``feeds`` (input -> samples), and the part's model."""
        m = tpl.model_of(body, ins, outs, inits)
        ev = FusedEval(m, {})
        n = len(next(iter(feeds.values())))
        rg: dict = {}
        for j in range(n):
            for k, a in ev.trace(
                **{t: np.asarray(v[j], F32) for t, v in feeds.items()}
            ).items():
                if k in ev.consts or np.asarray(a).dtype.kind != "f":
                    continue
                lo, hi = float(np.min(a)), float(np.max(a))
                r = rg.setdefault(k, [lo, hi])
                r[0], r[1] = min(r[0], lo), max(r[1], hi)
        return rg, m

    @staticmethod
    def _union(*rgs):
        out: dict = {}
        for rg in rgs:
            for k, (lo, hi) in rg.items():
                r = out.setdefault(k, [lo, hi])
                r[0], r[1] = min(r[0], lo), max(r[1], hi)
        return out

    @staticmethod
    def _calib(rg, model):
        """U16 MinMax of every range, and of the graph's float constants."""
        q = {k: ur.unsigned_minmax(lo, hi) for k, (lo, hi) in rg.items()}
        for i in model.graph.initializer:
            a = onnx.numpy_helper.to_array(i)
            if a.dtype.kind == "f":
                q[i.name] = ur.unsigned_minmax(a.min(), a.max())
        return ur.propagate(model, q)

    def merged(self, C, D, key, name, rgs, ml, extra=None, tie=()):
        """Retarget merged program ``name`` at the union of its parts' ranges.
        Each pair in ``tie`` gets one calibration (the union of both ranges):
        a Div by an RMS of already normalized values (``d = u / r``, r ~ 1)
        has its output's zero point equal to its input's in every template,
        and a record holding the one zero point cannot be told apart, so the
        target has to keep them equal too."""
        d = os.path.join(C, key)
        os.makedirs(d, exist_ok=True)
        model = os.path.join(d, "model.onnx")
        fm = tpl.float_graph(name, tpl.Dims.of(self.cfg))
        onnx.save(fm, model)
        rg = self._union(*rgs)
        for a, b in tie:
            rg[a] = rg[b] = self._union({a: rg[a]}, {a: rg[b]})[a]
        q = self._calib(rg, fm)
        e = self._retarget_fused(os.path.join(D, f"{key}.axmodel"), name, model, q)
        e.update(extra or {})
        ml[key] = e
        return e

    def norm2(self, C, D, key, ml, g, x=None, a=None, b=None):
        """The two-pass RMSNorm as one program (``nrm<H>``), or with the
        residual Add before it (``anorm<H>``: ``x = a + b``, outputs ``x`` and
        ``y``). Each part is calibrated as its own program was: the Add on
        ``a`` / ``b`` + x2 copies, the coarse pass on ``x`` + x2, the second
        pass on the coarse pass's ``u`` in fake-quant + x2."""
        H = self.cfg.H
        inits = tpl.graph(f"nrm{H}", tpl.Dims.of(self.cfg))[4]
        rgs = []
        if a is not None:
            rg, _ = self._stage(
                "  x = Add(a, b)\n",
                {"a": [1, H], "b": [1, H]},
                {"x": [1, H]},
                {"a": list(a) + list(a * F32(2)), "b": list(b) + list(b * F32(2))},
                {},
            )
            rgs.append(rg)
            x = (a + b).astype(F32)
        rg, mc = self._stage(
            tpl.NORM2_C,
            {"x": [1, H]},
            {"u": [1, H]},
            {"x": list(x) + list(x * F32(2))},
            inits,
        )
        rgs.append(rg)
        ev = FusedEval(mc, self._calib(rg, mc))
        u = np.stack([ev(x=xi) for xi in x])
        rg, _ = self._stage(
            tpl.NORM2_R,
            {"u": [1, H], "g": [1, H]},
            {"y": [1, H]},
            {"u": list(u) + list(u * F32(2)), "g": [g] * (2 * len(u))},
            inits,
        )
        rgs.append(rg)
        name = f"anorm{H}" if a is not None else f"nrm{H}"
        return self.merged(
            C,
            D,
            key,
            name,
            rgs,
            ml,
            dict(u_absmax=float(np.abs(u).max())),
            [("u", "d")],
        )

    def qk(self, st, li, C, D, A, ml):
        """``rq`` + ``kc`` + ``rk`` as one program (``qkf<nh*hd>``), each part
        calibrated as its own program was (``layer``)."""
        cfg = self.cfg
        nh, nkv, hd, kvd = cfg.nh, cfg.nkv, cfg.hd, cfg.kvd
        w = f"L{li}"
        if st == "first" and cfg.first_pos == 0:
            ml["qk"] = dict(self._manifest("rest")[w]["qk"], shared="rest")
            return
        name = f"qkf{nh * hd}"
        inits = tpl.graph(name, tpl.Dims.of(cfg))[4]
        both = st == "rest" and cfg.first_pos == 0
        F_ = self._acts("first", w) if both else None
        tiles = [cfg.rope_tiles(int(p), li) for p in A["p"]]
        t0 = cfg.rope_tiles(0, li)

        def tab(c, n):
            return [t[c] for t in tiles] + ([t0[c]] * n if both else [])

        nq, nk = nh * hd, kvd
        q = A["qp"].astype(F32)
        if both:
            q = np.concatenate([q, F_["qp"].astype(F32)])
        r = cfg.rot(q, nh)
        nf = len(F_["qp"]) if both else 0
        rq, _ = self._stage(
            tpl.QKF_Q,
            {"q": [1, nq], "qr": [1, nq], "cq": [1, nq], "sq": [1, nq]},
            {"yq": [1, nq]},
            {
                "q": list(q) + list(q * K2),
                "qr": list(r) + list(r * K2),
                "cq": tab("cos_q", nf) * 2,
                "sq": tab("sin_q", nf) * 2,
            },
            inits,
        )
        k = A["kp"].astype(F32)
        if both:
            k = np.concatenate([k, F_["kp"].astype(F32)])
        kr = cfg.rot(k, nkv)
        rkc, mkc = self._stage(
            tpl.QKF_KC,
            {"k": [1, nk], "kr": [1, nk]},
            {"uk": [1, nk], "ur": [1, nk]},
            {"k": list(k) + list(k * F32(2)), "kr": list(kr) + list(kr * F32(2))},
            inits,
        )
        ev = FusedEval(mkc, self._calib(rkc, mkc))
        uu = [ev(k=a, kr=b) for a, b in zip(k, kr)]
        uk = np.stack([x[0] for x in uu])
        urr = np.stack([x[1] for x in uu])
        rk, _ = self._stage(
            tpl.QKF_K,
            {"uk": [1, nk], "ur": [1, nk], "ck": [1, nk], "sk": [1, nk]},
            {"yk": [1, nk]},
            {
                "uk": list(uk) + list(uk * K2),
                "ur": list(urr) + list(urr * K2),
                "ck": tab("cos_k", nf) * 2,
                "sk": tab("sin_k", nf) * 2,
            },
            inits,
        )
        self.merged(
            C,
            D,
            "qk",
            name,
            [rq, rkc, rk],
            ml,
            dict(u_absmax=float(np.abs(uk).max())),
            [("kt", "ykh")],
        )

    def layer_fused(self, st, li, man):
        """One layer of the merged layout (``pf_layout.LAYER_FUSED``)."""
        cfg = self.cfg
        w = f"L{li}"
        A = self._acts(st, w)
        D = os.path.join(self.out, st, w)
        C = os.path.join(self.out, "cal", st, w)
        os.makedirs(D, exist_ok=True)
        g1, g2 = cfg.gains(li)
        ml = {}
        x = A["x"].astype(F32)
        if li == 0:
            self.norm2(C, D, "n1", ml, g1, x=x)
        else:
            P = self._acts(st, f"L{li - 1}")
            self.norm2(
                C, D, "n1", ml, g1, a=P["x1"].astype(F32), b=P["down"].astype(F32)
            )
        wq = cfg.merged_weight("qkv5", li)
        ml["qkv5"] = self.lin(
            st, li, "qkv5", f"{D}/qkv5.axmodel", wq, "a", ["qp", "kp", "vp"]
        )
        offs = pf_model.lay.offsets(cfg.QKV)
        ml["qkv5"]["parts"] = {k: dict(col=offs[k], n=n) for k, n in cfg.QKV}
        if not cfg.qk_norm:
            raise NotImplementedError("fuse=1 without a per-head q / k norm")
        self.qk(st, li, C, D, A, ml)
        self._attn_mlp(st, li, A, C, D, ml, fused=True)
        man[w] = ml

    def post_fused(self, st, man):
        cfg = self.cfg
        P = self._acts(st, f"L{cfg.L - 1}")
        D = os.path.join(self.out, st, "post")
        C = os.path.join(self.out, "cal", st, "post")
        os.makedirs(D, exist_ok=True)
        ml = {}
        self.norm2(
            C, D, "nf", ml, cfg.ln_f, a=P["x1"].astype(F32), b=P["down"].astype(F32)
        )
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
        self._attn_mlp(st, li, A, C, D, ml)
        man[w] = ml

    def _attn_mlp(self, st, li, A, C, D, ml, fused=False):
        """attn, o_proj, then (``fused``) n2 = res1 + norm, or res1, n2_c,
        n2_rms; gu, silu_f, down_proj, and res2 unless ``fused`` (the next
        layer's ``n1`` holds it)."""
        cfg = self.cfg
        nh, hd = cfg.nh, cfg.hd
        x = A["x"].astype(F32)
        _, g2 = cfg.gains(li)
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
        x1 = A["x1"].astype(F32)
        if fused:
            self.norm2(C, D, "n2", ml, g2, a=x, b=A["op"].astype(F32))
        else:
            ml["res1"] = self.add(
                f"{C}/res1", f"{D}/res1.axmodel", x, A["op"].astype(F32)
            )
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
        if not fused:
            ml["res2"] = self.add(
                f"{C}/res2", f"{D}/res2.axmodel", x1, A["down"].astype(F32)
            )

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
        (gen.layer_fused if cfg.fuse else gen.layer)(st, li, man)
        json.dump(man, open(mpath, "w"), indent=1)
        ml = man[f"L{li}"]
        tnames = sorted(
            {
                os.path.basename(os.path.dirname(e.get("template", "split/x")))
                for e in ml.values()
            }
        )
        n1, n2 = ("n1", "n2") if cfg.fuse else ("n1_rms", "n2_rms")
        print(
            f"{st} L{li}: {len(ml)} programs, templates {tnames}, u max "
            f"{ml[n1]['u_absmax']:.1f}/{ml[n2]['u_absmax']:.1f}, "
            f"{time.time() - t0:.0f}s",
            flush=True,
        )
    if hi_l == cfg.L - 1:
        (gen.post_fused if cfg.fuse else gen.post)(st, man)
        json.dump(man, open(mpath, "w"), indent=1)
        print(f"{st} post done, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
