"""A model's Pulsar-free decode on the AX650N card in the guest runner's plan
mode (``axcl_batch_runner``'s ``PLAN``: one plan per token, every tensor on
the device).

Both program sets stay resident (``first`` for position ``first_pos``,
``rest`` after it), bound to shared buffers: x0 (the embedding row), the RoPE
tables, the mask, and per layer a token-major K / V cache ``[S + 1, kvd]``.
Per step and layer: rk's output is rebound onto K row p (no copy), v's slice
of the merged qkv5 output is copied into V row p (one device copy),
attention reads the whole cache with the mask (rows 0..p valid). Position 0
is skipped when the model makes it zero end to end (llama-160m's BOS): its
K / V rows stay zero and valid. A program shared by both sets is loaded once.

--fake: ``pf_layout.FakeAX`` runs each program's float function instead of
the card (no rounding), to check the wiring against the float model offline.
--float-set: with --fake, use float stand-ins for every program
(``write_float_set``, under ``<out>/float_set``) instead of a generated
program set: needs neither Pulsar2 nor the card.

usage: MODEL=llama160m pf_plan.py [--fake [--float-set]] [--doc 0] [--prompt 48]
       -> <out>/plan_doc<d>[_fake].json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import onnx
from onnx.parser import parse_model

HERE = os.path.dirname(os.path.abspath(__file__))
AXERA = os.path.dirname(HERE)
for _p in (HERE, AXERA):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pf_layout as lay  # noqa: E402
import pf_model  # noqa: E402
import pf_templates as tpl  # noqa: E402
from pf_fake_quant import FusedEval  # noqa: E402

F32 = np.float32


def _parts(e, k):
    """The loaded programs of a manifest entry: a split linear is hi, lo and their Add."""
    return [f"{k}_{p}" for p in ("hi", "lo", "add")] if e["kind"] == "split" else [k]


class Plan:
    """Loads both program sets on ``ax`` (an ``AXSession`` or a ``FakeAX``),
    binds them once, and runs one ``PLAN`` per token (``step``)."""

    def __init__(self, ax, cfg: pf_model.Config, fake=False, root=None):
        self.ax, self.cfg, self.fake = ax, cfg, fake
        self.root = root or cfg.out
        self.g, self.h = ax.guest_dir, ax.host_dir
        self.lines: list[str] = []
        self.M = {}
        self.man = {
            st: json.load(open(os.path.join(self.root, f"manifest_{st}.json")))
            for st in ("first", "rest")
        }
        loaded = {}
        for st in ("first", "rest"):
            for wh in [f"L{li}" for li in range(cfg.L)] + ["post"]:
                for k in cfg.post_programs if wh == "post" else cfg.layer_programs:
                    e = self.man[st][wh][k]
                    src = e.get(
                        "shared", st
                    )  # a shared program: the rest set's serves both
                    for pk in _parts(e, k):
                        path = os.path.join(self.root, src, wh, f"{pk}.axmodel")
                        if path not in loaded:
                            if fake:
                                loaded[path] = ax.load(
                                    path, self._float_fn(wh, k, pk, e)
                                )
                            else:
                                loaded[path] = ax.load(open(path, "rb").read())
                        self.M[(st, wh, pk)] = loaded[path]
        self.n_programs = len(loaded)
        self.n_runs = sum(
            len(_parts(self.man["rest"][wh][k], k))
            for wh in [f"L{li}" for li in range(cfg.L)] + ["post"]
            for k in (cfg.post_programs if wh == "post" else cfg.layer_programs)
        )
        self.out_bufs = {}
        B = self.buf
        H, S, kvd = cfg.H, cfg.S, cfg.kvd
        self.x0, self.mask = B(H * 4), B((S + 1) * 4)
        # RoPE tables [cos_q | sin_q | cos_k | sin_k], one row per layer with
        # qk_norm (the layer's gains are folded in), else one row; one W per step
        self.tw = 2 * cfg.nh * cfg.hd + 2 * kvd
        self.trows = cfg.L if cfg.qk_norm else 1
        self.tab = B(self.trows * self.tw * 4)
        self.K = [B((S + 1) * kvd * 4) for _ in range(cfg.L)]
        self.V = [B((S + 1) * kvd * 4) for _ in range(cfg.L)]
        self.gain = []
        for li in range(cfg.L):
            g1, g2 = cfg.gains(li)
            b1, b2 = B(H * 4), B(H * 4)
            self.write(b1, g1, f"g{li}a")
            self.write(b2, g2, f"g{li}b")
            self.gain.append((b1, b2))
        self.gf = B(H * 4)
        self.write(self.gf, cfg.ln_f, "gf")
        self.io = {st: self.bind_set(st) for st in ("first", "rest")}
        self.run("setup")

    def _float_fn(self, wh, k, pk, e):
        """A program's float function keyed by its input names (fake mode)."""
        cfg = self.cfg
        li = None if wh == "post" else int(wh[1:])
        if pk.endswith("_add") or e["kind"] == "add":
            return lambda f: f["a"] + f["b"]
        if e["kind"] == "split":
            q, r = cfg.split_weights(cfg.merged_weight(k, li, gptq=False))
            w = q if pk.endswith("_hi") else r
            return lambda f: f["x"] @ w
        if e["kind"] == "linear":
            w = cfg.merged_weight(k, li)
            return lambda f: f["x"] @ w
        ev = FusedEval(os.path.join(self.root, e["model"]), {}, ())
        return lambda f: ev(**f)

    def buf(self, n):
        return int(self.ax._cmd(f"BUF {n}").split()[1])

    def bind(self, st, wh, k, io_, name, b, off=0, now=True):
        m = self.M[(st, wh, k)]
        specs = m.inputs if io_ == "I" else m.outputs
        idx = name if isinstance(name, int) else [s.name for s in specs].index(name)
        cmd = f"{m.id} {io_} {idx} {b} {off}"
        if now:
            r = self.ax._cmd(f"BIND {cmd}")
            if not r.startswith("OK"):
                raise RuntimeError(f"BIND {cmd} ({st} {wh} {k}): {r}")
        else:
            self.lines.append(f"B {cmd}")

    def _out(self, st, wh, key, prog, idx=0):
        # one output buffer per (layer, program, output) for both sets: a shared
        # program binds identically
        o = self.M[(st, wh, prog)].outputs[idx]
        if (wh, key, o.name) not in self.out_bufs:
            self.out_bufs[(wh, key, o.name)] = self.buf(o.nbytes)
        self.bind(st, wh, prog, "O", idx, self.out_bufs[(wh, key, o.name)])
        return self.out_bufs[(wh, key, o.name)]

    def outputs(self, st, wh, key, skip=()):
        """Bind every output of a program but ``skip`` (output names);
        ``{name: buffer}``."""
        m = self.M[(st, wh, key)]
        return {
            o.name: self._out(st, wh, key, key, i)
            for i, o in enumerate(m.outputs)
            if o.name not in skip
        }

    def chain(self, st, wh, key, **ins):
        """Bind a program's inputs (buffer or (buffer, byte offset)) and its
        output (a merged program with several outputs: ``outputs``)."""
        if self.man[st][wh].get(key, {}).get("kind") == "split":
            # hi and lo linears on the same input, then their Add
            hi = self.chain(st, wh, f"{key}_hi", **ins)
            lo = self.chain(st, wh, f"{key}_lo", **ins)
            self.bind(st, wh, f"{key}_add", "I", "a", hi)
            self.bind(st, wh, f"{key}_add", "I", "b", lo)
            return self._out(st, wh, key, f"{key}_add")
        for n, v in ins.items():
            b, off = v if isinstance(v, tuple) else (v, 0)
            self.bind(st, wh, key, "I", n, b, off)
        return self._out(st, wh, key, key)

    def tables(self, li):
        """(buffer, byte offset) of the layer's cos_q, sin_q, cos_k, sin_k rows."""
        cfg = self.cfg
        r = (li if cfg.qk_norm else 0) * self.tw * 4
        nq = cfg.nh * cfg.hd * 4
        t = self.tab
        return (t, r), (t, r + nq), (t, r + 2 * nq), (t, r + 2 * nq + cfg.kvd * 4)

    def write(self, b, arr, name, off=0):
        a = np.ascontiguousarray(arr, F32)
        a.tofile(os.path.join(self.h, "t", f"{name}.bin"))
        self.lines.append(f"W {b} {off} {a.nbytes} {self.g}/t/{name}.bin")

    def run(self, tag):
        path = os.path.join(self.h, "t", f"{tag}.txt")
        open(path, "w").write("\n".join(self.lines) + "\n")
        self.lines = []
        r = self.ax._cmd(f"PLAN {path}" if self.fake else f"PLAN {self.g}/t/{tag}.txt")
        if not r.startswith("OK"):
            raise RuntimeError(f"PLAN {tag}: {r}")
        return [int(v) for v in r.split()[1:]]

    def bind_set_fused(self, st):
        """``bind_set`` of the merged layout (``pf_layout.LAYER_FUSED``)."""
        cfg = self.cfg
        oq, og = lay.offsets(cfg.QKV), lay.offsets(cfg.GU)
        io = {}
        xp = dn = None
        for li in range(cfg.L):
            wh = f"L{li}"
            b1, b2 = self.gain[li]
            if li == 0:
                self.bind(st, wh, "n1", "I", "x", self.x0)
                self.bind(st, wh, "n1", "I", "g", b1)
                x, a = self.x0, self.outputs(st, wh, "n1")["y"]
            else:
                for n, v in dict(a=xp, b=dn, g=b1).items():
                    self.bind(st, wh, "n1", "I", n, v)
                o = self.outputs(st, wh, "n1")
                x, a = o["x"], o["y"]
            y5 = self.chain(st, wh, "qkv5", x=a)

            def o5(k, y5=y5):
                return (y5, oq[k] * 4)

            cq, sq, ck, sk = self.tables(li)
            ins = dict(q=o5("q_proj"), qr=o5("q_rot"), k=o5("k_proj"), kr=o5("k_rot"))
            ins |= dict(cq=cq, sq=sq, ck=ck, sk=sk)
            for n, v in ins.items():
                self.bind(st, wh, "qk", "I", n, *(v if isinstance(v, tuple) else (v,)))
            qr = self.outputs(st, wh, "qk", skip=("yk",))["yq"]  # yk: K row p, per step
            att = self.chain(
                st, wh, "attn", q=qr, k=self.K[li], v=self.V[li], mask=self.mask
            )
            op = self.chain(st, wh, "o_proj", x=att)
            for n, v in dict(a=x, b=op, g=b2).items():
                self.bind(st, wh, "n2", "I", n, v)
            o = self.outputs(st, wh, "n2")
            x1, m = o["x"], o["y"]
            gu = self.chain(st, wh, "gu", x=m)
            act = self.chain(
                st,
                wh,
                "silu_f",
                gate=(gu, og["gate_proj"] * 4),
                up=(gu, og["up_proj"] * 4),
            )
            dn = self.chain(st, wh, "down_proj", x=act)
            xp = x1
            io[li] = dict(y5=y5)
        for n, v in dict(a=xp, b=dn, g=self.gf).items():
            self.bind(st, "post", "nf", "I", n, v)
        xn = self.outputs(st, "post", "nf")["y"]
        io["logits"] = self.chain(st, "post", "lm_head", x=xn)
        return io

    def bind_set(self, st):
        cfg = self.cfg
        if cfg.fuse:
            return self.bind_set_fused(st)
        oq, og = lay.offsets(cfg.QKV), lay.offsets(cfg.GU)
        io, x = {}, self.x0
        for li in range(cfg.L):
            wh = f"L{li}"
            b1, b2 = self.gain[li]
            u = self.chain(st, wh, "n1_c", x=x)
            a = self.chain(st, wh, "n1_rms", x=u, g=b1)
            y5 = self.chain(st, wh, "qkv5", x=a)

            def o(k, y5=y5):
                return (y5, oq[k] * 4)

            cq, sq, ck, sk = self.tables(li)
            qr = self.chain(st, wh, "rq", q=o("q_proj"), r=o("q_rot"), c=cq, sn=sq)
            kq, kr = o("k_proj"), o("k_rot")
            if cfg.qk_norm:
                uk = self.chain(st, wh, "kc", x=o("k_proj"))  # [k | k_rot] is one slice
                kq, kr = (uk, 0), (uk, cfg.kvd * 4)
            for n, v in dict(q=kq, r=kr, c=ck, sn=sk).items():
                self.bind(st, wh, "rk", "I", n, *v)  # rk's output: bound per step
            att = self.chain(
                st, wh, "attn", q=qr, k=self.K[li], v=self.V[li], mask=self.mask
            )
            op = self.chain(st, wh, "o_proj", x=att)
            x1 = self.chain(st, wh, "res1", a=x, b=op)
            u2 = self.chain(st, wh, "n2_c", x=x1)
            m = self.chain(st, wh, "n2_rms", x=u2, g=b2)
            gu = self.chain(st, wh, "gu", x=m)
            act = self.chain(
                st,
                wh,
                "silu_f",
                gate=(gu, og["gate_proj"] * 4),
                up=(gu, og["up_proj"] * 4),
            )
            dn = self.chain(st, wh, "down_proj", x=act)
            x = self.chain(st, wh, "res2", a=x1, b=dn)
            io[li] = dict(y5=y5)
        u = self.chain(st, "post", "nf_c", x=x)
        xn = self.chain(st, "post", "nf_rms", x=u, g=self.gf)
        io["logits"] = self.chain(st, "post", "lm_head", x=xn)
        return io

    def step(self, token, p):
        """One token at position p: returns (logits [V], the runner's timings)."""
        cfg = self.cfg
        st = "first" if p == cfg.first_pos else "rest"
        kvd = cfg.kvd
        rows = []
        for li in range(self.trows):
            t = cfg.rope_tiles(p, li)
            rows.append(
                np.concatenate([t["cos_q"], t["sin_q"], t["cos_k"], t["sin_k"]], axis=1)
            )
        self.write(self.x0, cfg.model.embed([int(token)]).reshape(1, cfg.H), "x0")
        self.write(self.tab, np.concatenate(rows, axis=0), "tables")
        self.write(self.mask, cfg.mask_plan(p), "mask")

        def X(wh, k):
            for pk in _parts(self.man[st][wh][k], k):
                self.lines.append(f"X {self.M[(st, wh, pk)].id}")

        ov = lay.offsets(cfg.QKV)["v_proj"]
        kp, kout = cfg.k_program
        progs = cfg.layer_programs
        before_k, after_k = progs[: progs.index(kp)], progs[progs.index(kp) + 1 :]
        for li in range(cfg.L):
            wh = f"L{li}"
            for k in before_k:
                X(wh, k)
            # the K program's output: K row p
            self.bind(st, wh, kp, "O", kout, self.K[li], p * kvd * 4, now=False)
            X(wh, kp)
            y5 = self.io[st][li]["y5"]
            self.lines.append(f"C {self.V[li]} {p * kvd * 4} {y5} {ov * 4} {kvd * 4}")
            for k in after_k:
                X(wh, k)
        for k in cfg.post_programs:
            X("post", k)
        nb = self.M[(st, "post", "lm_head")].outputs[0].nbytes
        self.lines.append(f"R {self.io[st]['logits']} 0 {nb} {self.g}/t/logits.bin")
        tm = self.run("step")
        return np.fromfile(os.path.join(self.h, "t", "logits.bin"), F32), tm


# ---- float stand-ins of a program set (host-only wiring check) --------------------------
def _io_stub(k, n):
    """A linear's IO only ([1, k] -> [1, n]); FakeAX computes x @ W itself."""
    return parse_model(
        f'<ir_version: 9, opset_import: ["" : 17]> lin (float[1,{k}] x) => (float[1,{n}] y) '
        "{ y = Identity(x) }"
    )


def write_float_set(cfg: pf_model.Config, root: str) -> None:
    """Manifests and float stand-ins of both program sets under ``root``, laid
    out as ``pf_generate.py`` would (sharing and split linears included): fused
    programs as their float graphs (``pf_templates.float_graph``), linears and
    Adds as IO stubs. For ``Plan(..., fake=True, root=root)`` and
    ``pf_estimate.Chain(..., exact=True, root=root)``."""
    d = tpl.Dims.of(cfg)
    qd, kvd = cfg.nh * cfg.hd, cfg.kvd
    rope = "qkrope" if cfg.qk_norm else "rope"
    fused = {
        "n1": f"anorm{cfg.H}",
        "n2": f"anorm{cfg.H}",
        "nf": f"anorm{cfg.H}",
        "qk": f"qkf{qd}",
        "n1_c": f"crms{cfg.H}",
        "n2_c": f"crms{cfg.H}",
        "nf_c": f"crms{cfg.H}",
        "n1_rms": f"rms{cfg.H}",
        "n2_rms": f"rms{cfg.H}",
        "nf_rms": f"rms{cfg.H}",
        "rq": f"{rope}{qd}",
        "rk": f"{rope}{kvd}",
        "kc": f"hcrms{2 * kvd}",
        "attn": f"attn{qd}",
        "silu_f": f"silu{cfg.I}",
    }
    widths = {
        "qkv5": (cfg.H, sum(n for _, n in cfg.QKV)),
        "o_proj": (qd, cfg.H),
        "gu": (cfg.H, 2 * cfg.I),
        "down_proj": (cfg.I, cfg.H),
        "lm_head": (cfg.H, cfg.vocab),
    }
    man_rest: dict = {}
    for st in ("rest", "first"):  # rest first: the first set refers to it
        man = {}
        for wh in [f"L{li}" for li in range(cfg.L)] + ["post"]:
            li = None if wh == "post" else int(wh[1:])
            D = os.path.join(root, st, wh)
            os.makedirs(D, exist_ok=True)
            ml = {}
            for k in cfg.post_programs if wh == "post" else cfg.layer_programs:
                shared = ("rq", "rk", "kc", "qk")
                if st == "first" and cfg.first_pos == 0 and k in shared:
                    ml[k] = dict(man_rest[wh][k], shared="rest")
                elif k in fused:
                    path = os.path.join(D, f"{k}.axmodel")
                    name = f"nrm{cfg.H}" if (k, wh) == ("n1", "L0") else fused[k]
                    onnx.save(tpl.float_graph(name, d), path)
                    ml[k] = dict(
                        kind="fused", model=os.path.relpath(path, root), signed=[]
                    )
                elif k in ("res1", "res2"):
                    onnx.save(tpl.add_model(cfg.H), os.path.join(D, f"{k}.axmodel"))
                    ml[k] = dict(kind="add")
                elif cfg.is_split(st, li, k):
                    kk, n = widths[k]
                    onnx.save(_io_stub(kk, n), os.path.join(D, f"{k}_hi.axmodel"))
                    onnx.save(_io_stub(kk, n), os.path.join(D, f"{k}_lo.axmodel"))
                    onnx.save(tpl.add_model(n), os.path.join(D, f"{k}_add.axmodel"))
                    ml[k] = dict(kind="split")
                elif st == "first" and cfg.share_linears:
                    ml[k] = dict(man_rest[wh][k], shared="rest")
                else:
                    onnx.save(_io_stub(*widths[k]), os.path.join(D, f"{k}.axmodel"))
                    ml[k] = dict(kind="linear", shape=list(widths[k]))
            man[wh] = ml
        json.dump(man, open(os.path.join(root, f"manifest_{st}.json"), "w"), indent=1)
        if st == "rest":
            man_rest = man


def decode(plan: Plan, cfg: pf_model.Config, ids) -> list[dict]:
    """Teacher-forced steps over ``ids`` vs ``float_step``; one row per step."""
    fK, fV = cfg.new_cache()
    rows = []
    for p, tok in enumerate(ids):
        if p == 0 and cfg.zero_pos0:
            continue
        ts = time.time()
        lg, tm = plan.step(tok, p)
        wall = time.time() - ts
        flg, _ = cfg.float_step(fK, fV, tok, p)
        rows.append(
            dict(
                p=p,
                top1=int(lg.argmax()),
                float_top1=int(np.argmax(flg)),
                wall_s=wall,
                exec_us=tm[0] if tm else None,
                plan_us=tm[1] if len(tm) > 1 else None,
                rel=lay.rel_err(lg, flg),
            )
        )
        r = rows[-1]
        print(
            f"p={p} top1 {r['top1']} float {r['float_top1']} rel {r['rel']:.2e} "
            f"{wall * 1e3:.0f} ms",
            flush=True,
        )
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--float-set", action="store_true")
    ap.add_argument("--doc", type=int, default=0)
    ap.add_argument("--prompt", type=int, default=48)
    a = ap.parse_args()
    cfg = pf_model.load()
    ids = cfg.passage_ids(lay.EVAL_DOCS[a.doc])[: a.prompt]
    root, sess, A = None, None, None
    if a.float_set:
        a.fake = True
        root = os.path.join(cfg.out, "float_set")
        write_float_set(cfg, root)
    if a.fake:
        ax = lay.FakeAX(os.path.join(cfg.out, "plan_fake"))
    else:
        import axcl_session as A

        sess = A.AXSession().__enter__()
        print("health before", A.health_check(sess), flush=True)
        ax = sess
    t0 = time.time()
    pl = Plan(ax, cfg, a.fake, root)
    t_setup = time.time() - t0
    print(
        f"{pl.n_programs} programs loaded, {pl.n_runs} per token, setup {t_setup:.1f}s",
        flush=True,
    )
    rows = decode(pl, cfg, ids)
    res = dict(
        model=cfg.name,
        fake=a.fake,
        doc=lay.EVAL_DOCS[a.doc],
        programs=pl.n_programs,
        runs_per_token=pl.n_runs,
        fuse=cfg.fuse,
    )
    if sess is not None:
        res["health_after"] = A.health_check(sess)
        print("health after", res["health_after"], flush=True)
        sess.__exit__(None, None, None)
    res.update(lay.summarize(rows), setup_s=t_setup)
    res["s_per_token"] = float(np.mean([r["wall_s"] for r in rows]))
    res["steps"] = rows
    json.dump(
        res,
        open(
            os.path.join(cfg.out, f"plan_doc{a.doc}{'_fake' if a.fake else ''}.json"),
            "w",
        ),
        indent=1,
    )
    print({k: v for k, v in res.items() if k != "steps"})


if __name__ == "__main__":
    main()
