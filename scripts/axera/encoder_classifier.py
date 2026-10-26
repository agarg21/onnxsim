#!/usr/bin/env python3
"""BERT-style encoder classifiers (cross-encoders / rerankers, NLI / zero-shot,
sequence classifiers) on the AX650N: export -> onnxsim -> Pulsar2 -> card.

    python scripts/axera/encoder_classifier.py \\
        cross-encoder/ms-marco-MiniLM-L-6-v2 --seq 128 --precision U8 U16 \\
        --work /path/to/work --json results.json

For each model:

1. Export with PyTorch's legacy (TorchScript) exporter at a static [1, seq]
   shape, opset 17, with the model's default attention (SDPA in current
   transformers, so the `Where(IsNaN(P), 0, P)` guard and a -inf padding mask
   are in the graph).
2. `pulsar2_backend.simplify()`: no attention fusion, and the SDPA
   `IsNaN` guard removed exactly (see that function and
   docs/axera-encoder-classifiers.md).
3. `pulsar2 build` (Docker), Numpy calibration on real tokenized text pairs
   padded to `seq`, MinMax, at each requested precision (`U8` = Pulsar2's
   default int8; `U16` = every op type at 16-bit activations).
4. Run on the card through `axcl_session.AXSession` (health check before and
   after), against onnxruntime on the simplified fp32 graph: logit relative
   error, argmax agreement, and per-query rerank order (Kendall tau and top-1).

Needs torch + transformers (export), a loaded Pulsar2 image, and the AXCL
LXD VM for the card; `--no-device` stops after the builds. Nothing here runs
in CI.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

IMAGE = "pulsar2:7.0-lite"

# (query, passage, relevant) -- the evaluation set: a few queries, each with
# relevant and irrelevant passages, so rerank order can be checked per query.
EVAL_PAIRS = [
    (
        "how many people live in berlin",
        "Berlin has a population of about 3.7 million registered inhabitants in an area of 891 square kilometers.",
        1,
    ),
    (
        "how many people live in berlin",
        "New York City is famous for the Metropolitan Museum of Art and its many theaters.",
        0,
    ),
    (
        "how many people live in berlin",
        "Berlin is the capital and largest city of Germany, both by area and by population.",
        1,
    ),
    (
        "how many people live in berlin",
        "The recipe calls for two cups of flour, one egg and a pinch of salt.",
        0,
    ),
    (
        "how many people live in berlin",
        "Munich is the capital of Bavaria and hosts the annual Oktoberfest.",
        0,
    ),
    (
        "what causes rain",
        "Rain forms when water vapour condenses into droplets heavy enough to fall from clouds.",
        1,
    ),
    (
        "what causes rain",
        "The stock market fell sharply after the central bank raised interest rates.",
        0,
    ),
    (
        "what causes rain",
        "Clouds are made of tiny water droplets; when they merge and grow heavy, precipitation falls.",
        1,
    ),
    (
        "what causes rain",
        "Deserts receive very little precipitation throughout the year.",
        0,
    ),
    ("what causes rain", "The football match was postponed because of the storm.", 0),
    (
        "best way to learn python",
        "Start with the official Python tutorial, then write small programs every day to practice.",
        1,
    ),
    (
        "best way to learn python",
        "Pythons are large constricting snakes found in Africa, Asia and Australia.",
        0,
    ),
    (
        "best way to learn python",
        "Online courses and coding exercises are an effective way to pick up a programming language.",
        1,
    ),
    (
        "best way to learn python",
        "The Eiffel Tower was completed in 1889 for the World's Fair.",
        0,
    ),
    (
        "best way to learn python",
        "Java is a statically typed language used for enterprise software.",
        0,
    ),
    (
        "symptoms of the flu",
        "Influenza commonly causes fever, cough, sore throat, muscle aches and fatigue.",
        1,
    ),
    (
        "symptoms of the flu",
        "Flu symptoms usually appear suddenly and can include chills and headache.",
        1,
    ),
    (
        "symptoms of the flu",
        "The airline added three new flights to Tokyo this summer.",
        0,
    ),
    (
        "symptoms of the flu",
        "A healthy diet includes vegetables, fruit and whole grains.",
        0,
    ),
    (
        "symptoms of the flu",
        "Chimney flues should be cleaned once a year to prevent fires.",
        0,
    ),
]

# Calibration: different text, same distribution (query + passage, padded).
CALIB_PAIRS = [
    (
        "who wrote hamlet",
        "Hamlet is a tragedy written by William Shakespeare around 1600.",
    ),
    ("who wrote hamlet", "The Danish prince seeks revenge for his father's murder."),
    ("capital of japan", "Tokyo is the capital and most populous city of Japan."),
    ("capital of japan", "Kyoto was the imperial capital for over a thousand years."),
    (
        "how to boil an egg",
        "Place the egg in boiling water for nine minutes, then cool it in ice water.",
    ),
    (
        "how to boil an egg",
        "Chickens lay eggs roughly once a day under good conditions.",
    ),
    (
        "what is photosynthesis",
        "Plants convert light, water and carbon dioxide into glucose and oxygen.",
    ),
    ("what is photosynthesis", "The museum is closed on Mondays and public holidays."),
    (
        "speed of light",
        "Light travels at about 299,792 kilometres per second in a vacuum.",
    ),
    (
        "speed of light",
        "Sound is much slower than light, at roughly 343 metres per second in air.",
    ),
    (
        "why is the sky blue",
        "Rayleigh scattering makes short blue wavelengths scatter more than red ones.",
    ),
    ("why is the sky blue", "The painting uses a palette of deep blues and greens."),
    (
        "largest ocean",
        "The Pacific Ocean is the largest and deepest of the world's oceans.",
    ),
    ("largest ocean", "The Atlantic separates the Americas from Europe and Africa."),
    (
        "how do vaccines work",
        "Vaccines train the immune system to recognise a pathogen without causing disease.",
    ),
    ("how do vaccines work", "The bakery opens at seven every morning."),
]

# NLI pairs (premise, hypothesis) for NLI heads; labels come from the fp32 model.
NLI_PAIRS = [
    ("A man is playing a guitar on stage.", "A person is performing music."),
    ("A man is playing a guitar on stage.", "The man is sleeping at home."),
    ("Two dogs are running through a field.", "Animals are outside."),
    ("Two dogs are running through a field.", "The dogs are sitting indoors."),
    ("A woman is cutting vegetables in the kitchen.", "Someone is preparing food."),
    ("A woman is cutting vegetables in the kitchen.", "A woman is driving a car."),
    (
        "The children are building a sandcastle on the beach.",
        "Kids are at the seaside.",
    ),
    (
        "The children are building a sandcastle on the beach.",
        "The children are in a classroom.",
    ),
    ("A cyclist rides up a steep mountain road.", "Someone is on a bicycle."),
    ("A cyclist rides up a steep mountain road.", "The road is flat."),
    ("An old man reads a newspaper on a bench.", "A man is reading."),
    ("An old man reads a newspaper on a bench.", "A young girl is swimming."),
]


def dev_pairs(eval_pairs, n=64, seed=0):
    """Text pairs for choosing a mixed-precision mix: every first/second text
    of the built-in sets crossed, minus `eval_pairs`, shuffled -- so the
    choice is never made on the pairs it is then evaluated on."""
    firsts = {a for a, *_ in CALIB_PAIRS + EVAL_PAIRS} | {a for a, _ in NLI_PAIRS}
    seconds = {b for _, b, *_ in CALIB_PAIRS + EVAL_PAIRS} | {b for _, b in NLI_PAIRS}
    held_out = {(a, b) for a, b, *_ in eval_pairs}
    out = [
        (a, b)
        for a in sorted(firsts)
        for b in sorted(seconds)
        if (a, b) not in held_out
    ]
    random.Random(seed).shuffle(out)
    return out[:n]


def _tokenize(tok, pairs, seq, names):
    enc = tok(
        [a for a, *_ in pairs],
        [b for _, b, *_ in pairs],
        padding="max_length",
        truncation=True,
        max_length=seq,
        return_tensors="np",
    )
    return [
        {n: enc[n][i : i + 1].astype(np.int64) for n in names}
        for i in range(len(pairs))
    ]


def export(model_id: str, seq: int, out_path: str, attn: str = "sdpa"):
    """Legacy-exporter ONNX at a static [1, seq]; returns (tokenizer, input names)."""
    import torch
    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_id, attn_implementation=attn
    ).eval()
    import inspect

    accepted = inspect.signature(model.forward).parameters
    names = [
        n
        for n in ("input_ids", "attention_mask", "token_type_ids")
        if n in tok.model_input_names and n in accepted
    ]
    sample = _tokenize(tok, CALIB_PAIRS[:1], seq, names)[0]
    args = tuple(torch.from_numpy(sample[n]) for n in names)

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, *xs):
            return self.m(**dict(zip(names, xs))).logits

    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        torch.onnx.export(
            Wrap(model), args, out_path, input_names=names, output_names=["logits"],
            opset_version=17, dynamo=False,
        )  # fmt: skip
    return tok, names


def _ort(model_path, feeds):
    import onnxruntime as ort

    s = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return [s.run(None, f)[0][0] for f in feeds]


def build(work, tag, onnx_path, calib, precision, method="MinMax"):
    """`pulsar2 build` at `precision` ("U8", "U16", or a `layer_configs`
    list), cached by `mixed_precision.pulsar2_build`."""
    import mixed_precision as mp
    import onnx

    if isinstance(precision, list):  # explicit layer_configs (a mixed build)
        cfg = precision
    elif precision == "U8":
        cfg = None
    else:
        cfg = mp.u16_layer_configs(onnx.load(onnx_path, load_external_data=False))
        cfg[0]["data_type"] = precision
    return mp.pulsar2_build(
        os.path.join(work, tag), onnx_path, calib, cfg, method=method, image=IMAGE
    )


def metrics(ref, got, pairs):
    ref, got = np.stack(ref), np.stack(got)
    out = {
        "logit_rel_err": float(np.linalg.norm(got - ref) / np.linalg.norm(ref)),
        "logit_max_abs": float(np.abs(got - ref).max()),
    }
    if ref.shape[1] > 1:
        out["argmax_agree"] = float((ref.argmax(1) == got.argmax(1)).mean())
    else:  # one relevance logit: rank passages per query
        from itertools import combinations

        taus, top1 = [], []
        queries = sorted({q for q, *_ in pairs})
        for q in queries:
            idx = [i for i, p in enumerate(pairs) if p[0] == q]
            r, g = ref[idx, 0], got[idx, 0]
            conc = sum(
                np.sign(r[a] - r[b]) == np.sign(g[a] - g[b])
                for a, b in combinations(range(len(idx)), 2)
            )
            n = len(idx) * (len(idx) - 1) // 2
            taus.append(2 * conc / n - 1)
            top1.append(int(r.argmax() == g.argmax()))
        out["kendall_tau"] = float(np.mean(taus))
        out["top1_agree"] = float(np.mean(top1))
        out["sign_agree"] = float((np.sign(ref[:, 0]) == np.sign(got[:, 0])).mean())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("models", nargs="+")
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument(
        "--precision",
        nargs="*",
        default=["U8"],
        help='U8, U16, and/or "auto" (mixed_precision.auto: build U8 + U16, '
        "search a mix on dev_pairs(), check it on the card against the budget)",
    )
    ap.add_argument("--slack-db", type=float, default=1.0, help="auto: dB below U16")
    ap.add_argument("--max-err", type=float, help="auto: relative-error budget instead")
    ap.add_argument("--max-latency-ms", type=float, help="auto: latency budget")
    ap.add_argument(
        "--mix",
        nargs="*",
        default=[],
        metavar="NAME=FILE",
        help="extra builds with these quant.layer_configs (a JSON list, or a "
        "mixed_precision.py --json output); reported under NAME",
    )
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument(
        "--calib-method", default="MinMax", help="Pulsar2 calibration_method"
    )
    ap.add_argument("--nli", action="store_true", help="evaluate on NLI pairs")
    ap.add_argument("--work", required=True)
    ap.add_argument("--json")
    ap.add_argument(
        "--dev-npz",
        help="also write dev_pairs() tokenized as <dir>/<model>.dev.npz, the "
        "--feeds for mixed_precision.py",
    )
    ap.add_argument("--no-device", action="store_true")
    ap.add_argument("--runs", type=int, default=20, help="timed runs per model")
    a = ap.parse_args(argv)

    import mixed_precision as mp
    import onnx
    import pulsar2_backend

    results, evals = [], {}
    pairs = NLI_PAIRS if a.nli else EVAL_PAIRS
    for mid in a.models:
        tag = mid.replace("/", "__") + f"_s{a.seq}_{a.attn}"
        os.makedirs(a.work, exist_ok=True)
        raw = os.path.join(a.work, tag + ".onnx")
        tok, names = export(mid, a.seq, raw, a.attn)
        simp, ok = pulsar2_backend.simplify(onnx.load(raw))
        sp = os.path.join(a.work, tag + ".sim.onnx")
        onnx.save(simp, sp)
        ops = sorted({n.op_type for n in simp.graph.node})
        risks = pulsar2_backend.ax650_build_risks(simp)
        feeds = _tokenize(tok, pairs, a.seq, names)
        calib = _tokenize(tok, CALIB_PAIRS, a.seq, names)
        if a.dev_npz:
            dev = _tokenize(tok, dev_pairs(pairs), a.seq, names)
            os.makedirs(a.dev_npz, exist_ok=True)
            np.savez(
                os.path.join(a.dev_npz, tag + ".dev.npz"),
                **{n: np.concatenate([d[n] for d in dev]) for n in names},
            )
        ref = _ort(sp, feeds)
        raw_ref = _ort(raw, feeds)
        rec = {
            "model": mid, "seq": a.seq, "simplify_ok": ok, "ops": ops, "build_risks": risks,
            "sim_vs_export_max_abs": float(np.abs(np.stack(ref) - np.stack(raw_ref)).max()),
            "builds": {},
        }  # fmt: skip
        mixes = {}
        for spec in a.mix:
            label, path = spec.split("=", 1)
            with open(path) as fh:
                cfg = json.load(fh)
            mixes[label] = cfg["layer_configs"] if isinstance(cfg, dict) else cfg
        for prec in [p for p in [*a.precision, *mixes] if p != "auto"]:
            r = build(
                a.work,
                f"{tag}_{prec}",
                sp,
                calib,
                mixes.get(prec, prec),
                a.calib_method,
            )
            b = {
                "success": r.build_dir is not None,
                "build_s": round(r.seconds, 1),
                "cached": r.cached,
                "axmodel": r.axmodel,
            }
            if r.build_dir is None:
                b["error"] = r.error
            else:
                b["tensors_16bit"] = mp.count_16bit(r.build_dir)
            rec["builds"][prec] = b
        if "auto" in a.precision:
            dev = _tokenize(tok, dev_pairs(pairs), a.seq, names)
            rep = mp.auto(
                sp,
                calib,
                dev,
                os.path.join(a.work, tag + "_auto"),
                slack_db=a.slack_db,
                max_err=a.max_err,
                max_latency_ms=a.max_latency_ms,
            )
            rec["auto"] = rep
            if rep["chosen"]:
                c = next(c for c in rep["candidates"] if c["label"] == rep["chosen"])
                rec["builds"]["auto:" + rep["chosen"]] = {
                    "success": True,
                    "build_s": c["build_s"],
                    "axmodel": c["axmodel"],
                    "tensors_16bit": c["tensors_16bit"],
                }
        results.append(rec)
        evals[mid] = feeds, ref
        print(json.dumps(rec, indent=1, default=str), flush=True)

    if not a.no_device:
        from axcl_session import AXSession, health_check

        with AXSession() as s:
            print("health before:", health_check(s), flush=True)
            for rec in results:
                feeds, ref = evals[rec["model"]]
                for prec, b in rec["builds"].items():
                    if not b["success"]:
                        continue
                    m = s.load(b["axmodel"])
                    order = [i.name for i in m.inputs]
                    got = [
                        s.run(m, [f[n] for n in order])[0].reshape(-1) for f in feeds
                    ]
                    us0, n0 = s.exec_us, s.runs
                    for _ in range(a.runs):
                        s.run(m, [feeds[0][n] for n in order])
                    b["latency_ms"] = round((s.exec_us - us0) / (s.runs - n0) / 1000, 3)
                    b.update(metrics(ref, got, pairs))
                    s.unload(m)
                    shown = {k: v for k, v in b.items() if k != "axmodel"}
                    print(rec["model"], prec, json.dumps(shown), flush=True)
            print("health after:", health_check(s), flush=True)

    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
