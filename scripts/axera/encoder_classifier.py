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
import shutil
import sys
import time
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


def build(work, tag, onnx_path, names, calib, precision, op_types, method="MinMax"):
    import pulsar2_docker as pd

    root = os.path.join(work, tag)
    os.makedirs(os.path.join(root, "dataset"), exist_ok=True)
    os.makedirs(os.path.join(root, "config"), exist_ok=True)
    shutil.copy(onnx_path, os.path.join(root, "model.onnx"))
    inputs = []
    for n in names:
        pd.make_numpy_calibration_tar(
            os.path.join(root, f"dataset/{n}.tar"), [c[n] for c in calib]
        )
        inputs.append(
            {
                "tensor_name": n,
                "calibration_dataset": f"./dataset/{n}.tar",
                "calibration_format": "Numpy",
                "calibration_size": len(calib),
            }
        )
    quant = {
        "input_configs": inputs,
        "calibration_method": method,
        "precision_analysis": False,
    }
    if precision != "U8":
        quant["layer_configs"] = [
            {"op_types": sorted(op_types), "data_type": precision}
        ]
    cfg = {
        "model_type": "ONNX",
        "npu_mode": "NPU1",
        "quant": quant,
        "compiler": {"check": 0},
    }
    with open(os.path.join(root, "config/c.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    t0 = time.time()
    r = pd.build(
        root,
        "model.onnx",
        "out",
        config_path="config/c.json",
        image=IMAGE,
        timeout=3600,
    )
    return r, time.time() - t0


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
    ap.add_argument("--precision", nargs="+", default=["U8"])
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument(
        "--calib-method", default="MinMax", help="Pulsar2 calibration_method"
    )
    ap.add_argument("--nli", action="store_true", help="evaluate on NLI pairs")
    ap.add_argument("--work", required=True)
    ap.add_argument("--json")
    ap.add_argument("--no-device", action="store_true")
    ap.add_argument("--runs", type=int, default=20, help="timed runs per model")
    a = ap.parse_args(argv)

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
        ref = _ort(sp, feeds)
        raw_ref = _ort(raw, feeds)
        rec = {
            "model": mid, "seq": a.seq, "simplify_ok": ok, "ops": ops, "build_risks": risks,
            "sim_vs_export_max_abs": float(np.abs(np.stack(ref) - np.stack(raw_ref)).max()),
            "builds": {},
        }  # fmt: skip
        for prec in a.precision:
            r, secs = build(
                a.work,
                f"{tag}_{prec}",
                sp,
                names,
                calib,
                prec,
                set(ops),
                a.calib_method,
            )
            b = {
                "success": r.success,
                "build_s": round(secs, 1),
                "axmodel": r.axmodel_path,
            }
            if not r.success:
                b["error"] = (r.error or r.stdout_tail)[-1500:]
            rec["builds"][prec] = b
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
