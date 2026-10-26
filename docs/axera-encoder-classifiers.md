# BERT-style encoder classifiers on the AX650N

Cross-encoders and rerankers, NLI / zero-shot classifiers and plain sequence
classifiers are small BERT-family encoders with a classification head. This
page covers getting them from a Hugging Face checkpoint through onnxsim and
Pulsar2 onto the AX8850 card (AX650N), and how accurate and fast they are
there. It follows up the "Axera Pulsar2" section of PR #2217, which found
three blockers by static checks only.

Reproduce with `scripts/axera/encoder_classifier.py` (export -> onnxsim ->
`pulsar2 build` -> card). It needs torch, transformers, a loaded
`pulsar2:7.0-lite` image and the AXCL VM.

## What was in the way, and what changed

| Blocker | Pulsar2 result (confirmed) | Fix |
|---|---|---|
| Since #2217, plain `simplify()` fuses masked encoder attention into com.microsoft `Attention` | `KeyError('dont support Attention opr in AXOPS/ONNXOPS/CUSTOM_OPS')` | Axera flows pass `attention_fusion_backend="none"` |
| SDPA exports guard the attention Softmax with `Where(IsNaN(P), 0, P)` | `KeyError('dont support IsNaN opr ...')` | New opt-in pass `eliminate_sdpa_nan_guard` |
| RoBERTa position ids use `CumSum`, which is not on the published AX650 list | Builds into the NPU subgraph and is exact on the card | None needed |

All Axera entry points that simplify for Pulsar2 now go through
`pulsar2_backend.simplify()` / `simplify_kwargs()`. These are `worker.py`,
`convert_onnxmodelzoo.py`, `screen_onnxmodelzoo.py`,
`build_resident_train_step.py` and `build_calib_signal_probe.py`. The helper
applies `pulsar2_backend.SIMPLIFY_KWARGS`, which sets
`attention_fusion_backend="none"` and
`extra_optimizers=["eliminate_sdpa_nan_guard"]`. The suite fixture
`axera_masked_encoder_sdpa` (`scripts/axera/models.py`) is a legacy-exported
SDPA encoder layer with RoBERTa position ids. Plain `simplify()` turns it into
an `Attention` blocker. `tests/test_pulsar2_compat.py` checks that the worker
path leaves only AX650 ops and still matches onnxruntime.

On the tiny BERT export, the old path fails `pulsar2 build` on `Attention`.
With `attention_fusion_backend="none"` alone, it fails on `IsNaN`. With both
fixes, it builds and runs.

### `eliminate_sdpa_nan_guard`

`onnxsim/passes/eliminate_sdpa_nan_guard.h` is opt-in (`PassType::Other`):

```python
onnxsim.simplify(model, attention_fusion_backend="none",
                 extra_optimizers=["eliminate_sdpa_nan_guard"])
```

It matches `Where(IsNaN(P), 0, P)` where `P = Softmax(scores + mask)` and
`scores` is the scaled Q@K^T. It rewrites the guard in two cases:

1. **The mask is provably finite.** That covers a finite constant, an
   integer/bool tensor cast to float, or a `Where` that picks between such
   values behind shape-only ops. The dynamo exporter's `finfo.min` mask is
   this case, and so is unmasked attention. A finite Softmax row is never
   NaN, so the guard is dropped.
2. **The mask is `Where(c, 0, -inf)`**, which is what the legacy exporter
   emits. Swapping -inf for `finfo.min` alone would change the all-padding
   row: it would come out as a uniform average over the padding instead of
   zeros. The pass keeps the zeros by replacing the guard with an explicit
   row-validity multiply:

   ```
   valid = ReduceMax(Cast(keep, float), axes=[-1], keepdims=1)
   out   = Softmax(scores + Where(c, 0, finfo.min)) * valid
   ```

   For a row with at least one kept key, `exp(score + finfo.min - max)`
   underflows to exactly 0, just as `exp(-inf)` does. The Softmax is then
   bit-identical. For an all-masked row, the result is finite × 0 = 0, which
   matches the guard's output. All of these ops are on the AX650 list.

Both cases assume the attention scores themselves are finite and above
about -1e31. With NaN/Inf already in Q or K, the original guard would zero a
NaN row that the rewrite propagates. Eager attention, which has no guard,
makes the same assumption. Because of it, the pass is a legalization you opt
into rather than a default simplification. The pass declines in these cases:

- a float mask that comes from a graph input, since it can hold -inf;
- a Softmax that is not over the key axis;
- for case 2, a `P` or `IsNaN(P)` with other consumers.

Tests:

- `tests/test_eliminate_sdpa_nan_guard.py` builds models with the parser.
  It covers both mask forms, both operand orders, opset 17/18 (`ReduceMax`
  axes as attribute or input), a mask shared across layers, and the decline
  cases. Each test compares outputs bit for bit, including the all-padding
  row.
- `tests/test_encoder_classifier_hf_integration.py::test_sdpa_nan_guard_removed_without_fusion`
  runs real exports: 4 tiny HF models × both exporters, with no guard or
  `Attention` left.

On every model below, the simplified fp32 graph matches the export exactly
(max |diff| 0.0 on the evaluation inputs).

### `CumSum`

A RoBERTa position-id subgraph (`Equal/Not/Cast/CumSum/Mul/Add` -> `Gather`
of a table with row i = i) was built with Pulsar2 and run on the card with 20
padded inputs. It matched onnxruntime exactly (0 error). Whole RoBERTa
classifiers build and run with `CumSum` in place (see the table).

An exact rewrite, `CumSum` -> `MatMul` with a constant triangular matrix on
the int32 data, was tried and dropped. Pulsar2 rejects the integer MatMul
with `OnnxOptimizationError: transformation fail`, so the rewrite would turn
a working op into a broken one. `CumSum` is now recorded in
`pulsar2_ops.AX650_CONFIRMED_WORKING_OPS`, and `IsNaN` in
`AX650_CONFIRMED_BROKEN_OPS`.

## On the card

Setup for every row:

- Export: legacy exporter, opset 17, static `[1, seq]`, transformers' default
  SDPA attention, so the -inf mask and guard are in the graph.
- Build: `pulsar2:7.0-lite`, MinMax calibration on 16 tokenized query /
  passage pairs, Numpy calibration on the int64 inputs (which Pulsar2 turns
  into int32).
- Precision: `U8` is Pulsar2's default int8. `U16` sets `layer_configs` to
  every ONNX op type at U16, which gave 312 tensors at 16 bits and 125 at 8
  bits on MiniLM.
- Reference: onnxruntime fp32 on the same simplified graph.
- Latency: NPU execution time per call (`AXSession` `exec_us`), averaged
  over 20 runs.
- The card health check was 0.0 LSB before and after every batch.

| Model | Arch | seq | Precision | Build | Latency | Logit rel. err | Agreement with fp32 |
|---|---|---|---|---|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` (reranker) | BERT, 6L/384 | 128 | U8 | 34 s | 2.93 ms | 9.1 % | rerank Kendall τ 1.00, top-1 4/4, sign 95 % |
| | | | U16 | 38 s | 3.43 ms | 1.6 % | τ 0.95, top-1 4/4, sign 100 % |
| `cross-encoder/nli-distilroberta-base` (NLI) | RoBERTa, 6L/768, `CumSum` | 128 | U8 | 56 s | 6.73 ms | 17 % | argmax 12/12 |
| | | | U16 | 59 s | 8.01 ms | 5.1 % | argmax 12/12 |
| `distilbert-base-uncased-finetuned-sst-2-english` | DistilBERT, 6L/768 | 128 | U8 | 45 s | 6.63 ms | **101 %** | argmax 5/12 |
| | | | U16 | 48 s | 7.73 ms | 15 % | argmax 12/12 |
| `hf-internal-testing/tiny-random-Bert…` | BERT | 64 | U8 | 25 s | 0.36 ms | 0.8 % | (random weights: logits ~0, not meaningful) |
| `hf-internal-testing/tiny-random-Roberta…` | RoBERTa, `CumSum` | 64 | U8 | 25 s | 0.43 ms | 0.9 % | (random weights) |
| `hf-internal-testing/tiny-random-DistilBert…` | DistilBERT | 64 | U8 | 24 s | 0.36 ms | 1.6 % | (random weights) |

How each model was evaluated:

- **Reranker:** 4 queries × 5 passages, scored on a single relevance logit.
  The rerank order per query and the top-1 passage match fp32 at U8. At U16,
  one near-tied pair swaps.
- **NLI:** 12 premise/hypothesis pairs.
- **SST-2:** the same 12 texts, fed as sentence pairs. The labels compared
  are the fp32 model's own.

Notes:

- **Most models are fine at int8, but DistilBERT SST-2 is not.** Its logits
  are essentially wrong at U8. The cause is not the padding mask: with an
  all-ones mask the error is still 101 %. MSE calibration does not help
  either (100 %). U16 brings it back to full label agreement. MiniLM and
  DistilRoBERTa keep their decisions at U8, and U16 cuts their logit error
  by 3–6× for 15–20 % more latency.
- **The `finfo.min` mask survives int8.** Every evaluation input is padded
  (pairs of 20–40 tokens in 128 positions), and the reranker's order is
  exact at U8.
- Build time is under a minute per model. Each model compiles to one NPU
  subgraph.

Exports from newer torch no longer have the guard. torch 2.11 (used for
every result here) emits it, while CI's torch 2.14.1 exports SDPA without
`IsNaN`. On those exports the pass simply finds nothing to do, and
`attention_fusion_backend="none"` is still required.

## Mixed precision: the residual stream at 16 bits

Whole-model U16 fixes every model above but costs 15–20 % latency. Running
only the **residual stream** at 16 bits gets almost all of that accuracy at
2–4 % latency. The residual stream is every `LayerNorm` and residual `Add`,
i.e. the tensors entering them. Rows are U8 / the best mix /
whole-model U16. The mixes were chosen by `scripts/axera/mixed_precision.py`
on a 64-pair dev set disjoint from the evaluation pairs, and all were built
and run on the card. Health was 0.0 LSB before and after every batch.

| Model | Build | 16-bit tensors | Latency | Logit rel. err | Agreement with fp32 |
|---|---|---|---|---|---|
| DistilBERT SST-2 | U8 | 0 | 6.63 ms | 101 % | argmax 5/12 |
| | **LN + Add (25 ops)** | 43 | **6.76 ms** | **22 %** | **12/12** |
| | U16 | 164 | 7.73 ms | 15 % | 12/12 |
| MiniLM-L6 reranker | U8 | 0 | 2.93 ms | 9.1 % | τ 1.00, top-1 4/4 |
| | **LN only (13 ops)** | 13 | **2.92 ms** | **3.9 %** | τ 0.95, top-1 4/4, sign 100 % |
| | LN + Add (26 ops) | 45 | 2.98 ms | 1.8 % | τ 0.95, top-1 4/4, sign 100 % |
| | U16 | 168 | 3.43 ms | 1.6 % | τ 0.95, top-1 4/4 |
| DistilRoBERTa NLI | U8 | 0 | 6.73 ms | 17 % | argmax 12/12 |
| | **LN + Add (26 ops)** | 45 | **6.99 ms** | **5.2 %** | **12/12** |
| | U16 | 169 | 8.01 ms | 5.1 % | 12/12 |

The 16-bit tensor counts are unique activation tensors at 16 bits in
`quant_axmodel.json`. The MiniLM τ of 0.95 is one near-tied passage pair, the
same pair that U16 swaps.

### How the mix is chosen, and what did not work

`mixed_precision.py` never rebuilds per candidate. It needs one U8 and one
U16 build of the model:

1. **Replay.** `replay.py` reproduces the card from a build's own scale table.
   Replaying the U8 and U16 builds gives logit errors of 100.9 % / 25 %
   (DistilBERT), 9.0 % / 2.0 % (MiniLM) and 16.3 % / 5.2 % (DistilRoBERTa).
   On the card they are 101 % / 15 %, 9.1 % / 1.6 % and 17 % / 5.1 %.
2. **Splice.** A mix is emulated by taking the U16 build's scales for the
   tensors the mix widens and the U8 build's scales for everything else.
3. **What a mix widens.** This took a wrong build to learn: `layer_configs`
   promotes an op's **inputs**, not its output. A promoted `Add` reads 16-bit
   operands and still writes 8 bits. An operand that some 8-bit op also reads
   stays 8-bit, and Pulsar2 inserts an `AxRequantizeLinear` into the
   promoted op, which gains nothing. So a tensor is 16-bit only when every
   op reading it is promoted (`Build.widened()`).
   - Before this rule, the first DistilBERT mix promoted the op producing
     the outlier tensor. It went 101 % → 94 % on the card while the splice
     predicted 38 %.
   - With the rule, the splice predicts three real mixed builds to within a
     few points: 0.93 / 0.91 / 0.39, against 0.94 / 0.91 / 0.41 on the card.
4. **Search.** Start from all-U8, and at each step add the op-type group
   whose addition scores best (output SNR on the dev set). Stop within 1 dB
   of all ops promoted. On DistilBERT and DistilRoBERTa this picks
   `LayerNorm` then `Add`. On MiniLM `LayerNorm` alone is enough.
5. **Emit.** The chosen ops' post-fusion names become
   `[{"layer_names": [...], "data_type": "U16"}]`, which go to
   `encoder_classifier.py --mix NAME=FILE`.

What did not work: ranking single tensors (or single ops) by their gain
alone, then promoting them greedily.

- **DistilBERT.** One tensor stands out: the layer-4 FFN residual `Add`
  output, +7.6 dB alone, which is DistilBERT's known outlier layer.
  Promoting it and the next five fixes the label on 11 of 12 pairs, but the
  logit error stays at 44–45 % on the card.
- **MiniLM.** The error is spread out, and 2–4 greedy tensors do nothing on
  the card (9.7–9.9 %).
- **DistilRoBERTa.** Greedy reaches 9.2 % with 4 tensors and 6.3 % with 10
  (7.15 / 7.42 ms). Both are worse and slower than LN + Add.

The residual stream only pays as a whole: widening one residual tensor
helps little while its neighbours are still 8-bit. That is why `search()`
adds whole groups rather than ranked tensors.

Reproduce:

```bash
# U8 + U16 builds (and the dev set)
python scripts/axera/encoder_classifier.py MODEL --precision U8 U16 --work W --dev-npz W
python scripts/axera/mixed_precision.py W/TAG.sim.onnx W/TAG_U8/out W/TAG_U16/out \
    --feeds W/TAG.dev.npz --json mix.json
python scripts/axera/encoder_classifier.py MODEL --precision --mix best=mix.json --work W
```

## Not covered

- Dynamo exports were not built with Pulsar2. Their `finfo.min` mask takes
  the simpler path (case 1) and is covered by the integration test.
- Eager-attention exports were not built either. They have no guard, and
  #2217 found all their ops on the AX650 list.
- Larger checkpoints (BERT-base 12L, `bge-reranker`) were not run.
- seq 512 was not run.
- Mixed precision was chosen and checked on 12–20 evaluation pairs per
  model (64 dev pairs for the choice). A larger task set (SST-2 / MNLI
  validation) would tighten the error figures.
- Batch > 1 was not run.
