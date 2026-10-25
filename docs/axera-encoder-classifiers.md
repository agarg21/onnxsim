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

## Not covered

- Dynamo exports were not built with Pulsar2. Their `finfo.min` mask takes
  the simpler path (case 1) and is covered by the integration test.
- Eager-attention exports were not built either. They have no guard, and
  #2217 found all their ops on the AX650 list.
- Larger checkpoints (BERT-base 12L, `bge-reranker`) were not run.
- seq 512 was not run.
- No mixed precision short of whole-model U16 was tried for DistilBERT.
- Batch > 1 was not run.
