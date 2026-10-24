# Whole-model LLM decode with Pulsar-free programs (AX650N / AX8850)

This page covers the decode of a whole LLM on the AX650N card where no program is
compiled per model. Every program of every layer is made without Pulsar2, at the model's
own calibration. Fused programs, the residual Add and the linear layers are retargeted from
native template builds (`docs/axera-width-and-linear-emit.md`), and the calibration is
U16 MinMax computed on the host (`docs/axera-pulsar-free-calibration.md`). Pulsar2 runs
only once per program shape, to build the templates. A new checkpoint of the same shapes
needs no Pulsar2 run at all.

Two checkpoints were run on the card: JackFram/llama-160m and Qwen/Qwen3-0.6B. Both are
compared with Pulsar2's own `pulsar2 llm_build` output on the same steps.

- Code: `scripts/axera/llm_pulsar_free/` (one script per step, below).
- Tests (no card, no Pulsar2, no checkpoint): `tests/test_axera_llm_pulsar_free.py`.
- Related: `docs/axera-llm-rpc-decode.md` (the `llm_build` decode loop used for the
  baseline).

## Pipeline

```sh
cd scripts/axera/llm_pulsar_free
export MODEL=qwen3_0.6b LLM_PF_OUT=/data/pf/qwen3_0.6b   # LLM_PF_TEMPLATES: default <out>/../templates
python pf_templates.py            # native U16 template builds (Pulsar2, Docker), 4 variants per shape
python pf_record.py               # float decode over the calibration passages -> <out>/acts
python pf_gptq.py --device cuda   # optional: GPTQ weights on the card's s8 grid -> <out>/gptq
GPTQ=1 python pf_generate.py rest # every program of the `rest` set (rest before first)
GPTQ=1 python pf_generate.py first
GPTQ=1 python pf_estimate.py --doc 0      # host estimate (fake-quant), no card
python pf_plan.py --fake --float-set      # wiring check against the float model, no card / Pulsar2
GPTQ=1 python pf_plan.py --doc 0          # on the card -> <out>/plan_doc0.json
python pf_baseline_llm_build.py --build-dir <llm_build out>   # Pulsar2 llm_build on the same steps
python pf_profile.py                      # per-program card time of the rest set
```

| script | what it does |
| --- | --- |
| `pf_model.py` | `Config`: the checkpoint (pinned Hugging Face revision, or `LLM_PF_SNAPSHOT`), the sizes, the float reference of one step (`llm_layer_decomposition.layer_graph`), RoPE tables, merged weights |
| `pf_layout.py` | plan layout (mask, merged-linear column offsets, program order), the evaluation passages, `FakeAX` (numpy stand-in for the runner's plan mode) |
| `pf_templates.py` | every fused graph the programs use, and the Pulsar2 template builds |
| `pf_record.py` | each program's inputs over real-text float decode steps |
| `pf_generate.py` | calibration and emission of every program (`u16_retarget.retarget_graph` / `retarget`, `linear_emit.emit_linear_u16`), `manifest_<set>.json` |
| `pf_fake_quant.py` | fake-quant evaluation of a program the way the card runs it |
| `pf_estimate.py` | the whole decode in fake-quant on the host |
| `pf_plan.py` | the decode on the card: every program resident, one `PLAN` per token |
| `pf_gptq.py` | `linear_emit.gptq_weight` for every linear (CUDA torch) |
| `pf_baseline_llm_build.py`, `pf_profile.py` | the `llm_build` reference point; per-program card time |

Calibration text: the prose of `README.md`, `docs/qat.md`, `docs/accuracy-drop.md` and
`docs/attention-quantization.md` (positions 1..127 of each). Evaluation text: the first 48
tokens of `docs/adaquant.md` and `docs/activation-memory-planning.md`, teacher-forced. Both
are read from the checkout. The calibration changes when those files change.

## The program set

Per layer, in order (`pf_layout.LAYER`), with the native template each one is retargeted
from:

| program | computes | template |
| --- | --- | --- |
| `n1_c` | coarse norm `u = x / sqrt(mean(x^2) + eps)` | `crms<H>` |
| `n1_rms` | RMSNorm of `u` with the gain as an input | `rms<H>` |
| `qkv5` | one linear: q, k, v, `q_rot`, `k_rot` (rotate_half folded into the weights) | `k<H>n<..>` |
| `rq`, `rk` | RoPE apply `q*c + r*sn`, or with Qwen3's q / k norm `(q*c + r*sn) / rms_head(q)` | `rope<n>` / `qkrope<n>` |
| `kc` (Qwen3) | per-head coarse norm of `[k \| k_rot]` (two-pass k norm) | `hcrms<2 kvd>` |
| `attn` | the whole layer's attention: q, the token-major K / V cache `[129, nkv*hd]`, mask | `attn<nh*hd>` |
| `o_proj`, `res1` | linear, residual Add | `k<..>n<H>`, `add<H>` |
| `n2_c`, `n2_rms`, `gu`, `silu_f`, `down_proj`, `res2` | second norm, gate \| up linear, fused SiLU, down linear, residual Add | |

The post programs are `nf_c`, `nf_rms` and `lm_head`. All activations are U16 and the
weights s8 per output channel. The two-pass norm keeps the fused RMSNorm's inner `x^2`
inside 16 bits. Its second pass is calibrated on the card's `u`, which is the coarse
program evaluated in fake-quant.

Calibration follows the SmolLM2 pipeline (`docs/axera-width-and-linear-emit.md`). Fused and
single-op programs are calibrated on their recorded samples plus x2 copies (norms,
residual Adds) or x sqrt(2) copies (products, attention). Linears use the real range x2;
a merged linear uses the union of its parts. Each program is retargeted from the first of
four template variants that accepts the target.

Two calibration sets cover the position whose activations are massive:

- `first`: position `first_pos`, with up to 512 different tokens.
- `rest`: every later position.

llama-160m's BOS embedding is all zeros, so position 0 is zero end to end and is skipped.
Position 1 carries residuals of about 1.7e3-2e3 at layers 3..10, against at most about 300
later. Qwen3-0.6B has no BOS: position 0 reaches about 6.4e3-7e3 from layer 4 on, against
at most about 400 later.

### The plan

`pf_plan.Plan` loads both sets and binds every program to shared device buffers once. The
shared buffers are the embedding row, the RoPE tables (one row per layer with Qwen3, whose
q / k norm gains are folded into the tables), the mask, and per layer a token-major
K / V cache. A step is one `PLAN` of the runner (`scripts/axera/vm/axcl_batch_runner.c`):

- three `W` lines for the token's inputs;
- for `rk`, a rebinding of its output onto K row `p` (no copy);
- one device copy of v's slice of the `qkv5` output into V row `p`;
- one `X` per program;
- one `R` for the logits.

The mask leaves rows `0..p` valid. In float this equals `layer_graph`'s layout (earlier
rows plus the new row at the end), and the test checks that.

### What Qwen3-0.6B needed beyond llama-160m

- **q / k RMSNorm before RoPE.** `qkrope` computes `(q*c + rot(q)*sn) / rms_head(q)` with
  the gain folded into the host's tables (`c = g * cos`, `sn = rot(g) * sin`). This is
  exact because `rot` permutes within a head. The k norm needs the two-pass form: the U16
  step of k's `x^2` is up to 0.83x its typical mean (layer 8). So `kc` computes `u = k / r0`
  per head on `[k | k_rot]`, one slice since `qkv5` is laid out `q | q_rot | k | k_rot | v`,
  and `rk` then normalizes `u` exactly.
- **Two Adds at one shift.** qkrope's two Adds (`q*c + r*sn`, and `+ eps`) shift 15 in every
  first template, so Pulsar2 writes `0x1ea0` once. A target whose RoPE Add needs 14 (L11
  `rk`) needs both writes. `retarget_graph` refuses that template, and the `qkrope<n>t`
  variants, whose Adds shift differently, supply it.
- **GQA attention** in one program: q as `[nkv, G, hd]`, K / V token-major.
- **Shared linears.** A linear's program serves both sets, so one copy of its weights sits
  on the card, when position 0's ranges are within 2x of the rest's. It is calibrated on
  the union of both sets. 12 first-set linears keep their own program.
- **Split first-set MLP at layers 0-2.** Position 0 creates the massive activation there.
  s8 weights times that activation put K / V row 0, the attention sink, off. `gu` and
  `down_proj` of the first set are therefore two s8 linears, `W ~ Q8 + R8`, and an Add
  (~16-bit weights).
- **RoPE programs calibrated on both sets.** The first set shares the rest set's `rq` /
  `rk` / `kc`, because sin = 0 at position 0 degenerates their own calibration. They are
  calibrated on both sets: position 0's cos = 1 output reaches past the rest range
  (`rk` 30.7 against 21.6 at L18), and without position 0 in the calibration K row 0 is
  clipped.
- **U16 linears at k = 1024 and 2048** (segments `[240, 272]` and `[208, 272 x 3]`) and the
  1024 x 151936 lm_head (9 DEQUANT jobs, a 180-byte IO tail) are byte-equal to native
  builds. `linear_emit.U16_K` / `U16_TAILS` now include them.

## Results on the card

AX8850 (AX650N) card in the LXD VM, 2026-10-24. 48 teacher-forced steps per passage
(llama-160m: 47, position 0 skipped), compared with the float model (`Config.float_step`).
Top-1 means agreement with float's top-1. The logit error is the RMS error of the logits
over their RMS, as a median (p90) over the steps. The card's health check read 0.0 before
and after every run.

| model | program set | adaquant | activation-memory-planning | s / token |
| --- | --- | --- | --- | --- |
| llama-160m | Pulsar-free, round-to-nearest | 44/47, 1.31% (1.82%) | 46/47, 1.22% (1.48%) | 0.12 (card 85 ms) |
| llama-160m | Pulsar2 `llm_build` | 44/47, 1.89% (2.62%) | 46/47, 1.56% (2.03%) | 0.15 |
| Qwen3-0.6B | Pulsar-free, round-to-nearest | 44/48, 4.4% | 46/48, 4.0% | 0.21-0.22 |
| Qwen3-0.6B | Pulsar-free, GPTQ | 45/48, 3.36% (5.63%) | 44/48, 2.91% (3.99%) | 0.18 (card 163 ms) |
| Qwen3-0.6B | Pulsar-free, GPTQ from `pf_gptq.py` | 43/48, 3.29% (5.16%) | 44/48, 2.89% (4.27%) | 0.19 (card 162 ms) |
| Qwen3-0.6B | Pulsar2 `llm_build` | 45/48, 7.47% (9.73%) | 47/48, 7.27% (9.66%) | 0.29 |

Notes on the table:

- **Program counts.** Qwen3-0.6B: 673 programs (827 MB) resident, 423 programs run per
  token (15 per layer + 3). llama-160m: 342 programs (336 MB), 171 per token.
- **Card time.** "card" is the runner's execution time of the step's plan. The rest of
  the wall time is the host side: tables and logits over the shared folder, and the plan
  file.
- **`llm_build` timing.** The `llm_build` rows run through the session path of
  `llm_layer_loop` (K / V caches uploaded per call). That path is slower than its RPC
  path with resident caches (`docs/axera-llm-rpc-decode.md`: Qwen3-0.6B at 224 ms per
  token).
- **Qwen3 `pf_gptq.py` row.** The programs `pf_generate.py` emits from this repo's
  `linear_emit.gptq_weight` weights (2026-10-25). The GPTQ row above used the scratch
  GPTQ loop that `gptq_weight` reimplements. Its adaquant misses are at positions
  0, 3, 10, 13 and 46, and its activation-memory-planning misses at 0, 13, 28 and 34.
  Run back to back, both GPTQ sets take 162 ms of card time per token.
- **Qwen3 round-to-nearest row.** These are the card results reported from the run before
  GPTQ. Its JSON was overwritten by the GPTQ run, so no p90 is given.

Every top-1 miss of the Pulsar-free runs picks the float model's runner-up. The float
logit gap between the two is at most 0.2, against a logit standard deviation of 2-4:

| model, passage, position | float top-1 / top-2 gap | missed by |
| --- | --- | --- |
| llama-160m, adaquant, 24 / 30 / 46 | 0.033 / 0.036 / 0.002 | Pulsar-free (24 also `llm_build`) |
| llama-160m, activation-memory-planning, 46 | 0.072 | both |
| Qwen3-0.6B, both passages, 0 | 0.008 | Pulsar-free; `llm_build` picks a token 1.45 below top-1 |
| Qwen3-0.6B, adaquant, 3 / 10 | 0.021 / 0.068 | Pulsar-free (GPTQ) |
| Qwen3-0.6B, activation-memory-planning, 13 / 28 / 34 | 0.040 / 0.006 / 0.202 | Pulsar-free (GPTQ) |

### Where the error comes from

The host estimate (`pf_estimate.py`) evaluates every program in fake-quant. With
round-to-nearest weights it reproduces the card's Qwen3 accuracy closely: 44/48 and 47/48,
3.41% and 3.32%. Keeping every linear's float weights while still rounding every
activation (`--float-weights qkv5,o_proj,gu,down_proj,lm_head`) brings that to 45/48 and
48/48, 0.82% and 0.72%. Most of the remaining error is therefore s8 weight rounding, not
the U16 activations. The scratch run reported the same ablation on the card as 4.15% -> 1.45% median; that run's JSON was not kept.
GPTQ (`linear_emit.gptq_weight`, `docs/axera-width-and-linear-emit.md`) takes part of it
back.

llama-160m's host estimate gives 45/47 and 46/47, 1.26% and 1.14%.

## Checks

- **Offline** (`tests/test_axera_llm_pulsar_free.py`):
  - The plan mask attends to the same rows as `layer_graph`'s.
  - `FusedEval` without a calibration equals `ReferenceEvaluator` on every fused graph.
  - The split weights sum to the original.
  - `FakeAX` runs the runner's plan lines.
  - **Whole-plan wiring check.** On `llm_layer_decomposition.make_synthetic`'s qwen3-like,
    llama GQA and llama MHA models, the whole plan reproduces the float model step by step
    to 1e-4 relative error, and so does the host chain with every program exact. Wiring
    covered: both program sets, shared and split programs, K row rebinding, V row copies,
    merged-linear slices. Float stand-ins replace the programs (`write_float_set`).
- **Against the runs above.** The scripts in this directory regenerate those runs' programs
  from the same templates and recorded activations byte for byte:
  - Qwen3-0.6B: layer 0 of both sets, with GPTQ weights, including `kc`, the shared RoPE
    programs and the split linears.
  - llama-160m: layer 11 and the post programs of both sets.

  `pf_estimate.py` reproduces the runs' host estimates exactly, and `pf_plan.py --fake` the
  float wiring check (llama-160m: median 1.1e-4).
- **GPTQ.** `pf_gptq.py` uses master's `linear_emit.gptq_weight`. That function is a
  reimplementation of the GPTQ loop the card runs used, and about 5% of its codes differ
  by a rounding step. Its host estimate is 44/48 and 45/48, 2.30% and 2.31%, against
  44/48 and 43/48, 2.38% and 2.42% for the scratch weights. On the card its programs
  give 43/48 and 44/48, 3.29% and 2.89% (the `pf_gptq.py` row of the results table).

## Limits

- **Teacher-forced decode only.** The runs cover 48 steps of two passages; nothing here
  measures free-running generation or a long context. The context is 128 positions (the
  cache has 129 rows), and there is no prefill: a prompt goes through one token at a time.
- **Calibration covers positions 0..127 only.** Beyond that, ranges are unchecked.
- **Not in the repository.** Templates, recorded activations and program sets stay local
  scratch. Qwen3-0.6B's program set is 827 MB, and its templates are 49 Pulsar2 builds
  (four per fused / Add shape, one per linear shape; llama-160m: 29).
- **Not ported from the scratch run.** The per-program card-vs-host diagnostics and the
  attention calibration-margin experiment (`m = 1.0 / 1.2 / 1.414` on recorded inputs) are
  left out; the margin in use is sqrt(2).
