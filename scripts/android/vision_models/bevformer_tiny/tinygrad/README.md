# BEVFormer-tiny encoder in tinygrad, run on the Hexagon DSP

The tinygrad counterpart of the hand-written path in `../msda_hvx/` (the frame runner that reaches
~90 ms/frame on the phone). Here the encoder is re-expressed in tinygrad, and its multi-scale deformable
attention (MSDA), the part that dominates the encoder, is generated as Hexagon C by tinygrad's DSP backend
and timed on the phone against the hand kernel.

## Files

| file | what |
|---|---|
| `bevformer_tg.py` | the 3-layer encoder (TSA, SCA, FFN, LayerNorm) in tinygrad, weights from `model.Encoder.state_dict()`; `msda()` is the vectorized form of the hand kernel's contract (`../../msda_hvx/msda_ref.py`) |
| `check.py` | `unit`: `msda()` vs torch `model.msda_rank5` on random TSA/SCA-shaped calls; `encoder`: `TinyEncoder` vs torch `model.Encoder` on the saved frames |
| `capture_msda.py` | runs `msda()` on one real call (`msda_io/l0_tsa`, `l0_sca`: layer 0 of a real frame, with the torch reference output) under `DEV=DSP MOCKDSP=1`, traces the launches and writes `kernels.c` (the kernels) and `run.c` (the launch sequence as straight-line C) |
| `phone/build.sh` | builds one skel = the hand MSDA skel (`msda_hvx/`) + both tinygrad sequences; `RUN=1` pushes and benchmarks under the phone lock |
| `phone/make_impl.py` | patches `msda_impl.c`: flags bit 16 runs the tinygrad sequence on one HVX thread, bits 17-23 run a launch prefix (profiling) |
| `phone/tg_dispatch.c` | routes a call by shape: TSA = NV 2 without visibility, SCA = NV 6 with it |

## Correctness

- Encoder, tinygrad (CPU) vs torch, scene-0103 frames 0-2: max abs 5.5e-6 / 7.6e-6 / 6.9e-6, cos 1.000000000.
- `msda()` vs torch on random calls: CPU 2.8e-6 (TSA) / 9.2e-7 (SCA); DSP-mock (qemu) the same (`check.py unit`).
- The real layer-0 calls (`msda_io/l0_tsa`, `l0_sca`): tinygrad DSP-mock 4.9e-6 / 4.3e-6; generated C on the
  host 4.9e-6 / 4.3e-6; on the phone 5.0e-6 / 4.3e-6, `PASS` in the bench client's tolerance check.

Found on the way: the gathers must add the reference and the offset *before* the `- 0.5`, in the contract's
order `loc = ref + off / W`, `px = loc * W - 0.5`. The other order rounds differently and flips `floor()` at a
pixel boundary (one sample wrong by 0.1 on the real TSA frame).

## Speed on the phone

Xiaomi 12S (Hexagon V69), turbo, one msda call on a real layer-0 frame, median of 10, DSP-side time:

| call | hand, 1 thread | hand, 4 threads (the config in `msda_hvx/README.md`) | tinygrad, 1 thread | tinygrad vs hand 4T |
|---|---:|---:|---:|---:|
| TSA (2 frames, 4 points) | 19.96 ms | **5.05 ms** | 185.4 ms | 37x |
| SCA (6 cameras, 8 points, visibility) | 14.44 ms | **4.38 ms** | 1008.6 ms | 230x |

Scaled to the 3-layer encoder, the tinygrad sampling alone is about 3.6 s per frame (3 x (185 + 1009) ms),
against 55 ms for the whole hand encoder. The hand numbers match `msda_hvx/README.md`'s (4.4 / 4.0 ms) in turbo mode.
The non-MSDA encoder ops were not run on the phone through tinygrad, so this is the MSDA part only.

Where the tinygrad time goes (the prefix mode, one launch per step, phone):

- TSA, 6 launches: the four tap gathers, about 21 ms each; the final weighted reduction, about 100 ms.
- SCA, 14 launches: the visibility mask, 67 ms; twelve per-tap gather and weight kernels, 67-84 ms each;
  the final reduction, 152 ms.

So each kernel is a single-threaded pass over an intermediate of (NV, Q, M, P, D) = 3.8 M floats (15 MB), and
the taps don't share work. The hand kernel makes one pass per (query, head) with direct row reads and runs
on four threads. I did not read the generated instruction stream, so the exact cause of the per-element cost
is not verified.

The fork's qemu instruction counts (`test/external/dsp/hand/README.md`: MSDA TSA 1.7x, SCA 6.0x against the
hand kernel) point the same way but understate the device gap. Those counts are for a 96-query synthetic case,
and the device gap is measured on the real 2500-query calls.

## Levers not done

1. Fuse the four taps and the reduction into one kernel per value map, so no (NV, Q, M, P, D) tensor is
   written per tap.
2. Vectorize over the head dimension D = 32 (one HVX vector per gathered row), and check that the generated
   gathers actually use vector loads.
3. Split queries across four threads, as the hand kernel does. Each thread needs the kernel sequence for
   its query block, so this means generating the sequence with a query range.
4. Compute the SCA visibility mask on the host (it depends only on `bev_mask`), which removes its 67 ms kernel.

## Fusion: direct gathers through a rangeify fix (`rangeify_index_gather.patch`)

tinygrad's own indexing (`__getitem__`, `gather`) is a one-hot compare plus a sum over every row of the source.
Rangeify realizes each such reduce, so each tap of the MSDA became its own kernel. A direct `INDEX(buffer,
computed_index)` load has no reduce, but rangeify did not handle it: the buffer was indexed by the output ranges
and the index was never given them. The patch (against the fork's `dsp-consolidated`, commit `9a4680353`) fixes
that in `tinygrad/schedule/indexing.py`: the index args of an `INDEX` take the output's ranges, and the addressed
buffer is used as is. Tested on the CPU backend: `test_indexing`, `test_setitem_schedule`, `test_getitem_ops`,
`test_callify`, `test_tensor_data`, `test_dtype_weak`, `test_assign`, `test_realize_is_realize`,
`test_schedule_cache`, `test_jit`, `test_symbolic_tensor` all pass (240 tests, skips and one xfail as before).

`gather_rows()` in `bevformer_tg.py` uses it. The MSDA value must be a flat 1-D buffer: a `reshape` of a buffer
stays a `RESHAPE` view, and every index then goes through div/mod (the first version of the fix did this, with
about 2500 lines of generated C).

| version | TSA launches | TSA phone ms | SCA launches | SCA phone ms |
|---|---:|---:|---:|---:|
| one-hot gather (before) | 6 | 185 | 14 | 1009 |
| direct gather, value shaped (3-D) | 6 | 459 | 10 | 2807 |
| direct gather, flat value | 6 | 177 | 14 | 972 |
| shared sample coordinates (current) | 3 | 81 | 3 | 308 |

The gather is now one vector load per row in the final kernel (`*(float32*)(data1 + row << 5)`), with no
div/mod. The speedup is small (4% on SCA, 4% on TSA), because the time is elsewhere. TSA prefix profile (phone):
each of the four per-tap weight kernels takes about 21 ms and the final kernel about 90 ms. The tap kernels each
recompute the bilinear coordinates with a software float divide (`off / W`, needed for the contract's rounding
order) and the floor bit tricks, and the taps are realized because they have several consumers.

Then the coordinates are computed once: `msda` stacks the floor, fraction and attention weight into one tensor and
realizes it, so the taps read it instead of recomputing it (TSA 6 -> 3 launches, SCA 14 -> 3). Phone, prefix
profile: the coordinate kernel is 20 ms (TSA) / 84 ms (SCA) and the tap-and-reduce kernel 61 / 222 ms.

Tried and dropped: replacing `off / W` with a reciprocal multiply. The real cases stay at 4.9e-6 / 4.3e-6, but the
phone time doesn't move (81 / 307 ms), so the divide is not the cost. The generated coordinate kernel is scalar:
each lane of the float4 loads is extracted and goes through the scalar float path, and `floor` is a bit-manipulation
sequence. Remaining gap to the hand kernel (4T): about 16x (TSA) and 73x (SCA).

Next levers: vectorize the coordinate kernel over the 8 heads (codegen's upcast choice for these elementwise
chains), replace the bit-trick floor where it is exact, and move the SCA visibility mask to the host.

## Custom kernel (`Tensor.custom_kernel`, interop with a kernel written in UOps)

`msda_custom()` in `bevformer_tg.py` runs the tap-and-reduce stage as a kernel written in UOps and placed in the
tinygrad graph with `Tensor.custom_kernel(out, value, shared, inv, fxn=...)`. The coordinates still come from the
shared tensor, so only the tap stage changes. It matches `msda()` on CPU (max abs 1.5e-7 to 3.0e-7) and on the
DSP mock, and it passes the phone check (`PASS`):

| tap stage | TSA launches | TSA phone ms | SCA launches | SCA phone ms |
|---|---:|---:|---:|---:|
| generated by tinygrad (default) | 3 | 81 | 3 | 308 |
| UOp custom kernel (`msda_custom`) | 4 | 120 | 3 | 640 |

The custom kernel is slower because each thread handles one (query, head, channel) element and reads the five
coordinate fields for each channel, so the coordinate loads repeat 32 times. The next step is one thread per
(query, head) with the 32 channels as a vector accumulated over the (view, point, tap) reduction, built from
`Ops.STACK` loads. I have not verified that the fork's codegen takes a vector reduction. The default stays the
generated path.

## End to end (`e2e_tg.py`): images -> backbone -> encoder -> decoder -> boxes, in tinygrad

The whole model now runs in tinygrad: `backbone_tg.py` (ResNet-50 + FPN, frozen BN folded into the convs),
`bevformer_tg.py` (encoder, with the shared-coordinate MSDA), and `decoder_tg.py` (6 decoder layers with
self-attention and the 900-query deformable cross-attention through `msda`, plus the box refinement and the
last layer's heads). The host keeps the NMS-free decode, the geometry and the prev_bev rotation, as in torch.

Against the torch reference on scene-0103 frames 0-2 (tinygrad CPU backend):

| frame | feats max abs | bev max abs | cls max abs | bbox max abs | GT matched (tinygrad / torch) |
|---|---:|---:|---:|---:|---:|
| 0 | 2.0e-5 | 7.8e-6 | 5.0e-5 | 7.9e-5 | 8 / 8 of 23 |
| 1 | 1.6e-5 | 9.3e-6 | 9.0e-5 | 2.2e-4 | 11 / 11 of 29 |
| 2 | 1.8e-5 | 1.0e-5 | 5.9e-5 | 8.6e-5 | 16 / 16 of 30 |

Measured time per frame on this host's tinygrad CPU backend (single process, includes JIT compile on frame 0):

| frame | backbone | encoder | decoder | total |
|---|---:|---:|---:|---:|
| 0 | 24.2 s | 12.6 s | 16.6 s | 53.3 s |
| 1 | 18.2 s | 6.1 s | 9.0 s | 33.4 s |
| 2 | 18.2 s | 6.1 s | 8.4 s | 32.8 s |

These are not Hexagon numbers. What runs on the phone today is the MSDA only. The encoder's MSDA alone is
3 x (TSA 81 ms + SCA 308 ms), about 1.2 s per frame on the phone, against the hand frame at about 90 ms. The
decoder's six MSDA calls (900 queries) are not measured on the phone. Running the backbone (about 18 s on the CPU
backend) on the DSP needs codegen for convolution, which this path does not have yet. That is the next
milestone for an end-to-end phone number.

Reproduce: `python3 e2e_tg.py --ckpt <pth> --work <work>` with the same tinygrad fork on `PYTHONPATH`
(`work/frames/*.pt` from `validate.py`).

## Torch -> ONNX -> tinygrad: is the current model on that path?

No. The tinygrad model above is hand-written Tensor code that mirrors the torch modules weight by weight. It does not
go through the repo's export. The repo's export (`export.py`, `torch.onnx.export` at opset 17, then `onnxsim`) produces
`backbone6.sim.onnx` and the other pieces, but only the ORT and phone paths use them. The fork's `tinygrad.nn.onnx.OnnxRunner`
is the ONNX -> tinygrad lowering. Two checks on that path:

- `check_onnx_backbone.py`: `backbone6.sim.onnx` run by `OnnxRunner` on CPU, six cameras, frame 0: max abs 1.8e-5 vs the torch
  features (24.5 s on this host).
- `check_conv_dsp.py`: one 3x3 Conv + Relu from ONNX on the DSP mock: 4.8e-6.

Phone, float conv through the fork's graph path (`phone_conv.py`: `dsp_graph.capture` -> `emit` -> `build_skel` ->
client), one 3x3 64 -> 64 conv + bias + Relu at 30 x 50, fused into one kernel: the output is correct (max abs 4.3e-6 vs
torch, measured by a patched client; the client's own check compares raw bytes, which fails on last-bit rounding), but one
inference takes 664 ms. That is about 120 ns per MAC, scalar code. Float convolution through tinygrad's DSP codegen
cannot run the backbone on the phone. The fork's int8 QDQ path (`nn/onnx_qdq.py`, HMX via `dsp_graph`) is the one
built for convolution on this phone; the hand HMX conv measured in the fork is far faster. Next step: lower the
backbone's convolutions through that QDQ path, not the float one.

## The int8 backbone on the HMX through the tinygrad lowering (`phone_qdq_backbone.py`)

The repo's one-camera int8 backbone (`backbone1.q8.onnx`: uint8 NHWC in, uint8 feats out; 55 convs, 16 adds) goes through
`nn/onnx_qdq.py` (QDQ_HMX=1) -> `dsp_graph` -> skel -> phone. Three things were needed in the lowering
(`qdq_add_relayout.patch` against the fork's `dsp-consolidated`):

- The QDQ net's scales must be built before the capture (the lowering reads them with `.numpy()`; inside the capture no
  kernel runs, so they came back 0.0 and the QAdd constants were NaN). `phone_qdq_backbone.py` builds it first.
- A residual add whose two inputs were laid out with different row strides or row counts (a stride-1 conv's grid straight
  in, or a canonical one) now re-lays the second input to the first's layout (`relayout`), the ring keeping its zero point.
- The numpy emulation of the lowering (`qdq_emulate`) against onnxruntime on the real frame: 0 of 96000 values off. (On
  random uint8 input it is off by 4422 values, which is probably onnxruntime's own AVX2 int8 rounding; not checked on an
  AVX-VNNI host.)

Captured: 474 calls over 185 kernels, 24.8 MB constants, 165 MB scratch. On the phone, one inference:
- **output bit-exact vs onnxruntime: 0 of 96000 bytes-as-floats mismatched** (the client compares bytes; the patched
  client's float check is not run here);
- **time: 359 s per inference (1 iteration).** That is not usable. The profile run of 3 iterations shows two calls
  dominating (6.3 s and 2.7 s, the rest under 0.1 s), which does not add up to 359 s, so the cause is not found: possibly
  the 165 MB scratch arena, the HMX reported as unused (`hmx 0`), or per-call overhead. For comparison, the repo's QNN
  HTP run of this backbone is 20.9 ms.

So the path is correct end to end on the phone, and it is far too slow to use yet. Next: find where the 359 s goes
(per-call timing with a single iteration, the arena size, and whether the HMX kernels execute on the VTCM).

## Tap-loop variants that did not help (phone, 1 thread, turbo)

Both were measured against the 81 / 308 ms version above; neither is kept.
- Branch-free taps (clamp the pixel and multiply by the 0/1 mask, `MSDA_BRANCH_FREE=1`, off by default):
  TSA 81 ms (same), SCA 341 ms (slower). Correct: 4.9e-6 / 4.3e-6 on the DSP mock.
- Integer pixel corners as their own int32 tensor (no float compares or casts per tap): TSA 80 ms, SCA 318 ms.
  Correct, but no gain, so reverted.

The generated final kernel is already HVX-vectorized: the 32-channel accumulator uses vector loads and `dcfetch`. The
remaining cost is the scalar work per (query, head, point, tap), which the tap formulation repeats, and it is not in the
float compares. The next lever is the loop structure (fewer taps per output, one pass over the value per head), not the
arithmetic of each tap.

## Per-call profile of the int8 backbone (fixed)

The profiler's slot count was 256 in both `skel.c` and `client.c`, below the 474 calls of the backbone, so the earlier
per-call breakdown was truncated (`dsp_graph_profile_slots.patch`: 1024). With the cap raised, the per-call times sum to
the device total: 438.4 ms against 438.5 ms, HMX=1, bit-exact (0 of 96000 mismatches).

The time is in the convolution kernels (HMX). The heaviest calls, by kernel name from `graph.h`:

| call | kernel (shape encoded in the name) | ms |
|---:|---|---:|
| 257 | r_16_190_..._8 | 36.0 |
| 219 | r_8_379_2_2_..._2 | 33.9 |
| 423 | r_16_13_..._3_3_16 | 32.1 |
| 204 | r_8_379_2_2_... | 30.9 |
| 197 | r_2_1512_..._4_4 | 24.0 |
| 235 | r_8_379_2_2_... | 21.8 |
| 202 | r_2_48384_32_3_3 | 18.2 |

344 calls have nonzero time. The next concrete target is the efficiency of these HMX kernels (the fork's own notes put
its 3x3 kernels about 4x behind the hand HMX kernels), not the graph overhead.

## fp16 on the DSP mock: what works and what doesn't (`fp16_mock_helpers.patch`)

- The MOCKDSP link (v65) has no fp16 conversion helpers. `fp16_mock_helpers.patch` adds `__extendhfsf2` and
  `__truncsfhf2` (round to nearest even) and prepends them to a kernel only under the mock and only when the kernel uses
  `__fp16` or `_Float16`. The first version missed this trigger: clang lowers the renderer's `__fp16` casts to those calls
  without the source naming them.
- Scalar conversions are then bit-exact against numpy (checked on values including subnormals, infinities and -0).
- Vector fp16 conversions are now lane by lane under the mock (the renderer emits a scalar cast per lane): all 65536 half
  values convert to float32 exactly (checked against numpy).
- Still broken on the mock: fp16 vector arithmetic. Clang's native fp16 vector multiply on v65 gives zeros (a 4-element
  multiply returns 0). Fixing it needs the upcast heuristic to skip half kernels under the mock, or a per-op lane loop.
- Earlier note, now resolved for conversions only: vector fp16 conversions. The renderer emits
  `__builtin_convertvector` from an fp16 vector of 128 lanes to float32, and the v65 backend lowers it wrongly. That
  gives 255 of 256 wrong values in a 256-element cast, and fp16 elementwise multiply and add fail for the same reason.
  The fp16 matmul still has NaNs with HMX=0. v69 would have native fp16 vector instructions, but qemu aborts on them.
- So fp16 kernels can't be validated on the host yet. The next fix is either a scalar fallback for fp16 vector
  conversions under the mock, or integer-vector half-to-float code for v65 HVX.

## Reproduce

Needs the tinygrad fork with DSP support (checkout of `dsp-consolidated`, commit `9a4680353` at the time of
writing; upstream tinygrad was not tested) on `PYTHONPATH`, `clang-19`, and the msda case directories written by
`../msda_hvx/split.py dump` (`~/.cache/onnxsim-bevformer/work/msda_io`).

```sh
export PYTHONPATH=<tinygrad fork>:. CC=/usr/bin/clang-19 PARALLEL=0 CACHELEVEL=0
python3 check.py unit
python3 check.py encoder --ckpt $C/bevformer_tiny_epoch_24.pth --work $C/work --frames 0,1,2
G=<scratch>; K=$C/work/msda_io
python3 capture_msda.py $K/l0_tsa --out $G/gen_l0_tsa
python3 capture_msda.py $K/l0_sca --out $G/gen_l0_sca
# with the fork patched: git -C <tinygrad fork> apply <this dir>/rangeify_index_gather.patch
cd phone && GEN=$G OUT=$G/phone_build RUN=1 FLAGS=1,4,65536 TURBO=1 PHONE_LOCK_OWNER=<branch> ./build.sh
```

Phone runs go through `~/.cache/android-phone/phone-run` (set by `PHONE_RUN`), so they respect the shared
phone's lock. The profile build is `TG_PROF=1` with `FLAGS=65536+(p<<17)` for prefixes `p`.

### Why 359 s: the matmuls were not on the HMX (fixed with `HMX=1`)

Without `HMX=1` the tensor-core matcher used the scalar v65 vrmpy form: 38 of the 185 kernels built each 128-lane operand
by 128 scalar inserts per iteration (~25 M iterations for the 22.7 s kernel). With `HMX=1` no vrmpy kernel is left, and 54
kernels use the HMX tile op. Output stays bit-exact: 0 of 96000 values differ from onnxruntime.

| build | ms per inference (client, device) | mismatches vs ORT |
|---|---:|---:|
| default (no HMX) | 359 123 (1 iteration) | 0 |
| `HMX=1` | 437 (3-iteration mean) | 0 |

The per-call profile does not add up to the total in either build (about 15 % of the device time is attributed to calls),
so the breakdown is not trusted yet. The profile slot count (256 in skel.c and client.c) is also smaller than the 476 calls.
