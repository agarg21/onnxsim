#pragma once

// Which attention `Attention` form fuse_attention (passes/fuse_attention.h)
// is allowed to emit. Mirrors gemm_fusion_backend.h's pattern: selected from
// the ``attention_fusion_backend`` argument of ``simplify()`` through the
// ONNXSIM_ATTENTION_FUSION_BACKEND environment variable (see SimplifyImpl in
// onnxsim.cpp), read by FuseAttention on every match attempt.
//
// The two forms fuse_attention can emit:
//   * MatchAttention    -> com.microsoft `Attention` (ORT contrib, unmasked
//                          self-attention, QKV weights concatenated).
//   * MatchCausalAttention -> ai.onnx `Attention` (opset 23), with
//                          `is_causal` and/or an `attn_mask` input.
// fuse_gqa's `GroupQueryAttention` is a separate pass and is not gated here.

#include <string>

namespace onnxsim {

enum class AttentionFusionBackend {
  // Default: both forms, exactly as before this option existed.
  kAuto,
  // Only the ai.onnx `Attention` (opset 23) causal/masked form.
  kOnnx,
  // Only the com.microsoft `Attention` form (MatchAttention).
  kOrtContrib,
  // No attention fusion at all.
  kNone,
  // Only the com.microsoft `Attention` form, which never carries an
  // `attn_mask`/`mask_index`, the input onnxruntime-web's WebGPU Attention
  // kernel rejects at compute time ("Mask not supported"). The ai.onnx causal
  // form is excluded entirely, as for kOrtContrib.
  kWebGpu,
};

// Process-global; SimplifyImpl sets it at the start of every
// Simplify()/SimplifyConsumeInput() call, defaulting to kAuto when the env var
// is unset, so it never carries state over from an unrelated prior call.
void SetAttentionFusionBackend(AttentionFusionBackend backend);
AttentionFusionBackend GetAttentionFusionBackend();

// Parses the binding-facing spelling ("auto" / "onnx" / "ort_contrib" /
// "none" / "webgpu"). Throws std::invalid_argument on an unknown name.
AttentionFusionBackend ParseAttentionFusionBackend(const std::string& name);

// Whether each form may be emitted under `backend`.
inline bool AttentionBackendAllowsMicrosoftForm(
    AttentionFusionBackend backend) {
  return backend == AttentionFusionBackend::kAuto ||
         backend == AttentionFusionBackend::kOrtContrib ||
         backend == AttentionFusionBackend::kWebGpu;
}
inline bool AttentionBackendAllowsOnnxCausalForm(
    AttentionFusionBackend backend) {
  return backend == AttentionFusionBackend::kAuto ||
         backend == AttentionFusionBackend::kOnnx;
}

}  // namespace onnxsim
