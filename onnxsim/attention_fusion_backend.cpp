#include "attention_fusion_backend.h"

#include <atomic>
#include <stdexcept>

namespace onnxsim {

namespace {
std::atomic<AttentionFusionBackend> g_attention_fusion_backend{
    AttentionFusionBackend::kAuto};
}  // namespace

void SetAttentionFusionBackend(AttentionFusionBackend backend) {
  g_attention_fusion_backend.store(backend, std::memory_order_relaxed);
}

AttentionFusionBackend GetAttentionFusionBackend() {
  return g_attention_fusion_backend.load(std::memory_order_relaxed);
}

AttentionFusionBackend ParseAttentionFusionBackend(const std::string& name) {
  if (name == "auto") {
    return AttentionFusionBackend::kAuto;
  }
  if (name == "onnx") {
    return AttentionFusionBackend::kOnnx;
  }
  if (name == "ort_contrib") {
    return AttentionFusionBackend::kOrtContrib;
  }
  if (name == "none") {
    return AttentionFusionBackend::kNone;
  }
  if (name == "webgpu") {
    return AttentionFusionBackend::kWebGpu;
  }
  throw std::invalid_argument(
      "unknown attention_fusion_backend '" + name +
      "' (expected 'auto', 'onnx', 'ort_contrib', 'none' or 'webgpu')");
}

}  // namespace onnxsim
