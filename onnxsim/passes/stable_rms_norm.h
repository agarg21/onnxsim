// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// A decomposed RMSNorm `X / Sqrt(ReduceMean(X * X) + eps)` becomes its
// two-pass, range-safe form:
//
//   r0 = Sqrt(ReduceMean(X * X) + eps)          (the original denominator)
//   u  = Div(X, r0)
//   r1 = Sqrt(ReduceMean(u * u) + Div(eps, Mul(r0, r0)))
//   y  = Div(u, r1)
//
// In real arithmetic mean(u^2) + eps / r0^2 = (mean(X^2) + eps) / r0^2 = 1,
// so y equals the original exactly; the rewrite changes nothing for a float
// backend. It matters for a low-precision backend that runs the chain op by
// op, each op's output quantized to its own calibrated range: X * X's range
// is set by the rarest, largest token (LLM residual streams carry a few
// outlier channels of hundreds to thousands), so the squares of an ordinary
// token fall below one quantization step and their mean -- the RMS -- comes
// out wrong, rescaling the whole normalized output. The second pass sees
// |u| <= sqrt(N) (N the normalized width) up to the first pass's own error,
// whatever the token, so its u * u keeps its precision, and it corrects the
// coarse first pass (r1 absorbs its error). The first pass must not collapse
// (every square below one step, r0 = sqrt(eps)); at the proportions below it
// was off by 10-18%, not more. Measured on SmolLM2-135M run op by op in
// 16-bit (U16) on an AX650 NPU: the one-pass RMS was ~10% off at layers 3..7
// (x^2 range ~2.6e6 against a typical mean of ~10-16), the two-pass one
// within 0.1%, which took the decode's logit error from ~6% to ~2-3.5%.
//
// Matched spellings are fuse_rms_norm.h's (`FuseRMSNorm::MatchNorm`): the
// square is `Mul(X, X)` or `Pow(X, 2)`, the ReduceMean is over the last axis
// with keepdims, eps is a constant scalar, and the normalization is
// `Div(X, Sqrt(..))`, `Mul(X, Reciprocal(Sqrt(..)))` or
// `Mul(X, Div(1, Sqrt(..)))`, every intermediate single-use. Any following
// gain Mul is left as it is (the rewrite replaces the normalized value), so
// gain-less norms are covered too, at any opset. The rewritten graph does
// not match again: its first Sqrt now has several uses and the second
// pass's eps is not a constant.
//
// A pure graph-shape rewrite that adds nodes, so it is `PassType::Other` and
// never runs by default. Opt in with `extra_optimizers=["stable_rms_norm"]`
// (Python) or `--enable-optimization stable_rms_norm` (CLI). Run it without
// `fuse_rms_norm` (`skipped_optimizers=["fuse_rms_norm"]`) on opset >= 23
// models, where that pass would otherwise fuse the chain into one
// RMSNormalization first.

#include <string>
#include <vector>

#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/fuse_rms_norm.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

struct StableRMSNorm final : public PredicateBasedPass {
  explicit StableRMSNorm()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override { return "stable_rms_norm"; }

  // The matched chain's pieces: X, the first-pass Sqrt (r0), its Add (eps),
  // and its ReduceMean (axes / keepdims to copy).
  struct Match {
    Value* x = nullptr;
    Node* sqrt = nullptr;
    Node* add = nullptr;
    Node* mean = nullptr;
  };

  static bool MatchAt(Node* n, Match& m) {
    if (!(CheckKind(n, kDiv) || CheckKind(n, kMul)) ||
        n->outputs().size() != 1) {
      return false;
    }
    Value* x;
    int64_t axis;
    float eps;
    std::vector<Node*> chain;
    if (!FuseRMSNorm::MatchNorm(n->output(), x, axis, eps, chain)) {
      return false;
    }
    m = Match{};
    m.x = x;
    for (Node* c : chain) {
      if (c->kind() == kSqrt) m.sqrt = c;
      if (c->kind() == kAdd) m.add = c;
      if (c->kind() == kReduceMean) m.mean = c;
    }
    return m.sqrt != nullptr && m.add != nullptr && m.mean != nullptr;
  }

  bool patternMatchPredicate(Node* n) override {
    Match m;
    return MatchAt(n, m);
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    Match m;
    if (!MatchAt(n, m)) {
      return false;
    }
    Value* r0 = m.sqrt->output();
    Value* eps = CheckKind(m.add->input(0), kReduceMean) ? m.add->input(1)
                                                         : m.add->input(0);
    Value* out = n->output();

    // type and shape only: a created value already has its own fresh name
    auto like = [](Value* v, Value* from) {
      v->setElemType(from->elemType());
      if (from->has_sizes()) {
        v->setSizes(from->sizes());
      }
      return v;
    };
    auto make = [&](NodeKind kind, std::vector<Value*> ins, Value* from) {
      Node* node = graph.create(kind, 1);
      for (Value* v : ins) node->addInput(v);
      like(node->output(), from);
      node->insertBefore(n);
      return node->output();
    };

    Value* u = make(kDiv, {m.x, r0}, out);
    Value* uu = make(kMul, {u, u}, out);
    Node* mean2 = graph.create(kReduceMean, 1);
    mean2->copyAttributes(*m.mean);
    mean2->addInput(uu);
    for (size_t i = 1; i < m.mean->inputs().size(); ++i) {
      mean2->addInput(m.mean->input(i));  // axes as an input (opset >= 18)
    }
    like(mean2->output(), m.mean->output());
    mean2->insertBefore(n);
    Value* rr = make(kMul, {r0, r0}, r0);
    Value* e2 = make(kDiv, {eps, rr}, r0);
    Value* a2 = make(kAdd, {mean2->output(), e2}, r0);
    Value* r1 = make(kSqrt, {a2}, r0);
    // replaceAllUsesWith keeps a graph output's name on the new value
    Value* y = make(kDiv, {u, r1}, out);

    if (!tryReplacingAllUsesWith(out, y)) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
