// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Fuses a residual Add that feeds a last-axis LayerNormalization into one
// ONNX Runtime contrib op, `com.microsoft::SkipLayerNormalization`:
//
//   s = Add(x, skip)                       // x, skip: non-constant, same shape
//   y = LayerNormalization(s, gamma[, beta], axis=-1, epsilon=e)
//     ==>
//   y = SkipLayerNormalization(x, skip, gamma[, beta], epsilon=e)
//
// and the bias variant, where a [C] constant is added to the residual sum
// before the norm:
//
//   s = Add(Add(x, skip), bias)            // bias: constant [C]
//   y = LayerNormalization(s, gamma[, beta], axis=-1)
//     ==>
//   y = SkipLayerNormalization(x, skip, gamma[, beta], bias, epsilon=e)
//
// This is a memory-bound inter-op fusion: the Add output is no longer
// written to memory and read back by the norm.
//
// Deliberately conservative, so the emitted node only uses the subset of
// SkipLayerNormalization's schema this pass is certain of:
//   * Opset >= 17 (LayerNormalization is in the default domain from 17).
//   * LayerNormalization's axis is the last axis (-1, or 2 for rank 3).
//   * Only the Y output of LayerNormalization is consumed. Its optional
//     Mean/InvStdDev outputs must be unused.
//   * The Add output (and, for the bias variant, the inner Add output) has
//     exactly one use -- that norm (or the bias Add) -- and is not a graph
//     output. Otherwise the pass declines.
//   * x and skip are FLOAT, rank 3, with statically known and identical
//     shapes (SkipLayerNormalization's kernel takes [B, S, H] inputs; no
//     broadcasting of skip is attempted).
//   * The bias variant needs a beta input, since SkipLayerNormalization's
//     positional inputs put beta before bias.
//   * The emitted node carries an explicit `epsilon` equal to the norm's
//     own epsilon. (SkipLayerNormalization's own default epsilon differs
//     from LayerNormalization's, so omitting it would change the numerics.)
//
// The pass is PassType::Other, so it is never in onnxsim's default pass
// set: a default simplify() run leaves the graph unchanged. It is opted into
// by name, through the same `extra_optimizers` plumbing that
// defuse_matmul_integer_to_float and the batched-Gemm rewrite use:
// simplify(..., extra_optimizers=["fuse_skip_layer_norm"]).
//
// The output node's domain is "com.microsoft". On first fire the pass adds
// that domain to the graph's opset_import (version 1), the same way
// fuse_attention.h does, and the result needs a runtime that implements
// SkipLayerNormalization (ONNX Runtime's CPU and CUDA EPs do).

#pragma once

#include <vector>

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

struct FuseSkipLayerNorm final : public PredicateBasedPass {
  explicit FuseSkipLayerNorm()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override { return "fuse_skip_layer_norm"; }

  struct Match {
    Value* input = nullptr;  // x
    Value* skip = nullptr;   // skip (same shape as x)
    Value* gamma = nullptr;  // LayerNormalization's Scale
    Value* beta = nullptr;   // LayerNormalization's B, may be null
    Value* bias = nullptr;   // constant [C] bias, may be null
    float epsilon = 1e-5f;
    // The Add output the norm consumes (the one whose only use is the norm).
    Value* sum_out = nullptr;
    // Nodes made dead by the fusion, outermost (closest to the norm) first,
    // so each one has no uses left by the time it is destroyed.
    std::vector<Node*> dead_chain;
  };

  static bool IsFloatStatic(Value* v) {
    return v->elemType() == TensorProto_DataType_FLOAT;
  }

  // Rank-3 with every dimension statically known.
  static bool IsStaticRank3(Value* v) {
    if (!v->has_sizes() || v->sizes().size() != 3) {
      return false;
    }
    for (const auto& d : v->sizes()) {
      if (!d.is_int) {
        return false;
      }
    }
    return true;
  }

  static bool SameStaticShape(Value* a, Value* b) {
    if (!IsStaticRank3(a) || !IsStaticRank3(b)) {
      return false;
    }
    for (size_t i = 0; i < 3; ++i) {
      if (a->sizes()[i].dim != b->sizes()[i].dim) {
        return false;
      }
    }
    return true;
  }

  static bool IsGraphOutputValue(Graph& graph, Value* v) {
    for (const Value* out : graph.outputs()) {
      if (out == v) {
        return true;
      }
    }
    return false;
  }

  // An Add in the default domain whose output has exactly one use and is not
  // a graph output.
  static bool IsSingleUseAdd(Graph& graph, Value* v) {
    Node* add = v->node();
    return add->kind() == kAdd &&
           (!add->has_domain() || add->domain() == "" ||
            add->domain() == "ai.onnx") &&
           add->inputs().size() == 2 && v->uses().size() == 1 &&
           !IsGraphOutputValue(graph, v);
  }

  static bool TryMatch(Node* ln, Match& m) {
    if (ln->kind() != Symbol("LayerNormalization") ||
        (ln->has_domain() && !ln->domain().empty() &&
         ln->domain() != "ai.onnx")) {
      return false;
    }
    if (ln->inputs().size() < 2 || ln->inputs().size() > 3 ||
        ln->outputs().empty()) {
      return false;
    }
    // LayerNormalization is only in the default domain from opset 17.
    Graph& graph = *ln->owningGraph();
    if (getOpsetVersion(graph) < 17) {
      return false;
    }
    // Only Y may be consumed; the optional Mean/InvStdDev outputs are not
    // part of SkipLayerNormalization's emitted form.
    for (size_t i = 1; i < ln->outputs().size(); ++i) {
      if (ln->outputs()[i]->uses().size() != 0) {
        return false;
      }
    }
    const int64_t axis =
        GetValueFromAttrWithDefault<int64_t>(ln, Symbol("axis"), -1);

    Value* sum_out = ln->input(0);
    if (!IsSingleUseAdd(graph, sum_out)) {
      return false;
    }
    Node* sum_n = sum_out->node();
    if (sum_n->owningGraph() != ln->owningGraph()) {
      return false;
    }
    Value* lhs = sum_n->input(0);
    Value* rhs = sum_n->input(1);
    const bool lhs_const = FetchConstantTensor(lhs) != nullptr;
    const bool rhs_const = FetchConstantTensor(rhs) != nullptr;

    Value* x = nullptr;
    Value* skip = nullptr;
    Value* bias = nullptr;
    std::vector<Node*> chain;
    if (!lhs_const && !rhs_const) {
      // Plain residual: Add(x, skip).
      x = lhs;
      skip = rhs;
      chain = {sum_n};
    } else if (lhs_const != rhs_const) {
      // Bias variant: Add(Add(x, skip), bias) with a constant [C] bias.
      Value* inner_out = lhs_const ? rhs : lhs;
      Value* bias_v = lhs_const ? lhs : rhs;
      if (!IsSingleUseAdd(graph, inner_out)) {
        return false;
      }
      Node* inner_n = inner_out->node();
      if (inner_n->owningGraph() != ln->owningGraph() ||
          FetchConstantTensor(inner_n->input(0)) != nullptr ||
          FetchConstantTensor(inner_n->input(1)) != nullptr) {
        return false;
      }
      x = inner_n->input(0);
      skip = inner_n->input(1);
      bias = bias_v;
      chain = {sum_n, inner_n};
    } else {
      return false;  // Add(const, const) or Add(x, const) without a skip.
    }

    if (!IsFloatStatic(x) || !IsFloatStatic(skip) ||
        !SameStaticShape(x, skip)) {
      return false;
    }
    if (!IsStaticRank3(x)) {
      return false;
    }
    const int64_t hidden = x->sizes()[2].dim;
    if (axis != -1 && axis != 2) {
      return false;
    }
    if (bias != nullptr) {
      const Tensor* bias_t = FetchConstantTensor(bias);
      if (bias_t == nullptr || bias_t->sizes().size() != 1 ||
          bias_t->sizes()[0] != hidden) {
        return false;
      }
      // Positional inputs: beta comes before bias, so the bias variant
      // needs beta present.
      if (ln->inputs().size() < 3) {
        return false;
      }
    }

    m.input = x;
    m.skip = skip;
    m.gamma = ln->input(1);
    m.beta = ln->inputs().size() == 3 ? ln->input(2) : nullptr;
    m.bias = bias;
    m.epsilon =
        GetValueFromAttrWithDefault<float>(ln, Symbol("epsilon"), 1e-5f);
    m.sum_out = sum_out;
    m.dead_chain = chain;
    return true;
  }

  bool patternMatchPredicate(Node* n) override {
    Match m;
    return TryMatch(n, m);
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    Match m;
    if (!TryMatch(n, m)) {
      return false;
    }
    ONNX_ASSERT(!m.dead_chain.empty());

    Node* sln = graph.create(Symbol("SkipLayerNormalization"), 1);
    sln->addInput(m.input);
    sln->addInput(m.skip);
    sln->addInput(m.gamma);
    if (m.beta != nullptr) {
      sln->addInput(m.beta);
    }
    if (m.bias != nullptr) {
      sln->addInput(m.bias);
    }
    sln->f_(Symbol("epsilon"), m.epsilon);
    sln->setDomain("com.microsoft");
    sln->outputs()[0]->copyMetadata(n->outputs()[0]);
    sln->insertBefore(n);

    bool has_ms_domain = false;
    for (const OpSetID& opset : graph.opset_versions_mutable()) {
      if (opset.domain() == "com.microsoft") {
        has_ms_domain = true;
        break;
      }
    }
    if (!has_ms_domain) {
      graph.opset_versions_mutable().emplace_back("com.microsoft", 1);
    }

    n->outputs()[0]->replaceAllUsesWith(sln->outputs()[0]);

    // `n` itself is destroyed by the driving iterator (DestroyOne). Detach its
    // data edge into the Add chain first, then tear the chain down outermost
    // first, so each node has no uses left when it is destroyed.
    for (size_t i = 0; i < n->inputs().size(); ++i) {
      if (n->input(i) == m.sum_out) {
        n->removeInput(i);
        break;
      }
    }
    for (Node* dead : m.dead_chain) {
      dead->destroy();
    }

    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
