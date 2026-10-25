// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Removes the `Where(IsNaN(P), 0, P)` guard that PyTorch's
// `scaled_dot_product_attention` export puts after the attention Softmax,
//
//   P = Softmax(scores + mask)        (axis = last)
//   out = Where(IsNaN(P), 0, P)
//
// where `scores` is the scaled Q@K^T (a MatMul, or a Mul/Div of one). The
// guard exists for one case: a row whose keys are all masked with -inf, where
// Softmax computes 0/0 = NaN and the guard turns that row into zeros. Some
// backends have no `IsNaN` at all (Axera's Pulsar2 frontend rejects it:
// `KeyError('dont support IsNaN opr ...')`), so this pass rewrites the guard
// away in the two cases where that is exact:
//
//  1. The mask is provably finite: a constant with only finite values, an
//     integer/bool tensor cast to float, or a `Where` choosing between such
//     values, possibly behind shape-only ops (Expand, Reshape, Unsqueeze,
//     ...). This is what the dynamo exporter emits (`Where(m, 0, finfo.min)`),
//     and unmasked attention (no Add at all) is covered too. Softmax of a
//     finite row never produces NaN, so the guard selects P everywhere and is
//     dropped.
//
//  2. The mask is `Where(c, a, b)` with one branch the scalar -inf and the
//     other a provably finite scalar -- the legacy (TorchScript) exporter's
//     form. Then -inf is replaced by finfo.min (float32 lowest) and the guard
//     by an explicit row-validity multiply:
//
//       keep  = c                    (or Not(c) when -inf is the true branch)
//       valid = ReduceMax(Cast(keep, float), axes=[-1], keepdims=1)
//       out   = Softmax(scores + Where(c, a, finfo.min)) * valid
//
//     For a row with at least one kept key, the masked logits are
//     score + finfo.min, whose exp() underflows to exactly 0 against any kept
//     logit, just as exp(-inf) does, so Softmax is bit-identical. For an
//     all-masked row the original produced NaN -> 0; the rewrite produces a
//     finite Softmax times valid = 0 -> 0. The all-padding row keeps its
//     zeros: unlike swapping -inf for finfo.min alone, which would turn it
//     into a uniform average over the padding.
//
// Both rewrites assume the attention scores themselves are finite (and above
// about -1e31, where score + finfo.min would round to -inf): with NaN/Inf
// already in Q or K, the original guard would zero a NaN row that the
// rewritten graph propagates. That is the same assumption an export without
// SDPA (eager attention, which has no guard) makes, and why this pass is not
// on by default: it is a backend-legalization rewrite. Opt in with
// `extra_optimizers=["eliminate_sdpa_nan_guard"]` (Python) or
// `--enable-optimization eliminate_sdpa_nan_guard` (CLI).
//
// Only float32 is handled (what the exporters emit); fuse_attention, when it
// runs first, absorbs the guard into the fused node instead, so the pattern
// is simply not found then.

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "passes/quantize_conv_common.h"  // ReadFloatTensorFlat

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

struct EliminateSdpaNanGuard final : public PredicateBasedPass {
  explicit EliminateSdpaNanGuard()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "eliminate_sdpa_nan_guard";
  }

  static bool IsIntegerOrBool(int32_t t) {
    switch (t) {
      case TensorProto_DataType_BOOL:
      case TensorProto_DataType_INT8:
      case TensorProto_DataType_UINT8:
      case TensorProto_DataType_INT16:
      case TensorProto_DataType_UINT16:
      case TensorProto_DataType_INT32:
      case TensorProto_DataType_UINT32:
      case TensorProto_DataType_INT64:
      case TensorProto_DataType_UINT64:
        return true;
      default:
        return false;
    }
  }

  // The float32 values of a constant `v`, or false when `v` is not a float32
  // constant.
  static bool ConstFloats(const Value* v, std::vector<float>& out) {
    const Tensor* t = FetchConstantTensor(v);
    if (t == nullptr || t->elem_type() != TensorProto_DataType_FLOAT) {
      return false;
    }
    out = ReadFloatTensorFlat(*t);
    return true;
  }

  // Whether every value `v` can take is finite, whatever the graph inputs
  // are. Conservative: arithmetic (Add/Mul/...) is never trusted, since it
  // can overflow.
  static bool ProvablyFinite(const Value* v, int depth = 16) {
    if (depth <= 0) {
      return false;
    }
    if (IsIntegerOrBool(v->elemType())) {
      return true;
    }
    if (const Tensor* t = FetchConstantTensor(v)) {
      if (IsIntegerOrBool(t->elem_type())) {
        return true;
      }
      std::vector<float> f;
      if (!ConstFloats(v, f)) {
        return false;
      }
      for (float x : f) {
        if (!std::isfinite(x)) {
          return false;
        }
      }
      return true;
    }
    const Node* n = v->node();
    const NodeKind k = n->kind();
    if (k == kCast) {
      // Any integer fits float32/float64 (|int64| < 2^63 << FLT_MAX).
      if (!n->hasAttribute(kto) || n->inputs().size() != 1) {
        return false;
      }
      const int64_t to = n->i(kto);
      return (to == TensorProto_DataType_FLOAT ||
              to == TensorProto_DataType_DOUBLE) &&
             IsIntegerOrBool(n->input(0)->elemType());
    }
    if (k == Symbol("Where")) {
      return n->inputs().size() == 3 &&
             ProvablyFinite(n->input(1), depth - 1) &&
             ProvablyFinite(n->input(2), depth - 1);
    }
    if (k == kConcat) {
      for (const Value* in : n->inputs()) {
        if (!ProvablyFinite(in, depth - 1)) {
          return false;
        }
      }
      return !n->inputs().empty();
    }
    // Ops whose output values are a selection/rearrangement of input 0's.
    static const char* const kSelect[] = {
        "Expand",         "Reshape",  "Unsqueeze", "Squeeze", "Transpose",
        "Identity",       "Flatten",  "Tile",      "Slice",   "Gather",
        "GatherElements", "GatherND", "Neg",       "Abs"};
    for (const char* s : kSelect) {
      if (k == Symbol(s)) {
        return !n->inputs().empty() && ProvablyFinite(n->input(0), depth - 1);
      }
    }
    return false;
  }

  // The scaled Q@K^T operand: MatMul, or Mul/Div of one (the 1/sqrt(d) scale
  // applied after the MatMul).
  static bool IsScores(const Value* v) {
    const NodeKind k = v->node()->kind();
    if (k == kMatMul) {
      return true;
    }
    if (k == kMul || k == kDiv) {
      const Node* n = v->node();
      return n->inputs().size() == 2 &&
             (n->input(0)->node()->kind() == kMatMul ||
              (k == kMul && n->input(1)->node()->kind() == kMatMul));
    }
    return false;
  }

  // Whether `v` is a one-element float32 constant that is -inf
  // (`neg_inf`) or finite (otherwise).
  static bool IsScalar(const Value* v, bool neg_inf) {
    std::vector<float> f;
    if (!ConstFloats(v, f) || f.size() != 1) {
      return false;
    }
    return neg_inf ? std::isinf(f[0]) && f[0] < 0.0f : std::isfinite(f[0]);
  }

  bool patternMatchPredicate(Node* node) override {
    if (node->kind() != Symbol("Where") || node->inputs().size() != 3) {
      return false;
    }
    const Value* isnan = node->input(0);
    const Value* p = node->input(2);
    return isnan->node()->kind() == Symbol("IsNaN") &&
           isnan->node()->inputs().size() == 1 &&
           isnan->node()->input(0) == p && p->node()->kind() == kSoftmax &&
           p->elemType() == TensorProto_DataType_FLOAT &&
           FetchConstantTensor(node->input(1)) != nullptr;
  }

  // Whether Softmax `sm` normalizes over the last axis.
  static bool SoftmaxOverLastAxis(const Node* sm, const Graph& graph) {
    const Value* x = sm->input(0);
    const int opset = getOpsetVersion(graph);
    int64_t axis = opset >= 13 || opset == 0 ? -1 : 1;
    if (sm->hasAttribute(kaxis)) {
      axis = sm->i(kaxis);
    }
    if (axis == -1) {
      return true;
    }
    return x->has_sizes() &&
           axis == static_cast<int64_t>(x->sizes().size()) - 1;
  }

  bool runTransform(Node* guard, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    Value* p = guard->input(2);
    Node* sm = p->node();
    Value* logits = sm->input(0);

    // Case 1: Softmax input finite given finite scores.
    bool finite = false;
    Node* add = nullptr;
    int mask_idx = -1;
    if (IsScores(logits)) {
      finite = true;  // unmasked
    } else if (logits->node()->kind() == kAdd &&
               logits->node()->inputs().size() == 2) {
      add = logits->node();
      for (int i = 0; i < 2; ++i) {
        if (IsScores(add->input(1 - i))) {
          mask_idx = i;
          break;
        }
      }
      if (mask_idx >= 0 && ProvablyFinite(add->input(mask_idx))) {
        finite = true;
      }
    }
    if (finite) {
      guard->output()->replaceAllUsesWith(p);
      destroy_current = NodeDestroyType::DestroyOne;
      return true;
    }

    // Case 2: Where(c, a, b) mask with one -inf scalar branch. This changes
    // P itself (an all-masked row becomes finite), so P and IsNaN(P) must
    // feed only the guard.
    if (add == nullptr || mask_idx < 0 || logits->uses().size() != 1 ||
        p->uses().size() != 2 || guard->input(0)->uses().size() != 1 ||
        !SoftmaxOverLastAxis(sm, graph)) {
      return false;
    }
    Value* mask = add->input(mask_idx);
    Node* where = mask->node();
    if (where->kind() != Symbol("Where") || where->inputs().size() != 3) {
      return false;
    }
    Value* cond = where->input(0);
    int inf_branch = -1;
    for (int b = 1; b <= 2; ++b) {
      if (IsScalar(where->input(b), /*neg_inf=*/true) &&
          IsScalar(where->input(3 - b), /*neg_inf=*/false)) {
        inf_branch = b;
      }
    }
    if (inf_branch < 0 || cond->elemType() != TensorProto_DataType_BOOL ||
        !cond->has_sizes() || cond->sizes().empty()) {
      return false;
    }
    // The scalar branches must not raise the mask's rank above cond's, so
    // cond's last axis is the key axis the Softmax normalizes over.
    for (int b = 1; b <= 2; ++b) {
      const Value* br = where->input(b);
      if (!br->has_sizes() || br->sizes().size() > cond->sizes().size()) {
        return false;
      }
    }

    // finfo.min branch, as a fresh initializer (the -inf one may be shared).
    Tensor lowest;
    lowest.elem_type() = TensorProto_DataType_FLOAT;
    lowest.sizes() = FetchConstantTensor(where->input(inf_branch))->sizes();
    lowest.floats().push_back(std::numeric_limits<float>::lowest());
    Value* lowest_v = graph.addInitializerAndCreateValue(lowest);

    Node* new_where = graph.create(Symbol("Where"), 1);
    new_where->addInput(cond);
    new_where->addInput(inf_branch == 1 ? lowest_v : where->input(1));
    new_where->addInput(inf_branch == 2 ? lowest_v : where->input(2));
    new_where->output()->setElemType(TensorProto_DataType_FLOAT);
    if (mask->has_sizes()) {
      new_where->output()->setSizes(mask->sizes());
    }
    new_where->insertBefore(add);
    add->replaceInput(mask_idx, new_where->output());

    // valid = ReduceMax(Cast(keep, float), axes=[-1], keepdims=1)
    Value* keep = cond;
    if (inf_branch == 1) {  // -inf where cond is true: keep = Not(cond)
      Node* not_n = graph.create(Symbol("Not"), 1);
      not_n->addInput(cond);
      not_n->output()->setElemType(TensorProto_DataType_BOOL);
      not_n->output()->setSizes(cond->sizes());
      not_n->insertBefore(add);
      keep = not_n->output();
    }
    Node* cast = graph.create(kCast, 1);
    cast->addInput(keep);
    cast->i_(kto, TensorProto_DataType_FLOAT);
    cast->output()->setElemType(TensorProto_DataType_FLOAT);
    cast->output()->setSizes(cond->sizes());
    cast->insertBefore(add);

    Node* reduce = graph.create(Symbol("ReduceMax"), 1);
    reduce->addInput(cast->output());
    reduce->i_(kkeepdims, int64_t{1});
    if (getOpsetVersion(graph) >= 18) {
      Tensor axes_t;
      axes_t.elem_type() = TensorProto_DataType_INT64;
      axes_t.sizes().push_back(1);
      axes_t.int64s().push_back(-1);
      reduce->addInput(graph.addInitializerAndCreateValue(axes_t));
    } else {
      reduce->is_(kaxes, std::vector<int64_t>{-1});
    }
    reduce->output()->setElemType(TensorProto_DataType_FLOAT);
    std::vector<Dimension> rsizes = cond->sizes();
    rsizes.back() = Dimension(1);
    reduce->output()->setSizes(rsizes);
    reduce->insertBefore(add);

    Node* mul = graph.create(kMul, 1);
    mul->addInput(p);
    mul->addInput(reduce->output());
    mul->output()->setElemType(TensorProto_DataType_FLOAT);
    if (p->has_sizes()) {
      mul->output()->setSizes(p->sizes());
    }
    mul->insertBefore(guard);
    guard->output()->replaceAllUsesWith(mul->output());
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
