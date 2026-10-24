// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// N >= 2 MatMuls that read the same activation with constant 2-D weights
// `[K, N_i]` (same K and element type, any N_i) become one MatMul against the
// weights concatenated on their last axis, followed by a Split into the N_i
// column blocks:
//
//   y_i = MatMul(x, W_i)  ->
//   [y_1..y_n] = Split(MatMul(x, Concat(W_1..W_n, -1)), -1, [N_1..N_n])
//
// Exact: every output element is the same dot product. It is fuse_qkv.h
// generalized: that pass needs exactly three consumers of equal weight shape,
// which misses grouped-query attention (K and V narrower than Q: 576 / 192 /
// 192 in SmolLM2-135M), the MLP's gate/up pair, and any other fan-out of
// linears. One launch instead of N pays off on backends that dispatch op by
// op (each launch costs a fixed overhead); on an AX650 NPU the q/k/v (and
// RoPE-rotated q/k) projections and gate/up merged this way, together with
// fused attention, took the SmolLM2-135M decode from 63 to 16 programs per
// layer. Consumers are grouped by (K, element type); a consumer whose weight
// is not a constant 2-D tensor, whose output is unused, or that reads the
// activation as its second operand is left alone.
//
// The new nodes go immediately before the earliest consumer of the group (so
// after the activation's producer, whatever it is -- a graph input included,
// which fuse_qkv must decline because it anchors after the producer node);
// a weight produced by a node after that point (a Constant placed late)
// declines the group. The concatenated weight is a Concat node that constant
// folding turns into one initializer.
//
// A graph-shape rewrite, so it is `PassType::Other` and never runs by default.
// Opt in with `extra_optimizers=["fuse_parallel_matmuls"]` (Python) or
// `--enable-optimization fuse_parallel_matmuls` (CLI). fuse_qkv (on by
// default) still takes the exact three-equal-weights case first.

#include <string>
#include <unordered_set>
#include <vector>

#include "onnx/defs/tensor_util.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace onnxsim_passes {

struct FuseParallelMatMuls final : public PredicateBasedPass {
  explicit FuseParallelMatMuls()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override { return "fuse_parallel_matmuls"; }

  // A MatMul consumer that can join a group: activation as operand 0, a
  // constant 2-D weight, a used output. Sets its (K, N, element type).
  static bool Eligible(Node* n, int64_t& k, int64_t& cols, int32_t& type) {
    if (!CheckKind(n, kMatMul) || n->inputs().size() != 2 ||
        n->output()->uses().empty() || !IsConstantTensor(n, 1)) {
      return false;
    }
    const Tensor* w = FetchConstantTensor(n->input(1));
    if (w == nullptr || w->sizes().size() != 2) {
      return false;
    }
    k = w->sizes()[0];
    cols = w->sizes()[1];
    type = w->elem_type();
    return true;
  }

  // The consumers of n's activation that group with n (n included), in the
  // activation's use order.
  static std::vector<Node*> Group(Node* n) {
    std::vector<Node*> group;
    int64_t k0, cols;
    int32_t t0;
    if (!Eligible(n, k0, cols, t0)) {
      return group;
    }
    std::unordered_set<Node*> seen;
    for (const Use& use : n->input(0)->uses()) {
      int64_t k;
      int32_t t;
      if (use.offset == 0 && !seen.count(use.user) &&
          Eligible(use.user, k, cols, t) && k == k0 && t == t0) {
        seen.insert(use.user);
        group.push_back(use.user);
      }
    }
    return group;
  }

  bool patternMatchPredicate(Node* n) override { return Group(n).size() >= 2; }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    std::vector<Node*> group = Group(n);
    if (group.size() < 2) {
      return false;
    }
    // earliest member in node order, and the nodes before it
    std::unordered_set<Node*> members(group.begin(), group.end());
    std::unordered_set<Node*> before;
    Node* first = nullptr;
    for (Node* node : graph.nodes()) {
      if (members.count(node)) {
        first = node;
        break;
      }
      before.insert(node);
    }
    if (first == nullptr) {
      return false;
    }
    for (Node* m : group) {
      Node* producer = m->input(1)->node();
      if (producer->kind() != kParam && !before.count(producer)) {
        return false;  // the weight is defined after the insertion point
      }
    }

    Node* cat = graph.create(kConcat, 1);
    for (Node* m : group) cat->addInput(m->input(1));
    cat->i_(kaxis, 1);
    cat->insertBefore(first);

    Node* matmul = graph.create(kMatMul, 1);
    matmul->addInput(n->input(0));
    matmul->addInput(cat->output());
    matmul->output()->setElemType(n->output()->elemType());
    matmul->insertBefore(first);

    std::vector<int64_t> sizes;
    for (Node* m : group) {
      sizes.push_back(FetchConstantTensor(m->input(1))->sizes()[1]);
    }
    Node* split = graph.create("Split"_sym, group.size());
    split->i_(kaxis, -1);
    split->addInput(matmul->output());
    if (getOpsetVersion(graph) >= 13) {
      Tensor t;
      t.sizes().push_back(static_cast<int64_t>(sizes.size()));
      t.elem_type() = ONNX_NAMESPACE::TensorProto_DataType_INT64;
      for (int64_t s : sizes) t.int64s().push_back(s);
      split->addInput(graph.addInitializerAndCreateValue(t));
    } else {
      split->is_(ksplit, std::vector<int64_t>(sizes));
    }
    split->insertBefore(first);

    for (size_t i = 0; i < group.size(); ++i) {
      if (!tryReplacingAllUsesWith(group[i]->output(), split->outputs()[i])) {
        return false;
      }
    }
    // n's output now has no uses; the other members are dead too and are
    // removed by eliminate_deadend (destroying them here would invalidate
    // the pass's node iteration)
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace onnxsim_passes
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
