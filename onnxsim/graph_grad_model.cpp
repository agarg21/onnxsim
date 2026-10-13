/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * BuildGradientModel: the gradient of a whole model (graph_grad_model.h).
 *
 * BuildBackward (graph_grad.h) differentiates a slice of nodes into a
 * GraphBuilder and leaves its caller to choose the slice, get the shapes, seed
 * the loss, give an unreached target a zero, inline the model-local functions
 * the templated rules call and make every name unique. This file does those
 * steps once, so a caller can ask for the gradient of a model by name.
 */
#include "graph_grad_model.h"

#include <onnx/inliner/inliner.h>
#include <onnx/onnx_pb.h>
#include <onnx/shape_inference/implementation.h>

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "graph_grad.h"
#include "qat_graph_builder.h"

using onnx::TensorProto;

namespace {

bool IsFloat(int32_t elem_type) {
  return elem_type == TensorProto::FLOAT || elem_type == TensorProto::DOUBLE;
}

}  // namespace

onnx::ModelProto BuildGradientModel(const onnx::ModelProto& forward,
                                    const std::string& loss,
                                    const std::vector<std::string>& wrt,
                                    const std::string& prefix) {
  // Shape inference fills in value_info for every intermediate tensor. Only
  // static shapes are recorded: a dynamic value is an error only if the
  // differentiated slice touches it.
  onnx::ModelProto model = forward;
  onnx::shape_inference::InferShapes(model);
  const onnx::GraphProto& fwd = model.graph();

  std::map<std::string, std::vector<int64_t>> shapes;
  std::map<std::string, int32_t> dtypes;
  auto note = [&](const onnx::ValueInfoProto& vi) {
    if (!vi.type().has_tensor_type()) return;
    const auto& tt = vi.type().tensor_type();
    dtypes[vi.name()] = tt.elem_type();
    if (!tt.has_shape()) return;
    std::vector<int64_t> dims;
    for (const auto& d : tt.shape().dim()) {
      if (!d.has_dim_value()) return;
      dims.push_back(d.dim_value());
    }
    shapes[vi.name()] = dims;
  };
  for (const auto& v : fwd.input()) note(v);
  for (const auto& v : fwd.value_info()) note(v);
  for (const auto& v : fwd.output()) note(v);
  for (const TensorProto& t : fwd.initializer()) {
    dtypes[t.name()] = t.data_type();
    shapes[t.name()] = std::vector<int64_t>(t.dims().begin(), t.dims().end());
  }

  if (!dtypes.count(loss)) {
    throw std::invalid_argument("the loss " + loss +
                                " is not a value of the model");
  }
  if (!shapes.count(loss)) {
    throw std::invalid_argument("the loss " + loss + " has no static shape");
  }
  if (!shapes[loss].empty()) {
    throw std::invalid_argument("the loss " + loss +
                                " must be a scalar, it has rank " +
                                std::to_string(shapes[loss].size()));
  }
  if (!IsFloat(dtypes[loss])) {
    throw std::invalid_argument("the loss " + loss +
                                " must be float32 or float64");
  }
  const int32_t loss_type = dtypes[loss];
  for (const std::string& w : wrt) {
    if (!dtypes.count(w)) {
      throw std::invalid_argument("cannot differentiate with respect to " + w +
                                  ": not a value of the model");
    }
    if (dtypes[w] != loss_type) {
      throw std::invalid_argument("cannot differentiate with respect to " + w +
                                  ": its element type differs from the loss");
    }
    if (!shapes.count(w)) {
      throw std::invalid_argument("cannot differentiate with respect to " + w +
                                  ": it has no static shape");
    }
  }

  // The forward nodes the loss needs, and of those the ones downstream of a
  // wrt: the slice to differentiate. Nodes are in topological order.
  const int num_nodes = fwd.node_size();
  std::set<std::string> needed = {loss};
  std::vector<bool> need(static_cast<size_t>(num_nodes), false);
  for (int k = num_nodes; k-- > 0;) {
    const onnx::NodeProto& nd = fwd.node(k);
    bool used = false;
    for (const std::string& o : nd.output()) used = used || needed.count(o) > 0;
    if (!used) continue;
    need[static_cast<size_t>(k)] = true;
    for (const std::string& i : nd.input()) {
      if (!i.empty()) needed.insert(i);
    }
  }
  std::set<std::string> downstream(wrt.begin(), wrt.end());
  std::vector<onnx::NodeProto> slice;
  for (int k = 0; k < num_nodes; ++k) {
    if (!need[static_cast<size_t>(k)]) continue;
    const onnx::NodeProto& nd = fwd.node(k);
    bool depends = false;
    for (const std::string& i : nd.input()) {
      depends = depends || (!i.empty() && downstream.count(i) > 0);
    }
    if (!depends) continue;
    for (const std::string& o : nd.output()) downstream.insert(o);
    slice.push_back(nd);
  }
  for (const onnx::NodeProto& nd : slice) {
    std::vector<std::string> names(nd.input().begin(), nd.input().end());
    names.insert(names.end(), nd.output().begin(), nd.output().end());
    for (const std::string& n : names) {
      if (!n.empty() && !shapes.count(n)) {
        throw std::invalid_argument("value " + n + " has no static shape, so " +
                                    nd.op_type() + " cannot be differentiated");
      }
    }
  }

  // A wrt the loss does not reach gets a zero gradient below. BuildBackward
  // throws for such a target, so drop it and retry.
  std::vector<std::string> targets;
  for (const std::string& w : wrt) {
    if (needed.count(w)) targets.push_back(w);
  }
  const std::string kNoGradient = "no gradient reaches '";
  std::unique_ptr<GraphBuilder> b;
  std::map<std::string, std::string> got;
  for (;;) {
    b.reset(new GraphBuilder(prefix));
    b->SetFloatType(loss_type);
    std::map<std::string, std::string> seeds;
    seeds[loss] = b->Const(1.0f, "seed");
    try {
      got = BuildBackward(*b, slice, shapes, seeds, targets);
      break;
    } catch (const UnsupportedOpError&) {
      throw;
    } catch (const std::invalid_argument& e) {
      const std::string msg = e.what();
      if (msg.compare(0, kNoGradient.size(), kNoGradient) != 0) throw;
      const size_t end = msg.find('\'', kNoGradient.size());
      const std::string who =
          msg.substr(kNoGradient.size(), end - kNoGradient.size());
      std::vector<std::string> keep;
      for (const std::string& t : targets) {
        if (t != who) keep.push_back(t);
      }
      if (keep.size() == targets.size()) throw;
      targets = keep;
    }
  }

  // One Identity per wrt gives every result a name of its own. A wrt that was
  // not reached is an Expand of a zero to its shape.
  std::vector<std::string> results;
  std::vector<onnx::NodeProto> tail;
  std::vector<TensorProto> tail_init;
  for (size_t k = 0; k < wrt.size(); ++k) {
    const std::string result = prefix + "grad_" + std::to_string(k);
    auto it = got.find(wrt[k]);
    std::string src;
    if (it != got.end()) {
      src = it->second;
    } else {
      GraphBuilder z(prefix + "z" + std::to_string(k) + "_");
      z.SetFloatType(loss_type);
      src = z.Const(0.0f, "zero");
      if (!shapes[wrt[k]].empty()) {
        src = z.Op("Expand", {src, z.ConstInt64(shapes[wrt[k]], "shape")},
                   "zeros");
      }
      for (const auto& nd : z.nodes()) tail.push_back(nd);
      for (const auto& t : z.initializer()) tail_init.push_back(t);
    }
    onnx::NodeProto id;
    id.set_op_type("Identity");
    id.add_input(src);
    id.add_output(result);
    tail.push_back(id);
    results.push_back(result);
  }

  // The templated rules call model-local functions written for opset 17. They
  // are inlined in a model of their own, with the forward values they read
  // declared as inputs, because the inliner does not convert opset versions.
  std::vector<onnx::NodeProto> gnodes(b->nodes().begin(), b->nodes().end());
  if (!b->functions().empty()) {
    onnx::ModelProto gm;
    gm.set_ir_version(model.ir_version());
    onnx::OperatorSetIdProto* base = gm.add_opset_import();
    base->set_domain("");
    base->set_version(17);
    std::set<std::string> domains;
    for (const onnx::FunctionProto& fn : b->functions()) {
      *gm.add_functions() = fn;
      if (domains.insert(fn.domain()).second) {
        onnx::OperatorSetIdProto* o = gm.add_opset_import();
        o->set_domain(fn.domain());
        o->set_version(1);
      }
    }
    onnx::GraphProto* gg = gm.mutable_graph();
    gg->set_name("rules");
    std::set<std::string> made;
    for (const auto& t : b->initializer()) {
      made.insert(t.name());
      *gg->add_initializer() = t;
    }
    for (const auto& nd : b->nodes()) {
      for (const std::string& o : nd.output()) made.insert(o);
    }
    std::set<std::string> declared;
    for (const auto& nd : b->nodes()) {
      for (const std::string& i : nd.input()) {
        if (i.empty() || made.count(i) || !declared.insert(i).second) continue;
        onnx::ValueInfoProto* vi = gg->add_input();
        vi->set_name(i);
        onnx::TypeProto_Tensor* tt = vi->mutable_type()->mutable_tensor_type();
        tt->set_elem_type(dtypes.at(i));
        tt->mutable_shape();
        for (int64_t d : shapes.at(i))
          tt->mutable_shape()->add_dim()->set_dim_value(d);
      }
    }
    for (const auto& nd : b->nodes()) *gg->add_node() = nd;
    onnx::shape_inference::InferShapes(gm);
    onnx::inliner::InlineLocalFunctions(gm, false);
    gnodes.assign(gg->node().begin(), gg->node().end());
  }

  // Every new name starts with the prefix. Inlined bodies bring names such as
  // `tmp`, which may belong to the user's model, so they are prefixed too.
  std::set<std::string> fwd_names;
  for (const auto& v : fwd.input()) fwd_names.insert(v.name());
  for (const auto& t : fwd.initializer()) fwd_names.insert(t.name());
  for (int k = 0; k < num_nodes; ++k) {
    for (const std::string& o : fwd.node(k).output()) fwd_names.insert(o);
  }
  auto fresh = [&](const std::string& name) {
    if (name.empty() || fwd_names.count(name) ||
        name.compare(0, prefix.size(), prefix) == 0) {
      return name;
    }
    return prefix + name;
  };
  for (auto& nd : gnodes) {
    for (std::string& i : *nd.mutable_input()) i = fresh(i);
    for (std::string& o : *nd.mutable_output()) o = fresh(o);
  }
  for (auto& t : b->initializer()) t.set_name(fresh(t.name()));
  for (auto& nd : tail) {
    for (std::string& i : *nd.mutable_input()) i = fresh(i);
  }

  onnx::ModelProto out;
  out.set_ir_version(model.ir_version());
  out.set_producer_name("onnxsim-graph-grad");
  *out.mutable_opset_import() = model.opset_import();
  *out.mutable_functions() = model.functions();
  onnx::GraphProto* og = out.mutable_graph();
  og->set_name("gradient");
  for (const auto& v : fwd.input()) *og->add_input() = v;
  for (const TensorProto& t : fwd.initializer()) *og->add_initializer() = t;
  for (int k = 0; k < num_nodes; ++k) {
    if (need[static_cast<size_t>(k)]) *og->add_node() = fwd.node(k);
  }
  for (const auto& nd : gnodes) *og->add_node() = nd;
  for (const auto& t : b->initializer()) *og->add_initializer() = t;
  for (const auto& nd : tail) *og->add_node() = nd;
  for (const auto& t : tail_init) *og->add_initializer() = t;
  for (size_t k = 0; k < wrt.size(); ++k) {
    onnx::ValueInfoProto* vi = og->add_output();
    vi->set_name(results[k]);
    onnx::TypeProto_Tensor* tt = vi->mutable_type()->mutable_tensor_type();
    tt->set_elem_type(dtypes[wrt[k]]);
    tt->mutable_shape();
    for (int64_t d : shapes[wrt[k]])
      tt->mutable_shape()->add_dim()->set_dim_value(d);
  }
  return out;
}
