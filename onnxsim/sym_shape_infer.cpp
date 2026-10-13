#include "sym_shape_infer.h"

#include <set>
#include <string>

#include "shaping_rules.h"

namespace onnxsim {
namespace {

std::optional<int64_t> ConcreteInt(const SymExpr& e) {
  if (e.is_symbolic()) return std::nullopt;
  return e.to_int();
}

// Ops whose output(0) shape equals input(0) shape. Elementwise ops with extra
// data inputs (Clip's min/max, PRelu's slope, LayerNormalization's scale/bias)
// still take their shape from input(0); so do the in-place scatters
// (ScatterND/ScatterElements), whose output has the shape of `data` (input 0)
// -- this is what lets M2 shape the causal-mask subgraph in encoder/decoder
// models like BART instead of stalling at Shape(ScatterND(...)).
bool IsUnaryShapePreserving(const std::string& op) {
  static const std::set<std::string> kUnary = {
      "Relu",
      "LeakyRelu",
      "PRelu",
      "Sigmoid",
      "HardSigmoid",
      "HardSwish",
      "Tanh",
      "Erf",
      "Gelu",
      "Exp",
      "Log",
      "Sqrt",
      "Reciprocal",
      "Neg",
      "Abs",
      "Sign",
      "Softmax",
      "LogSoftmax",
      "Hardmax",
      "Softplus",
      "Softsign",
      "Elu",
      "Selu",
      "Celu",
      "Mish",
      "Cast",
      "CastLike",
      "Identity",
      "Clip",
      "Round",
      "Floor",
      "Ceil",
      "Not",
      "Shrink",
      "ThresholdedRelu",
      "LayerNormalization",
      "BatchNormalization",
      "InstanceNormalization",
      "GroupNormalization",
      "RMSNormalization",
      "SimplifiedLayerNormalization",
      "Dropout",
      "LpNormalization",
      "QuantizeLinear",
      "DequantizeLinear",
      "ScatterND",
      "ScatterElements",
  };
  return kUnary.count(op) != 0;
}

// Ops that broadcast their (n) inputs to a common shape.
bool IsBroadcastNary(const std::string& op) {
  static const std::set<std::string> kBroadcast = {
      "Add",         "Sub",        "Mul",       "Div",        "Pow",
      "Mod",         "Max",        "Min",       "Mean",       "Sum",
      "Greater",     "Less",       "Equal",     "And",        "Or",
      "Xor",         "BitwiseAnd", "BitwiseOr", "BitwiseXor", "GreaterOrEqual",
      "LessOrEqual",
  };
  return kBroadcast.count(op) != 0;
}

// The dispatch layer: reads each node's operands and attributes out of the
// SymGraph, then calls the pure rule in shaping_rules.h. The rules themselves
// (broadcasting, Reshape's 0/-1, Conv/pool output sizes, ...) live there.
class ShapeInferer {
 public:
  explicit ShapeInferer(const ShapeGraph& graph) : graph_(graph) {
    for (const auto& [name, shape] : graph.value_info) shapes_[name] = shape;
  }

  std::map<std::string, SymShape> Run() {
    for (const SymNode& node : graph_.node) {
      auto shape = Infer(node);
      if (shape && !node.output.empty() && !node.output[0].empty()) {
        shapes_[node.output[0]] = std::move(*shape);
      }
    }
    return shapes_;
  }

 private:
  const SymShape* Shape(const std::string& name) const {
    if (name.empty()) return nullptr;
    auto it = shapes_.find(name);
    return it == shapes_.end() ? nullptr : &it->second;
  }

  // A copy of the known shape of `name`, or nullopt if it is unknown.
  std::optional<SymShape> Operand(const std::string& name) const {
    const SymShape* s = Shape(name);
    if (!s) return std::nullopt;
    return *s;
  }

  std::optional<int64_t> IntAttr(const SymNode& node, const std::string& name,
                                 int64_t fallback) const {
    if (const SymAttr* a = node.attr(name)) {
      if (a->i) return *a->i;
    }
    return fallback;
  }

  // The int list from attribute `attr_name`, else from initializer input
  // `input_index` (opset >= 13 moved axes/starts/... to data inputs). Requires
  // every element concrete.
  std::optional<std::vector<int64_t>> IntListOperand(
      const SymNode& node, const std::string& attr_name,
      std::size_t input_index) const {
    if (const SymAttr* a = node.attr(attr_name)) return a->ints;
    if (input_index < node.input.size()) {
      auto it = graph_.initializer.find(node.input[input_index]);
      if (it != graph_.initializer.end()) {
        std::vector<int64_t> out;
        for (const SymExpr& e : it->second.data) {
          auto c = ConcreteInt(e);
          if (!c) return std::nullopt;
          out.push_back(*c);
        }
        return out;
      }
    }
    return std::nullopt;
  }

  // Conv/pool attributes as plain values. An absent attribute becomes an empty
  // vector, which the rules read as the per-axis default.
  shaping::WindowAttrs WindowAttrsOf(const SymNode& node) const {
    shaping::WindowAttrs a;
    if (const SymAttr* s = node.attr("strides")) a.strides = s->ints;
    if (const SymAttr* d = node.attr("dilations")) a.dilations = d->ints;
    if (const SymAttr* p = node.attr("pads")) a.pads = p->ints;
    a.ceil_mode = IntAttr(node, "ceil_mode", 0).value() != 0;
    return a;
  }

  // --- dispatch -----------------------------------------------------------

  std::optional<SymShape> Infer(const SymNode& node) {
    const std::string& op = node.op_type;
    if (op == "MatMul") return InferMatMul(node);
    if (op == "Gemm") return InferGemm(node);
    if (op == "Conv") return InferConvPool(node, /*is_pool=*/false);
    if (op == "MaxPool" || op == "AveragePool" || op == "LpPool")
      return InferConvPool(node, /*is_pool=*/true);
    if (op == "GlobalAveragePool" || op == "GlobalMaxPool" ||
        op == "GlobalLpPool")
      return InferGlobalPool(node);
    if (op == "Transpose") return InferTranspose(node);
    if (op == "Concat") return InferConcat(node);
    if (op == "Flatten") return InferFlatten(node);
    if (op == "Reshape") return InferReshape(node);
    if (op == "Squeeze") return InferSqueeze(node);
    if (op == "Unsqueeze") return InferUnsqueeze(node);
    if (op == "Expand") return InferExpand(node);
    if (op == "Gather") return InferGather(node);
    if (op == "Slice") return InferSlice(node);
    if (op.rfind("Reduce", 0) == 0) return InferReduce(node);
    if (op == "Where") return InferWhere(node);
    if (IsUnaryShapePreserving(op)) {
      if (const SymShape* s = Shape(node.input.empty() ? "" : node.input[0]))
        return *s;
      return std::nullopt;
    }
    if (IsBroadcastNary(op)) return InferBroadcast(node);
    return std::nullopt;
  }

  std::optional<SymShape> InferBroadcast(const SymNode& node) {
    std::vector<SymShape> ins;
    for (const std::string& name : node.input) {
      if (name.empty()) continue;
      auto s = Operand(name);
      if (!s) return std::nullopt;
      ins.push_back(std::move(*s));
    }
    if (ins.empty()) return std::nullopt;
    return shaping::Broadcast(ins, fresh_);
  }

  std::optional<SymShape> InferWhere(const SymNode& node) {
    if (node.input.size() < 3) return std::nullopt;
    std::vector<SymShape> ins;
    for (int i = 0; i < 3; ++i) {
      auto s = Operand(node.input[i]);
      if (!s) return std::nullopt;
      ins.push_back(std::move(*s));
    }
    return shaping::Broadcast(ins, fresh_);
  }

  std::optional<SymShape> InferMatMul(const SymNode& node) {
    if (node.input.size() < 2) return std::nullopt;
    auto a = Operand(node.input[0]);
    auto b = Operand(node.input[1]);
    if (!a || !b) return std::nullopt;
    return shaping::MatMul(*a, *b, fresh_);
  }

  std::optional<SymShape> InferGemm(const SymNode& node) {
    if (node.input.size() < 2) return std::nullopt;
    auto a = Operand(node.input[0]);
    auto b = Operand(node.input[1]);
    if (!a || !b) return std::nullopt;
    return shaping::Gemm(*a, *b, IntAttr(node, "transA", 0).value() != 0,
                         IntAttr(node, "transB", 0).value() != 0);
  }

  // Conv (kernel from the weight) and pooling (kernel from kernel_shape).
  std::optional<SymShape> InferConvPool(const SymNode& node, bool is_pool) {
    if (node.input.empty()) return std::nullopt;
    auto x = Operand(node.input[0]);
    if (!x) return std::nullopt;
    const shaping::WindowAttrs attrs = WindowAttrsOf(node);
    if (!is_pool) {
      if (node.input.size() < 2) return std::nullopt;
      auto w = Operand(node.input[1]);
      if (!w) return std::nullopt;
      return shaping::Conv(*x, *w, attrs, fresh_);
    }
    const SymAttr* k = node.attr("kernel_shape");
    if (!k) return std::nullopt;
    return shaping::Pool(*x, k->ints, attrs, fresh_);
  }

  std::optional<SymShape> InferGlobalPool(const SymNode& node) {
    if (node.input.empty()) return std::nullopt;
    auto x = Operand(node.input[0]);
    if (!x) return std::nullopt;
    return shaping::GlobalPool(*x);
  }

  std::optional<SymShape> InferTranspose(const SymNode& node) {
    if (node.input.empty()) return std::nullopt;
    auto x = Operand(node.input[0]);
    if (!x) return std::nullopt;
    std::optional<std::vector<int64_t>> perm;
    if (const SymAttr* a = node.attr("perm")) perm = a->ints;
    return shaping::Transpose(*x, perm);
  }

  std::optional<SymShape> InferConcat(const SymNode& node) {
    std::vector<SymShape> ins;
    for (const std::string& name : node.input) {
      auto s = Operand(name);
      if (!s) return std::nullopt;
      ins.push_back(std::move(*s));
    }
    return shaping::Concat(ins, IntAttr(node, "axis", 0).value());
  }

  std::optional<SymShape> InferFlatten(const SymNode& node) {
    if (node.input.empty()) return std::nullopt;
    auto x = Operand(node.input[0]);
    if (!x) return std::nullopt;
    return shaping::Flatten(*x, IntAttr(node, "axis", 1).value());
  }

  std::optional<SymShape> InferReshape(const SymNode& node) {
    if (node.input.size() < 2) return std::nullopt;
    auto target = IntListOperand(node, "shape", 1);
    if (!target) return std::nullopt;
    // The input shape may be unknown; the rule then makes fresh symbols.
    return shaping::Reshape(Operand(node.input[0]), *target, fresh_);
  }

  std::optional<SymShape> InferSqueeze(const SymNode& node) {
    if (node.input.empty()) return std::nullopt;
    auto x = Operand(node.input[0]);
    if (!x) return std::nullopt;
    return shaping::Squeeze(*x, IntListOperand(node, "axes", 1));
  }

  std::optional<SymShape> InferUnsqueeze(const SymNode& node) {
    if (node.input.empty()) return std::nullopt;
    auto x = Operand(node.input[0]);
    auto axes = IntListOperand(node, "axes", 1);
    if (!x || !axes) return std::nullopt;
    return shaping::Unsqueeze(*x, *axes);
  }

  std::optional<SymShape> InferExpand(const SymNode& node) {
    if (node.input.size() < 2) return std::nullopt;
    auto x = Operand(node.input[0]);
    auto target = IntListOperand(node, "shape", 1);
    if (!x || !target) return std::nullopt;
    return shaping::Expand(*x, *target, fresh_);
  }

  std::optional<SymShape> InferGather(const SymNode& node) {
    if (node.input.size() < 2) return std::nullopt;
    auto data = Operand(node.input[0]);
    auto idx = Operand(node.input[1]);
    if (!data || !idx) return std::nullopt;
    return shaping::Gather(*data, *idx, IntAttr(node, "axis", 0).value());
  }

  std::optional<SymShape> InferSlice(const SymNode& node) {
    if (node.input.empty()) return std::nullopt;
    auto x = Operand(node.input[0]);
    if (!x) return std::nullopt;
    auto starts = IntListOperand(node, "starts", 1);
    auto ends = IntListOperand(node, "ends", 2);
    auto axes = IntListOperand(node, "axes", 3);
    auto steps = IntListOperand(node, "steps", 4);
    if (!starts || !ends) return std::nullopt;
    return shaping::Slice(*x, *starts, *ends, axes, steps, fresh_);
  }

  std::optional<SymShape> InferReduce(const SymNode& node) {
    if (node.input.empty()) return std::nullopt;
    auto x = Operand(node.input[0]);
    if (!x) return std::nullopt;
    const bool keepdims = IntAttr(node, "keepdims", 1).value() != 0;
    auto axes = IntListOperand(node, "axes", 1);
    const bool noop_empty =
        IntAttr(node, "noop_with_empty_axes", 0).value() != 0;
    return shaping::Reduce(*x, keepdims, axes, noop_empty);
  }

  const ShapeGraph& graph_;
  std::map<std::string, SymShape> shapes_;
  shaping::FreshDims fresh_;
};

}  // namespace

std::map<std::string, SymShape> InferSymbolicShapes(const ShapeGraph& graph) {
  return ShapeInferer(graph).Run();
}

}  // namespace onnxsim
