#include "shaping_rules.h"

#include <algorithm>
#include <cstddef>
#include <set>
#include <string>
#include <utility>

namespace onnxsim::shaping {

std::int64_t NormalizeAxis(std::int64_t axis, std::int64_t rank) {
  if (axis < 0) axis += rank;
  return axis;
}

std::optional<std::int64_t> Concrete(const SymExpr& dim) {
  if (dim.is_symbolic()) return std::nullopt;
  return dim.to_int();
}

SymExpr FreshDims::operator()() {
  return SymExpr::Symbol("unk_" + std::to_string(next_++));
}

// --- broadcasting ----------------------------------------------------------

SymExpr BroadcastDim(const SymExpr& a, const SymExpr& b, FreshDims& fresh) {
  auto ca = Concrete(a);
  auto cb = Concrete(b);
  if (ca && *ca == 1) return b;
  if (cb && *cb == 1) return a;
  auto eq = TryEqual(a, b);
  if (eq && *eq) return a;            // provably the same dim
  if (ca && cb) return SymExpr(*ca);  // both concrete, differing (invalid ONNX)
  return fresh();                     // differing symbolic dims: keep distinct
}

Shape Broadcast(const std::vector<Shape>& shapes, FreshDims& fresh) {
  std::size_t rank = 0;
  for (const Shape& s : shapes) rank = std::max(rank, s.size());
  Shape out(rank, SymExpr(1));
  for (std::size_t i = 0; i < rank; ++i) {
    bool have = false;
    for (const Shape& s : shapes) {
      // Right-align: output position i maps to this shape's dim only if the
      // shape is long enough.
      const std::int64_t idx = static_cast<std::int64_t>(s.size()) -
                               static_cast<std::int64_t>(rank) +
                               static_cast<std::int64_t>(i);
      if (idx < 0) continue;
      const SymExpr& d = s[static_cast<std::size_t>(idx)];
      out[i] = have ? BroadcastDim(out[i], d, fresh) : d;
      have = true;
    }
  }
  return out;
}

Shape Expand(const Shape& input, const std::vector<std::int64_t>& target,
             FreshDims& fresh) {
  Shape t;
  t.reserve(target.size());
  for (std::int64_t v : target) t.push_back(SymExpr(v));
  return Broadcast({input, t}, fresh);
}

// --- reshaping -------------------------------------------------------------

Shape Reshape(const std::optional<Shape>& input,
              const std::vector<std::int64_t>& target, FreshDims& fresh) {
  Shape out(target.size());
  int neg_one = -1;
  for (std::size_t i = 0; i < target.size(); ++i) {
    const std::int64_t t = target[i];
    if (t == -1) {
      neg_one = static_cast<int>(i);
      continue;
    }
    if (t == 0) {  // copy the input dim at this position (allowzero=0)
      if (input && i < input->size())
        out[i] = (*input)[i];
      else
        out[i] = fresh();
    } else {
      out[i] = SymExpr(t);
    }
  }
  if (neg_one >= 0) {
    // Resolve the -1 as total // product(other dims) when the input is fully
    // known; otherwise leave that single slot a fresh symbol.
    SymExpr resolved;
    bool ok = false;
    if (input) {
      SymExpr total(1);
      for (const SymExpr& d : *input) total = total * d;
      SymExpr denom(1);
      for (std::size_t i = 0; i < out.size(); ++i)
        if (static_cast<int>(i) != neg_one) denom = denom * out[i];
      if (auto q = TryExactDivide(total, denom)) {
        resolved = *q;
        ok = true;
      }
    }
    out[static_cast<std::size_t>(neg_one)] = ok ? resolved : fresh();
  }
  return out;
}

std::optional<Shape> Flatten(const Shape& input, std::int64_t axis) {
  const std::int64_t rank = static_cast<std::int64_t>(input.size());
  axis = NormalizeAxis(axis, rank);
  if (axis < 0 || axis > rank) return std::nullopt;
  SymExpr outer(1), inner(1);
  for (std::int64_t i = 0; i < axis; ++i)
    outer = outer * input[static_cast<std::size_t>(i)];
  for (std::int64_t i = axis; i < rank; ++i)
    inner = inner * input[static_cast<std::size_t>(i)];
  return Shape{outer, inner};
}

Shape Squeeze(const Shape& input,
              const std::optional<std::vector<std::int64_t>>& axes) {
  const std::int64_t rank = static_cast<std::int64_t>(input.size());
  std::set<std::int64_t> drop;
  if (axes) {
    for (std::int64_t a : *axes) drop.insert(NormalizeAxis(a, rank));
  } else {
    // No axes: drop every dim provably equal to 1.
    for (std::int64_t i = 0; i < rank; ++i) {
      auto c = Concrete(input[static_cast<std::size_t>(i)]);
      if (c && *c == 1) drop.insert(i);
    }
  }
  Shape out;
  for (std::int64_t i = 0; i < rank; ++i)
    if (!drop.count(i)) out.push_back(input[static_cast<std::size_t>(i)]);
  return out;
}

std::optional<Shape> Unsqueeze(const Shape& input,
                               const std::vector<std::int64_t>& axes) {
  const std::int64_t out_rank =
      static_cast<std::int64_t>(input.size() + axes.size());
  std::set<std::int64_t> ins;
  for (std::int64_t a : axes) ins.insert(NormalizeAxis(a, out_rank));
  Shape out;
  std::size_t src = 0;
  for (std::int64_t i = 0; i < out_rank; ++i) {
    if (ins.count(i))
      out.push_back(SymExpr(1));
    else if (src < input.size())
      out.push_back(input[src++]);
    else
      return std::nullopt;
  }
  return out;
}

std::optional<Shape> Transpose(
    const Shape& input, const std::optional<std::vector<std::int64_t>>& perm) {
  const std::int64_t rank = static_cast<std::int64_t>(input.size());
  std::vector<std::int64_t> order;
  if (perm)
    order = *perm;
  else
    for (std::int64_t i = rank - 1; i >= 0; --i) order.push_back(i);
  if (static_cast<std::int64_t>(order.size()) != rank) return std::nullopt;
  Shape out;
  for (std::int64_t p : order) {
    p = NormalizeAxis(p, rank);
    if (p < 0 || p >= rank) return std::nullopt;
    out.push_back(input[static_cast<std::size_t>(p)]);
  }
  return out;
}

// --- joining and indexing --------------------------------------------------

std::optional<Shape> Concat(const std::vector<Shape>& inputs,
                            std::int64_t axis) {
  if (inputs.empty()) return std::nullopt;
  const std::int64_t rank = static_cast<std::int64_t>(inputs[0].size());
  axis = NormalizeAxis(axis, rank);
  if (axis < 0 || axis >= rank) return std::nullopt;
  Shape out = inputs[0];
  SymExpr sum = out[static_cast<std::size_t>(axis)];
  for (std::size_t k = 1; k < inputs.size(); ++k) {
    if (static_cast<std::int64_t>(inputs[k].size()) != rank)
      return std::nullopt;
    sum = sum + inputs[k][static_cast<std::size_t>(axis)];
  }
  out[static_cast<std::size_t>(axis)] = sum;
  return out;
}

std::optional<Shape> Gather(const Shape& data, const Shape& indices,
                            std::int64_t axis) {
  if (data.empty()) return std::nullopt;
  const std::int64_t rank = static_cast<std::int64_t>(data.size());
  axis = NormalizeAxis(axis, rank);
  if (axis < 0 || axis >= rank) return std::nullopt;
  // out = data[:axis] + indices.shape + data[axis+1:]
  Shape out;
  for (std::int64_t i = 0; i < axis; ++i)
    out.push_back(data[static_cast<std::size_t>(i)]);
  for (const SymExpr& d : indices) out.push_back(d);
  for (std::int64_t i = axis + 1; i < rank; ++i)
    out.push_back(data[static_cast<std::size_t>(i)]);
  return out;
}

std::optional<Shape> Slice(
    const Shape& input, const std::vector<std::int64_t>& starts,
    const std::vector<std::int64_t>& ends,
    const std::optional<std::vector<std::int64_t>>& axes,
    const std::optional<std::vector<std::int64_t>>& steps, FreshDims& fresh) {
  const std::int64_t rank = static_cast<std::int64_t>(input.size());
  if (starts.size() != ends.size()) return std::nullopt;
  // Guard for the out-of-bounds read in the earlier code (axes shorter than
  // starts). Longer axes were already tolerated, so they are still accepted.
  if (axes && axes->size() < starts.size()) return std::nullopt;
  Shape out = input;
  for (std::size_t j = 0; j < starts.size(); ++j) {
    const std::int64_t axis =
        axes ? NormalizeAxis((*axes)[j], rank) : static_cast<std::int64_t>(j);
    if (axis < 0 || axis >= rank) return std::nullopt;
    const std::int64_t step = (steps && j < steps->size()) ? (*steps)[j] : 1;
    auto dim = Concrete(input[static_cast<std::size_t>(axis)]);
    if (!dim || step == 0) {
      out[static_cast<std::size_t>(axis)] = fresh();
      continue;
    }
    const std::int64_t len = *dim;
    std::int64_t s = starts[j], e = ends[j];
    if (s < 0) s += len;
    if (e < 0) e += len;
    std::int64_t count;
    if (step > 0) {
      s = std::clamp<std::int64_t>(s, 0, len);
      e = std::clamp<std::int64_t>(e, 0, len);
      count = e > s ? (e - s + step - 1) / step : 0;
    } else {
      s = std::clamp<std::int64_t>(s, 0, len - 1);
      e = std::clamp<std::int64_t>(e, -1, len - 1);
      count = s > e ? (s - e - step - 1) / (-step) : 0;
    }
    out[static_cast<std::size_t>(axis)] = SymExpr(count);
  }
  return out;
}

Shape Reduce(const Shape& input, bool keepdims,
             const std::optional<std::vector<std::int64_t>>& axes,
             bool noop_with_empty_axes) {
  const std::int64_t rank = static_cast<std::int64_t>(input.size());
  std::set<std::int64_t> reduce;
  if (axes && !axes->empty()) {
    for (std::int64_t a : *axes) reduce.insert(NormalizeAxis(a, rank));
  } else if (axes && axes->empty() && noop_with_empty_axes) {
    return input;  // explicit empty axes + noop flag: identity
  } else {
    for (std::int64_t i = 0; i < rank; ++i) reduce.insert(i);  // reduce all
  }
  Shape out;
  for (std::int64_t i = 0; i < rank; ++i) {
    if (reduce.count(i)) {
      if (keepdims) out.push_back(SymExpr(1));
    } else {
      out.push_back(input[static_cast<std::size_t>(i)]);
    }
  }
  return out;
}

// --- matrix products -------------------------------------------------------

std::optional<Shape> MatMul(const Shape& a, const Shape& b, FreshDims& fresh) {
  if (a.empty() || b.empty()) return std::nullopt;
  // numpy matmul: a rank-1 lhs gets a leading 1, a rank-1 rhs a trailing 1;
  // both are stripped from the result.
  Shape la = a, lb = b;
  const bool a_vec = la.size() == 1;
  const bool b_vec = lb.size() == 1;
  if (a_vec) la.insert(la.begin(), SymExpr(1));
  if (b_vec) lb.push_back(SymExpr(1));
  // Broadcast the batch dims (everything but the trailing 2x2 matrix).
  Shape a_batch(la.begin(), la.end() - 2);
  Shape b_batch(lb.begin(), lb.end() - 2);
  Shape out = Broadcast({a_batch, b_batch}, fresh);
  // M's position is fixed once the batch dims are laid down; erase it there
  // (rather than at out.end() - 2) so a_vec's erase is still correct after
  // b_vec's pop_back has already shortened `out` -- e.g. both inputs rank-1
  // (a plain dot product), where out.end() - 2 would underflow before
  // begin() once N has already been popped.
  const std::size_t m_pos = out.size();
  out.push_back(la[la.size() - 2]);  // M
  out.push_back(lb[lb.size() - 1]);  // N
  if (b_vec) out.pop_back();
  if (a_vec) out.erase(out.begin() + static_cast<std::ptrdiff_t>(m_pos));
  return out;
}

std::optional<Shape> Gemm(const Shape& a, const Shape& b, bool trans_a,
                          bool trans_b) {
  if (a.size() != 2 || b.size() != 2) return std::nullopt;
  const SymExpr m = trans_a ? a[1] : a[0];
  const SymExpr n = trans_b ? b[0] : b[1];
  return Shape{m, n};
}

// --- windowed ops ----------------------------------------------------------

namespace {

// Shared spatial-output computation for Conv and pooling. `x` is [N, C,
// spatial...] and `kernel` has one entry per spatial axis.
std::optional<Shape> WindowedOutput(const Shape& x, SymExpr channels_out,
                                    const std::vector<std::int64_t>& kernel,
                                    const WindowAttrs& attrs,
                                    FreshDims& fresh) {
  const std::int64_t spatial = static_cast<std::int64_t>(x.size()) - 2;
  if (static_cast<std::int64_t>(kernel.size()) != spatial) return std::nullopt;

  Shape out;
  out.push_back(x[0]);          // N
  out.push_back(channels_out);  // C or M
  for (std::int64_t i = 0; i < spatial; ++i) {
    const SymExpr& in_dim = x[static_cast<std::size_t>(i + 2)];
    auto in = Concrete(in_dim);
    const std::int64_t s = i < static_cast<std::int64_t>(attrs.strides.size())
                               ? attrs.strides[i]
                               : 1;
    const std::int64_t d = i < static_cast<std::int64_t>(attrs.dilations.size())
                               ? attrs.dilations[i]
                               : 1;
    // ONNX lays pads out as [begin_0, ..., begin_{n-1}, end_0, ..., end_{n-1}].
    const std::int64_t pb = i < static_cast<std::int64_t>(attrs.pads.size())
                                ? attrs.pads[i]
                                : 0;
    const std::int64_t pe =
        spatial + i < static_cast<std::int64_t>(attrs.pads.size())
            ? attrs.pads[spatial + i]
            : 0;
    // Exact only when the input spatial dim is a concrete int; a strided window
    // over a symbolic length is not a polynomial, so it gets a fresh symbol.
    if (!in) {
      out.push_back(fresh());
      continue;
    }
    const std::int64_t k = kernel[static_cast<std::size_t>(i)];
    const std::int64_t eff = d * (k - 1) + 1;  // dilated kernel extent
    const std::int64_t numer = *in + pb + pe - eff;
    if (numer < 0) {
      out.push_back(fresh());
      continue;
    }
    const std::int64_t o =
        attrs.ceil_mode ? (numer + s - 1) / s + 1 : numer / s + 1;
    out.push_back(SymExpr(o));
  }
  return out;
}

}  // namespace

std::optional<Shape> Conv(const Shape& x, const Shape& w,
                          const WindowAttrs& attrs, FreshDims& fresh) {
  if (x.size() < 3) return std::nullopt;  // [N, C, spatial...]
  const std::int64_t spatial = static_cast<std::int64_t>(x.size()) - 2;
  if (static_cast<std::int64_t>(w.size()) != spatial + 2) return std::nullopt;
  std::vector<std::int64_t> kernel;
  for (std::int64_t i = 0; i < spatial; ++i) {
    auto k = Concrete(w[static_cast<std::size_t>(i + 2)]);
    if (!k) return std::nullopt;
    kernel.push_back(*k);
  }
  return WindowedOutput(x, w[0] /* M */, kernel, attrs, fresh);
}

std::optional<Shape> Pool(const Shape& x,
                          const std::vector<std::int64_t>& kernel,
                          const WindowAttrs& attrs, FreshDims& fresh) {
  if (x.size() < 3) return std::nullopt;  // [N, C, spatial...]
  return WindowedOutput(x, x[1] /* C preserved */, kernel, attrs, fresh);
}

std::optional<Shape> GlobalPool(const Shape& input) {
  if (input.size() < 2) return std::nullopt;
  Shape out{input[0], input[1]};
  for (std::size_t i = 2; i < input.size(); ++i) out.push_back(SymExpr(1));
  return out;
}

}  // namespace onnxsim::shaping
