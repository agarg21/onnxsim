#pragma once

// Pure ONNX shape rules over symbolic dimensions.
//
// Dependencies: this header and shaping_rules.cpp use only the C++ standard
// library and sym_expr.h. There is no onnx, protobuf, SymEngine or sympy
// dependency, so other projects can call these functions with plain C++ types.
//
// Conventions shared by every function below:
//   * A shape is a std::vector<SymExpr>, one entry per dimension. A dimension
//   is
//     a SymExpr: a concrete size is SymExpr(d), a named symbol is
//     SymExpr::Symbol("batch").
//   * Attribute and initializer values are passed as plain values. An
//     std::optional argument means the attribute or operand is absent. Where an
//     ONNX default applies, the caller passes that default, except where a
//     parameter says otherwise.
//   * Axes are passed raw. Negative axes count from the end and are normalized
//     inside the rule, as ONNX does.
//   * A function returns std::nullopt when the rule cannot be applied (invalid
//     rank, axis out of range, mismatched operands). It never returns a guessed
//     shape.
//   * A dimension that a rule cannot compute exactly is a fresh symbol from
//     FreshDims ("unk_0", "unk_1", ...). Fresh symbols are never merged with
//     another dimension, so a result is only ever less resolved, never wrongly
//     equated.
//
// These rules deliberately do not validate operands the way onnx.checker does.
// Each function's comment says where it is lenient on invalid input.

#include <cstdint>
#include <optional>
#include <vector>

#include "sym_expr.h"

namespace onnxsim::shaping {

// A tensor shape: one SymExpr per dimension.
using Shape = std::vector<SymExpr>;

// Source of fresh, never-merged symbols. Each call returns "unk_<n>" with n
// counting up from 0. The sequence is deterministic: no RNG, no global state.
class FreshDims {
 public:
  SymExpr operator()();

 private:
  std::int64_t next_ = 0;
};

// ONNX negative-axis rule: a negative axis `a` denotes `a + rank`. Values out
// of range are returned unchanged, so the caller range-checks them.
std::int64_t NormalizeAxis(std::int64_t axis, std::int64_t rank);

// The integer value of `dim` when it has no symbols; nullopt otherwise.
std::optional<std::int64_t> Concrete(const SymExpr& dim);

// --- broadcasting ----------------------------------------------------------

// Numpy/ONNX multidirectional broadcasting of one pair of dimensions:
//   1 broadcasts to the other dimension; equal dimensions stay as they are;
//   a symbolic dimension against a different dimension gives a fresh symbol.
// Lenient: two different concrete dimensions, neither 1 (invalid ONNX), return
// `a` without an error.
SymExpr BroadcastDim(const SymExpr& a, const SymExpr& b, FreshDims& fresh);

// ONNX multidirectional broadcasting of any number of shapes. Shapes are
// right-aligned, the output rank is the maximum input rank, and each output
// dimension is BroadcastDim over the inputs that have that dimension. Also used
// for Where (its three operands), Expand, and the batch dimensions of MatMul.
// An empty input list gives an empty shape.
Shape Broadcast(const std::vector<Shape>& shapes, FreshDims& fresh);

// ONNX Expand: the broadcast of `input` with the constant shape `target`.
Shape Expand(const Shape& input, const std::vector<std::int64_t>& target,
             FreshDims& fresh);

// --- reshaping -------------------------------------------------------------

// ONNX Reshape with allowzero=0 (the default; the allowzero attribute is not
// modelled):
//   * a 0 copies input dimension i (same position), or a fresh symbol if the
//     input shape is unknown or has no dimension i;
//   * a -1 is inferred as total_elements / product(other dims), the exact
//     division via TryExactDivide. If that does not divide evenly, or the input
//     is unknown, the slot is a fresh symbol;
//   * any other value is a literal dimension.
// Lenient: more than one -1 is invalid ONNX and is not diagnosed. The earlier
// -1 slots stay SymExpr(0), and the last -1 slot becomes a fresh symbol (its
// denominator includes that zero).
Shape Reshape(const std::optional<Shape>& input,
              const std::vector<std::int64_t>& target, FreshDims& fresh);

// ONNX Flatten: [prod(dims[:axis]), prod(dims[axis:])]. `axis` is normalized
// against the input rank, and must end up in [0, rank] (nullopt otherwise).
// Default axis is 1 (caller passes it).
std::optional<Shape> Flatten(const Shape& input, std::int64_t axis);

// ONNX Squeeze.
//   * `axes` present: removes those axes. It does not check that they are 1.
//   * `axes` absent: removes every dimension that is provably the concrete
//     value 1. Symbolic dimensions are kept.
// Lenient: `axes` is not checked against the input rank.
Shape Squeeze(const Shape& input,
              const std::optional<std::vector<std::int64_t>>& axes);

// ONNX Unsqueeze: the output rank is input rank + len(axes). Each axis is
// normalized against the output rank, and inserts a concrete 1. The remaining
// positions take the input dims in order. nullopt if the axes do not fit.
std::optional<Shape> Unsqueeze(const Shape& input,
                               const std::vector<std::int64_t>& axes);

// ONNX Transpose: output dim i = input dim perm[i]. If `perm` is absent it is
// the reversed dimension order. nullopt if perm's length differs from the rank
// or an entry is out of range.
std::optional<Shape> Transpose(
    const Shape& input, const std::optional<std::vector<std::int64_t>>& perm);

// --- joining and indexing --------------------------------------------------

// ONNX Concat: all inputs share a rank. The output is input 0's shape with the
// `axis` dimension replaced by the sum over all inputs. Lenient: the other
// dimensions are copied from input 0 without checking that the inputs agree.
// nullopt for no inputs, a rank mismatch, or an axis out of range.
std::optional<Shape> Concat(const std::vector<Shape>& inputs,
                            std::int64_t axis);

// ONNX Gather: data[:axis] + indices.shape + data[axis+1:]. `axis` is
// normalized against the data rank. nullopt for empty data or an axis out of
// range.
std::optional<Shape> Gather(const Shape& data, const Shape& indices,
                            std::int64_t axis);

// ONNX Slice, with starts/ends/axes/steps read as plain values:
//   * axes absent: the listed entries apply to axes 0, 1, ...
//   * steps absent: each step is 1.
//   * For each sliced axis: a negative start or end adds the dim length. Then
//   it
//     is clamped as Python slicing clamps it. Positive steps clamp to [0, len].
//     Negative steps clamp the start to [0, len-1] and the end to [-1, len-1].
//     The output length is ceil((end - start) / step), or 0.
//   * A symbolic dimension, or a step of 0, gives a fresh symbol.
// nullopt if starts and ends differ in length, or if axes is shorter than
// starts. The second case is an out-of-bounds read in the earlier code, which
// now returns nullopt instead.
std::optional<Shape> Slice(
    const Shape& input, const std::vector<std::int64_t>& starts,
    const std::vector<std::int64_t>& ends,
    const std::optional<std::vector<std::int64_t>>& axes,
    const std::optional<std::vector<std::int64_t>>& steps, FreshDims& fresh);

// ONNX Reduce* (ReduceSum, ReduceMean, ...):
//   * non-empty `axes`: reduce those axes;
//   * `axes` present but empty, with noop_with_empty_axes: identity;
//   * otherwise (axes absent, or empty without the noop flag): reduce all axes.
//   * keepdims: a reduced axis stays as a concrete 1; otherwise it is removed.
Shape Reduce(const Shape& input, bool keepdims,
             const std::optional<std::vector<std::int64_t>>& axes,
             bool noop_with_empty_axes);

// --- matrix products -------------------------------------------------------

// ONNX MatMul (numpy matmul semantics). A rank-1 lhs gets a leading 1 and a
// rank-1 rhs a trailing 1, and both are removed from the result. The batch
// dims broadcast with Broadcast, and the result is batch + [M, N]. nullopt if
// either input is empty (rank 0).
std::optional<Shape> MatMul(const Shape& a, const Shape& b, FreshDims& fresh);

// ONNX Gemm: both operands are rank 2. The result is [M, N], where M and N are
// taken from a and b after transA/transB. nullopt if either operand is not
// rank 2. alpha, beta and C do not affect shape.
std::optional<Shape> Gemm(const Shape& a, const Shape& b, bool trans_a,
                          bool trans_b);

// --- windowed ops (Conv, pooling) -----------------------------------------

// The attributes shared by Conv and the pooling ops. An empty vector means the
// attribute is absent, and it defaults per spatial axis: stride 1, dilation 1,
// pad 0. ceil_mode defaults to false.
//
// KNOWN DEVIATION FROM ONNX (kept so behavior is unchanged): `pads` is read as
// interleaved [begin_0, end_0, begin_1, end_1, ...]. The ONNX spec gives
// [begin_0, begin_1, ..., end_0, end_1, ...]. The two agree whenever begin ==
// end on every axis, which covers all symmetric padding. They disagree on
// asymmetric padding, where the output size comes out wrong (for example
// pads=[0,0,1,1] on a 5x5 input). Fixing this changes existing behavior, so it
// is left for a deliberate follow-up.
//
// KNOWN GAP: the `auto_pad` string attribute is not modelled. SAME_UPPER and
// SAME_LOWER are treated as zero padding, so their output sizes are wrong.
struct WindowAttrs {
  std::vector<std::int64_t> strides;
  std::vector<std::int64_t> dilations;
  std::vector<std::int64_t> pads;
  bool ceil_mode = false;
};

// ONNX Conv on [N, C, spatial...] input and [M, C/group, kernel...] weight:
// the output is [N, M, out_spatial...]. The kernel is the weight's spatial
// dimensions, so each must be concrete. Otherwise, or if the weight rank is not
// spatial + 2, the result is nullopt.
//
// Spatial output size, per axis, with input size `in`, kernel k, stride s,
// dilation d, and pads p_begin + p_end:
//   effective = d * (k - 1) + 1
//   floor mode: out = floor((in + p_begin + p_end - effective) / s) + 1
//   ceil mode:  out = ceil ((in + p_begin + p_end - effective) / s) + 1
// A symbolic `in`, or a negative numerator, gives a fresh symbol. The strided
// window over a symbolic length is not a polynomial.
std::optional<Shape> Conv(const Shape& x, const Shape& w,
                          const WindowAttrs& attrs, FreshDims& fresh);

// ONNX MaxPool, AveragePool and LpPool: channels are preserved, and the spatial
// output follows the same formula as Conv. `kernel` is the kernel_shape
// attribute, which must have one entry per spatial axis. nullopt otherwise.
std::optional<Shape> Pool(const Shape& x,
                          const std::vector<std::int64_t>& kernel,
                          const WindowAttrs& attrs, FreshDims& fresh);

// ONNX GlobalAveragePool, GlobalMaxPool and GlobalLpPool: [N, C, 1, 1, ...].
// nullopt if the input has rank below 2.
std::optional<Shape> GlobalPool(const Shape& input);

}  // namespace onnxsim::shaping
