// Standalone test (no ONNX, no protobuf):
//   g++ -std=c++17 shaping_rules.cpp sym_expr.cpp shaping_rules_test.cpp -o t
//   && ./t
// (Builds identically under Emscripten: em++ -std=c++17 ...)
#undef NDEBUG  // keep the asserts live in Release builds
#include "shaping_rules.h"

#include <cassert>
#include <initializer_list>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

using onnxsim::SymExpr;
using onnxsim::shaping::Broadcast;
using onnxsim::shaping::BroadcastDim;
using onnxsim::shaping::Concat;
using onnxsim::shaping::Concrete;
using onnxsim::shaping::Conv;
using onnxsim::shaping::Expand;
using onnxsim::shaping::Flatten;
using onnxsim::shaping::FreshDims;
using onnxsim::shaping::Gather;
using onnxsim::shaping::Gemm;
using onnxsim::shaping::GlobalPool;
using onnxsim::shaping::MatMul;
using onnxsim::shaping::NormalizeAxis;
using onnxsim::shaping::Pool;
using onnxsim::shaping::Reduce;
using onnxsim::shaping::Reshape;
using onnxsim::shaping::Shape;
using onnxsim::shaping::Slice;
using onnxsim::shaping::Squeeze;
using onnxsim::shaping::Transpose;
using onnxsim::shaping::Unsqueeze;
using onnxsim::shaping::WindowAttrs;

namespace {

// "[batch, seq, 8]" -- the printed form of a shape, for comparisons.
std::string Str(const Shape& s) {
  std::string r = "[";
  for (std::size_t i = 0; i < s.size(); ++i) {
    if (i) r += ", ";
    r += s[i].str();
  }
  return r + "]";
}

Shape Dims(std::initializer_list<SymExpr> xs) { return Shape(xs); }

// True iff `d` is a fresh placeholder symbol ("unk_N").
bool IsFresh(const SymExpr& d) {
  return d.is_symbolic() && d.str().rfind("unk_", 0) == 0;
}

}  // namespace

int main() {
  const SymExpr batch = SymExpr::Symbol("batch");
  const SymExpr seq = SymExpr::Symbol("seq");

  // --- helpers --------------------------------------------------------------
  assert(NormalizeAxis(-1, 3) == 2);
  assert(NormalizeAxis(1, 3) == 1);
  assert(!Concrete(batch).has_value());
  assert(Concrete(SymExpr(5)).value() == 5);

  // FreshDims: deterministic, never repeats.
  {
    FreshDims f;
    assert(f().str() == "unk_0");
    assert(f().str() == "unk_1");
  }

  // --- broadcasting with symbols ------------------------------------------
  {
    FreshDims f;
    // 1 broadcasts to the other side; equal symbols stay symbolic.
    assert(BroadcastDim(batch, SymExpr(1), f).str() == "batch");
    assert(BroadcastDim(SymExpr(1), seq, f).str() == "seq");
    assert(BroadcastDim(batch, batch, f).str() == "batch");
    // Differing symbolic dims cannot be decided: a fresh symbol, not batch.
    const SymExpr unk = BroadcastDim(batch, seq, f);
    assert(IsFresh(unk) && unk.str() != "batch" && unk.str() != "seq");
    // A symbol against a different concrete size is also undecided.
    assert(IsFresh(BroadcastDim(batch, SymExpr(512), f)));
  }
  {
    FreshDims f;
    // [batch,1,8] with [1,seq,8] -> [batch, seq, 8]
    const Shape out = Broadcast({Dims({batch, SymExpr(1), SymExpr(8)}),
                                 Dims({SymExpr(1), seq, SymExpr(8)})},
                                f);
    assert(Str(out) == "[batch, seq, 8]");
  }
  {
    FreshDims f;
    // Right-aligned rank promotion: [4] with [batch,1,4] -> [batch, 1, 4]
    const Shape out = Broadcast(
        {Dims({SymExpr(4)}), Dims({batch, SymExpr(1), SymExpr(4)})}, f);
    assert(Str(out) == "[batch, 1, 4]");
  }
  {
    FreshDims f;
    // batch vs 512 is undecided: the output dim becomes a fresh unk_ symbol.
    const Shape out = Broadcast({Dims({batch}), Dims({SymExpr(512)})}, f);
    assert(out.size() == 1 && IsFresh(out[0]));
    // No inputs: an empty shape.
    assert(Broadcast({}, f).empty());
  }
  {
    FreshDims f;
    // Expand: [1,4] -> [3,1] target gives [3,4]; symbolic batch survives a 1.
    assert(Str(Expand(Dims({SymExpr(1), SymExpr(4)}), {3, 1}, f)) == "[3, 4]");
    assert(Str(Expand(Dims({batch, SymExpr(4)}), {1, 4}, f)) == "[batch, 4]");
  }

  // --- Reshape: 0 copies, -1 infers ----------------------------------------
  {
    FreshDims f;
    // 0 copies input dim 0 (batch); -1 = batch*12 / batch = 12.
    assert(Str(Reshape(Dims({batch, SymExpr(3), SymExpr(4)}), {0, -1}, f)) ==
           "[batch, 12]");
  }
  {
    FreshDims f;
    // Matches the TryExactDivide example: (batch*1024*128) // (1024*128) =
    // batch gives 1024 for the middle dim here.
    assert(Str(Reshape(Dims({batch, SymExpr(1024), SymExpr(128)}), {0, -1, 128},
                       f)) == "[batch, 1024, 128]");
  }
  {
    FreshDims f;
    // Concrete -1: 2*3*4 / 4 = 6.
    assert(Str(Reshape(Dims({SymExpr(2), SymExpr(3), SymExpr(4)}), {-1, 4},
                       f)) == "[6, 4]");
    // 0 copies both leading dims, -1 = 24 / 6 = 4.
    assert(Str(Reshape(Dims({SymExpr(2), SymExpr(3), SymExpr(4)}), {0, 0, -1},
                       f)) == "[2, 3, 4]");
  }
  {
    FreshDims f;
    // Symbolic -1 that divides exactly: (batch*4) / 2 = 2*batch.
    assert(Str(Reshape(Dims({batch, SymExpr(4)}), {-1, 2}, f)) ==
           "[2*batch, 2]");
    // Not divisible (3*batch / 2): the slot stays undecided.
    const Shape undiv = Reshape(Dims({batch, SymExpr(3)}), {-1, 2}, f);
    assert(IsFresh(undiv[0]) && Str(Shape{undiv[1]}) == "[2]");
  }
  {
    FreshDims f;
    // Unknown input: 0 and -1 both become fresh symbols, in that order.
    assert(Str(Reshape(std::nullopt, {0, -1}, f)) == "[unk_0, unk_1]");
  }
  {
    FreshDims f;
    // A 0 with no matching input dim (input rank 2, position 2) is fresh.
    assert(Str(Reshape(Dims({SymExpr(2), SymExpr(3)}), {0, 0, 0}, f)) ==
           "[2, 3, unk_0]");
  }

  // --- Flatten / Squeeze / Unsqueeze --------------------------------------
  {
    assert(Str(*Flatten(Dims({batch, SymExpr(3), SymExpr(4), SymExpr(5)}),
                        1)) == "[batch, 60]");
    // axis 0: outer product is empty (1), inner is everything.
    assert(Str(*Flatten(Dims({batch, SymExpr(3), SymExpr(4), SymExpr(5)}),
                        0)) == "[1, 60*batch]");
    // A negative axis normalizes: -1 on rank 3 -> 2.
    assert(Str(*Flatten(Dims({SymExpr(2), SymExpr(3), SymExpr(4)}), -1)) ==
           "[6, 4]");
    assert(!Flatten(Dims({batch}), 3).has_value());  // axis past rank
  }
  {
    // No axes: drop concrete 1s, keep symbols.
    assert(Str(Squeeze(Dims({batch, SymExpr(1), SymExpr(4)}), std::nullopt)) ==
           "[batch, 4]");
    assert(Str(Squeeze(Dims({batch, SymExpr(1)}), std::nullopt)) == "[batch]");
    // Explicit axes, negative: -2 on rank 3 is axis 1.
    assert(Str(Squeeze(Dims({batch, SymExpr(1), SymExpr(4)}),
                       std::vector<std::int64_t>{-2})) == "[batch, 4]");
  }
  {
    // Output rank 4: axes 0 and -1 (=3) become 1; the rest keep input order.
    assert(Str(*Unsqueeze(Dims({batch, SymExpr(4)}), {0, -1})) ==
           "[1, batch, 4, 1]");
  }

  // --- Transpose / Concat / Gather ----------------------------------------
  {
    assert(
        Str(*Transpose(Dims({batch, SymExpr(3), SymExpr(4)}), std::nullopt)) ==
        "[4, 3, batch]");  // default: reversed
    assert(Str(*Transpose(Dims({batch, SymExpr(3), SymExpr(4)}),
                          std::vector<std::int64_t>{0, 2, 1})) ==
           "[batch, 4, 3]");
    assert(!Transpose(Dims({batch, SymExpr(3)}), std::vector<std::int64_t>{0})
                .has_value());
  }
  {
    assert(Str(*Concat({Dims({batch, SymExpr(4)}), Dims({batch, SymExpr(6)})},
                       1)) == "[batch, 10]");
    // Axis -2 on rank 2 is axis 0: batch + batch.
    assert(Str(*Concat({Dims({batch, SymExpr(4)}), Dims({batch, SymExpr(4)})},
                       -2)) == "[2*batch, 4]");
    assert(!Concat({}, 0).has_value());
  }
  {
    // Gather along axis 0 with indices [2,3]: [2, 3] + data[1:].
    assert(Str(*Gather(Dims({SymExpr(5), batch, SymExpr(7)}),
                       Dims({SymExpr(2), SymExpr(3)}), 0)) ==
           "[2, 3, batch, 7]");
  }

  // --- Slice ---------------------------------------------------------------
  {
    FreshDims f;
    // Concrete: x[2:100] on dim 10 has 8 elements.
    assert(Str(*Slice(Dims({SymExpr(10), SymExpr(8)}), {2}, {100},
                      std::vector<std::int64_t>{0}, std::nullopt, f)) ==
           "[8, 8]");
  }
  {
    FreshDims f;
    // Negative step: x[9:-11:-2] on length 10 -> indices 9,7,5,3,1.
    assert(Str(*Slice(Dims({SymExpr(10)}), {9}, {-11}, std::nullopt,
                      std::vector<std::int64_t>{-2}, f)) == "[5]");
  }
  {
    FreshDims f;
    // A symbolic dim being sliced gives a fresh symbol; a concrete one does
    // not.
    const Shape sliced = *Slice(Dims({batch, SymExpr(8)}), {0}, {4},
                                std::vector<std::int64_t>{0}, std::nullopt, f);
    assert(IsFresh(sliced[0]) && Str(Shape{sliced[1]}) == "[8]");
    assert(Str(*Slice(Dims({batch, SymExpr(8)}), {0}, {4},
                      std::vector<std::int64_t>{1}, std::nullopt, f)) ==
           "[batch, 4]");
  }
  {
    FreshDims f;
    // Malformed: starts/ends differ in length, or axes is shorter than starts.
    assert(
        !Slice(Dims({SymExpr(4)}), {0, 0}, {2}, std::nullopt, std::nullopt, f)
             .has_value());
    assert(!Slice(Dims({SymExpr(4), SymExpr(6)}), {0, 0}, {2, 3},
                  std::vector<std::int64_t>{0}, std::nullopt, f)
                .has_value());
  }

  // --- Reduce --------------------------------------------------------------
  {
    assert(Str(Reduce(Dims({batch, SymExpr(4), SymExpr(8)}), false,
                      std::vector<std::int64_t>{1}, false)) == "[batch, 8]");
    assert(Str(Reduce(Dims({batch, SymExpr(4), SymExpr(8)}), true,
                      std::vector<std::int64_t>{1}, false)) == "[batch, 1, 8]");
    // No axes: reduce everything.
    assert(Str(Reduce(Dims({batch, SymExpr(4), SymExpr(8)}), true, std::nullopt,
                      false)) == "[1, 1, 1]");
    assert(Str(Reduce(Dims({batch, SymExpr(4), SymExpr(8)}), false,
                      std::nullopt, false)) == "[]");
    // Explicit empty axes: identity with the noop flag, reduce-all without it.
    assert(Str(Reduce(Dims({batch, SymExpr(4)}), false,
                      std::vector<std::int64_t>{}, true)) == "[batch, 4]");
    assert(Str(Reduce(Dims({SymExpr(2), SymExpr(3)}), false,
                      std::vector<std::int64_t>{}, false)) == "[]");
  }

  // --- MatMul / Gemm -------------------------------------------------------
  {
    FreshDims f;
    assert(Str(*MatMul(Dims({batch, SymExpr(4), SymExpr(8)}),
                       Dims({SymExpr(8), SymExpr(16)}), f)) ==
           "[batch, 4, 16]");
    assert(Str(*MatMul(Dims({batch, SymExpr(4), SymExpr(8)}),
                       Dims({batch, SymExpr(8), SymExpr(16)}), f)) ==
           "[batch, 4, 16]");
    // Dot product of two vectors: a scalar (rank 0).
    assert(Str(*MatMul(Dims({SymExpr(8)}), Dims({SymExpr(8)}), f)) == "[]");
    // Vector @ matrix: the promoted leading 1 is removed again.
    assert(Str(*MatMul(Dims({SymExpr(4)}), Dims({SymExpr(4), batch}), f)) ==
           "[batch]");
    assert(!MatMul(Dims({}), Dims({SymExpr(4)}), f).has_value());
  }
  {
    assert(Str(*Gemm(Dims({batch, SymExpr(128)}),
                     Dims({SymExpr(128), SymExpr(10)}), false, false)) ==
           "[batch, 10]");
    assert(Str(*Gemm(Dims({batch, SymExpr(128)}),
                     Dims({SymExpr(10), SymExpr(128)}), false, true)) ==
           "[batch, 10]");
    assert(Str(*Gemm(Dims({SymExpr(128), batch}),
                     Dims({SymExpr(128), SymExpr(10)}), true, false)) ==
           "[batch, 10]");
    assert(!Gemm(Dims({batch}), Dims({SymExpr(10)}), false, false).has_value());
  }

  // --- Conv / pooling output formulas -------------------------------------
  {
    // ResNet stem: 224 input, kernel 7, stride 2, pad 3 -> floor((224+6-7)/2)+1
    // = 112.
    FreshDims f;
    WindowAttrs a;
    a.strides = {2, 2};
    a.pads = {3, 3, 3, 3};
    assert(Str(*Conv(Dims({batch, SymExpr(3), SymExpr(224), SymExpr(224)}),
                     Dims({SymExpr(64), SymExpr(3), SymExpr(7), SymExpr(7)}), a,
                     f)) == "[batch, 64, 112, 112]");
  }
  {
    // Dilation 2, kernel 3: effective extent 5, so 10 - 5 + 1 = 6.
    FreshDims f;
    WindowAttrs a;
    a.dilations = {2, 2};
    assert(Str(*Conv(Dims({SymExpr(1), SymExpr(1), SymExpr(10), SymExpr(10)}),
                     Dims({SymExpr(1), SymExpr(1), SymExpr(3), SymExpr(3)}), a,
                     f)) == "[1, 1, 6, 6]");
  }
  {
    // Asymmetric pads in ONNX layout: begins [0, 0], ends [1, 1]. Kernel 3 on
    // a 5x5 input gives (5 + 0 + 1 - 3) + 1 = 4 on each axis.
    FreshDims f;
    WindowAttrs a;
    a.pads = {0, 0, 1, 1};
    assert(Str(*Conv(Dims({SymExpr(1), SymExpr(1), SymExpr(5), SymExpr(5)}),
                     Dims({SymExpr(1), SymExpr(1), SymExpr(3), SymExpr(3)}), a,
                     f)) == "[1, 1, 4, 4]");
  }
  {
    // Symbolic spatial length: a strided window is not a polynomial -> fresh.
    FreshDims f;
    WindowAttrs a;
    a.pads = {1, 1};
    const Shape out = *Conv(Dims({batch, SymExpr(3), seq}),
                            Dims({SymExpr(64), SymExpr(3), SymExpr(3)}), a, f);
    assert(out.size() == 3 && out[0].str() == "batch" && out[1].str() == "64" &&
           IsFresh(out[2]));
  }
  {
    // A symbolic kernel cannot be read as an integer: no shape.
    FreshDims f;
    assert(!Conv(Dims({batch, SymExpr(3), SymExpr(8)}),
                 Dims({SymExpr(4), SymExpr(3), seq}), WindowAttrs{}, f)
                .has_value());
  }
  {
    // Input 2 with kernel 5 and no pads: numerator -3 is undecided -> fresh.
    FreshDims f;
    assert(Str(*Conv(Dims({SymExpr(1), SymExpr(1), SymExpr(2)}),
                     Dims({SymExpr(1), SymExpr(1), SymExpr(5)}), WindowAttrs{},
                     f)) == "[1, 1, unk_0]");
  }
  {
    // Floor vs ceil: in 5, kernel 2, stride 2. floor -> 2, ceil -> 3.
    FreshDims f;
    WindowAttrs floor_a;
    floor_a.strides = {2, 2};
    WindowAttrs ceil_a = floor_a;
    ceil_a.ceil_mode = true;
    const Shape x = Dims({SymExpr(1), SymExpr(8), SymExpr(5), SymExpr(5)});
    assert(Str(*Pool(x, {2, 2}, floor_a, f)) == "[1, 8, 2, 2]");
    assert(Str(*Pool(x, {2, 2}, ceil_a, f)) == "[1, 8, 3, 3]");
  }
  {
    // MaxPool on a symbolic batch: channels and batch are kept; 32-2 -> 16.
    FreshDims f;
    WindowAttrs a;
    a.strides = {2, 2};
    assert(Str(*Pool(Dims({batch, SymExpr(8), SymExpr(32), SymExpr(32)}),
                     {2, 2}, a, f)) == "[batch, 8, 16, 16]");
    // Absent strides default to 1: 32 - 3 + 1 = 30.
    assert(Str(*Pool(Dims({batch, SymExpr(8), SymExpr(32), SymExpr(32)}),
                     {3, 3}, WindowAttrs{}, f)) == "[batch, 8, 30, 30]");
    // kernel_shape with the wrong number of entries: no shape.
    assert(!Pool(Dims({batch, SymExpr(8), SymExpr(32), SymExpr(32)}), {2}, a, f)
                .has_value());
  }
  {
    assert(Str(*GlobalPool(Dims({batch, SymExpr(64), SymExpr(7),
                                 SymExpr(7)}))) == "[batch, 64, 1, 1]");
    assert(!GlobalPool(Dims({batch})).has_value());
  }

  std::cout << "all shaping_rules tests passed\n";
  return 0;
}
