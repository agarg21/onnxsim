"""Tests for the ``fuse_parallel_matmuls`` C++ pass
(onnxsim/passes/fuse_parallel_matmuls.h).

N >= 2 MatMuls reading the same activation with constant 2-D weights of the
same K become one MatMul against the concatenated weights plus a Split --
fuse_qkv generalized to any count and unequal widths (grouped-query attention's
q/k/v, the MLP's gate/up). Opt-in via ``extra_optimizers=["fuse_parallel_matmuls"]``.

Models are built with ``onnx.parser`` (AGENTS.md); random weights are numpy
initializers attached after parsing.
"""

import numpy as np
import pytest
from onnx import numpy_helper, parser
from onnx.reference import ReferenceEvaluator

import onnxsim

K = 32


def _model(body, weights, opset=17):
    model = parser.parse_model(
        f"""
        <ir_version: 8, opset_import: ["": {opset}]>
        {body}
        """
    )
    rng = np.random.RandomState(0)
    for name, cols in weights.items():
        w = rng.randn(K, cols).astype(np.float32) * 0.1
        model.graph.initializer.append(numpy_helper.from_array(w, name))
    return model


def _ops(model):
    return [n.op_type for n in model.graph.node]


def _run(model, feeds):
    return ReferenceEvaluator(model).run(None, feeds)


def _simplify(model, feeds):
    sim, ok = onnxsim.simplify(
        model,
        check_n=1,
        input_data=feeds,
        extra_optimizers=["fuse_parallel_matmuls"],
    )
    assert ok
    for want, got in zip(_run(model, feeds), _run(sim, feeds)):
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)
    return sim


def _decls(weights):
    return ", ".join(f"float[{K},{c}] {n}" for n, c in weights.items())


def _x(rows=4):
    return {"x": np.random.RandomState(1).randn(rows, K).astype(np.float32)}


def test_gate_up_pair_from_a_computed_activation():
    w = {"wg": 48, "wu": 48}
    model = _model(
        f"""
        g (float[4,{K}] x) => (float[4,48] y)
        <{_decls(w)}>
        {{
          h = Relu(x)
          gate = MatMul(h, wg)
          up = MatMul(h, wu)
          y = Mul(gate, up)
        }}
        """,
        w,
    )
    sim = _simplify(model, _x())
    assert _ops(sim).count("MatMul") == 1
    assert "Split" in _ops(sim)


def test_gqa_qkv_unequal_widths_on_a_graph_input():
    """576/192/192-like widths (here 64/16/16), read straight from the graph
    input -- both cases fuse_qkv declines."""
    w = {"wq": 64, "wk": 16, "wv": 16}
    model = _model(
        f"""
        g (float[4,{K}] x) => (float[4,64] q, float[4,16] k, float[4,16] v)
        <{_decls(w)}>
        {{
          q = MatMul(x, wq)
          k = MatMul(x, wk)
          v = MatMul(x, wv)
        }}
        """,
        w,
    )
    sim = _simplify(model, _x())
    assert _ops(sim).count("MatMul") == 1
    split = next(n for n in sim.graph.node if n.op_type == "Split")
    sizes = next(
        numpy_helper.to_array(i)
        for i in sim.graph.initializer
        if i.name == split.input[1]
    )
    assert sizes.tolist() == [64, 16, 16]
    assert [o.name for o in sim.graph.output] == ["q", "k", "v"]


def test_five_way():
    w = {f"w{i}": c for i, c in enumerate([64, 16, 16, 64, 16])}
    outs = ", ".join(f"float[4,{c}] y{i}" for i, c in enumerate(w.values()))
    body = "\n".join(f"y{i} = MatMul(h, w{i})" for i in range(len(w)))
    model = _model(
        f"""
        g (float[4,{K}] x) => ({outs})
        <{_decls(w)}>
        {{
          h = Tanh(x)
          {body}
        }}
        """,
        w,
    )
    sim = _simplify(model, _x())
    assert _ops(sim).count("MatMul") == 1


def test_ineligible_consumers_are_left_alone():
    """A runtime weight, and the activation as MatMul's second operand, do not
    join the group; the two constant-weight consumers still fuse."""
    w = {"wa": 24, "wb": 8}
    model = _model(
        f"""
        g (float[4,{K}] x, float[{K},8] wr)
          => (float[4,24] a, float[4,8] b, float[4,8] r, float[{K},{K}] t)
        <{_decls(w)}>
        {{
          h = Relu(x)
          a = MatMul(h, wa)
          b = MatMul(h, wb)
          r = MatMul(h, wr)
          ht = Transpose(h)
          t = MatMul(ht, h)
        }}
        """,
        w,
    )
    feeds = _x()
    feeds["wr"] = np.random.RandomState(2).randn(K, 8).astype(np.float32)
    sim = _simplify(model, feeds)
    # the fused one, the runtime-weight one, and Transpose(h) @ h
    assert _ops(sim).count("MatMul") == 3


@pytest.mark.parametrize("opset", [11, 13, 18])
def test_split_sizes_follow_the_opset(opset):
    w = {"wa": 16, "wb": 8}
    model = _model(
        f"""
        g (float[4,{K}] x) => (float[4,16] a, float[4,8] b)
        <{_decls(w)}>
        {{
          h = Relu(x)
          a = MatMul(h, wa)
          b = MatMul(h, wb)
        }}
        """,
        w,
        opset=opset,
    )
    sim = _simplify(model, _x())
    split = next(n for n in sim.graph.node if n.op_type == "Split")
    if opset < 13:
        assert [a.ints for a in split.attribute if a.name == "split"] == [[16, 8]]
    else:
        assert len(split.input) == 2


def test_not_run_by_default():
    w = {"wa": 16, "wb": 8}
    model = _model(
        f"""
        g (float[4,{K}] x) => (float[4,16] a, float[4,8] b)
        <{_decls(w)}>
        {{
          h = Relu(x)
          a = MatMul(h, wa)
          b = MatMul(h, wb)
        }}
        """,
        w,
    )
    sim, ok = onnxsim.simplify(model)
    assert ok
    assert _ops(sim).count("MatMul") == 2
