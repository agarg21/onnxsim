"""fuse_skip_layer_norm: Add(x, skip) -> LayerNormalization  ==>  com.microsoft::SkipLayerNormalization.

Opt-in only (``extra_optimizers=["fuse_skip_layer_norm"]``); a default
``simplify()`` must leave the graph unchanged. The fused model is checked
numerically against the unfused one, both run through onnxruntime's CPU EP.
"""

import collections

import numpy as np
import onnxruntime as ort
import onnxsim.onnxsim_cpp2py_export as C
import pytest
from onnx import numpy_helper, parser

import onnxsim

# _list_other_optimizers only sees onnxsim's custom passes once they are registered.
C._list_other_optimizers()

PASS = "fuse_skip_layer_norm"
SHAPE = (2, 4, 8)
H = SHAPE[-1]


def simplify_isolated_extra(model, *pass_names, check_n=3):
    """Run only the named opt-in pass(es): every default pass is skipped.

    Local copy of tests/_formal_verify_common.simplify_isolated_extra, which
    skips the whole module when the optional z3 dependency is missing.
    """
    names = set(pass_names)
    assert names <= set(C._list_other_optimizers())
    sim_model, check_ok = onnxsim.simplify(
        model,
        check_n=check_n,
        extra_optimizers=sorted(names),
        skipped_optimizers=sorted(C._list_optimizers()),
    )
    assert check_ok, "simplified model failed onnxsim's own equivalence check"
    return sim_model, collections.Counter(n.op_type for n in sim_model.graph.node)


def _model(body, initializers=(), opset=17):
    model = parser.parse_model(f'<ir_version: 8, opset_import: ["": {opset}]> {body}')
    model.graph.initializer.extend(initializers)
    return model


def _init(name, arr):
    return numpy_helper.from_array(arr.astype(np.float32), name)


def _params(seed=0, with_bias=False):
    rng = np.random.default_rng(seed)
    inits = [
        _init("gamma", rng.standard_normal(H) + 1.0),
        _init("beta", rng.standard_normal(H)),
    ]
    if with_bias:
        inits.append(_init("bias", rng.standard_normal(H)))
    return inits


def _plain_model(**kw):
    return _model(
        """
        g (float[2,4,8] X, float[2,4,8] S, float[8] gamma, float[8] beta) => (float[2,4,8] Y)
        {
          A = Add(X, S)
          Y = LayerNormalization<axis = -1, epsilon = 0.001>(A, gamma, beta)
        }
        """,
        _params(),
        **kw,
    )


def _bias_model():
    return _model(
        """
        g (float[2,4,8] X, float[2,4,8] S, float[8] bias, float[8] gamma, float[8] beta) => (float[2,4,8] Y)
        {
          A = Add(X, S)
          B = Add(A, bias)
          Y = LayerNormalization<axis = -1, epsilon = 0.001>(B, gamma, beta)
        }
        """,
        _params(with_bias=True),
    )


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def _feeds(seed=1):
    rng = np.random.default_rng(seed)
    return {
        "X": rng.standard_normal(SHAPE).astype(np.float32),
        "S": rng.standard_normal(SHAPE).astype(np.float32),
    }


def _domains(model):
    return {(o.domain, o.version) for o in model.opset_import}


def test_fuses_residual_add_and_matches_unfused():
    model = _plain_model()
    sim, ops = simplify_isolated_extra(model, PASS)
    assert ops["SkipLayerNormalization"] == 1
    assert ops["Add"] == 0 and ops["LayerNormalization"] == 0
    node = next(n for n in sim.graph.node if n.op_type == "SkipLayerNormalization")
    assert node.domain == "com.microsoft"
    # Three inputs (no bias); beta is the LayerNormalization B input.
    assert len(node.input) == 4
    attrs = {a.name: a.f for a in node.attribute}
    assert attrs == {"epsilon": pytest.approx(0.001)}
    assert ("com.microsoft", 1) in _domains(sim)

    feeds = _feeds()
    np.testing.assert_allclose(
        _run(sim, feeds)[0], _run(model, feeds)[0], rtol=1e-5, atol=1e-5
    )


def test_fuses_bias_variant_and_matches_unfused():
    model = _bias_model()
    sim, ops = simplify_isolated_extra(model, PASS)
    assert ops["SkipLayerNormalization"] == 1
    assert ops["Add"] == 0 and ops["LayerNormalization"] == 0
    node = next(n for n in sim.graph.node if n.op_type == "SkipLayerNormalization")
    # input, skip, gamma, beta, bias.
    assert len(node.input) == 5
    feeds = _feeds(seed=2)
    np.testing.assert_allclose(
        _run(sim, feeds)[0], _run(model, feeds)[0], rtol=1e-5, atol=1e-5
    )


def test_declines_when_add_output_has_two_consumers():
    model = _model(
        """
        g (float[2,4,8] X, float[2,4,8] S, float[8] gamma, float[8] beta) => (float[2,4,8] Y, float[2,4,8] Z)
        {
          A = Add(X, S)
          Y = LayerNormalization<axis = -1, epsilon = 0.001>(A, gamma, beta)
          Z = Neg(A)
        }
        """,
        _params(),
    )
    sim, ops = simplify_isolated_extra(model, PASS)
    assert ops["SkipLayerNormalization"] == 0
    assert ops["Add"] == 1 and ops["LayerNormalization"] == 1
    feeds = _feeds(seed=3)
    for got, want in zip(_run(sim, feeds), _run(model, feeds)):
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)


def test_declines_when_add_output_is_graph_output():
    model = _model(
        """
        g (float[2,4,8] X, float[2,4,8] S, float[8] gamma, float[8] beta) => (float[2,4,8] Y, float[2,4,8] A)
        {
          A = Add(X, S)
          Y = LayerNormalization<axis = -1, epsilon = 0.001>(A, gamma, beta)
        }
        """,
        _params(),
    )
    sim, ops = simplify_isolated_extra(model, PASS)
    assert ops["SkipLayerNormalization"] == 0
    assert ops["Add"] == 1 and ops["LayerNormalization"] == 1


def test_declines_when_norm_mean_output_is_used():
    model = _model(
        """
        g (float[2,4,8] X, float[2,4,8] S, float[8] gamma, float[8] beta)
          => (float[2,4,8] Y, float[2,4,1] M, float[2,4,1] I)
        {
          A = Add(X, S)
          Y, M, I = LayerNormalization<axis = -1, epsilon = 0.001>(A, gamma, beta)
        }
        """,
        _params(),
    )
    sim, ops = simplify_isolated_extra(model, PASS)
    assert ops["SkipLayerNormalization"] == 0
    assert ops["LayerNormalization"] == 1


def test_declines_without_a_skip_operand():
    # Add(x, constant) with no second non-constant operand is a plain bias
    # add, not a residual: there is no skip tensor to fuse.
    model = _model(
        """
        g (float[2,4,8] X, float[8] bias, float[8] gamma, float[8] beta) => (float[2,4,8] Y)
        {
          A = Add(X, bias)
          Y = LayerNormalization<axis = -1, epsilon = 0.001>(A, gamma, beta)
        }
        """,
        _params(with_bias=True),
    )
    sim, ops = simplify_isolated_extra(model, PASS)
    assert ops["SkipLayerNormalization"] == 0
    assert ops["Add"] == 1 and ops["LayerNormalization"] == 1


def test_default_simplify_leaves_graph_unchanged():
    # The pass is opt-in: a default simplify() must neither fire it nor add
    # the com.microsoft domain.
    model = _plain_model()
    before = collections.Counter(n.op_type for n in model.graph.node)
    sim, ok = onnxsim.simplify(model, check_n=3)
    assert ok
    after = collections.Counter(n.op_type for n in sim.graph.node)
    assert after["SkipLayerNormalization"] == 0
    assert after["Add"] == before["Add"] == 1
    assert after["LayerNormalization"] == before["LayerNormalization"] == 1
    assert all(o.domain != "com.microsoft" for o in sim.opset_import)
    assert [n.op_type for n in sim.graph.node] == [n.op_type for n in model.graph.node]
    feeds = _feeds(seed=4)
    np.testing.assert_allclose(
        _run(sim, feeds)[0], _run(model, feeds)[0], rtol=1e-6, atol=1e-6
    )


def test_registered_as_opt_in_only():
    # Other (opt-in) pass: named in the opt-in list, absent from the default set.
    assert PASS in C._list_other_optimizers()
    assert PASS not in C._list_optimizers()
