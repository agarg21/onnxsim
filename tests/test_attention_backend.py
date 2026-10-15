"""Tests for ``simplify(..., attention_fusion_backend=...)``.

The option selects which ``Attention`` form ``fuse_attention``
(``onnxsim/passes/fuse_attention.h``) may emit:

  * ``com.microsoft::Attention`` -- unmasked hand-written self-attention
    (``MatchAttention``),
  * ``ai.onnx::Attention`` (opset 23) -- the causal path, either
    ``is_causal=1`` with no mask input, or with an ``attn_mask`` input.

Each test builds a small model in one of three shapes and checks which
form(s) appear for every backend value, and that the simplified model still
matches the original numerically. The model builders and numeric helpers are
reused from ``test_fuse_attention.py`` and ``test_fuse_attention_causal_kv.py``.
"""

import onnx
import pytest
from test_fuse_attention import _attention_model
from test_fuse_attention_causal_kv import (
    _attr_int,
    _causal_kv_model,
    _check_equal,
    _feeds,
)

import onnxsim

ort = pytest.importorskip("onnxruntime")


def _self_attention():
    # Unmasked self-attention: the MatchAttention (com.microsoft) shape.
    return _attention_model(opset=23)


def _causal():
    # Square lower-triangular constant mask: ai.onnx Attention, is_causal=1.
    return _causal_kv_model(mask="static_causal")


def _masked():
    # Constant non-causal padding mask: ai.onnx Attention with attn_mask.
    return _causal_kv_model(mask="constant_padding")


MODELS = {"self": _self_attention, "causal": _causal, "masked": _masked}

COM_MS = "com.microsoft"
ONNX_CAUSAL = "ai.onnx:is_causal"
ONNX_MASK = "ai.onnx:attn_mask"

# Which Attention forms each backend must produce on each model shape.
EXPECTED = {
    "auto": {
        "self": {COM_MS},
        "causal": {ONNX_CAUSAL},
        "masked": {ONNX_MASK},
    },
    "onnx": {
        "self": set(),
        "causal": {ONNX_CAUSAL},
        "masked": {ONNX_MASK},
    },
    "ort_contrib": {
        "self": {COM_MS},
        "causal": set(),
        "masked": set(),
    },
    "none": {
        "self": set(),
        "causal": set(),
        "masked": set(),
    },
    # Same forms as ort_contrib. The com.microsoft form never carries the
    # attn_mask/mask_index input that onnxruntime-web's WebGPU kernel rejects,
    # and the ai.onnx causal form is excluded, so a masked case is declined.
    "webgpu": {
        "self": {COM_MS},
        "causal": set(),
        "masked": set(),
    },
}


def _form(node):
    if node.domain == COM_MS:
        return COM_MS
    if len(node.input) > 3 and node.input[3]:
        return ONNX_MASK
    if _attr_int(node, "is_causal"):
        return ONNX_CAUSAL
    return "ai.onnx:plain"


def _forms(model):
    return {_form(n) for n in model.graph.node if n.op_type == "Attention"}


def _has_no_mask_inputs(model):
    # Every Attention node has no attn_mask (input 3) or mask/past (input 4+)
    # input wired up -- the WebGPU kernel's unsupported inputs.
    return all(
        not any(node.input[3:])
        for node in model.graph.node
        if node.op_type == "Attention"
    )


@pytest.mark.parametrize("case", ["self", "causal", "masked"])
@pytest.mark.parametrize("backend", list(EXPECTED))
def test_attention_form_per_backend(backend, case):
    model = MODELS[case]()
    simplified, ok = onnxsim.simplify(model, attention_fusion_backend=backend)
    assert ok
    onnx.checker.check_model(simplified)
    assert _forms(simplified) == EXPECTED[backend][case]
    _check_equal(model, simplified, _feeds(200))


@pytest.mark.parametrize("case", ["self", "causal", "masked"])
def test_default_is_identical_to_explicit_auto(case):
    model = MODELS[case]()
    default, _ = onnxsim.simplify(model)
    explicit, _ = onnxsim.simplify(model, attention_fusion_backend="auto")
    assert default.SerializeToString() == explicit.SerializeToString()


def test_webgpu_declines_masked_but_keeps_unmasked_self_attention():
    masked, _ = onnxsim.simplify(_masked(), attention_fusion_backend="webgpu")
    assert _forms(masked) == set()
    assert _has_no_mask_inputs(masked)

    unmasked, _ = onnxsim.simplify(_self_attention(), attention_fusion_backend="webgpu")
    attn = [n for n in unmasked.graph.node if n.op_type == "Attention"]
    assert len(attn) == 1
    assert attn[0].domain == COM_MS
    assert len(attn[0].input) == 3  # input, weights, bias: no mask/past.
    assert _has_no_mask_inputs(unmasked)


@pytest.mark.parametrize("case", ["self", "causal", "masked"])
def test_webgpu_never_emits_attn_mask(case):
    simplified, _ = onnxsim.simplify(MODELS[case](), attention_fusion_backend="webgpu")
    assert _has_no_mask_inputs(simplified)


def test_backend_does_not_leak_between_calls():
    model = _self_attention()
    none_out, _ = onnxsim.simplify(model, attention_fusion_backend="none")
    assert _forms(none_out) == set()
    after, _ = onnxsim.simplify(model)
    assert _forms(after) == {COM_MS}


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="attention_fusion_backend"):
        onnxsim.simplify(_self_attention(), attention_fusion_backend="bogus")
