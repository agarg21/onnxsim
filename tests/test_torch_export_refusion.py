# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Re-fusion of attention in real ``torch.onnx.export`` (dynamo) models.

The synthetic graphs in ``test_fuse_attention*.py`` cover the exporter's
shapes by hand; these build actual torch modules, export them, simplify, and
check numerics against torch itself. Skipped when torch, onnxscript or
onnxruntime is missing.
"""

import contextlib
import io

import numpy as np
import onnx
import pytest

import onnxsim

torch = pytest.importorskip("torch")
pytest.importorskip("onnxscript")
ort = pytest.importorskip("onnxruntime")

import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

H, D = 4, 16
C = H * D


class _ManualCausal(nn.Module):
    """Hand-written causal attention: scores.masked_fill(~tril, -inf)."""

    def __init__(self, seq, diagonal=0):
        super().__init__()
        self.q = nn.Linear(C, C)
        self.k = nn.Linear(C, C)
        self.v = nn.Linear(C, C)
        self.o = nn.Linear(C, C)
        # diagonal=1 also attends the superdiagonal: not the causal pattern.
        self.register_buffer(
            "mask", torch.tril(torch.ones(seq, seq, dtype=torch.bool), diagonal)
        )

    def forward(self, x):
        B, S, _ = x.shape
        q = self.q(x).view(B, S, H, D).transpose(1, 2)
        k = self.k(x).view(B, S, H, D).transpose(1, 2)
        v = self.v(x).view(B, S, H, D).transpose(1, 2)
        s = (q @ k.transpose(-1, -2)) / D**0.5
        s = s.masked_fill(~self.mask[:S, :S], float("-inf"))
        y = torch.softmax(s, -1) @ v
        return self.o(y.transpose(1, 2).reshape(B, S, C))


class _SDPACausal(nn.Module):
    """F.scaled_dot_product_attention(is_causal=True)."""

    def __init__(self):
        super().__init__()
        self.q = nn.Linear(C, C)
        self.k = nn.Linear(C, C)
        self.v = nn.Linear(C, C)
        self.o = nn.Linear(C, C)

    def forward(self, x):
        B, S, _ = x.shape
        q = self.q(x).view(B, S, H, D).transpose(1, 2)
        k = self.k(x).view(B, S, H, D).transpose(1, 2)
        v = self.v(x).view(B, S, H, D).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, S, C))


def _export(module, x, path, dynamic=False):
    kwargs = {"opset_version": 23, "dynamo": True}
    if dynamic:
        kwargs["dynamic_shapes"] = ({1: torch.export.Dim("seq")},)
    with contextlib.redirect_stdout(io.StringIO()):
        torch.onnx.export(module, (x,), str(path), **kwargs)
    return onnx.load(str(path))


def _run_ort(model, x):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, {sess.get_inputs()[0].name: x})[0]


def _attention_nodes(model):
    return [n for n in model.graph.node if n.op_type == "Attention"]


def _attr_int(node, name, default=0):
    for a in node.attribute:
        if a.name == name:
            return int(onnx.helper.get_attribute_value(a))
    return default


@pytest.fixture
def seq8_input():
    torch.manual_seed(0)
    return torch.randn(1, 8, C)


def test_masked_fill_causal_export_refuses_to_is_causal_attention(tmp_path, seq8_input):
    module = _ManualCausal(8).eval()
    with torch.no_grad():
        expected = module(seq8_input).numpy()
    raw = _export(module, seq8_input, tmp_path / "manual_static.onnx")
    assert _attention_nodes(raw) == []  # the exporter emits the decomposed form

    simplified, ok = onnxsim.simplify(raw)
    assert ok
    nodes = _attention_nodes(simplified)
    assert len(nodes) == 1
    assert _attr_int(nodes[0], "is_causal") == 1
    assert len(nodes[0].input) == 3
    assert not any(n.op_type == "Softmax" for n in simplified.graph.node)
    np.testing.assert_allclose(
        _run_ort(simplified, seq8_input.numpy()), expected, atol=1e-5
    )


def test_sdpa_causal_export_is_already_attention_and_stays_correct(
    tmp_path, seq8_input
):
    module = _SDPACausal().eval()
    with torch.no_grad():
        expected = module(seq8_input).numpy()
    raw = _export(module, seq8_input, tmp_path / "sdpa_static.onnx")
    # torch's own exporter emits the standard Attention op for SDPA.
    assert len(_attention_nodes(raw)) == 1

    simplified, ok = onnxsim.simplify(raw)
    assert ok
    assert len(_attention_nodes(simplified)) == 1
    np.testing.assert_allclose(
        _run_ort(simplified, seq8_input.numpy()), expected, atol=1e-5
    )


def test_masked_fill_causal_export_with_dynamic_seq_fuses(tmp_path):
    # The sequence length is a runtime dimension (dynamic_shapes on the input),
    # so the mask is a Slice of the 8x8 buffer by Shape(x). The matcher accepts
    # it because the slice length is the attention's own sequence length.
    torch.manual_seed(0)
    module = _ManualCausal(8).eval()
    seq0 = torch.randn(1, 8, C)
    raw = _export(module, seq0, tmp_path / "manual_dynamic.onnx", dynamic=True)
    simplified, ok = onnxsim.simplify(raw)
    assert ok
    nodes = _attention_nodes(simplified)
    assert len(nodes) == 1
    assert _attr_int(nodes[0], "is_causal") == 1
    assert len(nodes[0].input) == 3
    assert not any(n.op_type == "Softmax" for n in simplified.graph.node)
    # The buffer has 8 rows, so valid lengths are 1..8; torch is the reference.
    for seq in (1, 3, 5, 8):
        x = torch.randn(1, seq, C)
        with torch.no_grad():
            expected = module(x).numpy()
        np.testing.assert_allclose(_run_ort(simplified, x.numpy()), expected, atol=1e-5)


def test_non_causal_dynamic_mask_export_never_becomes_is_causal(tmp_path):
    # Same dynamic-length export, but the buffer attends the superdiagonal too.
    # It must not fuse to is_causal=1; whatever it becomes must stay correct.
    torch.manual_seed(0)
    module = _ManualCausal(8, diagonal=1).eval()
    seq0 = torch.randn(1, 8, C)
    raw = _export(module, seq0, tmp_path / "noncausal_dynamic.onnx", dynamic=True)
    simplified, ok = onnxsim.simplify(raw)
    assert ok
    assert all(_attr_int(n, "is_causal") == 0 for n in _attention_nodes(simplified))
    for seq in (1, 3, 5, 8):
        x = torch.randn(1, seq, C)
        with torch.no_grad():
            expected = module(x).numpy()
        np.testing.assert_allclose(_run_ort(simplified, x.numpy()), expected, atol=1e-5)
