# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Causal masks built in the graph from a dynamic length are proven causal.

The mask is computed from the sequence length with Trilu, Range and Greater
(the form a transformers-style causal mask exports to). The matcher proves it
is strictly causal and emits is_causal, with no mask input. Checked against
torch for several sequence lengths. Skipped when torch, onnxscript or
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

H, D = 2, 8
C = H * D


class _InGraphCausal(torch.nn.Module):
    def forward(self, x):
        B, S, _ = x.shape
        q = x.view(B, S, H, D).transpose(1, 2)
        sc = (q @ q.transpose(-1, -2)) / D**0.5
        # Causal mask from the length alone: fill where column > row.
        rows = torch.arange(S).view(-1, 1)
        cols = torch.arange(S + 1)
        fill = torch.full((S, S + 1), -3.4028234663852886e38)
        mask = torch.triu(fill, diagonal=1) * (cols > rows).to(fill.dtype)
        mask = mask[None, None, :, :S]
        y = torch.softmax(sc + mask, -1) @ q
        return y.transpose(1, 2).reshape(B, S, C)


def _export(module, args, path, dynamic=None):
    kwargs = {"opset_version": 23, "dynamo": True}
    if dynamic is not None:
        kwargs["dynamic_shapes"] = dynamic
    with contextlib.redirect_stdout(io.StringIO()):
        torch.onnx.export(module, args, str(path), **kwargs)
    return onnx.load(str(path))


def _run(model, x):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, {sess.get_inputs()[0].name: x})[0]


def test_in_graph_causal_mask_from_dynamic_length_is_is_causal(tmp_path):
    torch.manual_seed(0)
    module = _InGraphCausal().eval()
    x0 = torch.randn(1, 6, C)
    seq = torch.export.Dim("seq", min=2, max=32)
    raw = _export(module, (x0,), tmp_path / "causal.onnx", dynamic=({1: seq},))
    sim, ok = onnxsim.simplify(raw)
    assert ok
    nodes = [n for n in sim.graph.node if n.op_type == "Attention"]
    assert len(nodes) == 1
    causal = [a.i for a in nodes[0].attribute if a.name == "is_causal"]
    assert causal == [1]
    assert len(nodes[0].input) == 3
    for S in (2, 5, 11):
        x = torch.randn(1, S, C)
        with torch.no_grad():
            ref = module(x).numpy()
        np.testing.assert_allclose(_run(sim, x.numpy()), ref, rtol=1e-4, atol=1e-5)
