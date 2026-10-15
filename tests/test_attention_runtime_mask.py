# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Attention fusion with a runtime additive float mask of any broadcastable shape.

The mask is a graph input here, so it is not a constant: the matcher emits it
as Attention's attn_mask as-is. Checked against torch for each mask shape and
for a dynamic sequence length. Skipped when torch, onnxscript or onnxruntime
is missing.
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


class _MaskedAttention(torch.nn.Module):
    def forward(self, q, k, v, mask):
        sc = (q @ k.transpose(-1, -2)) / D**0.5 + mask
        return torch.softmax(sc, -1) @ v


def _export(module, args, path, dynamic=None):
    kwargs = {"opset_version": 23, "dynamo": True}
    if dynamic is not None:
        kwargs["dynamic_shapes"] = dynamic
    with contextlib.redirect_stdout(io.StringIO()):
        torch.onnx.export(module, args, str(path), **kwargs)
    return onnx.load(str(path))


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, {n.name: feeds[n.name] for n in sess.get_inputs()})[0]


def _ops(model):
    return [n.op_type for n in model.graph.node]


@pytest.mark.parametrize("mask_shape", ["1x1xSxS", "1xHxSxS", "1x1x1xS"])
def test_runtime_mask_of_any_broadcastable_shape_fuses_and_matches_torch(
    tmp_path, mask_shape
):
    torch.manual_seed(0)
    S = 6
    shape = {
        "1x1xSxS": (1, 1, S, S),
        "1xHxSxS": (1, H, S, S),
        "1x1x1xS": (1, 1, 1, S),
    }[mask_shape]
    module = _MaskedAttention().eval()
    q, k, v = (torch.randn(1, H, S, D) for _ in range(3))
    mask = torch.randn(*shape)
    with torch.no_grad():
        ref = module(q, k, v, mask).numpy()
    raw = _export(module, (q, k, v, mask), tmp_path / "m.onnx")
    sim, ok = onnxsim.simplify(raw)
    assert ok
    assert "Attention" in _ops(sim)
    assert "Softmax" not in _ops(sim)
    feeds = {"q": q.numpy(), "k": k.numpy(), "v": v.numpy(), "mask": mask.numpy()}
    np.testing.assert_allclose(_run(sim, feeds), ref, rtol=1e-4, atol=1e-5)


def test_runtime_mask_with_dynamic_sequence_length_matches_torch(tmp_path):
    torch.manual_seed(0)
    module = _MaskedAttention().eval()
    S0 = 6
    q, k, v = (torch.randn(1, H, S0, D) for _ in range(3))
    mask = torch.randn(1, 1, S0, S0)
    seq = torch.export.Dim("seq", min=2, max=32)
    dyn = ({2: seq}, {2: seq}, {2: seq}, {2: seq, 3: seq})
    raw = _export(module, (q, k, v, mask), tmp_path / "dyn.onnx", dynamic=dyn)
    sim, ok = onnxsim.simplify(raw)
    assert ok
    assert "Attention" in _ops(sim)
    for S in (3, 9):
        q, k, v = (torch.randn(1, H, S, D) for _ in range(3))
        mask = torch.randn(1, 1, S, S)
        with torch.no_grad():
            ref = module(q, k, v, mask).numpy()
        feeds = {"q": q.numpy(), "k": k.numpy(), "v": v.numpy(), "mask": mask.numpy()}
        np.testing.assert_allclose(_run(sim, feeds), ref, rtol=1e-4, atol=1e-5)
