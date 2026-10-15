# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Attention fusion with a runtime additive float mask of any broadcastable shape.

The mask is a graph input here, so it is not a constant: the matcher emits it
as Attention's attn_mask as-is. The module has the head split and merge that
the causal matcher expects. Checked against torch for each mask shape and for
a dynamic sequence length. Skipped when torch, onnxscript or onnxruntime is
missing.
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


class _MaskedAttention(torch.nn.Module):
    def forward(self, x, mask):
        B, S, _ = x.shape
        q = x.view(B, S, H, D).transpose(1, 2)
        sc = (q @ q.transpose(-1, -2)) / D**0.5 + mask
        y = torch.softmax(sc, -1) @ q
        return y.transpose(1, 2).reshape(B, S, C)


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


@pytest.mark.parametrize("shape", ["1x1xSxS", "1xHxSxS"])
def test_runtime_mask_of_any_broadcastable_shape_fuses_and_matches_torch(
    tmp_path, shape
):
    torch.manual_seed(0)
    S = 6
    mask_shape = {
        "1x1xSxS": (1, 1, S, S),
        "1xHxSxS": (1, H, S, S),
    }[shape]
    module = _MaskedAttention().eval()
    x = torch.randn(1, S, C)
    mask = torch.randn(*mask_shape)
    with torch.no_grad():
        ref = module(x, mask).numpy()
    raw = _export(module, (x, mask), tmp_path / "m.onnx")
    sim, ok = onnxsim.simplify(raw)
    assert ok
    assert "Attention" in _ops(sim)
    assert "Softmax" not in _ops(sim)
    feeds = {"x": x.numpy(), "mask": mask.numpy()}
    np.testing.assert_allclose(_run(sim, feeds), ref, rtol=1e-4, atol=1e-5)


def test_runtime_mask_with_dynamic_sequence_length_matches_torch(tmp_path):
    torch.manual_seed(0)
    module = _MaskedAttention().eval()
    S0 = 6
    x = torch.randn(1, S0, C)
    mask = torch.randn(1, 1, S0, S0)
    seq = torch.export.Dim("seq", min=2, max=32)
    dyn = ({1: seq}, {2: seq, 3: seq})
    raw = _export(module, (x, mask), tmp_path / "dyn.onnx", dynamic=dyn)
    sim, ok = onnxsim.simplify(raw)
    assert ok
    assert "Attention" in _ops(sim)
    for S in (3, 9):
        x = torch.randn(1, S, C)
        mask = torch.randn(1, 1, S, S)
        with torch.no_grad():
            ref = module(x, mask).numpy()
        feeds = {"x": x.numpy(), "mask": mask.numpy()}
        np.testing.assert_allclose(_run(sim, feeds), ref, rtol=1e-4, atol=1e-5)


def test_runtime_mask_broadcasting_over_the_query_axis_declines_and_stays_correct(
    tmp_path,
):
    # [1, 1, 1, S] covers only one query row: ORT's Attention cannot take it,
    # and an Expand would materialize a full score-sized tensor, so it stays
    # unfused. The graph must still be correct.
    torch.manual_seed(0)
    S = 6
    module = _MaskedAttention().eval()
    x = torch.randn(1, S, C)
    mask = torch.randn(1, 1, 1, S)
    with torch.no_grad():
        ref = module(x, mask).numpy()
    sim, ok = onnxsim.simplify(_export(module, (x, mask), tmp_path / "b.onnx"))
    assert ok
    assert "Attention" not in _ops(sim)
    np.testing.assert_allclose(
        _run(sim, {"x": x.numpy(), "mask": mask.numpy()}), ref, rtol=1e-4, atol=1e-5
    )
