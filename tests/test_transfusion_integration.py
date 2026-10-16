"""Integration test for onnxsim on Autoware's lidar TransFusion model.

Autoware's lidar TransFusion node (autoware_lidar_transfusion, ONNX Runtime backend)
runs AutowareFoundation/lidar_transfusion's transfusion.onnx. This test simplifies that
real model with onnxsim and checks it under ONNX Runtime.

The model is about 170 MB, so the test is skipped unless TRANSFUSION_ONNX points at a
local copy (the weekly workflow fetches it at a pinned revision).

Final outputs are not compared element-wise. The head selects peaks with
``Equal(sigmoid(heatmap), maxpool(heatmap))``, which depends on exact float ties.
Folding BatchNorm into the preceding ConvTranspose changes float rounding by about
1e-6, which flips a few hundred peak decisions and so changes the selected
proposals. The test therefore checks the continuous heatmap tensor element-wise,
and checks the final outputs for interface and sanity only.
"""

import os

import numpy as np
import onnx
import pytest

import onnxsim

ort = pytest.importorskip("onnxruntime")

MODEL_PATH = os.environ.get("TRANSFUSION_ONNX")
pytestmark = pytest.mark.skipif(
    not MODEL_PATH or not os.path.isfile(MODEL_PATH),
    reason="set TRANSFUSION_ONNX to a local transfusion.onnx to run",
)

# Heatmap tensor before the peak-selection Equal. It is continuous, so it is compared
# element-wise.
HEATMAP_TENSOR = "/pts_bbox_head/Sigmoid_output_0"
OUTPUT_NAMES = ["cls_score0", "bbox_pred0", "dir_cls_pred0"]
NUM_VOXELS = 50
HEATMAP_ATOL = 1e-5


@pytest.fixture(scope="module")
def original():
    return onnx.load(MODEL_PATH)


@pytest.fixture(scope="module")
def simplified(original):
    sim, check_ok = onnxsim.simplify(original, check_n=0)
    assert check_ok
    return sim


def _expose(model, names):
    exposed = onnx.ModelProto()
    exposed.CopyFrom(model)
    for name in names:
        exposed.graph.output.append(onnx.helper.make_empty_tensor_value_info(name))
    return exposed


def _frame(num_voxels):
    rng = np.random.default_rng(0)
    return {
        "voxels": rng.standard_normal((num_voxels, 20, 5)).astype(np.float32),
        "num_points": rng.integers(1, 20, size=(num_voxels,)).astype(np.int32),
        "coors": np.column_stack(
            [np.zeros(num_voxels), rng.integers(0, 400, (num_voxels, 3))]
        ).astype(np.int32),
    }


def _run(model, names, feeds):
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return dict(zip(names, session.run(names, feeds)))


def test_simplify_keeps_transfusion_interface(original, simplified):
    session = ort.InferenceSession(
        simplified.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    inputs = {i.name: (i.type, i.shape) for i in session.get_inputs()}
    assert inputs["voxels"] == ("tensor(float)", ["voxels_num", 20, 5])
    assert inputs["num_points"] == ("tensor(int32)", ["voxels_num"])
    assert inputs["coors"] == ("tensor(int32)", ["voxels_num", 4])
    outputs = {o.name: o.shape for o in session.get_outputs()}
    assert outputs == {
        "cls_score0": [1, 5, 500],
        "bbox_pred0": [1, 8, 500],
        "dir_cls_pred0": [1, 2, 500],
    }


def test_simplify_reduces_graph(original, simplified):
    assert len(simplified.graph.node) < len(original.graph.node)


def test_heatmap_matches_original(original, simplified):
    feeds = _frame(NUM_VOXELS)
    want = _run(_expose(original, [HEATMAP_TENSOR]), [HEATMAP_TENSOR], feeds)[
        HEATMAP_TENSOR
    ]
    got = _run(_expose(simplified, [HEATMAP_TENSOR]), [HEATMAP_TENSOR], feeds)[
        HEATMAP_TENSOR
    ]
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=0, atol=HEATMAP_ATOL)


def test_final_outputs_are_finite_and_shaped(simplified):
    outputs = _run(simplified, OUTPUT_NAMES, _frame(NUM_VOXELS))
    for name in OUTPUT_NAMES:
        assert np.all(np.isfinite(outputs[name])), name
