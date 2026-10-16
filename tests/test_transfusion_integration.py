"""Integration test for the ONNX -> onnxsim -> ONNX Runtime path used by Autoware's
lidar TransFusion node (autoware_lidar_transfusion, ONNX Runtime backend).

The node feeds the model three tensors that change shape every frame (the voxel
count is dynamic) and reads three fixed-shape outputs back. This test builds a
graph with the same I/O contract, simplifies it with onnxsim, and checks that the
simplified model still matches the original under ONNX Runtime for several voxel
counts. It is a stand-in for the real TransFusion weights, which are not in this
repository.
"""

import numpy as np
import onnx
import pytest
from onnx import parser

import onnxsim

ort = pytest.importorskip("onnxruntime")

POINTS_PER_PILLAR = 20
NUM_FEATURES = 4
NUM_COORD_VALUES = 2
NUM_CLASSES = 3
NUM_PROPOSALS = 8
NUM_BOX_VALUES = 7

OUTPUT_NAMES = ["cls_score0", "bbox_pred0", "dir_cls_pred0"]


def _transfusion_like_model():
    # Opset 12 keeps ReduceSum's axes as an attribute, which keeps the text form readable.
    # Inline Constant nodes hold the fixed output shapes; the voxel count NV stays symbolic.
    model = parser.parse_model(
        f"""
        <ir_version: 7, opset_import: ["": 12]>
        transfusion (float[NV, {POINTS_PER_PILLAR}, {NUM_FEATURES}] voxels,
                     int32[NV] num_points,
                     int32[NV, {NUM_COORD_VALUES}] coors)
          => (float[1, {NUM_CLASSES}, {NUM_PROPOSALS}] cls_score0,
              float[1, {NUM_BOX_VALUES}, {NUM_PROPOSALS}] bbox_pred0,
              float[1, 2, {NUM_PROPOSALS}] dir_cls_pred0)
        {{
          feat_sum = ReduceSum<axes = [0, 1, 2], keepdims = 0>(voxels)
          points = Cast<to = 1>(num_points)
          points_sum = ReduceSum<axes = [0], keepdims = 0>(points)
          coors_f = Cast<to = 1>(coors)
          coors_sum = ReduceSum<axes = [0, 1], keepdims = 0>(coors_f)
          partial = Add(feat_sum, points_sum)
          total = Add(partial, coors_sum)
          total_1d = Reshape(total, one_shape)
          cls_shape = Constant<value = int64[3] {{1, {NUM_CLASSES}, {NUM_PROPOSALS}}}>()
          box_shape = Constant<value = int64[3] {{1, {NUM_BOX_VALUES}, {NUM_PROPOSALS}}}>()
          dir_shape = Constant<value = int64[3] {{1, 2, {NUM_PROPOSALS}}}>()
          cls_score0 = Expand(total_1d, cls_shape)
          bbox_pred0 = Expand(total_1d, box_shape)
          dir_cls_pred0 = Expand(total_1d, dir_shape)
        }}
        """
    )
    one_shape = onnx.helper.make_tensor("one_shape", onnx.TensorProto.INT64, [1], [1])
    model.graph.initializer.append(one_shape)
    return model


def _frame(num_voxels, rng):
    return {
        "voxels": rng.standard_normal(
            (num_voxels, POINTS_PER_PILLAR, NUM_FEATURES)
        ).astype(np.float32),
        "num_points": rng.integers(1, POINTS_PER_PILLAR, size=(num_voxels,)).astype(
            np.int32
        ),
        "coors": rng.integers(0, 100, size=(num_voxels, NUM_COORD_VALUES)).astype(
            np.int32
        ),
    }


def _session(model):
    return ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )


def test_simplify_keeps_transfusion_io_contract():
    model = _transfusion_like_model()
    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok

    session = _session(sim_model)
    inputs = {i.name: (i.type, i.shape) for i in session.get_inputs()}
    assert inputs["voxels"] == (
        "tensor(float)",
        ["NV", POINTS_PER_PILLAR, NUM_FEATURES],
    )
    assert inputs["num_points"] == ("tensor(int32)", ["NV"])
    assert inputs["coors"] == ("tensor(int32)", ["NV", NUM_COORD_VALUES])
    assert [o.name for o in session.get_outputs()] == OUTPUT_NAMES


@pytest.mark.parametrize("num_voxels", [1, 7, 33])
def test_simplified_model_matches_original_under_onnxruntime(num_voxels):
    model = _transfusion_like_model()
    sim_model, check_ok = onnxsim.simplify(model)
    assert check_ok

    feeds = _frame(num_voxels, np.random.default_rng(num_voxels))
    want = _session(model).run(OUTPUT_NAMES, feeds)
    got = _session(sim_model).run(OUTPUT_NAMES, feeds)

    for name, w, g in zip(OUTPUT_NAMES, want, got):
        assert g.shape == w.shape, name
        np.testing.assert_allclose(g, w, rtol=1e-5, atol=1e-5, err_msg=name)


def test_simplify_reduces_nodes_and_keeps_dynamic_voxel_dim():
    model = _transfusion_like_model()
    sim_model, _ = onnxsim.simplify(model)

    assert len(sim_model.graph.node) <= len(model.graph.node)
    dims = sim_model.graph.input[0].type.tensor_type.shape.dim
    assert dims[0].dim_param == "NV"
