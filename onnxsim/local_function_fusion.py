# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Outline a residual Add feeding a LayerNormalization into a model-local function.

``Add(x, skip) -> LayerNormalization`` is a memory-bound pair: the sum is
written by the Add and read again by the norm. Packaging the pair as a model-
local function ``FunctionProto`` makes the fused unit explicit: the call is one
node, so the per-kernel traffic model (``kernel_traffic_breakdown`` with
``decompose_functions=False``) charges only its boundary tensors, and the sum
drops out of the total.

The gate is the traffic model itself: each pair is outlined on its own, the
model's total before and after is compared, and only pairs whose measured
saving reaches the threshold are kept together.

The saving is only real if the runtime runs the call as one kernel. Runtimes
that inline model-local functions back to primitive ops (ORT does) keep the
sum in memory again, so ``decompose_functions=True`` costs the same as the
original graph. By default a pair is only outlined when a fused kernel can take
it (the conditions of ``fuse_skip_layer_norm``, see ``_has_fused_target``); the
kernel itself is still the runtime's responsibility.
"""

import copy
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

import onnx
from onnx import helper

from onnxsim.model_info import (
    _collect_dtypes,
    _collect_shapes,
    _is_std_domain,
    _known,
    kernel_traffic_breakdown,
)

__all__ = ["LOCAL_DOMAIN", "OutlineResult", "outline_residual_add_norm"]

LOCAL_DOMAIN = "onnxsim.local"
_NORM = "LayerNormalization"


@dataclass(frozen=True)
class OutlineResult:
    """``model``: the outlined copy. ``saved_bytes``: the traffic model's measured
    saving for each outlined pair, in graph order."""

    model: onnx.ModelProto
    saved_bytes: Tuple[int, ...]

    @property
    def count(self) -> int:
        return len(self.saved_bytes)


def _has_subgraph(node: onnx.NodeProto) -> bool:
    return any(a.HasField("g") or len(a.graphs) for a in node.attribute)


def _default_opset(model: onnx.ModelProto) -> int:
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            return opset.version
    raise ValueError("model has no default ONNX opset import")


def _build_function(
    name: str,
    add: onnx.NodeProto,
    norm: onnx.NodeProto,
    opset: int,
) -> onnx.FunctionProto:
    """``name(x, skip, scale[, bias]) -> outputs of norm`` whose body is the
    Add followed by the LayerNormalization, with the norm's attributes copied.
    """
    inputs = ["x", "skip", "scale"]
    if len(norm.input) > 2 and norm.input[2]:
        inputs.append("bias")
    body_norm_inputs = ["sum", "scale"] + (["bias"] if len(inputs) == 4 else [])
    body_outputs: List[str] = []
    formal_outputs: List[str] = []
    for k, out in enumerate(norm.output):
        if out:
            body_outputs.append(f"out{k}")
            formal_outputs.append(f"out{k}")
        else:
            body_outputs.append("")
    body = [
        helper.make_node("Add", ["x", "skip"], ["sum"]),
        helper.make_node(_NORM, body_norm_inputs, body_outputs, name="norm"),
    ]
    body[1].attribute.extend(copy.deepcopy(list(norm.attribute)))
    return helper.make_function(
        LOCAL_DOMAIN,
        name,
        inputs,
        formal_outputs,
        body,
        [helper.make_opsetid("", opset)],
    )


_FLOAT_BYTES = 4  # DTypeMap stores bytes per element


def _has_fused_target(
    add: onnx.NodeProto,
    norm: onnx.NodeProto,
    shapes: Dict[str, List],
    dtypes: Dict[str, int],
    opset: int,
) -> bool:
    """Whether the pair is one a fused kernel can take, i.e. what
    ``fuse_skip_layer_norm`` requires: opset >= 17, a rank-3 float sum with
    identical static shapes for both Add operands, a last-axis norm whose
    extra outputs are unused, and 1-D scale and bias."""
    if opset < 17:
        return False
    x, skip = add.input[0], add.input[1]
    sx, ss = shapes.get(x), shapes.get(skip)
    if not (_known(sx) and _known(ss)) or len(sx) != 3 or sx != ss:
        return False
    if dtypes.get(x) != _FLOAT_BYTES or dtypes.get(skip) != _FLOAT_BYTES:
        return False
    if any(out for out in norm.output[1:]):
        return False
    axis = next((a.i for a in norm.attribute if a.name == "axis"), -1)
    if axis not in (-1, 2):
        return False
    for name in norm.input[1:]:
        if name and not (_known(shapes.get(name)) and len(shapes[name]) == 1):
            return False
    return True


def _eligible_pairs(model: onnx.ModelProto) -> List[Tuple[int, int]]:
    """Structural matches ``(add_index, norm_index)``: the Add has two non-empty
    inputs and its only use is the norm, which is not a graph output; neither
    node has a subgraph."""
    graph = model.graph
    graph_outputs = {o.name for o in graph.output}
    producer: Dict[str, int] = {}
    consumers: Dict[str, List[int]] = {}
    for i, node in enumerate(graph.node):
        for name in node.input:
            if name:
                consumers.setdefault(name, []).append(i)
        for name in node.output:
            if name:
                producer[name] = i

    pairs: List[Tuple[int, int]] = []
    used: Set[int] = set()
    for i, norm in enumerate(graph.node):
        if not (_is_std_domain(norm) and norm.op_type == _NORM and norm.input):
            continue
        x = norm.input[0]
        j = producer.get(x)
        if j is None or j in used or i in used:
            continue
        add = graph.node[j]
        if not (
            _is_std_domain(add)
            and add.op_type == "Add"
            and len(add.input) == 2
            and all(add.input)
        ):
            continue
        if _has_subgraph(add) or _has_subgraph(norm):
            continue
        if consumers.get(x, []) != [i] or x in graph_outputs:
            continue
        pairs.append((j, i))
        used.update((j, i))
    return pairs


def _outline(
    model: onnx.ModelProto, pairs: Sequence[Tuple[int, int]]
) -> onnx.ModelProto:
    """Copy of ``model`` with each ``(add, norm)`` pair replaced by a call to a
    new model-local function. ``pairs`` must come from ``_eligible_pairs``."""
    out = copy.deepcopy(model)
    if not pairs:
        return out
    opset = _default_opset(out)
    graph = out.graph
    by_norm = {i: j for j, i in pairs}
    removed = {j for j, _ in pairs}
    new_nodes = []
    functions: List[onnx.FunctionProto] = []
    for i, node in enumerate(graph.node):
        if i in removed:
            continue
        if i not in by_norm:
            new_nodes.append(node)
            continue
        add = graph.node[by_norm[i]]
        name = f"ResidualAddNorm_{len(functions)}"
        functions.append(_build_function(name, add, node, opset))
        call_inputs = [add.input[0], add.input[1], node.input[1]]
        if len(node.input) > 2 and node.input[2]:
            call_inputs.append(node.input[2])
        new_nodes.append(
            helper.make_node(
                name,
                call_inputs,
                [o for o in node.output if o],
                name=f"{name}_call",
                domain=LOCAL_DOMAIN,
            )
        )

    dropped = {graph.node[j].output[0] for j in removed}
    del graph.node[:]
    graph.node.extend(new_nodes)
    kept = [v for v in graph.value_info if v.name not in dropped]
    del graph.value_info[:]
    graph.value_info.extend(kept)
    out.functions.extend(functions)
    if not any(o.domain == LOCAL_DOMAIN for o in out.opset_import):
        out.opset_import.append(helper.make_opsetid(LOCAL_DOMAIN, 1))
    return out


def _total_traffic(model: onnx.ModelProto) -> int:
    return sum(kernel_traffic_breakdown(model).values())


def outline_residual_add_norm(
    model: onnx.ModelProto,
    min_bytes_saved: int = 0,
    require_fused_target: bool = True,
) -> OutlineResult:
    """Outline each eligible ``Add -> LayerNormalization`` pair whose measured
    saving under the per-kernel traffic model is at least ``min_bytes_saved``.

    Each pair is measured on its own: the model's total traffic with and
    without that pair outlined (``decompose_functions=False``). Pairs are
    disjoint, so the accepted pairs are then outlined together.

    With ``require_fused_target`` (the default), a pair is only considered if
    a fused kernel can take it (see ``_has_fused_target``). Pass ``False`` to
    outline every structural match and rely on the measured saving alone.
    """
    base = _total_traffic(model)
    accepted: List[Tuple[int, int]] = []
    saved: List[int] = []
    shapes, dtypes = {}, {}
    opset = _default_opset(model)
    if require_fused_target:
        inferred = onnx.shape_inference.infer_shapes(model, data_prop=True)
        shapes = _collect_shapes(inferred.graph, {})
        dtypes = _collect_dtypes(inferred.graph, {})
    for j, i in _eligible_pairs(model):
        if require_fused_target and not _has_fused_target(
            model.graph.node[j], model.graph.node[i], shapes, dtypes, opset
        ):
            continue
        pair = (j, i)
        gain = base - _total_traffic(_outline(model, [pair]))
        if gain >= min_bytes_saved:
            accepted.append(pair)
            saved.append(gain)
    return OutlineResult(_outline(model, accepted), tuple(saved))
