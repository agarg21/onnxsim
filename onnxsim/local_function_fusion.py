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

The saving is only real if the runtime runs the call as one kernel. Runtimes
that inline model-local functions back to primitive ops (ORT does) keep the
sum in memory again, so ``decompose_functions=True`` costs the same as the
original graph. The gate here therefore uses the advisor's per-candidate
``bytes_saved`` estimate; callers that need the saving must also have a fused
kernel for the function body (see ``fuse_skip_layer_norm``).
"""

import copy
from typing import List

import onnx
from onnx import helper

from onnxsim.model_info import (
    _collect_dtypes,
    _collect_shapes,
    _is_std_domain,
    _tensor_bytes,
)

__all__ = ["LOCAL_DOMAIN", "outline_residual_add_norm"]

LOCAL_DOMAIN = "onnxsim.local"
_NORM = "LayerNormalization"


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
    func = helper.make_function(
        LOCAL_DOMAIN,
        name,
        inputs,
        formal_outputs,
        body,
        [helper.make_opsetid("", opset)],
    )
    return func


def outline_residual_add_norm(
    model: onnx.ModelProto,
    min_bytes_saved: int = 0,
) -> "tuple[onnx.ModelProto, int]":
    """Copy of ``model`` with each eligible ``Add -> LayerNormalization`` pair
    replaced by a call to a new model-local function. Returns the new model and
    the number of pairs outlined.

    A pair is eligible when the Add has two non-empty inputs, its output feeds
    only the norm and is not a graph output, neither node has a subgraph, and
    the predicted saving (``2 * size(sum)``, as in ``fusion_advisor``) is at
    least ``min_bytes_saved``. Shapes come from shape inference; a pair whose
    sum size is unknown is skipped.
    """
    out = copy.deepcopy(model)
    opset = _default_opset(out)
    inferred = onnx.shape_inference.infer_shapes(out, data_prop=True)
    shapes = _collect_shapes(inferred.graph, {})
    dtypes = _collect_dtypes(inferred.graph, {})
    graph = out.graph
    graph_outputs = {o.name for o in graph.output}

    producer = {}
    consumers = {}
    for i, node in enumerate(graph.node):
        for name in node.input:
            if name:
                consumers.setdefault(name, []).append(i)
        for name in node.output:
            if name:
                producer[name] = i

    pairs = []  # (add_index, norm_index)
    used = set()
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
        size = _tensor_bytes(x, shapes, dtypes)
        if size is None or 2 * size < min_bytes_saved:
            continue
        pairs.append((j, i))
        used.update((j, i))

    if not pairs:
        return out, 0

    by_norm = {i: j for j, i in pairs}
    removed = {j for j, _ in pairs}
    new_nodes = []
    functions = []
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
        call = helper.make_node(
            name,
            call_inputs,
            [o for o in node.output if o],
            name=f"{name}_call",
            domain=LOCAL_DOMAIN,
        )
        new_nodes.append(call)

    dropped = {graph.node[j].output[0] for j in removed}
    del graph.node[:]
    graph.node.extend(new_nodes)
    kept = [v for v in graph.value_info if v.name not in dropped]
    del graph.value_info[:]
    graph.value_info.extend(kept)
    out.functions.extend(functions)
    if not any(o.domain == LOCAL_DOMAIN for o in out.opset_import):
        out.opset_import.append(helper.make_opsetid(LOCAL_DOMAIN, 1))
    return out, len(pairs)
