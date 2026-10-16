"""Heuristic placement of a graph's nodes onto a primary device with a fallback.

This is the planning half of what an execution provider (EP) partitioner does
in ONNX Runtime: decide which device runs each node, then cut the graph into
same-device regions and count the tensors that cross between them. It is
advisory: it reports a plan and never rewrites the model.

Devices are op-type sets. A device claims a node when its set holds the node's
op type and the device-specific checks in :mod:`onnxsim.webnn_target` and
:mod:`onnxsim.webgpu_target` do not flag it. ``"cpu"`` supports every op and is
always the last resort.

Placement heuristic, in order:

1. Heavy ops (``Conv``, ``ConvTranspose``, ``MatMul``, ``Gemm``, ``Attention``,
   ``MultiHeadAttention``) go to the first device in priority order that
   supports them. This keeps the compute on the primary device whenever it can
   run it.
2. Activation fusion: an elementwise op that reads a placed heavy op's output,
   with that output used only there, follows the heavy op onto its device when
   the device supports it. This matches the epilogue fusion of the kernel
   traffic model, so a fused kernel is never split across devices.
3. Everything else, in graph order: stay on the device of its producers when
   that device supports it, else on the device of its already-placed consumers,
   else the first supporting device. Glue ops follow their neighbours and do not
   add boundaries needlessly.

Partitions are the connected same-device regions. A region is *convex* when no
path leaves it and re-enters it, which an ORT-style scheduler needs; non-convex
regions are counted in :attr:`PartitionPlan.nonconvex_partitions`.

Op support for the WebGPU device comes from :data:`WEBGPU_OPS`. It is an
assumption: onnxruntime-web's operator table is not in this repository, so the
set is derived from the WebNN list plus ops commonly implemented on WebGPU.
Check it against ``js/web/docs/webgpu-operators.md`` before trusting a result.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple, Union

import onnx

from onnxsim.model_info import (
    _collect_dtypes,
    _collect_shapes,
    _is_kernel_elementwise,
    _is_std_domain,
    _kernel_graph,
    _tensor_bytes,
)
from onnxsim.rustnn_runtime import WEBNN_SUPPORTED_OPS
from onnxsim.webgpu_target import (
    _flagged_attention_nodes,
    _flagged_conv3d_nodes,
    _flagged_resize_align_corners_downsample_nodes,
)
from onnxsim.webnn_target import _flagged_reshape_expand_nodes

__all__ = [
    "CPU",
    "WEBGPU_OPS",
    "HEAVY_OPS",
    "PartitionPlan",
    "Partition",
    "partition_graph",
]

CPU = "cpu"
HEAVY_OPS: FrozenSet[str] = frozenset(
    {"Conv", "ConvTranspose", "MatMul", "Gemm", "Attention", "MultiHeadAttention"}
)

#: Assumed WebGPU op set (default domain). See the module docstring.
WEBGPU_OPS: FrozenSet[str] = WEBNN_SUPPORTED_OPS | frozenset(
    {"MatMul", "Attention", "MultiHeadAttention", "Erf", "Sqrt", "Pow", "Neg", "Shape"}
)

_BUILTIN_DEVICES = {"webnn": WEBNN_SUPPORTED_OPS, "webgpu": WEBGPU_OPS}

#: Contrib-domain ops a device runs (attention is ``com.microsoft`` in fused exports).
_CONTRIB_OPS: Dict[str, FrozenSet[Tuple[str, str]]] = {
    "webgpu": frozenset(
        {("com.microsoft", "Attention"), ("com.microsoft", "MultiHeadAttention")}
    ),
}


@dataclasses.dataclass
class Partition:
    """A connected region of nodes on one device.

    :param device: the device the region runs on.
    :param node_indices: indices into the graph's node list, in graph order.
    :param convex: whether the region is convex (see the module docstring).
    """

    device: str
    node_indices: List[int]
    convex: bool


@dataclasses.dataclass
class PartitionPlan:
    """Result of :func:`partition_graph`.

    :param assignment: device for every node index.
    :param partitions: the same-device regions.
    :param boundary_tensors: ``(tensor name, producer device, consumer device)``
            for every node-to-node tensor that crosses devices.
    :param transfer_bytes: bytes moved across those boundaries, each tensor
            counted once per consuming device; ``None`` when a shape is unknown.
    :param heavy_on_primary: how many heavy ops landed on the primary device.
    :param heavy_total: how many heavy ops the graph has.
    :param hazards: node indices a device check flags for the primary device.
            The plan still places them on a fallback. A runtime that commits a
            node at partition time, as ONNX Runtime does for WebGPU ``Attention``
            masks, would ignore that placement.
    """

    assignment: Dict[int, str]
    partitions: List[Partition]
    boundary_tensors: List[Tuple[str, str, str]]
    transfer_bytes: Optional[int]
    heavy_on_primary: int
    heavy_total: int
    hazards: List[int]

    @property
    def nonconvex_partitions(self) -> int:
        return sum(1 for p in self.partitions if not p.convex)

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = defaultdict(int)
        for device in self.assignment.values():
            out[device] += 1
        return dict(out)


def _device_supports(device: str, node: onnx.NodeProto) -> bool:
    if device == CPU:
        return True
    ops = _BUILTIN_DEVICES.get(device)
    if ops is None:
        raise ValueError(
            f"unknown device {device!r}; known: {sorted(_BUILTIN_DEVICES)} and 'cpu'"
        )
    if node.domain in ("", "ai.onnx"):
        return node.op_type in ops
    return (node.domain, node.op_type) in _CONTRIB_OPS.get(device, frozenset())


def _device_hazards(graph: onnx.GraphProto, device: str) -> Set[int]:
    """Node indices a device's documented gaps flag, from the existing checkers."""
    if device == "webnn":
        return {i for i, _ in _flagged_reshape_expand_nodes(graph)}
    if device == "webgpu":
        return (
            {i for i, _ in _flagged_attention_nodes(graph)}
            | {i for i, _ in _flagged_conv3d_nodes(graph)}
            | {i for i, _ in _flagged_resize_align_corners_downsample_nodes(graph)}
        )
    return set()


def partition_graph(
    model: Union[str, onnx.ModelProto],
    primary: str,
    fallback: Sequence[str] = (CPU,),
) -> PartitionPlan:
    """Place ``model``'s nodes on ``primary``, falling back in order.

    :param model: the ONNX model, or a file path. Run it on a simplified model.
    :param primary: the preferred device, one of ``"webnn"``, ``"webgpu"``.
    :param fallback: devices tried in order after ``primary``. ``"cpu"`` is
            appended when missing, so every node always has a home.
    :returns: a :class:`PartitionPlan`
    """
    if isinstance(model, str):
        model = onnx.load(model, load_external_data=False)
    devices: List[str] = [primary] + [d for d in fallback if d != primary]
    if CPU not in devices:
        devices.append(CPU)
    graph = _kernel_graph(model, decompose=False).graph
    nodes = list(graph.node)
    n = len(nodes)

    shapes = _collect_shapes(graph, {})
    dtypes = _collect_dtypes(graph, {})
    producer: Dict[str, int] = {}
    consumers: Dict[str, List[int]] = defaultdict(list)
    for i, node in enumerate(nodes):
        for name in node.output:
            if name:
                producer[name] = i
        for name in node.input:
            if name:
                consumers[name].append(i)
    graph_outputs = {o.name for o in graph.output}

    hazards_by_device = {
        d: _device_hazards(model.graph, d) for d in devices if d != CPU
    }
    primary_hazards = hazards_by_device.get(primary, set())

    def supported(device: str, i: int) -> bool:
        return _device_supports(device, nodes[i]) and i not in hazards_by_device.get(
            device, set()
        )

    assignment: Dict[int, str] = {}

    # 1. Heavy ops: the first device in priority order that supports them.
    for i, node in enumerate(nodes):
        if node.op_type in HEAVY_OPS and _is_std_domain_or_contrib(node):
            for device in devices:
                if supported(device, i):
                    assignment[i] = device
                    break

    # 2. Activation fusion: elementwise readers of a placed heavy op's sole output
    #    follow it onto the same device, down the chain.
    for i in sorted(assignment):
        cur = i
        while True:
            outs = [o for o in nodes[cur].output if o]
            if len(outs) != 1 or outs[0] in graph_outputs:
                break
            readers = consumers.get(outs[0], [])
            if len(readers) != 1:
                break
            nxt = readers[0]
            if nxt in assignment or not _is_kernel_elementwise(nodes[nxt]):
                break
            if not supported(assignment[i], nxt):
                break
            assignment[nxt] = assignment[i]
            cur = nxt

    # 3. Everything else in graph order: stay with producers, then consumers.
    for i in range(n):
        if i in assignment:
            continue
        candidates = [d for d in devices if supported(d, i)]
        if not candidates:
            candidates = [CPU]
        placed_neighbours: List[str] = []
        for name in nodes[i].input:
            p = producer.get(name)
            if p is not None and p in assignment:
                placed_neighbours.append(assignment[p])
        for name in nodes[i].output:
            for c in consumers.get(name, []):
                if c in assignment:
                    placed_neighbours.append(assignment[c])
        chosen = next(
            (d for d in placed_neighbours if d in candidates),
            candidates[0],
        )
        assignment[i] = chosen

    partitions = _partitions(nodes, assignment, producer, consumers)
    boundary, transfer = _boundaries(
        nodes, assignment, producer, consumers, graph_outputs, shapes, dtypes
    )
    heavy = [i for i, node in enumerate(nodes) if node.op_type in HEAVY_OPS]
    return PartitionPlan(
        assignment=assignment,
        partitions=partitions,
        boundary_tensors=boundary,
        transfer_bytes=transfer,
        heavy_on_primary=sum(1 for i in heavy if assignment[i] == primary),
        heavy_total=len(heavy),
        hazards=sorted(i for i in primary_hazards if i < n),
    )


def _is_std_domain_or_contrib(node: onnx.NodeProto) -> bool:
    # Attention is a com.microsoft contrib op in fused exports; accept it too.
    return _is_std_domain(node) or (
        node.domain == "com.microsoft" and node.op_type in HEAVY_OPS
    )


def _partitions(
    nodes: List[onnx.NodeProto],
    assignment: Dict[int, str],
    producer: Dict[str, int],
    consumers: Dict[str, List[int]],
) -> List[Partition]:
    n = len(nodes)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, node in enumerate(nodes):
        for name in node.input:
            p = producer.get(name)
            if p is not None and assignment[p] == assignment[i]:
                parent[find(p)] = find(i)

    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    out: List[Partition] = []
    for members in groups.values():
        members.sort()
        member_set = set(members)
        device = assignment[members[0]]
        out.append(
            Partition(
                device=device,
                node_indices=members,
                convex=_is_convex(member_set, nodes, consumers),
            )
        )
    out.sort(key=lambda p: p.node_indices[0])
    return out


def _is_convex(
    members: Set[int],
    nodes: List[onnx.NodeProto],
    consumers: Dict[str, List[int]],
) -> bool:
    """False when a path leaves ``members`` and re-enters it."""
    successors: Dict[int, Set[int]] = defaultdict(set)
    for i, node in enumerate(nodes):
        for name in node.output:
            for c in consumers.get(name, []):
                successors[i].add(c)
    # Nodes reachable from the region without passing through it again.
    outside: Set[int] = set()
    stack = [s for m in members for s in successors[m] if s not in members]
    while stack:
        cur = stack.pop()
        if cur in outside:
            continue
        outside.add(cur)
        for nxt in successors[cur]:
            if nxt in members:
                return False
            stack.append(nxt)
    return True


def _boundaries(
    nodes: List[onnx.NodeProto],
    assignment: Dict[int, str],
    producer: Dict[str, int],
    consumers: Dict[str, List[int]],
    graph_outputs: Set[str],
    shapes,
    dtypes,
) -> Tuple[List[Tuple[str, str, str]], Optional[int]]:
    boundary: List[Tuple[str, str, str]] = []
    total: Optional[int] = 0
    for name, readers in consumers.items():
        p = producer.get(name)
        if p is None:
            continue  # graph inputs and initializers are loaded on each device
        seen_devices = set()
        for c in readers:
            dst = assignment[c]
            if dst == assignment[p] or dst in seen_devices:
                continue
            seen_devices.add(dst)
            boundary.append((name, assignment[p], dst))
            size = _tensor_bytes(name, shapes, dtypes)
            if size is None or total is None:
                total = None
            else:
                total += int(size)
    return boundary, total
