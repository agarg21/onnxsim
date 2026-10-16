# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Advisory layer: which fusions are worth doing, judged by memory traffic.

The fusion passes under ``onnxsim/passes`` fire on structural pattern match
alone. This module asks the per-kernel traffic model (``onnxsim.model_info``)
whether a candidate fusion would remove any memory traffic at all. It is
advisory: nothing here changes a pass or the default simplification, and no
pass calls it yet.

Each candidate reports ``bytes_saved``: the bytes of intermediate tensors that
the fused kernel would neither write nor re-read. A tensor that is written once
and read once by its consumer saves ``size * 2``. When the tensor must still be
materialized for another consumer (or is a graph output) only the re-read is
avoided, so the saving is ``size``.
"""

import copy
import dataclasses
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import onnx
from onnx import helper

from onnxsim.model_info import (
    DTypeMap,
    Macs,
    ShapeMap,
    _collect_dtypes,
    _collect_shapes,
    _is_std_domain,
    _is_symbolic,
    _kernel_graph,
    _kernel_groups,
    _known,
    _tensor_bytes,
    kernel_traffic_breakdown,
)

__all__ = ["FusionCandidate", "fusion_candidates", "should_fuse"]

_NORM_OPS = ("LayerNormalization", "RMSNormalization")


@dataclasses.dataclass(frozen=True)
class FusionCandidate:
    """One fusion the kernel model could apply.

    ``kind`` is ``"elementwise_chain"``, ``"residual_add_norm"``,
    ``"masked_softmax"`` or ``"gridsample_fusion"`` (a GridSample's saving over
     its decomposition, see ``_gridsample_fusion_saving``). ``node_names`` and
    ``op_types`` list the member nodes in graph order (names may be empty for unnamed nodes). ``tensors`` are the
    tensors whose traffic the fusion removes. ``bytes_saved`` is an ``int``, a
    symbolic ``sympy`` expression when a dimension is a ``dim_param``, or
    ``None`` when a shape or element size is unknown.
    """

    kind: str
    node_names: Tuple[str, ...]
    op_types: Tuple[str, ...]
    tensors: Tuple[str, ...]
    bytes_saved: Optional[Macs]


def _total(sizes: Iterable[Optional[Macs]]) -> Optional[Macs]:
    total: Optional[Macs] = 0
    for size in sizes:
        if size is None or total is None:
            return None
        total = total + size
    return total


_FLOAT_BYTES = 4  # DTypeMap stores bytes per element


def _gridsample_fusion_saving(
    node: onnx.NodeProto, shapes: ShapeMap, dtypes: DTypeMap, opset: int
) -> Optional[Macs]:
    """Bytes a fused ``GridSample`` saves over its decomposition.

    The node is decomposed on its own, in a one-node model with the same static
    shapes, by ``rewrite_gridsample_to_gather``. The saving is the decomposed
    traffic minus the fused op's, so it is what keeping the fused op avoids
    moving. ``None`` when a shape or dtype is unknown, or when the pass declines
    to decompose this node (it stays a ``GridSample``, nothing is saved).
    """
    import onnxsim  # local: onnxsim imports this module

    names = [n for n in node.input if n]
    if not names or not node.output or not node.output[0]:
        return None
    tensors = names + [node.output[0]]
    # Static integer dims only: a symbolic dim has no single size to model.
    if not all(
        _known(shapes.get(n))
        and all(isinstance(d, int) for d in shapes[n])
        and dtypes.get(n) == _FLOAT_BYTES
        for n in tensors
    ):
        return None
    inputs = [
        helper.make_tensor_value_info(n, onnx.TensorProto.FLOAT, shapes[n])
        for n in names
    ]
    output = helper.make_tensor_value_info(
        node.output[0], onnx.TensorProto.FLOAT, shapes[node.output[0]]
    )
    single = helper.make_node("GridSample", names, [node.output[0]], name="gridsample")
    single.attribute.extend(copy.deepcopy(list(node.attribute)))
    graph = helper.make_graph([single], "gridsample_fusion", inputs, [output])
    model = helper.make_model(
        graph, ir_version=10, opset_imports=[helper.make_opsetid("", opset)]
    )
    decomposed, _ = onnxsim.simplify(
        model, check_n=0, extra_optimizers=["rewrite_gridsample_to_gather"]
    )
    if any(n.op_type == "GridSample" for n in decomposed.graph.node):
        return None
    fused = sum(kernel_traffic_breakdown(model).values())
    split = sum(kernel_traffic_breakdown(decomposed).values())
    return split - fused


def _collect_candidates(
    graph: onnx.GraphProto,
    inherited_shapes: ShapeMap,
    inherited_dtypes: DTypeMap,
    out: List[FusionCandidate],
    opset: int = 13,
) -> None:
    shapes = _collect_shapes(graph, inherited_shapes)
    dtypes = _collect_dtypes(graph, inherited_dtypes)
    consumers: Dict[str, List[int]] = defaultdict(list)
    producer: Dict[str, int] = {}
    for i, node in enumerate(graph.node):
        for name in node.input:
            if name:
                consumers[name].append(i)
        for name in node.output:
            if name:
                producer[name] = i
    graph_outputs = {o.name for o in graph.output}

    def nbytes(name: str) -> Optional[Macs]:
        return _tensor_bytes(name, shapes, dtypes) if name else None

    def sole_consumer(name: str) -> bool:
        return len(consumers.get(name, [])) == 1 and name not in graph_outputs

    # Elementwise chains: every multi-node kernel group of the kernel model.
    # An intermediate is a member output whose consumers are all members and
    # that is not a graph output; the fused kernel keeps it in registers.
    for group in _kernel_groups(graph, consumers):
        if len(group) < 2:
            continue
        members = [graph.node[i] for i in group]
        member_set = set(group)
        produced = [o for n in members for o in n.output if o]
        intermediates = [
            o
            for o in produced
            if consumers.get(o)
            and o not in graph_outputs
            and all(c in member_set for c in consumers[o])
        ]
        total = _total(nbytes(o) for o in intermediates)
        out.append(
            FusionCandidate(
                kind="elementwise_chain",
                node_names=tuple(n.name for n in members),
                op_types=tuple(n.op_type for n in members),
                tensors=tuple(intermediates),
                bytes_saved=None if total is None else 2 * total,
            )
        )

    # Residual Add feeding a normalization. The sum is written by the Add and
    # read by the norm; a fused kernel avoids both when the sum has no other
    # use, and at least the re-read when it must stay in memory.
    for node in graph.node:
        if not (
            _is_std_domain(node)
            and node.op_type in _NORM_OPS
            and node.input
            and node.input[0]
        ):
            continue
        x = node.input[0]
        add_index = producer.get(x)
        if add_index is None:
            continue
        add = graph.node[add_index]
        if not (_is_std_domain(add) and add.op_type == "Add"):
            continue
        size = nbytes(x)
        if size is None:
            saved = None
        else:
            # The norm is multi-pass (see model_info._KERNEL_MULTIPASS_OPS): it
            # reads the sum for its statistics and again to normalize. A sole-
            # consumer sum is also written once by the Add: 3 * size in all. A
            # sum that must stay in memory keeps its write, so only the two
            # reads go: 2 * size.
            saved = 3 * size if sole_consumer(x) else 2 * size
        out.append(
            FusionCandidate(
                kind="residual_add_norm",
                node_names=(add.name, node.name),
                op_types=(add.op_type, node.op_type),
                tensors=(x,),
                bytes_saved=saved,
            )
        )

    # Where feeding Softmax: the masked scores. Same accounting as above.
    for node in graph.node:
        if not (
            _is_std_domain(node)
            and node.op_type == "Softmax"
            and node.input
            and node.input[0]
        ):
            continue
        x = node.input[0]
        where_index = producer.get(x)
        if where_index is None:
            continue
        where = graph.node[where_index]
        if not (_is_std_domain(where) and where.op_type == "Where"):
            continue
        size = nbytes(x)
        if size is None:
            saved = None
        else:
            # Softmax is multi-pass too: a sole-consumer intermediate is written,
            # then read twice; a shared one keeps its write, so only the two reads go.
            saved = 3 * size if sole_consumer(x) else 2 * size
        out.append(
            FusionCandidate(
                kind="masked_softmax",
                node_names=(where.name, node.name),
                op_types=(where.op_type, node.op_type),
                tensors=(x,),
                bytes_saved=saved,
            )
        )

    # A GridSample: what keeping the fused op saves over its decomposition.
    for node in graph.node:
        if not (_is_std_domain(node) and node.op_type == "GridSample"):
            continue
        saved = _gridsample_fusion_saving(node, shapes, dtypes, opset)
        if saved is None:
            continue
        out.append(
            FusionCandidate(
                kind="gridsample_fusion",
                node_names=(node.name,),
                op_types=(node.op_type,),
                tensors=tuple(n for n in node.input if n),
                bytes_saved=saved,
            )
        )

    # Control-flow bodies are traversed like the kernel model does.
    for node in graph.node:
        for attr in node.attribute:
            bodies = ([attr.g] if attr.HasField("g") else []) + list(attr.graphs)
            for body in bodies:
                _collect_candidates(body, shapes, dtypes, out, opset)


def fusion_candidates(
    model: onnx.ModelProto, attention_tiled: bool = True
) -> List[FusionCandidate]:
    """Fusion candidates in ``model``, with the bytes each one would save.

    The graph is shape-inferred exactly as the kernel model infers it, so sizes
    agree with ``kernel_traffic_breakdown``. ``attention_tiled`` is accepted for
    signature parity with that function; the three candidate kinds never touch
    an attention score matrix, so it does not change any result.
    """
    out: List[FusionCandidate] = []
    opset = next(
        (o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), 13
    )
    _collect_candidates(_kernel_graph(model, decompose=False).graph, {}, {}, out, opset)
    return out


def should_fuse(
    candidate: FusionCandidate, min_bytes_saved: int = 4096
) -> Optional[bool]:
    """Whether fusing ``candidate`` saves enough traffic to be worth doing.

    Returns True when ``bytes_saved >= min_bytes_saved``, False below it, and
    None (undecided) when the saving is symbolic or unknown: a formula in a
    ``dim_param`` has no single value to compare, and guessing one would make
    the gate look more precise than it is.

    Why a floor at all: a kernel has a fixed launch and scheduling cost, a few
    microseconds on a GPU, that does not shrink with its data. Moving 4 KiB at
    about 1 TB/s takes about 4 ns, so removing a few KiB of traffic is lost in
    that floor. Bandwidth only sets the kernel time once the tensors are large.
    The 4096-byte default is a conservative starting point, not a measured
    crossover, and should be tuned per target. The gate counts traffic only; a
    fusion also removes a launch, which this gate does not credit.
    """
    value = candidate.bytes_saved
    if value is None or _is_symbolic(value):
        return None
    return int(value) >= min_bytes_saved
