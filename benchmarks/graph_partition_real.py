#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Partition real simplified models across device pairs and print the plan.

For each model and each (primary, fallback) pair this prints how many nodes
each device gets, how many heavy ops (Conv, MatMul, Gemm, Attention) stay on
the primary device, the number of same-device partitions, the cross-device
tensors and their bytes, and the hazards the primary device would commit.

usage: graph_partition_real.py MODEL.onnx [MODEL.onnx ...]
"""

import sys

import onnx

from onnxsim.graph_partition import partition_graph

PAIRS = [("webnn", ["webgpu", "cpu"]), ("webgpu", ["cpu"])]


def main(paths):
    print(
        f"{'model':16s} {'primary/fallback':22s} {'nodes per device':34s} "
        f"{'heavy on primary':>17s} {'parts':>6s} {'nonconvex':>10s} "
        f"{'cross tensors':>14s} {'transfer bytes':>15s} {'hazards':>8s}"
    )
    for path in paths:
        name = path.rsplit("/", 1)[-1].removesuffix(".onnx")
        model = onnx.load(path, load_external_data=False)
        for primary, fallback in PAIRS:
            plan = partition_graph(model, primary, fallback)
            counts = plan.counts()
            per_device = ", ".join(f"{d} {c}" for d, c in sorted(counts.items()))
            transfer = (
                "unknown" if plan.transfer_bytes is None else f"{plan.transfer_bytes:,}"
            )
            pair = f"{primary}/{fallback[0]}"
            print(
                f"{name:16s} {pair:22s} {per_device:34s} "
                f"{plan.heavy_on_primary:>8d}/{plan.heavy_total:<8d} "
                f"{len(plan.partitions):>6d} {plan.nonconvex_partitions:>10d} "
                f"{len(plan.boundary_tensors):>14d} {transfer:>15s} "
                f"{len(plan.hazards):>8d}"
            )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
