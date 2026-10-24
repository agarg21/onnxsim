#!/bin/bash
# onnx-remote-compiler command for v68+ cDSPs (qfloat HVX; HMX with HMX=1): an ONNX model -> a tinygrad-generated standalone DSP
# program, packed as a `tghx-v69` artifact for onnx-remote-hexagon-worker (the runner for the v65 artifacts too).
#
#   onnx-remote-compiler --port 39503 --cache-dir ~/.cache/onnxsim-v69 --target hexagon-v69 \
#     --compiler-id "tinygrad-$(git -C "$TINYGRAD_ROOT" rev-parse --short HEAD)" --command '.../onnx_remote/compile_v69.sh {input} {output} {manifest}'
#
# The fork's examples/dsp_graph_onnx.py captures the model's kernels without running them (qemu can't run v68+ code), emits them
# with dsp_graph_v65's emitter and builds the FastRPC skel for HEXAGON_ARCH (default v69). There is no compile-time correctness
# check: compare the runner's outputs with onnxruntime's (check_outputs.py). Needs TINYGRAD_ROOT, HEXAGON_SDK_ROOT,
# HEXAGON_TOOLCHAIN; DSP_THREADS (default 4) is baked into the program; HMX=1 QDQ_HMX=1 HMX_VTCM_KB=... for int8 QDQ models on HMX.
set -eo pipefail
input=$1; output=$2; manifest=$3
: "${TINYGRAD_ROOT:?}" "${HEXAGON_SDK_ROOT:?}" "${HEXAGON_TOOLCHAIN:?}"
work=$(mktemp -d "${TMPDIR:-/tmp}/onnxsim-v69-XXXXXX")
trap 'rm -rf "$work"' EXIT
export PYTHONUNBUFFERED=1 PYTHONPATH="$TINYGRAD_ROOT${PYTHONPATH:+:$PYTHONPATH}" CC="${CC:-clang-19}"
log="$work/compile.log"
"${PYTHON:-python3}" "$TINYGRAD_ROOT/examples/dsp_graph_onnx.py" "$input" "$work/g" --arch "${HEXAGON_ARCH:-v69}" \
  --artifact "$output" --manifest "$manifest" > "$log" 2>&1 || { grep -v "unable to find CPU" "$log" | tail -30 >&2; exit 1; }
grep -E "^(emitted|artifact)" "$log" >&2
