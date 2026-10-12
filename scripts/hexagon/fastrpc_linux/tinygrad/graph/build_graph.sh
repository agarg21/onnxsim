#!/bin/bash
# End to end: tinygrad lowers the ONNX graph to rendered DSP kernels, gen_graph.py emits one C
# file, the skel is built for v68, and run_graph_e2e.py checks the result on the cDSP.
#   MODEL=path/to/model.onnx INPUT_NAME=Input3 INPUT_SHAPE=1,1,28,28 PYTHON=/path/python ./build_graph.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3}"
MODEL="${MODEL:?set MODEL to the ONNX model (e.g. onnxsim output)}"
INPUT_NAME="${INPUT_NAME:?set INPUT_NAME}"
INPUT_SHAPE="${INPUT_SHAPE:?set INPUT_SHAPE}"
HEXAGON_SDK_ROOT="${HEXAGON_SDK_ROOT:-$HOME/opt/hexagon-sdk/Hexagon_SDK/6.4.0.2}"
HEXAGON_TOOLCHAIN="${HEXAGON_TOOLCHAIN:-$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/19.0.04/Tools}"
X86_SYSROOT="${X86_SYSROOT:-$HOME/opt/x86-sysroot}"
QEMU_X86="${QEMU_X86:-/usr/bin/qemu-x86_64}"
HEX_ARCH="${HEX_ARCH:-v68}"
OUT="${OUT:-$HERE/out}"
QAIC="$HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/Ubuntu/qaic"
export QEMU_LD_PREFIX="$X86_SYSROOT"
export LD_LIBRARY_PATH="$HEXAGON_TOOLCHAIN/lib"
x86() { "$QEMU_X86" -L "$X86_SYSROOT" "$@"; }

# tinygrad's optimized render vectorizes weight loads into float4 at a stride that is not a
# multiple of 16 bytes, which faults on the cDSP. NOOPT=1 renders the unvectorized kernels.
export NOOPT="${NOOPT:-1}"

mkdir -p "$OUT"
echo "== capture (tinygrad DEV=DSP, no compile)"
TG_TMPDIR="${TG_TMPDIR:-$HOME/opt/tgtmp}"
mkdir -p "$TG_TMPDIR"
TG_TMPDIR="$TG_TMPDIR" "$PY" "$HERE/capture_graph.py" "$MODEL" --input-name "$INPUT_NAME" \
  --input-shape "$INPUT_SHAPE" --out "$OUT"
echo "== generate DSP C"
"$PY" "$HERE/gen_graph.py" "$OUT"
cp "$HERE/graph_rpc.idl" "$OUT/"
cd "$OUT"
echo "== qaic"
x86 "$QAIC" -mdll -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" -o . graph_rpc.idl
echo "== DSP skel ($HEX_ARCH)"
QURT="$HEXAGON_SDK_ROOT/rtos/qurt/compute${HEX_ARCH}/include"
x86 "$HEXAGON_TOOLCHAIN/bin/hexagon-clang" -m${HEX_ARCH} -mhvx=${HEX_ARCH} -mhvx-length=128b -O2 -fPIC -shared -Wall \
  -I . -I "$HERE" -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" \
  -I "$HEXAGON_SDK_ROOT/ipc/fastrpc/rpcmem/inc" -I "$QURT/qurt" -I "$QURT/posix" \
  "$HERE/graph_rpc_impl.c" tg_graph_gen.c graph_rpc_skel.c -o graph_rpc.so
echo "== host library"
gcc -O2 -Wall -fPIC -shared -I . -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" \
  "$HERE/graph_host.c" graph_rpc_stub.c -L/usr/lib/aarch64-linux-gnu -lcdsprpc -o libgraph_host.so
echo "== run on cDSP"
ADSP_LIBRARY_PATH="$OUT" "$PY" "$HERE/run_graph_e2e.py" "$MODEL" "$OUT" "$OUT"
