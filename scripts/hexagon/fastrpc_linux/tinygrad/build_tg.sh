#!/bin/bash
# Builds tinygrad's rendered matmul as a FastRPC skel for the cDSP and runs tg_matmul_test.py.
# Uses the same SDK/sysroot/qemu setup as ../build.sh.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
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

mkdir -p "$OUT"; cd "$OUT"
cp "$HERE/tg_rpc.idl" .
echo "== qaic"
x86 "$QAIC" -mdll -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" -o . tg_rpc.idl
echo "== DSP skel (tinygrad kernel, $HEX_ARCH)"
QURT="$HEXAGON_SDK_ROOT/rtos/qurt/compute${HEX_ARCH}/include"
x86 "$HEXAGON_TOOLCHAIN/bin/hexagon-clang" -m${HEX_ARCH} -mhvx=${HEX_ARCH} -mhvx-length=128b -O2 -fPIC -shared -Wall \
  -I . -I "$HERE" -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" \
  -I "$HEXAGON_SDK_ROOT/ipc/fastrpc/rpcmem/inc" -I "$QURT/qurt" -I "$QURT/posix" \
  "$HERE/tg_rpc_impl.c" tg_rpc_skel.c -o tg_rpc.so
echo "== host library"
gcc -O2 -Wall -fPIC -shared -I . -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" \
  "$HERE/tg_host.c" tg_rpc_stub.c -L/usr/lib/aarch64-linux-gnu -lcdsprpc -o libtg_host.so
echo "== run on cDSP"
ADSP_LIBRARY_PATH="$OUT" "${PYTHON:-python3}" "$HERE/tg_matmul_test.py" --out "$OUT"
