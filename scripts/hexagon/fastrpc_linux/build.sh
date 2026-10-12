#!/bin/bash
# Reproduces the Hexagon FastRPC round trip on a Linux aarch64 host with a Hexagon cDSP
# (Radxa Dragon Q8B / SC8280XP, cDSP firmware e_flags V68), end to end:
#
#   ew_rpc.idl --(qaic)--------------------------> ew_rpc_{skel,stub}.c, ew_rpc.h
#   ew_rpc_skel.c + ew_rpc_impl.c --(hexagon-clang -mv68 -shared)--> ew_rpc.so   (DSP side)
#   ew_client.c + ew_rpc_stub.c   --(native gcc, -lcdsprpc)-------> ew_client    (AP side)
#   run ew_client with ADSP_LIBRARY_PATH pointing at ew_rpc.so
#
# The Hexagon SDK's tools (qaic, hexagon-clang) are x86-64 binaries. On an aarch64 host they
# run under qemu-x86_64 against an x86-64 root filesystem. The sysroot is built with
# debootstrap --arch=amd64 and needs libatomic1 (see the note in README.md).
#
# Environment (defaults match the layout set up on this machine):
#   HEXAGON_SDK_ROOT   Hexagon SDK 6.4.0.2 root (contains incs/, ipc/, tools/)
#   HEXAGON_TOOLCHAIN  SDK Hexagon Tools, e.g. .../tools/HEXAGON_Tools/19.0.04/Tools
#   X86_SYSROOT        amd64 root for qemu-x86_64 -L (default ~/opt/x86-sysroot)
#   QEMU_X86           qemu-x86_64 binary (default /usr/bin/qemu-x86_64)
#   HEX_ARCH           Hexagon arch flag (default v68, matching the cDSP firmware)
#   RUN_ON_DEVICE      1 to run ew_client on this board (default 1)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
HEXAGON_SDK_ROOT="${HEXAGON_SDK_ROOT:-$HOME/opt/hexagon-sdk/Hexagon_SDK/6.4.0.2}"
HEXAGON_TOOLCHAIN="${HEXAGON_TOOLCHAIN:-$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/19.0.04/Tools}"
X86_SYSROOT="${X86_SYSROOT:-$HOME/opt/x86-sysroot}"
QEMU_X86="${QEMU_X86:-/usr/bin/qemu-x86_64}"
HEX_ARCH="${HEX_ARCH:-v68}"
RUN_ON_DEVICE="${RUN_ON_DEVICE:-1}"
OUT="${OUT:-$HERE/out}"

QAIC="$HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/Ubuntu/qaic"
for p in "$QAIC" "$HEXAGON_TOOLCHAIN/bin/hexagon-clang" "$QEMU_X86" "$X86_SYSROOT/lib64/ld-linux-x86-64.so.2"; do
  [ -e "$p" ] || { echo "missing: $p" >&2; exit 2; }
done

# qemu-user passes QEMU_LD_PREFIX to child processes; the Hexagon driver spawns clang-19.
export QEMU_LD_PREFIX="$X86_SYSROOT"
export LD_LIBRARY_PATH="$HEXAGON_TOOLCHAIN/lib"
x86() { "$QEMU_X86" -L "$X86_SYSROOT" "$@"; }

mkdir -p "$OUT"
cd "$OUT"
cp "$HERE/ew_rpc.idl" .

echo "== qaic"
x86 "$QAIC" -mdll -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" -o . ew_rpc.idl

echo "== DSP skel ($HEX_ARCH)"
QURT="$HEXAGON_SDK_ROOT/rtos/qurt/compute${HEX_ARCH}/include"
x86 "$HEXAGON_TOOLCHAIN/bin/hexagon-clang" -m${HEX_ARCH} -O2 -fPIC -shared -Wall \
  -I . -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" \
  -I "$HEXAGON_SDK_ROOT/ipc/fastrpc/rpcmem/inc" -I "$QURT/qurt" -I "$QURT/posix" \
  "$HERE/ew_rpc_impl.c" ew_rpc_skel.c -o ew_rpc.so
x86 "$HEXAGON_TOOLCHAIN/bin/hexagon-readelf" -h ew_rpc.so | grep -E 'Machine|Flags' || true

echo "== AP client (native aarch64)"
gcc -O2 -Wall -I . -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" \
  "$HERE/ew_client.c" ew_rpc_stub.c -L/usr/lib/aarch64-linux-gnu -lcdsprpc -o ew_client

echo "== host library for the Python driver"
gcc -O2 -Wall -fPIC -shared -I . -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" \
  "$HERE/ew_host_session.c" ew_rpc_stub.c -L/usr/lib/aarch64-linux-gnu -lcdsprpc -o libew_host.so

if [ "$RUN_ON_DEVICE" = 1 ]; then
  echo "== run on cDSP"
  ADSP_LIBRARY_PATH="$OUT" ./ew_client
fi
