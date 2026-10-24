#!/bin/bash
# The hand MSDA skel (../../../../msda_hvx: msda_impl.c, msda_kernel.h) with tinygrad's generated launch sequences
# for the two real BEVFormer calls (TSA, SCA) next to it, plus the same bench client. Flags bit 16 (0x10000) runs
# tinygrad's sequence; the other flags run the hand kernel, so one phone session can time both.
#   GEN=<dir with gen_l0_tsa/ and gen_l0_sca/ from capture_msda.py>   (required)
#   CASES=<dir with l0_tsa/ and l0_sca/ msda_io case directories>     (default ~/.cache/onnxsim-bevformer/work/msda_io)
#   RUN=1 to push and run on the phone under the phone lock: FLAGS (default "4,65536"), REPS (10), TURBO (0)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MSDA="$(cd "$HERE/../../../../msda_hvx" && pwd)"
: "${GEN:?set GEN to the capture_msda.py output root}"
HEXAGON_SDK_ROOT="${HEXAGON_SDK_ROOT:-/mnt/data/cache/tvm-hexagon/qualcomm/Hexagon_SDK/6.4.0.2}"
HEXAGON_TOOLCHAIN="${HEXAGON_TOOLCHAIN:-$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/19.0.04/Tools}"
NDK_CLANG="${NDK_CLANG:-/usr/lib/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang}"
OUT="${OUT:-$HERE/build}"
DEVICE_SERIAL="${DEVICE_SERIAL:-239dbd8f}"
CASES="${CASES:-$HOME/.cache/onnxsim-bevformer/work/msda_io}"
D="${D:-/data/local/tmp/msda_tg}"

mkdir -p "$OUT" && cd "$OUT"
cp "$MSDA"/msda_rpc.idl "$MSDA"/msda_kernel.h "$MSDA"/msda_shape.h "$MSDA"/msda_io.h "$MSDA"/msda_client.c .
cp "$HERE/tg_dispatch.c" .
python3 "$HERE/make_impl.py" "$MSDA/msda_impl.c" msda_impl.c
"$HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/Ubuntu/qaic" -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" msda_rpc.idl
INC=(-I . -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef")
QURT_INC=(-I "$HEXAGON_SDK_ROOT/rtos/qurt/computev69/include/qurt" -I "$HEXAGON_SDK_ROOT/rtos/qurt/computev69/include/posix")
HX=("$HEXAGON_TOOLCHAIN/bin/hexagon-clang" -O2 -fPIC -mcpu=hexagonv69 -mhvx -mhvx-length=128b -Wno-unused-variable)
[ -n "${TG_PROF:-}" ] && HX+=(-DTG_PROF)   # per-launch timers (tg_dispatch.c reports them through FARF)
"${HX[@]}" -c "${INC[@]}" -o skel.o msda_rpc_skel.c
"${HX[@]}" -c -Wall "${INC[@]}" "${QURT_INC[@]}" -o impl.o msda_impl.c
"${HX[@]}" -c "${INC[@]}" -o dispatch.o tg_dispatch.c
objs=()
for t in ${TAGS:-l0_tsa l0_sca}; do
  mkdir -p "$t"
  cp "$GEN/gen_$t/run.c" "$GEN/gen_$t/kernels.c" "$t/"
  "${HX[@]}" -c "${INC[@]}" -o "tg_$t.o" "$t/run.c"
  objs+=("tg_$t.o")
done
"$HEXAGON_TOOLCHAIN/bin/hexagon-link" -Bdynamic -shared -export-dynamic -o msda_rpc.so skel.o impl.o dispatch.o "${objs[@]}" \
  "$HEXAGON_TOOLCHAIN/target/hexagon/lib/v69/G0/pic/libgcc.so"
"$NDK_CLANG" -O2 -c "${INC[@]}" -I "$HEXAGON_SDK_ROOT/ipc/fastrpc/rpcmem/inc" -o msda_stub.o msda_rpc_stub.c
"$NDK_CLANG" -O2 "${INC[@]}" -I "$HEXAGON_SDK_ROOT/ipc/fastrpc/rpcmem/inc" -o msda_client msda_client.c msda_stub.o \
  -L "$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64" -lcdsprpc -lm
echo "built $OUT/msda_rpc.so and $OUT/msda_client"
[ "${RUN:-0}" = 1 ] || exit 0

PR="${PHONE_RUN:-$HOME/.cache/android-phone/phone-run}"
S="$DEVICE_SERIAL"
"$PR" adb -s "$S" shell "rm -rf $D && mkdir -p $D"
"$PR" adb -s "$S" push msda_client msda_rpc.so "$D/" >/dev/null
for t in ${TAGS:-l0_tsa l0_sca}; do "$PR" adb -s "$S" push "$CASES/$t" "$D/$t" >/dev/null; done
"$PR" adb -s "$S" shell "chmod 755 $D/msda_client && cd $D && LD_LIBRARY_PATH=/vendor/lib64 ADSP_LIBRARY_PATH=$D \
  ./msda_client 'file:///msda_rpc.so?msda_rpc_skel_handle_invoke&_modver=1.0&_dom=cdsp' ${REPS:-10} ${TURBO:-0} ${FLAGS:-4,65536} ${TAGS:-l0_tsa l0_sca}"
