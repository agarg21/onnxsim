#!/bin/bash
# Per-phase cycles of the generated sparse SCA on hexagon-sim: one binary per phase prefix, the simulator's Pcycles
# total for each, and the differences are the phases' costs (see ../README.md, "Profiling the sparse SCA").
#   GEN=<dir with run.c and kernels.c (the sca_sparse_capture.py output)>
#   OUT=<work dir> (default ./sim_build)
#   TOOLS=<HEXAGON_Tools/.../Tools dir>   SHIM=<dir with libncurses.so.5 -> libncursesw.so.6, libtinfo.so.5 -> libtinfo.so.6>
#   NPHASE=<number of phases in the generated run.c> (default 10)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
: "${GEN:?set GEN to the generated sca directory}"
: "${TOOLS:?set TOOLS to the HEXAGON_Tools Tools directory}"
: "${SHIM:?set SHIM to the ncurses shim directory}"
OUT="${OUT:-$PWD/sim_build}"
NPHASE="${NPHASE:-10}"
mkdir -p "$OUT" && cd "$OUT"
cp "$GEN/run.c" "$GEN/kernels.c" .
cp "$HERE/sca_sim_main.c" main.c
: > prefix_summary.txt
for p in $(seq 0 "$NPHASE"); do
  "$TOOLS/bin/hexagon-clang" -O2 -mcpu=hexagonv69 -mhvx -mhvx-length=128b -Wno-unused-variable -Wno-unused-function \
    -DNP="$p" -I . -o "sim_sca_$p" main.c run.c
  LD_LIBRARY_PATH="$SHIM" "$TOOLS/bin/hexagon-sim" --march v69 "sim_sca_$p" > "prefix_$p.log" 2>&1
  echo "prefix $p $(grep -o 'Total: .*' "prefix_$p.log")" >> prefix_summary.txt
done
echo "wrote $OUT/prefix_summary.txt"
