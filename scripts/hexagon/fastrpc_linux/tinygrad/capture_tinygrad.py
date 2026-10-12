"""Capture the DSP C source that tinygrad renders for an 8x16 @ 16x4 float32 matmul.

tinygrad's DSP device renders C for every kernel, then compiles it with the mock DSP
boilerplate. This script runs the matmul through DEV=DSP MOCKDSP=1 with a no-op CC, so the
compile step does nothing, and reads the rendered kernel back from DEBUG=4 output. The mock
boilerplate (syscalls, _start) is dropped; only the typedef and the kernel function are kept.

Usage: python capture_tinygrad.py [--out tg_kernel_r8_4_16.c]
"""
import argparse, os, re, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ANSI = re.compile(r"\x1b\[[0-9;]*m")

MATMUL = r'''
import numpy as np
from tinygrad import Tensor, dtypes
rng = np.random.default_rng(0)
a = Tensor(rng.random((8, 16), dtype=np.float32), dtype=dtypes.float32)
b = Tensor(rng.random((16, 4), dtype=np.float32), dtype=dtypes.float32)
(a @ b).realize()
'''

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "tg_kernel_r8_4_16.c"))
    args = ap.parse_args()
    env = dict(os.environ, DEV="DSP", MOCKDSP="1", DEBUG="4",
               CC=os.path.join(HERE, "noop_cc"), PATH=HERE + os.pathsep + os.environ["PATH"],
               TMPDIR=os.environ.get("TG_TMPDIR", tempfile.gettempdir()))
    p = subprocess.run([sys.executable, "-c", MATMUL], env=env, capture_output=True, text=True)
    log = ANSI.sub("", p.stdout + p.stderr).splitlines()
    try:
        start = next(i for i, l in enumerate(log) if l.startswith("typedef float float16"))
    except StopIteration:
        sys.exit("no rendered kernel found in tinygrad DEBUG output")
    end = next(i for i in range(start, len(log)) if log[i] == "}")
    body = "\n".join(log[start:end + 1]) + "\n"
    with open(args.out, "w") as f:
        f.write("/* Rendered by tinygrad (DEV=DSP) from capture_tinygrad.py. Do not edit. */\n")
        f.write(body)
    print(f"wrote {args.out} ({end - start + 1} lines)")

if __name__ == "__main__":
    main()
