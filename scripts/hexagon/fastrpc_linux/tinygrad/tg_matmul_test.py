"""Run tinygrad's rendered 8x16 @ 16x4 float32 matmul on the cDSP and check it against numpy.

Usage: ADSP_LIBRARY_PATH=<dir with tg_rpc.so> python tg_matmul_test.py [--out DIR] [--trials N]
"""
import argparse, ctypes, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--trials", type=int, default=20)
    args = ap.parse_args()

    lib = ctypes.CDLL(os.path.join(args.out, "libtg_host.so"))
    h = ctypes.c_uint64(0)
    rc = lib.tg_host_open(ctypes.byref(h))
    if rc:
        sys.exit(f"tg_host_open failed rc=0x{rc & 0xffffffff:x}")
    lib.tg_rpc_matmul_8x16x4.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int,
                                         ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]

    rng = np.random.default_rng(1)
    worst = 0.0
    failures = 0
    for t in range(args.trials):
        a = rng.standard_normal((8, 16)).astype(np.float32)
        b = rng.standard_normal((16, 4)).astype(np.float32)
        c = np.zeros((8, 4), dtype=np.float32)
        rc = lib.tg_rpc_matmul_8x16x4(h, a.ctypes.data, a.nbytes, b.ctypes.data, b.nbytes,
                                      c.ctypes.data, c.nbytes)
        if rc:
            sys.exit(f"matmul rc=0x{rc & 0xffffffff:x}")
        ref = a @ b
        err = float(np.max(np.abs(c - ref)))
        worst = max(worst, err)
        if not np.allclose(c, ref, atol=1e-5, rtol=1e-5):
            failures += 1
    print(f"{args.trials - failures}/{args.trials} trials within 1e-5; worst max|err| = {worst:.3g}")
    print("PASSED" if failures == 0 else f"FAILED: {failures} trial(s)")
    return 1 if failures else 0

if __name__ == "__main__":
    sys.exit(main())
