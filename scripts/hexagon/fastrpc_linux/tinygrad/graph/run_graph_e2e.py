"""End-to-end: run the ONNX graph through tinygrad-generated kernels on the cDSP, one RPC.

Packs the arena (constants from layout.json, the input), calls graph_rpc.run_graph, reads the
output back, and compares it with onnxruntime on the original model.

Usage: ADSP_LIBRARY_PATH=<dir with graph_rpc.so> python run_graph_e2e.py MODEL.onnx LAYOUT_DIR OUT_DIR [--seed N]
"""
import argparse, base64, ctypes, json, os, sys
import numpy as np
import onnxruntime as ort

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("layout_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()

    layout = json.load(open(os.path.join(args.layout_dir, "layout.json")))
    base = {int(k): v for k, v in layout["arena"]["base"].items()}
    arena_bytes = layout["arena"]["bytes"]
    lib = ctypes.CDLL(os.path.join(args.out_dir, "libgraph_host.so"))
    h = ctypes.c_uint64(0)
    rc = lib.graph_host_open(ctypes.byref(h))
    if rc:
        sys.exit(f"graph_host_open failed rc=0x{rc & 0xffffffff:x}")
    lib.graph_rpc_run_graph.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int,
                                        ctypes.c_void_p, ctypes.c_int]

    inp, outp = layout["input"], layout["output"]
    ref = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    in_name = ref.get_inputs()[0].name
    rng = np.random.default_rng(args.seed)
    failures = 0
    for t in range(args.trials):
        x = rng.random(inp["shape"], dtype=np.float32)
        arena = np.zeros(arena_bytes, dtype=np.uint8)
        for k, b64 in layout["consts"].items():
            data = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
            arena[base[int(k)]:base[int(k)] + data.size] = data
        arena[base[inp["key"]]:base[inp["key"]] + x.nbytes] = np.frombuffer(x.tobytes(), dtype=np.uint8)

        out_arena = np.zeros(arena_bytes, dtype=np.uint8)
        rc = lib.graph_rpc_run_graph(h, arena.ctypes.data, arena.nbytes, out_arena.ctypes.data,
                                     out_arena.nbytes)
        if rc:
            sys.exit(f"run_graph rc=0x{rc & 0xffffffff:x}")
        n = int(np.prod(outp["shape"]))
        got = out_arena[base[outp["key"]]:base[outp["key"]] + n * 4].view(np.float32).reshape(outp["shape"])
        want = ref.run(None, {in_name: x})[0]
        err = float(np.max(np.abs(got - want)))
        ok = np.allclose(got, want, atol=1e-4, rtol=1e-4) and int(np.argmax(got)) == int(np.argmax(want))
        failures += not ok
        print(f"[{'OK' if ok else 'FAIL'}] trial {t}: max|err| vs onnxruntime = {err:.3g}, "
              f"argmax {int(np.argmax(got))} vs {int(np.argmax(want))}")
    print("PASSED" if failures == 0 else f"FAILED: {failures} trial(s)")
    return 1 if failures else 0

if __name__ == "__main__":
    sys.exit(main())
