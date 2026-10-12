"""Lower an ONNX graph through tinygrad's DSP device and emit one C file that runs all kernels.

tinygrad schedules the graph and renders one C kernel per compute op (DEV=DSP). This script:
  1. builds tinygrad's OnnxRunner for the model and schedules it,
  2. lowers the schedule to rendered kernel source (no compile, no execution),
  3. assigns every kernel argument a byte offset in one aligned arena,
  4. writes tg_graph_gen.c (the kernels plus a run function) and layout.json (offsets,
     input/output, and the constant bytes taken from the ONNX initializers).

Usage: python capture_graph.py MODEL.onnx --input-name Input3 --input-shape 1,1,28,28 --out DIR
Environment: TG_TMPDIR (a non-noexec dir); CC is set to noop_cc automatically.
"""
import argparse, base64, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ANSI = re.compile(r"\x1b\[[0-9;]*m")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--input-name", required=True)
    ap.add_argument("--input-shape", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    shape = tuple(int(v) for v in args.input_shape.split(","))

    os.environ.setdefault("DEV", "DSP")
    os.environ.setdefault("MOCKDSP", "1")
    os.environ.setdefault("CC", os.path.join(HERE, "..", "noop_cc"))
    if os.environ.get("TG_TMPDIR"): os.environ["TMPDIR"] = os.environ["TG_TMPDIR"]

    import numpy as np
    import onnx
    from onnx import numpy_helper
    from tinygrad import Tensor
    from tinygrad.nn.onnx import OnnxRunner
    from tinygrad.uop.ops import Ops
    import tinygrad.engine.realize as R

    runner = OnnxRunner(args.model)
    x = Tensor(np.zeros(shape, dtype=np.float32))
    out_t = list(runner({args.input_name: x}).values())[0]
    lin = out_t.schedule_linear()
    lin = R.lower_and_compile(lin)

    def buf_of(u):
        """Follow views (SHRINK/BITCAST/RESHAPE) down to the BUFFER they index."""
        while u.op is not Ops.BUFFER:
            if not u.src: raise RuntimeError(f"no BUFFER under {u.op}")
            u = u.src[0]
        return u

    def buf_key(b): return b.arg.slot
    def nelem(b): return int(b.src[0].arg)
    def nbytes(b): return nelem(b) * b.dtype.itemsize

    def view_offset(a):
        """Byte offset of a kernel argument inside its base buffer (sum of SHRINK begins)."""
        off, u = 0, a
        while u.op is not Ops.BUFFER:
            if u.op is Ops.SHRINK:
                off += int(u.src[1].arg) * u.src[0].dtype.itemsize
            u = u.src[0]
        return off

    # Initializer names -> the COPY that places them on the DSP.
    inits = {i.name: numpy_helper.to_array(i) for i in onnx.load(args.model).graph.initializer}
    buffers = {}   # key -> {"nbytes": ..., "dtype": "float32"}
    consts = {}    # key -> base64 bytes, from the ONNX initializers
    kernels = []   # {"name", "src", "args": [keys]}
    input_key = None
    last_out = None

    for item in lin.src:
        head = item.src[0]
        dest = buf_of(item.src[1]) if len(item.src) > 1 else None
        if head.op is Ops.COPY:
            if dest is None: continue
            key = buf_key(dest)
            buffers.setdefault(key, {"nbytes": nbytes(dest), "dtype": str(dest.dtype)})
            # The input is the copy fed by the host-side (NPY) parameter, not a file constant.
            if any(v.op is Ops.PARAM and v.arg.device == "NPY" for v in head.toposort()):
                input_key = key
            continue
        if head.op is not Ops.PROGRAM:
            continue
        srcs = [u for u in head.toposort() if u.op is Ops.SOURCE]
        text = ANSI.sub("", srcs[0].arg)
        name = ANSI.sub("", head.arg.name)
        refs = []
        for a in item.src[1:]:
            b = buf_of(a)
            k = buf_key(b)
            buffers.setdefault(k, {"nbytes": nbytes(b), "dtype": str(b.dtype)})
            refs.append({"key": k, "offset": view_offset(a)})
        kernels.append({"name": name, "src": text, "args": refs})
        last_out = refs[0]["key"]

    # Constants: each initializer's Tensor reaches a DSP BUFFER with the same slot the kernels use.
    for name, t in runner.graph_values.items():
        if name not in inits or not hasattr(t, "uop"):
            continue
        dsp = [v for v in t.uop.toposort() if v.op is Ops.BUFFER and v.arg.device == "DSP"]
        if not dsp: continue
        k = buf_key(dsp[0])
        arr = np.ascontiguousarray(inits[name], dtype=np.float32)
        if k in buffers and buffers[k]["nbytes"] == arr.nbytes:  # float32 constants
            consts[str(k)] = base64.b64encode(arr.tobytes()).decode()

    layout = {"input": {"key": input_key, "shape": list(shape)},
              "output": {"key": last_out, "shape": list(out_t.shape)},
              "buffers": {str(k): v for k, v in buffers.items()},
              "consts": consts,
              "kernels": kernels}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "layout.json"), "w") as f:
        json.dump(layout, f, indent=1)
    print(f"kernels={len(kernels)} buffers={len(buffers)} consts={len(consts)} input={input_key} output={last_out}")

if __name__ == "__main__":
    main()
