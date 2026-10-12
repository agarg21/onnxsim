"""Run an ONNX model's supported ops on the Hexagon cDSP through ew_rpc, node by node.

Host code handles Reshape and broadcasting. Conv, MaxPool, MatMul, Add, and Relu run on
the DSP. Each DSP output is compared with onnxruntime on the same graph, and the final
output is also compared with onnxruntime on the original, unsimplified model.

Usage (after ./build.sh has produced out/ew_rpc.so and out/libew_host.so):
  python onnx_dsp_run.py MODEL.onnx [--simplified MODEL_SIM.onnx] [--seed 0]
"""
import argparse
import ctypes
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT", os.path.join(HERE, "out"))
ATOL = 1e-4
RTOL = 1e-4


class Dsp:
    def __init__(self, lib_path):
        self.lib = ctypes.CDLL(lib_path)
        self.handle = ctypes.c_uint64(0)
        rc = self.lib.ew_host_open(ctypes.byref(self.handle))
        if rc:
            raise RuntimeError(f"ew_host_open failed rc=0x{rc & 0xffffffff:x}")
        self.lib.ew_rpc_elementwise.argtypes = [ctypes.c_uint64, ctypes.c_int32,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
        self.lib.ew_rpc_matmul_f32.argtypes = [ctypes.c_uint64, ctypes.c_int32, ctypes.c_int32,
            ctypes.c_int32, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_int]
        self.lib.ew_rpc_conv2d_f32.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
        self.lib.ew_rpc_maxpool2d_f32.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]

    def _check(self, rc, what):
        if rc:
            raise RuntimeError(f"{what} failed rc=0x{rc & 0xffffffff:x}")

    def elementwise(self, op, a, b=None):
        a = np.ascontiguousarray(a, dtype=np.float32)
        c = np.empty_like(a)
        if b is None:
            b_ptr, b_len = None, 0
        else:
            b = np.ascontiguousarray(b, dtype=np.float32)
            b_ptr, b_len = b.ctypes.data, b.nbytes
        rc = self.lib.ew_rpc_elementwise(self.handle, op, a.ctypes.data, a.nbytes, b_ptr, b_len,
                                         c.ctypes.data, c.nbytes)
        self._check(rc, f"elementwise op={op}")
        return c

    def matmul(self, a, b):
        a = np.ascontiguousarray(a, dtype=np.float32)
        b = np.ascontiguousarray(b, dtype=np.float32)
        m, k = a.shape
        k2, n = b.shape
        assert k == k2
        c = np.empty((m, n), dtype=np.float32)
        rc = self.lib.ew_rpc_matmul_f32(self.handle, m, k, n, a.ctypes.data, a.nbytes,
                                        b.ctypes.data, b.nbytes, c.ctypes.data, c.nbytes)
        self._check(rc, "matmul")
        return c

    def conv2d(self, x, w, bias, strides, pads):
        x = np.ascontiguousarray(x, dtype=np.float32)
        w = np.ascontiguousarray(w, dtype=np.float32)
        N, C, H, W = x.shape
        M, Cw, kH, kW = w.shape
        assert Cw == C
        sH, sW = strides
        pT, pL, pB, pR = pads
        Ho = (H + pT + pB - kH) // sH + 1
        Wo = (W + pL + pR - kW) // sW + 1
        y = np.empty((N, M, Ho, Wo), dtype=np.float32)
        p = np.array([N, C, H, W, M, kH, kW, sH, sW, pT, pL, pB, pR,
                      0 if bias is None else 1], dtype=np.int32)
        if bias is None:
            bias_arr = np.zeros(0, dtype=np.float32)
        else:
            bias_arr = np.ascontiguousarray(bias, dtype=np.float32)
        rc = self.lib.ew_rpc_conv2d_f32(self.handle, p.ctypes.data, p.size * 4, x.ctypes.data,
                                        x.nbytes, w.ctypes.data, w.nbytes, bias_arr.ctypes.data,
                                        bias_arr.nbytes, y.ctypes.data, y.nbytes)
        self._check(rc, "conv2d")
        return y

    def maxpool(self, x, kernel, strides, pads):
        x = np.ascontiguousarray(x, dtype=np.float32)
        N, C, H, W = x.shape
        kH, kW = kernel
        sH, sW = strides
        pT, pL, pB, pR = pads
        Ho = (H + pT + pB - kH) // sH + 1
        Wo = (W + pL + pR - kW) // sW + 1
        y = np.empty((N, C, Ho, Wo), dtype=np.float32)
        p = np.array([N, C, H, W, kH, kW, sH, sW, pT, pL, pB, pR], dtype=np.int32)
        rc = self.lib.ew_rpc_maxpool2d_f32(self.handle, p.ctypes.data, p.size * 4, x.ctypes.data,
                                           x.nbytes, y.ctypes.data, y.nbytes)
        self._check(rc, "maxpool2d")
        return y


def attrs(node):
    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}


def conv_pads(a, in_hw, kernel, strides):
    """Explicit [top, left, bottom, right] pads, resolving ONNX auto_pad SAME_UPPER/LOWER."""
    auto = a.get("auto_pad", b"NOTSET")
    auto = auto.decode() if isinstance(auto, bytes) else auto
    if auto in ("SAME_UPPER", "SAME_LOWER"):
        pads_hw = []
        for size, k, s in zip(in_hw, kernel, strides):
            out = -(-size // s)
            total = max((out - 1) * s + k - size, 0)
            begin = total // 2 if auto == "SAME_UPPER" else total - total // 2
            pads_hw.append((begin, total - begin))
        (pT, pB), (pL, pR) = pads_hw
        return [pT, pL, pB, pR]
    if auto not in ("NOTSET", "VALID"):
        raise NotImplementedError(f"auto_pad {auto}")
    return list(a.get("pads", [0, 0, 0, 0]))


def run_graph(dsp, model, feeds):
    env = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    env.update(feeds)
    log = []
    for node in model.graph.node:
        ins = [env[i] if i else None for i in node.input]
        a = attrs(node)
        if node.op_type == "Conv":
            assert a.get("group", 1) == 1 and all(d == 1 for d in a.get("dilations", [1, 1]))
            strides = a.get("strides", [1, 1])
            pads = conv_pads(a, ins[0].shape[2:], ins[1].shape[2:], strides)
            out = dsp.conv2d(ins[0], ins[1], ins[2] if len(ins) > 2 else None, strides, pads)
        elif node.op_type == "MaxPool":
            assert all(d == 1 for d in a.get("dilations", [1, 1])) and not a.get("ceil_mode", 0)
            kernel = a["kernel_shape"]
            strides = a.get("strides", [1, 1])
            pads = conv_pads(a, ins[0].shape[2:], kernel, strides)
            out = dsp.maxpool(ins[0], kernel, strides, pads)
        elif node.op_type == "MatMul":
            assert ins[0].ndim == 2 and ins[1].ndim == 2, "only 2-D MatMul is on the DSP"
            out = dsp.matmul(ins[0], ins[1])
        elif node.op_type == "Add":
            shape = np.broadcast_shapes(ins[0].shape, ins[1].shape)
            x = np.ascontiguousarray(np.broadcast_to(ins[0], shape), dtype=np.float32)
            y = np.ascontiguousarray(np.broadcast_to(ins[1], shape), dtype=np.float32)
            out = dsp.elementwise(1, x, y).reshape(shape)
        elif node.op_type == "Relu":
            out = dsp.elementwise(0, ins[0]).reshape(ins[0].shape)
        elif node.op_type == "Gemm":
            # Y = alpha * op(A) @ op(B) + beta * C, on the DSP as matmul then broadcast add.
            # Scaling by alpha/beta is applied on the host (exact for 1.0).
            A, B = ins[0], ins[1]
            if a.get("transA", 0):
                A = A.T
            if a.get("transB", 0):
                B = B.T
            alpha, beta = float(a.get("alpha", 1.0)), float(a.get("beta", 1.0))
            prod = dsp.matmul(A, B)
            if alpha != 1.0:
                prod = prod * np.float32(alpha)
            if len(ins) > 2 and ins[2] is not None:
                C = np.asarray(ins[2], dtype=np.float32)
                if beta != 1.0:
                    C = C * np.float32(beta)
                shape = np.broadcast_shapes(prod.shape, C.shape)
                x = np.ascontiguousarray(np.broadcast_to(prod, shape), dtype=np.float32)
                y = np.ascontiguousarray(np.broadcast_to(C, shape), dtype=np.float32)
                out = dsp.elementwise(1, x, y).reshape(shape)
            else:
                out = prod
        elif node.op_type == "Reshape":
            out = ins[0].reshape([int(v) for v in ins[1]])
        else:
            raise NotImplementedError(f"op {node.op_type} is not on the DSP yet")
        env[node.output[0]] = out
        log.append((node.op_type, node.output[0], out))
    return env, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--simplified", help="onnxsim output to run on the DSP (default: the model)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    original = onnx.load(args.model)
    graph_model = onnx.load(args.simplified) if args.simplified else original
    inputs = [i for i in graph_model.graph.input
              if i.name not in {init.name for init in graph_model.graph.initializer}]
    rng = np.random.default_rng(args.seed)
    feeds = {}
    for i in inputs:
        shape = [d.dim_value for d in i.type.tensor_type.shape.dim]
        feeds[i.name] = rng.random(shape, dtype=np.float32)

    # Reference: onnxruntime on the original model.
    ref = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    ref_out = ref.run(None, feeds)[0]

    # Intermediate reference: onnxruntime on the graph that runs on the DSP, with every
    # node output exposed.
    probe = onnx.ModelProto()
    probe.CopyFrom(graph_model)
    known = {o.name for o in probe.graph.output}
    for node in probe.graph.node:
        for name in node.output:
            if name not in known:
                probe.graph.output.append(onnx.helper.make_empty_tensor_value_info(name))
                known.add(name)
    probe_sess = ort.InferenceSession(probe.SerializeToString(), providers=["CPUExecutionProvider"])
    probe_names = [o.name for o in probe_sess.get_outputs()]
    probe_vals = dict(zip(probe_names, probe_sess.run(None, feeds)))

    dsp = Dsp(os.path.join(OUT, "libew_host.so"))
    env, log = run_graph(dsp, graph_model, feeds)

    failures = 0
    for op, name, val in log:
        want = probe_vals[name]
        ok = val.shape == want.shape and np.allclose(val, want, atol=ATOL, rtol=RTOL)
        err = float(np.max(np.abs(val - want))) if val.shape == want.shape else float("nan")
        failures += not ok
        print(f"[{'OK' if ok else 'FAIL'}] {op:8s} {name:32s} shape={tuple(val.shape)} max|err|={err:.3g}")

    final = env[graph_model.graph.output[0].name]
    ok = final.shape == ref_out.shape and np.allclose(final, ref_out, atol=ATOL, rtol=RTOL)
    failures += not ok
    same_class = int(np.argmax(final)) == int(np.argmax(ref_out))
    print(f"[{'OK' if ok else 'FAIL'}] final output vs onnxruntime(original): "
          f"max|err|={float(np.max(np.abs(final - ref_out))):.3g} argmax_match={same_class}")
    print("PASSED" if failures == 0 else f"FAILED: {failures} mismatch(es)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
