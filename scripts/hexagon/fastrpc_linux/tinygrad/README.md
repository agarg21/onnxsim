# tinygrad-generated DSP code on the cDSP

tinygrad renders C for its DSP device. This directory takes one rendered kernel and runs it
through FastRPC on the Q8B cDSP (V68), checked against numpy.

1. `capture_tinygrad.py` runs an 8x16 @ 16x4 float32 matmul through `DEV=DSP MOCKDSP=1` with
   `noop_cc` as the compiler, so nothing is compiled or executed. It reads the rendered kernel
   from `DEBUG=4` output and writes `tg_kernel_r8_4_16.c`. The kernel is tinygrad's own code,
   unchanged.
2. `tg_rpc.idl`, `tg_rpc_impl.c`, `tg_host.c`, and `tg_matmul_test.py` wrap that kernel. The DSP
   side copies inputs into 128-byte-aligned statics, because the kernel's vector loads assume
   alignment that FastRPC marshalled buffers don't guarantee.
3. `build_tg.sh` builds the skel for v68, builds the host library, and runs the test.

Result on the Q8B: 20/20 random trials within 1e-5 of numpy (worst max |err| = 4.8e-7).

```sh
python capture_tinygrad.py                  # optional: re-render the kernel
PYTHON=/path/to/python-with-numpy ./build_tg.sh
```

Note: `capture_tinygrad.py` needs `qemu-hexagon-static` on PATH, since tinygrad's mock program
run fails after the kernel has been printed. The kernel is captured before that point. Set
`TG_TMPDIR` to a directory without noexec if the mock run can't map its file.

## End to end: a whole ONNX graph on the cDSP from tinygrad-generated kernels

`graph/` runs a complete simplified ONNX model (MNIST-8 from onnxsim) on the cDSP with every
compute kernel rendered by tinygrad:

1. `graph/capture_graph.py` schedules the model with tinygrad (`DEV=DSP`), lowers it to rendered
   kernel source, and writes `layout.json`: each kernel's arguments as (buffer, byte offset), the
   input, the output, and the constants from the ONNX initializers.
2. `graph/gen_graph.py` puts every buffer into one aligned arena (with guard bytes on both ends)
   and writes `tg_graph_gen.c`, which calls the kernels in schedule order.
3. `graph/build_graph.sh` builds the v68 skel, the host library, and runs `run_graph_e2e.py`.
   One RPC call carries the arena in and returns it after the kernels run.

```sh
MODEL=models/mnist-8.sim.onnx INPUT_NAME=Input3 INPUT_SHAPE=1,1,28,28 PYTHON=/path/python \
  ./graph/build_graph.sh
```

Result on the Q8B cDSP (5 kernels, 3 random inputs): output identical to onnxruntime
(max |err| = 0) and the same argmax.

`NOOPT=1` is the default. With tinygrad's optimized render, the weight loads are vectorized
into `float4` at a stride that isn't a multiple of 16 bytes, and the cDSP faults on them
(`rc=0x4e`). The unoptimized kernels avoid this.
