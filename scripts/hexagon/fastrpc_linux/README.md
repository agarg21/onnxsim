# Hexagon FastRPC on Linux: reproduction and elementwise extension

`build.sh` rebuilds a small FastRPC service for the cDSP on a Linux aarch64 host
(Radxa Dragon Q8B / SC8280XP) and runs its host test on the device. The
interface is `ew_rpc.idl`:

| method | purpose |
| --- | --- |
| `run_add(a, b)` | smoke test: scalar add on the DSP |
| `elementwise(op, a, b, c)` | float32 elementwise over byte payloads. `op` 0..7 = relu, add, sub, mul, max, min, abs, neg; unary ops ignore `b` |

`ew_client.c` checks every op against the same expression on the host, bit for
bit, for sizes 1, 7, 1024 and 65537, and checks that bad op codes and length
mismatches are rejected.

## Real model: MNIST-8 on the cDSP

`onnx_dsp_run.py` runs an ONNX graph node by node. Conv, MaxPool, MatMul, Add, and Relu
go to the DSP. Reshape and broadcasting stay on the host. Every DSP node output is compared
with onnxruntime on the same graph, and the final output is compared with onnxruntime on
the original model.

```sh
./build.sh                                            # builds out/ and runs ew_client
mkdir -p models && curl -fsSL -o models/mnist-8.onnx \
  https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-8.onnx
ADSP_LIBRARY_PATH=$PWD/out python onnx_dsp_run.py models/mnist-8.onnx
```

The driver needs `numpy`, `onnx`, and `onnxruntime`. Add `--simplified` to run an onnxsim
output on the DSP instead of the original graph.

Passing run (`mnist-8`, random input, seed 0): all 12 DSP nodes match onnxruntime exactly
(max |err| = 0), and the final output matches with the same argmax.

## Run

```sh
./build.sh                 # builds into ./out and runs ew_client on the cDSP
RUN_ON_DEVICE=0 ./build.sh # build only
```

## Prerequisites

- Hexagon SDK 6.4.0.2 extracted (`HEXAGON_SDK_ROOT`). The SDK zip extracted with
  Python's `zipfile` loses symlinks and execute bits; restore them from the zip
  metadata before running the tools.
- Hexagon Tools 19.0.04 inside the SDK (`HEXAGON_TOOLCHAIN`).
- `qemu-user` (provides `qemu-x86_64`) and an amd64 root at `X86_SYSROOT`,
  built with `debootstrap --arch=amd64`. Install `libatomic1` into it, since
  `hexagon-clang` needs `libatomic.so.1`.
- The cDSP skel is unsigned. `ew_client` enables unsigned modules for the cDSP
  domain with `remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE)`. That is
  fine for development, not for production.

## Notes

- `sequence<int32>` parameters reach the skel as byte lengths, so the skel checks
  `pLen == 4 * count`.
- Conv `auto_pad` (SAME_UPPER/SAME_LOWER) is resolved on the host into explicit pads, since
  MNIST uses it.

- The DSP side uses scalar float. The output is compared bit for bit, so any
  difference from host IEEE arithmetic shows up as a failure.
- Output length is fixed by the caller (`rout` sequence capacity). The DSP
  requires `len(c) == len(a)`.
