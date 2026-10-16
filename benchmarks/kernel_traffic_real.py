#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real-architecture benchmark of the per-kernel traffic model.

Builds three architectures with random weights (no download): a tiny Llama and
a tiny BERT from transformers' configs, and a small residual CNN. Each is
exported to ONNX with torch, simplified with onnxsim (the way a user would
deploy it), and costed with ``kernel_traffic_breakdown`` before and after
epilogue fusion (an elementwise op absorbed into the MatMul, Gemm or Conv that
produces its input). Prints one row per model.

Every run also compares with tinygrad: its per-kernel memory estimate for the
same model (``estimate_uop`` summed over the compiled kernels) is printed next
to onnxsim's. The two conventions both count each byte once per kernel, but
tinygrad also fuses differently, so the gap is informative, not an error. When
tinygrad cannot be imported or fails on a model, that cell says so instead of
being left out.

usage: kernel_traffic_real.py [WORKDIR]    (needs torch and transformers)
"""

import os
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")

try:
    import numpy as np
    import onnx
    import torch
    import torch.nn as nn
    from transformers import BertConfig, BertModel, LlamaConfig, LlamaForCausalLM

    import onnxsim
    import onnxsim.model_info as model_info
    from onnxsim.model_info import kernel_traffic_breakdown
except ImportError as e:
    print("SKIP kernel_traffic_real: %s" % e)
    sys.exit(0)

try:
    from tinygrad import Tensor
    from tinygrad.engine.realize import compile_linear, estimate_uop
    from tinygrad.nn.onnx import OnnxRunner
    from tinygrad.uop.ops import Ops
except ImportError as e:
    Tensor = None
    _TINYGRAD_ERROR = "tinygrad not installed (%s)" % e
else:
    _TINYGRAD_ERROR = ""


def _export(path, model, args):
    torch.onnx.export(
        model.eval(),
        args,
        path,
        input_names=["x"],
        opset_version=17,
        dynamo=False,
        do_constant_folding=True,
    )


class _LlamaLogits(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, ids):
        return self.m(input_ids=ids, use_cache=False).logits


class _BertHidden(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, ids):
        return self.m(input_ids=ids).last_hidden_state


class _ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)
        self.b2 = nn.BatchNorm2d(c)

    def forward(self, x):
        y = torch.relu(self.b1(self.c1(x)))
        return torch.relu(x + self.b2(self.c2(y)))


class _SmallResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU()
        )
        self.blocks = nn.Sequential(_ResBlock(16), _ResBlock(16))
        self.head = nn.Linear(16, 10)

    def forward(self, x):
        x = self.blocks(self.stem(x))
        return self.head(x.mean(dim=(2, 3)))


def _models(workdir):
    torch.manual_seed(0)
    llama = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=176,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
        )
    )
    bert = BertModel(
        BertConfig(
            vocab_size=256,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=256,
            max_position_embeddings=64,
        ),
        add_pooling_layer=False,
    )
    specs = [
        ("llama_tiny", _LlamaLogits(llama), (torch.randint(0, 256, (1, 16)),)),
        ("bert_tiny", _BertHidden(bert), (torch.randint(0, 256, (1, 16)),)),
        ("resnet_small", _SmallResNet(), (torch.randn(1, 3, 32, 32),)),
    ]
    for name, model, args in specs:
        path = os.path.join(workdir, name + ".onnx")
        _export(path, model, args)
        simplified, _ = onnxsim.simplify(onnx.load(path), check_n=0)
        onnx.save(simplified, path)
        yield name, path, simplified


def _total(model, epilogue):
    real = model_info._is_kernel_epilogue_producer
    model_info._is_kernel_epilogue_producer = real if epilogue else (lambda node: False)
    try:
        breakdown = kernel_traffic_breakdown(model)
    finally:
        model_info._is_kernel_epilogue_producer = real
    return sum(breakdown.values()), len(breakdown)


def _feeds(model):
    # Random inputs of the model's static shapes; the values only pick a path.
    rng = np.random.default_rng(0)
    init = {i.name for i in model.graph.initializer}
    feeds = {}
    for vi in model.graph.input:
        if vi.name in init:
            continue
        shape = [d.dim_value or 1 for d in vi.type.tensor_type.shape.dim]
        if vi.type.tensor_type.elem_type in (6, 7):
            arr = rng.integers(0, 200, shape).astype(np.int64)
        else:
            arr = rng.uniform(0.5, 2.0, shape).astype(np.float32)
        feeds[vi.name] = Tensor(arr)
    return feeds


def _tinygrad_totals(path, model):
    """(bytes, kernels) of tinygrad's per-kernel memory estimate for ``model``."""
    outputs = OnnxRunner(path)(_feeds(model))
    names = sorted(outputs)
    linear = compile_linear(Tensor.schedule_linear(*[outputs[n] for n in names]))
    mem, kernels = 0, 0
    for call in linear.src:
        if call.src[0].op is Ops.COPY:
            continue
        mem += int(estimate_uop(call).mem)
        kernels += 1
    return mem, kernels


def main():
    workdir = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp()
    os.makedirs(workdir, exist_ok=True)
    print(
        f"{'model':14s} {'before':>10s} {'after':>10s} {'saved':>7s} {'kernels':>9s}"
        f" | {'tinygrad mem':>12s} {'tg kernels':>10s} {'after vs tg':>12s}"
    )
    for name, path, model in _models(workdir):
        before, nb = _total(model, epilogue=False)
        after, na = _total(model, epilogue=True)
        pct = 100.0 * (before - after) / before if before else 0.0
        if _TINYGRAD_ERROR:
            tg = f"{'n/a':>12s} {'n/a':>10s} {'n/a':>12s}"
        else:
            try:
                tg_mem, tg_kernels = _tinygrad_totals(path, model)
            except Exception as e:  # report it in the table, keep the other rows
                reason = str(e).splitlines()[0][:40] if str(e) else type(e).__name__
                tg = f"{'failed':>12s} {reason}"
            else:
                ratio = 100.0 * (after - tg_mem) / tg_mem if tg_mem else 0.0
                tg = f"{tg_mem:>12d} {tg_kernels:>10d} {ratio:>+11.1f}%"
        print(
            f"{name:14s} {before:>10d} {after:>10d} {pct:>6.1f}% {nb:>4d}/{na:<4d}"
            f" | {tg}"
        )
    if _TINYGRAD_ERROR:
        print(f"tinygrad comparison unavailable: {_TINYGRAD_ERROR}")


if __name__ == "__main__":
    main()
