#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real-architecture benchmark of the per-kernel traffic model.

Builds three architectures with random weights (no download): a tiny Llama and
a tiny BERT from transformers' configs, and a small residual CNN. Each is
exported to ONNX with torch, simplified with onnxsim (the way a user would
deploy it), and costed with ``kernel_traffic_breakdown`` before and after
epilogue fusion (an elementwise op absorbed into the MatMul, Gemm or Conv that
produces its input). Prints one row per model.

If tinygrad is installed, its per-kernel memory estimate is printed next to
onnxsim's: the two conventions both count each byte once per kernel, but
tinygrad also fuses differently, so the gap is informative, not an error.

usage: kernel_traffic_real.py [WORKDIR]    (needs torch and transformers)
"""

import os
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")

try:
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
        yield name, simplified


def _total(model, epilogue):
    real = model_info._is_kernel_epilogue_producer
    model_info._is_kernel_epilogue_producer = real if epilogue else (lambda node: False)
    try:
        breakdown = kernel_traffic_breakdown(model)
    finally:
        model_info._is_kernel_epilogue_producer = real
    return sum(breakdown.values()), len(breakdown)


def main():
    workdir = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp()
    os.makedirs(workdir, exist_ok=True)
    print(f"{'model':14s} {'before':>10s} {'after':>10s} {'saved':>7s} {'kernels':>9s}")
    for name, model in _models(workdir):
        before, nb = _total(model, epilogue=False)
        after, na = _total(model, epilogue=True)
        pct = 100.0 * (before - after) / before if before else 0.0
        print(f"{name:14s} {before:>10d} {after:>10d} {pct:>6.1f}% {nb:>4d}/{na:<4d}")


if __name__ == "__main__":
    main()
