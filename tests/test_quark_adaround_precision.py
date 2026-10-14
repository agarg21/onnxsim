"""Float32 AdaRound avoids 16-bit rounding differences against Quark."""

import numpy as np
import pytest
import test_quark_finetune_coverage_parity as C
import test_quark_finetune_parity as P
from onnx import numpy_helper, parser

from onnxsim import quark_finetune as qf

pytestmark = C.pytestmark


@pytest.mark.parametrize("seed", [0, 5, 11])
def test_int16_adaround_rounding_matches_quark(seed, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(seed)
    model = parser.parse_model(
        '<ir_version: 9, opset_import: ["": 17]> g (float[8,12] x) => (float[8,7] y) { y=MatMul(x,w) }'
    )
    model.graph.node[0].name = "mm"
    model.graph.initializer.append(
        numpy_helper.from_array(rng.normal(size=(12, 7)).astype(np.float32), "w")
    )
    data = [{"x": rng.normal(size=(8, 12)).astype(np.float32)} for _ in range(3)]
    ff = P._ff("adaround", NumIterations=100, BatchSize=3, OutputQDQ=True)
    used = []
    original = qf._adam_step32

    def adam32(p, *args):
        used.append(p.dtype)
        return original(p, *args)

    monkeypatch.setattr(qf, "_adam_step32", adam32)
    quark, mine, *_ = C._both(model, data, ff, preset="INT16_CNN_DEFAULT")
    assert used and set(used) == {np.dtype(np.float32)}
    mismatch32 = max(C._mismatch(quark, mine).values())

    # The same run in float64. float32 is there because float64 drifts from
    # Quark's torch float32 codes, so compare the two rather than demanding exact
    # codes: a host's float32 kernels can flip a few rounding ties.
    orig_options = P._options

    def float64_options(ff64):
        o = orig_options(ff64)
        o.float32 = False
        return o

    monkeypatch.setattr(P, "_options", float64_options)
    quark64, mine64, *_ = C._both(model, data, ff, preset="INT16_CNN_DEFAULT")
    monkeypatch.setattr(P, "_options", orig_options)
    mismatch64 = max(C._mismatch(quark64, mine64).values())
    assert mismatch32 <= mismatch64
    # ... and within a few rounding ties of Quark's codes (one flip of 84 here)
    assert mismatch32 <= 0.02
