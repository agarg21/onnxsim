# Integration test: onnxsim's attention fusion on real Hugging Face encoder
# sequence classifiers -- the cross-encoder / reranker / zero-shot (NLI)
# family that scores a text against choices or answers yes/no -- exported
# with PyTorch's own exporters, not hand-built graphs.
#
# For each model, attention implementation (eager, and SDPA: transformers'
# default) and exporter (dynamo with a dynamic batch/sequence, legacy
# TorchScript with static shapes), the export is simplified with plain
# onnxsim.simplify() and checked for:
#   * every encoder layer's attention fused into one com.microsoft Attention
#     whose padding mask is its attention_bias input (fuse_attention.h),
#   * the classifier's probabilities matching transformers' own forward pass,
#     with a padded batch row, and for the dynamo export at other shapes,
#   * a fully padded row matching the unsimplified export (with SDPA's -inf
#     mask that row is NaN before the exported guard zeroes it).
# tests/test_fuse_attention_padding_mask.py covers the same rewrite on
# parser-built and small torch-built models; this checks real checkpoints.
#
# Same optional-dependency conventions as tests/test_export_transformers.py:
# torch and transformers are not normal test dependencies, so this skips
# unless they are importable, and skips (rather than fails) on a network
# error downloading a model. Run by the `optimum` job of
# .github/workflows/backend-integration.yml. Locally::
#
#     pip install torch --index-url https://download.pytorch.org/whl/cpu
#     pip install transformers onnxscript
#     pytest tests/test_encoder_classifier_hf_integration.py -v

import contextlib
import io
import warnings

import numpy as np
import onnx
import pytest

import onnxsim

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
ort = pytest.importorskip("onnxruntime")

MODELS = [
    "hf-internal-testing/tiny-random-BertForSequenceClassification",
    "hf-internal-testing/tiny-random-RobertaForSequenceClassification",
    "hf-internal-testing/tiny-random-DistilBertForSequenceClassification",
    "hf-internal-testing/tiny-random-ElectraForSequenceClassification",
]
B, S = 2, 7
NAMES = ["input_ids", "attention_mask"]


def _load(model_id, attn):
    try:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_id, attn_implementation=attn
        )
    except Exception as e:  # network/hub errors surface as a variety of types
        pytest.skip(f"Could not load {model_id} from Hugging Face Hub: {e}")
    return model.eval()


def _inputs(vocab_size, batch=B, seq=S, padded_from=4):
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(3, vocab_size, (batch, seq), generator=g)
    mask = torch.ones(batch, seq, dtype=torch.long)
    mask[-1, padded_from:] = 0
    return ids, mask


def _export(model, path, dynamo):
    args = _inputs(model.config.vocab_size)
    kwargs = {"input_names": NAMES, "output_names": ["logits"], "dynamo": dynamo}
    if dynamo:
        dims = {0: "batch", 1: "seq"}
        kwargs.update(opset_version=18, dynamic_shapes={n: dims for n in NAMES})
    else:
        kwargs.update(opset_version=17)
    with (
        warnings.catch_warnings(),
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        warnings.simplefilter("ignore")
        torch.onnx.export(model, args, str(path), **kwargs)
    return onnx.load(str(path))


def _run(model, ids, mask):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    feeds = {"input_ids": ids.numpy(), "attention_mask": mask.numpy()}
    names = {i.name for i in sess.get_inputs()}
    return sess.run(None, {k: v for k, v in feeds.items() if k in names})[0]


@pytest.mark.parametrize(
    "dynamo", [True, False], ids=["dynamo_dynamic", "legacy_static"]
)
@pytest.mark.parametrize("attn", ["eager", "sdpa"])
@pytest.mark.parametrize("model_id", MODELS, ids=lambda m: m.split("-")[-1])
def test_every_encoder_layer_fuses_and_matches_transformers(
    tmp_path, model_id, attn, dynamo
):
    if dynamo:
        pytest.importorskip("onnxscript")
    model = _load(model_id, attn)
    exported = _export(model, tmp_path / "model.onnx", dynamo)

    simplified, ok = onnxsim.simplify(exported)
    assert ok
    attention = [n for n in simplified.graph.node if n.op_type == "Attention"]
    assert len(attention) == model.config.num_hidden_layers
    for node in attention:
        assert node.domain == "com.microsoft"
        assert len(node.input) == 6 and node.input[5]  # attention_bias
    assert not any(n.op_type == "Softmax" for n in simplified.graph.node)
    assert len(simplified.graph.node) < len(exported.graph.node)

    shapes = [(B, S, 4)] + ([(1, 3, 2), (3, 6, 5)] if dynamo else [])
    for batch, seq, padded_from in shapes:
        ids, mask = _inputs(model.config.vocab_size, batch, seq, padded_from)
        with torch.no_grad():
            expected = model(input_ids=ids, attention_mask=mask).logits.numpy()
        np.testing.assert_allclose(
            _run(simplified, ids, mask), expected, rtol=1e-4, atol=1e-5
        )

    ids, mask = _inputs(model.config.vocab_size, padded_from=0)
    expected = _run(exported, ids, mask)
    assert not np.isnan(expected).any()
    np.testing.assert_allclose(
        _run(simplified, ids, mask), expected, rtol=1e-4, atol=1e-5
    )


@pytest.mark.parametrize(
    "dynamo", [True, False], ids=["dynamo_dynamic", "legacy_static"]
)
@pytest.mark.parametrize("model_id", MODELS, ids=lambda m: m.split("-")[-1])
def test_sdpa_nan_guard_removed_without_fusion(tmp_path, model_id, dynamo):
    # The NPU-legalization path (scripts/axera/pulsar2_backend.SIMPLIFY_KWARGS):
    # no attention fusion, and eliminate_sdpa_nan_guard removes every SDPA
    # `Where(IsNaN(P), 0, P)` -- the dynamo export's finfo.min mask directly,
    # the legacy export's -inf mask via finfo.min plus a row-validity multiply.
    # A fully padded row must still match the export (zeros, for -inf).
    if dynamo:
        pytest.importorskip("onnxscript")
    model = _load(model_id, "sdpa")
    exported = _export(model, tmp_path / "model.onnx", dynamo)
    # torch 2.11 exports the guard; torch 2.14 no longer does. Either way the
    # result must be guard- and Attention-free and match the export.

    simplified, ok = onnxsim.simplify(
        exported,
        attention_fusion_backend="none",
        extra_optimizers=["eliminate_sdpa_nan_guard"],
    )
    assert ok
    ops = {n.op_type for n in simplified.graph.node}
    assert not ops & {"IsNaN", "Attention"}

    for padded_from in (4, 0):
        ids, mask = _inputs(model.config.vocab_size, padded_from=padded_from)
        np.testing.assert_allclose(
            _run(simplified, ids, mask),
            _run(exported, ids, mask),
            rtol=1e-4,
            atol=1e-5,
        )
