"""Attention fusion for BERT-style encoder classifiers with a padding mask.

Cross-encoders, rerankers and zero-shot/NLI classifiers -- models that score a
text against a set of choices, or answer yes/no -- are encoders whose
attention adds a padding mask computed from the ``attention_mask`` graph
input::

    ext = (1 - Cast(attention_mask[:, None, None, :])) * finfo(float32).min
    scores = Q @ K^T / sqrt(head_size) + ext

and they are mostly exported below opset 23, where there is no ai.onnx
``Attention``. ``fuse_attention`` (``onnxsim/passes/fuse_attention.h``) emits
com.microsoft ``Attention`` for them, with the mask passed through as its
``attention_bias`` input (Expand-ed to the full [S, S] block). Each test checks
the fused form and that onnxruntime gives the same probabilities as the
unsimplified model, padded rows included.

The models are a small post-LayerNorm BERT (embeddings + LayerNorm, encoder
layers with an Erf-GELU FFN, a tanh pooler on the first token and a softmax
classifier head) built with the ONNX text parser, in the node shapes a
PyTorch export of Hugging Face's eager BERT produces.
"""

import collections

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
from onnx import parser

import onnxsim

ort = pytest.importorskip("onnxruntime")

B, S, C, H, F, VOCAB, LABELS = 2, 8, 32, 4, 64, 50, 3
D = C // H
FINFO_MIN = float(np.finfo(np.float32).min)


def _f32(rng, name, *shape):
    return onnx.numpy_helper.from_array(
        (rng.standard_normal(shape) * 0.2).astype(np.float32), name
    )


def _ln(x, prefix, out):
    return (
        f"{out} = LayerNormalization <axis=-1, epsilon=1e-12> "
        f"({x}, {prefix}_g, {prefix}_b)\n"
    )


def _linear(x, prefix, out):
    return f"{prefix}_mm = MatMul({x}, {prefix}_w)\n{out} = Add({prefix}_b, {prefix}_mm)\n"


def _layer(i, x, mask, mask_first):
    p = f"l{i}"
    s = ""
    for t in "qkv":
        s += _linear(x, f"{p}{t}", f"{p}{t}_lin")
        s += f"{p}{t}_rs = Reshape({p}{t}_lin, heads)\n"
        s += f"{p}{t}_h = Transpose <perm=[0,2,1,3]> ({p}{t}_rs)\n"
    scores = f"{p}msk = Add({mask}, {p}sc)" if mask_first else f"{p}msk = Add({p}sc, {mask})"
    s += f"""
    {p}k_t = Transpose <perm=[0,1,3,2]> ({p}k_h)
    {p}qk = MatMul({p}q_h, {p}k_t)
    {p}sc = Div({p}qk, sqrt_d)
    {scores}
    {p}p = Softmax <axis=-1> ({p}msk)
    {p}ctx4 = MatMul({p}p, {p}v_h)
    {p}ctxt = Transpose <perm=[0,2,1,3]> ({p}ctx4)
    {p}ctx = Reshape({p}ctxt, merged)
    """
    s += _linear(f"{p}ctx", f"{p}o", f"{p}o_lin")
    s += f"{p}r1 = Add({p}o_lin, {x})\n"
    s += _ln(f"{p}r1", f"{p}ln1", f"{p}h1")
    s += _linear(f"{p}h1", f"{p}f1", f"{p}f1_lin")
    s += f"""
    {p}g_d = Div({p}f1_lin, sqrt2)
    {p}g_e = Erf({p}g_d)
    {p}g_a = Add({p}g_e, one)
    {p}g_m = Mul({p}f1_lin, {p}g_a)
    {p}g = Mul({p}g_m, half)
    """
    s += _linear(f"{p}g", f"{p}f2", f"{p}f2_lin")
    s += f"{p}r2 = Add({p}f2_lin, {p}h1)\n"
    s += _ln(f"{p}r2", f"{p}ln2", f"{p}out")
    return s, f"{p}out"


def _encoder_classifier(opset=17, layers=2, batch=B, seq=S, mask_first=False):
    """A BERT-style sequence classifier. `batch`/`seq` may be dim_param names."""
    rng = np.random.default_rng(0)
    inits = [
        _f32(rng, "word_emb", VOCAB, C),
        _f32(rng, "pos_emb", S, C),
        _f32(rng, "tt_emb", 2, C),
        _f32(rng, "pool_w", C, C),
        _f32(rng, "pool_b", C),
        _f32(rng, "cls_w", LABELS, C),
        _f32(rng, "cls_b", LABELS),
    ]
    for prefix in ["emb_ln"] + [f"l{i}ln{j}" for i in range(layers) for j in (1, 2)]:
        inits += [_f32(rng, f"{prefix}_g", C), _f32(rng, f"{prefix}_b", C)]
    for i in range(layers):
        for name, k, n in [("q", C, C), ("k", C, C), ("v", C, C), ("o", C, C),
                           ("f1", C, F), ("f2", F, C)]:
            inits += [_f32(rng, f"l{i}{name}_w", k, n), _f32(rng, f"l{i}{name}_b", n)]

    body = f"""
    heads = Constant <value = int64[4] {{0, 0, {H}, {D}}}> ()
    merged = Constant <value = int64[3] {{0, 0, {C}}}> ()
    sqrt_d = Constant <value = float {{{float(np.sqrt(D))}}}> ()
    sqrt2 = Constant <value = float {{1.4142135}}> ()
    one = Constant <value = float {{1.0}}> ()
    half = Constant <value = float {{0.5}}> ()
    fmin = Constant <value = float {{{FINFO_MIN!r}}}> ()
    ax1 = Constant <value = int64[1] {{1}}> ()
    ax2 = Constant <value = int64[1] {{2}}> ()
    zero = Constant <value = int64[1] {{0}}> ()
    cls = Constant <value = int64 {{0}}> ()
    ids_shape = Shape(input_ids)
    seq_len = Slice(ids_shape, ax1, ax2)
    seq_s = Squeeze(seq_len, zero)
    one_i = Constant <value = int64 {{1}}> ()
    zero_i = Constant <value = int64 {{0}}> ()
    pos = Range(zero_i, seq_s, one_i)
    we = Gather(word_emb, input_ids)
    pe = Gather(pos_emb, pos)
    te = Gather(tt_emb, token_type_ids)
    e1 = Add(we, pe)
    e2 = Add(e1, te)
    {_ln("e2", "emb_ln", "h0")}
    m1 = Unsqueeze(attention_mask, ax1)
    m2 = Unsqueeze(m1, ax2)
    mf = Cast <to=1> (m2)
    inv = Sub(one, mf)
    ext = Mul(inv, fmin)
    """
    x = "h0"
    for i in range(layers):
        s, x = _layer(i, x, "ext", mask_first)
        body += s
    body += f"""
    first = Gather <axis=1> ({x}, cls)
    pool_mm = Gemm <transB=1> (first, pool_w, pool_b)
    pooled = Tanh(pool_mm)
    logits = Gemm <transB=1> (pooled, cls_w, cls_b)
    probs = Softmax <axis=-1> (logits)
    """
    ids = f"int64[{batch},{seq}]"
    model = parser.parse_model(
        f"""<ir_version: 10, opset_import: ["": {opset}]>
        classifier ({ids} input_ids, {ids} attention_mask, {ids} token_type_ids)
            => (float[{batch},{LABELS}] probs) {{
        {body}
        }}"""
    )
    model.graph.initializer.extend(inits)
    onnx.checker.check_model(model)
    return model


def _feeds(batch=B, seq=S):
    rng = np.random.default_rng(1)
    mask = np.ones((batch, seq), np.int64)
    mask[-1, seq // 2 + 1 :] = 0  # Padding on the last row.
    tt = np.zeros((batch, seq), np.int64)
    tt[:, seq // 2 :] = 1
    return {
        "input_ids": rng.integers(0, VOCAB, (batch, seq)).astype(np.int64),
        "attention_mask": mask,
        "token_type_ids": tt,
    }


def _run(model, feeds):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)[0]


def _attention_nodes(model):
    return [n for n in model.graph.node if n.op_type == "Attention"]


def _assert_masked_ms_attention(model, layers):
    attn = _attention_nodes(model)
    assert len(attn) == layers
    for n in attn:
        assert n.domain == "com.microsoft"
        assert len(n.input) == 6
        assert n.input[3] == "" and n.input[4] == ""  # mask_index, past
        assert n.input[5]  # attention_bias
    ops = collections.Counter(n.op_type for n in model.graph.node)
    assert ops["Softmax"] == 1  # Only the classifier head's.


def test_padding_mask_fuses_to_ms_attention_with_attention_bias():
    model = _encoder_classifier()
    simplified, ok = onnxsim.simplify(model)
    assert ok
    onnx.checker.check_model(simplified)
    _assert_masked_ms_attention(simplified, layers=2)
    feeds = _feeds()
    np.testing.assert_allclose(
        _run(simplified, feeds), _run(model, feeds), rtol=1e-4, atol=1e-5
    )


def test_padding_mask_is_expanded_once_per_layer_to_the_score_block():
    # [B,1,1,S] does not cover the query axis, which attention_bias must.
    simplified, _ = onnxsim.simplify(_encoder_classifier(layers=1))
    (attn,) = _attention_nodes(simplified)
    producer = {o: n for n in simplified.graph.node for o in n.output}
    assert producer[attn.input[5]].op_type == "Expand"


def test_mask_as_first_add_operand():
    model = _encoder_classifier(mask_first=True)
    simplified, _ = onnxsim.simplify(model)
    _assert_masked_ms_attention(simplified, layers=2)
    feeds = _feeds()
    np.testing.assert_allclose(
        _run(simplified, feeds), _run(model, feeds), rtol=1e-4, atol=1e-5
    )


def test_dynamic_batch_and_sequence_length():
    model = _encoder_classifier(batch="batch", seq="seq")
    simplified, _ = onnxsim.simplify(model)
    _assert_masked_ms_attention(simplified, layers=2)
    for batch, seq in [(1, 3), (3, 8)]:
        feeds = _feeds(batch, seq)
        np.testing.assert_allclose(
            _run(simplified, feeds), _run(model, feeds), rtol=1e-4, atol=1e-5
        )


def test_opset_23_keeps_the_onnx_form_and_ort_contrib_gets_attention_bias():
    model = _encoder_classifier(opset=23)
    feeds = _feeds()
    expected = _run(model, feeds)

    auto, _ = onnxsim.simplify(model)
    assert {n.domain for n in _attention_nodes(auto)} == {""}
    np.testing.assert_allclose(_run(auto, feeds), expected, rtol=1e-4, atol=1e-5)

    contrib, _ = onnxsim.simplify(model, attention_fusion_backend="ort_contrib")
    _assert_masked_ms_attention(contrib, layers=2)
    np.testing.assert_allclose(_run(contrib, feeds), expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("backend", ["webgpu", "none", "onnx"])
def test_backends_that_cannot_take_the_mask_leave_it_unfused(backend):
    model = _encoder_classifier()
    simplified, _ = onnxsim.simplify(model, attention_fusion_backend=backend)
    assert _attention_nodes(simplified) == []
    feeds = _feeds()
    np.testing.assert_allclose(
        _run(simplified, feeds), _run(model, feeds), rtol=1e-4, atol=1e-5
    )



def _torch_encoder_classifier(torch, sdpa):
    # Hugging Face's BERT attention and padding mask, in a few lines: what its
    # PyTorch export emits, without needing transformers. Eager attention adds
    # the finfo.min extended mask to the scores; SDPA (transformers' default)
    # takes a bool mask, which the exporters turn into an additive one plus a
    # Where(IsNaN(P), 0, P) guard after the Softmax.
    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q, self.k, self.v, self.o = (torch.nn.Linear(C, C) for _ in range(4))
            self.ln1, self.ln2 = torch.nn.LayerNorm(C), torch.nn.LayerNorm(C)
            self.ff1, self.ff2 = torch.nn.Linear(C, F), torch.nn.Linear(F, C)

        def forward(self, x, ext):
            b, s, _ = x.shape

            def heads(t):
                return t.view(b, s, H, D).transpose(1, 2)

            q, k, v = heads(self.q(x)), heads(self.k(x)), heads(self.v(x))
            if sdpa:
                ctx = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=ext
                )
            else:
                scores = q @ k.transpose(-1, -2) / D**0.5 + ext
                ctx = torch.softmax(scores, -1) @ v
            ctx = ctx.transpose(1, 2).reshape(b, s, C)
            h = self.ln1(self.o(ctx) + x)
            return self.ln2(self.ff2(torch.nn.functional.gelu(self.ff1(h))) + h)

    class Classifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(VOCAB, C)
            self.layers = torch.nn.ModuleList(Layer() for _ in range(2))
            self.head = torch.nn.Linear(C, LABELS)

        def forward(self, input_ids, attention_mask):
            if sdpa:
                ext = attention_mask[:, None, None, :].bool()
            else:
                ext = (1.0 - attention_mask[:, None, None, :].float()) * FINFO_MIN
            x = self.emb(input_ids)
            for layer in self.layers:
                x = layer(x, ext)
            return torch.softmax(self.head(torch.tanh(x[:, 0])), -1)

    torch.manual_seed(0)
    return Classifier().eval()


def _torch_export(torch, module, path, dynamo):
    import contextlib
    import io

    feeds = _feeds()
    args = tuple(torch.from_numpy(feeds[n]) for n in ("input_ids", "attention_mask"))
    kwargs = dict(input_names=["input_ids", "attention_mask"], dynamo=dynamo)
    if dynamo:
        dims = {0: "batch", 1: "seq"}
        kwargs.update(opset_version=18, dynamic_shapes={"input_ids": dims, "attention_mask": dims})
    else:
        kwargs.update(opset_version=17)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        torch.onnx.export(module, args, str(path), **kwargs)
    return onnx.load(str(path))


@pytest.mark.parametrize("sdpa", [False, True], ids=["eager", "sdpa"])
@pytest.mark.parametrize("dynamo", [True, False], ids=["dynamo_dynamic", "legacy_static"])
def test_torch_exported_encoder_classifier(tmp_path, dynamo, sdpa):
    torch = pytest.importorskip("torch")
    if dynamo:
        pytest.importorskip("onnxscript")
    module = _torch_encoder_classifier(torch, sdpa)
    model = _torch_export(torch, module, tmp_path / "m.onnx", dynamo)
    simplified, ok = onnxsim.simplify(model)
    assert ok
    _assert_masked_ms_attention(simplified, layers=2)
    cases = [_feeds()] + ([_feeds(1, 3), _feeds(3, 6)] if dynamo else [])
    for feeds in cases:
        feeds = {k: feeds[k] for k in ("input_ids", "attention_mask")}
        with torch.no_grad():
            expected = module(*(torch.from_numpy(feeds[k]) for k in feeds))
        np.testing.assert_allclose(
            _run(simplified, feeds), expected.numpy(), rtol=1e-4, atol=1e-5
        )
    # An all-padding row: with the legacy SDPA export's -inf mask its
    # probabilities are NaN, which the exported guard turns into zeros, and so
    # must the fused graph. (Compared with the export rather than torch, whose
    # kernels and exporters disagree on such a row.)
    feeds = {k: v for k, v in _feeds().items() if k != "token_type_ids"}
    feeds["attention_mask"][-1] = 0
    expected = _run(model, feeds)
    assert not np.isnan(expected).any()
    np.testing.assert_allclose(_run(simplified, feeds), expected, rtol=1e-4, atol=1e-5)
