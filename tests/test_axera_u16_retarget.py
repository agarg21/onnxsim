"""No-device checks for ``scripts/axera/u16_retarget.py`` against native Pulsar2
U16 single-op builds.

The builds are local scratch (not committed), from ``AXERA_U16_PROGRAM_BUILDS``
(default: the pulsar-free scratch tree); tests whose build is missing are
skipped. ``recal_l0/*`` and ``u16_elementwise/*`` are the SmolLM2-135M layer-0
programs (real-activation calibration); ``u16_pf/<op>/v*`` are calibration
variants of one op and shape (inputs uniform over chosen ranges plus the two
range ends).
"""

import itertools
import os
import sys

import numpy as np
import onnx
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "axera"))

import graph_stitch as gs  # noqa: E402
import u16_retarget as ur  # noqa: E402

B = os.environ.get(
    "AXERA_U16_PROGRAM_BUILDS", "/mnt/data/cache/claude-work/tmp/llm-pulsar-free"
)
RC, EW, V = f"{B}/recal_l0", f"{B}/u16_elementwise", f"{B}/u16_pf"


def _v(name, n):
    return [(f"{V}/{name}/v{i}", "y") for i in range(n)]


# group -> (op, inputs, attrs, [(build, output name)]): one op and shape each
GROUPS = {
    "mulxx576": (
        "Mul",
        ("x", "x"),
        None,
        [
            (f"{RC}/n1_sq", "y"),
            (f"{RC}/nf_sq", "y"),
            (f"{EW}/u16_ew_n_sq_1x576_2x", "z"),
            (f"{EW}/u16_ew_n_sq_1x576_real", "z"),
        ],
    ),
    "mean576": (
        "ReduceMean",
        ("x",),
        {"count": 576},
        [
            (f"{RC}/n1_m", "y"),
            (f"{RC}/nf_m", "y"),
            (f"{EW}/u16_ew_mean_1x576_2x", "z"),
            (f"{EW}/u16_ew_mean_1x576_real", "z"),
        ],
    ),
    "div576": (
        "Div",
        ("x", "r"),
        None,
        [
            (f"{RC}/n1_d", "y"),
            (f"{RC}/nf_d", "y"),
            (f"{EW}/u16_ew_n_d_1x576_2x", "z"),
            (f"{EW}/u16_ew_n_d_1x576_real", "z"),
        ],
    ),
    "mulg576": (
        "Mul",
        ("x", "g"),
        None,
        [
            (f"{RC}/n1_g", "y"),
            (f"{RC}/nf_g", "y"),
            (f"{EW}/u16_ew_m_1x576_2x", "z"),
            (f"{EW}/u16_ew_m_1x576_real", "z"),
        ],
    ),
    "addeps": (
        "Add",
        ("x", "eps"),
        None,
        [(f"{RC}/n1_e", "y"), (f"{RC}/n2_e", "y"), (f"{RC}/nf_e", "y")],
    ),
    "sqrt": (
        "Sqrt",
        ("x",),
        None,
        [(f"{RC}/n1_r", "y"), (f"{RC}/n2_r", "y"), (f"{RC}/nf_r", "y")],
    ),
    "add576": (
        "Add",
        ("a", "b"),
        None,
        [(f"{RC}/res1", "y"), (f"{RC}/res2", "y"), (f"{RC}/rq_add16", "y")]
        + _v("add576", 4)[1:],
    ),
    "add192": ("Add", ("a", "b"), None, [(f"{RC}/rk_add16", "y")] + _v("add192", 4)),
    "mul576": (
        "Mul",
        ("a", "b"),
        None,
        [(f"{RC}/rq_a16", "y"), (f"{RC}/rq_b16", "y")] + _v("mul576", 4)[1:],
    ),
    "mul192": ("Mul", ("a", "b"), None, [(f"{RC}/rk_a16", "y"), (f"{RC}/rk_b16", "y")]),
    "mul1536": (
        "Mul",
        ("a", "b"),
        None,
        [(f"{RC}/silu16", "y"), (f"{RC}/act16", "y")] + _v("mul1536", 1),
    ),
    # the attention-mask Add between score and softmax (mask [-32, 0] -> zero point 65535)
    "madd129": ("Add", ("x", "mask"), None, _v("madd129", 4)),
    "sig1536": ("Sigmoid", ("x",), None, [(f"{RC}/sig16", "y")] + _v("sig1536", 4)),
    "smx129": ("Softmax", ("x",), None, [(f"{RC}/smx16", "y")] + _v("smx129", 4)),
    "score": ("MatMul", ("a", "b"), None, [(f"{RC}/score16", "y")] + _v("score", 4)),
    "pv": ("MatMul", ("a", "b"), None, [(f"{RC}/pv16", "y")] + _v("pv", 4)),
    # llama-160m's residual Add (an input over exactly [-1, 1]: zero point 32768)
    "add768": ("Add", ("a", "b"), None, _v("add768", 4)),
}
PAIRS = [
    (g, t, n)
    for g, (_, _, _, members) in GROUPS.items()
    for t, n in itertools.permutations(range(len(members)), 2)
]
REFUSALS = (
    "cannot be located",
    "ambiguous template",
    "repeats a write",
    "elided Add-shift write",
)


def _need(directory):
    path = os.path.join(directory, "out", "compiled.axmodel")
    if not os.path.exists(path):
        pytest.skip(f"native build not present: {directory}")
    return path


def _q(directory, out):
    return {("y" if k == out else k): v for k, v in ur.quant_of(directory).items()}


@pytest.mark.parametrize("group,t,n", PAIRS)
def test_retarget_equals_the_native_build_or_refuses(group, t, n):
    """Every template -> target pair of a group: the retargeted program equals
    the native build of the target calibration, or the template is refused
    for a stated reason (a zero point below 256, two tensors sharing a zero
    point, a write the template already left out). Never a silent difference."""
    op, ins, attrs, members = GROUPS[group]
    (td, to), (nd, no) = members[t], members[n]
    template, want = _need(td), gs.load_model(_need(nd))
    try:
        got = ur.retarget(template, op, _q(td, to), _q(nd, no), ins, attrs=attrs)
    except ValueError as e:
        assert any(r in str(e) for r in REFUSALS), e
        return
    assert ur.compare(got, want) == []


@pytest.mark.parametrize("group", list(GROUPS))
def test_every_group_has_a_working_template(group):
    op, ins, attrs, members = GROUPS[group]
    ok = 0
    for (td, to), (nd, no) in itertools.permutations(members, 2):
        _need(td), _need(nd)
        try:
            got = ur.retarget(
                f"{td}/out/compiled.axmodel",
                op,
                _q(td, to),
                _q(nd, no),
                ins,
                attrs=attrs,
            )
        except ValueError:
            continue
        ok += ur.compare(got, gs.load_model(f"{nd}/out/compiled.axmodel")) == []
    assert ok >= 1


def test_elision_of_a_zero_zero_point():
    """act16 (both input zero points nonzero) to silu16's calibration, whose
    sigmoid input has zero point 0: the core and DEQUANT writes of 0x1ad0 are
    left out and the segment loses its padding, as in the native build."""
    act, silu = _need(f"{RC}/act16"), f"{RC}/silu16"
    got = ur.retarget(act, "Mul", _q(f"{RC}/act16", "y"), _q(silu, "y"), ("a", "b"))
    want = gs.load_model(_need(silu))
    assert ur.compare(got, want) == []
    assert len(ur._segs(got)[gs.MAIN]) < len(ur._segs(gs.load_model(act))[gs.MAIN])
    with pytest.raises(ValueError, match="cannot be located"):
        ur.retarget(
            _need(silu), "Mul", _q(silu, "y"), _q(f"{RC}/act16", "y"), ("a", "b")
        )


CALIBRATED = [
    (f"{RC}/{k}", op)
    for k, op in [
        ("n1_sq", "Mul"),
        ("n1_m", "ReduceMean"),
        ("n1_e", "Add"),
        ("nf_e", "Add"),
        ("n1_r", "Sqrt"),
        ("n1_d", "Div"),
        ("nf_d", "Div"),
        ("n1_g", "Mul"),
        ("nf_g", "Mul"),
        ("res1", "Add"),
        ("res2", "Add"),
        ("rq_a16", "Mul"),
        ("rq_b16", "Mul"),
        ("rk_b16", "Mul"),
        ("rq_add16", "Add"),
        ("score16", "MatMul"),
        ("smx16", "Softmax"),
        ("pv16", "MatMul"),
        ("sig16", "Sigmoid"),
        ("silu16", "Mul"),
        ("act16", "Mul"),
    ]
] + [(f"{EW}/u16_ew_mean_1x576_2x", "ReduceMean"), (f"{EW}/u16_ew_n_d_1x576_2x", "Div")]
CALIBRATED += [
    (f"{V}/madd129/v{i}", "Add") for i in range(4)
]  # v3: an exact .5 at an asymmetric range
CALIBRATED += [
    (f"{V}/add768/v{i}", "Add") for i in range(4)
]  # an Add input over [-1, 1]: 32768, where a Mul operand gets 32767


@pytest.mark.parametrize("directory,op", CALIBRATED)
def test_calibrate_reproduces_pulsar2(directory, op):
    _need(directory)
    assert ur.calibrate(directory, op) == ur.quant_of(directory)


def test_calibrate_reduce_mean_output_within_two_ulps():
    # the one known difference: Pulsar2's float forward reduces in an order not reproduced
    d = f"{RC}/nf_m"
    _need(d)
    mine, native = ur.calibrate(d, "ReduceMean"), ur.quant_of(d)
    assert mine["x"] == native["x"]
    assert mine["y"][1] == native["y"][1]
    assert (
        abs(
            np.float32(mine["y"][0]).view(np.int32)
            - np.float32(native["y"][0]).view(np.int32)
        )
        <= 2
    )


def test_symmetric_range_zero_point():
    s, zp = ur.unsigned_minmax(-0.1751, 0.1751)
    assert zp == 32767  # rint(-lo / s) = rint(32767.5...) would give 32768
    assert (
        ur.unsigned_minmax(-36.0, 4.0)[1] == 58982
    )  # an asymmetric exact .5 rounds up
    assert ur.unsigned_minmax(0.0, 2.0)[1] == 0
    assert ur.unsigned_minmax(-2.0, 0.0)[1] == 65535


def test_add_shift_follows_the_ratio_constant():
    q = {"a": (2e-4, 300), "b": (5e-5, 400), "y": (1.5e-4, 500)}  # s_a / s_y > 1
    c = ur.candidates("Add", q, ("a", "b"))
    assert c["add_shift@0x1ea0"] == 14
    lo, hi = c["add_const"] & 0xFFFF, c["add_const"] >> 16
    assert (lo, hi) == (round(2e-4 / 1.5e-4 * 2**14), round(5e-5 / 1.5e-4 * 2**14))


def test_sigmoid_table_is_monotonic_and_saturates():
    c = ur.candidates("Sigmoid", {"x": (1e-4, 30000), "y": (1.5e-5, 0)}, ("x",))
    t = []
    for k in range(128):
        v = c[f"sig_t{k}@{0x1050 + 16 * k:#06x}"]
        t += [v & 0xFFFF, v >> 16]
    assert all(a <= b for a, b in zip(t, t[1:]))
    assert c["sig_tail@0x1850"] >= t[-1]


def test_unknown_op_refused():
    with pytest.raises(ValueError, match="not a retargetable"):
        ur.candidates("Conv", {"x": (1.0, 300), "y": (1.0, 300)}, ("x",))


# ---- fused (multi-op) programs: u16_fused/<name>/v0..v3, four calibrations each ----------
FUSED = {
    "rms576": "RMSNorm x, g -> y (Mul x*x, ReduceMean, Add eps, Sqrt, Div, Mul gain)",
    "silu1536": "SiLU gate, up -> y (Sigmoid, Mul, Mul)",
    "rope576": "RoPE apply q, r, c, sn -> y (Mul, Mul, Add)",
    "rope192": "RoPE apply at 192",
    "crms576": "coarse RMS x -> x / Sqrt(ReduceMean(x*x) + eps), no gain",
    "attn3": "attention of one kv group: q3 [3,64], K, V [129,64], mask -> [3,64]; "
    "7 CORE jobs for 5 nodes (Softmax takes two, Transpose quantizes its output)",
    "attn_mha768": "llama-160m's whole-layer MHA attention: q [1,768], K, V [129,768], mask "
    "-> [1,768], Reshape / Transpose inside; four inputs whose listing order flips",
    "crms768": "coarse RMS at 768 (eps 1e-6: the eps constant gets MIN_SCALE)",
    "rms768": "RMSNorm at 768",
    "rope768": "RoPE apply at 768",
    "silu3072": "SiLU at 3072",
    # Qwen3-0.6B
    "rms1024": "RMSNorm at 1024",
    "crms1024": "coarse RMS at 1024",
    "hcrms2048": "per-head coarse RMS of [k | k_rot]: Reshape to heads, 5 CORE jobs for 7 nodes",
    "qkrope1024": "per-head k norm + RoPE: (q c + r sn) / rms_head(q), 8 CORE jobs for 11 "
    "nodes, x*x runs first; both Adds shift 15, so 0x1ea0 is written once for them",
    "qkrope2048": "per-head q norm + RoPE at 2048",
    "qkrope1024t": "qkrope1024 with r ~ -q: the RoPE Add shifts 11-12, both 0x1ea0 writes kept",
    "qkrope2048t": "qkrope2048 with r ~ -q",
    "attn_gqa2048": "Qwen3's whole-layer GQA attention (16 q heads, 8 kv heads), 54 CORE jobs",
}
# calibrate_graph's largest scale distance to Pulsar2's, in float32 ulps (zero points equal)
FUSED_ULPS = {
    "rms576": 1,
    "attn3": 2,
    "attn_mha768": 4,
    "crms576": 2,
    "crms768": 1,
    "rms768": 2,
    "rms1024": 2,
    "crms1024": 2,
    "hcrms2048": 2,
    "qkrope1024": 2,
    "qkrope2048": 2,
    "qkrope1024t": 2,
    "qkrope2048t": 2,
    "attn_gqa2048": 4,
}
FUSED_PAIRS = [(g, t, n) for g in FUSED for t, n in itertools.permutations(range(4), 2)]


def _fused(name, i):
    return f"{B}/u16_fused/{name}/v{i}"


@pytest.mark.parametrize("group,t,n", FUSED_PAIRS)
def test_retarget_graph_equals_the_native_build_or_refuses(group, t, n):
    td, nd = _fused(group, t), _fused(group, n)
    template, want = _need(td), gs.load_model(_need(nd))
    try:
        got = ur.retarget_graph(
            template, f"{td}/model.onnx", ur.quant_of(td), ur.quant_of(nd)
        )
    except ValueError as e:
        assert any(r in str(e) for r in REFUSALS), e
        return
    assert ur.compare(got, want) == []


@pytest.mark.parametrize("group", list(FUSED))
def test_every_fused_group_has_a_working_template(group):
    ok = 0
    for t, n in itertools.permutations(range(4), 2):
        td, nd = _fused(group, t), _fused(group, n)
        _need(td), _need(nd)
        try:
            got = ur.retarget_graph(
                f"{td}/out/compiled.axmodel",
                f"{td}/model.onnx",
                ur.quant_of(td),
                ur.quant_of(nd),
            )
        except ValueError:
            continue
        ok += ur.compare(got, gs.load_model(f"{nd}/out/compiled.axmodel")) == []
    assert ok >= 1


@pytest.mark.parametrize("group,i", [(g, i) for g in FUSED for i in range(4)])
def test_calibrate_graph_reproduces_pulsar2(group, i):
    """Every zero point equal; scales equal except a fused RMSNorm's inner
    scales, which can be one float32 ulp off."""
    d = _fused(group, i)
    _need(d)
    mine, native = ur.calibrate_graph(d), ur.quant_of(d)
    assert set(native) <= set(mine)
    for t, (s, zp) in native.items():
        assert mine[t][1] == zp, t
        ulps = abs(
            int(np.float32(mine[t][0]).view(np.int32))
            - int(np.float32(s).view(np.int32))
        )
        assert ulps <= FUSED_ULPS.get(group, 0), (t, mine[t], (s, zp))


def test_graph_nodes_reduce_mean_width():
    model = onnx.parser.parse_model(
        '<ir_version: 9, opset_import: ["" : 17]> '
        "g (float[1,576] x) => (float[1,1] y) "
        "{ sq = Mul(x, x) y = ReduceMean<axes=[-1], keepdims=1>(sq) }"
    )
    nodes = ur.graph_nodes(model)
    assert [n[0] for n in nodes] == ["Mul", "ReduceMean"]
    assert nodes[1][3] == {"count": 576}


def test_add_shift_keeps_an_exact_ratio_of_one():
    """A ratio of exactly 1.0 (an eps Add whose input and output share a
    scale) is stored as 32768 at shift 15; the U8 rule would shift it."""
    assert ur.add_shift(1.0, 0.5) == 0
    assert ur.add_shift(1.0 + 0.4 / 32768, 0.1) == 0  # still rounds to 32768
    assert ur.add_shift(1.0 + 1.0 / 32768, 0.1) == 1
    assert ur.add_shift(1.9, 0.1) == 1 and ur.add_shift(2.5, 0.1) == 2
    q = {"x": (2e-5, 300), "eps": (1e-10, 0), "y": (2e-5, 300)}
    c = ur.candidates("Add", q, ("x", "eps"))
    assert c["add_shift@0x1ea0"] == 15 and c["add_const"] & 0xFFFF == 32768


def test_symmetric_zero_point_depends_on_the_op():
    assert ur.unsigned_minmax(-1.0, 1.0)[1] == 32767  # a Mul operand (RoPE's sin)
    assert (
        ur.unsigned_minmax(-1.0, 1.0, symmetric_32767=False)[1] == 32768
    )  # an Add input
    assert ur.unsigned_minmax(-1.0, 2.0, symmetric_32767=False)[1] == 21845


def test_min_scale():
    s, zp = ur.unsigned_minmax(0.0, 1e-6)
    assert s == float(np.float32(1e-10)) and zp == 0
    assert ur.unsigned_minmax(0.0, 1e-5)[0] > float(np.float32(1e-10))


def test_propagate_through_data_movement():
    model = onnx.parser.parse_model(
        '<ir_version: 9, opset_import: ["" : 17]> '
        "g (float[1,768] x) => (float[12,64] y) "
        "{ t = Relu(x) shape = Constant<value = int64[2] {12, 64}>() "
        "y = Reshape(t, shape) }"
    )
    q = ur.propagate(model, {"x": (1e-3, 100), "t": (2e-3, 0)})
    assert q["y"] == (2e-3, 0)


# Qwen3-0.6B layer 11's k norm + RoPE, built natively at its own calibration: its RoPE
# Add shifts 14 and its eps Add 15, so the native program writes 0x1ea0 for both
NATIVE_QKROPE = f"{B}/gen_model/native_L11_rk"


@pytest.mark.parametrize("i", range(4))
def test_elided_add_shift_write_refused(i):
    """qkrope1024's two Adds both shift 15, so Pulsar2 wrote 0x1ea0 once for
    both; retargeted to layer 11 the eps Add would run at the RoPE Add's 14
    (the card's k was 29% off). qkrope1024t's Adds shift differently, keeps
    both writes and gives the native program."""
    td, tt = _fused("qkrope1024", i), _fused("qkrope1024t", i)
    _need(td), _need(tt), _need(NATIVE_QKROPE)
    target = ur.quant_of(NATIVE_QKROPE)
    with pytest.raises(ValueError, match="elided Add-shift write"):
        ur.retarget_graph(_need(td), f"{td}/model.onnx", ur.quant_of(td), target)
    got = ur.retarget_graph(_need(tt), f"{tt}/model.onnx", ur.quant_of(tt), target)
    want = gs.load_model(_need(NATIVE_QKROPE))
    assert ur.compare(got, want) == []


def test_core_jobs_follow_the_records_not_the_node_order():
    """qkrope runs x*x (node 4) first; an RMSNorm keeps the ONNX order."""
    for name, first in (("qkrope1024", "n4"), ("rms576", "n0")):
        d = _fused(name, 0)
        template = _need(d)
        nodes = ur.graph_nodes(f"{d}/model.onnx")
        q = ur.propagate(onnx.load(f"{d}/model.onnx"), ur.quant_of(d))
        comp = [i for i, n in enumerate(nodes) if n[0] not in ur.PASS_THROUGH]
        main = ur._segs(gs.load_model(template))[gs.MAIN]
        got = ur._assign_core_jobs(main, comp, ur._graph_candidates(nodes, q))
        assert got[0] == first
        assert sorted(got.values()) == sorted(f"n{i}" for i in comp)


# ---- merged programs of the Pulsar-free decode (pf_templates: nrm / anorm / qkf /
# lanorm / slanorm), four calibrations each, under AXERA_U16_MERGED_BUILDS ------------
M = os.environ.get(
    "AXERA_U16_MERGED_BUILDS", "/mnt/data/cache/claude-work/tmp/qwen-fuse/templates"
)
#: a linear (FullyConnected, its weight table written by linear_emit) inside the
#: program: every pair equals the native build or is refused
MERGED_EXACT = ("lanorm2048", "lanorm3072", "slanorm3072")
#: no linear; some pairs differ from the native build in job order or IO slot
#: numbering only (Pulsar2 rebuild noise), at least one pair is byte-equal
MERGED = ("nrm1024", "anorm1024", "qkf2048") + MERGED_EXACT


def _merged(name, i):
    return f"{M}/fused/{name}/v{i}"


@pytest.mark.parametrize(
    "group,t,n",
    [(g, t, n) for g in MERGED_EXACT for t, n in itertools.permutations(range(4), 2)],
)
def test_merged_linear_program_equals_the_native_build_or_refuses(group, t, n):
    td, nd = _merged(group, t), _merged(group, n)
    template, want = _need(td), gs.load_model(_need(nd))
    try:
        got = ur.retarget_graph(
            template, f"{td}/model.onnx", ur.quant_of(td), ur.quant_of(nd)
        )
    except ValueError as e:
        assert any(r in str(e) for r in REFUSALS), e
        return
    assert ur.compare(got, want) == []


@pytest.mark.parametrize("group", MERGED)
def test_every_merged_group_has_a_working_template(group):
    ok = 0
    for t, n in itertools.permutations(range(4), 2):
        td, nd = _merged(group, t), _merged(group, n)
        _need(td), _need(nd)
        try:
            got = ur.retarget_graph(
                f"{td}/out/compiled.axmodel",
                f"{td}/model.onnx",
                ur.quant_of(td),
                ur.quant_of(nd),
            )
        except ValueError:
            continue
        ok += ur.compare(got, gs.load_model(f"{nd}/out/compiled.axmodel")) == []
    assert ok >= 1


def test_merged_linear_with_new_weights_equals_the_native_build():
    """lanorm2048 retargeted to another build's calibration AND weights
    (``default_rng(2)``, not the template's) is that build, npu_params included."""
    td, hd = _merged("lanorm2048", 0), f"{M}/heldout/lanorm2048_w2"
    template, want = _need(td), gs.load_model(_need(hd))
    w = onnx.numpy_helper.to_array(
        next(
            i for i in onnx.load(f"{hd}/model.onnx").graph.initializer if i.name == "w"
        )
    )
    got = ur.retarget_graph(
        template, f"{hd}/model.onnx", ur.quant_of(td), ur.quant_of(hd), weights=w
    )
    assert ur.compare(got, want) == []


def test_core_jobs_take_a_topological_order():
    """anorm1024 v0 calibrates m, e and r alike and u and d alike: the jobs
    after the second ReduceMean are still Add, Sqrt, Div in that order."""
    d = _merged("anorm1024", 0)
    template = _need(d)
    nodes = ur.graph_nodes(f"{d}/model.onnx")
    q = ur.propagate(onnx.load(f"{d}/model.onnx"), ur.quant_of(d))
    comp = [i for i, n in enumerate(nodes) if n[0] not in ur.PASS_THROUGH]
    main = ur._segs(gs.load_model(template))[gs.MAIN]
    got = ur._assign_core_jobs(main, comp, ur._graph_candidates(nodes, q), nodes)
    ops = [nodes[int(got[j][1:])][0] for j in sorted(got)]
    assert ops[-4:] == ["Add", "Sqrt", "Div", "Mul"]
