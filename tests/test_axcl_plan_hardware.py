"""Hardware check of the runner's plan mode (AXSession.buffer / bind / run_plan):
a chain of two models that hands its tensor over through a shared device
buffer (at an offset) in one PLAN command equals two plain runs; a plain run
after BIND gets the model's own buffers back; unloading a bound model is safe.
Uses the committed health-check Relu fixture; skipped without an AXCL device."""

import gzip
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "axera"))

import axcl_session  # noqa: E402
import pulsar2_docker  # noqa: E402

pytestmark = pytest.mark.skipif(
    not pulsar2_docker.axcl_available(), reason="needs an AXCL device"
)


def _relu():
    with gzip.open(axcl_session.HEALTH_MODEL, "rb") as f:
        return f.read()


def test_plan_chain_equals_plain_runs():
    with axcl_session.AXSession(subdir="plan_mode_test") as s:
        a, b = s.load(_relu()), s.load(_relu())
        spec = a.inputs[0]
        x = np.random.default_rng(0).uniform(-1, 1, spec.shape).astype(np.float32)
        want = s.run(b, s.run(a, [x]))[0]

        n = spec.nbytes
        src, mid = s.buffer(n), s.buffer(2 * n)
        s.bind(a, "I", 0, src)
        s.bind(a, "O", 0, mid, n)  # a's output at an offset inside mid ...
        s.bind(b, "I", 0, mid, n)  # ... is b's input: no copy between them
        out = s.buffer(n)
        s.bind(b, "O", 0, out)
        plan = axcl_session.Plan(s, tag="t")
        plan.write(src, x).execute(a).execute(b)
        key = plan.read(out, n)
        execute_us, total_us = s.run_plan(plan)
        got = plan.result(key, np.float32, want.shape)
        assert np.array_equal(got, want)
        assert 0 < execute_us <= total_us

        # a plain run after BIND uses the model's own buffers again
        assert np.array_equal(s.run(a, [x])[0], s.run(b, [x])[0])
        with pytest.raises(axcl_session.DeviceError, match="past the buffer end"):
            s.bind(a, "I", 0, src, 4)
        s._cmd(f"UNLOAD {a.id}")
        s._cmd(f"UNLOAD {b.id}")
        for buf in (src, mid, out):
            s.free_buffer(buf)
        assert axcl_session.health_check(s) <= 1.0
