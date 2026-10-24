#!/usr/bin/env python3
"""Run TLC on every RosDiscoveryLease config and check the expected outcome.

The default and health-reflecting configs must hold every property. The
verify-off config with the current announce must violate one, which shows the
model still detects the known gap. Set TLA2TOOLS_JAR to the tla2tools.jar path.
"""

import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
MODULE = "RosDiscoveryLease.tla"
EXPECTED = {
    "RosDiscoveryLease.cfg": "pass",
    "NoVerifyReflect.cfg": "pass",
    "NoVerify.cfg": "violation",
}
PASS_MARKER = "Model checking completed. No error has been found."
VIOLATION_MARKERS = (
    "Invariant",
    "Temporal properties were violated",
    "Action property",
)


def outcome(output):
    if PASS_MARKER in output:
        return "pass"
    if any(marker in output for marker in VIOLATION_MARKERS):
        return "violation"
    return "error"


def main():
    jar = os.environ.get("TLA2TOOLS_JAR")
    if not jar:
        sys.exit("run_tlc: set TLA2TOOLS_JAR to the tla2tools.jar path")

    failures = []
    for config, expected in EXPECTED.items():
        proc = subprocess.run(
            ["java", "-XX:+UseParallelGC", "-cp", jar, "tlc2.TLC",
             "-workers", "auto", "-config", config, MODULE],
            cwd=HERE, capture_output=True, text=True,
        )
        output = proc.stdout + proc.stderr
        actual = outcome(output)
        status = "ok" if actual == expected else "MISMATCH"
        print(f"{status}: {config} expected {expected}, got {actual}")
        if actual != expected:
            failures.append(config)
            print(output)

    if failures:
        sys.exit(f"run_tlc: unexpected outcome for {', '.join(failures)}")


if __name__ == "__main__":
    main()
