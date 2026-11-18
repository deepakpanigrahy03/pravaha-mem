"""pravaha verify: do the counters report the known bytes of light, built in workloads?

Each check: pravaha observes the whole GPU, measures one second of background, then
starts the demo (a separate process) which moves a known number of bytes and prints
when its work ran. The bytes in that window, minus the background, are compared with
what the demo moved. Each check runs 3 times; the middle value is reported.

Safety: the workloads are copies and simple writes and reads of a few GiB (about one
second of GPU work), never dense compute. Before each check the GPU's thermal headroom
is read; a check starts only with enough headroom, and checks are separated by a cool down.

PASS means: this counter agreed with this known workload within the stated tolerance,
on this device, driver and software. It is not a claim about every workload.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

from .observer import JSON_NAMES, METRICS, observe
from .status import provenance, thermal

MIB = 2 ** 20
MIN_HEADROOM_START = 35.0     # C, a check starts only above this
MIN_HEADROOM_ABORT = 15.0     # C, verification stops below this
# name, demo mode, copies, MiB, field, tolerance
CHECKS = [
    ("copy host to GPU", "h2d", 8, 256, "copy", 0.002),
    ("copy GPU to host", "d2h", 8, 256, "copy", 0.002),
    ("kernel write (cudaMemset)", "memset", 16, 256, "kernel_write", 0.01),
    ("kernel read (GPU to GPU copy)", "d2d", 16, 256, "kernel_read", 0.02),
]


def _demo(mode, n, mib):
    out = subprocess.run([sys.executable, "-m", "pravaha_mem.demo", mode, str(n), str(mib)],
                         capture_output=True, text=True, env={**os.environ, "PRAVAHA_DEMO_WINDOW": "1"})
    if out.returncode:
        raise RuntimeError(out.stderr.strip() or "demo failed")
    w = [l.split() for l in out.stdout.splitlines() if l.startswith("WINDOW")][0]
    return int(w[1]), int(w[2])


def _wait_for_headroom(cooldown_s):
    """Cool down, then wait (up to 10 minutes) until headroom allows a check. Returns headroom."""
    time.sleep(cooldown_s)
    deadline = time.time() + 600
    while True:
        h = thermal()["headroom_c"]
        if h is None or h >= MIN_HEADROOM_START:
            return h
        if h < MIN_HEADROOM_ABORT or time.time() > deadline:
            raise RuntimeError(f"GPU thermal headroom {h:.0f} C is too low: verification stopped to protect the machine")
        print(f"  waiting for the GPU to cool (headroom {h:.0f} C, need {MIN_HEADROOM_START:.0f} C)", flush=True)
        time.sleep(15)


def verify(json_path: str | None = None, cooldown_s: float = 30.0, quiet: bool = False) -> dict:
    say = (lambda s: None) if quiet else (lambda s: print(s, flush=True))
    say("pravaha verify: do the counters report the bytes of known, light workloads? (about 3 minutes)")
    report = {"provenance": provenance(), "meaning": __doc__.split("PASS means:")[1].strip(),
              "metrics": {JSON_NAMES[k]: {"counter": c, "conversion": v} for k, (c, v) in METRICS.items()},
              "checks": []}
    h = _wait_for_headroom(0)
    with observe() as m:
        time.sleep(2)
    idle = m.total()
    report["checks"].append({"check": "idle machine: no copies", "field": "copy_bytes",
                             "measured_bytes": idle["copy"], "expected_bytes": 0,
                             "tolerance_bytes": MIB, "pass": idle["copy"] < MIB,
                             "headroom_before_c": h})
    for i, (name, mode, n, mib, field, tol) in enumerate(CHECKS):
        h0 = _wait_for_headroom(cooldown_s if i else 1.0)
        exp = n * mib * MIB
        runs = []
        for _ in range(3):   # the middle of 3 runs: a burst from other programs cannot decide the result
            with observe() as m:
                b0 = time.time_ns(); time.sleep(1.0); b1 = time.time_ns()
                t0, t1 = _demo(mode, n, mib)
            a, b = t0 - 40_000_000, t1 + 40_000_000
            q = m._sum(b0, b1)
            bg = q[field] / q["seconds"] * (b - a) / 1e9 if q["seconds"] else 0.0
            raw = m._sum(a, b)[field]
            runs.append({"raw_bytes": raw, "background_bytes": bg, "net_bytes": raw - bg})
        runs.sort(key=lambda r: r["net_bytes"])
        mid = runs[1]
        err = mid["net_bytes"] / exp - 1
        report["checks"].append({
            "check": name, "workload": f"demo {mode} {n} x {mib} MiB", "field": JSON_NAMES[field],
            "counter": METRICS[field][0], "expected_bytes": exp, "measured_bytes": mid["net_bytes"],
            "ratio": mid["net_bytes"] / exp, "relative_error": err, "tolerance": tol,
            "pass": abs(err) <= tol, "runs": runs, "headroom_before_c": h0, "headroom_after_c": thermal()["headroom_c"]})
    report["all_pass"] = all(c["pass"] for c in report["checks"])
    for c in report["checks"]:
        ratio = f"  ratio {c['ratio']:.4f}" if "ratio" in c else ""
        say(f"  {'PASS' if c['pass'] else 'FAIL'}  {c['check']:32} {c['measured_bytes'] / MIB:10.1f} MiB"
            f"  expected {c['expected_bytes'] / MIB:8.1f}{ratio}")
    say("all checks passed on this machine (" + report["provenance"]["gpu"] + ", driver "
        + str(report["provenance"]["driver"]) + ")" if report["all_pass"] else
        "some checks failed. pravaha sees the whole GPU, so other programs add traffic; run again when the GPU "
        "is quieter. If it still fails, please report the JSON (--json FILE).")
    if json_path:
        with open(json_path, "w") as fh:
            json.dump(report, fh, indent=1)
        say(f"report: {json_path}")
    return report
