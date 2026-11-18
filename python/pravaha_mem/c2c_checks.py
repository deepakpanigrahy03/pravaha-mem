"""pravaha test: check the live C2C counters against programs that move known bytes.

Each check starts a separate worker process (the CUDA runtime through ctypes, so
nothing is compiled) while pravaha watches the whole GPU from outside. pravaha first
measures one second of the machine's background (the desktop and other programs),
then starts the worker, which prints when its work ran. pravaha sums the samples in
that window, subtracts the background, and compares with the bytes the worker moved.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time

from .c2c_sampler import c2c

MIB = 2 ** 20

# name, worker mode, copies, MiB, measured field, tolerance (relative)
CHECKS = [
    ("copy host to GPU", "h2d", 8, 256, "copy", 0.002),
    ("copy GPU to host", "d2h", 8, 256, "copy", 0.002),
    ("kernel write (cudaMemset)", "memset", 16, 256, "kernel_write", 0.01),
    ("kernel read and write (GPU to GPU copy)", "d2d", 16, 256, "kernel_read", 0.02),
]


def _worker(mode, n, mib):
    import os
    out = subprocess.run([sys.executable, "-m", "pravaha_mem.demo", mode, str(n), str(mib)],
                         capture_output=True, text=True, env={**os.environ, "PRAVAHA_DEMO_WINDOW": "1"})
    if out.returncode:
        raise RuntimeError(out.stderr.strip() or "demo failed")
    w = [l.split() for l in out.stdout.splitlines() if l.startswith("WINDOW")][0]
    return int(w[1]), int(w[2])


def main(json_path: str | None = None) -> int:
    results, ok = [], True
    print("pravaha test: does pravaha report exactly the bytes a test program moves? (about 1 minute)")
    with c2c() as m:
        time.sleep(2)
    idle = m.total()
    r = {"check": "idle machine: no copies", "measured_mib": idle["copy"] / MIB, "expected_mib": 0,
         "pass": idle["copy"] < 1 * MIB}
    results.append(r)
    for name, mode, n, mib, field, tol in CHECKS:
        exp = n * mib * MIB
        tries = []
        for _ in range(3):   # the middle of 3 runs: a burst from the desktop cannot decide the result
            with c2c() as m:
                b0 = time.time_ns(); time.sleep(1.0); b1 = time.time_ns()   # background first
                t0, t1 = _worker(mode, n, mib)
            a, b = t0 - 40_000_000, t1 + 40_000_000   # covers the edge samples, little background
            q = m._sum(b0, b1)
            bg = q[field] / q["seconds"] * (b - a) / 1e9 if q["seconds"] else 0.0
            tries.append((m._sum(a, b)[field] - bg, bg))
        tries.sort()
        got, bg = tries[1]
        results.append({"check": name, "measured_mib": got / MIB, "expected_mib": exp / MIB,
                        "ratio": got / exp, "pass": abs(got / exp - 1) <= tol, "tolerance": tol,
                        "runs_mib": [round(t[0] / MIB, 1) for t in tries], "background_mib": bg / MIB})
    for r in results:
        ok &= r["pass"]
        extra = f"  ratio {r['ratio']:.4f}  (middle of 3 runs)" if "ratio" in r else ""
        print(f"  {'PASS' if r['pass'] else 'FAIL'}  {r['check']:42} {r['measured_mib']:10.1f} MiB"
              f"  expected {r['expected_mib']:8.1f}{extra}")
    print("all checks passed: pravaha measures this GPU correctly" if ok else
          "some checks failed. pravaha sees the whole GPU, so busy windows (videos, browsers) add traffic.\n"
          "Close them and run again; if it still fails, please report the output.")
    if json_path:
        with open(json_path, "w") as fh:
            json.dump(results, fh, indent=1)
    return 0 if ok else 1
