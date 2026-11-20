"""Overhead of the separate pravaha reader on a measured program (light A/B test).

What it does, in words:
  The same small GPU job (known_bytes: copy 32 x 256 MiB from one GPU buffer to another,
  8 GiB of memory traffic, about 0.1 s of GPU time) is timed many times, alternately
    A: with nothing observing the GPU
    B: with the pravaha reader running as a separate process, sampling every 10 ms.
  The job's own GPU time comes from known_bytes (WINDOW t0 t1, around cudaDeviceSynchronize).
  The result is the difference of the median times, set against the spread between runs.

Safety (the GPU is never stressed):
  memory copies only, no dense compute; a 10 s cool down between blocks;
  a block starts only with >= 35 C thermal headroom, and the test stops below 15 C.

Run (about 3 minutes):
  python validation/overhead_ab.py
Writes validation/overhead_ab_<hostname>.json (run from the repository folder).
"""
import json
import os
import platform
import statistics
import subprocess
import sys
import time

from pravaha_mem.observer import observe
from pravaha_mem.status import provenance, thermal

JOB = ["d2d", "32", "256"]          # 32 x 256 MiB = 8 GiB copied GPU buffer to GPU buffer
BLOCKS = 5                          # each block: A B B A A B B A (4 of each, order balanced)
ORDER = "ABBAABBA"
COOLDOWN_S = 10
START_MIN_C, ABORT_C = 35.0, 15.0


def headroom_ok():
    """Wait (up to 10 minutes) for enough thermal headroom; stop the test if it is too low."""
    deadline = time.time() + 600
    while True:
        h = thermal()["headroom_c"]
        if h is None or h >= START_MIN_C:
            return h
        if h < ABORT_C or time.time() > deadline:
            sys.exit(f"stopped: GPU thermal headroom {h:.0f} C is too low")
        print(f"  waiting for the GPU to cool (headroom {h:.0f} C)", flush=True)
        time.sleep(10)


def job_seconds():
    """Run the job once and return its GPU time in seconds, from known_bytes' own window."""
    env = dict(os.environ, PRAVAHA_KNOWN_WINDOW="1", PRAVAHA_KNOWN_PAUSE="0.3")
    out = subprocess.run([sys.executable, "-m", "pravaha_mem.known_bytes", *JOB],
                         capture_output=True, text=True, env=env)
    if out.returncode:
        sys.exit(out.stderr.strip() or "known_bytes failed")
    for line in out.stdout.splitlines():
        if line.startswith("WINDOW"):
            _, t0, t1 = line.split()
            return (int(t1) - int(t0)) / 1e9
    sys.exit("known_bytes printed no WINDOW line")


def main():
    times = {"A": [], "B": []}
    heads = []
    for b in range(BLOCKS):
        if b:
            time.sleep(COOLDOWN_S)
        heads.append(headroom_ok())
        print(f"block {b + 1} of {BLOCKS} (headroom {heads[-1]} C)", flush=True)
        for mode in ORDER:
            if mode == "B":
                o = observe(interval_ms=10).start()      # the separate reader process
                times["B"].append(job_seconds())
                o.stop()
                time.sleep(1.0)                          # one reader at a time: let it release the GPU
            else:
                times["A"].append(job_seconds())
    med = {m: statistics.median(v) for m, v in times.items()}
    spread = {m: (max(v) - min(v)) / statistics.median(v) * 100 for m, v in times.items()}
    diff = (med["B"] - med["A"]) / med["A"] * 100
    res = {"job": "known_bytes " + " ".join(JOB), "runs_each": len(times["A"]),
           "median_s": med, "spread_pct": spread, "diff_pct": diff,
           "times_s": times, "headroom_c_per_block": heads, "provenance": provenance(),
           "date": time.strftime("%Y-%m-%d %H:%M")}
    os.makedirs("validation", exist_ok=True)
    out = f"validation/overhead_ab_{platform.node()}.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nwithout reader: median {med['A'] * 1e3:.2f} ms (spread {spread['A']:.1f}%)")
    print(f"with reader:    median {med['B'] * 1e3:.2f} ms (spread {spread['B']:.1f}%)")
    print(f"difference:     {diff:+.2f}%   ({len(times['A'])} runs each)   saved {out}")


if __name__ == "__main__":
    main()
