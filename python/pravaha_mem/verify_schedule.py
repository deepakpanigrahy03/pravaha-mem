"""pravaha verify --schedule: the schedule runner checked on the GPU, under the L2 observer.

A short probe first measures this machine's rates; then a fixture schedule, each task
sized to last about slot_seconds with gap_s of idle between tasks (so no window catches
its neighbour), runs once while pravaha observes the whole GPU. Checks, kept
apart on purpose (as in verify --kernels):

1. Exact (no counter): every read checksum matches; payload bytes equal the work the
   schedule asked for.
2. Counter observation, background subtracted (idle slot rate x window, window widened by
   window_pad_ms on each side):
   streaming read   observed read miss bytes per payload byte within miss_tolerance of 1
                    ([verify] of pravaha.ini; footprint far larger than the 24 MiB L2)
   resident read    observed read miss bytes per payload byte at most resident_max_ratio
                    (footprint inside L2, many passes: the cache serves the reads)
   sham             observed memory bytes per second at most sham_max_bytes_per_s
                    (paced empty kernels move no data)
   copy             observed copy bytes per payload byte within copy_tolerance of 1
   square wave      every half within half_tolerance of period / 2; off halves move no payload
Settings: [schedule] and [verify] of pravaha.ini, then a file named by $PRAVAHA_CONFIG.
The thermal headroom guard of pravaha verify runs before the fixture.
Part of pravaha-mem. MIT License. Copyright (c) 2026 MANTRA Labs.
"""
from __future__ import annotations

import configparser
import json
import os
import subprocess
import sys
import tempfile

from .schedule import parse

KEYS_SCHEDULE = {"idle_s": float, "slot_seconds": float, "gap_s": float, "streaming_mib": int,
                 "resident_mib": int, "resident_max_ratio": float, "sham_pace_ms": float,
                 "sham_max_bytes_per_s": float, "copy_mib": int, "copy_tolerance": float,
                 "square_period_s": float, "square_periods": int, "square_pace_ms": float,
                 "half_tolerance": float}
KEYS_VERIFY = {"miss_tolerance": float, "window_pad_ms": float}


def settings(paths=None):
    """[schedule] and [verify] of pravaha.ini (shipped with the library), then $PRAVAHA_CONFIG."""
    if paths is None:
        from .thermal_limits import PACKAGE_CONFIG
        paths = [PACKAGE_CONFIG, os.environ.get("PRAVAHA_CONFIG")]
    cp = configparser.ConfigParser()
    cp.read([p for p in paths if p])
    out = {}
    for sec, keys in (("schedule", KEYS_SCHEDULE), ("verify", KEYS_VERIFY)):
        for k, typ in keys.items():
            if not cp.has_option(sec, k):
                raise KeyError("pravaha.ini [%s] %s missing" % (sec, k))
            out[k] = typ(cp.get(sec, k))
    return out


def probe_schedule(cfg):
    """A short schedule that measures this machine's rates for the three timed tasks."""
    return {"slots": [
        {"name": "probe_streaming", "kind": "kernel", "op": "read", "footprint_mib": cfg["streaming_mib"],
         "logical_mib": cfg["streaming_mib"] * 4},
        {"name": "probe_resident", "kind": "kernel", "op": "read", "footprint_mib": cfg["resident_mib"],
         "logical_mib": cfg["resident_mib"] * 512},
        {"name": "probe_copy", "kind": "copy", "direction": "h2d", "host": "pinned", "size_mib": cfg["copy_mib"],
         "count": 4},
    ]}


def counts(probe, cfg):
    """Operations per timed task so each lasts about slot_seconds, from the probe slots (at least 1)."""
    by = {s["name"]: s for s in probe["slots"]}

    def n(name):
        s = by[name]
        per_op = s["seconds"] / s["ops"] if s["ops"] and s["seconds"] > 0 else None
        return max(1, int(round(cfg["slot_seconds"] / per_op))) if per_op else 1
    return {"streaming_passes": n("probe_streaming"), "resident_passes": n("probe_resident"), "copy_count": n("probe_copy")}


def fixture(cfg, n):
    """The validation schedule: each task about slot_seconds long, gap_s of idle between tasks."""
    copy = {"direction": "h2d", "host": "pinned", "size_mib": cfg["copy_mib"]}
    gap = lambda i: {"name": "gap_%d" % i, "kind": "idle", "seconds": cfg["gap_s"]}  # noqa: E731
    return {"pause_after_setup_s": 1, "slots": [
        {"name": "idle", "kind": "idle", "seconds": cfg["idle_s"]}, gap(1),
        {"name": "read_streaming", "kind": "kernel", "op": "read", "footprint_mib": cfg["streaming_mib"],
         "logical_mib": cfg["streaming_mib"] * n["streaming_passes"]}, gap(2),
        {"name": "read_resident", "kind": "kernel", "op": "read", "footprint_mib": cfg["resident_mib"],
         "logical_mib": cfg["resident_mib"] * n["resident_passes"]}, gap(3),
        {"name": "sham", "kind": "noop", "pace_ms": cfg["sham_pace_ms"], "seconds": cfg["slot_seconds"]}, gap(4),
        dict(copy, name="copy_h2d", kind="copy", count=n["copy_count"]), gap(5),
        {"name": "square", "kind": "square", "period_s": cfg["square_period_s"], "periods": cfg["square_periods"],
         "on": dict(copy, kind="copy", pace_ms=cfg["square_pace_ms"]),
         "off": {"kind": "noop", "pace_ms": cfg["square_pace_ms"]}},
    ]}


def _within(ratio, tol):
    return None if ratio is None else abs(ratio - 1.0) <= tol


def evaluate(parsed, window_sum, cfg):
    """Checks from the parsed schedule output and window_sum(t0, t1) -> observer totals (bytes)."""
    pad = int(cfg["window_pad_ms"] * 1e6)
    slots = {s["name"]: s for s in parsed["slots"]}
    idle = slots["idle"]
    base = window_sum(idle["t0"], idle["t1"])
    rate = {k: base[k] / base["seconds"] for k in ("copy", "kernel_read", "memory")} if base["seconds"] else None

    def observed(name, field):
        s = slots[name]
        w = window_sum(s["t0"] - pad, s["t1"] + pad)
        return w[field] - (rate[field] * w["seconds"] if rate else 0.0), w["seconds"]

    checks = []
    for s in parsed["slots"]:
        if s.get("check") is not None:
            checks.append({"check": "checksum %s" % s["name"], "pass": bool(s["check"]["ok"]), "exact": True})
    for name, key in (("read_streaming", "streaming_mib"), ("read_resident", "resident_mib")):  # exact
        s = slots[name]
        want = s["ops"] * cfg[key] * (1 << 20)
        checks.append({"check": "payload %s" % name, "exact": True, "pass": s["ops"] > 0 and s["payload_bytes"] == want})
    obs, _ = observed("read_streaming", "kernel_read")
    r = obs / slots["read_streaming"]["payload_bytes"] if slots["read_streaming"]["payload_bytes"] else None
    checks.append({"check": "streaming read misses per payload byte", "value": r,
                   "tolerance": cfg["miss_tolerance"], "pass": _within(r, cfg["miss_tolerance"])})
    obs, _ = observed("read_resident", "kernel_read")
    r = obs / slots["read_resident"]["payload_bytes"] if slots["read_resident"]["payload_bytes"] else None
    checks.append({"check": "resident read misses per payload byte", "value": r,
                   "limit": cfg["resident_max_ratio"], "pass": None if r is None else r <= cfg["resident_max_ratio"]})
    obs, sec = observed("sham", "memory")
    v = obs / sec if sec else None
    checks.append({"check": "sham memory bytes per second", "value": v, "limit": cfg["sham_max_bytes_per_s"],
                   "pass": None if v is None else v <= cfg["sham_max_bytes_per_s"]})
    obs, _ = observed("copy_h2d", "copy")
    r = obs / slots["copy_h2d"]["payload_bytes"] if slots["copy_h2d"]["payload_bytes"] else None
    checks.append({"check": "copy bytes per payload byte", "value": r, "tolerance": cfg["copy_tolerance"],
                   "pass": _within(r, cfg["copy_tolerance"])})
    half = cfg["square_period_s"] / 2.0
    hs = parsed["halves"]
    worst = max(abs((h["t1"] - h["t0"]) / 1e9 - half) / half for h in hs) if hs else None
    checks.append({"check": "square wave half duration error", "value": worst, "limit": cfg["half_tolerance"],
                   "pass": None if worst is None else worst <= cfg["half_tolerance"]})
    off_payload = sum(h["payload_bytes"] for h in hs if h["half"] == "off")
    checks.append({"check": "square wave off halves move no payload", "exact": True, "pass": off_payload == 0 and bool(hs)})
    return {"checks": checks, "pass": all(c["pass"] is True for c in checks)}


def verify_schedule(json_path=None, cooldown_s=20.0, quiet=False):
    """Run the fixture under the observer; returns the report (and writes it as JSON if asked)."""
    from .observer import observe
    from .status import provenance
    from .verify import _wait_for_headroom
    cfg = settings()
    _wait_for_headroom(cooldown_s)
    probe = _run_file(probe_schedule(cfg))
    if probe["done"] is None:
        rep = {"pass": False, "error": "probe schedule failed: %s" % probe.get("stderr")}
        if json_path:
            with open(json_path, "w") as fh:
                json.dump(rep, fh, indent=1)
        print("schedule verify: FAIL (%s)" % rep["error"])
        return rep
    n = counts(probe, cfg)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(fixture(cfg, n), fh)
        path = fh.name
    with observe() as m:
        p = subprocess.run([sys.executable, "-m", "pravaha_mem.schedule", path], capture_output=True, text=True)
    os.unlink(path)
    parsed = parse(p.stdout.splitlines())
    if p.returncode not in (0, 3) or parsed["done"] is None:
        rep = {"pass": False, "error": "schedule exit %s: %s" % (p.returncode, (p.stderr or "")[-300:])}
    else:
        rep = evaluate(parsed, m._sum, cfg)
    rep.update({"settings": cfg, "sized_counts": n, "provenance": provenance(), "quality": m.quality()})
    if json_path:
        with open(json_path, "w") as fh:
            json.dump(rep, fh, indent=1)
    if not quiet:
        for c in rep.get("checks", []):
            val = "" if "value" not in c or c["value"] is None else " %.4g" % c["value"]
            print("%-6s %s%s" % ({True: "PASS", False: "FAIL", None: "N/A"}[c["pass"]], c["check"], val))
        print("schedule verify: %s" % ("PASS" if rep["pass"] else "FAIL"))
    return rep


def _run_file(sched):
    """Run a schedule in a child process; parsed lines plus stderr tail."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(sched, fh)
        path = fh.name
    try:
        p = subprocess.run([sys.executable, "-m", "pravaha_mem.schedule", path], capture_output=True, text=True)
    finally:
        os.unlink(path)
    return dict(parse(p.stdout.splitlines()), stderr=(p.stderr or "")[-300:])


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="verify_schedule", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=None)
    ap.add_argument("--cooldown", type=float, default=20.0)
    a = ap.parse_args(argv)
    return 0 if verify_schedule(a.json, a.cooldown)["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
