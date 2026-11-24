"""pravaha verify --kernels: known kernel workloads in each memory placement.

Two kinds of check, kept apart on purpose:

1. Logical bytes (exact). known_kernel reads or writes TIMES x SIZE in each supported
   placement and checks a checksum outside its work window. PASS = the program did exactly
   the work it reports. This needs no counter.

2. Counter observation. pravaha observes each run. For a small buffer read repeatedly the
   cache serves most reads, so the bytes that reach memory are far below the logical bytes:
   reported as observed bytes per logical byte, never judged against 1.0. Only the miss
   fixture (a buffer much larger than the GPU's L2, device placement, few passes) is
   expected to send every logical byte to memory; it is judged with a tolerance.

Each run lasts about [verify] kernel_seconds of GPU work (pass count sized from a short probe),
so it spans many 10 ms samples; the window is widened by window_pad_ms on each side.
Background: one second before each run, scaled to the window (as in pravaha verify).
Settings: [verify] in pravaha.ini, shipped with the library.
Part of pravaha-mem. MIT License. Copyright (c) 2026 MANTRA Labs.
"""
from __future__ import annotations

import configparser
import json
import math
import os
import time

from . import known_kernel
from .observer import observe
from .status import provenance, thermal
from .verify import _wait_for_headroom

FIELD = {"read": "kernel_read", "write": "kernel_write"}


def settings():
    """[verify] of pravaha.ini (shipped with the library), then a file named by $PRAVAHA_CONFIG."""
    from .thermal_limits import PACKAGE_CONFIG
    cp = configparser.ConfigParser()
    cp.read([p for p in (PACKAGE_CONFIG, os.environ.get("PRAVAHA_CONFIG")) if p])
    v = cp["verify"]
    return {"kernel_seconds": v.getfloat("kernel_seconds"), "window_pad_ms": v.getfloat("window_pad_ms"),
            "small_mib": v.getint("small_mib"), "miss_mib": v.getint("miss_mib"),
            "miss_tolerance": v.getfloat("miss_tolerance")}


def _times_for(placement, op, mib, prefetch, seconds):
    """Pass count giving about `seconds` of work, from a one pass probe (repeated pass time)."""
    r = known_kernel.run(placement, op, 2, mib, prefetch=prefetch, timeout=600)
    if r["returncode"] != 0 or r["t0"] is None or r["first_t1"] is None:
        raise RuntimeError("probe failed: %s" % r["stderr_tail"])
    per_pass = max((r["t1"] - r["first_t1"]) / 1e9, 1e-6)
    return max(1, math.ceil(seconds / per_pass))


def _observed(placement, op, mib, prefetch, cfg):
    times = _times_for(placement, op, mib, prefetch, cfg["kernel_seconds"])
    pad = int(cfg["window_pad_ms"] * 1e6)
    with observe() as m:
        b0 = time.time_ns(); time.sleep(1.0); b1 = time.time_ns()
        r = known_kernel.run(placement, op, times, mib, prefetch=prefetch, timeout=600)
    if r["returncode"] != 0 or r["t0"] is None:
        return {"error": "exit %s: %s" % (r["returncode"], r["stderr_tail"]), "check": r.get("check"), "times": times}
    q = m._sum(b0, b1)
    a, b = r["t0"] - pad, r["t1"] + pad
    bg = q[FIELD[op]] / q["seconds"] * (b - a) / 1e9 if q["seconds"] else None
    raw = m._sum(a, b)[FIELD[op]]
    net = raw - bg if bg is not None else None
    logical = r["check"]["logical_bytes"]
    return {"check": r["check"], "times": times, "raw_bytes": raw, "background_bytes": bg, "observed_bytes": net,
            "observed_per_logical": (net / logical) if net is not None and logical else None,
            "first_pass_s": (r["first_t1"] - a) / 1e9 if r["first_t1"] else None,
            "window_s": (b - a) / 1e9, "attrs_before": r["attrs_before"], "attrs_after": r["attrs_after"]}


def verify_kernels(json_path: str | None = None, cooldown_s: float = 20.0, quiet: bool = False) -> dict:
    say = (lambda s: None) if quiet else (lambda s: print(s, flush=True))
    q, cfg = known_kernel.query(), settings()
    report = {"provenance": provenance(), "attributes": q["attributes"], "supported": q["supported"], "settings": cfg,
              "checks": []}
    conds = [(pl, None) for pl in ("device", "pinned", "pageable")] + [("managed", "none"), ("managed", "device")]
    i = 0
    for pl, pf in conds:
        if not q["supported"][pl]["ok"]:
            report["checks"].append({"placement": pl, "prefetch": pf, "skipped": q["supported"][pl]["reason"]})
            continue
        for op in ("read", "write"):
            h = _wait_for_headroom(cooldown_s if i else 1.0); i += 1
            say("  known_kernel %s %s %d MiB for about %g s%s ..." % (pl, op, cfg["small_mib"], cfg["kernel_seconds"],
                                                                   " prefetch " + pf if pf else ""))
            r = _observed(pl, op, cfg["small_mib"], pf, cfg)
            r.update(placement=pl, prefetch=pf or "not_applicable", op=op, kind="logical",
                     pass_=bool(r.get("check") and r["check"]["ok"]), headroom_before_c=h)
            report["checks"].append(r)
    if q["supported"]["device"]["ok"]:
        _wait_for_headroom(cooldown_s)
        say("  miss fixture: device read %d MiB (far larger than L2) for about %g s ..."
            % (cfg["miss_mib"], cfg["kernel_seconds"]))
        r = _observed("device", "read", cfg["miss_mib"], None, cfg)
        tol = cfg["miss_tolerance"]
        ok = r.get("observed_per_logical") is not None and abs(r["observed_per_logical"] - 1) <= tol
        r.update(placement="device", prefetch="not_applicable", op="read", kind="miss_fixture",
                 tolerance=tol, pass_=bool(r.get("check") and r["check"]["ok"] and ok))
        report["checks"].append(r)
    report["all_pass"] = all(c.get("pass_", True) for c in report["checks"])
    for c in report["checks"]:
        if "skipped" in c:
            say("  SKIP  %-8s %s" % (c["placement"], c["skipped"]))
            continue
        opl = c.get("observed_per_logical")
        say("  %s  %-8s %-14s %-5s %-12s observed/logical %s" % (
            "PASS" if c["pass_"] else "FAIL", c["placement"], c["prefetch"], c["op"], c["kind"],
            "%.4f" % opl if opl is not None else "n/a"))
    if json_path:
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        with open(json_path, "w") as fh:
            json.dump(report, fh, indent=1, default=str)
        say("report: %s" % json_path)
    return report
