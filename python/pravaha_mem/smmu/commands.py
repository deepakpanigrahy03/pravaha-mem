"""pravaha smmu status | run | observe: GPU address translations (counts, never bytes)."""
from __future__ import annotations

import os
import subprocess
import sys
import time

from ..status import LIBEXEC
from . import discover

EVENTS = ("transaction", "tlb_miss")
MEANING = {"transaction": "translation requests", "tlb_miss": "translation misses"}
os.environ.setdefault("PRAVAHA_PMCG", os.path.join(LIBEXEC, "pravaha-pmcg"))


def _unit_name(x):
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return x.get("unit") or x.get("name") or next(iter(x.values()))
    if isinstance(x, (tuple, list)):
        return next((str(i) for i in x if str(i).startswith("smmuv3_pmcg")), str(x[-1]))
    return getattr(x, "unit", None) or getattr(x, "name", None) or str(x)


def _pmcg_names(x):
    """Every smmuv3_pmcg_* name inside whatever discovery returns (string, list, tuple, dict)."""
    if isinstance(x, str):
        return [x] if x.startswith("smmuv3_pmcg") else []
    if isinstance(x, dict):
        x = list(x.keys()) + list(x.values())
    if isinstance(x, (list, tuple, set)):
        return [n for i in x for n in _pmcg_names(i)]
    return _pmcg_names(str(x))


def default_unit():
    """The GPU's SMMU translation unit (TCU PMCG), as found by discovery."""
    names = _pmcg_names(discover.candidates())
    if not names:
        raise discover.DiscoveryError("no SMMU PMCG counter for the GPU found")
    return names[0]


def _open(unit):
    """Open counters: Linux perf directly if allowed, else the pravaha-pmcg reader (requests only)."""
    from .perf import Counter, PerfError
    try:
        return {e: Counter(unit, e) for e in EVENTS}, "perf"
    except (PerfError, PermissionError, OSError):
        from .helper import HelperCounter
        return {"transaction": HelperCounter(unit)}, "pravaha-pmcg"


def status() -> int:
    print("pravaha smmu status (address translations, not bytes)")
    try:
        gpus = discover.find_gpus()
        print(f"  GPUs              : {', '.join(map(str, gpus)) or 'none found'}")
        cands = discover.candidates()
        print(f"  GPU SMMU counters : {', '.join(_pmcg_names(cands)) or 'none'}")
        print(f"  default unit      : {default_unit()}")
        units = discover.pmcg_units()
        print(f"  all PMCG units    : {len(units)}")
    except discover.DiscoveryError as e:
        print(f"  {e}")
        return 1
    try:
        ctr, how = _open(default_unit())
        print(f"  readable via      : {how} ({', '.join(MEANING[e] for e in ctr)})")
        for c in ctr.values():
            c.close()
    except Exception as e:
        print(f"  not readable: {e} (run pravaha install)")
        return 1
    return 0


def run(cmd=None, seconds=0.0, live=False, unit=None, interval_ms=100) -> int:
    unit = unit or default_unit()
    ctr, how = _open(unit)
    say = lambda s: print(f"pravaha smmu: {s}", file=sys.stderr, flush=True)
    for c in ctr.values():
        c.start()
    start = {e: c.read() for e, c in ctr.items()}
    t0 = time.time()
    say(f"counting GPU address translations on {unit} via {how}"
        + (f" while '{' '.join(cmd)}' runs" if cmd else (f" for {seconds:g} s" if seconds else " until Ctrl+C")))
    if live:
        print("t_ns," + ",".join(ctr), flush=True)
    rc, p = 0, (subprocess.Popen(cmd) if cmd else None)
    last = dict(start)
    try:
        while True:
            if p is not None and p.poll() is not None:
                rc = p.returncode
                break
            if p is None and seconds and time.time() - t0 >= seconds:
                break
            time.sleep(interval_ms / 1000)
            if live:
                now = {e: c.read() for e, c in ctr.items()}
                print(f"{time.time_ns()}," + ",".join(str(now[e] - last[e]) for e in ctr), flush=True)
                last = now
    except KeyboardInterrupt:
        rc = 130
        if p is not None:
            p.wait()
    end = {e: c.read() for e, c in ctr.items()}
    dt = time.time() - t0
    for c in ctr.values():
        c.stop(); c.close()
    print(f"{'event':14}{'count':>14}{'per second':>14}   meaning")
    for e in ctr:
        n = end[e] - start[e]
        print(f"{e:14}{n:14d}{n / dt:14.0f}   {MEANING[e]}")
    print(f"window {dt:.2f} s, device wide. These are address translations, not bytes "
          f"(for bytes: pravaha run / observe).")
    return rc
