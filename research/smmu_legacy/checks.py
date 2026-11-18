"""The validation suite behind `pravaha test`: the letter's checks, on your machine.

Author: Deepak Panigrahy
Part of pravaha (GB10 GPU memory-traffic measurement). License: MIT.

Each check gives the counters work whose bytes we know, and compares. Every
check returns a dict:

    {"check": name, "plan": experiment id in the letter's plan,
     "status": "PASS" | "FAIL" | "INFO" | "SKIP",
     "rule": the pass rule in words, "values": {...}, "reason": why, if SKIP}

INFO means the check reports a fact with no pass rule (for example, how the
copy counter reacts to CPU-only traffic). SKIP always says why (missing
permission, missing program, no calibration); nothing fails silently.

Test programs come from the repository's build/ folder (make build), or from
$PRAVAHA_BUILD.
"""

import os
import re
import subprocess
import time

from . import ncu
from .calib import Calibration, CalibrationMissing
from .cuda import Buffers
from .measure import default_unit
from .perf import Counter
from .status import ncu_permitted

MIB = 2 ** 20
SECTORS_256 = (256 * MIB) // 32          # one 256 MiB round = 8,388,608 sectors


# ---- helpers ------------------------------------------------------------------
def _build(name):
    """Path of a test program, or None if it is not built."""
    roots = [os.environ.get("PRAVAHA_BUILD"), os.path.join(os.getcwd(), "build"),
             os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "build")]
    for root in roots:
        if root:
            path = os.path.join(root, name)
            if os.access(path, os.X_OK):
                return path
    return None


def _result(check, plan, status, rule, values=None, reason=None):
    return {"check": check, "plan": plan, "status": status, "rule": rule,
            "values": values or {}, "reason": reason}


def _bg(unit, windows=3, seconds=0.5):
    """Background rate measured NOW (median of short windows, so one burst cannot dominate)."""
    c = Counter(unit)
    rates = []
    for _ in range(windows):
        c.start()
        time.sleep(seconds)
        c.stop()
        rates.append(c.read() / seconds)
    c.close()
    return sorted(rates)[len(rates) // 2]


def _count_process(unit, argv):
    """Raw transactions while a program runs, and its duration in seconds."""
    c = Counter(unit)
    c.start()
    t0 = time.monotonic()
    subprocess.run(argv, check=True, stdout=subprocess.DEVNULL)
    dt = time.monotonic() - t0
    c.stop()
    n = c.read()
    c.close()
    return n, dt


def _count_copies(unit, buf, direction, copies, idle_rate):
    """Background-subtracted transactions for `copies` in-process copies."""
    c = Counter(unit)
    buf.copy(direction)                      # warm-up, not counted
    buf.sync()
    c.start()
    t0 = time.monotonic()
    for _ in range(copies):
        buf.copy(direction)
    buf.sync()
    dt = time.monotonic() - t0
    c.stop()
    net = c.read() - idle_rate * dt
    c.close()
    return net


def _linfit(xs, ys):
    """Least-squares slope and R^2."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    ss_res = sum((y - (my + b * (x - mx))) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return b, 1 - ss_res / ss_tot


# ---- the checks ------------------------------------------------------------------
def check_calibration():
    """Is there a calibration for this machine, is it current, which sizes are unstable?"""
    rule = "calibration present and not stale"
    try:
        cal = Calibration.load()
    except CalibrationMissing as e:
        return _result("calibration", "E5", "FAIL", rule, reason=str(e))
    stale = cal.stale()
    values = {"calibration_id": cal.calibration_id, "stale_fields": stale,
              "unstable_sizes": cal.unstable_sizes(), "range": cal.bounds(),
              "background_rate": cal.idle_rate, "background_sd": cal.idle_rate_sd}
    return _result("calibration", "E5", "FAIL" if stale else "PASS", rule, values)


def check_linearity(unit, cal, sizes=(4, 8, 16), repeats=5):
    """Do transactions grow in a straight line with bytes copied, in both directions?

    Each size is copied `repeats` times and the median is fitted, so one copy disturbed
    by another program using the GPU cannot bend the line. Copies more than 5% from
    their size's median are reported by name.
    """
    rule = "R^2 >= 0.999 per direction on the median of each size; slopes within 2%"
    buf = Buffers(256 * MIB)
    slopes, r2s, points, outliers = {}, {}, {}, []
    for d in ("h2d", "d2h"):
        xs, ys = [], []
        points[d] = {}
        for n in sizes:
            runs = [_count_copies(unit, buf, d, n, _bg(unit)) for _ in range(repeats)]
            med = sorted(runs)[len(runs) // 2]
            points[d][str(n)] = runs
            xs.append(n * 256 * MIB)
            ys.append(med)
            for i, r in enumerate(runs):
                dev = (r / med - 1) * 100 if med else 0.0
                if abs(dev) > 5:
                    outliers.append({"direction": d, "copies": n, "repeat": i + 1,
                                     "deviation_pct": round(dev, 1)})
        slopes[d], r2s[d] = _linfit(xs, ys)
    buf.free()
    diff = abs(slopes["h2d"] - slopes["d2h"]) / max(slopes.values()) * 100
    ok = min(r2s.values()) >= 0.999 and diff <= 2.0
    reason = None
    if outliers:
        reason = (f"{len(outliers)} of {2 * len(sizes) * repeats} copies were more than 5% from "
                  "their median; another program probably used the GPU (a browser or the desktop). "
                  "Close it and run the check again.")
    return _result("linearity", "E4", "PASS" if ok else "FAIL", rule,
                   {"r2": r2s, "bytes_per_txn": {d: 1 / s for d, s in slopes.items()},
                    "direction_diff_pct": diff, "points_txn": points, "outliers": outliers},
                   reason=reason)


def check_copy_blind_to_kernels(unit, cal):
    """Does the copy counter stay quiet while kernels read memory directly?"""
    rule = "kernel reads add under 10% of what the same bytes give as copies"
    zc = _build("zc")
    if not zc:
        return _result("copy_blind_to_kernels", "E6", "SKIP", rule, reason="build/zc missing: make build")
    values = {}
    expected = 8 * 256 * MIB / cal.bytes_per_txn(256, "h2d")
    worst = 0.0
    for mode in ("pinned", "malloc", "device"):
        deltas = []
        for _ in range(3):                                        # median of 3 pairs
            b0 = _bg(unit)
            n0, t0 = _count_process(unit, [zc, mode, "0", "r"])  # start-up only
            b8 = _bg(unit)
            n8, t8 = _count_process(unit, [zc, mode, "8", "r"])  # start-up + 8 rounds
            deltas.append((n8 - b8 * t8) - (n0 - b0 * t0))
        delta = sorted(deltas)[1]
        values[mode] = {"added_txn": delta, "pct_of_copy_expectation": delta / expected * 100}
        worst = max(worst, delta / expected * 100)
    values["copy_expectation_txn"] = expected
    return _result("copy_blind_to_kernels", "E6", "PASS" if worst < 10 else "FAIL", rule, values)


def check_kernels(modes=("pinned", "malloc", "device", "managed", "managed+prefetch")):
    """Do the L2 events match known kernel reads and writes in each memory type?

    managed = cudaMallocManaged filled by the CPU; managed+prefetch = the same,
    moved to the GPU first (N2 records both residencies). Managed deviations are
    reported as a limit, not a failure, as the plan states.
    """
    rule = ("reads, and pinned/device writes, within 1% of 8,388,608 sectors; "
            "managed and pageable deviations reported as limits")
    zc = _build("zc")
    if not zc:
        return _result("kernels", "E7/N2", "SKIP", rule, reason="build/zc missing: make build")
    if not ncu_permitted():
        return _result("kernels", "E7/N2", "SKIP", rule, reason="GPU counters admin-only: run with sudo")
    values, ok, limits = {}, True, []
    for mode in modes:
        base, prefetch = mode.split("+")[0], mode.endswith("+prefetch")
        for op in ("r", "w"):
            argv = [zc, base, "1", op] + (["prefetch"] if prefetch else [])
            k = ncu.kernel_traffic(argv)
            sectors = (k.read_bytes if op == "r" else k.write_bytes) / 32
            err = (sectors / SECTORS_256 - 1) * 100
            values[f"{mode}_{op}"] = {"sectors": sectors, "error_pct": err,
                                      "extra_read_sectors": k.read_bytes / 32 if op == "w" else None}
            strict = base in ("pinned", "device") or (op == "r" and base == "malloc")
            if abs(err) > 1.0:
                if strict:
                    ok = False
                else:
                    limits.append(f"{mode} {op}: {err:+.2f}%")
    values["limits"] = limits
    return _result("kernels", "E7/N2", "PASS" if ok else "FAIL", rule, values)


def _mixed_copy_pass(unit, cal, argv):
    """One run of build/mixed with the copy counter around the work only."""
    bg = _bg(unit)
    p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    if p.stdout.readline().strip() != "READY":
        raise RuntimeError("mixed did not report READY")
    c = Counter(unit)
    c.start()
    t0 = time.monotonic()
    p.stdin.write("\n"); p.stdin.flush()                     # go
    done = p.stdout.readline().strip()                       # waits for the work to finish
    dt = time.monotonic() - t0
    c.stop()
    raw = c.read()
    c.close()
    p.stdin.write("\n"); p.stdin.flush()
    p.wait()
    known = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", done)}
    return raw - bg * dt, known


def check_mixed(unit, cal, copy_mib=256, copies=8, rounds=4, repeats=5, kernel_pass=True):
    """Do the copy counter and the L2 events together account for known traffic?

    The two paths come from SEPARATE runs of the same program (copy counter live,
    L2 events in an Nsight Compute pass): accounting across repeated runs, not
    one simultaneous measurement.
    """
    rule = "copy path, kernel path and combined total each within 5% of known bytes"
    exe = _build("mixed")
    if not exe:
        return _result("mixed", "N1", "SKIP", rule, reason="build/mixed missing: make build")
    argv = [exe, str(copy_mib), str(copies), str(rounds)]
    profile = {(copy_mib, "h2d", False): copies, (copy_mib, "d2h", False): copies}
    copy_errs, known = [], {}
    for _ in range(repeats):
        net, known = _mixed_copy_pass(unit, cal, argv)
        est = cal.estimate(net, profile).bytes
        copy_errs.append((est / known["copy_bytes"] - 1) * 100)
    values = {"copy_error_pct": copy_errs, "known": known, "passes": "separate"}
    ok = max(abs(e) for e in copy_errs) <= 5
    if not kernel_pass or not ncu_permitted():
        values["kernel_pass"] = "skipped: " + ("not requested" if not kernel_pass else "no GPU counter permission")
        return _result("mixed", "N1", "PASS" if ok else "FAIL", rule + " (copy path only)", values)
    k = ncu.kernel_traffic(argv, stdin_text="\n\n")
    r_err = (k.read_bytes / known["read_bytes"] - 1) * 100
    w_err = (k.write_bytes / known["write_bytes"] - 1) * 100
    mean_copy = known["copy_bytes"] * (1 + sum(copy_errs) / len(copy_errs) / 100)
    total_known = known["copy_bytes"] + known["read_bytes"] + known["write_bytes"]
    total_err = ((mean_copy + k.read_bytes + k.write_bytes) / total_known - 1) * 100
    values.update({"kernel_read_error_pct": r_err, "kernel_write_error_pct": w_err,
                   "combined_error_pct": total_err})
    ok = ok and abs(r_err) <= 5 and abs(w_err) <= 5 and abs(total_err) <= 5
    return _result("mixed", "N1", "PASS" if ok else "FAIL", rule, values)


def check_cpu_only(unit, cal, repeats=5):
    """How does the copy counter react to memory traffic made only by the CPU?"""
    rule = "information: background-subtracted increment per run vs the CPU's known bytes"
    exe = _build("cpu_stream")
    if not exe:
        return _result("cpu_only", "N4", "SKIP", rule, reason="build/cpu_stream missing: make build")
    incs = []
    for _ in range(repeats):
        bg = _bg(unit)
        n, dt = _count_process(unit, [exe, "1024", "8"])
        incs.append(n - bg * dt)
    return _result("cpu_only", "N4", "INFO", rule,
                   {"increments_txn": incs, "cpu_bytes_moved": 2 * 1024 * MIB * 8,
                    "supports": "the counter does not capture the tested CPU-only workload"
                                if max(incs) < 0.01 * 2 * 1024 * MIB * 8 / cal.bounds()[1] else
                                "the counter rose during CPU-only traffic: see values"})


def check_directions(unit, cal, copies=8):
    """Host-to-device, device-to-host and device-to-device copies, reported separately."""
    rule = "information: bytes per transaction for each direction; device-to-device may not pass the SMMU"
    buf = Buffers(256 * MIB)
    values = {}
    for d in ("h2d", "d2h", "d2d"):
        net = _count_copies(unit, buf, d, copies, _bg(unit))
        values[d] = {"net_txn": net,
                     "bytes_per_txn": copies * 256 * MIB / net if net > 0 else None}
    buf.free()
    return _result("directions", "N5", "INFO", rule, values)


def check_async(unit, cal, copies=8, repeats=5):
    """Does the calibration hold for asynchronous copies on two streams?"""
    rule = "median async bytes per transaction (5 repeats) within the E5 spread of the calibration at 256 MiB"
    buf = Buffers(256 * MIB)
    buf.copy_async("h2d", 1)                 # warm-up, not counted
    buf.sync()
    runs = []
    for _ in range(repeats):                 # measured like the calibration: 5 repeats, median
        bg = _bg(unit)                       # background just before this repeat
        c = Counter(unit)
        c.start()
        t0 = time.monotonic()
        buf.copy_async("h2d", copies)
        dt = time.monotonic() - t0
        c.stop()
        runs.append(copies * 256 * MIB / (c.read() - bg * dt))
        c.close()
    buf.free()
    bpt = sorted(runs)[len(runs) // 2]
    sync_ref = cal.bytes_per_txn(256, "h2d")
    has_async = ("async", "h2d") in cal.table   # calibrations from fix 4 on carry async entries
    mode = "async" if has_async else False
    ref = cal.bytes_per_txn(256, "h2d", mode)
    dev = abs(bpt / ref - 1) * 100
    tol = cal.spread_pct(256, "h2d", mode)   # the plan's rule: within the E5 spread, same size and kind
    return _result("async", "N6", "PASS" if dev <= tol else "FAIL", rule,
                   {"bytes_per_txn": bpt, "runs": runs,
                    "run_spread_pct": (max(runs) - min(runs)) / min(runs) * 100,
                    "calibration": ref, "compared_with": "async entry" if has_async else "sync entry",
                    "gap_vs_sync_pct": (bpt / sync_ref - 1) * 100,
                    "deviation_pct": dev, "tolerance_pct": tol})


def check_repeatability():
    """N3: do the size table and the unstable sizes recur across independent campaigns?

    A campaign = one `pravaha setup`. The plan asks for three, each after a reboot;
    campaigns from the same boot are counted once (the latest).
    """
    rule = "information: spread per size across >= 3 campaigns from different boots; recurring instability"
    by_boot = {}
    for c in Calibration.history():
        by_boot[c.data.get("boot_id") or c.calibration_id] = c   # latest per boot wins
    camps = list(by_boot.values())
    if len(camps) < 3:
        return _result("repeatability", "N3", "SKIP", rule,
                       reason=f"{len(camps)} campaign(s) from different boots; reboot and run 'pravaha setup' until 3")
    sizes = {}
    for c in camps:
        for e in c.data["entries"]:
            sizes.setdefault((e["host"], e["direction"], e["mib"]), []).append(e)
    table = []
    for (host, d, mib), es in sorted(sizes.items()):
        v = [e["bytes_per_txn"] for e in es]
        n_unstable = sum(bool(e.get("unstable")) for e in es)
        table.append({"host": host, "direction": d, "mib": mib, "campaigns": len(v),
                      "median_per_campaign": v, "spread_pct": (max(v) - min(v)) / min(v) * 100,
                      "unstable_in": n_unstable, "recurring_instability": n_unstable >= 2})
    return _result("repeatability", "N3", "INFO", rule,
                   {"campaigns": [c.calibration_id for c in camps], "sizes": table,
                    "recurring_unstable": [(t["host"], t["direction"], t["mib"])
                                           for t in table if t["recurring_instability"]]})


def check_pattern():
    """N7: what do the L2 events count when reads do not fill whole sectors?"""
    rule = "information: measured sectors vs expected UNIQUE sectors (never vs logical bytes)"
    exe = _build("pattern")
    if not exe:
        return _result("pattern", "N7", "SKIP", rule, reason="build/pattern missing: make build")
    if not ncu_permitted():
        return _result("pattern", "N7", "SKIP", rule, reason="GPU counters admin-only: run with sudo")
    values = {}
    for name, args in (("seq", ["seq"]), ("stride_8B", ["stride", "256", "1", "8"]),
                       ("stride_32B", ["stride", "256", "1", "32"]),
                       ("stride_128B", ["stride", "256", "1", "128"]), ("random", ["random"])):
        p = subprocess.run([exe] + args, capture_output=True, text=True)
        known = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", p.stdout)}
        k = ncu.kernel_traffic([exe] + args)
        measured = k.read_bytes / 32
        values[name] = {"logical_bytes": known.get("logical_bytes"),
                        "expected_sectors": known.get("expected_sectors"), "measured_sectors": measured,
                        "ratio_to_expected": measured / known["expected_sectors"] if known.get("expected_sectors") else None}
    return _result("pattern", "N7", "INFO", rule, values)


def check_warm_cold(mib=16, rounds=3):
    """N8 (artifact): kernel traffic with caches flushed per replay vs kept warm.

    cold = --cache-control all; warm = --cache-control none, where the first round
    primes a buffer smaller than the 24 MiB L2. Verify the option's meaning for your
    Nsight Compute version: the result records the version string.
    """
    rule = "information: sectors per round, cold vs warm (difference reported)"
    exe = _build("pattern")
    if not exe:
        return _result("warm_cold", "N8", "SKIP", rule, reason="build/pattern missing: make build")
    if not ncu_permitted():
        return _result("warm_cold", "N8", "SKIP", rule, reason="GPU counters admin-only: run with sudo")
    argv = [exe, "seq", str(mib), str(rounds)]
    cold = [k["read_sectors"] for k in ncu.kernel_traffic(argv, cache_control="all").kernels]
    warm = [k["read_sectors"] for k in ncu.kernel_traffic(argv, cache_control="none").kernels]
    ver = subprocess.run([ncu.find_ncu(), "--version"], capture_output=True, text=True).stdout.strip()
    return _result("warm_cold", "N8", "INFO", rule,
                   {"buffer_mib": mib, "expected_sectors_per_round": mib * MIB // 32,
                    "cold_per_round": cold, "warm_per_round": warm,
                    "warm_minus_cold_after_priming": sum(warm[1:]) - sum(cold[1:]),
                    "ncu_version": ver.splitlines()[-1] if ver else None})


# ---- the suite ----------------------------------------------------------------------
QUICK = ("calibration", "linearity", "mixed_copy")
FULL = ("calibration", "linearity", "copy_blind_to_kernels", "kernels", "mixed",
        "repeatability", "cpu_only", "directions", "async", "pattern", "warm_cold")


def run(level="quick", only=None, unit=None, log=print):
    """Run the checks of `level` ('quick' or 'full') or just `only`; return the list of results."""
    unit = unit or default_unit()
    names = [only] if only else (QUICK if level == "quick" else FULL)
    results = []
    cal_result = check_calibration()
    results.append(cal_result)
    log(_line(cal_result))
    if cal_result["status"] == "FAIL" and "calibration_id" not in cal_result["values"]:
        return results                       # nothing else can be interpreted without one
    cal = Calibration.load()
    table = {
        "linearity": lambda: check_linearity(unit, cal),
        "copy_blind_to_kernels": lambda: check_copy_blind_to_kernels(unit, cal),
        "kernels": lambda: check_kernels(),
        "mixed": lambda: check_mixed(unit, cal),
        "mixed_copy": lambda: check_mixed(unit, cal, repeats=3, kernel_pass=False),
        "cpu_only": lambda: check_cpu_only(unit, cal),
        "directions": lambda: check_directions(unit, cal),
        "async": lambda: check_async(unit, cal),
        "repeatability": check_repeatability,
        "pattern": check_pattern,
        "warm_cold": check_warm_cold,
    }
    for name in names:
        if name == "calibration":
            continue
        try:
            r = table[name]()
        except Exception as e:               # a broken check is reported, never hidden
            r = _result(name, "", "FAIL", "", reason=f"{type(e).__name__}: {e}")
        results.append(r)
        log(_line(r))
    return results


def _line(r):
    extra = f" ({r['reason']})" if r.get("reason") else ""
    return f"[{r['status']:4s}] {r['check']:22s} {r['plan']:6s} {r['rule']}{extra}"
