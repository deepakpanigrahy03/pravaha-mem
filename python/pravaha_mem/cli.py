"""The `pravaha` command: show, setup, measure, watch, test.

Author: Deepak Panigrahy
Part of pravaha (GB10 GPU memory-traffic measurement). License: MIT.

    pravaha show                     what pravaha sees on this machine, and what is missing
    pravaha setup                    confirm the counter, calibrate, save (once per machine/driver)
    pravaha measure -- CMD ...       copy traffic while CMD runs (system wide)
    pravaha measure --kernels -- CMD also a separate Nsight Compute pass for kernel traffic
    pravaha watch                    raw counter every N ms, as CSV
    pravaha test [--full|--only X]   the letter's checks on this machine

Every command prints what it measures (system-wide copy traffic) and refuses
to print bytes it cannot stand behind (no calibration -> transactions only).
Old 0.2 verb names (find, idle, calibrate, monitor, kernel, info) still work.
"""

import argparse
import json
import subprocess
import sys
import time

from . import __version__, calibrate, config, checks, ncu, status
from .calib import Calibration, CalibrationMissing
from .discover import DiscoveryError, candidates
from .measure import Measurement, watch

REPORT_SCHEMA = 1
ALIASES = {"find": "show", "info": "show", "idle": "setup", "calibrate": "setup",
           "monitor": "watch", "kernel": "measure"}


def _err(msg, code=2):
    print(f"pravaha: {msg}", file=sys.stderr)
    return code


def _write_json(path, obj):
    """Write a report; '-' means standard output."""
    text = json.dumps(obj, indent=2, default=str)
    if path == "-":
        print(text)
    else:
        with open(path, "w") as f:
            f.write(text + "\n")


# ---- show ------------------------------------------------------------------------
def cmd_show(a):
    """Print the status dict in plain lines (or JSON)."""
    s = status.show()
    if a.json:
        _write_json("-", s)
        return 0
    print(f"pravaha {s['pravaha_version']}")
    print(f"  GPUs found          : {', '.join(s['gpus']) or 'none'}")
    print(f"  GPU SMMU candidates : {', '.join(s['candidates']) or 'none'}")
    print(f"  unit in use         : {s['unit'] or '-'} ({'confirmed' if s['unit_confirmed'] else 'not confirmed'})")
    print(f"  calibration         : {s['calibration']} {s['calibration_id'] or ''}")
    print(f"  perf permission     : {'yes' if s['perf_permitted'] else 'no'}")
    print(f"  GPU counter (ncu)   : {'yes' if s['ncu_permitted'] else 'admin only'}")
    print(f"  firmware APMT table : {'present' if s['apmt_table'] else 'absent'}")
    print(f"  C2C traffic counter : {'exposed' if s.get('c2c_counter') else 'not exposed (pravaha measures copies and kernel misses instead)'}")
    for r in s["reasons"]:
        print(f"  ! {r}")
    return 0


# ---- setup -----------------------------------------------------------------------
def _confirm_unit(a):
    """Pick the copy counter: the one given, the only candidate, or the one that moves with copies."""
    if a.unit:
        return a.unit
    saved = config.saved_unit()                  # confirmed earlier (also by 0.2)
    if saved:
        return saved
    _bdf, units = candidates()                   # candidates() gives (gpu address, units)
    if len(units) == 1:
        return units[0]
    print("Several counter units found; copying 256 MiB x 8 under each to find the GPU's:")
    from .cuda import Buffers
    from .perf import Counter
    buf, best = Buffers(256 << 20), (None, 0)
    for u in units:
        c = Counter(u)
        c.start()
        for _ in range(8):
            buf.copy("h2d")
        buf.sync()
        c.stop()
        n = c.read()
        c.close()
        print(f"  {u:28s} {n:>12,d} transactions")
        best = max(best, (u, n), key=lambda x: x[1])
    buf.free()
    config.save_unit(best[0])
    return best[0]


def cmd_setup(a):
    """Confirm the unit, calibrate, save calibration.json and calibration.tsv."""
    try:
        unit = _confirm_unit(a)
    except DiscoveryError as e:
        return _err(str(e))
    if not a.yes:
        reply = input(f"Use {unit} as the GPU copy counter and calibrate now (about 3 min)? [y/N] ")
        if reply.strip().lower() not in ("y", "yes"):
            return _err("setup cancelled", 1)
    pinned = [int(x) for x in a.pinned.split(",")]
    pageable = [int(x) for x in a.pageable.split(",")] if a.pageable else []
    asyncs = [int(x) for x in a.async_sizes.split(",")] if a.async_sizes else []
    cal, _rows = calibrate.run(unit, pinned, pageable, asyncs, repeats=a.repeats)
    path = cal.save(system=a.system)
    print(f"saved {path} (id {cal.calibration_id})")
    for host, d, mib in cal.unstable_sizes():
        print(f"  ! unstable: {host} {d} {mib} MiB (spread > {calibrate.UNSTABLE_PCT}%): bounds only there")
    return 0


# ---- measure ---------------------------------------------------------------------
def _parse_profile(text):
    """'256:h2d:8,256:d2h:8' or '64:h2d:pageable:4' -> {(mib, dir, pageable): copies}."""
    prof = {}
    for item in filter(None, (text or "").split(",")):
        f = item.split(":")
        pageable = (f[2] == "async" and "async") or f[2] == "pageable" if len(f) == 4 else False
        prof[(int(f[0]), f[1], pageable)] = int(f[-1])
    return prof or None


def cmd_measure(a):
    """Copy traffic around a command; optionally a separate kernel pass."""
    cmd = [x for x in a.command if x != "--"]
    if not cmd:
        return _err("give a command after --, e.g. pravaha measure -- python app.py")
    profile = _parse_profile(a.profile)
    m = Measurement(a.unit)
    m.window = ("the whole command, including CUDA start and memory allocation; "
                "for copies only, measure inside your program (docs/MANUAL.md)")
    m.start()
    state, rc = "complete", 0
    try:
        rc = subprocess.call(cmd)
        if rc:
            state = f"partial: command exited {rc}"
    except KeyboardInterrupt:
        state = "partial: interrupted"
    finally:
        m.stop(state)
    report = {"schema": REPORT_SCHEMA, "pravaha_version": __version__,
              "command": cmd, "copy": m.to_dict(profile)}
    print(m.summary(profile))
    if a.kernels:
        print("kernel pass: running the command again under Nsight Compute (separate run, cold caches)")
        try:
            k = ncu.kernels(cmd, match=a.match)
            report["kernels"] = {"pass": "separate", "read_bytes": k.read_bytes,
                                 "write_bytes": k.write_bytes, "per_kernel": k.kernels}
            print("  " + k.summary())
        except (RuntimeError, FileNotFoundError) as e:
            report["kernels"] = {"pass": "failed", "reason": str(e)}
            print(f"  ! kernel pass failed: {e}")
    if a.json:
        _write_json(a.json, report)
    m.close()
    return rc


# ---- watch -----------------------------------------------------------------------
def cmd_watch(a):
    """CSV lines: utc_ns, mono_ns, raw count, delta since the last line."""
    print("t_utc_ns,t_mono_ns,raw_count,delta")
    end = time.monotonic() + a.seconds if a.seconds else None
    try:
        for s, d in watch(a.interval, a.unit):
            print(f"{s.t_utc_ns},{s.t_mono_ns},{s.count},{d}", flush=True)
            if end and time.monotonic() >= end:
                break
    except KeyboardInterrupt:
        pass
    return 0


# ---- test ------------------------------------------------------------------------
def cmd_test(a):
    if not a.smmu:
        from .c2c_checks import main as c2c_main
        return c2c_main(a.json)
    return cmd_test_smmu(a)


def cmd_test_smmu(a):
    """Run the checks; write a JSON report; exit 1 if any check failed."""
    level = "full" if a.full else "quick"
    results = checks.run(level, a.only, a.unit)
    try:
        cal_id = Calibration.load().calibration_id
    except CalibrationMissing:
        cal_id = None
    report = {"schema": REPORT_SCHEMA, "pravaha_version": __version__, "level": a.only or level,
              "status": status.show(), "calibration_id": cal_id, "results": results,
              "scope": {"copy_path": "system-wide SMMU transactions; host<->GPU copies",
                        "kernel_path": "per-kernel L2 sysmem sectors, separate Nsight Compute pass",
                        "not_covered": "process attribution, timing, CPU-only traffic"}}
    if a.json:
        _write_json(a.json, report)
    failed = [r["check"] for r in results if r["status"] == "FAIL"]
    print(f"{len(results)} checks, {len(failed)} failed" + (f": {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


# ---- argument parsing ------------------------------------------------------------
def cmd_permissions(a):
    """Report who may read the counters on this machine. Changes nothing."""
    from . import permissions
    return permissions.report()


def cmd_helper(a):
    """Compile the reader pravaha-pmcg into the current folder."""
    from . import helper
    try:
        path = helper.build()
    except (helper.HelperError, OSError) as e:
        return _err(str(e))
    print(f"built {path}")
    print("to let every user measure copies without sudo, an administrator runs once:")
    print(f"  sudo install -m 0755 {path} /usr/local/bin/")
    print("  sudo setcap cap_perfmon=ep /usr/local/bin/pravaha-pmcg")
    return 0


def cmd_install_sampler(a):
    from .c2c_sampler import install_sampler
    return install_sampler()


def cmd_uninstall_sampler(a):
    from .c2c_sampler import uninstall_sampler
    return uninstall_sampler()


def cmd_c2c(a):
    from .c2c_sampler import run
    cmd = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not cmd and not a.seconds:
        return _err("usage: pravaha c2c [--idle S] [--csv FILE] [--json FILE] -- COMMAND\n"
                    "   or: pravaha c2c --seconds N [--csv FILE]   (observe whatever runs)")
    observe = 0.0 if cmd else a.seconds
    try:
        rc, m = run(cmd, interval_ms=a.interval, idle_s=a.idle, observe=observe)
    except (RuntimeError, FileNotFoundError) as e:
        return _err(str(e))
    print(m.summary(), file=sys.stderr)
    if a.csv:
        m.to_csv(a.csv)
        print(f"trace: {a.csv}", file=sys.stderr)
    if a.json:
        import json
        with open(a.json, "w") as fh:
            json.dump({"total": m.total(), "phases": m.phases()}, fh, indent=1)
        print(f"json: {a.json}", file=sys.stderr)
    return rc


def _parser():
    p = argparse.ArgumentParser(prog="pravaha", description="GPU memory traffic on the NVIDIA GB10")
    p.add_argument("--version", action="version", version=f"pravaha {__version__}")
    sub = p.add_subparsers(dest="verb")
    sub.add_parser("install-sampler", help="once per machine: install the counter readers (asks for the password once)")
    sub.add_parser("uninstall-sampler", help="remove the counter readers")
    s = sub.add_parser("c2c", help="live GPU traffic over C2C while a command runs (L2 counters)")
    s.add_argument("--csv", metavar="FILE", help="per sample trace")
    s.add_argument("--idle", type=float, default=0.0, help="seconds of idle baseline first")
    s.add_argument("--interval", type=int, default=10, help="sample interval, ms")
    s.add_argument("--seconds", type=float, default=0, help="with no command: observe the GPU for N seconds")
    s.add_argument("--json", metavar="FILE", help="summary (total and phases, bytes) as JSON")
    s.add_argument("command", nargs=argparse.REMAINDER)
    s = sub.add_parser("show", help="what pravaha sees, and what is missing")
    s.add_argument("--json", action="store_true")
    s = sub.add_parser("setup", help="confirm the counter and calibrate (once per machine/driver)")
    s.add_argument("--unit"); s.add_argument("--yes", "-y", action="store_true")
    s.add_argument("--pinned", default=",".join(map(str, calibrate.DEFAULT_PINNED)))
    s.add_argument("--pageable", default=",".join(map(str, calibrate.DEFAULT_PAGEABLE)))
    s.add_argument("--async-sizes", default=",".join(map(str, calibrate.DEFAULT_ASYNC)),
                   help="sizes for async (2-stream) copies; empty to skip")
    s.add_argument("--repeats", type=int, default=5)
    s.add_argument("--system", action="store_true", help="save to /etc/pravaha (needs root)")
    s = sub.add_parser("measure", help="copy traffic while a command runs")
    s.add_argument("--unit"); s.add_argument("--json", metavar="FILE")
    s.add_argument("--profile", help="known copies, e.g. 256:h2d:8,64:h2d:async:4 (else bounds)")
    s.add_argument("--kernels", action="store_true", help="also a separate Nsight Compute pass")
    s.add_argument("--match", help="kernel-name regex for the kernel pass")
    s.add_argument("command", nargs=argparse.REMAINDER)
    s = sub.add_parser("watch", help="raw counter every N ms, as CSV")
    s.add_argument("--unit"); s.add_argument("--interval", type=int, default=100)
    s.add_argument("--seconds", type=float, default=0)
    sub.add_parser("permissions", help="who may read the counters here (changes nothing)")
    s = sub.add_parser("helper", help="build the reader pravaha-pmcg (for use without sudo)")
    s.add_argument("action", choices=["build"])
    s = sub.add_parser("test", help="run the validation checks on this machine")
    s.add_argument("--unit"); s.add_argument("--json", metavar="FILE")
    g = s.add_mutually_exclusive_group()
    s.add_argument("--smmu", action="store_true", help="the older SMMU translation counter checks")
    g.add_argument("--quick", action="store_true"); g.add_argument("--full", action="store_true")
    g.add_argument("--only", choices=sorted(set(checks.FULL) | set(checks.QUICK)))
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ALIASES:                     # 0.2 names keep working
        print(f"pravaha: '{argv[0]}' is now '{ALIASES[argv[0]]}'", file=sys.stderr)
        argv[0] = ALIASES[argv[0]]
    a = _parser().parse_args(argv)
    verbs = {"show": cmd_show, "setup": cmd_setup, "measure": cmd_measure,
             "watch": cmd_watch, "test": cmd_test, "permissions": cmd_permissions,
             "helper": cmd_helper, "c2c": cmd_c2c,
             "install-sampler": cmd_install_sampler, "uninstall-sampler": cmd_uninstall_sampler}
    if not a.verb:
        _parser().print_help()
        return 0
    try:
        return verbs[a.verb](a)
    except DiscoveryError as e:
        return _err(str(e))


if __name__ == "__main__":
    sys.exit(main())
