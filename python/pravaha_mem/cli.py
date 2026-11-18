"""pravaha command line. Thin: every command calls one function of the package."""
from __future__ import annotations

import argparse
import sys

from . import __version__

EPILOG = """start here:
  pravaha install              once per machine (asks for your password once)
  pravaha status               is this machine ready?
  pravaha verify               check the counters against known bytes (about 3 minutes, light)
  pravaha run -- CMD           GPU memory traffic while CMD runs (device wide)
  pravaha observe --seconds 60 GPU memory traffic over a time window, e.g. a running server

more:
  pravaha smmu status|run|observe   GPU address translations (counts, not bytes)
  pravaha kernels -- CMD            L2 traffic per kernel (Nsight Compute replay)
  pravaha uninstall                 remove what install created
docs: docs/TUTORIAL.md, docs/MANUAL.md"""


def _err(msg: str) -> int:
    print(f"pravaha: {msg}", file=sys.stderr)
    return 1


def _measure_opts(s):
    s.add_argument("--idle", type=float, default=0.0, metavar="S", help="measure S seconds of idle baseline first")
    s.add_argument("--interval", type=int, default=10, metavar="MS", help="sample interval in ms (default 10)")
    s.add_argument("--device", type=int, default=0, help="GPU index (default 0)")
    s.add_argument("--csv", metavar="FILE", help="one row per sample")
    s.add_argument("--json", metavar="FILE", help="summary with provenance")


def _parser():
    p = argparse.ArgumentParser(prog="pravaha", formatter_class=argparse.RawDescriptionHelpFormatter,
                                description="Live GPU memory traffic over NVLink C2C on the NVIDIA GB10, "
                                            "from the GPU's L2 cache counters.", epilog=EPILOG)
    p.add_argument("--version", action="version", version=f"pravaha {__version__}")
    sub = p.add_subparsers(dest="verb", metavar="COMMAND")

    s = sub.add_parser("install", help="once per machine: build the readers and allow counter reading")
    s.add_argument("--prefix", default=None, help="install folder (default /usr/local/libexec/pravaha)")
    s.add_argument("--cap", default="cap_perfmon", choices=["cap_perfmon", "cap_sys_admin"],
                   help="capability for the L2 reader (default cap_perfmon)")
    s = sub.add_parser("uninstall", help="remove exactly what install created")
    s.add_argument("--prefix", default=None)
    sub.add_parser("status", help="is this machine ready? GPU, readers, counters, temperature")
    s = sub.add_parser("verify", help="check the counters against known bytes (light, about 3 minutes)")
    s.add_argument("--json", metavar="FILE", help="full report")
    s.add_argument("--cooldown", type=float, default=30.0, metavar="S", help="seconds between checks (default 30)")

    s = sub.add_parser("run", help="GPU memory traffic while a command runs (device wide)")
    _measure_opts(s)
    s.add_argument("command", nargs=argparse.REMAINDER, help="-- CMD [ARGS]")
    s = sub.add_parser("observe", help="GPU memory traffic over a time window (device wide)")
    _measure_opts(s)
    s.add_argument("--seconds", type=float, default=0.0, help="how long (default: until Ctrl+C)")
    s.add_argument("--live", action="store_true", help="also print each sample as CSV while observing")

    s = sub.add_parser("smmu", help="GPU address translations (counts, never bytes)")
    ss = s.add_subparsers(dest="smmu_verb", metavar="status|run|observe")
    ss.add_parser("status", help="SMMU counters of this GB10 and which one is the GPU's")
    r = ss.add_parser("run", help="translations while a command runs")
    r.add_argument("--unit"); r.add_argument("--live", action="store_true")
    r.add_argument("command", nargs=argparse.REMAINDER)
    o = ss.add_parser("observe", help="translations over a time window")
    o.add_argument("--unit"); o.add_argument("--live", action="store_true")
    o.add_argument("--seconds", type=float, default=0.0)

    s = sub.add_parser("kernels", help="L2 traffic per kernel (Nsight Compute replay; slower, needs profiling rights)")
    s.add_argument("--match", help="regular expression for kernel names")
    s.add_argument("command", nargs=argparse.REMAINDER)
    return p


def _cmd(a):
    return a.command[1:] if a.command[:1] == ["--"] else a.command


def _live_row(s):
    print(f"{s.t0},{s.t1},{int(s.copy)},{int(s.kernel_read)},{int(s.kernel_write)},{int(s.all_read)},{int(s.all_write)}",
          flush=True)


def _measure(a, cmd):
    from .observer import run
    if getattr(a, "live", False):
        print("t_start_ns,t_end_ns,copy_bytes,kernel_read_miss_bytes,kernel_write_bytes,"
              "all_read_miss_bytes,all_write_bytes", flush=True)
    rc, m = run(cmd or None, interval_ms=a.interval, idle_s=a.idle, seconds=getattr(a, "seconds", 0.0),
                device=a.device, live=_live_row if getattr(a, "live", False) else None)
    print(m.summary(), file=sys.stderr)
    if a.csv:
        m.to_csv(a.csv); print(f"csv: {a.csv}", file=sys.stderr)
    if a.json:
        m.to_json(a.json); print(f"json: {a.json}", file=sys.stderr)
    return rc


def main(argv=None):
    a = _parser().parse_args(argv)
    try:
        if a.verb == "install":
            from .install import DEFAULT_PREFIX, install
            return install(a.prefix or DEFAULT_PREFIX, a.cap)
        if a.verb == "uninstall":
            from .install import DEFAULT_PREFIX, uninstall
            return uninstall(a.prefix or DEFAULT_PREFIX)
        if a.verb == "status":
            from .status import report
            return report()
        if a.verb == "verify":
            from .verify import verify
            return 0 if verify(a.json, a.cooldown)["all_pass"] else 1
        if a.verb == "run":
            cmd = _cmd(a)
            if not cmd:
                return _err("usage: pravaha run [--idle S] [--csv FILE] [--json FILE] -- CMD [ARGS]")
            return _measure(a, cmd)
        if a.verb == "observe":
            return _measure(a, None)
        if a.verb == "smmu":
            from .smmu import commands as sm
            if a.smmu_verb == "status":
                return sm.status()
            if a.smmu_verb == "run":
                cmd = _cmd(a)
                if not cmd:
                    return _err("usage: pravaha smmu run [--unit U] [--live] -- CMD [ARGS]")
                return sm.run(cmd, unit=a.unit, live=a.live)
            if a.smmu_verb == "observe":
                return sm.run(None, seconds=a.seconds, unit=a.unit, live=a.live)
            return _err("usage: pravaha smmu status | run -- CMD | observe [--seconds N]")
        if a.verb == "kernels":
            from .kernels import kernel_traffic
            cmd = _cmd(a)
            if not cmd:
                return _err("usage: pravaha kernels [--match REGEX] -- CMD [ARGS]")
            k = kernel_traffic(cmd, match=a.match)
            print(k.summary())
            return 0
        _parser().print_help()
        return 0
    except KeyboardInterrupt:
        return 130
    except (RuntimeError, FileNotFoundError, OSError) as e:
        return _err(str(e))


if __name__ == "__main__":
    sys.exit(main())
