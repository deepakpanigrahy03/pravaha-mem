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
    s.add_argument("--idle", type=float, default=0.0, metavar="SECONDS",
                   help="first measure SECONDS of the GPU before anything starts, shown as its own row "
                        "'idle' (your baseline: what the desktop and other programs move)")
    s.add_argument("--interval", type=int, default=10, metavar="MS",
                   help="take one sample every MS milliseconds (default 10; use 1 for phases of a few ms)")
    s.add_argument("--device", type=int, default=0, metavar="N", help="which GPU (default 0; a GB10 has one)")
    s.add_argument("--csv", metavar="FILE", help="also save one row per sample to FILE (times in ns)")
    s.add_argument("--json", metavar="FILE", help="also save the summary with all details to FILE")


OUTPUT = """output columns (all in GiB, for the whole GPU):
  copy    bytes moved by copy engines (cudaMemcpy, .cuda(), .cpu(), model loading)
  kread   bytes GPU code read from memory
  kwrite  bytes GPU code wrote to memory
  C2C     all GPU memory traffic; on GB10 it crosses NVLink C2C
  GB/s    C2C bytes per second in that row
rows: total; idle (with --idle); run, or one row per phase"""


def _sub(sub, name, help, description, examples):
    return sub.add_parser(name, help=help, formatter_class=argparse.RawDescriptionHelpFormatter,
                          description=description, epilog=examples)


def _parser():
    p = argparse.ArgumentParser(prog="pravaha", formatter_class=argparse.RawDescriptionHelpFormatter,
                                description="Live GPU memory traffic over NVLink C2C on the NVIDIA GB10, "
                                            "from the GPU's L2 cache counters. Help for one command: "
                                            "pravaha COMMAND --help", epilog=EPILOG)
    p.add_argument("--version", action="version", version=f"pravaha {__version__}")
    sub = p.add_subparsers(dest="verb", metavar="COMMAND")

    s = _sub(sub, "install", "once per machine: build the readers and allow counter reading",
             "Builds two small readers and installs them in /usr/local/libexec/pravaha, each with one\n"
             "capability (cap_perfmon: may read performance counters). Asks for your password once.\n"
             "Changes nothing else on the system. Undo with: pravaha uninstall",
             "example:\n  pravaha install")
    s.add_argument("--prefix", default=None, metavar="DIR",
                   help="install folder (default /usr/local/libexec/pravaha); must be root owned")
    s.add_argument("--cap", default="cap_perfmon", choices=["cap_perfmon", "cap_sys_admin"],
                   help="capability for the L2 reader: cap_perfmon (default, smallest) or cap_sys_admin "
                        "(only if your driver refuses cap_perfmon)")
    s = _sub(sub, "uninstall", "remove exactly what install created",
             "Removes the files listed in the install MANIFEST (after checking they are unchanged),\n"
             "the folder, and any parent folders install created. Confirms nothing is left.",
             "example:\n  pravaha uninstall\n  pip uninstall pravaha-mem      (the Python package)")
    s.add_argument("--prefix", default=None, metavar="DIR")
    _sub(sub, "status", "is this machine ready? GPU, readers, counters, temperature",
         "Answers: which GPU and driver; are the readers installed with their capability; are the\n"
         "GPU counters free right now (or used by another profiler); GPU temperature and headroom\n"
         "to its thermal limit. Ends with 'ready' or what is missing.", "example:\n  pravaha status")
    s = _sub(sub, "verify", "check pravaha against a program that moves known bytes (light, about 3 min)",
             "Runs known_bytes (a small, readable test program) four times: copy 2 GiB into the GPU,\n"
             "copy 2 GiB out, GPU code writes 4 GiB, GPU code reads 4 GiB. pravaha measures each from\n"
             "outside and compares with the known amount. PASS = agreement within the stated tolerance\n"
             "on this machine. Light work only; waits between checks; stops if the GPU gets warm.",
             "example:\n  pravaha verify --json verify.json")
    s.add_argument("--json", metavar="FILE", help="save the full report (every run, background, temperatures)")
    s.add_argument("--cooldown", type=float, default=30.0, metavar="SECONDS",
                   help="wait between checks (default 30)")
    s.add_argument("--schedule", action="store_true",
                   help="instead: the schedule runner checked under the observer (streaming and "
                        "resident reads, sham, copy, square wave); settings in [schedule] of pravaha.ini")
    s.add_argument("--kernels", action="store_true",
                   help="instead: known GPU kernels reading and writing in each memory placement "
                        "(device, pinned, pageable, managed); logical bytes exact, counter traffic reported")

    s = _sub(sub, "run", "measure GPU memory traffic while a command runs",
             "Starts observing, runs your command, stops when it ends, prints the table.\n"
             "Device wide: counts every process on the GPU during that time, not only your command.\n\n"
             + OUTPUT,
             "examples:\n"
             "  pravaha run -- python my_app.py\n"
             "  pravaha run --idle 2 --csv trace.csv -- ./my_cuda_program\n"
             "  pravaha run -- python -m pravaha_mem.known_bytes h2d 8 256   (copies 2 GiB into the GPU)")
    _measure_opts(s)
    s.add_argument("command", nargs=argparse.REMAINDER, help="-- then the command and its arguments")
    s = _sub(sub, "observe", "measure GPU memory traffic over a time window (e.g. a running server)",
             "Observes the whole GPU for --seconds, or until Ctrl+C, without starting anything.\n"
             "Use it for a server that is already running. Device wide.\n\n" + OUTPUT,
             "examples:\n"
             "  pravaha observe --seconds 60\n"
             "  pravaha observe --live > samples.csv          (each sample as it arrives; Ctrl+C stops)")
    _measure_opts(s)
    s.add_argument("--seconds", type=float, default=0.0, help="how long (default: until Ctrl+C)")
    s.add_argument("--live", action="store_true", help="also print each sample as a CSV line as it arrives")

    s = _sub(sub, "smmu", "GPU address translations (counts, never bytes)",
             "The GPU's SMMU translates the addresses programs use into real memory addresses, one\n"
             "page at a time. These counts show translation requests and misses: useful to study page\n"
             "sizes and translation caching. They are not bytes; for bytes use run or observe.",
             "examples:\n  pravaha smmu status\n  pravaha smmu run -- python my_app.py\n"
             "  pravaha smmu observe --seconds 10")
    ss = s.add_subparsers(dest="smmu_verb", metavar="status|run|observe")
    ss.add_parser("status", help="the SMMU counters of this GB10 and which one belongs to the GPU")
    r = ss.add_parser("run", help="translation counts while a command runs")
    r.add_argument("--unit", help="which SMMU counter unit (default: the GPU's)")
    r.add_argument("--live", action="store_true", help="also print counts every 100 ms")
    r.add_argument("command", nargs=argparse.REMAINDER)
    o = ss.add_parser("observe", help="translation counts over a time window")
    o.add_argument("--unit", help="which SMMU counter unit (default: the GPU's)")
    o.add_argument("--live", action="store_true", help="also print counts every 100 ms")
    o.add_argument("--seconds", type=float, default=0.0, help="how long (default: until Ctrl+C)")

    s = _sub(sub, "kernels", "L2 traffic per named GPU kernel (Nsight Compute)",
             "Runs your command under NVIDIA Nsight Compute, which replays each GPU kernel to measure\n"
             "it. Slower than run, and needs profiling rights. Use it to see which kernel moves the\n"
             "bytes that run reports in total.",
             "example:\n  pravaha kernels --match gemm -- python my_app.py")
    s.add_argument("--match", metavar="REGEX", help="only kernels whose name matches REGEX")
    s.add_argument("--name-base", default="demangled", choices=("demangled", "function", "mangled"),
                   help="which kernel name REGEX sees (default demangled: includes template arguments)")
    s.add_argument("--timeout", type=float, default=None, metavar="SECONDS")
    s.add_argument("--probe", action="store_true", help="only check that kernel traffic can be measured here")
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
            if a.schedule:
                from .verify_schedule import verify_schedule
                return 0 if verify_schedule(a.json, a.cooldown)["pass"] else 1
            if a.kernels:
                from .verify_kernels import verify_kernels
                return 0 if verify_kernels(a.json, a.cooldown)["all_pass"] else 1
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
            from .kernels import kernel_traffic, probe
            if a.probe:
                r = probe()
                print(("ready: " if r["ok"] else "not ready: ") + (r["reason"] or str(r["version"])))
                return 0 if r["ok"] else 1
            cmd = _cmd(a)
            if not cmd:
                return _err("usage: pravaha kernels [--match REGEX] [--name-base B] [--timeout S] -- CMD [ARGS]")
            k = kernel_traffic(cmd, match=a.match, name_base=a.name_base, timeout=a.timeout)
            print(k.summary())
            return 0
        _parser().print_help()
        return 0
    except KeyboardInterrupt:
        return 130
    except (RuntimeError, FileNotFoundError, OSError, TimeoutError) as e:
        return _err(str(e))


if __name__ == "__main__":
    sys.exit(main())
