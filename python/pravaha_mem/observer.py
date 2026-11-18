"""The observer: live, device wide GPU memory traffic from the GPU's L2 counters.

    import pravaha_mem as pv
    with pv.observe() as m:
        m.metadata({"model": "qwen3-4b"})
        m.mark("load");   load_model()
        m.mark("decode"); generate()
    print(m.summary())
    m.to_csv("trace.csv"); m.to_json("result.json")

The reader (pravaha-reader) is a separate process. It watches the whole GPU and
runs no GPU work, so it observes any program without changing it. All values are
device wide: they include every process using the GPU during the window.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from . import __version__

SECTOR = 32
GIB = 2 ** 30
FIELDS = ("copy", "kernel_read", "kernel_write", "all_read", "all_write")
JSON_NAMES = {"copy": "copy_bytes", "kernel_read": "kernel_read_miss_bytes",
              "kernel_write": "kernel_write_bytes", "all_read": "all_read_miss_bytes",
              "all_write": "all_write_bytes", "memory": "memory_traffic_bytes"}
METRICS = {
    "copy": ("lts__t_sectors_srcunit_ce.sum", "sectors x 32 / 2: each copied byte passes L2 twice (read, write)"),
    "kernel_read": ("lts__t_sectors_srcunit_tex_op_read_lookup_miss_realtime.sum", "sectors x 32"),
    "kernel_write": ("lts__t_sectors_srcunit_tex_op_write_realtime.sum", "sectors x 32"),
    "all_read": ("lts__t_sectors_op_read_lookup_miss_realtime.sum", "sectors x 32"),
    "all_write": ("lts__t_sectors_op_write_realtime.sum", "sectors x 32"),
}
LIBEXEC = "/usr/local/libexec/pravaha"
_last_stop = 0.0          # the driver needs about 1 s to release the counters between readers


@dataclass
class Sample:
    t0: int               # ns, system clock (same as time.time_ns())
    t1: int
    copy: float           # bytes
    kernel_read: float
    kernel_write: float
    all_read: float
    all_write: float


def find_reader() -> str:
    """Locate pravaha-reader: $PRAVAHA_READER, the install folder, then PATH."""
    for p in (os.environ.get("PRAVAHA_READER"), os.path.join(LIBEXEC, "pravaha-reader"),
              shutil.which("pravaha-reader")):
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError("GPU counter reader not installed on this machine. "
                            "Run once (asks for the admin password): pravaha install")


class observe:
    """Observe device wide GPU memory traffic while the `with` block runs (or between
    start() and stop()). interval_ms: sample interval, 10 by default."""

    def __init__(self, interval_ms: int = 10, device: int = 0):
        self.interval_ms, self.device = interval_ms, device
        self.samples: list[Sample] = []
        self.marks: list[tuple[str, int]] = []
        self.meta: dict = {}
        self.clock_pairs: list[tuple[int, int, int]] = []
        self.t_start = self.t_end = 0
        self.reader_exit = ""
        self._p = None
        self._err: list[str] = []
        self._ready = threading.Event()

    # lifecycle
    def start(self) -> "observe":
        global _last_stop
        wait = 1.0 - (time.time() - _last_stop)
        if wait > 0:
            time.sleep(wait)
        self._p = subprocess.Popen([find_reader(), "--interval-ms", str(self.interval_ms),
                                    "--device", str(self.device)],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=lambda: self._err.extend(self._p.stderr), daemon=True).start()
        deadline = time.time() + 15   # return only once samples really flow
        while not self._ready.is_set() and self._p.poll() is None and time.time() < deadline:
            self._ready.wait(0.05)
        if self._p.poll() is not None:
            msg = "".join(self._err).strip() or "pravaha-reader stopped"
            raise RuntimeError(msg.replace("pravaha-reader: ", "") + " (check: pravaha status)")
        if not self._ready.is_set():
            self._p.kill()
            raise RuntimeError("pravaha-reader produced no samples within 15 s (check: pravaha status)")
        self.t_start = time.time_ns()
        self.marks.append(("start", self.t_start))
        return self

    def stop(self) -> "observe":
        global _last_stop
        self.t_end = time.time_ns()
        time.sleep(0.1 + self.interval_ms / 1000)     # let the last samples arrive
        self._p.send_signal(signal.SIGINT)
        self._p.wait(timeout=10)
        time.sleep(0.05)
        _last_stop = time.time()
        return self

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False

    def mark(self, name: str) -> None:
        """Start a named phase now; it lasts until the next mark or the end."""
        self.marks.append((name, time.time_ns()))

    def metadata(self, d: dict) -> None:
        """Attach your own description of the run (workload, model, bytes expected ...)."""
        self.meta.update(d)

    def _read(self):
        first = True
        for line in self._p.stdout:
            f = line.split()
            if not f:
                continue
            if f[0] == "C":
                self.clock_pairs.append((int(f[1]), int(f[2]), int(f[3])))
            elif f[0] == "E":
                self.reader_exit = line.strip()
            elif f[0] == "S":
                if first:          # the first sample can hold residue from before start
                    first = False
                    continue
                self._ready.set()
                v = [float(x) * SECTOR for x in f[4:9]]
                v[0] /= 2          # copy engine: each copied byte passes L2 twice
                self.samples.append(Sample(int(f[2]), int(f[3]), *v))

    # results
    def _sum(self, a: int, b: int) -> dict:
        tot = dict.fromkeys(FIELDS, 0.0)
        split = 0.0
        for s in self.samples:     # a sample across an edge counts by the share of its time inside
            ov = min(b, s.t1) - max(a, s.t0)
            if ov > 0:
                frac = ov / (s.t1 - s.t0) if s.t1 > s.t0 else 1.0
                for k in FIELDS:
                    tot[k] += getattr(s, k) * frac
                if frac < 1.0:
                    split += (s.all_read + s.all_write) * frac
        tot["seconds"] = (b - a) / 1e9
        tot["memory"] = tot["all_read"] + tot["all_write"]
        tot["boundary_share"] = split / tot["memory"] if tot["memory"] else 0.0
        return tot

    def phases(self) -> dict[str, dict]:
        """Bytes per phase; phases are delimited by mark() calls."""
        edges = self.marks + [("end", self.t_end)]
        return {n: self._sum(a, b) for (n, a), (_, b) in zip(edges, edges[1:]) if b - a > 1_000_000}

    def total(self) -> dict:
        return self._sum(self.t_start, self.t_end)

    def quality(self) -> dict:
        """Sampling and timing facts that let a reader judge the result."""
        inside = [s for s in self.samples if s.t1 > self.t_start and s.t0 < self.t_end]
        gaps = [b.t0 - a.t1 for a, b in zip(inside, inside[1:]) if b.t0 - a.t1 > 1_000_000]
        lengths = [s.t1 - s.t0 for s in inside]
        offs = [r - g for g, r, _ in self.clock_pairs]
        return {
            "samples_received": len(inside),
            "gaps": len(gaps),
            "time_not_sampled_ms": sum(gaps) / 1e6,
            "interval_requested_ms": self.interval_ms,
            "interval_observed_ms": statistics.median(lengths) / 1e6 if lengths else None,
            "clock_pairs": len(self.clock_pairs),
            "clock_offset_ns": statistics.median(offs) if offs else None,
            "clock_offset_spread_ns": (max(offs) - min(offs)) if len(offs) > 1 else None,
            "reader_exit": self.reader_exit,
        }

    def baseline(self, phase: str = "idle") -> dict | None:
        """Background rate from an idle phase, with its variation, if one was measured."""
        p = self.phases().get(phase)
        if not p or not p["seconds"]:
            return None
        a = dict(self.marks)[phase]
        per = [(s.all_read + s.all_write) / ((s.t1 - s.t0) / 1e9) for s in self.samples
               if s.t0 >= a and s.t1 <= a + p["seconds"] * 1e9 and s.t1 > s.t0]
        return {"phase": phase, "rate_bytes_per_s": {k: p[k] / p["seconds"] for k in FIELDS},
                "memory_rate_stdev_bytes_per_s": statistics.pstdev(per) if len(per) > 1 else None}

    def estimate(self) -> dict | None:
        """Phase 'run' minus (idle rate x run time), per column, never below 0; None without both phases.
        An estimate of the command's own traffic, assuming the background stayed as during idle."""
        ph = self.phases()
        i, r = ph.get("idle"), ph.get("run")
        if not i or not r or not i["seconds"] or not r["seconds"]:
            return None
        k = r["seconds"] / i["seconds"]
        out = {JSON_NAMES.get(f, f): max(0.0, r[f] - i[f] * k) for f in (*FIELDS, "memory")}
        out["seconds"] = r["seconds"]
        return out

    def summary(self) -> str:
        rows = [("total", self.total())] + (list(self.phases().items()) if len(self.marks) > 1 else [])
        head = f"{'phase':14}{'sec':>9}{'copy GiB':>11}{'kread GiB':>11}{'kwrite GiB':>11}{'C2C GiB':>10}{'GB/s':>8}"
        lines = [head]
        for n, t in rows:
            gbps = t["memory"] / t["seconds"] / 1e9 if t["seconds"] else 0
            lines.append(f"{n:14}{t['seconds']:9.2f}{t['copy'] / GIB:11.4f}{t['kernel_read'] / GIB:11.4f}"
                         f"{t['kernel_write'] / GIB:11.4f}{t['memory'] / GIB:10.4f}{gbps:8.1f}")
        ph = dict(rows[1:])
        if "idle" in ph and "run" in ph and ph["idle"]["seconds"] and ph["run"]["seconds"]:
            i, r = ph["idle"], ph["run"]
            k = r["seconds"] / i["seconds"]           # background bytes in run = idle rate x run time
            est = {f: max(0.0, r[f] - i[f] * k) for f in ("copy", "kernel_read", "kernel_write", "memory")}
            gbps = est["memory"] / r["seconds"] / 1e9
            lines.append(f"{'run - idle':14}{r['seconds']:9.2f}{est['copy'] / GIB:11.4f}{est['kernel_read'] / GIB:11.4f}"
                         f"{est['kernel_write'] / GIB:11.4f}{est['memory'] / GIB:10.4f}{gbps:8.1f}")
            lines.append("note: 'run - idle' = run minus (idle rate x run time): an estimate of your command's "
                         "own traffic, assuming the background stayed as it was during idle")
        short = [n for n, t in rows[1:] if t["seconds"] * 1000 < 10 * self.interval_ms]
        if short:
            lines.append(f"note: phase {', '.join(short)} spans fewer than 10 samples; for sharper "
                         f"edges use interval_ms=1 (command line: --interval 1)")
        q = self.quality()
        if q["time_not_sampled_ms"]:
            lines.append(f"note: {q['time_not_sampled_ms']:.0f} ms not sampled in {q['gaps']} gaps; totals exclude it")
        lines.append("device wide: all processes on the GPU. copy: copy engines; kread/kwrite: GPU code; "
                     "C2C: all L2 read misses and writes (GPU memory traffic)")
        return "\n".join(lines)

    def to_csv(self, path: str) -> None:
        """One row per sample (bytes), with the phase it falls in."""
        edges = self.marks + [("end", self.t_end or time.time_ns())]
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t_start_ns", "t_end_ns", "phase", *(JSON_NAMES[k] for k in FIELDS)])
            for s in self.samples:
                if s.t1 <= self.t_start or (self.t_end and s.t0 >= self.t_end):
                    continue
                mid = (s.t0 + s.t1) // 2
                ph = next((n for (n, a), (_, b) in zip(edges, edges[1:]) if a <= mid < b),
                          "before_start" if mid < self.t_start else "after_end")
                w.writerow([s.t0, s.t1, ph, *(int(getattr(s, k)) for k in FIELDS)])

    def to_dict(self) -> dict:
        from .status import provenance
        name = lambda t: {JSON_NAMES.get(k, k): v for k, v in t.items()}
        return {
            "pravaha_version": __version__, "reader_protocol": 2,
            "scope": "device wide (all processes on the GPU)",
            "provenance": provenance(),
            "metrics": {JSON_NAMES[k]: {"counter": c, "conversion": conv} for k, (c, conv) in METRICS.items()},
            "memory_traffic_bytes": "all_read_miss_bytes + all_write_bytes; crosses NVLink C2C on GB10 "
                                    "(all GPU memory is the shared LPDDR5X behind the CPU die)",
            "time_base": "system clock (CLOCK_REALTIME), ns",
            "t_start_ns": self.t_start, "t_end_ns": self.t_end,
            "total": name(self.total()),
            "phases": {n: name(t) for n, t in self.phases().items()},
            "baseline": self.baseline(),
            "estimate_run_minus_idle": self.estimate(),
            "quality": self.quality(),
            "metadata": self.meta,
        }

    def to_json(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=1)


def run(cmd: list[str] | None = None, interval_ms: int = 10, idle_s: float = 0.0, seconds: float = 0.0,
        device: int = 0, live=None) -> tuple[int, observe]:
    """Observe while a command runs (optional idle baseline first), or, with no command,
    for `seconds` (or until Ctrl+C). live: a callable given each new sample."""
    say = lambda s: print(f"pravaha: {s}", file=sys.stderr, flush=True)
    m = observe(interval_ms=interval_ms, device=device).start()
    if idle_s:
        m.marks[-1] = ("idle", m.t_start)
        say(f"measuring {idle_s:g} s of idle baseline ...")
        time.sleep(idle_s)
        m.mark("run")
    rc = 0
    try:
        if cmd:
            say(f"observing the whole GPU while '{' '.join(cmd)}' runs; results print when it ends "
                f"(Ctrl+C stops it and still prints them)")
            if live:
                p = subprocess.Popen(cmd)
                rc = _follow(m, live, lambda: p.poll() is not None)
                rc = p.wait()
            else:
                rc = subprocess.call(cmd)
        else:
            what = f"for {seconds:g} s" if seconds else "until Ctrl+C"
            say(f"observing the whole GPU {what}; results print at the end")
            end = time.time() + seconds if seconds else None
            if live:
                _follow(m, live, lambda: end is not None and time.time() >= end)
            else:
                while end is None or time.time() < end:
                    time.sleep(0.1)
    except KeyboardInterrupt:
        rc = 130
        say("stopped by Ctrl+C")
    m.stop()
    if live:
        for s in m.samples[getattr(m, "_followed", 0):]:
            live(s)
    return rc, m


def _follow(m, live, done) -> int:
    m._followed = 0
    while not done():
        while m._followed < len(m.samples):
            live(m.samples[m._followed]); m._followed += 1
        time.sleep(0.05)
    return 0
