"""Live GPU traffic over NVLink C2C on GB10, from the GPU's L2 counters.

    import pravaha_mem as pv
    with pv.c2c() as m:
        m.mark("load");    load_model()
        m.mark("request"); generate()
    print(m.summary())
    m.to_csv("trace.csv")

The sampler (pravaha-c2c) is a separate process. It watches the whole GPU and
runs no GPU work, so it measures any program, including one it did not start.
"""
from __future__ import annotations

import csv
import os
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

SECTOR = 32
GIB = 2 ** 30
FIELDS = ("copy", "kernel_read", "kernel_write", "all_read", "all_write")


@dataclass
class Sample:
    t0: int          # ns, CLOCK_REALTIME (same clock as time.time_ns())
    t1: int
    copy: float      # bytes moved by copy engines (CE sectors x 16)
    kernel_read: float
    kernel_write: float
    all_read: float  # all L2 read misses, bytes
    all_write: float  # all L2 writes, bytes


def find_sampler() -> str:
    """Locate pravaha-c2c: $PRAVAHA_C2C, then PATH, then next to this package."""
    p = os.environ.get("PRAVAHA_C2C") or shutil.which("pravaha-c2c")
    if p:
        return p
    local = Path(__file__).with_name("pravaha-c2c")
    if local.exists():
        return str(local)
    raise FileNotFoundError(
        "pravaha-c2c not found. Run ./install.sh (builds it and grants its one capability), "
        "or set PRAVAHA_C2C=/path/to/pravaha-c2c.")


def build(out: str | None = None, cuda: str = "/usr/local/cuda") -> str:
    """Compile pravaha-c2c from the bundled source. Returns the binary path."""
    src = Path(__file__).with_name("pravaha_c2c.cpp")
    out = out or str(Path(__file__).with_name("pravaha-c2c"))
    t = Path(cuda) / "targets" / "sbsa-linux"
    inc, lib = (t / "include", t / "lib") if t.exists() else (Path(cuda) / "include", Path(cuda) / "lib64")
    cmd = ["g++", "-O2", "-o", out, str(src), f"-I{inc}", f"-L{lib}",
           f"-L{cuda}/extras/CUPTI/lib64", "-lcupti", "-lcuda", f"-Wl,-rpath,{lib}"]
    subprocess.run(cmd, check=True)
    return out


BIN = "/usr/local/bin"


def install_sampler(cuda: str | None = None) -> int:
    """Build the two small readers and install them with one capability each.

    pravaha-c2c   GPU L2 sampler   cap_sys_admin  (reads GPU counters, runs no GPU work)
    pravaha-pmcg  SMMU reader      cap_perfmon    (optional fallback)
    Builds as the current user; only the copy and setcap steps use sudo (one password).
    Changes no system setting.
    """
    import tempfile
    cuda = cuda or os.environ.get("CUDA_HOME", "/usr/local/cuda")
    here = Path(__file__).parent
    with tempfile.TemporaryDirectory() as d:
        c2c_bin = build(os.path.join(d, "pravaha-c2c"), cuda)
        pmcg_bin = os.path.join(d, "pravaha-pmcg")
        subprocess.run(["cc", "-O2", "-o", pmcg_bin, str(here / "pravaha_pmcg.c")], check=True)
        print("built pravaha-c2c and pravaha-pmcg; installing to /usr/local/bin (asks for your password once)")
        script = (f"install -m 0755 {c2c_bin} {pmcg_bin} {BIN}/ && "
                  f"setcap cap_sys_admin=ep {BIN}/pravaha-c2c && "
                  f"setcap cap_perfmon=ep {BIN}/pravaha-pmcg && "
                  f"getcap {BIN}/pravaha-c2c {BIN}/pravaha-pmcg")
        return subprocess.call(["sudo", "sh", "-c", script])


def uninstall_sampler() -> int:
    """Remove the two readers from /usr/local/bin."""
    return subprocess.call(["sudo", "rm", "-f", f"{BIN}/pravaha-c2c", f"{BIN}/pravaha-pmcg"])


class c2c:
    """Context manager that samples GPU C2C traffic while its block runs."""

    def __init__(self, interval_ms: int = 10, warmup_s: float = 0.3, tail_s: float = 0.1):
        self.interval_ms, self.warmup_s, self.tail_s = interval_ms, warmup_s, tail_s
        self.samples: list[Sample] = []
        self.marks: list[tuple[str, int]] = []
        self.t_start = self.t_end = 0
        self._p = None
        self._err: list[str] = []
        self._ready = threading.Event()

    # lifecycle
    def start(self) -> "c2c":
        self._p = subprocess.Popen([find_sampler(), "--interval-ms", str(self.interval_ms)],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=lambda: self._err.extend(self._p.stderr), daemon=True).start()
        # return only once the counters are really running: wait for the second sample
        # (the first is skipped), so no work done after start() can be missed
        deadline = time.time() + 15
        while not self._ready.is_set() and self._p.poll() is None and time.time() < deadline:
            self._ready.wait(0.05)
        if self._p.poll() is None and not self._ready.is_set():
            self._p.kill()
            raise RuntimeError("pravaha-c2c started but produced no samples within 15 s")
        if self._p.poll() is not None:
            raise RuntimeError("pravaha-c2c stopped: " + "".join(self._err).strip()
                               + " (check: pravaha permissions)")
        self.t_start = time.time_ns()
        self.marks.append(("start", self.t_start))
        return self

    def stop(self) -> "c2c":
        self.t_end = time.time_ns()
        time.sleep(self.tail_s + self.interval_ms / 1000)
        self._p.send_signal(signal.SIGINT)
        self._p.wait(timeout=10)
        time.sleep(0.05)
        return self

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False

    def mark(self, name: str) -> None:
        """Start a named phase now; it lasts until the next mark or the end."""
        self.marks.append((name, time.time_ns()))

    def _read(self):
        first = True
        for line in self._p.stdout:
            if not line.startswith("S "):
                continue
            if first:  # the first sample can hold residue from before start
                first = False
                continue
            self._ready.set()
            f = line.split()
            v = [float(x) * SECTOR for x in f[4:9]]
            v[0] /= 2  # copy engine counts each byte once read, once written
            self.samples.append(Sample(int(f[2]), int(f[3]), *v))

    # results
    def _sum(self, a: int, b: int) -> dict:
        tot = dict.fromkeys(FIELDS, 0.0)
        for s in self.samples:  # a sample straddling an edge counts by the share of its time inside
            ov = min(b, s.t1) - max(a, s.t0)
            if ov > 0:
                f = ov / (s.t1 - s.t0) if s.t1 > s.t0 else 1.0
                for k in FIELDS:
                    tot[k] += getattr(s, k) * f
        tot["seconds"] = (b - a) / 1e9
        tot["c2c"] = tot["all_read"] + tot["all_write"]
        return tot

    def phases(self) -> dict[str, dict]:
        """Bytes per phase, in bytes; phases are delimited by mark() calls."""
        edges = self.marks + [("end", self.t_end)]
        out = {}
        for (n, a), (_, b) in zip(edges, edges[1:]):
            if b - a > 1_000_000:  # skip empty phases (a mark right after start)
                out[n] = self._sum(a, b)
        return out

    def total(self) -> dict:
        return self._sum(self.t_start, self.t_end)

    def summary(self) -> str:
        rows = [("total", self.total())] + (list(self.phases().items()) if len(self.marks) > 1 else [])
        head = f"{'phase':14}{'sec':>9}{'copy GiB':>11}{'kread GiB':>11}{'kwrite GiB':>11}{'C2C GiB':>10}{'GB/s':>8}"
        lines = [head]
        for n, t in rows:
            gbps = t["c2c"] / t["seconds"] / 1e9 if t["seconds"] else 0
            lines.append(f"{n:14}{t['seconds']:9.2f}{t['copy'] / GIB:11.4f}{t['kernel_read'] / GIB:11.4f}"
                         f"{t['kernel_write'] / GIB:11.4f}{t['c2c'] / GIB:10.4f}{gbps:8.1f}")
        short = [n for n, t in rows[1:] if t["seconds"] * 1000 < 10 * self.interval_ms]
        if short:
            lines.append(f"note: phase {', '.join(short)} spans fewer than 10 samples; for sharper phase edges "
                         f"use interval_ms=1 (command line: --interval 1)")
        lost = sum(max(0, b.t0 - a.t1) for a, b in zip(self.samples, self.samples[1:]) if b.t0 - a.t1 > 1_000_000)
        if lost:
            lines.append(f"note: {lost / 1e6:.0f} ms not sampled (counter buffer reset); totals exclude it")
        lines.append("copy: copy engines; kread/kwrite: kernels; C2C: all GPU L2 misses and writes "
                     "(whole GPU, includes other programs and the desktop)")
        return "\n".join(lines)

    def to_csv(self, path: str) -> None:
        """One row per sample (bytes), with the phase it falls in."""
        edges = self.marks + [("end", self.t_end or time.time_ns())]
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t_start_ns", "t_end_ns", "phase", *FIELDS])
            for s in self.samples:
                if s.t1 <= self.t_start or (self.t_end and s.t0 >= self.t_end):
                    continue
                mid = (s.t0 + s.t1) // 2
                ph = next((n for (n, a), (_, b) in zip(edges, edges[1:]) if a <= mid < b), "")
                w.writerow([s.t0, s.t1, ph, *(int(getattr(s, k)) for k in FIELDS)])


def run(cmd: list[str] | None = None, interval_ms: int = 10, idle_s: float = 0.0,
        observe: float = 0.0) -> tuple[int, c2c]:
    """Run a command under the sampler (optional idle baseline first), or, with observe=N
    and no command, watch the whole GPU for N seconds."""
    cmd = cmd or ["sleep", str(observe)]
    import sys
    say = lambda s: print(f"pravaha: {s}", file=sys.stderr, flush=True)
    m = c2c(interval_ms=interval_ms).start()
    if idle_s:
        m.marks[-1] = ("idle", m.t_start)
        say(f"measuring {idle_s:g} s of idle baseline ...")
        time.sleep(idle_s)
        m.mark("run")
    if observe:
        say(f"observing the whole GPU for {observe:g} s; results print at the end (Ctrl+C stops early)")
    else:
        say(f"measuring while '{' '.join(cmd)}' runs; results print when it ends (Ctrl+C stops it and still prints them)")
    try:
        rc = subprocess.call(cmd)
    except KeyboardInterrupt:
        rc = 130
        say("stopped by Ctrl+C")
    m.stop()
    return rc, m
