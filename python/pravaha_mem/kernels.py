"""Kernel path: bytes GPU kernels read from and write to memory, via Nsight Compute.

Author: Deepak Panigrahy
Part of pravaha (GB10 GPU memory-traffic measurement). License: MIT.

On the GB10 all GPU memory, even cudaMalloc memory, is "system memory" to the
GPU's L2 cache, so two L2 events see kernel traffic in every allocation type:

    lts__d_sectors_fill_sysmem.sum               sectors read from memory (L2 misses)
    lts__t_sectors_aperture_sysmem_op_write.sum  sectors written to memory
    one sector = 32 bytes

Nsight Compute replays each kernel and flushes the caches before each pass, so
run this as its own pass: never time it or measure its energy.

    from pravaha_mem import kernel_traffic
    r = kernel_traffic(["python", "my_app.py"])
    print(r.read_bytes, r.write_bytes)
    for k in r.kernels: print(k["kernel"], k["read_mb"], k["write_mb"])

Permission: run as root, or allow GPU counters for everyone once
(options nvidia NVreg_RestrictProfilingToAdminUsers=0, then reboot).

A counter Nsight Compute could not read is None, never 0: the aggregate of that direction is
then None while the other direction stays complete. Errors are distinct: NoPermission,
NoKernelMatched, NoKernelData, WorkloadFailed, and TimeoutError.
Note: cudaMemset and cudaMemcpy are driver operations, not kernels; they never appear here.
"""

import csv
import os
import re
import shutil
import subprocess

SECTOR = 32
READ = "lts__d_sectors_fill_sysmem.sum"
WRITE = "lts__t_sectors_aperture_sysmem_op_write.sum"
WRITE_REQ = "lts__t_requests_aperture_sysmem_op_write.sum"
DEVICE_FILL = "lts__d_sectors_fill_device.sum"
METRICS = [READ, WRITE, WRITE_REQ, DEVICE_FILL]


def find_ncu():
    for p in (os.environ.get("NCU"), shutil.which("ncu"), "/usr/local/cuda/bin/ncu"):
        if p and os.access(p, os.X_OK):
            return p
    raise FileNotFoundError("ncu not found; install Nsight Compute or set NCU=/path/to/ncu")


# --- parsing (also used by validate/ncu_parse.py) --------------------------------
def parse(lines):
    """ncu --csv output -> (ordered list of (id, kernel name), {(id, name): {metric: value}}).
    Lines that are not ncu CSV (your program's own output) are ignored."""
    body = [line for line in lines if line.startswith('"')]
    if not body:
        return [], {}
    order, vals = [], {}
    for row in csv.DictReader(body):
        kid = (row.get("ID"), row.get("Kernel Name"))
        if kid not in vals:
            order.append(kid)
            vals[kid] = {}
        v = row.get("Metric Value", "").replace(",", "").strip()
        try:
            vals[kid][row["Metric Name"]] = float(v)
        except ValueError:
            pass
    return order, vals


def rows_out(order, vals, total=False, match=None):
    """Table rows: kernel, one column per metric (sorted), read_mb, write_mb."""
    metrics = sorted({m for d in vals.values() for m in d})
    keep = [k for k in order if not match or re.search(match, k[1] or "")]
    if total:
        out = [("TOTAL", {m: sum(vals[k].get(m, 0.0) for k in keep) for m in metrics})]
    else:
        out = [((k[1] or "").replace(",", ";"), vals[k]) for k in keep]
    header = ["kernel"] + metrics + ["read_mb", "write_mb"]
    res = []
    for name, d in out:
        r = [name] + [f"{d.get(m, 0):.0f}" for m in metrics]
        r.append(f"{d.get(READ, 0) * SECTOR / 1e6:.3f}")
        r.append(f"{d.get(WRITE, 0) * SECTOR / 1e6:.3f}")
        res.append(r)
    return header, res


# --- running ----------------------------------------------------------------
class NoPermission(RuntimeError):
    """GPU counters need profiling rights (ERR_NVGPUCTRPERM)."""


class NoKernelMatched(RuntimeError):
    """The command ran but no kernel name matched the regex."""


class NoKernelData(RuntimeError):
    """Nsight Compute returned no kernel rows (the command launched no kernel, or ncu failed)."""


class WorkloadFailed(RuntimeError):
    """The profiled command exited non zero (kernels profiled before the failure are attached)."""

    def __init__(self, returncode, msg, result=None):
        super().__init__(msg)
        self.returncode, self.result = returncode, result


class KernelTraffic:
    """Result of one profiled run. Missing counters are None; see `missing` per kernel."""

    def __init__(self, order, vals, returncode, ncu_args=None, ncu_version=None):
        self.returncode = returncode
        self.ncu_args, self.ncu_version, self.metrics = list(ncu_args or []), ncu_version, list(METRICS)
        self.kernels = []
        for k in order:
            d = vals[k]
            rs, ws = d.get(READ), d.get(WRITE)
            self.kernels.append({
                "id": k[0], "kernel": k[1], "read_sectors": rs, "write_sectors": ws,
                "write_requests": d.get(WRITE_REQ), "device_fill_sectors": d.get(DEVICE_FILL),
                "read_mb": None if rs is None else rs * SECTOR / 1e6,
                "write_mb": None if ws is None else ws * SECTOR / 1e6,
                "missing": [m for m in (READ, WRITE) if m not in d],
            })

    def _total(self, key):
        vals = [k[key] for k in self.kernels]
        return None if any(v is None for v in vals) else sum(vals) * SECTOR

    @property
    def read_bytes(self):
        """Sum over kernels, or None when any kernel lacks the read counter."""
        return self._total("read_sectors")

    @property
    def write_bytes(self):
        """Sum over kernels, or None when any kernel lacks the write counter."""
        return self._total("write_sectors")

    def summary(self):
        def gb(v):
            return "n/a" if v is None else "%.3f GB" % (v / 1e9)
        return (f"{len(self.kernels)} kernels: {gb(self.read_bytes)} read, "
                f"{gb(self.write_bytes)} written (L2 sysmem events)")


def ncu_version(ncu=None):
    """Nsight Compute version string, or None."""
    try:
        out = subprocess.run([ncu or find_ncu(), "--version"], capture_output=True, text=True, timeout=30).stdout
        return (out.strip().splitlines() or [None])[-1]
    except (OSError, subprocess.SubprocessError):
        return None


def kernel_traffic(command, match=None, launch_count=None, ncu=None, extra_args=(), stdin_text=None,
                   cache_control="all", name_base="demangled", timeout=None, replay_mode="kernel"):
    """Run `command` (a list) under Nsight Compute and return a KernelTraffic.

    match: only kernels whose name matches this regular expression (passed to ncu,
           so other kernels are not even profiled).
    name_base: which name `match` sees: "demangled" (default: full name with template
           arguments, so "MulFunctor" matches), "function" (short name) or "mangled".
    launch_count: stop after this many profiled kernels (ncu -c), to keep long runs short.
    stdin_text: text fed to the program's standard input.
    cache_control: "all" (default) flushes caches before each replay = cold cache;
           "none" keeps them = warm cache.
    timeout: seconds; TimeoutError when exceeded.

    This is a SEPARATE run of the program: Nsight Compute replays each kernel with
    cold caches, so its counts describe the kernels, not the timing of a normal run.
    """
    ncu = ncu or find_ncu()
    args = [ncu, "--csv", "--print-units", "base", "--replay-mode", replay_mode,
            "--cache-control", cache_control, "--metrics", ",".join(METRICS)]
    if match:
        args += ["--kernel-name-base", name_base, "--kernel-name", f"regex:{match}"]
    if launch_count:
        args += ["-c", str(launch_count)]
    args += list(extra_args) + list(command)
    try:
        p = subprocess.run(args, capture_output=True, text=True, input=stdin_text, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"Nsight Compute exceeded {timeout} s")
    order, vals = parse(p.stdout.splitlines())
    tail = (p.stderr.strip().splitlines()[-1:] or [""])[0]
    if "ERR_NVGPUCTRPERM" in p.stdout + p.stderr:
        raise NoPermission("no permission for GPU counters: run as root, or set "
                           "NVreg_RestrictProfilingToAdminUsers=0 and reboot (see docs/TUTORIAL.md)")
    res = KernelTraffic(order, vals, p.returncode, args, ncu_version(ncu)) if order else None
    if p.returncode != 0:
        raise WorkloadFailed(p.returncode, f"command exited {p.returncode}: {tail}", res)
    if res is None:
        if match:
            raise NoKernelMatched(f"no kernel name ({name_base}) matched regex {match!r}")
        raise NoKernelData(f"Nsight Compute returned no kernel data. {tail}")
    return res


def probe(ncu=None, timeout=300):
    """Can kernel traffic be measured here? Runs known_kernel (one tiny GPU kernel) under the
    full metric set. Returns {"ok", "reason", "ncu", "version"}; never raises."""
    import sys
    try:
        ncu = ncu or find_ncu()
    except FileNotFoundError as e:
        return {"ok": False, "reason": str(e), "ncu": None, "version": None}
    ver = ncu_version(ncu)
    cmd = [sys.executable, "-m", "pravaha_mem.known_kernel", "device", "write", "1", "1"]
    try:
        r = kernel_traffic(cmd, ncu=ncu, timeout=timeout)
    except Exception as e:   # every failure becomes a reason
        return {"ok": False, "reason": f"{type(e).__name__}: {e}", "ncu": ncu, "version": ver}
    if r.write_bytes is None:
        return {"ok": False, "reason": "write counter unavailable", "ncu": ncu, "version": ver}
    if r.write_bytes <= 0:
        return {"ok": False, "reason": "probe kernel wrote 0 bytes", "ncu": ncu, "version": ver}
    return {"ok": True, "reason": None, "ncu": ncu, "version": ver}


# 0.3 name: pravaha.kernels(...) is the same call
kernels = kernel_traffic
