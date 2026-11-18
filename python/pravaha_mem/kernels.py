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
class KernelTraffic:
    """Result of one profiled run."""

    def __init__(self, order, vals, returncode):
        self.returncode = returncode
        self.kernels = []
        for k in order:
            d = vals[k]
            self.kernels.append({
                "id": k[0], "kernel": k[1],
                "read_sectors": d.get(READ, 0.0), "write_sectors": d.get(WRITE, 0.0),
                "write_requests": d.get(WRITE_REQ, 0.0), "device_fill_sectors": d.get(DEVICE_FILL, 0.0),
                "read_mb": d.get(READ, 0.0) * SECTOR / 1e6, "write_mb": d.get(WRITE, 0.0) * SECTOR / 1e6,
            })

    @property
    def read_bytes(self):
        return sum(k["read_sectors"] for k in self.kernels) * SECTOR

    @property
    def write_bytes(self):
        return sum(k["write_sectors"] for k in self.kernels) * SECTOR

    def summary(self):
        return (f"{len(self.kernels)} kernels: {self.read_bytes / 1e9:.3f} GB read, "
                f"{self.write_bytes / 1e9:.3f} GB written (L2 sysmem events)")


def kernel_traffic(command, match=None, launch_count=None, ncu=None, extra_args=(), stdin_text=None,
                   cache_control="all"):
    """Run `command` (a list) under Nsight Compute and return a KernelTraffic.

    match: only kernels whose name matches this regular expression (passed to ncu,
           so other kernels are not even profiled).
    launch_count: stop after this many profiled kernels (ncu -c), to keep long runs short.
    stdin_text: text fed to the program's standard input (used by the mixed-workload
           test program, which waits for Enter before and after its work).
    cache_control: "all" (default) flushes caches before each replay = cold cache;
           "none" keeps them = warm cache (used only by the N8 check).

    This is a SEPARATE run of the program: Nsight Compute replays each kernel with
    cold caches, so its counts describe the kernels, not the timing of a normal run.
    """
    args = [ncu or find_ncu(), "--csv", "--print-units", "base", "--replay-mode", "kernel",
            "--cache-control", cache_control, "--metrics", ",".join(METRICS)]
    if match:
        args += ["--kernel-name", f"regex:{match}"]
    if launch_count:
        args += ["-c", str(launch_count)]
    args += list(extra_args) + list(command)
    p = subprocess.run(args, capture_output=True, text=True, input=stdin_text)
    order, vals = parse(p.stdout.splitlines())
    if not order:
        hint = p.stderr.strip().splitlines()[-1:] or [""]
        if "ERR_NVGPUCTRPERM" in p.stdout + p.stderr:
            hint = ["no permission for GPU counters: run as root, or set "
                    "NVreg_RestrictProfilingToAdminUsers=0 and reboot (see docs/TUTORIAL.md)"]
        raise RuntimeError(f"Nsight Compute returned no kernel data. {hint[0]}")
    return KernelTraffic(order, vals, p.returncode)


# 0.3 name: pravaha.kernels(...) is the same call
kernels = kernel_traffic
