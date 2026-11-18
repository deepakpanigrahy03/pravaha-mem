"""`pravaha permissions`: report who may read the two counters. Changes nothing.

Author: Deepak Panigrahy
Part of pravaha-mem (GB10 GPU memory traffic measurement). License: MIT.

pravaha never changes system settings. It only reports them:

  copy counter (SMMU PMCG, through perf)   readable by users when kernel.perf_event_paranoid <= 0,
                                           or through the reader pravaha-pmcg with cap_perfmon
  kernel counters (Nsight Compute)         readable by users when the NVIDIA driver allows it
                                           (RmProfilingAdminOnly: 0); otherwise run kernel
                                           measurements with sudo

How a machine is configured is the administrator's decision.
"""

import os

from . import helper


def _read(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


def perf_level():
    """The current kernel.perf_event_paranoid value as text, or None."""
    v = _read("/proc/sys/kernel/perf_event_paranoid")
    return v.strip() if v else None


def perf_open():
    level = perf_level()
    return level is not None and level.lstrip("-").isdigit() and int(level) <= 0


def driver_admin_only():
    """True if the NVIDIA driver keeps GPU counters admin only; None if the driver is not loaded."""
    params = _read("/proc/driver/nvidia/params")
    if params is None:
        return None
    return "RmProfilingAdminOnly: 1" in params


def report():
    """Print who may read each counter, and how to measure in each case. Changes nothing."""
    admin = driver_admin_only()
    print("pravaha permissions (read only; pravaha never changes system settings)")
    import shutil, subprocess
    c2 = shutil.which("pravaha-c2c")
    if c2:
        cap = subprocess.run(["getcap", c2], capture_output=True, text=True).stdout
        ok = "cap_sys_admin: ready" if "cap_sys_admin" in cap else "missing cap_sys_admin: run pravaha install-sampler"
        print(f"  L2 sampler pravaha-c2c : {c2} ({ok})")
    else:
        print("  L2 sampler pravaha-c2c : not installed (run pravaha install-sampler)")
    print(f"  copy counter (perf)    : {'readable by users' if perf_open() else 'root only'}"
          f" (kernel.perf_event_paranoid = {perf_level()})")
    if admin is None:
        print("  kernel counters (ncu)  : unknown (NVIDIA driver not loaded)")
    else:
        print(f"  kernel counters (ncu)  : {'root only' if admin else 'readable by users'}")
    path = helper.find()
    cap = helper.capability(path) if path else None
    if path:
        print(f"  {'reader ' + helper.NAME:<23}: {path} ({'cap_perfmon: ready' if cap == 'cap_perfmon' else 'missing cap_perfmon'})")
    else:
        print(f"  {'reader ' + helper.NAME:<23}: not installed")
    if not perf_open() and cap != "cap_perfmon":
        print("  copy measurements need root here, unless the administrator installs the reader once:")
        print("    pravaha helper build && sudo install -m 0755 pravaha-pmcg /usr/local/bin/ \\")
        print("      && sudo setcap cap_perfmon=ep /usr/local/bin/pravaha-pmcg")
    if admin:
        print("  live C2C traffic (copies and kernels) without root: pravaha c2c -- CMD")
    return 0
