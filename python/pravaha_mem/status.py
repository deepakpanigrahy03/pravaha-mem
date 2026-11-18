"""What pravaha finds on this machine, and whether it can measure (pravaha show).

Author: Deepak Panigrahy
Part of pravaha (GB10 GPU memory-traffic measurement). License: MIT.

show() never raises: anything missing becomes a False/None field plus a reason
in plain words, so a caller (for example an A-LEMS reader's is_available())
can decide and explain without try/except.
"""

import os

from . import __version__, config, discover, identity
from .calib import Calibration, CalibrationMissing


def _read(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


def perf_permitted():
    """True if this process may open device-wide perf counters."""
    if os.geteuid() == 0:
        return True
    level = (_read("/proc/sys/kernel/perf_event_paranoid") or "2").strip()
    return level.lstrip("-").isdigit() and int(level) <= 0


def ncu_permitted():
    """True if GPU performance counters are open to this user (Nsight Compute)."""
    if os.geteuid() == 0:
        return True
    params = _read("/proc/driver/nvidia/params") or ""
    # the driver prints 'RmProfilingAdminOnly: 1' while counters are admin-only
    return "RmProfilingAdminOnly: 0" in params


def show(sysfs="/sys"):
    """Dict describing the machine and pravaha's readiness; never raises."""
    reasons = []
    info = {"pravaha_version": __version__, "gpus": discover.find_gpus(sysfs),
            "candidates": [], "unit": None, "unit_confirmed": False,
            "calibration": "absent", "calibration_id": None, "calibration_stale": [],
            "perf_permitted": perf_permitted(), "ncu_permitted": ncu_permitted(),
            "apmt_table": os.path.exists(f"{sysfs}/firmware/acpi/tables/APMT"),
            "c2c_counter": any(d.startswith("nvidia") for d in (
                os.listdir(f"{sysfs}/bus/event_source/devices")
                if os.path.isdir(f"{sysfs}/bus/event_source/devices") else [])),
            "identity": None, "reasons": reasons}
    try:
        _, info["candidates"] = discover.candidates(sysfs=sysfs)
    except discover.DiscoveryError as e:
        reasons.append(str(e))
    saved = config.saved_unit()
    info["unit"], info["unit_confirmed"] = saved, bool(saved)
    if not saved:
        reasons.append("counter unit not confirmed: run 'pravaha setup'")
    info["identity"] = identity.current(unit=saved, sysfs=sysfs)
    try:
        cal = Calibration.load()
        info["calibration_id"] = cal.calibration_id
        info["calibration_stale"] = cal.stale(info["identity"])
        info["calibration"] = "stale" if info["calibration_stale"] else "usable"
        if info["calibration_stale"]:
            reasons.append("calibration stale (" + ", ".join(info["calibration_stale"])
                           + " changed): run 'pravaha setup'")
    except CalibrationMissing as e:
        reasons.append(str(e))
    if not info["perf_permitted"]:
        reasons.append("copy counter is root only here: run 'sudo $(which pravaha) ...'")
    if not info["ncu_permitted"]:
        reasons.append("GPU counters are root only here: run kernel measurements as "
                       "'sudo $(which pravaha) measure --kernels -- CMD'")
    return info
