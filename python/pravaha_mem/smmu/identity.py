"""Who this machine is: the facts a calibration is valid for.

Author: Deepak Panigrahy
Part of pravaha (GB10 GPU memory-traffic measurement). License: MIT.

A calibration measured on one machine, driver or kernel may not hold on
another. Every calibration and every report therefore records this identity,
and a calibration whose identity differs from the current machine is "stale".

All values come from files Linux already provides (no subprocess, no root):

    vendor, product   /sys/class/dmi/id/sys_vendor, product_name
    gpu_pci           the NVIDIA display device found in sysfs
    driver            /proc/driver/nvidia/version
    kernel            platform.release()
    cuda_runtime      cudaRuntimeGetVersion() through libcudart, if present

Fields that cannot be read are None, never a guess.
"""

import platform
import re

from . import discover

# The fields that must match for a calibration to be usable.
KEYS = ("vendor", "product", "gpu_pci", "unit", "driver", "kernel", "cuda_runtime")


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip() or None
    except OSError:
        return None


def nvidia_driver():
    """Driver version from /proc/driver/nvidia/version, e.g. '580.159.03'."""
    text = _read("/proc/driver/nvidia/version") or ""
    m = re.search(r"Kernel Module(?: for \S+)?\s+([0-9]+\.[0-9]+(?:\.[0-9]+)?)", text)
    return m.group(1) if m else None


def cuda_runtime():
    """CUDA runtime version as 'major.minor', or None if libcudart is not found."""
    try:
        from .cuda import runtime_version
        return runtime_version()
    except Exception:  # no CUDA on this machine: the field is simply unknown
        return None


def current(unit=None, sysfs="/sys"):
    """The identity of this machine now. `unit` is the counter unit in use."""
    gpus = discover.find_gpus(sysfs)
    return {
        "vendor": _read(f"{sysfs}/class/dmi/id/sys_vendor"),
        "product": _read(f"{sysfs}/class/dmi/id/product_name"),
        "gpu_pci": gpus[0] if gpus else None,
        "unit": unit,
        "driver": nvidia_driver(),
        "kernel": platform.release(),
        "cuda_runtime": cuda_runtime(),
    }


def differences(saved, now):
    """Keys whose values differ between a saved identity and the current one.

    A key unknown on either side (None) is not counted as a difference,
    but it is listed separately by the caller if it matters.
    """
    out = []
    for k in KEYS:
        a, b = saved.get(k), now.get(k)
        if a is not None and b is not None and a != b:
            out.append(k)
    return out
