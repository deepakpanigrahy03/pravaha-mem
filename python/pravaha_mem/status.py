"""pravaha status: is this machine ready? Also the provenance recorded in every result."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess

from . import __version__

LIBEXEC = "/usr/local/libexec/pravaha"


def _q(*fields):
    """Query nvidia-smi; returns a list of strings, or Nones if unavailable."""
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        return [x.strip() for x in out[0].split(",")] if out else [None] * len(fields)
    except Exception:
        return [None] * len(fields)


def _cuda_version():
    try:
        for line in subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10).stdout.splitlines():
            if "CUDA Version" in line:
                return line.split("CUDA Version:")[1].split()[0]
    except Exception:
        pass
    return None


def thermal():
    """GPU temperature and headroom to its thermal limit (C), or None if unknown."""
    t, h = _q("temperature.gpu", "temperature.gpu.tlimit")
    num = lambda x: float(x) if x not in (None, "", "N/A", "[N/A]") else None
    return {"temperature_c": num(t), "headroom_c": num(h)}


def provenance() -> dict:
    name, uuid, driver = _q("name", "uuid", "driver_version")
    return {"gpu": name, "gpu_uuid": uuid, "driver": driver, "cuda": _cuda_version(),
            "host": platform.node(), "kernel": platform.release(), "pravaha": __version__}


def _cap(path):
    """The file's capability as getcap prints it (e.g. 'cap_perfmon=ep'), or ''."""
    try:
        out = subprocess.run(["getcap", path], capture_output=True, text=True).stdout.strip()
        return out.split(" ", 1)[1] if " " in out else ""
    except Exception:
        return ""


def readers() -> dict:
    out = {}
    for name in ("pravaha-reader", "pravaha-pmcg"):
        p = os.path.join(os.environ.get("PRAVAHA_PREFIX", LIBEXEC), name)
        out[name] = {"path": p if os.path.exists(p) else None, "capability": _cap(p) if os.path.exists(p) else ""}
    return out


def counters_free() -> str:
    """'free', 'busy', 'not installed' or an error message, from a 0.3 s read."""
    from .observer import find_reader
    try:
        r = subprocess.run([find_reader(), "--seconds", "0.3"], capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        return "not installed"
    last = (r.stdout.strip().splitlines() or [""])[-1]
    if last.startswith("E ok"):
        return "free"
    if last.startswith("E busy"):
        return "busy"
    return (r.stderr.strip().replace("pravaha-reader: ", "") or "not readable")[:160]


def status() -> dict:
    return {"provenance": provenance(), "readers": readers(), "counters": counters_free(), "thermal": thermal()}


def report() -> int:
    s = status()
    pv, th = s["provenance"], s["thermal"]
    ok = True
    print("pravaha status")
    print(f"  GPU            : {pv['gpu']}   driver {pv['driver']}   CUDA {pv['cuda']}")
    for name, r in s["readers"].items():
        if r["path"]:
            print(f"  {name:15}: installed, {r['capability'] or 'NO capability: run pravaha install'}")
            ok &= bool(r["capability"]) or name == "pravaha-pmcg"
        else:
            print(f"  {name:15}: not installed: run pravaha install")
            ok &= name == "pravaha-pmcg"
    print(f"  L2 counters    : {s['counters']}")
    ok &= s["counters"] == "free"
    if th["temperature_c"] is not None:
        hd = f", headroom to thermal limit {th['headroom_c']:.0f} C" if th["headroom_c"] is not None else ""
        print(f"  temperature    : {th['temperature_c']:.0f} C{hd}")
    stray = [p for p in ("/usr/local/bin/pravaha-c2c", "/usr/local/bin/pravaha-pmcg") if os.path.exists(p)]
    if stray:
        print(f"  note           : files from an earlier version: {' '.join(stray)} (remove with sudo rm)")
    print("  ready: measure with  pravaha run -- CMD   or   pravaha observe" if ok else
          "  not ready: see the lines above")
    return 0 if ok else 1
