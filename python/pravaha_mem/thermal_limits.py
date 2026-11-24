"""thermal_limits: when may pravaha start a check, and when must it stop?

Headroom = degrees the GPU is below its own thermal limit (nvidia-smi temperature.gpu.tlimit;
NVIDIA: 0 = software slowdown, -2 = hardware slowdown, -5 = shutdown). Each GPU reports its
own limit, so the rule below works on any GPU.

    start a check only while headroom >= start_c
    stop verification when headroom <  abort_c

start_c and abort_c come from [thermal] in pravaha.ini (shipped with the library), then a
file named by $PRAVAHA_CONFIG, then PRAVAHA_MIN_HEADROOM_START_C / _ABORT_C, then arguments.
Each resolved value is recorded with its source.
Part of pravaha-mem. MIT License. Copyright (c) 2026 MANTRA Labs.
"""
from __future__ import annotations

import configparser
import os

PACKAGE_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pravaha.ini")
KEYS = (("start_c", "PRAVAHA_MIN_HEADROOM_START_C"), ("abort_c", "PRAVAHA_MIN_HEADROOM_ABORT_C"))


def resolve(start_c=None, abort_c=None, env=None):
    """{"start_c", "abort_c", "source": {...}, "error"}; error is set when a value is missing."""
    env = os.environ if env is None else env
    args = {"start_c": start_c, "abort_c": abort_c}
    val, src = {"start_c": None, "abort_c": None}, {"start_c": None, "abort_c": None}
    for path in (PACKAGE_CONFIG, env.get("PRAVAHA_CONFIG")):
        if not path:
            continue
        cp = configparser.ConfigParser()
        if not cp.read(path):
            if path != PACKAGE_CONFIG:
                raise RuntimeError("PRAVAHA_CONFIG file not readable: %s" % path)
            continue
        for key, _ in KEYS:
            v = cp.get("thermal", key, fallback="").strip()
            if v:
                val[key], src[key] = float(v), "config:" + path
    for key, envname in KEYS:
        if env.get(envname):
            val[key], src[key] = float(env[envname]), "env:" + envname
        if args[key] is not None:
            val[key], src[key] = float(args[key]), "argument"
    err = None
    if val["start_c"] is None or val["abort_c"] is None:
        err = "start_c and abort_c must be set under [thermal] in %s" % PACKAGE_CONFIG
    elif val["start_c"] < val["abort_c"]:
        err = "start_c %.1f is below abort_c %.1f" % (val["start_c"], val["abort_c"])
    return dict(val, source=src, error=err)
