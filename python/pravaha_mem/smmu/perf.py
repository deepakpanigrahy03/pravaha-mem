"""A minimal Linux perf counter, opened directly with the perf_event_open system call.

Author: Deepak Panigrahy
Part of pravaha (GB10 GPU memory-traffic measurement). License: MIT.

This is what the ``perf`` command does underneath, without running ``perf``.
Only the Python standard library is used (ctypes), so nothing needs compiling.

The SMMU counter is an "uncore" counter: it belongs to a device, not to a
process. So it is opened for the whole system (pid = -1) on the one CPU the
kernel names in the unit's ``cpumask`` file. That needs permission:
``kernel.perf_event_paranoid`` at 0 or lower, the CAP_PERFMON capability, or root.
"""

import ctypes
import errno
import fcntl
import os
import platform
import struct
import time
from collections import namedtuple

# System call numbers for perf_event_open.
_SYSCALL = {"aarch64": 241, "arm64": 241, "x86_64": 298}

# ioctl requests from linux/perf_event.h: _IO('$', n)
_IOC_ENABLE = 0x2400
_IOC_DISABLE = 0x2401
_IOC_RESET = 0x2403

_FLAG_FD_CLOEXEC = 8
_ATTR_SIZE = 72          # PERF_ATTR_SIZE_VER1: accepted by every kernel since 2.6.33
_DISABLED_BIT = 1 << 0   # start the counter stopped; we enable it ourselves


# One raw reading of a counter.
#   count      the counter value, an unsigned 64-bit number that only grows while enabled
#   t_mono_ns  time.monotonic_ns() at the read: use it for durations and alignment
#   t_utc_ns   time.time_ns() at the read: use it to match other logs (UTC wall clock)
RawSample = namedtuple("RawSample", "count t_mono_ns t_utc_ns")


class PerfError(OSError):
    """perf_event_open failed. The message says why in plain words."""


def _read(path):
    with open(path) as f:
        return f.read().strip()


def _parse_bits(spec):
    """'config1:0-31' -> ('config1', [(0, 31)]); 'config:5' -> ('config', [(5, 5)])."""
    field, ranges = spec.split(":", 1)
    out = []
    for part in ranges.split(","):
        lo, _, hi = part.partition("-")
        out.append((int(lo), int(hi or lo)))
    return field.strip(), out


def event_config(unit, event, sysfs="/sys"):
    """Turn an event name into the numbers perf_event_open wants.

    The kernel describes each unit in sysfs:
      .../<unit>/type                 the unit's perf type number
      .../<unit>/events/<event>       e.g. 'event=0x1'
      .../<unit>/format/<term>        where each term goes, e.g. 'config:0-15'
    Returns (type, {'config': n, 'config1': n, 'config2': n}).
    """
    base = os.path.join(sysfs, "bus", "event_source", "devices", unit)
    if not os.path.isdir(base):
        raise PerfError(errno.ENOENT, f"perf unit {unit} not found in {sysfs}")
    ptype = int(_read(os.path.join(base, "type")))
    cfg = {"config": 0, "config1": 0, "config2": 0}
    for term in _read(os.path.join(base, "events", event)).split(","):
        name, _, value = term.strip().partition("=")
        value = int(value, 0) if value else 1
        field, ranges = _parse_bits(_read(os.path.join(base, "format", name)))
        for lo, hi in ranges:          # spread the value over the bit ranges, low bits first
            width = hi - lo + 1
            cfg[field] |= (value & ((1 << width) - 1)) << lo
            value >>= width
    return ptype, cfg


def unit_cpu(unit, sysfs="/sys"):
    """The CPU an uncore unit must be opened on (first entry of its cpumask)."""
    path = os.path.join(sysfs, "bus", "event_source", "devices", unit, "cpumask")
    try:
        first = _read(path).split(",")[0]
        return int(first.split("-")[0])
    except (OSError, ValueError):
        return 0


def _explain(err, unit):
    if err in (errno.EACCES, errno.EPERM):
        try:
            level = _read("/proc/sys/kernel/perf_event_paranoid")
        except OSError:
            level = "?"
        return (f"no permission to open {unit} (kernel.perf_event_paranoid = {level}) "
                "and no reader pravaha-pmcg with cap_perfmon. Run 'pravaha permissions' "
                "to see the options.")
    if err == errno.ENOENT:
        return f"{unit}: event not supported by this kernel"
    if err == errno.ENOSYS:
        return "this kernel has no perf_event_open"
    return f"{unit}: {os.strerror(err)}"


class Counter:
    """One running perf counter. Use start(), read(), stop(), close()."""

    def __init__(self, unit, event="transaction", sysfs="/sys"):
        self.unit = unit
        self.event = event
        self.fd = -1
        self._helper = None
        ptype, cfg = event_config(unit, event, sysfs)
        nr = _SYSCALL.get(platform.machine())
        if nr is None:
            raise PerfError(errno.ENOSYS, f"unsupported CPU architecture {platform.machine()}")

        # struct perf_event_attr, first 72 bytes:
        # type, size, config, sample_period, sample_type, read_format, flags,
        # wakeup_events, bp_type, config1, config2
        attr = ctypes.create_string_buffer(_ATTR_SIZE)
        struct.pack_into("<IIQQQQQIIQQ", attr, 0, ptype, _ATTR_SIZE, cfg["config"],
                         0, 0, 0, _DISABLED_BIT, 0, 0, cfg["config1"], cfg["config2"])

        libc = ctypes.CDLL(None, use_errno=True)
        libc.syscall.restype = ctypes.c_long
        fd = libc.syscall(ctypes.c_long(nr), attr, ctypes.c_int(-1),
                          ctypes.c_int(unit_cpu(unit, sysfs)), ctypes.c_int(-1),
                          ctypes.c_ulong(_FLAG_FD_CLOEXEC))
        if fd < 0:
            err = ctypes.get_errno()
            if err in (errno.EACCES, errno.EPERM) and event == "transaction":
                from . import helper            # no direct permission: try the reader
                try:
                    self._helper = helper.HelperCounter(unit)
                    return
                except helper.HelperError:
                    pass
            raise PerfError(err, _explain(err, unit))
        self.fd = fd

    def read(self):
        """Current count (64-bit; the kernel extends the hardware counter)."""
        if self._helper:
            return self._helper.read()
        return struct.unpack("<Q", os.read(self.fd, 8))[0]

    def read_raw(self):
        """Current count with monotonic and UTC timestamps (nanoseconds).

        Both clocks are read right around the counter read, so the sample can
        be lined up with other measurements (monotonic) and with logs (UTC).
        """
        t_mono = time.monotonic_ns()
        t_utc = time.time_ns()
        return RawSample(self.read(), t_mono, t_utc)

    def start(self):
        if self._helper:
            return self._helper.start()
        fcntl.ioctl(self.fd, _IOC_RESET, 0)
        fcntl.ioctl(self.fd, _IOC_ENABLE, 0)

    def stop(self):
        if self._helper:
            return self._helper.stop()
        fcntl.ioctl(self.fd, _IOC_DISABLE, 0)

    def close(self):
        if getattr(self, "_helper", None):
            self._helper.close()
            self._helper = None
        if getattr(self, "fd", -1) >= 0:
            os.close(self.fd)
            self.fd = -1

    def __del__(self):
        self.close()
