"""Measure bytes per SMMU transaction on this machine, for each copy size and direction.

Author: Deepak Panigrahy
Part of pravaha (GB10 GPU memory-traffic measurement). License: MIT.

Why in-process
--------------
Counting a whole program also counts CUDA start-up (about 200,000 transactions
on the tested GB10), which then has to be removed by fitting a line. Here the
counter runs only around the copies, inside one process, so start-up never
enters the count.

One measurement (function `measure`)
------------------------------------
  1. count the idle background for a moment, right before;
  2. count N copies of the given size (N x size = about 2 GiB);
  3. bytes per transaction = bytes copied / (transactions - background x time).

The calibration (function `run`)
--------------------------------
For every (host memory, direction, size): `repeats` measurements. The median goes
into the table, with the repeat spread. If the repeats split into two clearly
different levels (seen at 128 MiB on the GN100), the size is marked unstable
and both levels are kept, so nobody uses it as if it were steady.
"""

import statistics
import time

from . import __version__, identity
from .calib import SCHEMA, Calibration
from .cuda import Buffers
from .perf import Counter

DEFAULT_PINNED = [1, 2, 4, 8, 16, 32, 64, 128, 256]
DEFAULT_PAGEABLE = [4, 64, 256]
DEFAULT_ASYNC = [16, 64, 256]   # async copies on 2 streams cost ~1.4% more transactions (N6)
METHOD = "inprocess-2"          # 2: fresh buffer per repeat (fix 5)          # change this name if the procedure changes
UNSTABLE_PCT = 5.0              # repeats further apart than this = two levels


def measure(counter, buf, direction, copies, idle_seconds, use_async=False):
    """One measurement. Returns (bytes per transaction or None, net transactions, seconds)."""
    # background right before, over the same counter
    counter.start()
    time.sleep(idle_seconds)
    counter.stop()
    idle_rate = counter.read() / idle_seconds

    buf.copy(direction)          # warm-up copy, not counted
    buf.sync()
    counter.start()
    t0 = time.monotonic()
    if use_async:                # cudaMemcpyAsync over 2 streams, as frameworks copy
        buf.copy_async(direction, copies)
    else:
        for _ in range(copies):
            buf.copy(direction)
        buf.sync()
    dt = time.monotonic() - t0
    counter.stop()
    net = counter.read() - idle_rate * dt
    if net <= 0:                 # nothing measurable: report it, never invent a value
        return None, net, dt
    return copies * buf.nbytes / net, net, dt


def _summarise(values):
    """Median, spread and (if the values jump) the two levels."""
    v = sorted(values)
    med = statistics.median(v)
    spread = (v[-1] - v[0]) / med * 100 if len(v) > 1 else 0.0
    levels = []
    if spread > UNSTABLE_PCT:
        # split at the largest gap between neighbouring values
        gap, i = max((v[k + 1] - v[k], k) for k in range(len(v) - 1))
        levels = [round(statistics.median(v[:i + 1]), 1), round(statistics.median(v[i + 1:]), 1)]
    return med, spread, levels


def run(unit, pinned=DEFAULT_PINNED, pageable=DEFAULT_PAGEABLE, async_sizes=DEFAULT_ASYNC, repeats=5,
        total_mib=2048, idle_seconds=1.0, log=print):
    """Measure every configuration; returns (Calibration, list of raw rows)."""
    counter = Counter(unit)
    rows, entries = [], []
    for host, sizes in (("pinned", pinned), ("pageable", pageable), ("async", async_sizes)):
        for mib in sizes:
            copies = max(4, total_mib // mib)
            for direction in ("h2d", "d2h"):
                values = []
                for r in range(repeats):
                    # a new allocation each repeat: where its pages land can change the result
                    buf = Buffers(mib << 20, pageable=(host == "pageable"))
                    bpt, net, dt = measure(counter, buf, direction, copies, idle_seconds, use_async=(host == "async"))
                    rows.append({"host": host, "mib": mib, "dir": direction, "run": r + 1,
                                 "copies": copies, "net_txn": round(net), "seconds": round(dt, 4),
                                 "bytes_per_txn": round(bpt, 1) if bpt else None})
                    buf.free()
                    if bpt:
                        values.append(bpt)
                if not values:
                    log(f"  {host:8s} {mib:>4} MiB {direction}: no measurable transactions")
                    continue
                med, spread, levels = _summarise(values)
                entries.append({"host": host, "direction": direction, "mib": mib,
                                "bytes_per_txn": round(med, 1), "spread_pct": round(spread, 2),
                                "runs": len(values), "unstable": bool(levels), "levels": levels})
                flag = f"  UNSTABLE: levels {levels}" if levels else ""
                log(f"  {host:8s} {mib:>4} MiB {direction}: {med:7.0f} B/transaction "
                    f"(spread {spread:.1f}%){flag}")

    rates = _background(counter)
    counter.close()
    stamp = time.strftime("%Y-%m-%dT%H:%M", time.localtime())
    data = {
        "schema": SCHEMA,
        "calibration_id": f"pravaha-calibrate_{stamp}",
        "method": METHOD,
        "pravaha_version": __version__,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "identity": identity.current(unit=unit),
        "repeats": repeats,
        "background": {"rate": round(statistics.median(rates), 1),   # median: robust to bursts
                       "sd": round(statistics.stdev(rates), 1)},
        "entries": entries,
    }
    return Calibration(data), rows


def _background(counter, runs=5, seconds=2.0):
    """Idle background, measured once more at the end (transactions per second)."""
    rates = []
    for _ in range(runs):
        counter.start()
        time.sleep(seconds)
        counter.stop()
        rates.append(counter.read() / seconds)
    return rates
