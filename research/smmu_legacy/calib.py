"""Turning SMMU transactions into bytes, honestly.

Author: Deepak Panigrahy
Part of pravaha (GB10 GPU memory-traffic measurement). License: MIT.

Why this module exists
----------------------
The SMMU counter counts *transactions*, not bytes, and one transaction does
not stand for a fixed number of bytes: on the tested GB10 it ranged from about
2.1 KB (large pinned copies) to about 3.8 KB (copies of a few MiB, pageable
copies), and at some sizes it was not stable. So:

* a calibration is a table, one entry per (host memory type, direction, copy size);
* bytes can be computed only with a calibration measured on THIS machine
  (no fallback to published numbers);
* if the caller knows its copies (sizes and directions), we convert exactly
  ("known profile"); if not, we give a labelled range ("bound").

The calibration file (calibration.json, schema 2)
------------------------------------------------
    {
      "schema": 2,
      "calibration_id": "pravaha-calibrate_2026-11-05T04:51",
      "method": "inprocess-1",
      "pravaha_version": "0.3.0",
      "created_utc": "2026-11-05T10:51:00Z",
      "identity": {...},                 # see identity.py
      "background": {"rate": 3625.4, "sd": 1768.1},
      "entries": [
        {"host": "pinned", "direction": "h2d", "mib": 256,
         "bytes_per_txn": 2089.3, "spread_pct": 0.1, "runs": 5,
         "unstable": false, "levels": []},
        ...
      ]
    }

The same numbers are also written to calibration.tsv (one entry per line), which
the C library reads, so C and Python convert identically.
"""

import json
import math
import os
from collections import namedtuple

from . import config, identity

SCHEMA = 2
DIRECTIONS = ("h2d", "d2h")        # host-to-device, device-to-host
HOSTS = ("pinned", "pageable", "async")   # how the copy was made: pinned, pageable, or pinned async on 2 streams


def _host(pageable):
    """Profile flag -> table key: False pinned, True pageable, "async" pinned async copies."""
    return "async" if pageable == "async" else ("pageable" if pageable else "pinned")

# The result of a byte conversion. `bytes` is None in bound mode.
Estimate = namedtuple("Estimate", "bytes low high mode calibration_id notes")


class CalibrationMissing(RuntimeError):
    """No calibration for this machine: bytes cannot be estimated."""


def _boot_id():
    """Linux's id of the current boot ('' if unreadable): tells campaigns after a reboot apart."""
    try:
        with open("/proc/sys/kernel/random/boot_id") as f:
            return f.read().strip()
    except OSError:
        return ""


def _log2(x):
    return math.log2(max(float(x), 1e-9))


class Calibration:
    """A measured calibration table plus where and how it was measured."""

    def __init__(self, data, path=None):
        if data.get("schema") != SCHEMA:
            raise CalibrationMissing(
                f"calibration file has schema {data.get('schema')}, expected {SCHEMA}; "
                "run 'pravaha setup' again")
        self.data = data
        self.path = path
        self.calibration_id = data["calibration_id"]
        self.method = data.get("method", "unknown")
        self.identity = data.get("identity", {})
        bg = data.get("background", {})
        self.idle_rate = float(bg.get("rate", 0.0))
        self.idle_rate_sd = float(bg.get("sd", 0.0))
        # entries indexed by (host, direction) -> sorted list of entry dicts
        self.table = {}
        for e in data["entries"]:
            self.table.setdefault((e["host"], e["direction"]), []).append(e)
        for k in self.table:
            self.table[k].sort(key=lambda e: e["mib"])

    # ---- loading and saving ---------------------------------------------------
    @classmethod
    def load(cls, path=None):
        """The calibration of this machine; raises CalibrationMissing if there is none."""
        path = path or config.find_file("calibration.json")
        if not path:
            raise CalibrationMissing("no calibration on this machine; run 'pravaha setup'")
        with open(path) as f:
            return cls(json.load(f), path)

    def save(self, system=False):
        """Write calibration.json and calibration.tsv (for the C library)."""
        d = config.write_dir(system)
        self.data.setdefault("boot_id", _boot_id())     # N3 needs campaigns from different boots
        path = os.path.join(d, "calibration.json")
        with open(path, "w") as f:
            json.dump(self.data, f, indent=2)
            f.write("\n")
        with open(os.path.join(d, "calibration.tsv"), "w") as f:
            # line 1: id and background; then one line per entry
            f.write(f"#id\t{self.calibration_id}\t{self.idle_rate:.3f}\t{self.idle_rate_sd:.3f}\n")
            for e in self.data["entries"]:
                f.write(f"{e['host']}\t{e['direction']}\t{e['mib']}\t{e['bytes_per_txn']:.3f}"
                        f"\t{e['spread_pct']:.3f}\t{int(bool(e.get('unstable')))}\n")
        # every campaign is also kept in history/, so repeatability (N3) can compare them
        hist = os.path.join(d, "history")
        os.makedirs(hist, exist_ok=True)
        with open(os.path.join(hist, self.calibration_id.replace(":", "-") + ".json"), "w") as f:
            json.dump(self.data, f, indent=2)
        self.path = path
        return path

    @classmethod
    def history(cls, path=None):
        """Every saved campaign on this machine, oldest first (for the N3 check)."""
        d = os.path.dirname(path or config.find_file("calibration.json") or "")
        hist = os.path.join(d, "history")
        if not d or not os.path.isdir(hist):
            return []
        out = []
        for name in sorted(os.listdir(hist)):
            with open(os.path.join(hist, name)) as f:
                out.append(cls(json.load(f), os.path.join(hist, name)))
        return out

    # ---- validity ---------------------------------------------------------------
    def stale(self, now=None):
        """List of identity fields that changed since calibration (empty = usable)."""
        now = now or identity.current(unit=self.identity.get("unit"))
        return identity.differences(self.identity, now)

    # ---- lookups ----------------------------------------------------------------
    def _entries(self, host, direction):
        rows = self.table.get((host, direction))
        if not rows:
            raise CalibrationMissing(f"no calibration for {host} {direction} copies")
        return rows

    def bytes_per_txn(self, copy_mib, direction="h2d", pageable=False):
        """Bytes per transaction for copies of `copy_mib` MiB.

        Between measured sizes: interpolated on a log scale.
        Outside them: the nearest measured size (flagged by `covered`).
        """
        rows = self._entries(_host(pageable), direction)
        x = _log2(copy_mib)
        if x <= _log2(rows[0]["mib"]):
            return rows[0]["bytes_per_txn"]
        if x >= _log2(rows[-1]["mib"]):
            return rows[-1]["bytes_per_txn"]
        for a, b in zip(rows, rows[1:]):
            la, lb = _log2(a["mib"]), _log2(b["mib"])
            if la <= x <= lb:
                w = (x - la) / (lb - la)
                return a["bytes_per_txn"] + w * (b["bytes_per_txn"] - a["bytes_per_txn"])
        return rows[-1]["bytes_per_txn"]  # not reached

    def spread_pct(self, copy_mib, direction="h2d", pageable=False):
        """Repeat spread (%) of the nearest measured size."""
        rows = self._entries(_host(pageable), direction)
        near = min(rows, key=lambda e: abs(_log2(e["mib"]) - _log2(copy_mib)))
        return near["spread_pct"]

    def covered(self, copy_mib, direction="h2d", pageable=False):
        """True if the size lies within the measured range for that direction and memory."""
        rows = self._entries(_host(pageable), direction)
        return rows[0]["mib"] <= copy_mib <= rows[-1]["mib"]

    def unstable_sizes(self):
        """(host, direction, mib) of sizes whose repeats jumped between levels."""
        return [(e["host"], e["direction"], e["mib"]) for e in self.data["entries"] if e.get("unstable")]

    def bounds(self):
        """(lowest, highest) bytes per transaction over every entry."""
        values = [e["bytes_per_txn"] for e in self.data["entries"]]
        return min(values), max(values)

    # ---- conversion -------------------------------------------------------------
    def estimate(self, net_txn, profile=None):
        """Bytes behind `net_txn` (transactions after background subtraction).

        profile: what was copied, {(mib, direction, pageable): number_of_copies},
                 pageable = False (pinned), True (pageable) or "async" (pinned, cudaMemcpyAsync),
                 e.g. {(256, "h2d", False): 8, (256, "d2h", False): 8}.
                 Known profile: the copies' expected transactions give the
                 effective bytes per transaction; the result is `bytes`, with a
                 low/high from each entry's repeat spread.
                 No profile: only a bound from the lowest and highest calibrated
                 values; `bytes` is None.
        """
        notes = []
        if not profile:
            lo, hi = self.bounds()
            return Estimate(None, net_txn * lo, net_txn * hi, "bound", self.calibration_id,
                            ["copy sizes unknown: range over all calibrated sizes"])
        known_bytes = expected_txn = expected_lo = expected_hi = 0.0
        for (mib, direction, pageable), copies in profile.items():
            b = copies * mib * 2 ** 20                        # bytes this group copied
            bpt = self.bytes_per_txn(mib, direction, pageable)
            s = self.spread_pct(mib, direction, pageable) / 100.0
            known_bytes += b
            expected_txn += b / bpt                           # transactions it should cause
            expected_lo += b / (bpt * (1 + s))
            expected_hi += b / (bpt * (1 - min(s, 0.99)))
            if not self.covered(mib, direction, pageable):
                notes.append(f"{mib} MiB {direction} outside the calibrated sizes")
            if (_host(pageable), direction, mib) in self.unstable_sizes():
                notes.append(f"{mib} MiB {direction} is unstable in this calibration")
        bpt_eff = known_bytes / expected_txn                  # effective bytes per transaction
        return Estimate(net_txn * bpt_eff, net_txn * known_bytes / expected_hi,
                        net_txn * known_bytes / expected_lo, "known_profile",
                        self.calibration_id, notes)

    def to_dict(self):
        return self.data
