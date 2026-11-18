"""Measure GPU copy-engine traffic, and say exactly what was measured.

Author: Deepak Panigrahy
Part of pravaha (GB10 GPU memory-traffic measurement). License: MIT.

    import pravaha_mem as pravaha
    with pravaha.measure() as m:
        run_my_code()
    print(m.summary())
    m.estimate(profile={(256, "h2d", False): 8})     # bytes, if you know your copies

What the numbers mean (printed with every result)
-------------------------------------------------
* Scope: SYSTEM-WIDE. The SMMU counter sees every copy through the GPU's SMMU,
  from any process, during the measurement window. It cannot tell which
  process made a copy. "Process attribution: not available."
* raw_count: what the hardware counted (unsigned, never adjusted).
* background: idle transactions per second x window length, from the
  calibration (measured on this machine). Subtracted, not hidden.
* net_txn: raw_count - background. Signed: it can be negative when the window
  is short and quiet. It is never clamped to zero.
* bytes: only with a calibration of this machine; exact with a known transfer
  profile, otherwise a labelled range (see calib.Calibration.estimate).
* status: "complete", or "partial: <reason>" if the measured code failed.
"""

import time

from . import config, discover
from .calib import Calibration, CalibrationMissing
from .perf import Counter


def default_unit(sysfs="/sys"):
    """The counter unit to use: PRAVAHA_UNIT, the saved choice, or the only candidate.

    With several candidates and no confirmed choice we refuse to guess: on the
    tested GB10 seven units share the GPU's SMMU and only one counts copies.
    """
    unit = config.saved_unit()
    if unit:
        return unit
    _, cands = discover.candidates(sysfs=sysfs)
    if len(cands) == 1:
        return cands[0]
    raise discover.DiscoveryError(
        f"{len(cands)} candidate units ({', '.join(cands)}). "
        "Run 'pravaha setup' once to confirm which one counts copies, or set PRAVAHA_UNIT.")


def _load_calibration():
    """The machine's calibration, or None (raw counts are still reported)."""
    try:
        return Calibration.load()
    except CalibrationMissing:
        return None


class Measurement:
    """Counts the GPU's copy-engine transactions between start() and stop()."""

    window = "between start() and stop()"

    def __init__(self, unit=None, sysfs="/sys", background="live"):
        """background="live": measure the idle rate just before and just after the window
        (about 0.6 s each); background="calibration": use the rate saved by `pravaha setup`."""
        self.unit = unit or default_unit(sysfs)
        self.calibration = _load_calibration()
        self._counter = Counter(self.unit, "transaction", sysfs)
        self.start_sample = self.stop_sample = None
        self.status = "not started"
        self.background_mode = background
        self.bg_rate = self.bg_rate_after = None

    def _rate(self, windows=3, seconds=0.2):
        """Idle transactions per second now: median of short windows, so one burst cannot dominate."""
        import time as _time
        rates = []
        for _ in range(windows):
            self._counter.start()
            _time.sleep(seconds)
            self._counter.stop()
            rates.append(self._counter.read() / seconds)
        return sorted(rates)[len(rates) // 2]

    # ---- running --------------------------------------------------------------
    def start(self):
        if self.background_mode == "live":
            self.bg_rate = self._rate()
        elif self.calibration:
            self.bg_rate = self.calibration.idle_rate
        self._counter.start()
        self.start_sample = self._counter.read_raw()
        self.status = "running"
        return self

    def stop(self, status="complete"):
        """Stop counting. `status` labels the window (e.g. 'partial: exit code 1')."""
        self._counter.stop()
        self.stop_sample = self._counter.read_raw()
        if self.background_mode == "live":
            self.bg_rate_after = self._rate()
        self.status = status
        return self

    def close(self):
        self._counter.close()

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        # stop even if the measured code raised; label the window as partial
        status = "complete" if exc_type is None else f"partial: {exc_type.__name__}"
        self.stop(status)
        self.close()
        return False          # never swallow the caller's exception

    # ---- results ----------------------------------------------------------------
    @property
    def raw_count(self):
        """Transactions counted in the window (unsigned, as read)."""
        return self.stop_sample.count - self.start_sample.count

    @property
    def seconds(self):
        return (self.stop_sample.t_mono_ns - self.start_sample.t_mono_ns) / 1e9

    @property
    def background(self):
        """Expected idle transactions over the window, or None if no rate is known."""
        if self.bg_rate is None:
            return None
        return self.bg_rate * self.seconds

    @property
    def disturbed(self):
        """True if the idle rate after the window differs from before by more than a factor 2
        (rates under 1,000/s count as 1,000/s, so tiny changes do not trigger it)."""
        a, b = self.bg_rate, self.bg_rate_after
        if a is None or b is None:
            return False
        return max(a, b, 1000.0) > 2 * max(min(a, b), 1000.0)

    @property
    def net_txn(self):
        """raw_count - background (signed, never clamped); None without a calibration."""
        bg = self.background
        return None if bg is None else self.raw_count - bg

    def estimate(self, profile=None):
        """Bytes behind the window; raises CalibrationMissing without a calibration."""
        if not self.calibration:
            raise CalibrationMissing("no calibration on this machine; run 'pravaha setup'")
        return self.calibration.estimate(self.net_txn, profile)

    def to_dict(self, profile=None):
        """Everything needed to interpret the result later (for logs and reports)."""
        d = {"unit": self.unit, "scope": "system_wide", "window": self.window,
             "process_attribution": False,
             "status": self.status, "raw_count": self.raw_count, "seconds": self.seconds,
             "t_start_mono_ns": self.start_sample.t_mono_ns,
             "t_start_utc_ns": self.start_sample.t_utc_ns,
             "background_txn": self.background, "net_txn": self.net_txn,
             "background_source": self.background_mode,
             "background_rate_before": self.bg_rate, "background_rate_after": self.bg_rate_after,
             "disturbed": self.disturbed,
             "calibration_id": None, "calibration_stale": None, "estimate": None,
             "estimate_unavailable_reason": None}
        if not self.calibration:
            d["estimate_unavailable_reason"] = "no confirmed calibration"
            return d
        d["calibration_id"] = self.calibration.calibration_id
        d["calibration_stale"] = self.calibration.stale()
        d["estimate"] = self.estimate(profile)._asdict()
        return d

    def summary(self, profile=None):
        """Human-readable block, scope stated every time."""
        d = self.to_dict(profile)
        lines = ["GPU copy-engine traffic",
                 f"  raw transactions     {d['raw_count']:,} in {d['seconds']:.2f} s ({self.unit})"]
        if d["estimate"] is None:
            lines.append(f"  byte estimate        unavailable ({d['estimate_unavailable_reason']})")
        else:
            e = d["estimate"]
            if e["bytes"] is None:
                lines.append(f"  byte estimate        {e['low'] / 1e9:.2f} to {e['high'] / 1e9:.2f} GB "
                             "(bound: copy sizes unknown)")
            else:
                lines.append(f"  byte estimate        {e['bytes'] / 1e9:.2f} GB "
                             f"({e['low'] / 1e9:.2f} to {e['high'] / 1e9:.2f}, known profile)")
            src = ("measured just before" if self.background_mode == "live"
                   else "from the calibration")
            lines.append(f"  background           {d['background_txn']:,.0f} transactions subtracted "
                         f"({self.bg_rate:,.0f}/s, {src})")
            if self.disturbed:
                lines.append(f"  ! disturbed          idle rate changed from {self.bg_rate:,.0f}/s to "
                             f"{self.bg_rate_after:,.0f}/s: another program used the GPU; "
                             "measure again on a quiet GPU")
            stale = f" STALE: {', '.join(d['calibration_stale'])} changed" if d["calibration_stale"] else ""
            lines.append(f"  calibration          {d['calibration_id']}{stale}")
        lines += [f"  window               {self.window}",
                  "  scope                system-wide window",
                  "  process attribution  not available",
                  f"  status               {d['status']}"]
        return "\n".join(lines)


def measure(unit=None, background="live"):
    """Shortcut: `with pravaha.measure() as m: ...`."""
    return Measurement(unit, background=background)


def watch(interval_ms=100, unit=None):
    """Yield (RawSample, delta) every `interval_ms`, until the caller stops iterating.

    delta is the raw transaction count in that interval (unsigned). Use it beside
    another sampler (power, timing); convert in analysis with the calibration.
    """
    c = Counter(unit or default_unit())
    c.start()
    try:
        last = c.read_raw()
        next_tick = time.monotonic()
        while True:
            next_tick += interval_ms / 1000.0
            time.sleep(max(0.0, next_tick - time.monotonic()))
            now = c.read_raw()
            yield now, now.count - last.count
            last = now
    finally:
        c.close()
