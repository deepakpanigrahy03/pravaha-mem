"""verify --schedule: fixture and judgement from synthetic schedule output and observer sums (no GPU)."""
import pytest

from pravaha_mem import schedule, verify_schedule as vs

MIB = 1 << 20
CFG = {"idle_s": 3, "slot_seconds": 2.0, "gap_s": 1.0, "streaming_mib": 1024, "resident_mib": 8,
       "resident_max_ratio": 0.05, "sham_pace_ms": 10, "sham_max_bytes_per_s": 2e8,
       "copy_mib": 256, "copy_tolerance": 0.05, "square_period_s": 2.0, "square_periods": 4,
       "square_pace_ms": 50, "half_tolerance": 0.05, "miss_tolerance": 0.05, "window_pad_ms": 40}
S = 1_000_000_000


def parsed(half_s=1.0, bad_check=False, off_payload=0):
    slots = [
        {"name": "idle", "kind": "idle", "t0": 0, "t1": 3 * S, "ops": 0, "payload_bytes": 0, "check": None, "detail": {}},
        {"name": "read_streaming", "kind": "kernel", "t0": 4 * S, "t1": 5 * S, "ops": 4, "payload_bytes": 4096 * MIB,
         "check": {"ok": not bad_check}, "detail": {}},
        {"name": "read_resident", "kind": "kernel", "t0": 6 * S, "t1": 7 * S, "ops": 512, "payload_bytes": 4096 * MIB,
         "check": {"ok": True}, "detail": {}},
        {"name": "sham", "kind": "noop", "t0": 8 * S, "t1": 11 * S, "ops": 300, "payload_bytes": 0, "check": None, "detail": {}},
        {"name": "copy_h2d", "kind": "copy", "t0": 12 * S, "t1": 13 * S, "ops": 8, "payload_bytes": 2048 * MIB,
         "check": None, "detail": {}},
    ]
    h = int(half_s * S)
    halves = [{"slot": 5, "period": i // 2, "half": "on" if i % 2 == 0 else "off", "t0": 14 * S + i * h,
               "t1": 14 * S + (i + 1) * h, "ops": 20, "payload_bytes": 20 * 256 * MIB if i % 2 == 0 else off_payload}
              for i in range(8)]
    return {"slots": slots, "halves": halves, "done": {"checks_ok": True}}


def make_sum(read_factor=1.0, resident_ratio=0.01, copy_factor=1.0, background=1e8):
    """Observer totals: background everywhere plus the slot's own traffic, by window."""
    def window_sum(a, b):
        sec = (b - a) / S
        tot = {"copy": background * sec * 0.1, "kernel_read": background * sec * 0.5, "memory": background * sec,
               "seconds": sec}
        if a < 5 * S and b > 4 * S:
            tot["kernel_read"] += 4096 * MIB * read_factor
        if a < 7 * S and b > 6 * S:
            tot["kernel_read"] += 4096 * MIB * resident_ratio
        if a < 13 * S and b > 12 * S:
            tot["copy"] += 2048 * MIB * copy_factor
        return tot
    return window_sum


def by_name(rep):
    return {c["check"]: c for c in rep["checks"]}


def test_fixture_is_a_valid_schedule_with_gaps():
    sl = schedule.validate(vs.fixture(CFG, {"streaming_passes": 450, "resident_passes": 280000, "copy_count": 440}))
    assert len(sl) == 11 and [s["kind"] for s in sl[1::2]] == ["idle"] * 5


def test_counts_size_each_task_to_slot_seconds():
    probe = {"slots": [{"name": "probe_streaming", "seconds": 0.0179, "ops": 4},
                       {"name": "probe_resident", "seconds": 0.0036, "ops": 512},
                       {"name": "probe_copy", "seconds": 0.018, "ops": 4}]}
    n = vs.counts(probe, CFG)
    assert n["streaming_passes"] == 447 and n["copy_count"] == 444 and 280000 <= n["resident_passes"] <= 290000
    assert vs.probe_schedule(CFG)["slots"][0]["footprint_mib"] == 1024


def test_all_pass_on_good_data():
    rep = vs.evaluate(parsed(), make_sum(), CFG)
    assert rep["pass"], [c for c in rep["checks"] if c["pass"] is not True]
    assert by_name(rep)["streaming read misses per payload byte"]["value"] == pytest.approx(1.0, abs=0.01)


def test_each_failure_is_detected():
    assert by_name(vs.evaluate(parsed(), make_sum(read_factor=0.8), CFG))["streaming read misses per payload byte"]["pass"] is False
    assert by_name(vs.evaluate(parsed(), make_sum(resident_ratio=0.2), CFG))["resident read misses per payload byte"]["pass"] is False
    assert by_name(vs.evaluate(parsed(), make_sum(copy_factor=1.2), CFG))["copy bytes per payload byte"]["pass"] is False
    assert by_name(vs.evaluate(parsed(half_s=1.2), make_sum(), CFG))["square wave half duration error"]["pass"] is False
    assert by_name(vs.evaluate(parsed(off_payload=1), make_sum(), CFG))["square wave off halves move no payload"]["pass"] is False
    assert by_name(vs.evaluate(parsed(bad_check=True), make_sum(), CFG))["checksum read_streaming"]["pass"] is False


def test_settings_require_every_key(tmp_path):
    ini = tmp_path / "p.ini"
    ini.write_text("[schedule]\nidle_s = 3\n[verify]\nmiss_tolerance = 0.05\nwindow_pad_ms = 40\n")
    with pytest.raises(KeyError):
        vs.settings([str(ini)])
