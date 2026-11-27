"""schedule runner: validation, timing records and payload accounting with a fake backend (no GPU)."""
import json

import pytest

from pravaha_mem import schedule as sch

MIB = 1 << 20


class FakeBackend:
    """Records calls; payload sizes follow the real backend's rules."""

    def __init__(self, bad_checksum=False):
        self.calls, self.prepared, self.bad = [], [], bad_checksum
        self.reads = {}

    def prepare(self, part):
        self.prepared.append(part["kind"])

    def kernel_pass(self, part, stream=0):
        self.calls.append(("kernel", part["op"], part["footprint_mib"], stream))
        key = (part.get("placement", "device"), part["footprint_mib"])
        self.reads[key] = self.reads.get(key, 0) + (1 if part["op"] == "read" else 0)
        return part["footprint_mib"] * MIB

    def copy_once(self, part, stream=0):
        self.calls.append(("copy", part["direction"], part["host"], stream))
        return part["size_mib"] * MIB

    def compute_once(self, part, stream=0):
        self.calls.append(("compute", part["iters"], stream))
        return 0

    def noop(self, stream=0):
        self.calls.append(("noop",))
        return 0

    def sync(self):
        self.calls.append(("sync",))

    def read_checksum(self, part):
        key = (part.get("placement", "device"), part["footprint_mib"])
        n_vec = part["footprint_mib"] * MIB // 16
        passes = self.reads.pop(key, 0)
        got = sch._expected_read(n_vec, passes) + (1 if self.bad else 0)
        return got, n_vec

    def attributes(self):
        return {"blocks": 2, "threads": 4, "arch": "fake", "runtime_version": 0}


class Clock:
    """Deterministic clock: sleep advances time; every GPU call costs a little."""

    def __init__(self):
        self.t = 1000.0

    def mono(self):
        self.t += 1e-4
        return self.t

    def sleep(self, s):
        self.t += s

    def ns(self):
        return int(self.mono() * 1e9)


def run(slots, be=None):
    be, ck, lines = be or FakeBackend(), Clock(), []
    recs = sch.run_schedule({"slots": slots}, be, emit=lines.append, clock=ck.mono, sleep=ck.sleep, now_ns=ck.ns)
    return recs, sch.parse(lines), be


def test_validation_rejects_bad_schedules():
    bad = [
        {"slots": []},
        {"slots": [{"kind": "warp"}]},
        {"slots": [{"kind": "kernel", "op": "read", "footprint_mib": 16, "logical_mib": 40}]},
        {"slots": [{"kind": "copy", "direction": "x", "host": "pinned", "size_mib": 16, "count": 1}]},
        {"slots": [{"kind": "square", "period_s": 2, "periods": 1, "on": {"kind": "copy", "direction": "h2d",
                    "host": "pinned", "size_mib": 16}, "off": {"kind": "idle"}}]},
        {"slots": [{"kind": "square", "period_s": 2, "periods": 1, "on": {"kind": "noop", "pace_ms": 10},
                    "off": {"kind": "kernel", "op": "read", "footprint_mib": 8}}]},
    ]
    for b in bad:
        with pytest.raises(sch.ScheduleError):
            sch.validate(b)


def test_resident_and_streaming_issue_the_same_logical_bytes():
    recs, out, _ = run([
        {"name": "resident", "kind": "kernel", "op": "read", "footprint_mib": 8, "logical_mib": 1024},
        {"name": "streaming", "kind": "kernel", "op": "read", "footprint_mib": 1024, "logical_mib": 1024},
    ])
    assert [r["ops"] for r in recs] == [128, 1]
    assert recs[0]["payload_bytes"] == recs[1]["payload_bytes"] == 1024 * MIB
    assert all(r["check"]["ok"] for r in recs)
    assert out["done"]["checks_ok"] is True and len(out["slots"]) == 2


def test_failed_checksum_is_reported():
    recs, out, _ = run([{"kind": "kernel", "op": "read", "footprint_mib": 8, "logical_mib": 16}],
                       FakeBackend(bad_checksum=True))
    assert recs[0]["check"]["ok"] is False and out["done"]["checks_ok"] is False


def test_copy_count_and_paced_copy():
    recs, _, be = run([
        {"kind": "copy", "direction": "h2d", "host": "pinned", "size_mib": 16, "count": 4},
        {"kind": "copy", "direction": "d2h", "host": "pageable", "size_mib": 16, "pace_ms": 100, "seconds": 1.0},
    ])
    assert recs[0]["payload_bytes"] == 64 * MIB
    assert 9 <= recs[1]["ops"] <= 11 and recs[1]["payload_bytes"] == recs[1]["ops"] * 16 * MIB
    assert ("copy", "d2h", "pageable", 0) in be.calls


def test_noop_sham_moves_nothing_but_paces_like_copies():
    recs, _, be = run([{"kind": "noop", "pace_ms": 100, "seconds": 1.0}])
    assert recs[0]["payload_bytes"] == 0 and 9 <= recs[0]["ops"] <= 11
    assert sum(1 for c in be.calls if c == ("noop",)) == recs[0]["ops"]


def test_overlap_issues_both_parts_on_two_streams():
    recs, _, be = run([{"kind": "overlap", "compute": {"iters": 1000, "launches": 3},
                        "transfer": {"kind": "copy", "direction": "h2d", "host": "pinned", "size_mib": 16, "count": 5}}])
    assert recs[0]["detail"] == {"compute_launches": 3, "transfer_ops": 5}
    assert {c[2] for c in be.calls if c[0] == "compute"} == {0} and {c[3] for c in be.calls if c[0] == "copy"} == {1}
    assert recs[0]["payload_bytes"] == 80 * MIB


def test_square_wave_halves_are_timed_and_counted():
    recs, out, _ = run([{"kind": "square", "period_s": 2.0, "periods": 3,
                         "on": {"kind": "copy", "direction": "h2d", "host": "pinned", "size_mib": 16, "pace_ms": 100},
                         "off": {"kind": "noop", "pace_ms": 100}}])
    halves = out["halves"]
    assert [h["half"] for h in halves] == ["on", "off"] * 3
    assert all(0.99 <= (h["t1"] - h["t0"]) / 1e9 <= 1.05 for h in halves)
    assert all(h["payload_bytes"] == 0 for h in halves if h["half"] == "off")
    assert sum(h["payload_bytes"] for h in halves) == recs[0]["payload_bytes"] > 0


def test_setup_prepares_every_part_before_the_first_slot():
    _, out, be = run([{"kind": "idle", "seconds": 0.5},
                      {"kind": "overlap", "compute": {"iters": 10, "launches": 1},
                       "transfer": {"kind": "kernel", "op": "read", "footprint_mib": 8, "logical_mib": 8}}])
    assert be.prepared[:3] == ["idle", "compute", "kernel"]
    assert out["setup"]["t1"] <= out["slots"][0]["t0"]


def test_parse_roundtrip():
    lines = ['SETUP {"t0": 1, "t1": 2, "attributes": {}}', 'SLOT {"index": 0}', 'HALF {"slot": 0}', "noise",
             'DONE {"slots": 1}']
    p = sch.parse(lines)
    assert p["setup"]["t1"] == 2 and p["slots"] == [{"index": 0}] and p["halves"] == [{"slot": 0}]
    assert json.dumps(p["done"]) == '{"slots": 1}'


def test_table_is_readable():
    _, out, _ = run([{"name": "resident", "kind": "kernel", "op": "read", "footprint_mib": 8, "logical_mib": 64},
                     {"kind": "square", "period_s": 2.0, "periods": 1,
                      "on": {"kind": "copy", "direction": "h2d", "host": "pinned", "size_mib": 16, "pace_ms": 100},
                      "off": {"kind": "idle"}}])
    t = sch.table(out)
    assert "resident" in t and "ok" in t and "square wave: 2 halves (1 on)" in t and "checks ok" in t
