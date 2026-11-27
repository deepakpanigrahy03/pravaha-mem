"""GPU test (marker gpu): a tiny schedule on the real GPU, exact checks only (no counter needed).

    python -m pytest -m gpu tests
"""
import json
import subprocess
import sys

import pytest

pytestmark = pytest.mark.gpu

SCHED = {"slots": [
    {"name": "idle", "kind": "idle", "seconds": 0.2},
    {"name": "read_resident", "kind": "kernel", "op": "read", "footprint_mib": 8, "logical_mib": 64},
    {"name": "read_streaming", "kind": "kernel", "op": "read", "footprint_mib": 256, "logical_mib": 512},
    {"name": "write", "kind": "kernel", "op": "write", "footprint_mib": 64, "logical_mib": 128},
    {"name": "copy", "kind": "copy", "direction": "h2d", "host": "pinned", "size_mib": 16, "count": 4},
    {"name": "copy_pageable", "kind": "copy", "direction": "d2h", "host": "pageable", "size_mib": 16, "count": 2},
    {"name": "compute", "kind": "compute", "iters": 1000, "launches": 4},
    {"name": "overlap", "kind": "overlap", "compute": {"iters": 1000, "launches": 2},
     "transfer": {"kind": "copy", "direction": "h2d", "host": "pinned", "size_mib": 16, "count": 2}},
    {"name": "square", "kind": "square", "period_s": 0.4, "periods": 2,
     "on": {"kind": "copy", "direction": "h2d", "host": "pinned", "size_mib": 16, "pace_ms": 50},
     "off": {"kind": "noop", "pace_ms": 50}},
]}
MIB = 1 << 20


def test_schedule_runs_on_the_gpu(tmp_path):
    from pravaha_mem.schedule import parse
    f = tmp_path / "s.json"
    f.write_text(json.dumps(SCHED))
    p = subprocess.run([sys.executable, "-m", "pravaha_mem.schedule", str(f)], capture_output=True, text=True,
                       timeout=120)
    assert p.returncode == 0, p.stderr[-300:]
    out = parse(p.stdout.splitlines())
    s = {x["name"]: x for x in out["slots"]}
    assert out["done"]["checks_ok"] is True
    assert s["read_resident"]["ops"] == 8 and s["read_resident"]["check"]["ok"]
    assert s["read_streaming"]["payload_bytes"] == 512 * MIB and s["read_streaming"]["check"]["ok"]
    assert s["write"]["payload_bytes"] == 128 * MIB
    assert s["copy"]["payload_bytes"] == 64 * MIB and s["copy_pageable"]["payload_bytes"] == 32 * MIB
    assert s["overlap"]["detail"] == {"compute_launches": 2, "transfer_ops": 2}
    assert [h["half"] for h in out["halves"]] == ["on", "off", "on", "off"]
    assert all(x["t1"] >= x["t0"] for x in out["slots"]) and out["setup"]["t1"] <= out["slots"][0]["t0"]
