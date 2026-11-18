"""Tests that need no GPU: a fake reader speaks protocol v2 and the observer must
turn it into the right bytes, phases and quality figures. Run: python -m pytest tests"""
import os
import stat
import sys
import textwrap
import time

import pytest

FAKE = textwrap.dedent('''\
    #!{py}
    import signal, sys, time
    stop = False
    def h(*_):
        global stop; stop = True
    signal.signal(signal.SIGINT, h)
    print("V 2"); print("M a b c d e"); print(f"C 1 {{time.time_ns()}} 1", flush=True)
    i = 0
    while not stop:
        t0 = time.time_ns(); time.sleep(0.01); t1 = time.time_ns()
        # per sample: 1000 copy sectors, 100 read miss, 10 write, 200 all reads, 20 all writes
        print(f"S {{i}} {{t0}} {{t1}} 1000 100 10 200 20", flush=True); i += 1
    print("E ok", i, flush=True)
''')


@pytest.fixture
def fake_reader(tmp_path, monkeypatch):
    p = tmp_path / "pravaha-reader"
    p.write_text(FAKE.format(py=sys.executable))
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PRAVAHA_READER", str(p))
    return p


def test_bytes_and_phases(fake_reader):
    import pravaha_mem.observer as ob
    ob._last_stop = 0
    with ob.observe() as m:
        m.mark("a"); time.sleep(0.3)
        m.mark("b"); time.sleep(0.3)
    t = m.total()
    n = t["all_read"] / (200 * 32)                       # samples' worth inside the window
    assert n > 30
    assert t["copy"] == pytest.approx(n * 1000 * 32 / 2, rel=1e-6)   # x16 conversion
    assert t["kernel_read"] == pytest.approx(n * 100 * 32, rel=1e-6)
    assert t["memory"] == pytest.approx(t["all_read"] + t["all_write"])
    p = m.phases()
    assert set(p) == {"a", "b"}
    assert p["a"]["copy"] + p["b"]["copy"] == pytest.approx(t["copy"], rel=0.02)
    q = m.quality()
    assert q["samples_received"] > 30 and q["clock_pairs"] >= 1
    assert q["reader_exit"].startswith("E ok")


def test_json_has_provenance_and_scope(fake_reader, tmp_path):
    import json
    import pravaha_mem.observer as ob
    ob._last_stop = 0
    with ob.observe() as m:
        m.metadata({"workload": "test"}); time.sleep(0.2)
    m.to_json(tmp_path / "r.json")
    d = json.load(open(tmp_path / "r.json"))
    assert d["scope"].startswith("device wide")
    assert d["metrics"]["copy_bytes"]["conversion"].startswith("sectors x 32 / 2")
    assert d["metadata"] == {"workload": "test"}
    assert "provenance" in d and "quality" in d


def test_missing_reader(monkeypatch):
    import pravaha_mem.observer as ob
    monkeypatch.setenv("PRAVAHA_READER", "/nonexistent/reader")
    monkeypatch.setattr(ob, "LIBEXEC", "/nonexistent")
    monkeypatch.setattr(ob.shutil, "which", lambda *_: None)
    with pytest.raises(FileNotFoundError):
        ob.find_reader()
