"""GPU tests: the known_bytes checks, automated. Light work only (copies and simple writes
and reads of a few GiB). Skipped when pravaha's reader is not installed.

Run on a GB10:  python -m pytest -m gpu tests
"""
import time

import pytest

pytestmark = pytest.mark.gpu

GIB = 2 ** 30


def _ready():
    try:
        from pravaha_mem.observer import find_reader
        find_reader()
        return True
    except Exception:
        return False


skip = pytest.mark.skipif(not _ready(), reason="pravaha reader not installed (pravaha install)")


def _measure(direction, times, mib):
    """Bytes in the work window, minus the background rate measured for 1 s just before
    (the desktop and other programs also use the GPU), as pravaha verify does."""
    import os, subprocess, sys
    from pravaha_mem.observer import observe
    with observe() as m:
        b0 = time.time_ns(); time.sleep(1.0); b1 = time.time_ns()
        out = subprocess.run([sys.executable, "-m", "pravaha_mem.known_bytes", direction, str(times), str(mib)],
                             capture_output=True, text=True, env={**os.environ, "PRAVAHA_KNOWN_WINDOW": "1"})
    assert out.returncode == 0, out.stderr
    w = [l.split() for l in out.stdout.splitlines() if l.startswith("WINDOW")][0]
    t0, t1 = int(w[1]) - 40_000_000, int(w[2]) + 40_000_000
    q, s = m._sum(b0, b1), m._sum(t0, t1)
    k = s["seconds"] / q["seconds"]
    time.sleep(2)                                   # cool down between tests
    return {f: s[f] - q[f] * k for f in ("copy", "kernel_read", "kernel_write", "memory")}


@skip
@pytest.mark.parametrize("times,mib", [(8, 256), (4, 512), (16, 128)])
def test_copy_into_gpu_counts_bytes_not_copies(times, mib):
    s = _measure("h2d", times, mib)
    assert s["copy"] / (2 * GIB) == pytest.approx(1.0, abs=0.002)


@skip
def test_copy_out_of_gpu():
    s = _measure("d2h", 8, 256)
    assert s["copy"] / (2 * GIB) == pytest.approx(1.0, abs=0.002)


@skip
def test_gpu_code_writes():
    s = _measure("memset", 16, 256)
    assert s["kernel_write"] / (4 * GIB) == pytest.approx(1.0, abs=0.02)
    assert s["copy"] < 0.01 * GIB                  # no copy engine work


@skip
def test_gpu_code_reads_and_crosses_twice():
    s = _measure("d2d", 16, 256)
    assert s["kernel_read"] / (4 * GIB) == pytest.approx(1.0, abs=0.03)
    assert s["memory"] / (8 * GIB) == pytest.approx(1.0, abs=0.05)   # read 4 + write 4


@skip
def test_status_ready():
    from pravaha_mem.status import counters_free
    assert counters_free() == "free"
