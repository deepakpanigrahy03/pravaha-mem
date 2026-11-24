"""1.1.0 tests without a GPU: known_kernel parsing and arguments, kernel pass hardening."""
import subprocess
import types

import pytest

from pravaha_mem import kernels as k
from pravaha_mem import known_kernel as kk

MIB16 = 16 * 2 ** 20


def test_parse_machine_lines():
    r = kk.parse(['ATTRS {"when": "before", "managed_memory": 1}', "FIRST 10 20", "WINDOW 10 50",
                  'ATTRS {"when": "after", "managed_memory": 1}', 'CHECK ok {"ok": true, "logical_bytes": 5}'])
    assert (r["t0"], r["t1"], r["first_t1"]) == (10, 50, 20)
    assert r["attrs_before"] == {"managed_memory": 1} and r["check"]["logical_bytes"] == 5


def test_expected_checksum():
    n_vec = MIB16 // 16
    assert kk.expected_checksum("read", 3, n_vec) == (3 * n_vec * 4 * kk.READ_VALUE) % (1 << 64)
    assert kk.expected_checksum("write", 3, n_vec) == (n_vec * 4 * kk.WRITE_VALUE) % (1 << 64)


@pytest.mark.parametrize("argv", [["device", "read", "0", "16"], ["device", "read", "1", "16", "--prefetch", "device"],
                                  ["pinned", "read", "2", "16", "--hold", "1"], ["device", "read", "1", "0"]])
def test_bad_arguments_rejected(argv):
    with pytest.raises(SystemExit):
        kk._args(argv)


def test_support_shape():
    s = kk.support({"managed_memory": 0, "pageable_memory_access": 1})
    assert s["managed"] == {"ok": False, "reason": "managedMemory attribute is 0"}
    assert s["pageable"]["ok"] and set(s) == set(kk.PLACEMENTS)


def _csv(rows):
    out = ['"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"']
    for kid, name, metric, val in rows:
        out.append('"%s","%s","%s","sector","%s"' % (kid, name, metric, val))
    return "\n".join(out) + "\n"


def _fake(monkeypatch, stdout="", stderr="", rc=0, timeout=False):
    seen = {}

    def run(args, **kw):
        seen["args"] = args
        if timeout:
            raise subprocess.TimeoutExpired(args, 1)
        return types.SimpleNamespace(returncode=rc, stdout=stdout, stderr=stderr)
    monkeypatch.setattr(k.subprocess, "run", run)
    monkeypatch.setattr(k, "ncu_version", lambda ncu=None: "v")
    return seen


def test_missing_read_counter_is_none_write_kept(monkeypatch):
    _fake(monkeypatch, _csv([("0", "a", k.READ, "n/a"), ("0", "a", k.WRITE, "10")]))
    r = k.kernel_traffic(["x"], ncu="/ncu")
    assert r.read_bytes is None and r.write_bytes == 320 and r.kernels[0]["missing"] == [k.READ]


def test_name_base_and_timeout_and_errors(monkeypatch):
    seen = _fake(monkeypatch, _csv([("0", "a", k.READ, "1"), ("0", "a", k.WRITE, "1")]))
    k.kernel_traffic(["x"], ncu="/ncu", match="MulFunctor")
    a = seen["args"]
    assert a[a.index("--kernel-name-base") + 1] == "demangled" and "regex:MulFunctor" in a
    _fake(monkeypatch, timeout=True)
    with pytest.raises(TimeoutError):
        k.kernel_traffic(["x"], ncu="/ncu", timeout=1)
    _fake(monkeypatch, "", "==ERROR== ERR_NVGPUCTRPERM")
    with pytest.raises(k.NoPermission):
        k.kernel_traffic(["x"], ncu="/ncu")
    _fake(monkeypatch, "", "")
    with pytest.raises(k.NoKernelMatched):
        k.kernel_traffic(["x"], ncu="/ncu", match="zzz")
    with pytest.raises(k.NoKernelData):
        k.kernel_traffic(["x"], ncu="/ncu")
    _fake(monkeypatch, _csv([("0", "a", k.READ, "1"), ("0", "a", k.WRITE, "1")]), "boom", rc=3)
    with pytest.raises(k.WorkloadFailed) as e:
        k.kernel_traffic(["x"], ncu="/ncu")
    assert e.value.returncode == 3 and e.value.result.write_bytes == 32


def test_probe_never_raises(monkeypatch):
    monkeypatch.setattr(k, "find_ncu", lambda: (_ for _ in ()).throw(FileNotFoundError("ncu not found")))
    assert k.probe()["ok"] is False


def test_thermal_limits_layers(tmp_path):
    from pravaha_mem import thermal_limits as t
    r = t.resolve(env={})
    assert (r["start_c"], r["abort_c"], r["error"]) == (10.0, 5.0, None)
    assert r["source"]["start_c"].endswith("pravaha.ini")
    cfg = tmp_path / "c.ini"
    cfg.write_text("[thermal]\nstart_c = 8\n")
    r = t.resolve(env={"PRAVAHA_CONFIG": str(cfg), "PRAVAHA_MIN_HEADROOM_ABORT_C": "4"})
    assert r["start_c"] == 8.0 and r["source"]["start_c"] == "config:" + str(cfg)
    assert r["abort_c"] == 4.0 and r["source"]["abort_c"] == "env:PRAVAHA_MIN_HEADROOM_ABORT_C"
    assert t.resolve(start_c=12, env={})["source"]["start_c"] == "argument"
    assert t.resolve(start_c=3, env={})["error"]
    with pytest.raises(RuntimeError):
        t.resolve(env={"PRAVAHA_CONFIG": str(tmp_path / "missing.ini")})
