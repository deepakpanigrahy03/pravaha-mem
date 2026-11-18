"""Tests for the pravaha library (Python and C). No GPU and no perf permission needed:
everything runs against a fake sysfs tree shaped like the tested GB10, a made-up
schema-2 calibration, and a fake Nsight Compute.

Author: Deepak Panigrahy
Part of pravaha (GB10 GPU memory-traffic measurement). License: MIT.
"""
import os
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "python"))
from pravaha_mem import calib, config, discover, perf  # noqa: E402
from pravaha_mem.measure import default_unit  # noqa: E402

UNITS = ["13002", "13042", "13062", "13082", "138a2", "14902"]


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text + "\n")


@pytest.fixture
def sysfs(tmp_path):
    """Three SMMUs; the GPU (000f:01:00.0) sits behind the one at 0x13000000;
    a NIC (0000:01:00.0) is not NVIDIA; six counter units."""
    t = tmp_path / "sys"
    for s in ("0x0000000013000000", "0x0000000013800000", "0x0000000014900000"):
        d = t / "devices" / "platform" / "x" / "iommu" / f"smmu3.{s}"
        d.mkdir(parents=True)
        (t / "class" / "iommu").mkdir(parents=True, exist_ok=True)
        os.symlink(d, t / "class" / "iommu" / f"smmu3.{s}")
    gpu = t / "bus" / "pci" / "devices" / "000f:01:00.0"
    _write(str(gpu / "vendor"), "0x10de")
    _write(str(gpu / "class"), "0x030200")
    os.symlink(t / "devices/platform/x/iommu/smmu3.0x0000000013000000", gpu / "iommu")
    nic = t / "bus" / "pci" / "devices" / "0000:01:00.0"
    _write(str(nic / "vendor"), "0x14c3")
    _write(str(nic / "class"), "0x028000")
    for u in UNITS:
        d = t / "bus" / "event_source" / "devices" / f"smmuv3_pmcg_{u}"
        _write(str(d / "type"), "42")
        _write(str(d / "cpumask"), "0")
        _write(str(d / "events" / "transaction"), "event=0x01")
        _write(str(d / "format" / "event"), "config:0-15")
        _write(str(d / "format" / "filter_enable"), "config1:33")
    return str(t)


@pytest.fixture(autouse=True)
def clean_env(tmp_path, monkeypatch):
    monkeypatch.delenv("PRAVAHA_UNIT", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setenv("PRAVAHA_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


# --- discovery -----------------------------------------------------------------
def test_finds_only_the_nvidia_gpu(sysfs):
    assert discover.find_gpus(sysfs) == ["000f:01:00.0"]


def test_candidates_are_the_units_of_the_gpu_smmu(sysfs):
    bdf, units = discover.candidates(sysfs=sysfs)
    assert bdf == "000f:01:00.0"
    assert units == ["smmuv3_pmcg_13002", "smmuv3_pmcg_13042",
                     "smmuv3_pmcg_13062", "smmuv3_pmcg_13082"]


def test_device_without_iommu_is_rejected(sysfs):
    with pytest.raises(discover.DiscoveryError):
        discover.candidates(bdf="0000:01:00.0", sysfs=sysfs)


def test_several_candidates_need_a_confirmed_choice(sysfs):
    with pytest.raises(discover.DiscoveryError, match="pravaha setup"):
        default_unit(sysfs)


def test_saved_choice_and_environment_win(sysfs, monkeypatch):
    config.save_unit("smmuv3_pmcg_13002")
    assert default_unit(sysfs) == "smmuv3_pmcg_13002"
    monkeypatch.setenv("PRAVAHA_UNIT", "smmuv3_pmcg_13042")
    assert default_unit(sysfs) == "smmuv3_pmcg_13042"


# --- perf event description ------------------------------------------------------
def test_event_config_from_sysfs(sysfs):
    ptype, cfg = perf.event_config("smmuv3_pmcg_13002", "transaction", sysfs)
    assert ptype == 42
    assert cfg == {"config": 1, "config1": 0, "config2": 0}


def test_format_spreads_value_over_ranges():
    assert perf._parse_bits("config1:0-31") == ("config1", [(0, 31)])
    assert perf._parse_bits("config:5") == ("config", [(5, 5)])
    assert perf._parse_bits("config:0-7,32-39") == ("config", [(0, 7), (32, 39)])


def test_missing_unit_is_a_clear_error(sysfs):
    with pytest.raises(perf.PerfError, match="not found"):
        perf.event_config("smmuv3_pmcg_99999", "transaction", sysfs)


# --- calibration (schema 2, made-up numbers; no built-in table) -----------------
def _entry(host, d, mib, bpt, spread=0.5, unstable=False):
    return {"host": host, "direction": d, "mib": mib, "bytes_per_txn": bpt,
            "spread_pct": spread, "unstable": unstable, "runs": [bpt], "levels": [bpt]}


def _cal_data():
    e = []
    for d in ("h2d", "d2h"):
        e += [_entry("pinned", d, 16, 4000), _entry("pinned", d, 64, 3000),
              _entry("pinned", d, 128, 2500, spread=30, unstable=True), _entry("pinned", d, 256, 2000)]
        e += [_entry("pageable", d, 4, 3700), _entry("pageable", d, 256, 3800)]
    return {"schema": 2, "calibration_id": "test-cal_2026", "method": "inprocess-1",
            "identity": {"unit": "smmuv3_pmcg_13002"}, "background": {"rate": 1000.0, "sd": 50.0},
            "entries": e}


@pytest.fixture
def cal():
    return calib.Calibration(_cal_data())


def test_no_calibration_means_no_bytes():
    with pytest.raises(calib.CalibrationMissing, match="pravaha setup"):
        calib.Calibration.load()


def test_old_schema_is_refused():
    d = _cal_data()
    d["schema"] = 1
    with pytest.raises(calib.CalibrationMissing, match="schema"):
        calib.Calibration(d)


def test_measured_interpolated_and_outside_sizes(cal):
    assert cal.bytes_per_txn(64) == 3000
    assert cal.bytes_per_txn(32) == 3500                 # halfway on a log scale
    assert cal.bytes_per_txn(1024) == 2000 and not cal.covered(1024)
    assert cal.bytes_per_txn(256, "d2h", pageable=True) == 3800


def test_unknown_profile_gives_a_bound_not_bytes(cal):
    e = cal.estimate(1000)
    assert e.mode == "bound" and e.bytes is None
    assert (e.low, e.high) == (2000 * 1000, 4000 * 1000)
    assert e.calibration_id == "test-cal_2026"


def test_known_profile_converts_with_the_mix(cal):
    mib = 2 ** 20
    prof = {(256, "h2d", False): 8, (64, "d2h", False): 8}
    net = 8 * 256 * mib / 2000 + 8 * 64 * mib / 3000     # exactly what these copies should cause
    e = cal.estimate(net, prof)
    assert e.mode == "known_profile"
    assert abs(e.bytes - 8 * 320 * mib) < 1
    assert e.low < e.bytes < e.high


def test_unstable_size_is_flagged(cal):
    assert ("pinned", "h2d", 128) in cal.unstable_sizes()
    e = cal.estimate(1e5, {(128, "h2d", False): 4})
    assert any("unstable" in n for n in e.notes)


def test_identity_change_makes_calibration_stale(cal):
    saved = {"unit": "smmuv3_pmcg_13002", "driver": "580.95"}
    now = {"unit": "smmuv3_pmcg_13002", "driver": "590.10"}
    from pravaha_mem import identity
    assert identity.differences(saved, now) == ["driver"]


def test_saved_calibration_is_loaded_back(cal):
    path = cal.save()
    back = calib.Calibration.load()
    assert back.path == path and back.calibration_id == "test-cal_2026"
    assert back.bytes_per_txn(32, "d2h") == 3500
    tsv = open(os.path.join(os.path.dirname(path), "calibration.tsv")).read().splitlines()
    assert tsv[0].startswith("#id\ttest-cal_2026\t1000.000")


C_TEST = r"""
#include <stdio.h>
#include <math.h>
#include "pravaha.h"
int main(int argc, char **argv) {
    char u[PRAVAHA_MAX_UNITS][64];
    int n = pravaha_candidates(argv[1], NULL, u, PRAVAHA_MAX_UNITS);
    for (int i = 0; i < n; i++) printf("%s ", u[i]);
    pravaha_calibration cal;
    if (pravaha_calibration_load(&cal, NULL) == 0)
        printf("| %s %.0f %.0f %.0f %.0f %.0f |", cal.id, pravaha_bytes_per_txn(&cal, 32, PRAVAHA_H2D, 0),
               pravaha_bytes_per_txn(&cal, 256, PRAVAHA_D2H, 1), pravaha_bytes_per_txn(&cal, 1024, PRAVAHA_H2D, 0),
               pravaha_net(1000, 2.0, cal.idle_rate), cal.idle_rate);
    else
        printf("| nocal %s |", pravaha_last_error());
    char d[64];
    int rc = pravaha_default_unit(argv[1], d, sizeof d);
    printf(" %d %s\n", rc, rc == 0 ? d : pravaha_last_error());
    return 0;
}
"""


def _run_c(sysfs, tmp_path):
    src, exe = tmp_path / "t.c", tmp_path / "t"
    src.write_text(C_TEST)
    subprocess.run(["cc", "-std=gnu99", "-Wall", "-Werror", "-I", os.path.join(ROOT, "lib"),
                    str(src), os.path.join(ROOT, "lib", "pravaha.c"), "-lm", "-o", str(exe)], check=True)
    return subprocess.run([str(exe), sysfs], capture_output=True, text=True, check=True).stdout.split("|")


@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
def test_c_library_agrees_with_python(sysfs, tmp_path, cal):
    units, calpart, choice = _run_c(sysfs, tmp_path)
    assert units.split() == discover.candidates(sysfs=sysfs)[1]
    assert calpart.split()[0] == "nocal"                  # C has no built-in table either
    assert choice.strip().startswith("-1") and "pravaha setup" in choice
    cal.save()
    config.save_unit("smmuv3_pmcg_13002")                # the C side reads the same files
    units, calpart, choice = _run_c(sysfs, tmp_path)
    assert calpart.split() == ["test-cal_2026", f"{cal.bytes_per_txn(32):.0f}",
                               f"{cal.bytes_per_txn(256, 'd2h', True):.0f}", "2000", "-1000", "1000"]
    assert choice.split() == ["0", "smmuv3_pmcg_13002"]


# --- command line ------------------------------------------------------------------
def test_cli_profile_parsing():
    from pravaha_mem import cli
    assert cli._parse_profile("256:h2d:8,64:d2h:pageable:4") == {
        (256, "h2d", False): 8, (64, "d2h", True): 4}
    assert cli._parse_profile("") is None


def test_cli_old_verbs_map_to_new(monkeypatch, capsys):
    from pravaha_mem import cli
    seen = {}
    monkeypatch.setattr(cli, "cmd_show", lambda a: seen.setdefault("verb", "show") and 0)
    assert cli.main(["info"]) == 0 and seen["verb"] == "show"
    assert "now 'show'" in capsys.readouterr().err


def test_show_never_raises(sysfs):
    from pravaha_mem import status
    s = status.show(sysfs)
    assert s["calibration"] == "absent" and s["reasons"]


# --- kernel path (fake ncu) ---------------------------------------------------------
FAKE_NCU = '''#!/bin/sh
echo "my program prints this"
echo '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"'
echo '"0","readk(a, b)","lts__d_sectors_fill_sysmem.sum","sector","8,388,608"'
echo '"0","readk(a, b)","lts__t_sectors_aperture_sysmem_op_write.sum","sector","0"'
echo '"1","writek","lts__d_sectors_fill_sysmem.sum","sector","3,399"'
echo '"1","writek","lts__t_sectors_aperture_sysmem_op_write.sum","sector","8,391,743"'
'''


def test_kernel_traffic_runs_ncu_and_sums_bytes(tmp_path):
    from pravaha_mem import ncu
    fake = tmp_path / "ncu"
    fake.write_text(FAKE_NCU)
    fake.chmod(0o755)
    r = ncu.kernel_traffic(["./app"], ncu=str(fake))
    assert [k["kernel"] for k in r.kernels] == ["readk(a, b)", "writek"]
    assert r.read_bytes == (8388608 + 3399) * 32
    assert r.write_bytes == 8391743 * 32
    assert "2 kernels" in r.summary()


def test_kernel_traffic_explains_missing_permission(tmp_path):
    from pravaha_mem import ncu
    fake = tmp_path / "ncu"
    fake.write_text("#!/bin/sh\necho '==ERROR== ERR_NVGPUCTRPERM - no permission'\n")
    fake.chmod(0o755)
    with pytest.raises(RuntimeError, match="NVreg_RestrictProfilingToAdminUsers"):
        ncu.kernel_traffic(["./app"], ncu=str(fake))


# --- calibration arithmetic (fake counter and copies) ---------------------------------
class _FakeCounter:
    """Counts 1,000 transactions/s of background, plus 1 per 2,048 bytes copied."""
    def __init__(self, buf):
        self.buf, self.value, self.t0 = buf, 0, None

    def start(self):
        import time
        self.value, self.t0, self.buf.copied = 0, time.monotonic(), 0

    def stop(self):
        import time
        self.value = int(1000 * (time.monotonic() - self.t0) + self.buf.copied / 2048)

    def read(self):
        return self.value


class _FakeBuf:
    nbytes = 64 << 20
    copied = 0

    def copy(self, direction):
        self.copied += self.nbytes

    def sync(self):
        pass


def test_calibration_subtracts_background_measured_just_before():
    from pravaha_mem import calibrate
    buf = _FakeBuf()
    bpt, net, dt = calibrate.measure(_FakeCounter(buf), buf, 1, copies=8, idle_seconds=0.2)
    assert abs(bpt - 2048) / 2048 < 0.01


# --- N3 repeatability: campaigns kept in history, one per boot ------------------------
def test_repeatability_needs_three_boots_and_flags_recurring_instability(monkeypatch):
    from pravaha_mem import checks
    for i, boot in enumerate(["b1", "b1", "b2", "b3"]):        # two campaigns in boot b1
        d = _cal_data()
        d["calibration_id"] = f"cal-{i}"
        d["boot_id"] = boot
        calib.Calibration(d).save()
        if i == 1:
            r = checks.check_repeatability()
            assert r["status"] == "SKIP" and "1 campaign" in r["reason"]
    r = checks.check_repeatability()
    assert r["status"] == "INFO"
    assert r["values"]["campaigns"] == ["cal-1", "cal-2", "cal-3"]   # latest of boot b1 kept
    assert ("pinned", "h2d", 128) in r["values"]["recurring_unstable"]
