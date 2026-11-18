"""Tests for validate/analyze.py and validate/ncu_parse.py. No GPU needed: python3 -m pytest tests"""
import csv
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "validate"))
import analyze  # noqa: E402
import ncu_parse  # noqa: E402

SAMPLE = os.path.join(ROOT, "examples", "sample_results")


def _copy_sample(tmp_path):
    d = tmp_path / "r"
    shutil.copytree(SAMPLE, d)
    return str(d)


def test_sample_results_reproduce_paper_values(tmp_path):
    R = analyze.analyze(_copy_sample(tmp_path))
    assert R["all_ok"]
    assert round(R["cal_bytes_per_txn"]) == 2040
    assert round(R["cal"]["h2d"]["bytes_per_txn"]) == 2046
    assert round(R["cal"]["d2h"]["bytes_per_txn"]) == 2034
    assert R["read_err_max"] < 0.25
    assert abs(R["writes"]["malloc"]["err_pct"] - 15.17) < 0.01


def _write(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def test_idle_background_is_subtracted(tmp_path):
    """Synthetic run: 1,000 txn/s background, 100,000 startup, 130,000 per round, runs take 2 s + 0.1 s per round."""
    d = tmp_path / "syn"
    d.mkdir()
    _write(d / "platform.csv", ["key", "value"], [["unit", "smmuv3_pmcg_x"]])
    _write(d / "smmu_idle.csv", ["run", "seconds", "txn"], [[r, 10, 10000] for r in range(1, 6)])
    rows = []
    for r in range(1, 6):
        for dr in ("h2d", "d2h"):
            for k in (0, 8, 16, 32):
                el = 2 + 0.1 * k
                rows.append([dr, k, r, 100000 + 130000 * k + 1000 * el, el])
    _write(d / "smmu_calib.csv", ["dir", "rounds", "run", "txn", "elapsed_s"], rows)
    R = analyze.analyze(str(d))
    assert R["idle_source"] == "measured"
    c = R["cal"]["h2d"]
    assert abs(c["per_round"] - 130000) < 1e-6          # background removed from the slope
    assert abs(c["intercept"] - 100000) < 1e-6          # and from the intercept
    assert abs(c["intercept_raw"] - 102000) < 1e-6
    assert c["repeats"] == 5
    assert abs(R["cal_bytes_per_txn"] - 256 * 2**20 / 130000) < 1e-6


def test_failing_check_is_reported(tmp_path):
    d = _copy_sample(tmp_path)
    _write(os.path.join(d, "ncu_reads.csv"), ["mode", "fill_sysmem", "fill_device"],
           [["pinned", 9000000, 0], ["malloc", 8388608, 0], ["device", 8388608, 5]])
    R = analyze.analyze(d)
    assert not R["all_ok"]
    failed = {c["name"] for c in R["checks"] if not c["ok"]}
    assert "ncu reads: max |error| %" in failed
    assert "device-memory fill counter (all memory is sysmem)" in failed


NCU_CSV = '''==PROF== Connected to process 123
mode=pinned op=r rounds=1 bytes=268435456 status=no error
"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Block Size","Grid Size","Device","CC","Section Name","Metric Name","Metric Unit","Metric Value"
"0","123","zc","127.0.0.1","readk(const unsigned long long *, unsigned long, unsigned long long *)","1","7","(256, 1, 1)","(1024, 1, 1)","0","12.1","Command line profiler metrics","lts__d_sectors_fill_sysmem.sum","sector","8,407,066"
"0","123","zc","127.0.0.1","readk(const unsigned long long *, unsigned long, unsigned long long *)","1","7","(256, 1, 1)","(1024, 1, 1)","0","12.1","Command line profiler metrics","lts__t_sectors_aperture_sysmem_op_write.sum","sector","0"
"1","123","zc","127.0.0.1","writek(unsigned long long *, unsigned long)","1","7","(256, 1, 1)","(1024, 1, 1)","0","12.1","Command line profiler metrics","lts__d_sectors_fill_sysmem.sum","sector","18493"
"1","123","zc","127.0.0.1","writek(unsigned long long *, unsigned long)","1","7","(256, 1, 1)","(1024, 1, 1)","0","12.1","Command line profiler metrics","lts__t_sectors_aperture_sysmem_op_write.sum","sector","8391787"
'''


def test_ncu_parse_per_kernel_and_total():
    order, vals = ncu_parse.parse(NCU_CSV.splitlines())
    assert len(order) == 2
    header, rows = ncu_parse.rows_out(order, vals)
    assert header[1:3] == ["lts__d_sectors_fill_sysmem.sum", "lts__t_sectors_aperture_sysmem_op_write.sum"]
    assert rows[0][0].startswith("readk(") and "," not in rows[0][0]
    assert rows[0][1] == "8407066"
    assert rows[1][2] == "8391787"
    header, rows = ncu_parse.rows_out(order, vals, total=True)
    assert rows == [["TOTAL", "8425559", "8391787", f"{8425559*32/1e6:.3f}", f"{8391787*32/1e6:.3f}"]]
    _, rows = ncu_parse.rows_out(order, vals, match="^writek")
    assert len(rows) == 1


def test_copy_size_configurations(tmp_path):
    """Synthetic sizes: pinned 4 MiB at 2,000 B/txn, pageable 256 MiB at 4,000 B/txn, no background."""
    d = _copy_sample(tmp_path)
    rows = []
    for host, mib, bpt in (("pinned", 4, 2000.0), ("pageable", 256, 4000.0)):
        n = 4096 // mib
        for dr in ("h2d", "d2h"):
            for r in (1, 2, 3):
                rows.append([host, mib, dr, 0, r, 100000])
                rows.append([host, mib, dr, n, r, 100000 + n * mib * 2**20 / bpt])
    _write(os.path.join(d, "smmu_sizes.csv"), ["host", "mib", "dir", "rounds", "run", "txn"], rows)
    R = analyze.analyze(d)
    assert abs(R["sizes"]["pinned_4"]["bytes_per_txn"] - 2000) < 1e-6
    assert abs(R["sizes"]["pageable_256"]["bytes_per_txn"] - 4000) < 1e-6
    assert R["sizes"]["pinned_4"]["n"] == 6
