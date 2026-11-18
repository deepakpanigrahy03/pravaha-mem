<p align="center">
  <img src="assets/pravaha-banner.svg" alt="pravaha: GPU memory flow on the NVIDIA GB10" width="100%">
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
  <img alt="Python 3.8+" src="https://img.shields.io/badge/python-3.8%2B-blue.svg">
  <img alt="Platform: NVIDIA GB10" src="https://img.shields.io/badge/platform-NVIDIA%20GB10-76B900.svg">
  <img alt="Dependencies: none" src="https://img.shields.io/badge/dependencies-none-brightgreen.svg">
  <!-- PyPI and DOI badges are added after the first release -->
</p>

# pravaha-mem

**Live GPU memory traffic on the NVIDIA GB10 (DGX Spark, Acer GN100 and other GB10 systems), for any program, without changing it.**

On the GB10 the GPU has no memory of its own: every byte it reads or writes goes to the shared LPDDR5X behind the CPU die, over NVLink C2C. NVIDIA exposes no counter for that link on GB10. pravaha reads the GPU's own L2 cache counters every 10 ms, from a separate process, and reports the bytes moved: by copy engines, by GPU code reading and writing, and in total.

*Pravaha* (प्रवाह) is Sanskrit for *flow*.

## 🧭 What it gives you

| | feature | command |
|---|---|---|
| 1 | allow counter reading, once per machine | `pravaha install` |
| 2 | is this machine ready? | `pravaha status` |
| 3 | check the numbers against known bytes | `pravaha verify` |
| 4 | traffic while a program runs | `pravaha run -- CMD` |
| 5 | traffic over a time window, e.g. a running LLM server | `pravaha observe --seconds 60` |
| 6 | live samples | `pravaha observe --live` |
| 7 | phases inside your own Python code | `with pv.observe() as m: m.mark("decode")` |
| 8 | save data with full provenance | `--csv FILE`, `--json FILE` |
| 9 | GPU address translations (counts, not bytes) | `pravaha smmu status / run / observe` |
| 10 | traffic per kernel (Nsight Compute) | `pravaha kernels -- CMD` |
| 11 | remove it | `pravaha uninstall` |

## 🚀 Start

```bash
# from PyPI
pip install pravaha-mem
# or from GitHub (latest)
git clone https://github.com/deepakpanigrahy03/pravaha-mem && cd pravaha-mem && pip install .
```
Install it in the Python environment that will measure (a venv is fine). The program you measure needs nothing installed. Then:

```bash
pravaha install          # once per machine: asks for your password once
pravaha status           # ready?
pravaha verify           # optional: about 3 minutes, light work only
pravaha run --idle 2 -- python my_app.py
```

```
pravaha: measuring 1 s of idle baseline ...
pravaha: observing the whole GPU while 'python -m pravaha_mem.demo h2d 8 256' runs; results print when it ends
phase               sec   copy GiB  kread GiB kwrite GiB   C2C GiB    GB/s
total              1.52     2.0146     0.0669     0.2500    4.6802     3.3
idle               1.00     0.0000     0.0244     0.0000    0.0989     0.1
run                0.52     2.0146     0.0425     0.2500    4.5813     9.4
```
That demo copies 2 GiB; the extra 0.0146 GiB of copies and 0.25 GiB of writes are its own CUDA startup and buffer clearing.

## 🔐 What `pravaha install` puts on your system

Exactly one folder, and nothing else:
```
/usr/local/libexec/pravaha/
    pravaha-reader   reads the GPU's L2 counters       capability cap_perfmon
    pravaha-pmcg     reads the GPU's SMMU counters     capability cap_perfmon
    MANIFEST         what install created (with checksums)
```
Root owned, not writable by users, not in your PATH. No system setting, driver option, group, sudoers entry or service is changed. The readers run as the normal user and only read counters. `pravaha uninstall` removes exactly what MANIFEST lists and confirms nothing is left; then `pip uninstall pravaha-mem`. Manual cleanup, if ever needed: `sudo rm -r /usr/local/libexec/pravaha`.

Why a password at all: NVIDIA's driver lets only privileged programs read GPU counters, and only root can attach a capability to a file.

## 🐍 From Python

```python
import pravaha_mem as pv

with pv.observe() as m:
    m.metadata({"model": "qwen3-4b"})
    m.mark("load");    model = load_model()
    m.mark("decode");  out = model.generate(prompt)

print(m.summary())        # bytes per phase
m.to_csv("trace.csv")     # one row per sample, system clock in ns
m.to_json("run.json")     # totals, phases, baseline, sampling quality, provenance
```
`m.start()` and `m.stop()` work too, for harnesses with their own hooks (A-LEMS uses these).

## 📏 What the numbers mean

| column | from the GPU's L2 counter | meaning |
|---|---|---|
| copy | `lts__t_sectors_srcunit_ce` (sectors x 32 / 2) | bytes moved by copy engines: `cudaMemcpy`, `.cuda()`, model loading. Each copied byte passes L2 twice, hence / 2. |
| kread | `..._srcunit_tex_op_read_lookup_miss_realtime` (x 32) | bytes GPU code read from memory (L2 misses) |
| kwrite | `..._srcunit_tex_op_write_realtime` (x 32) | bytes GPU code wrote |
| C2C | all L2 read misses + all L2 writes (x 32) | all GPU memory traffic. On GB10 all GPU memory lies behind the CPU die, so this traffic crosses NVLink C2C. |

* **Device wide.** The counters see every process on the GPU. `run -- CMD` reports the GPU's traffic while CMD runs, not CMD's alone. Use `--idle` or phases for a baseline; the JSON reports raw, baseline and subtracted values separately.
* **One reader at a time.** The GPU has one sampling unit. While another profiler holds it, pravaha says "counters busy".
* **GPU side.** CPU memory traffic is not included.

## ✅ Checked against known bytes

`pravaha verify` on an Acer GN100 and a DGX Spark (driver 580, CUDA 13.0), each check the middle of 3 runs:
```
  PASS  copy host to GPU             2048.0 MiB  expected  2048.0  ratio 1.0000
  PASS  copy GPU to host             2048.0 MiB  expected  2048.0  ratio 1.0000
  PASS  kernel write (cudaMemset)    4096.0 MiB  expected  4096.0  ratio 1.0000
  PASS  kernel read (GPU to GPU copy) 4095.9 MiB expected  4096.0  ratio 1.0000
```
On real frameworks, observed from outside: PyTorch `.cuda()` 8 minus 4 copies of 1 GiB gives 3.9999 GiB; vLLM serving Qwen3 4B reads 7.52 GiB per generated token, against 7.49 GiB of weights. PASS means agreement for these workloads on that machine and software, not a claim about every workload.

## 🎓 Learn
[docs/TUTORIAL.md](docs/TUTORIAL.md): lessons with expected output. [docs/MANUAL.md](docs/MANUAL.md): every command, option and field. Examples: [examples/](examples).

## 📦 Requirements
NVIDIA GB10 system, Linux, NVIDIA driver 565 or newer with CUDA 13 (CUPTI included), `gcc` and `g++`, Python 3.8 or newer.

## 📖 Cite
See [CITATION.cff](CITATION.cff).

## License
MIT. Copyright (c) 2026 MANTRA Labs.
