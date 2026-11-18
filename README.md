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

For example, measuring pravaha's test program while it copies 2 GiB into the GPU:
```
pravaha: measuring 1 s of idle baseline ...
pravaha: observing the whole GPU while 'python -m pravaha_mem.known_bytes h2d 8 256' runs; results print when it ends
phase               sec   copy GiB  kread GiB kwrite GiB   C2C GiB    GB/s
total              1.52     2.0146     0.0669     0.2500    4.6802     3.3
idle               1.00     0.0000     0.0244     0.0000    0.0989     0.1
run                0.52     2.0146     0.0425     0.2500    4.5813     9.4
```
That command is pravaha's test program copying 2 GiB into the GPU (see below). The extra 0.0146 GiB of copies is CUDA's startup; the 0.25 GiB of kwrite is the program clearing its GPU buffer before the copies.

## 🧪 The test program: known_bytes

To check pravaha you need data whose amount is known in advance. `known_bytes` is a small, readable program ([python/pravaha_mem/known_bytes.py](python/pravaha_mem/known_bytes.py)) that moves exactly that, with NVIDIA's standard CUDA calls, and says what it did.

```bash
python -m pravaha_mem.known_bytes DIRECTION TIMES SIZE_MIB        # moves TIMES x SIZE_MIB
```

| DIRECTION | stands for | what happens | who moves it | pravaha column |
|---|---|---|---|---|
| `h2d` | host to device | copy from the computer's memory into the GPU | copy engines (`cudaMemcpy`) | copy |
| `d2h` | device to host | copy from the GPU back to the computer's memory | copy engines (`cudaMemcpy`) | copy |
| `d2d` | device to device | copy from one GPU buffer to another | GPU code reading and writing | kread, kwrite |
| `memset` | memory set | fill a GPU buffer with a value | GPU code writing (`cudaMemset`) | kwrite |

"Host" is NVIDIA's word for the computer side, "device" for the GPU. Examples: `h2d 8 256` copies 8 x 256 MiB = 2 GiB into the GPU; `memset 16 256` makes GPU code write 4 GiB. Before its work it reserves its buffers and clears the two GPU buffers (2 x SIZE_MIB in kwrite); every CUDA program also moves about 15 MiB of copies when it starts. `pravaha verify` runs these four directions and compares.

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

How the counting works: every GPU request to memory passes the GPU's L2 cache. If the data is already in L2 (a hit), nothing crosses C2C and nothing is counted. If not (a miss), it is fetched over C2C and counted (kread for GPU code). Every write passes L2 and goes to memory over C2C (kwrite). Copies pass L2 twice, as a read and a write, hence copy = sectors x 32 / 2. Diagrams: [GB10 architecture](docs/GB10_ARCHITECTURE.md).

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
Samples carry the system clock; on GB10 it agrees with the GPU's sample clock within 1 microsecond. On real frameworks, observed from outside: PyTorch `.cuda()` 8 minus 4 copies of 1 GiB gives 3.9999 GiB; vLLM serving Qwen3 4B reads 7.52 GiB per generated token, against 7.49 GiB of weights. PASS means agreement for these workloads on that machine and software, not a claim about every workload.

## 📚 Documentation

| read this | if you want to |
|---|---|
| [GB10 architecture](docs/GB10_ARCHITECTURE.md) | understand the GB10, NVLink C2C and what pravaha counts (with diagrams) |
| [Tutorial](docs/TUTORIAL.md) | learn by running: each lesson one command, with the expected output |
| [Manual](docs/MANUAL.md) | look up every command, option, column and word |
| [Developer guide](docs/DEVELOPER_GUIDE.md) | use pravaha inside your Python programs and harnesses |
| [Researcher guide](docs/RESEARCHER_GUIDE.md) | methodology, accuracy, time base, and what to report in a paper |
| [Validation record](docs/VALIDATION.md) and [test matrix](docs/TEST_MATRIX.md) | see every test, on which machine, with what result; add yours |
| [Design](docs/DESIGN.md) | the system design and its rules |
| [Contributing](CONTRIBUTING.md) | report results from your GB10, or change the code |

Examples: [examples/](examples).

## 📦 Requirements
NVIDIA GB10 system, Linux, NVIDIA driver 565 or newer with CUDA 13 (CUPTI included), `gcc` and `g++`, Python 3.8 or newer.

## 📖 Cite
See [CITATION.cff](CITATION.cff).

## License
MIT. Copyright (c) 2026 MANTRA Labs.
