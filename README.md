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

**Live GPU memory traffic over NVLink C2C on the NVIDIA GB10 (DGX Spark, Acer GN100 and other GB10 systems), for any program, without changing it and without root.**

The GB10 joins an Arm CPU die and a Blackwell GPU die with NVLink C2C. The GPU has no memory of its own: every byte it reads from or writes to the shared LPDDR5X crosses C2C. NVIDIA exposes no C2C traffic counter on GB10, so how much a program moves, and when, has been invisible.

pravaha reads the GPU's own L2 cache counters live, every 10 ms, from a separate process. Every L2 miss and every write leaves the GPU over C2C, so these counters give the traffic directly, split into **copies** (copy engines) and **kernels** (reads and writes by GPU code).

*Pravaha* (प्रवाह) is Sanskrit for *flow*.

## 🚀 Start

```bash
# from PyPI
pip install pravaha-mem

# or from GitHub (latest)
git clone https://github.com/deepakpanigrahy03/pravaha-mem
cd pravaha-mem && pip install .
```

Install it in the Python environment that will do the measuring (a venv is fine: `python3 -m venv ~/pravaha-venv && . ~/pravaha-venv/bin/activate`). Then, once per machine:

```bash
pravaha install-sampler          # asks for your password once
pravaha test                     # optional: check it against known bytes
pravaha c2c --idle 2 -- python my_app.py
```

For example, a program that copies 2 GiB from host to GPU:

```
pravaha: measuring 1 s of idle baseline ...
pravaha: measuring while 'python -m pravaha_mem.demo h2d 8 256' runs; results print when it ends (Ctrl+C stops it and still prints them)
phase               sec   copy GiB  kread GiB kwrite GiB   C2C GiB    GB/s
total              1.52     2.0146     0.0669     0.2500    4.6802     3.3
idle               1.00     0.0000     0.0244     0.0000    0.0989     0.1
run                0.52     2.0146     0.0425     0.2500    4.5813     9.4
```

2 GiB of copies plus the program's own CUDA startup (0.0146 GiB of copies, and 0.25 GiB of kernel writes from clearing its buffer).

The program you measure needs nothing installed. It can be Python, C++, a server, anything that uses the GPU.

## 🐍 From Python

```python
import pravaha_mem as pv

with pv.c2c() as m:
    m.mark("load");    model = load_model()
    m.mark("decode");  out = model.generate(prompt)

print(m.summary())        # GiB per phase: copies, kernel reads, kernel writes, total, GB/s
m.to_csv("trace.csv")     # one row per sample, in ns, tagged with its phase
```

`m.start()` and `m.stop()` work too, for frameworks that already have their own start and stop hooks. Timestamps use the system clock (`time.time_ns()`), so samples line up with energy or other traces.

## 👥 Who it helps

* **Anyone running GPU work on a GB10**: one command tells you how much your program moves through memory, and when. Run `pravaha c2c --seconds 60` to watch a server that is already running.
* **Developers**: mark phases in your code (`m.mark("prefill")`), get results as a table, a dict, CSV or JSON, and see which step is bound by memory bandwidth.
* **Researchers**: a 10 ms time series of GPU memory traffic, on the system clock, ready to join with power and energy; bytes per token for LLMs; a self test (`pravaha test`) to cite as evidence.

## ✅ Checked against known bytes

`pravaha test` runs these on your machine. Each check is a separate process that moves a known number of bytes while pravaha watches from outside (Acer GN100, driver 580, CUDA 13.0):

```
  PASS  idle machine: no copies                           0.0 MiB  expected      0.0
  PASS  copy host to GPU                               2048.0 MiB  expected   2048.0  ratio 1.0000
  PASS  copy GPU to host                               2048.0 MiB  expected   2048.0  ratio 1.0000
  PASS  kernel write (cudaMemset)                      2048.0 MiB  expected   2048.0  ratio 1.0000
  PASS  kernel read and write (GPU to GPU copy)        2047.9 MiB  expected   2048.0  ratio 1.0000
```

And on real frameworks, measured from outside without changing them:

| workload | pravaha | expected |
|---|---|---|
| PyTorch `.cuda()`: 8 copies minus 4 copies of 1 GiB | 3.9999 GiB | 4 GiB |
| vLLM serving Qwen3 4B: model load | 7.80 GiB copies | weights 7.49 GiB plus startup |
| vLLM: 400 generated tokens | 7.52 GiB kernel reads per token | 7.49 GiB of weights read once per token |

Copy counts are the same for pinned, registered, 2 MiB huge page and pageable host memory. A program's own CUDA startup appears as a small fixed amount (about 15 MiB for a plain CUDA program); `--idle` and phase marks separate it.

## 🎓 Learn by example

[docs/TUTORIAL.md](docs/TUTORIAL.md) walks through five lessons with expected output:

| example | shows |
|---|---|
| [01_any_command.sh](examples/01_any_command.sh) | measure any program, with an idle baseline, CSV and JSON |
| [02_python_phases.py](examples/02_python_phases.py) | phases inside your PyTorch code, measured against expected bytes |
| [03_llm_bytes_per_token.py](examples/03_llm_bytes_per_token.py) | GiB per token of a running LLM server |
| [04_join_with_other_data.py](examples/04_join_with_other_data.py) | line up samples with power, energy or logs |

## 📏 What the numbers mean

| column | counter (GPU L2, 32 B sectors) | meaning |
|---|---|---|
| copy | `lts__t_sectors_srcunit_ce` | bytes moved by copy engines (`cudaMemcpy` and friends) |
| kread | `lts__t_sectors_srcunit_tex_op_read_lookup_miss_realtime` | bytes GPU code read from memory (L2 misses) |
| kwrite | `lts__t_sectors_srcunit_tex_op_write_realtime` | bytes GPU code wrote |
| C2C | all L2 read misses plus all L2 writes | everything the GPU sent or fetched over C2C |

Good to know:

* **Whole GPU.** The counters see every program on the GPU, including the desktop. Use `--idle` or a phase mark for a baseline.
* **One reader at a time.** The GPU has one sampling unit. While another profiler holds it (another `pravaha c2c`, Nsight Compute, Nsight Systems GPU metrics, DCGM profiling), pravaha says so and stops.
* **GPU side.** CPU traffic to memory is not counted here.

## 🔐 Permissions, without changing your system

GPU counters normally need root. `pravaha install-sampler` builds one small program, `pravaha-c2c`, copies it to `/usr/local/bin` and gives that one file the `cap_sys_admin` capability. It only reads counters and runs no GPU work. pravaha changes no system setting, no driver option, no group and no sudoers file. `pravaha permissions` shows the state; `pravaha uninstall-sampler` removes it.

## 🧰 Commands

| command | does |
|---|---|
| `pravaha c2c [--idle S] [--csv FILE] [--json FILE] -- CMD` | measure while CMD runs |
| `pravaha c2c --seconds N [--csv FILE]` | observe the whole GPU for N seconds (a running server, say) |
| `pravaha install-sampler` / `uninstall-sampler` | set up or remove the reader (once per machine) |
| `pravaha permissions` | what can be read here, changes nothing |
| `pravaha test` | check pravaha against known bytes on this machine |

Diagnostics: pravaha also reads the GPU's SMMU (address translation) counters through Linux perf. They count page translations, not bytes, and are kept for studying page size and translation caching. See [docs/MANUAL.md](docs/MANUAL.md).

## 📦 Requirements

NVIDIA GB10 system, Linux, NVIDIA driver with CUDA 13 (CUPTI is part of it), `g++`, Python 3.8 or newer.

## 📖 Cite

If pravaha helps your work, please cite it (see [CITATION.cff](CITATION.cff)).

## License

MIT. Copyright (c) 2026 Deepak Panigrahy and Sirisha Kota.
