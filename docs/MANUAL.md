# pravaha manual

## 1. What pravaha measures

On the NVIDIA GB10 the GPU reaches all of its memory over NVLink C2C, through the CPU die. A byte the GPU needs that is not in its L2 cache is fetched over C2C; a byte it writes goes back over C2C. pravaha counts these at the GPU's L2 cache, which sees every access, and splits them by who made them:

* **copy engines**: `cudaMemcpy`, `.cuda()`, `.cpu()`, model weight loading
* **kernels**: GPU code reading and writing memory (matrix multiplies, attention, and so on)

It samples the counters every 10 ms through CUPTI PM sampling, from its own process, for the whole GPU.

## 2. Install

```bash
pip install pravaha-mem
pravaha install-sampler
pravaha permissions
```

`install-sampler` compiles two small readers from the source shipped in the package and installs them to `/usr/local/bin` with one capability each:

| reader | capability | role |
|---|---|---|
| `pravaha-c2c` | `cap_sys_admin` | GPU L2 counters (main path) |
| `pravaha-pmcg` | `cap_perfmon` | SMMU translation counters (diagnostics) |

Set `CUDA_HOME` if CUDA is not in `/usr/local/cuda`. Install the Python package in every environment that will measure; the readers are installed once per machine. The program being measured needs nothing.

## 3. Command line

```bash
pravaha c2c -- ./my_cuda_app
pravaha c2c --idle 2 --csv trace.csv -- python train.py
```

* `--idle S` measures S seconds of the idle machine first, reported as its own row.
* `--csv FILE` writes one row per sample: `t_start_ns, t_end_ns, phase, copy, kernel_read, kernel_write, all_read, all_write` (bytes).
* `--json FILE` writes the summary (total and phases, bytes) as JSON.
* `--seconds N` with no command observes the whole GPU for N seconds, for example a server that is already running.
* `--interval MS` sets the sample interval (default 10).

`pravaha test` checks the counters against small programs that move known bytes. `pravaha test --smmu` runs the older SMMU checks.

Long commands print a line at the start, so you know it is working. Ctrl+C stops the command and still prints the results.

## 4. Python

```python
import pravaha_mem as pv

with pv.c2c() as m:
    m.mark("prefill"); ...
    m.mark("decode");  ...

m.summary()      # text table
m.total()        # dict, bytes: copy, kernel_read, kernel_write, all_read, all_write, c2c, seconds
m.phases()       # {phase name: dict as above}
m.samples        # list of Sample(t0, t1, copy, kernel_read, kernel_write, all_read, all_write)
m.to_csv(path)
```

Without `with`: `m = pv.c2c().start()` ... `m.stop()`. To run a command: `rc, m = pv.c2c_run(["./app"], idle_s=2)`.

Timestamps are `CLOCK_REALTIME` in ns, the same as `time.time_ns()` and `date +%s%N`, so samples join directly with other measurements (power, energy, logs).

## 5. Reading the results

* **copy** is exact: copied bytes, whatever the host memory type.
* **kread** and **kwrite** are what GPU code moved. Data already in L2 is not counted, because it did not cross C2C.
* **C2C** is everything the GPU fetched or sent, from all sources, so it includes the desktop and other programs. Compare with the idle row.
* A program's CUDA startup moves a small, fixed amount (about 15 MiB of copies for a plain CUDA program). To isolate your work, mark phases, or compare two runs that differ only in the work (8 copies minus 4 copies gives exactly 4).
* If the summary ends with "N ms not sampled", the counter buffer was reset during a very long run (about every 16 minutes at 10 ms); totals exclude that time.

## 6. Errors

| message | meaning, what to do |
|---|---|
| GPU counters busy | another profiler holds the sampling unit; stop it and run again |
| pravaha-c2c not found | run `pravaha install-sampler` |
| missing cap_sys_admin | run `pravaha install-sampler` again (a rebuilt or copied file loses its capability) |

## 7. Diagnostics: SMMU translation counters

The GPU's SMMU on GB10 has Linux perf counters (`smmuv3_pmcg_*`). On the GPU's TCU PMCG, `transaction` equals `pcie_ats_trans_rq`: it counts address translation requests for host pages, not bytes. Bytes per request vary with allocation type, page size and history, so these counters are not used for traffic. They are useful for studying translations (4 KiB versus 2 MiB pages, translation cache reach). `pravaha show`, `pravaha watch` and `pravaha measure` expose them.

## 8. Uninstall

```bash
pravaha uninstall-sampler
pip uninstall pravaha-mem
```
