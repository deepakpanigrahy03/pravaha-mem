# pravaha manual (1.0)

## 1. What pravaha measures
| term | meaning | how |
|---|---|---|
| L2 traffic | 32 byte sectors counted at the GPU's L2 cache | measured: 5 counters, CUPTI PM sampling |
| memory traffic | L2 read misses (fetched from memory) and L2 writes (sent toward memory) | measured |
| C2C traffic | bytes crossing NVLink C2C | inferred: on GB10 all GPU memory is the shared LPDDR5X behind the CPU die; supported by `verify`. No link counter exists on GB10. |

How the counting works, with diagrams (L2 hit: not counted; L2 miss: fetched over C2C, counted): [GB10_ARCHITECTURE.md](GB10_ARCHITECTURE.md).

All values are **device wide**: every process on the GPU during the window. Live sampling on GB10 does not offer a split by memory type (aperture); the totals cover all of it.

## 2. Commands
| command | does |
|---|---|
| `pravaha install [--prefix DIR] [--cap cap_perfmon\|cap_sys_admin]` | build both readers, install them with one capability each, write MANIFEST; test that the L2 reader works |
| `pravaha uninstall [--prefix DIR]` | remove exactly the MANIFEST files (sha256 checked), the folder and any parents install created; confirm |
| `pravaha status` | GPU, driver, CUDA; each reader and its capability; counters free or busy; temperature and thermal headroom |
| `pravaha verify [--json FILE] [--cooldown S]` | light known byte checks with a thermal guard; PASS or FAIL per check |
| `pravaha run [opts] -- CMD` | observe while CMD runs; stops when CMD ends; Ctrl+C stops CMD and still prints results |
| `pravaha observe [opts] [--seconds N] [--live]` | observe for N seconds or until Ctrl+C; `--live` prints each sample as CSV |
| `pravaha smmu status` | the GB10's SMMU counters and which one belongs to the GPU |
| `pravaha smmu run [--unit U] [--live] -- CMD` | address translation counts while CMD runs |
| `pravaha smmu observe [--unit U] [--live] [--seconds N]` | address translation counts over a window |
| `pravaha kernels [--match REGEX] -- CMD` | L2 traffic per kernel through Nsight Compute (replay: slower; needs profiling rights) |

Options of `run` and `observe`: `--idle S` (idle baseline first, its own row), `--interval MS` (1 to 1000, default 10), `--device N`, `--csv FILE`, `--json FILE`.

## 3. Python
```python
import pravaha_mem as pv
with pv.observe(interval_ms=10, device=0) as m:
    m.metadata({...}); m.mark("phase"); ...
m.summary(); m.total(); m.phases(); m.baseline("idle"); m.quality(); m.samples
m.to_csv(path); m.to_json(path); m.to_dict()
rc, m = pv.run(["./app"], idle_s=2)        # same as `pravaha run`
pv.status(); pv.verify(json_path=None)
```
`total()` and each phase give bytes: `copy`, `kernel_read`, `kernel_write`, `all_read`, `all_write`, `memory` (all_read + all_write), plus `seconds` and `boundary_share` (the part of the phase total that came from samples split at a phase edge).

## 4. Results
CSV (phase `before_start` or `after_end` for samples at the edges of the window): `t_start_ns, t_end_ns, phase, copy_bytes, kernel_read_miss_bytes, kernel_write_bytes, all_read_miss_bytes, all_write_bytes`; times on the system clock (same as `time.time_ns()`, `date +%s%N`).

JSON: version and reader protocol; scope; provenance (GPU, UUID, driver, CUDA, host, kernel); every metric with its counter and conversion; totals and phases; the idle baseline with its variation; sampling quality (samples, gaps, time not sampled, observed interval, clock pairs and offset); your metadata.

## 5. Reading the numbers
* **copy** is exact for host to GPU and GPU to host copies (pinned, registered, huge page, pageable host memory).
* **kread / kwrite**: what GPU code moved; data that hits in L2 does not cross C2C and is not counted.
* **Startup**: each CUDA program moves a small fixed amount (about 15 MiB of copies for a plain CUDA program). Use phases, or compare two runs.
* **Phases** shorter than 10 samples have fuzzy edges; the summary says so. Use `interval_ms=1` for millisecond phases.
* **Idle baseline** (`--idle SECONDS`, or a phase named `idle`): see section 5a.

## 5a. The idle baseline: purpose and math

**Purpose.** pravaha counts every byte on the GPU, from every program: the desktop redrawing the screen, other users, a server in the background. `--idle` first watches the GPU for a few seconds before your command starts, to learn how much that background moves on its own. Then pravaha can estimate what your command added.

**Math.** With the idle row measured for $t_\text{idle}$ seconds and the run row for $t_\text{run}$ seconds, for each column (copy, kread, kwrite, C2C):

$$\text{background rate} = \frac{B_\text{idle}}{t_\text{idle}} \qquad
B_\text{command} \approx B_\text{run} - \frac{B_\text{idle}}{t_\text{idle}} \times t_\text{run}$$

The table prints this as the row `run - idle` (never below zero); the JSON has the same values as `estimate_run_minus_idle`, and Python as `m.estimate()`. The raw samples (CSV, `m.samples`) are never changed, so you can apply your own method instead.

**Example** (GN100, `pravaha run --idle 2 -- python -m pravaha_mem.known_bytes h2d 8 256`): idle kread 0.27 GiB in 1.00 s, so the background rate is 0.27 GiB/s; run lasted 1.10 s with kread 0.31 GiB; estimate 0.31 - 0.27 x 1.10 = 0.01 GiB. The test program reads nothing with GPU code, so about 0 is right: the 0.31 GiB was the desktop.

**Limits.**
* It assumes the background during the run was the same as during idle. If another program started or stopped in between, the estimate is wrong. The JSON (`baseline`) gives the background's variation from sample to sample: a large variation means the estimate is rough.
* It is an estimate, never an exact count of one program. Raw values (`run`) are always shown next to it.
* Copies by other programs are rare, so the copy column usually needs no baseline; GPU code reads and writes (kread, kwrite, C2C) are where background matters.
* **Sample length**: the GPU closes a sample at the requested interval when it is busy; when it is mostly idle samples last longer (about 22 ms at a 10 ms request on GB10). `quality()` reports the observed interval. Totals are unaffected: every sample covers its own time span.
* **Clocks**: the reader records the GPU sample clock and the system clock together every second. On GB10 they agree within 1 microsecond (measured offset -528 ns, spread 648 ns), so samples join with power, energy or logs on the system clock.

## 6. Errors
| message | meaning |
|---|---|
| GPU counters busy | another profiler holds the sampling unit; stop it and try again |
| pravaha-reader not found | run `pravaha install` |
| insufficient privileges | the reader lost its capability (rebuilt or copied); run `pravaha install` again |
| thermal headroom too low | `verify` stopped to protect the machine; let it cool |

## 7. Safety
pravaha's own workloads (`verify`, `known_bytes`) are light: copies and simple writes and reads of a few GiB, about 1 s of GPU work per check, with a cool down between checks and a thermal headroom guard (start at 35 C or more of headroom, stop below 15 C). pravaha waits 1 s between stopping and starting readers, as the driver needs time to release the counters.

## 8. Words used
| word | meaning |
|---|---|
| host | the computer side: CPU and the memory programs normally use |
| device | the GPU |
| copy engine | GPU hardware that copies blocks of memory (`cudaMemcpy`, `.cuda()`, `.cpu()`) |
| kernel, GPU code | a program running on the GPU's cores (matrix multiply, attention, `cudaMemset` ...) |
| L2 cache | the GPU's last cache; every byte the GPU fetches from or sends to memory passes it |
| sector | the L2 cache's unit of 32 bytes; the counters count sectors |
| NVLink C2C | the link between the CPU die and the GPU die on GB10; the GPU's memory traffic crosses it |
| device wide | counts every process on the GPU, not one program |
| headroom | degrees between the GPU's temperature and its thermal limit |
| capability | a single Linux permission attached to one file (here: may read performance counters) |
| SMMU, translation | the unit that turns program addresses into memory addresses, one page at a time |

## 9. Developer notes
* `PRAVAHA_READER`: path of another reader binary. `PRAVAHA_PMCG`: path of another SMMU reader.
* `python -m pravaha_mem.known_bytes DIRECTION TIMES SIZE_MIB` moves TIMES x SIZE_MIB: `h2d` (host to device: computer memory into the GPU, copy engines), `d2h` (device to host, copy engines), `d2d` (GPU buffer to GPU buffer, GPU code), `memset` (GPU code fills a buffer). `--help` explains everything. `PRAVAHA_KNOWN_PAUSE=S` sets the pause between setup and work (default 0.5 s); `PRAVAHA_KNOWN_WINDOW=1` prints when the work ran (used by verify).
* Reader protocol v2 (stdout): `V 2`; `M names`; `C cupti_ns realtime_ns monotonic_ns`; `S idx t0 t1 v1..v5` (sectors); `E ok n | E busy | E error code`.
* Tests without a GPU: `pip install .[test] && python -m pytest tests`. On a GB10 with pravaha installed: `python -m pytest -m gpu tests` (light).

## Schedules for energy experiments

`python -m pravaha_mem.schedule FILE.json [--device 0]` runs the slots of FILE in order, in one process with one CUDA context and every buffer prepared before the first slot, so slots differ only in the work they do.

### Slot kinds
| Kind | Keys | Work |
|---|---|---|
| `idle` | `seconds` | sleep |
| `noop` | `pace_ms`, `seconds` | one empty kernel per pace (sham: the submission loop, no data) |
| `kernel` | `op` read or write, `footprint_mib`, `logical_mib`, optional `placement` (device, pinned, managed, pageable) | passes over the footprint until `logical_mib` are done; `logical_mib` must be a multiple of `footprint_mib` |
| `copy` | `direction` h2d or d2h, `host` pinned or pageable, `size_mib`, and `count`, or `pace_ms` with `seconds` | copies back to back, or one per pace |
| `compute` | `iters`, `launches` | fused multiply adds on registers |
| `overlap` | `compute` {`iters`, `launches`}, `transfer` (a copy or kernel part) | both issued on two streams at once |
| `square` | `period_s`, `periods`, `on` (copy, kernel or noop part with `pace_ms`), `off` (idle or noop) | lock in modulation; every half reported |

### Output (stdout)
`SETUP {json}` buffers and compile, before any slot. `SLOT {json}` per slot: `index`, `name`, `kind`, `t0`, `t1` (ns, `time.time_ns()` clock), `ops`, `payload_bytes`, `check` (read checksum), `detail`. `HALF {json}` per square wave half: `slot`, `period`, `half`, `t0`, `t1`, `ops`, `payload_bytes`. `DONE {json}` totals. Exit 0 done, 1 CUDA error, 2 invalid schedule, 3 a check failed.

### What the numbers are
Payload bytes are program counted: what the code asked to move. The bytes that reached memory come from the observer over the slot's window; a read of a footprint inside the 24 MiB L2 reaches memory only on its first pass. A pageable copy may be staged through a pinned buffer by the driver. A square wave half includes its pacing waits; achieved rates follow from its payload and duration.

### Verification
`pravaha verify --schedule` runs a fixture under the observer and checks checksums, payloads, streaming and resident read misses, the sham, a copy and square wave timing, against `[schedule]` of `pravaha.ini`.
