# pravaha manual (1.0)

## 1. What pravaha measures
| term | meaning | how |
|---|---|---|
| L2 traffic | 32 byte sectors counted at the GPU's L2 cache | measured: 5 counters, CUPTI PM sampling |
| memory traffic | L2 read misses (fetched from memory) and L2 writes (sent toward memory) | measured |
| C2C traffic | bytes crossing NVLink C2C | inferred: on GB10 all GPU memory is the shared LPDDR5X behind the CPU die; supported by `verify`. No link counter exists on GB10. |

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
CSV: `t_start_ns, t_end_ns, phase, copy_bytes, kernel_read_miss_bytes, kernel_write_bytes, all_read_miss_bytes, all_write_bytes`; times on the system clock (same as `time.time_ns()`, `date +%s%N`).

JSON: version and reader protocol; scope; provenance (GPU, UUID, driver, CUDA, host, kernel); every metric with its counter and conversion; totals and phases; the idle baseline with its variation; sampling quality (samples, gaps, time not sampled, observed interval, clock pairs and offset); your metadata.

## 5. Reading the numbers
* **copy** is exact for host to GPU and GPU to host copies (pinned, registered, huge page, pageable host memory).
* **kread / kwrite**: what GPU code moved; data that hits in L2 does not cross C2C and is not counted.
* **Startup**: each CUDA program moves a small fixed amount (about 15 MiB of copies for a plain CUDA program). Use phases, or compare two runs.
* **Phases** shorter than 10 samples have fuzzy edges; the summary says so. Use `interval_ms=1` for millisecond phases.
* **Baseline**: subtracting an idle rate is an estimate; it assumes the background stayed the same. The JSON gives its variation.
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
pravaha's own workloads (`verify`, the demo) are light: copies and simple writes and reads of a few GiB, about 1 s of GPU work per check, with a cool down between checks and a thermal headroom guard (start at 35 C or more of headroom, stop below 15 C). pravaha waits 1 s between stopping and starting readers, as the driver needs time to release the counters.

## 8. Developer notes
* `PRAVAHA_READER`: path of another reader binary. `PRAVAHA_PMCG`: path of another SMMU reader.
* `python -m pravaha_mem.demo MODE COPIES MIB` (modes h2d, d2h, d2d, memset) moves known bytes; `PRAVAHA_DEMO_QUIET=S` waits S seconds between setup and work; `PRAVAHA_DEMO_WINDOW=1` prints when the work ran.
* Reader protocol v2 (stdout): `V 2`; `M names`; `C cupti_ns realtime_ns monotonic_ns`; `S idx t0 t1 v1..v5` (sectors); `E ok n | E busy | E error code`.
* Tests without a GPU: `pip install .[test] && python -m pytest tests`.
