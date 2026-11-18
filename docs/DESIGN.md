# pravaha-mem 1.0: system design (revision 2)

Revision 2 adds: precise terms (L2, memory, C2C), the byte conversion and its reason, `cap_perfmon` (tested), installer safeguards, device wide wording, baseline reporting, clock mapping, boundary samples, sampling validation, loss accounting, a narrower meaning of PASS, provenance, metadata, and safety rules for the user's machine.

## 1. Problem
On the NVIDIA GB10 the GPU has no memory of its own. Every byte it reads or writes goes to the LPDDR5X attached to the CPU die, over NVLink C2C. NVIDIA exposes no counter for that link on GB10, so nobody can see how many bytes a GPU program moves, or when.

## 2. What pravaha measures, in precise terms
| term | meaning | how pravaha gets it |
|---|---|---|
| L2 traffic | sectors counted at the GPU's L2 cache (32 bytes each) | measured directly: 5 counters, CUPTI PM sampling |
| memory traffic | L2 read misses (data fetched from memory) and L2 writes (data sent toward memory) | measured: the miss and write counters |
| C2C traffic | bytes physically crossing NVLink C2C | inferred: on GB10 all GPU memory is the shared LPDDR5X behind the CPU die, so GPU memory traffic must cross C2C. Supported by the known byte checks (ratio 1.0000 on two machines). Not measured by a link counter, because none exists on GB10. |

The documentation and outputs use these terms exactly. "C2C" in the CLI table means GPU memory traffic, and the manual states the inference above.

## 3. Design rules
1. **One meaning per number.** Bytes and translations are never mixed; every value carries its unit and source counter.
2. **Observer only.** pravaha does not modify, instrument, restart or control the program it watches. It reads hardware counters from its own process. Its overhead is measured and documented, not assumed to be zero.
3. **Least privilege.** One install step, one password. Each reader file gets the smallest capability that works (tested: `cap_perfmon` is enough for the L2 reader on driver 580). No system setting changes.
4. **Leave no trace.** Install creates one folder with a manifest; uninstall removes exactly that and confirms nothing is left.
5. **Prove it, and say exactly what was proven.** `verify` compares pravaha with known bytes for named workloads on this machine and software, and reports each result.
6. **No hidden features.** Every capability is a documented command, option or API (section 12).
7. **Safe for the machine.** pravaha's own workloads are light, short, separated by cool downs, and guarded by the GPU's thermal headroom (section 9).

## 4. Actors
| actor | goal |
|---|---|
| Machine admin | allow counter reading once, safely; remove it cleanly |
| End user | see how many bytes a program or a running server moves |
| Developer | measure phases inside their own code |
| Researcher | a time series to join with power and energy; evidence; provenance; address translation studies |
| Integrator (A-LEMS) | start, stop, mark, attach metadata, read samples from a harness |

## 5. Features and commands

### 5.1 Main feature: GPU memory traffic (L2 counters)
| # | feature | who | command | Python |
|---|---|---|---|---|
| 1 | get the tool | anyone | `pip install pravaha-mem` or from GitHub | |
| 2 | allow counter reading on this machine | admin, once | `pravaha install` | |
| 3 | is this machine ready? | anyone | `pravaha status` | `pv.status()` |
| 4 | check the numbers against known bytes | anyone | `pravaha verify [--json FILE]` | `pv.verify()` |
| 5 | device wide traffic while one program runs | user | `pravaha run -- CMD` | |
| 6 | device wide traffic over a time window (a server) | user, operator | `pravaha observe [--seconds N]` | |
| 7 | live samples | user | `pravaha observe --live` | |
| 8 | phases and metadata in my own code | developer | | `with pv.observe() as m: m.mark("x"); m.metadata({...})` |
| 9 | save data with provenance | researcher | `--csv FILE`, `--json FILE` | `m.to_csv()`, `m.to_json()`, `m.samples`, `m.phases()`, `m.total()` |
| 10 | remove it | admin | `pravaha uninstall` | |

* **Device wide.** `run` and `observe` count all traffic on the GPU during their window, from every process. `run` only adds convenience: it starts CMD and stops when CMD ends. It does not isolate CMD's traffic.
* `status`: GPU, driver, CUDA and CUPTI versions; each reader installed, with which capability; counters free right now; current GPU temperature and thermal headroom.
* Options for `run` and `observe`: `--idle S` (baseline first), `--interval MS` (default 10), `--device N`, `--csv FILE`, `--json FILE`.

### 5.2 Second feature: address translations (GPU SMMU counters)
| # | feature | who | command |
|---|---|---|---|
| 11 | which SMMU counters exist, which is the GPU's | researcher | `pravaha smmu status` |
| 12 | translations while a program runs | researcher | `pravaha smmu run -- CMD` |
| 13 | translations live, whatever runs | researcher | `pravaha smmu observe [--seconds N] [--live]` |

Output is always in translations (`transaction` = translation requests, `tlb_miss` = translation misses), never bytes: bytes per translation vary with allocation type, page size and history.

### 5.3 Third feature: per kernel detail (Nsight Compute)
| # | feature | who | command |
|---|---|---|---|
| 14 | L2 traffic per named kernel | developer, researcher | `pravaha kernels [--match REGEX] -- CMD` |

Uses Nsight Compute replay: slower than `run`, and needs profiling permission. The help and manual say so.

## 6. Counters, conversion and accuracy
| output column | JSON name | counter | conversion | why |
|---|---|---|---|---|
| copy | `copy_bytes` | `lts__t_sectors_srcunit_ce` | sectors x 32 / 2 | the copy engine passes each copied byte through L2 twice (read and write); measured ratio 1.0000 against known copies |
| kread | `kernel_read_miss_bytes` | `lts__t_sectors_srcunit_tex_op_read_lookup_miss_realtime` | sectors x 32 | sectors read by GPU code that missed L2 |
| kwrite | `kernel_write_bytes` | `lts__t_sectors_srcunit_tex_op_write_realtime` | sectors x 32 | sectors written by GPU code |
| (JSON only) | `all_read_miss_bytes` | `lts__t_sectors_op_read_lookup_miss_realtime` | sectors x 32 | all sources |
| (JSON only) | `all_write_bytes` | `lts__t_sectors_op_write_realtime` | sectors x 32 | all sources |
| C2C | `memory_traffic_bytes` | all read misses + all writes | sum of the two above | GPU memory traffic; C2C by inference (section 2) |

* `_realtime` counters use a low cost counting method that NVIDIA describes as approximate. pravaha reports the accuracy measured by `verify` for each counter, not an assumed one. Measured so far: within 0.3 %.
* No two counters that count the same sectors are added together.
* **Open test (before 1.0):** aperture variants (`aperture_device`, `aperture_sysmem`) to show how GB10 classifies `cudaMalloc` and host buffers, and whether either filter changes the totals. Read only query first; then a light sampling test.

## 7. Time, phases and sampling
* **Clock mapping.** The reader records the CUPTI GPU timestamp and the host clock (`CLOCK_REALTIME` and `CLOCK_MONOTONIC`) together at start and every second, and outputs the mapping. Samples carry both GPU time and mapped host time, with the mapping uncertainty. Measured so far: host and sample times agree within 5 ms.
* **Boundary samples.** A sample that spans a phase mark is split by time share; each phase also reports `boundary_share`, the fraction of its total that came from split samples, so readers can judge how exact the split is.
* **Interval.** Default 10 ms. Other intervals are offered only after validation: 1, 2, 5 and 10 ms are tested for real spacing, missing samples and accuracy; the accepted range is documented. A phase shorter than 10 samples gets a note.
* **Loss accounting.** The reader drains the hardware buffer every 100 ms into two alternating counter images, so a reset never loses a sample. Every result reports `samples_received`, `samples_expected`, `samples_lost`, `time_not_sampled_ms`.

## 8. Baseline and attribution
Every result reports separately: the raw device wide value; the idle baseline rate (if `--idle` or an `idle` phase was measured); the baseline subtracted estimate; and the variation of the baseline (so a user can see if background traffic was stable). The subtracted value is labelled an estimate. pravaha never presents it as one program's exact traffic.

## 9. Safety for the user's machine
1. pravaha's own workloads (`verify`, `known_bytes`) are light: copies and simple GPU reads and writes of a few GiB, about 1 s of GPU work per check. No dense compute.
2. 30 s cool down between verify checks is configurable; a minimum gap of 1 s between stopping one reader and starting the next (the driver needs time to release the counters).
3. Thermal guard: before each verify check, read `temperature.gpu.tlimit` (headroom to the thermal limit). Start only if headroom >= 35 C; abort if it drops below 15 C during a check. On GB10 the slowdown is at headroom -2 C and shutdown at -5 C; the hardware cuts power without any OS log, so the guard must act early.
4. `status` shows temperature and headroom, so users see the machine state before measuring.
5. Example and tutorial workloads are chosen the same way; nothing in the docs asks the user to run sustained full load.

## 10. Install and uninstall contract
`pravaha install` is the only command that touches the system. It asks for the password once and creates exactly:
```
/usr/local/libexec/pravaha/          root owned, mode 0755
    pravaha-reader   L2 reader      cap_perfmon      root owned, mode 0755
    pravaha-pmcg     SMMU reader    cap_perfmon      root owned, mode 0755
    MANIFEST         files, sizes, sha256, version, date, capability   root owned, mode 0644
```
Safeguards:
* the folder and files are root owned and not writable by users;
* `--prefix DIR` is accepted only if DIR is root owned and not writable by users; otherwise install refuses;
* readers load no code from user paths; their only inputs are fixed options;
* the capability is chosen by test at install time: `cap_perfmon` first; if the driver refuses it, install reports that and offers the documented alternative instead of silently using a broader one;
* uninstall deletes only paths listed in MANIFEST that lie inside the install folder, checks each sha256 first, removes the folder, and confirms "pravaha removed: nothing left on this system" or lists what remains;
* nothing else changes: no system setting, driver option, group, sudoers entry, service, or file in a home folder;
* full removal: `pravaha uninstall`, then `pip uninstall pravaha-mem`. The README states the footprint before the install instructions, with the one line manual cleanup `sudo rm -r /usr/local/libexec/pravaha`.

Why a password is needed: NVIDIA's driver lets only privileged processes read GPU counters, and only root can write to `/usr/local` and attach a capability. The readers then run as the normal user with that one capability. The capability is broader than "these five counters"; the reader's code limits itself to reading the counters pravaha needs.

## 11. Architecture
```
user / harness
   |  CLI (cli.py)                Python API (observe, status, verify)
   v
Observer (observer.py)  start / mark / metadata / stop / samples / phases / summary / export
   |  starts the reader, reads its stdout line protocol
   v
pravaha-reader (C++)  CUPTI PM sampling of 5 L2 counters, whole GPU, clock mapping
   v
GPU L2 cache counters

smmu/       SMMU discovery, pravaha-pmcg reader, run and observe (translations)
kernels.py  Nsight Compute per kernel pass
```
| component | file | responsibility |
|---|---|---|
| L2 reader | `reader.cpp` | sample 5 counters; two alternating images; clock mapping lines; clear busy error |
| observer | `observer.py` | start reader, wait until samples flow, collect, phases, boundary share, loss accounting, export with provenance |
| install | `install.py` | build, capability test, copy, setcap, MANIFEST; uninstall with checks |
| status | `status.py` | versions, readers, capability, counters free, temperature and headroom |
| verify | `verify.py`, `known_bytes.py` | known byte workloads, thermal guard, cool downs, background, middle of 3, detailed report |
| SMMU | `smmu/` | discover PMCGs, `pravaha-pmcg`, run and observe in translations |
| kernels | `kernels.py` | Nsight Compute per kernel L2 traffic |
| CLI | `cli.py` | thin mapping of commands to the above |

Reader protocol (stdout), version 2: `V 2` once; `M names` once; `C gpu_ns realtime_ns monotonic_ns` at start and every second; `S idx gpu_t0 gpu_t1 v1..v5` per sample (sectors); `L n` if n samples were lost; `E ok|busy|error`.

## 12. Results and provenance
Every JSON result contains: pravaha version; reader and protocol version; GPU name, UUID, architecture; driver, CUDA and CUPTI versions; metric names, units and conversions; interval requested and observed; scope ("device wide"); clock domains and mapping uncertainty; sample and loss counts; baseline method and values; phases with `boundary_share`; user metadata; and, if run, the last `verify` result on this machine.

`verify` reports, per check: workload, metric, expected bytes, measured bytes (unrounded), relative error, tolerance, the three runs, background, GPU headroom before and after, and PASS, FAIL or NOT SUPPORTED. PASS means: this counter agreed with this known workload within the stated tolerance, on this device, driver and software. It is not a claim about every workload.

## 13. Non functional requirements
* No root to use after install; one line errors, no tracebacks.
* Device wide scope stated in every output.
* One L2 reader at a time per GPU (hardware); a second gets "counters busy".
* No lost samples on long runs, or an exact count of what was lost.
* Overhead measured by a light A/B test (same light workload with and without pravaha, cool downs between runs, thermal guard), and documented.
* Verified so far: ratio 1.0000 on Acer GN100 and DGX Spark for copies and kernel writes; kernel reads within 0.3 %.

## 14. Feature inventory: today's code and where it goes
| today | file today | in 1.0 |
|---|---|---|
| L2 sampling, busy error, buffer, `--device` | `pravaha_c2c.cpp` | `reader.cpp`; adds clock mapping, alternating images, protocol v2 |
| start waits for samples, marks, edge split, notes, summary, dicts, samples, CSV | `c2c_sampler.py` | `observer.py`; adds metadata, boundary share, loss counts, JSON provenance |
| run with idle, Ctrl+C, observe | `c2c_sampler.run()` | `run`, `observe` |
| live CSV stream | `l2_commands.cmd_watch` | `observe --live` |
| readiness report | `l2_commands.cmd_show`, `permissions.py` | `status`, adds temperature and headroom |
| build, install, uninstall | `c2c_sampler` install functions | `install.py` with MANIFEST, safeguards, capability test |
| known byte demo | `known_bytes.py` | `known_bytes.py` |
| verification with JSON detail | `c2c_checks.py` | `verify.py`, adds thermal guard, cool downs, full report |
| reader location variable | `PRAVAHA_C2C` | `PRAVAHA_READER` |
| SMMU discovery, identity | `discover.py`, `identity.py` | `smmu/` |
| perf reader, `pravaha-pmcg` | `perf.py`, `helper.py`, `pravaha_pmcg.c` | `smmu/`, installed by `install` |
| SMMU measure, watch | `measure.py`, `copycounter.py` | `smmu run`, `smmu observe`, translations only |
| calibration to bytes | `calib.py`, `calibrate.py`, `config.py` | removed: translations do not convert to bytes reliably |
| SMMU validation checks | `checks.py` | research folder (paper evidence) |
| Nsight Compute kernel pass | `ncu.py` | `kernels.py` |
| CUDA test programs, Makefile | `src/`, `Makefile` | research folder |
| unit tests | `tests/` | rewritten for 1.0 |

## 15. Open items before building
1. Aperture variants on GB10 (section 6): read only query, then a light test.
2. Interval validation at 1, 2, 5, 10 ms (section 7): light workload only.
3. Light A/B overhead test (section 13), with thermal guard and cool downs; plan shown to the owner first.
