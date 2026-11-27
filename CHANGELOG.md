# Changelog

## 1.2.1 (2026-11-27)
Docs only, no code change: full documentation review (listed under 1.2.0 Docs below); PyPI page now shows the corrected README.

## 1.2.0 (2026-11-27)

New
- `schedule`: a timestamped schedule of GPU workload slots for energy experiments, run in one process with one CUDA context so slots differ only in their work. Slot kinds: `idle`; `noop` (paced empty kernels, a sham control with the submission loop of the other slots and no data); `kernel` (read or write over a chosen footprint until a logical byte count is done, so a cache resident and a streaming footprint can issue the same operations; read checksums verified outside the slot); `copy` (host to device or device to host, pinned or pageable host memory, back to back or paced); `compute` (an arithmetic kernel on registers); `overlap` (a compute part and a copy or kernel part on two streams); `square` (lock in modulation: timed on and off halves, each reported). Machine lines `SETUP`, `SLOT`, `HALF`, `DONE` carry wall clock windows (the clock of `time.time_ns()`) and program counted payload bytes. Compiled at run time with NVRTC, nothing to build. `pravaha schedule FILE.json` (a table per slot; `--json OUT`, `--raw`), `pravaha schedule --template` (a commented starter file), examples in `examples/schedules/`, lesson 9 of docs/LEARN.md; Python: `schedule.validate()`, `run_schedule()`, `parse()`.
- `pravaha verify --schedule`: the schedule runner checked under the observer. Exact: checksums and payload. Observed, background subtracted: streaming read misses per payload byte within `[verify] miss_tolerance`; resident read misses per payload byte at most `resident_max_ratio`; sham memory traffic at most `sham_max_bytes_per_s`; copy bytes per payload byte within `copy_tolerance`; square wave halves within `half_tolerance`. Settings in `[schedule]` of `pravaha.ini`.

Docs
- Schedules for energy experiments: README, MANUAL (section 10: slot kinds, output, the built in test, designing a schedule), LEARN lesson 9 (the idea), TUTORIAL lesson 7 (hands on), DEVELOPER_GUIDE (from Python), RESEARCHER_GUIDE (accuracy rows, checklist).
- `verify --kernels` and `known_kernel` (1.1.0) now in the README, MANUAL and THE_WHOLE_IDEA command lists.
- Corrected: thermal guard values in MANUAL and THE_WHOLE_IDEA (10 C and 5 C from `pravaha.ini` since 1.1.0, not 35 C and 15 C); RESEARCHER_GUIDE overhead (measured in 1.0.1); README install line (on PyPI); LEARN now nine lessons; README lists LEARN.

## 1.1.0 (2026-11-24)

New
- `known_kernel`: a GPU kernel that reads or writes an exact, known amount of data in a chosen memory placement (device `cudaMalloc`, pinned `cudaMallocHost`, managed `cudaMallocManaged` with optional prefetch, pageable `aligned_alloc`). It is compiled at run time with NVRTC, so nothing needs building. The checksum is verified outside the work window, and the first pass is timed separately. Memory attributes before and after are recorded as requests and policy, not residency. `--hold S` gives a duration matched control with no kernels, and `--query` lists the supported placements. Python: `known_kernel.run()`, `query()`, `parse()`.
- `pravaha verify --kernels`: logical bytes checked exactly in each placement. Counter traffic is reported as observed bytes per logical byte. A miss fixture (1 GiB device read) is judged within 5%.
- `pravaha kernels --probe` and `kernels.probe()`: can kernel traffic be measured here.

Changed (kernel pass)
- A counter Nsight Compute could not read is None, never 0. The aggregate of that direction is None and the other direction stays complete.
- `kernel_traffic(name_base="demangled")` by default, so a regex sees template arguments (`MulFunctor`).
- `timeout=`, and distinct errors: `NoPermission`, `NoKernelMatched`, `NoKernelData`, `WorkloadFailed` (with the kernels profiled before the failure).
- The result records `ncu_args`, `ncu_version` and `metrics`.

- Thermal gate of `verify`: start_c 10 and abort_c 5 (headroom below the GPU's own limit) now live in `pravaha.ini` shipped with the library, replaceable by a file named in `PRAVAHA_CONFIG`, `PRAVAHA_MIN_HEADROOM_START_C` / `_ABORT_C`, or arguments; the values and their sources are in the report. The fixed 35 C and 15 C of 1.0.x blocked every check on a GB10 idling at 61 C.

Fixed
- `__version__` reported 1.0.0 in release 1.0.1.

Docs
- `cudaMemset` and `cudaMemcpy` are driver operations, not profiled kernels.

## 1.0.1 (2026-11-20)

Documentation and validation release; no change to the measurement code or its numbers.

* Overhead of the separate reader measured on both machines: `validation/overhead_ab.py` (light A/B timing, thermal guard). GN100 -0.04%, Spark -0.53%, 20 runs each, inside the run to run spread: no measurable cost.
* VALIDATION.md: overhead results; which counter names exist (Nsight Compute query) and why the aperture counters cannot be sampled live; Nsight Systems GPU metrics on GB10 report no memory traffic; the GN100 read check and the GPU to GPU breakdown explained; the 1,510 token LLM run; the write back limit of the L2 write counter.
* TEST_MATRIX.md: every row now tested on both machines (DGX Spark rows added 2026-11-20); row 19, overhead.
* VALIDATION.md: every feature on the DGX Spark, with the GN100 for comparison.
* New docs/LEARN.md: the course in eight lessons.
* CITATION.cff: leftover placeholder removed.

## 1.0.0 (2026-11-18)

First public release.

* Live, device wide GPU memory traffic on the NVIDIA GB10 from the GPU's L2 cache counters (CUPTI PM sampling), observed from a separate process: copy engines, GPU code reads and writes, and all GPU memory traffic (which crosses NVLink C2C on GB10).
* Commands: `install`, `status`, `verify`, `run -- CMD`, `observe [--seconds N] [--live]`, `smmu status|run|observe`, `kernels -- CMD`, `uninstall`.
* Python: `pravaha_mem.observe()` with `mark()`, `metadata()`, `summary()`, `total()`, `phases()`, `baseline()`, `quality()`, `to_csv()`, `to_json()`.
* JSON results with provenance, metric definitions and conversions, idle baseline and its variation, sampling quality and clock mapping.
* Least privilege install: one folder in `/usr/local/libexec/pravaha` with two readers, `cap_perfmon` each, and a MANIFEST; clean, checked uninstall. No system setting is changed.
* `verify`: light known byte workloads (`python -m pravaha_mem.known_bytes`), thermal headroom guard, cool downs, background subtraction, middle of three runs, detailed JSON report.
* SMMU address translation counts (never converted to bytes) and Nsight Compute per kernel traffic as separate features.
