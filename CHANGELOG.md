# Changelog

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
