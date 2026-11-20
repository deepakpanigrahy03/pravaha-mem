# Changelog

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
