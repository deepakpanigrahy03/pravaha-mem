# Changelog

## 1.0.0 (2026-11-18)

First public release.

* Live, device wide GPU memory traffic on the NVIDIA GB10 from the GPU's L2 cache counters (CUPTI PM sampling), observed from a separate process: copy engines, GPU code reads and writes, and all GPU memory traffic (which crosses NVLink C2C on GB10).
* Commands: `install`, `status`, `verify`, `run -- CMD`, `observe [--seconds N] [--live]`, `smmu status|run|observe`, `kernels -- CMD`, `uninstall`.
* Python: `pravaha_mem.observe()` with `mark()`, `metadata()`, `summary()`, `total()`, `phases()`, `baseline()`, `quality()`, `to_csv()`, `to_json()`.
* JSON results with provenance, metric definitions and conversions, idle baseline and its variation, sampling quality and clock mapping.
* Least privilege install: one folder in `/usr/local/libexec/pravaha` with two readers, `cap_perfmon` each, and a MANIFEST; clean, checked uninstall. No system setting is changed.
* `verify`: light known byte workloads (`python -m pravaha_mem.demo`), thermal headroom guard, cool downs, background subtraction, middle of three runs, detailed JSON report.
* SMMU address translation counts (never converted to bytes) and Nsight Compute per kernel traffic as separate features.
