# Changelog

## 1.0.0 (2026-11-18)

First public release.

* `pravaha c2c`: live GPU memory traffic over NVLink C2C on the NVIDIA GB10, from the GPU's L2 counters (CUPTI PM sampling), for any program, measured from a separate process. Reports copies (copy engines), kernel reads and writes, and everything that crossed C2C, per phase, with an idle baseline, as a table, CSV or JSON.
* `pravaha c2c --seconds N`: observe the whole GPU, for example a server that is already running.
* Python API: `with pravaha_mem.c2c() as m`, `m.mark()`, `m.summary()`, `m.phases()`, `m.total()`, `m.samples`, `m.to_csv()`; samples on the system clock (ns) to join with power, energy or logs.
* `pravaha install-sampler`: builds the reader `pravaha-c2c` and gives that one file `cap_sys_admin`, so users measure without sudo. No system setting is changed.
* `pravaha test`: checks the counters against a demo program that moves known bytes (`python -m pravaha_mem.demo`), with background subtraction and the middle of three runs.
* Clear errors: counters busy (another profiler), reader missing, missing capability; progress lines and Ctrl+C handling for long runs; a note when a phase is too short for the sampling interval.
* SMMU translation counters (Linux perf) kept as diagnostics: `pravaha show`, `watch`, `measure`, `test --smmu`.
* Docs: README, docs/TUTORIAL.md (five lessons with expected output), docs/MANUAL.md; examples for any command, Python phases, LLM bytes per token, and joining with other data.
