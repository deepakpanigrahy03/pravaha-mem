# Validation record

Every test that supports pravaha 1.0's numbers: when, where, the command, what was expected, what was measured. Raw reports are in [validation/](../validation). The summary per feature and machine is the [test matrix](TEST_MATRIX.md).

Machines: Acer GN100 (GB10, with desktop) and NVIDIA DGX Spark (GB10, headless), driver 580.159.03, CUDA 13.0. All workloads are light (copies and simple writes and reads of a few GiB).

## 1. `pravaha verify` (2026-11-18, release 1.0)
Each check is the middle of three runs, background subtracted, measured only while the work ran.

| check | expected | GN100 | Spark |
|---|---|---|---|
| idle: no copies | 0 MiB | 0.0 | 0.0 |
| copy 2 GiB into the GPU (`known_bytes h2d 8 256`) | 2048 MiB | 2048.0, ratio 1.0000 | 2048.0, 1.0000 |
| copy 2 GiB out of the GPU (`d2h 8 256`) | 2048 MiB | 2048.0, 1.0000 | 2048.0, 1.0000 |
| GPU code writes 4 GiB (`memset 16 256`) | 4096 MiB | 4096.0, 1.0000 | 4096.0, 1.0000 |
| GPU code reads 4 GiB (`d2d 16 256`) | 4096 MiB | 4084.6, 0.9972 | 4095.9, 1.0000 |

Files: `validation/verify_gn100_2026-11-18.json`, `validation/verify_spark_2026-11-18.json`.

## 2. Each path by hand (`pravaha run -- python -m pravaha_mem.known_bytes ...`, GN100)
These include the program's own startup and setup, so they show what a user sees.

| command | known_bytes says | copy | kread | kwrite | C2C | explained |
|---|---|---|---|---|---|---|
| `h2d 8 256` | copied 2 GiB in; cleared 2 x 256 MiB | 2.0147 | 0.31 | 0.5000 | 5.72 | 2 GiB copied + 15 MiB CUDA startup; kwrite = setup; C2C = 4 (copy read and write) + 0.5 + desktop |
| `h2d 4 512` | copied 2 GiB in; cleared 2 x 512 MiB | 2.0158 | 0.03 | 1.0000 | 5.35 | same 2 GiB; setup now 1 GiB |
| `h2d 16 128` | copied 2 GiB in; cleared 2 x 128 MiB | 2.0141 | 0.02 | 0.2500 | 4.53 | same 2 GiB; setup now 0.25 GiB |
| `memset 16 256` | wrote 4 GiB; cleared 2 x 256 MiB | 0.0147 | 0.02 | 4.4999 | 4.80 | 4 + 0.5 written; no copy engine work; writes cross once |
| `d2d 16 256` | copied 4 GiB GPU to GPU; cleared 2 x 256 MiB | 0.0147 | 4.0292 | 4.4999 | 8.82 | reads 4 (L2 misses), writes 4 + 0.5; on GB10 a GPU to GPU copy crosses C2C twice |

Conclusions: pravaha counts bytes, not the number of copies (three splits of 2 GiB agree); every column is accounted for by the program's own steps.

## 3. One reader at a time (Spark, 2026-11-18)
A second reader started while one was running reported "GPU counters busy" and exited; the first finished normally (`E ok 291`, 291 samples in 10 s).

## 4. Install and permission (both machines, 2026-11-18)
`pravaha install` with `cap_perfmon` only: the reader samples (`status`: counters free). `cap_sys_admin` is not needed with driver 580.

## 5. Time and sampling (GN100, 2026-11-18)
`observe --seconds 3`: 150 samples, 0 gaps; GPU and system clocks agree within 1 µs (offset -528 ns, spread 648 ns); observed sample length about 22 ms at a 10 ms request on a mostly idle GPU. A 100 s run had 0 gaps (before 1.0, same reader logic).

## 6. SMMU translations (GN100, 2026-11-18)
`smmu status`: 7 GPU SMMU units, default `smmuv3_pmcg_13002` (the translation control unit), readable via perf. `smmu run -- known_bytes h2d 8 256`: 1,295,537 translation requests, 671,242 translation misses in 1.20 s.

## 7. Real frameworks (before 1.0, same counters and conversions)
| workload | measured | expected |
|---|---|---|
| PyTorch `.cuda()`, 8 minus 4 copies of 1 GiB | 3.9999 GiB | 4 GiB |
| vLLM serve Qwen3 4B, model load | 7.80 GiB copies | weights 7.49 GiB + startup |
| vLLM, 400 generated tokens in 17.22 s | 3,009 GiB kernel reads = 7.52 GiB per token; 187.6 GB/s | 7.49 GiB of weights read once per token |
| idle around the request | 0.02 to 0.03 GiB per 3 s | about 0 |

To be repeated on 1.0: see the [test matrix](TEST_MATRIX.md).

## 8. Known limits found during validation
* Live sampling on GB10 offers no aperture (memory type) split; `ConfigAddMetrics` rejects the aperture counters.
* Values are device wide; other GPU users (the desktop, another LLM) appear in kread and C2C. `verify` subtracts a background and takes the middle of three runs.
* Overhead (A/B with and without pravaha) not yet measured.
