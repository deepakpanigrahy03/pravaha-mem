# Validation record

Every test that supports pravaha's numbers: when, where, the command, what was expected, what was measured. Raw reports are in [validation/](../validation). The summary per feature and machine is the [test matrix](TEST_MATRIX.md).

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

The GN100 read check is the only one below 1.0000 (0.28% low): the desktop redraws the screen in bursts that read memory, and the background, measured just before the check, is subtracted at its average rate, so a burst pattern that differs during the check leaves a small error. The Spark has no desktop and agrees to the last digit shown.

## 2. Each path by hand (`pravaha run -- python -m pravaha_mem.known_bytes ...`, GN100)
These include the program's own startup and setup, so they show what a user sees.

| command | known_bytes says | copy | kread | kwrite | C2C | explained |
|---|---|---|---|---|---|---|
| `h2d 8 256` | copied 2 GiB in; cleared 2 x 256 MiB | 2.0147 | 0.31 | 0.5000 | 5.72 | 2 GiB copied + 15 MiB CUDA startup; kwrite = setup; C2C = 4 (copy read and write) + 0.5 + desktop |
| `h2d 4 512` | copied 2 GiB in; cleared 2 x 512 MiB | 2.0158 | 0.03 | 1.0000 | 5.35 | same 2 GiB; setup now 1 GiB |
| `h2d 16 128` | copied 2 GiB in; cleared 2 x 128 MiB | 2.0141 | 0.02 | 0.2500 | 4.53 | same 2 GiB; setup now 0.25 GiB |
| `memset 16 256` | wrote 4 GiB; cleared 2 x 256 MiB | 0.0147 | 0.02 | 4.4999 | 4.80 | 4 + 0.5 written; no copy engine work; writes cross once |
| `d2d 16 256` | copied 4 GiB GPU to GPU; cleared 2 x 256 MiB | 0.0147 | 4.0292 | 4.4999 | 8.82 | reads 4 (L2 misses), writes 4 + 0.5 setup; 8.82 = 4.03 + 4.50 + 0.29 from the desktop and CUDA startup; on GB10 a GPU to GPU copy crosses C2C twice |

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

On 1.0 (2026-11-18, GN100), from outside, Mistral 7B Instruct v0.3 served by vLLM (weights 14,496,078,512 bytes = 13.50 GiB): 200 tokens in 12.54 s; kernel reads 13.26 GiB per token; 15.9 tokens per second; 227 GB/s; no copies during generation; idle before and after 0.007 and 0.02 GiB. The embedding table (32,768 x 4,096 x 2 bytes = 0.25 GiB) is a lookup of one row per token, so 13.50 - 0.25 = 13.25 GiB is the expected read per token.

With a 1,510 token prompt (GN100, 2026-11-19): prefill 0.40 s, 38.2 GiB read, 7.88 GiB written (125 GB/s reads plus writes, about 2.9 passes over the weights); decode 13.47 GiB per token at 226 GB/s reads plus writes, the weights plus the KV cache (0.125 MiB per earlier token), about 83% of the 273 GB/s NVIDIA specifies for the memory.

`pravaha kernels` (Nsight Compute, 1.0, GN100): PyTorch `torch.ones(2**24)` then `.sum()`: 0.067 GB written, 0.067 GB read (64 MiB each). Driver operations (`cudaMemcpy`, `cudaMemset`) are not kernels to Nsight Compute ("No kernels were profiled" for known_bytes); they are visible to `run` and `observe`.

`pravaha uninstall` (1.0, GN100): removed the folder; `status` then reported not ready with the fix line; `install` restored it.

## 8. Overhead of the separate reader (both machines, 2026-11-20)
`python validation/overhead_ab.py`: one light GPU job (`known_bytes d2d 32 256`, 8 GiB copied GPU to GPU, about 0.07 s) timed by its own window, 20 times with nothing observing (A) and 20 times with the pravaha reader running as a separate process at 10 ms (B), in 5 blocks ordered A B B A A B B A, thermal guard (start at 35 C headroom or more, stop below 15 C), 10 s cool down between blocks.

| machine | without reader (median) | with reader (median) | difference | spread between runs |
|---|---|---|---|---|
| GN100 | 71.92 ms | 71.90 ms | -0.04% | 15.2% / 23.3% |
| Spark | 71.15 ms | 70.77 ms | -0.53% | 2.0% / 2.9% |

Conclusion: the reader has no measurable cost on the measured program; both differences are far inside the run to run spread. Files: `validation/overhead_ab_gn100.json`, `validation/overhead_ab_spark.json`.

## 9. Every feature on the DGX Spark (2026-11-20)
The same commands as the GN100 rows of the [test matrix](TEST_MATRIX.md), run on the headless DGX Spark with light work and 30 s cool downs; vLLM 0.23.0 and PyTorch 2.11.0 (CUDA 13.0) in their own environment, as on the GN100.

| feature | Spark result | GN100 for comparison |
|---|---|---|
| copy into the GPU, 3 splits of 2 GiB | copy 2.0147 / 2.0159 / 2.0141 GiB | 2.0147 / 2.0158 / 2.0141 GiB |
| observe 3 s | 112 samples, 0 gaps; clock offset -880 ns, spread 752 ns; samples about 27 ms on an idle GPU | 150 samples, 0 gaps; -528 ns, 648 ns |
| idle baseline | 0.08 GiB of total traffic in 2 s idle (no desktop) | desktop bursts every 100 ms |
| run - idle, `known_bytes h2d 8 256` | copy 2.0147, kread 0.0000, kwrite 0.5000 GiB | copy 2.0147, kread 0.005 |
| Python phases (`examples/02`) | load 1.000, compute 9.941, save 0.422 GiB | 0.992, 9.952, 0.421 |
| LLM from outside (`examples/03`, vLLM, Qwen3 4B Instruct 2507) | 400 tokens; 7.518 GiB read per token (weights 7.49 GiB); 23.6 tokens/s; 190.6 GB/s; no copies; idle before 0.08 GiB, after 0.52 GiB per 2 s | Qwen3 4B 7.52 GiB per token, 187.6 GB/s |
| SMMU translations, `h2d 8 256` | 1,267,880 requests, 709,203 misses in 1.31 s; 7 GPU units of 17 | 1,295,537 and 671,242 in 1.20 s |
| per kernel (`pravaha kernels`, PyTorch fill and sum of 64 MiB) | 0.067 GB written, 0.067 GB read | same |
| uninstall, status, install | nothing left; not ready with the fix line; reinstall ready | same |
| unit tests, GPU tests | 3 passed; 7 passed | 3 passed; 7 passed |

Conclusion: every feature gives the same result on both machines; the copy splits agree to the fourth decimal and the LLM read per token matches the weights on both.

## 10. Which counters exist, and which tools can read traffic (GN100, 2026-11-19)
Counter names, listed by Nsight Compute (needs root: `sudo env PATH=$PATH ncu --query-metrics --query-metrics-mode all`): `lts__t_sectors_op_read`, `lts__t_sectors_op_read_lookup_miss`, `lts__t_sectors_aperture_device`, `lts__t_sectors_aperture_sysmem` all exist. pravaha uses the `lookup_miss` reads (hits never leave the GPU); the aperture counters exist but PM sampling rejects them, so live sampling cannot split traffic by buffer type.

Nsight Systems GPU metrics (`sudo nsys profile --gpu-metrics-devices=all --gpu-metrics-set=gb20b -o /tmp/gm sleep 3`, then `nsys export --type sqlite` and `select metricName from TARGET_INFO_GPU_METRICS`): 19 metrics, clocks and activity percentages (copy engines, SMs, tensor cores, warps). None is memory or link traffic.

## 11. Known limits found during validation
* Live sampling on GB10 offers no aperture (memory type) split; `ConfigAddMetrics` rejects the aperture counters.
* Values are device wide; other GPU users (the desktop, another LLM) appear in kread and C2C. `verify` subtracts a background and takes the middle of three runs.
* The L2 write counter counts writes arriving at L2; data written more than once before eviction may be counted more often than it reaches memory. The checks above write each byte once and do not test this case.

## Schedule runner (1.2.0)

Date 2026-11-27, both machines, vLLM and other GPU programs stopped. Raw reports: [validation/schedule/](../validation/schedule).

### `pravaha verify --schedule` (each task sized by a probe to about 2 s, 1 s idle between tasks)
| check | limit | GN100 | SPARK |
|---|---|---|---|
| checksums (2 reads) and payloads (2 reads) | exact | ✅ | ✅ |
| streaming read (1 GiB, far above the 24 MiB L2): observed read misses per payload byte | 1 ± 0.05 | 1.000 | 1.000 |
| resident read (8 MiB, inside L2): observed read misses per payload byte | at most 0.05 | 1.8e-5 | 3.7e-6 |
| sham (paced empty kernels): memory bytes per second above idle | at most 1e8 | 8.7e7 | 1.0e5 |
| copy, pinned host to device: observed copy bytes per payload byte | 1 ± 0.05 | 1.000 | 1.000 |
| square wave half duration error | at most 0.05 | 0.0007 | 0.0008 |
| square wave off halves move no payload | exact | ✅ | ✅ |

The sham is close to zero on the headless DGX Spark; on the GN100 the desktop redraws the screen (the 100 ms bursts of row 10 of the test matrix), which stays below the limit.

### Rates (`pravaha schedule examples/schedules/probe_rates.json`)
| slot | GN100 GB/s | SPARK GB/s |
|---|---|---|
| read, 8 MiB resident in L2 | 1,161 | 1,165 |
| read, 1 GiB streaming from memory | 240 | 242 |
| copy host to device, pinned, 256 MiB | 59.2 | 58.9 |
| copy device to host, pinned, 256 MiB | 58.9 | 59.0 |
| copy host to device, pageable, 256 MiB | 59.3 | 59.3 |

Pageable copies reach the pinned rate on both machines. The probe slots are short (0.01 to 0.15 s) by design; they measure rates only.

### First run note
The first GN100 run of 1.2.0 code used fixed pass counts: tasks lasted 0.01 to 0.15 s, so windows widened by `window_pad_ms` caught neighbouring tasks (resident 0.995, sham 2.3 GB/s, copy 1.125 = 1 + one square wave copy). The released check sizes every task from a probe and separates tasks with idle.

### Tests
`python -m pytest tests`: 30 passed on both machines. `python -m pytest -m gpu tests`: 8 passed on both machines (includes a schedule of every slot kind on the GPU).
