# Researcher guide: methodology, accuracy, and how to report results

## 1. Measurement model
| quantity | definition | status |
|---|---|---|
| L2 traffic | 32 byte sectors counted by the GPU's L2 (LTS) counters | measured |
| memory traffic | L2 read misses (fetched from memory) + L2 writes (sent toward memory) | measured |
| NVLink C2C traffic | bytes crossing the CPU to GPU link | inferred: on GB10 all GPU memory is the LPDDR5X behind the CPU die, so GPU memory traffic crosses C2C; supported by `verify`. No link counter exists on GB10. |

Counters (CUPTI PM sampling, one pass, no replay, device wide):

| field | counter | conversion |
|---|---|---|
| copy | `lts__t_sectors_srcunit_ce.sum` | sectors x 32 / 2 (each copied byte passes L2 as a read and a write) |
| kernel_read | `lts__t_sectors_srcunit_tex_op_read_lookup_miss_realtime.sum` | sectors x 32 |
| kernel_write | `lts__t_sectors_srcunit_tex_op_write_realtime.sum` | sectors x 32 |
| all_read | `lts__t_sectors_op_read_lookup_miss_realtime.sum` | sectors x 32 |
| all_write | `lts__t_sectors_op_write_realtime.sum` | sectors x 32 |

The `_realtime` counters use a low cost counting method that NVIDIA describes as approximate; measured agreement with known workloads is reported in `verify` and [VALIDATION.md](VALIDATION.md). Live sampling on GB10 does not offer an aperture (memory type) split; totals cover all memory.

## 2. Accuracy (measured)
| path | workload | ratio to known bytes |
|---|---|---|
| copy, host to GPU and back | 2 GiB | 1.0000 (GN100, Spark) |
| GPU code writes | 4 GiB | 1.0000 (GN100, Spark) |
| GPU code reads | 4 GiB | 0.9972 to 1.0000 |
| PyTorch `.cuda()` | 8 minus 4 copies of 1 GiB | 3.9999 / 4 |
| vLLM decode, Qwen3 4B | 400 tokens | 7.52 GiB per token vs 7.49 GiB weights |

Host buffer type does not change copy counts (pinned, registered, 2 MiB huge pages, pageable within 0.2 %).

## 3. Time and sampling
* Samples carry the GPU sample clock; the reader records it together with the system clock every second. On GB10 they agree within 1 µs (offset -528 ns, spread 648 ns). pravaha reports times on the system clock.
* Requested interval 10 ms; when the GPU is mostly idle, samples last longer (about 22 ms). `quality()` reports the observed interval.
* A sample that spans a phase edge is split by time share; `boundary_share` says how much of a phase total came from such samples.
* Long runs: no gaps observed over 100 s; any lost time is reported (`time_not_sampled_ms`).

## 4. Background and attribution
Values are device wide. Report the raw value, the idle baseline (`--idle`, `m.baseline()`), and the subtracted estimate separately; the baseline's variation shows whether subtraction is meaningful. `verify` uses background subtraction plus the middle of three runs.

## 5. Overhead
pravaha does not modify or instrument the measured program; it reads counters from its own process. Its overhead has not yet been quantified with an A/B test; this is listed in the [test matrix](TEST_MATRIX.md). Run any A/B test with light workloads and cool downs (see the safety note in the manual).

## 6. Reproducibility checklist for a paper
1. `pravaha status` output (GPU, driver, CUDA).
2. `pravaha verify --json` on the machine used, attached as supplementary data.
3. Results saved with `--json` (provenance, metric definitions, conversions, quality, baseline).
4. The interval used, and whether phases were shorter than 10 samples.
5. What else ran on the GPU (device wide counters).

## 7. Cite
Use [CITATION.cff](../CITATION.cff) (GitHub shows "Cite this repository").
