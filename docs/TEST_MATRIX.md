# Test matrix: what has been tested, where

Every feature, every machine. Please help fill the gaps: run the commands on your GB10 and send the result (see "Report your results" below).

Legend: ✅ passed · ❌ failed · ⬜ not yet tested · — not applicable

## Machines
| id | machine | OS, kernel | driver | CUDA | date |
|---|---|---|---|---|---|
| GN100 | Acer Veriton GN100 (GB10) | Linux, with desktop | 580.159.03 | 13.0 | 2026-11-18 |
| SPARK | NVIDIA DGX Spark (GB10) | Linux, headless | 580.159.03 | 13.0 | 2026-11-18 |

## Features
| # | feature | command | GN100 | SPARK |
|---|---|---|---|---|
| 1 | install with `cap_perfmon` | `pravaha install` | ✅ | ✅ |
| 2 | readiness report | `pravaha status` | ✅ | ✅ |
| 3 | counters busy when another reader holds them | reader running + `status` / second reader | ✅ | ✅ |
| 4 | verification, all checks | `pravaha verify` | ✅ ratios 1.0000, 1.0000, 1.0000, 0.9972 | ✅ ratios 1.0000 x 4 |
| 5 | copy path: into the GPU, 3 splits of 2 GiB | `run -- known_bytes h2d 8 256 / 4 512 / 16 128` | ✅ 2.0147 / 2.0158 / 2.0141 GiB | ⬜ |
| 6 | copy path: out of the GPU | `run -- known_bytes d2h 8 256` | ✅ (in verify) | ✅ (in verify) |
| 7 | kernel path: GPU code writes | `run -- known_bytes memset 16 256` | ✅ kwrite 4.4999 (4 + 0.5 setup) | ✅ (in verify) |
| 8 | kernel path: GPU code reads | `run -- known_bytes d2d 16 256` | ✅ kread 4.0292, kwrite 4.4999, C2C 8.82 | ✅ (in verify) |
| 9 | observe a time window, quality report | `observe --seconds 3 --json F` | ✅ 0 gaps, clocks within 1 µs | ⬜ |
| 10 | live samples | `observe --live` | ⬜ | ⬜ |
| 11 | idle baseline, CSV, JSON | `run --idle 2 --csv F --json F -- CMD` | partly | ⬜ |
| 12 | Python phases | `examples/02_python_phases.py` | ⬜ on 1.0 (✅ before 1.0) | ⬜ |
| 13 | LLM bytes per token, from outside | `examples/03_llm_bytes_per_token.py` | ⬜ on 1.0 (✅ before 1.0: 7.52 GiB/token) | ⬜ |
| 14 | SMMU translations | `smmu status`, `smmu run -- CMD` | ✅ 7 GPU units; 1.30 M requests for 2 GiB | ⬜ |
| 15 | per kernel traffic | `pravaha kernels -- CMD` | ⬜ | ⬜ |
| 16 | uninstall leaves nothing | `pravaha uninstall` | ⬜ | ⬜ |
| 17 | unit tests (no GPU) | `python -m pytest tests` | ✅ 3 passed | ⬜ |
| 18 | GPU tests | `python -m pytest -m gpu tests` | ⬜ | ⬜ |

Details and raw outputs: [VALIDATION.md](VALIDATION.md) and the [validation/](../validation) folder.

## Report your results
1. `pravaha status` and `pravaha verify --json verify.json`
2. Open an issue with the title "Test report: <your machine>", paste the `status` output and attach `verify.json`.
3. Optional: any row above you tested, with its output.

Light work only: every command above moves a few GiB with copies or simple writes and reads. None runs heavy compute.
