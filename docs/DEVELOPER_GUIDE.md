# Developer guide: using pravaha in your programs

## 1. Install
```bash
pip install pravaha-mem          # in your project's or platform's environment (requirements.txt)
pravaha install                  # once per machine, by an admin (or your platform's install script)
```
The readers live in `/usr/local/libexec/pravaha` and are shared by all environments. The program you measure needs nothing.

At startup, check readiness and tell the user what to do, instead of failing in the middle of a run:
```python
import pravaha_mem as pv
s = pv.status()                  # s["ready"]: True or False; s["fix"]: plain instructions
if not s["ready"]:
    print("GPU traffic measurement disabled:", "; ".join(s["fix"]))
```

## 2. Measure a block of code
```python
import pravaha_mem as pv

with pv.observe() as m:              # starts the reader; returns once samples flow
    run_my_gpu_work()
print(m.summary())                   # table: copy, kread, kwrite, C2C, GB/s
```

## 3. Phases
A phase starts at `mark(name)` and lasts until the next mark or the end.
```python
with pv.observe() as m:
    m.mark("load");     model = load()
    m.mark("prefill");  out = model.prefill(prompt)
    m.mark("decode");   text = model.decode(out)

for name, p in m.phases().items():
    print(name, p["seconds"], p["copy"], p["kernel_read"], p["kernel_write"], p["memory"])
```
Phases shorter than about 10 samples have fuzzy edges (the summary says so); use `pv.observe(interval_ms=1)` for phases of a few milliseconds. Each phase also reports `boundary_share`: the part of its total that came from samples split at a phase edge.

## 4. Start and stop without `with`
For harnesses with their own hooks (A-LEMS uses this):
```python
m = pv.observe(interval_ms=10).start()
...                                  # m.mark(...) as needed
m.stop()
```

## 5. Your own description of the run
```python
m.metadata({"model": "qwen3-4b", "batch": 1, "expected_bytes": 2 * 2**30})
```
Stored with the result (`to_json`).

## 6. Results
| call | returns |
|---|---|
| `m.summary()` | text table |
| `m.total()` | dict for the whole window |
| `m.phases()` | `{phase: dict}` |
| `m.baseline("idle")` | rate of an idle phase and its variation, or `None` |
| `m.quality()` | samples, gaps, time not sampled, observed interval, clock offset |
| `m.samples` | list of `Sample(t0, t1, copy, kernel_read, kernel_write, all_read, all_write)` |
| `m.to_csv(path)`, `m.to_json(path)`, `m.to_dict()` | exports |

Dict keys (bytes): `copy`, `kernel_read`, `kernel_write`, `all_read`, `all_write`, `memory` (all_read + all_write), plus `seconds` and `boundary_share`. In JSON and CSV the same values are named `copy_bytes`, `kernel_read_miss_bytes`, `kernel_write_bytes`, `all_read_miss_bytes`, `all_write_bytes`, `memory_traffic_bytes`.

## 7. Time
`Sample.t0`, `t1`, `m.t_start`, mark times: system clock in ns, the same as `time.time_ns()`. Join with your own logs by time.

## 8. Run another program
```python
rc, m = pv.run(["python", "server.py"], idle_s=2)    # like `pravaha run --idle 2 -- ...`
rc, m = pv.run(None, seconds=60)                      # like `pravaha observe --seconds 60`
```

## 9. Errors to handle
| exception | when |
|---|---|
| `FileNotFoundError` | reader not installed: `pravaha install` |
| `RuntimeError("GPU counters busy ...")` | another profiler holds the counters; retry later |
| `RuntimeError("... insufficient privileges ...")` | reader lost its capability: `pravaha install` again |

## 10. Rules to remember
* **Device wide.** Values include every process on the GPU. Measure an idle phase for a baseline.
* **One reader per GPU.** Do not start two observers at the same time; reuse one, and mark phases.
* pravaha waits 1 s between stopping and starting a reader (the driver releases the counters).

## 11. Command line from scripts
`pravaha run --json result.json -- CMD` and `pravaha observe --seconds N --csv samples.csv` work unattended: no sudo, no prompts. Exit code: the command's exit code for `run`, 0 for `observe`.
