# pravaha tutorial

Each lesson: one idea, one command, the output to expect (measured on an Acer GN100).

## Lesson 0: install and check
```bash
pip install pravaha-mem      # or: git clone https://github.com/deepakpanigrahy03/pravaha-mem && cd pravaha-mem && pip install .
pravaha install              # once per machine; asks for your password once
pravaha status
pravaha verify
```
`status` should end with "ready". `verify` runs light workloads that move known bytes and compares; each line should say PASS with a ratio near 1.0000.

## Lesson 1: measure any program
```bash
pravaha run --idle 2 -- python -m pravaha_mem.known_bytes h2d 8 256
bash examples/01_any_command.sh            # the same, plus CSV and JSON
```
Rows: **idle** (the machine before your program: your baseline), **run** (while it ran), **total**. Columns: **copy** (copy engines), **kread / kwrite** (GPU code), **C2C** (all GPU memory traffic), **GB/s**. The command, word by word: `pravaha run --idle 2 --` means "measure 2 s of the quiet GPU, then measure while the following runs". `python -m pravaha_mem.known_bytes` is pravaha's test program; `h2d` means host to device (from the computer's memory into the GPU), `8` times, `256` MiB each: 2 GiB, moved by the copy engines. So the **copy** column should show about 2 GiB.

You see a little more: 0.0146 GiB of copies is CUDA's own startup, which every CUDA program does, and 0.25 GiB in **kwrite** is the program clearing its GPU buffer before copying. Try the other directions and watch which column moves:

| command | moves | column |
|---|---|---|
| `known_bytes h2d 8 256` | 2 GiB into the GPU | copy |
| `known_bytes d2h 8 256` | 2 GiB out of the GPU | copy |
| `known_bytes memset 16 256` | GPU code writes 4 GiB | kwrite |
| `known_bytes d2d 16 256` | GPU code reads 4 GiB (and writes it elsewhere) | kread (and kwrite) |

`python -m pravaha_mem.known_bytes --help` explains every argument.

## Lesson 2: phases in your own code
```bash
python examples/02_python_phases.py        # needs PyTorch
```
| phase | expected | measured |
|---|---|---|
| load (copy) | 1.00 GiB | 0.99 |
| compute (kernel reads: 10 passes over 1 GiB) | 10.0 GiB | 9.94 |
| save (copy) | 0.25 GiB | about 0.42 |

Save is higher on purpose: `.cpu()` into new, pageable memory goes through a staging step, and pravaha counts every byte the copy engines really move. Copying into pinned memory shows close to 0.25 GiB. The example samples every 1 ms because these phases last only milliseconds.

## Lesson 3: an LLM server, from outside
Start a server (for example `vllm serve Qwen/Qwen3-4B-Instruct-2507`), then:
```bash
python examples/03_llm_bytes_per_token.py --tokens 400
```
Measured: 7.52 GiB of kernel reads per generated token for a model whose weights are 7.49 GiB: decoding reads every weight once per token. To watch a server without sending requests yourself: `pravaha observe --seconds 60 --csv server.csv`.

## Lesson 4: join with power, energy or logs
Samples carry system clock times in ns, the same as `time.time_ns()`. Any series stamped with that clock joins directly:
```bash
python examples/04_join_with_other_data.py /tmp/pravaha_lesson1.csv
```

## Lesson 5: address translations (SMMU)
```bash
pravaha smmu status
pravaha smmu run -- python -m pravaha_mem.known_bytes h2d 8 256
```
```
pravaha smmu status (address translations, not bytes)
  GPU SMMU counters : smmuv3_pmcg_13002, smmuv3_pmcg_13042, ... (7 in total)
  default unit      : smmuv3_pmcg_13002
  readable via      : perf (translation requests, translation misses)
event                  count    per second   meaning
transaction          1295537       1075976   translation requests
tlb_miss              671242        557484   translation misses
```
These count translation requests and misses, not bytes: useful to study page sizes (2 MiB pages need about 10 times fewer translations than 4 KiB pages) and translation caching. The default unit is the GPU's translation control unit (TCU); the other six are its translation buffer units.

## Lesson 6: when something goes wrong
| you see | fix |
|---|---|
| GPU counters busy | stop Nsight, DCGM profiling or another pravaha, and run again |
| pravaha-reader not found | `pravaha install` |
| insufficient privileges | `pravaha install` again |
| a check fails in `verify` | another program was busy on the GPU; run again when it is quieter |
