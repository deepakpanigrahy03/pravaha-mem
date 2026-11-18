# pravaha tutorial

Five short lessons. Each one has a command to run and the output to expect, measured on an Acer GN100 (GB10). Your numbers will match closely; the background row differs from machine to machine.

## Lesson 0: install and check (5 minutes)

```bash
pip install pravaha-mem      # or from GitHub: git clone https://github.com/deepakpanigrahy03/pravaha-mem && cd pravaha-mem && pip install .
pravaha install-sampler      # once per machine; asks for your password once
pravaha permissions
pravaha test
```

`pravaha test` starts small programs that move a known number of bytes and checks pravaha against them:

```
  PASS  idle machine: no copies                           0.0 MiB  expected      0.0
  PASS  copy host to GPU                               2048.0 MiB  expected   2048.0  ratio 1.0000
  PASS  copy GPU to host                               2048.0 MiB  expected   2048.0  ratio 1.0000
  PASS  kernel write (cudaMemset)                      2048.0 MiB  expected   2048.0  ratio 1.0000
  PASS  kernel read and write (GPU to GPU copy)        2047.9 MiB  expected   2048.0  ratio 1.0000
all checks passed
```

If all pass, every number pravaha gives you on this machine rests on the same counters. On an Acer GN100 and a DGX Spark every check gives a ratio of 1.0000 (the GN100's desktop can move the GPU to GPU check by a few tenths of a percent).

## Lesson 1: measure any command

```bash
pravaha c2c --idle 2 -- python my_app.py
bash examples/01_any_command.sh            # pravaha's own demo: copies 2 GiB host to GPU
```

What the rows mean:

* **idle**: the machine before your command starts. The desktop and other programs use the GPU a little; this is your baseline.
* **run**: while your command ran.
* **total**: both together.

What the columns mean:

* **copy GiB**: bytes moved by copy engines (`cudaMemcpy`, `.cuda()`, `.cpu()`, weight loading).
* **kread GiB / kwrite GiB**: bytes GPU code read from and wrote to memory.
* **C2C GiB**: everything the GPU fetched or sent over C2C, from every source.
* **GB/s**: C2C bytes per second in that phase.

Your program's own CUDA startup moves a small fixed amount (about 15 MiB of copies for a plain CUDA program), so the demo, which copies 2 GiB, shows a little over 2 GiB. You can run the demo yourself: `python -m pravaha_mem.demo h2d 8 256` (modes: h2d, d2h, d2d, memset). Lesson 2 shows how to measure only the part you care about.

## Lesson 2: phases in your own code

```bash
python examples/02_python_phases.py
```

```python
with pv.c2c() as m:
    m.mark("load");    dev = host.cuda()
    m.mark("compute"); s = dev.sum()   # x 10
    m.mark("save");    back = dev[:N//4].cpu()
```

The example samples every 1 ms (`pv.c2c(interval_ms=1)`), because these phases last only a few milliseconds. Measured on the GN100:

| phase | expected | measured |
|---|---|---|
| load (copy) | 1.00 GiB | 0.998 |
| compute (kernel reads: 10 passes over a 1 GiB tensor, far larger than the L2 cache) | 10.0 GiB | 9.94 |
| save (copy) | 0.25 GiB | about 0.43 |

Save is higher on purpose: `.cpu()` into new, pageable memory goes through a staging step, and pravaha counts every byte the copy engines really move. Copying into pinned memory (`out.copy_(dev)`) shows close to 0.25 GiB. This is the kind of thing pravaha is for: seeing what your framework actually does.

How fine a phase can be: a sample lasts about one interval (10 ms by default). A phase shorter than about 10 samples has fuzzy edges, and the summary says so. Use `interval_ms=1` (command line `--interval 1`) for short phases; keep 10 ms for long runs, where it costs less.

## Lesson 3: an LLM server, from outside

Start a server in one terminal (for example `vllm serve Qwen/Qwen3-4B-Instruct-2507`), then:

```bash
python examples/03_llm_bytes_per_token.py --tokens 400
```

Measured on the GN100 with Qwen3 4B (weights 7.49 GiB on disk):

```
generated tokens     400
kernel reads / token 7.522 GiB
tokens per second    23.2
read bandwidth       187.6 GB/s
```

Decoding reads every weight once per token, so GiB per token is about the weight size, and the tokens per second follow from the memory bandwidth. The server is never changed or restarted. To watch a server without sending requests yourself:

```bash
pravaha c2c --seconds 60 --csv server.csv     # observe whatever runs for 60 s
```

## Lesson 4: join with power, energy or logs

Samples are stamped with the system clock in ns, the same as `time.time_ns()` and `date +%s%N`. Any other series stamped with that clock joins directly:

```bash
python examples/04_join_with_other_data.py /tmp/pravaha_lesson1.csv
```

Replace the made up readings with your power meter or request log to get, for example, joules per GiB moved.

## Lesson 5: when something goes wrong

| you see | why | fix |
|---|---|---|
| `GPU counters busy` | another profiler holds the counters | stop Nsight, DCGM profiling or another pravaha, run again |
| `pravaha-c2c not found` | readers not installed | `pravaha install-sampler` |
| `missing cap_sys_admin` | the reader was rebuilt or copied | `pravaha install-sampler` again |
| a check fails in `pravaha test` | a heavy job ran at the same time | run again on a quiet machine |
