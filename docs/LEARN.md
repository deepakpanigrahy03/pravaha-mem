# Learn pravaha in eight lessons

A short course, in the order the ideas build on each other. Each lesson ends with one thing to try on a GB10. All commands are light: they copy or write a few GiB, never heavy compute.

New to the project? Read [The whole idea](THE_WHOLE_IDEA.md) first; this course goes one level deeper.

## Lesson 1: the machine
The GB10 has two dies: an Arm CPU die and a Blackwell GPU die, joined by NVLink C2C (chip to chip). The memory (LPDDR5X, 128 GB) is attached to the CPU die. The GPU has no memory of its own.

So every byte the GPU fetches from memory, or writes to it, crosses C2C. The GPU's L2 cache sits in front of that path: a read that hits in L2 never leaves the GPU; a read that misses goes over C2C.

Try: `pravaha status` and read what it reports about the GPU and the counters.

## Lesson 2: the problem
NVIDIA's Grace servers expose counters for the bytes crossing C2C. On the GB10 none is exposed: DCGM profiling does not load, NVML has no traffic query, Nsight Compute's C2C section collects nothing, and the Linux C2C counter driver finds no units. Nsight Systems' GPU metrics on GB10 report activity percentages, not bytes.

So the question "how many bytes does my program move over C2C?" has no direct answer on this chip.

## Lesson 3: the idea
Every GPU request to memory passes the GPU's L2 cache, and the L2 counts what passes it, in 32 byte sectors, by which unit asked. Counting at L2 sees the same traffic, from the GPU side.

Five counters are enough:

| name in pravaha | counter | what it counts |
|---|---|---|
| copy | `lts__t_sectors_srcunit_ce` | copy engine traffic (`cudaMemcpy`) |
| kread | `lts__t_sectors_srcunit_tex_op_read_lookup_miss_realtime` | GPU code reads that missed L2 |
| kwrite | `lts__t_sectors_srcunit_tex_op_write_realtime` | GPU code writes |
| total (2 counters) | `lts__t_sectors_op_read_lookup_miss_realtime` + `lts__t_sectors_op_write_realtime` | all memory traffic, every unit |

The total already includes copy, kread and kwrite; never add them together.

## Lesson 4: how the counters were chosen
1. The argument: every GPU memory request passes L2, so L2 counters can see all of it.
2. The candidates: names listed by `ncu --query-metrics`.
3. The tests: programs that move a known number of bytes.
4. The rejections, each for a reason you can check: SMMU counters count address translations, not bytes (2,055 to 3,690 bytes per translation, depending on the buffer); L2 reads without `lookup_miss` include hits; the aperture (memory type) counters are refused by live sampling; Nsight Compute's DRAM counters need replay.
5. The confirmation: two machines and real frameworks.

Try: `pravaha verify`. It moves known bytes and compares.

## Lesson 5: the conversion
Kernel bytes are sectors x 32. A copy engine sits on the GPU die but both ends of a copy sit in LPDDR5X, so it reads each byte over C2C and writes it back over C2C: the copy counter sees each byte twice, so copied bytes are sectors x 32 / 2. That is why a 2 GiB copy shows about 2 GiB of copy and about 4 GiB of total traffic.

Try: `pravaha run -- python -m pravaha_mem.known_bytes h2d 8 256` and find the 2 GiB and the 4 GiB.

## Lesson 6: watching from outside
CUPTI PM sampling reads the counters at a fixed interval (10 ms) while programs run normally, without replaying anything. pravaha runs it in its own process (the reader), so the measured program needs no change, no library and no rebuild. One reader per GPU: a second one is refused while the first runs.

Measured cost on the measured program: none visible (about 0.04% and 0.53% on two machines, inside the run to run spread).

Try: start any GPU program, then `pravaha observe --seconds 10` in another terminal.

## Lesson 7: background and phases
The counters see the whole GPU, including the desktop redrawing the screen. pravaha measures the idle rate just before the work and subtracts it over the work's duration:

net = run - (idle bytes / idle seconds) x run seconds

This is an estimate (it assumes the background stays at its idle rate), so the raw numbers are always kept. A sample that straddles the edge of a phase is split in proportion to time.

Try: `pravaha run --idle 2 --csv out.csv -- CMD` and compare the run and run - idle rows.

## Lesson 8: what it shows
An unmodified vLLM server running Mistral 7B: each generated token reads 13.47 GiB, the weights plus the KV cache of the prompt, at about 226 GB/s, near the 273 GB/s the memory is specified for. Prefill reads the weights about three times in 0.4 s and writes the prompt's KV cache. Copies are zero once the model is loaded.

Try: [examples/03_llm_bytes_per_token.py](../examples/03_llm_bytes_per_token.py) on your own model.

## Where next
* Every command and option: [MANUAL.md](MANUAL.md)
* Using pravaha in your own Python: [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)
* The evidence behind every number: [VALIDATION.md](VALIDATION.md) and [TEST_MATRIX.md](TEST_MATRIX.md)
* Research uses and limits: [RESEARCHER_GUIDE.md](RESEARCHER_GUIDE.md)

## Lesson 9: your first schedule

A schedule is a written experiment: a list of GPU steps (slots) that `pravaha schedule` runs in order, printing when each one ran and how many bytes it moved. Run `pravaha observe` (or A-LEMS) at the same time, and every slot gets its observed traffic, and its energy, over the same clock.

**1. Get a starter file.**
```bash
pravaha schedule --template > my_test.json
```
It has an idle start, a streaming read, a gap, a copy and an idle end. Ready made examples: `examples/schedules/` (probe_rates, resident_vs_streaming, square_wave).

**2. Measure your machine's speeds.**
```bash
pravaha schedule examples/schedules/probe_rates.json
```
The table shows seconds and GB/s per slot. A GN100 gives about 1,161 GB/s for a read that fits the 24 MiB L2, 240 GB/s for a read from memory, and 59 GB/s for copies.

**3. Size your slots.** A slot must last long enough to be measured: at least 2 s for traffic (sampled every 10 ms), at least 60 s for energy (counters update about every 100 ms). For a read: `logical_mib = seconds x rate / 1,048,576`, rounded to a multiple of `footprint_mib`. Example: 60 s of streaming at 240 GB/s is about 13,730,000 MiB.

**4. Run it and read the table.**
```bash
pravaha schedule my_test.json --json my_result.json
```
One line per slot: kind, seconds, operations, payload GiB, GB/s, check. Payload is what the program asked to move; the observer shows what reached memory.

**5. Check your design.** Idle between measured slots (so windows never overlap), the same `logical_mib` when you compare two kinds of read, a `noop` sham when the work is paced, nothing else on the GPU.

**Exercise.** Write a schedule that compares an 8 MiB read with a 1,024 MiB read of the same total size, run it under `pravaha observe`, and explain why the observed bytes differ by about a thousand times while the payload is the same.
