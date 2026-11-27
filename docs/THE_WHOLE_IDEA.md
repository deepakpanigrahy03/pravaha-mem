# pravaha: the whole idea, in one read

## 1. The machine
The NVIDIA GB10 is two chips in one package:
* a **CPU chip** (Arm cores), with the memory attached to it: 128 GB of LPDDR5X;
* a **GPU chip** (Blackwell), with **no memory of its own**.

The two chips are joined by a short, fast link: **NVLink C2C** ("chip to chip").

So whenever the GPU needs data, the data comes **from the memory on the CPU side, across C2C**. Whenever the GPU writes a result, it goes **back across C2C** into that memory. Even memory a program calls "GPU memory" (`cudaMalloc`) is a part of the same LPDDR5X, behind the CPU chip.

> Picture a kitchen (the GPU) with no fridge. All food (data) is in a store across the road (memory). The road is C2C. Every ingredient fetched and every dish sent back crosses the road.

## 2. The problem
How much a program carries across that road decides its **speed** (the road has a limit), its **energy** (carrying costs power) and how much it **slows the CPU** (they share the road and the memory).

But NVIDIA gives **no counter for C2C** on the GB10. Nobody could see how many bytes a program moves across it, or when.

## 3. The solution
Every request the GPU makes to memory passes one place first: the GPU's **L2 cache**, a small store of recent data at the GPU's door (in the picture: a **pantry** at the kitchen door). The L2 keeps its own counts, in units of 32 bytes (**sectors**), and it records **who** made each request.

pravaha reads those L2 counts **every 10 ms, live, from its own small program**, without touching the program being measured. From them it reports:

| column | means | in the picture |
|---|---|---|
| **copy** | bytes carried by the **copy engines** (blocks moved, no calculation) | delivery vans |
| **kread** | bytes **GPU code** had to fetch from memory (L2 misses) | cooks fetching ingredients not in the pantry |
| **kwrite** | bytes **GPU code** wrote to memory | cooks sending dishes back |
| **C2C** | everything the GPU fetched or sent, by anyone | every trip across the road |

Reads that find their data already in L2 (**hits**) do not cross the road, so they are rightly **not** counted.

## 4. Two paths cover all GPU traffic
Data only reaches or leaves the GPU in two ways:

1. **The copy path.** A copy engine moves a block: into the GPU (h2d, host to device), out of it (d2h, device to host), or between two GPU buffers. No calculation. Examples: `cudaMemcpy`, PyTorch `.cuda()` and `.cpu()`, **loading a model's weights**.
2. **The kernel path.** GPU code (a **kernel**) works on data: it reads what it needs and writes results. Examples: adding numbers, **matrix multiplication**, attention, `cudaMemset`, and the GPU to GPU copy that the driver does with a kernel. In an LLM: **generating every token**. pravaha does not count the calculations; it counts the **bytes the calculations had to fetch and write**.

Both paths pass the L2, so pravaha sees both, in the same samples, at the same time. That is why one tool covers copies, computations, any direction, any buffer (pinned, pageable, huge pages, GPU buffers) and any framework (CUDA, PyTorch, vLLM, anything), without knowing anything about the program.

## 5. Why a copy shows more on C2C than its size
Both buffers of a copy live in the same LPDDR5X. Copying 2 GiB into the GPU = **read 2 GiB** across the road, then **write 2 GiB** back across it: **about 4 GiB on C2C for 2 GiB copied**. A GPU to GPU copy of 4 GiB reads 4 and writes 4: about 8 GiB. GPU code that only writes 4 GiB: about 4 GiB. On a normal PC with a graphics card none of this would cross the CPU link; on the GB10 all of it does. pravaha makes that visible.

## 6. How we know it is right
A test program, **known_bytes**, moves an exact amount of data that we choose. pravaha measures it from outside. If they agree, pravaha can be trusted on real programs where nobody knows the answer.

| known_bytes | moves | pravaha measured |
|---|---|---|
| `h2d 8 256` (also 4 x 512 and 16 x 128) | 2 GiB copied into the GPU | copy 2.01 GiB every time (bytes, not copies) |
| `d2h 8 256` | 2 GiB copied out | copy 2.00 |
| `memset 16 256` | GPU code writes 4 GiB | kwrite 4.00 (+0.5 setup) |
| `d2d 16 256` | GPU code reads 4 GiB, writes 4 GiB | kread 4.03, C2C 8.8 |

`pravaha verify` repeats these automatically: ratio **1.0000** on two different GB10 machines. And on real software, from outside: PyTorch copies exact to 0.003 %; a vLLM server running Qwen3 4B read **7.52 GiB per generated token**, against **7.49 GiB** of weights, because generating a token reads every weight once.

## 7. Reading a result
* **total**: the whole measurement.
* **idle** (with `--idle`): the GPU before your program, showing the background (the desktop redraws the screen about 10 times a second).
* **run**: while your program ran; it includes the background.
* **run - idle**: your program's own traffic, estimated as run minus (idle rate x run time).

pravaha counts the **whole GPU** (every program on it). That is why the idle baseline exists.

## 8. What pravaha does not do
* It does not count the CPU's own memory traffic: only the GPU's.
* It cannot tell programs apart: values are for the whole GPU; use idle and phases.
* It cannot split by memory type (the GB10's live counters do not offer that).
* Only one counter reader can run on the GPU at a time (the hardware has one sampling unit).
* It counts bytes, not calculations.
* It counts the payload the GPU requested at its L2, not the raw bytes on the wire: a link counter would also include protocol overhead (packet headers, coherence and flow control messages) and any traffic that does not pass the GPU's L2 (for example address translation requests). On NVIDIA's Grace CPUs such a link counter is exposed; the GB10's CPU die is made by MediaTek, and none is exposed there.
* The CPU's own memory traffic never crosses C2C (the CPU and the memory are on the same die), so it is not counted; it does compete for the same memory. CPU counters (for example through A-LEMS) cover that side.
* `pravaha kernels` (Nsight Compute) sees only kernels a program launches itself; driver operations such as `cudaMemcpy` and `cudaMemset` appear only in `run` and `observe`.

## 9. The commands, in the order you use them
```
pravaha install              once per machine: one password, one folder, one permission per reader
pravaha status               is the machine ready? (readers, permission, counters free, temperature)
pravaha verify               check the scale with known weights (light, about 3 minutes)
pravaha run -- CMD           measure while CMD runs
pravaha observe --seconds N  measure a time window (a server already running)
```
In Python: `with pravaha_mem.observe() as m: m.mark("step") ...`.
Also: `pravaha smmu` (address translations, a separate counter, never bytes), `pravaha kernels` (traffic per named kernel), `pravaha verify --kernels` (known kernels in each memory placement), `pravaha schedule` (timed GPU steps with known bytes, for energy experiments; `verify --schedule` checks it).

## 10. Where pravaha lives on your machine
| part | installed by | where | how often |
|---|---|---|---|
| Python package (`pravaha` command, `import pravaha_mem`) | `pip install pravaha-mem` | your venv, or a platform's venv (A-LEMS: its `requirements.txt`) | each environment that measures |
| two counter readers, one permission each | `pravaha install` (admin password once) | `/usr/local/libexec/pravaha` | once per machine, shared by every environment |

The program you measure needs nothing installed. pravaha reads the counters only while a measurement runs (`run`, `observe`, `verify`, `pv.observe()`); there is no background service.

## 11. Words
* **headroom**: how many degrees the GPU is below its thermal limit. `nvidia-smi` reports it (`temperature.gpu.tlimit`); `pravaha status` shows it. At about -5 the hardware switches the machine off; `verify` only starts with 10 or more and stops below 5 (`[thermal]` of `pravaha.ini`).
* **host / device**: NVIDIA's words for the computer side and the GPU.
* **sector**: 32 bytes, the unit the L2 counts in.
* **hit / miss**: data found in L2 / data that had to cross C2C.
* **device wide**: every program on the GPU together.

## In one sentence
**The GB10's GPU has no memory and no C2C counter; every byte it uses passes its L2 cache; pravaha reads the L2's counts live, from outside, for copies and for computation alike, and checks itself against known bytes.**
