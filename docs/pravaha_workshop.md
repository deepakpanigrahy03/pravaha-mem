# pravaha workshop: Table I explained, then pravaha step by step

Author: Deepak Panigrahy. For the CAL letter and for teaching others.
Every command here was run on the GN100 and the DGX Spark (Nov 7 to 11, 2026).

---

## Part 1. Table I, row by row

The table answers one question: **is there any ready made way to read how many bytes the GB10 GPU moves to memory?** Each row is one place we looked. Red means "no", green means "this is what we use instead".

### Group 1: NVIDIA tools

**Row 1. DCGM profiling: fails to load (both systems).**
- What DCGM is: NVIDIA's Data Center GPU Manager, the standard monitor on data center GPUs. Its profiling part reports bytes moved over PCIe and NVLink.
- How we tested: `dcgmi profile -l` asks DCGM to list the profiling metrics it can collect.
- What we saw: "This request is serviced by a module of DCGM that is not currently loaded." Versions 3.3.9 and 4.7.1 on the GN100, 3.3.9 on the Spark.
- What it means: the profiling module does not run on the GB10. NVIDIA staff state DCGM is not officially supported on the DGX Spark.
- How to say it if asked: "We tried two DCGM versions; the profiling module never loads on this chip."

**Row 2. NVML traffic query: lists 8 C2C links, no traffic query (both systems).**
- What NVML is: the NVIDIA Management Library, the library behind `nvidia-smi`. Most GPU monitors read it.
- How we tested: a short Python script (package `nvidia-ml-py`):
  - `NVML_FI_DEV_C2C_LINK_COUNT` asks how many C2C links exist → **8**.
  - `nvmlDeviceGetPcieThroughput` is NVML's only traffic query → **Not Supported**.
- What it means: NVML knows the link is there (like a building directory listing 8 doors) but has no function that counts the data through it.
- Careful wording: NVML has no C2C traffic function at all; we did not call one that failed. Its only traffic function (for PCIe) is not supported.
- Related: `nvidia-smi c2c -s` prints "C2C Link 0: 43.125 GB/s". That is the link **speed**, not traffic.

**Row 3. Nsight Compute C2C section: no metrics (both systems).**
- What it is: Nsight Compute, NVIDIA's kernel profiler, has a section called `C2CLink` meant to report traffic on this link.
- How we tested: `ncu --list-sections` (the section exists), `ncu --query-metrics | grep -i c2c` (0 metrics), then `ncu --section C2CLink ./build/zc device 1 r` (profile a real kernel).
- What we saw: the kernel ran and was profiled, then "No metrics to show for section(s): C2CLink".
- What it means: the profiler was built for chips with readable C2C counters; on the GB10 it finds none.

### Group 2: Linux and firmware

**Row 4. C2C counter units in Linux: none (both systems).**
- Background: on NVIDIA Grace servers, Linux shows counter units such as `nvidia_nvlink_c2c0_pmu_0`. The driver is `nvidia_cspmu`.
- How we tested: `ls /sys/bus/event_source/devices/ | grep -iE "nvlink_c2c|cspmu"` → nothing.
- What it means: no C2C counter is available to Linux tools such as `perf`.

**Row 5. APMT firmware table: absent (both systems).**
- What APMT is: an ACPI table in which the firmware tells Linux where system counters sit in hardware. The `nvidia_cspmu` driver needs it.
- How we tested: `ls /sys/firmware/acpi/tables/ | grep -i APMT` → nothing. `pravaha show` also reports `apmt_table: false`.
- What it means: even if the chip contains such counters, Linux is never told where they are. This is why the paper says "not exposed in the tested software stack", never "the chip has no counter". Whether the silicon has one is a question for NVIDIA or MediaTek.

### Group 3: what we use instead

**Row 6. SMMU PMCG units in the GPU's SMMU: 7 (1 responds) (both systems).**
- What it is: the SMMU translates the addresses of device transfers. It contains counter groups (PMCGs). Linux exposes each as `smmuv3_pmcg_<address>`.
- How we found it: pravaha follows the GPU's sysfs `iommu` link to its SMMU (at 0x13000000) and lists the 7 PMCG units inside that SMMU's register window. Then it counts all 7 while copying known bytes.
- What we saw: only `smmuv3_pmcg_13002` moved (on the Spark: 1,048,987 transactions against 0 for the other six).
- What it means: that one unit counts the GPU's copy traffic. It is our copy path counter.

**Row 7. Nsight Compute L2 sysmem events: available (both systems).**
- What they are: `lts__d_sectors_fill_sysmem.sum` (32 B sectors read into the L2 from system memory) and `lts__t_sectors_aperture_sysmem_op_write.sum` (sectors written).
- What it means: these count kernel traffic that leaves the GPU. They are our kernel path counter.

**One sentence summary:** six places that should report traffic report nothing on either system, so we found two counters that do respond and proved what they measure.

---

## Part 2. pravaha workshop

pravaha is our C and Python library plus a command `pravaha` with five verbs: `show`, `setup`, `measure`, `watch`, `test`.

### Step 0. Get ready (once per machine)

```
cd pravaha-mem
make build                              # builds the test programs into build/
export PYTHONPATH=$PWD/python
sudo sysctl kernel.perf_event_paranoid=0   # lets you read system counters without sudo
alias pravaha='python3 -m pravaha.cli'
```

Why `perf_event_paranoid=0`: the SMMU counter counts the whole system, and Linux allows that only to privileged users unless this setting is lowered. It resets on reboot.

### Step 1. `pravaha show`: what can pravaha see?

```
pravaha show
```

It never fails; it reports:
- `gpus`: the GPU's PCI address (`000f:01:00.0`).
- `candidates`: the 7 PMCG units inside the GPU's SMMU.
- `unit`: the chosen counter (`smmuv3_pmcg_13002`) once confirmed.
- `calibration`: `usable`, `stale` (machine changed since calibration) or `absent`.
- `perf_permitted`, `ncu_permitted`: whether you have the rights.
- `apmt_table: false`: Table I row 5, checked by the library.
- `identity`: vendor, product, kernel, driver, CUDA. A calibration is tied to these.

**Exercise:** run it on both machines and compare `identity`. Kernel versions differ (1021 vs 1026), so each machine needs its own calibration.

### Step 2. Make sure the machine is quiet

```
pravaha watch --seconds 3
```

Each line is one 100 ms sample: time, raw count, increase. On a quiet machine the increase is a few hundred per line. If it is thousands, something else is moving data; wait or stop it. A busy background (142,760 per second on Nov 7) broke every copy check.

### Step 3. `pravaha setup`: choose the counter and calibrate

```
pravaha setup        # asks you to confirm the counter
pravaha setup -y     # same, no question
```

What it does:
1. Counts all candidates during known copies and picks the one that responds. It never picks silently when there are several candidates.
2. For pinned, pageable and async copies, sizes 1 to 256 MiB, both directions, 5 repeats each, each repeat on a fresh buffer:
   - counts 1 s of idle background,
   - copies 2,048 MiB in copies of that size,
   - computes bytes per transaction = bytes / (count − background).
3. Saves `~/.config/pravaha/calibration.json` (and `.tsv` for C) plus a copy in `history/`.
4. Flags a size **unstable** when its 5 repeats spread more than 5%, and keeps both levels.

Takes about 10 minutes. **Exercise:** open `calibration.json` and find the 256 MiB entries; compare with the 1 MiB entries (about 2,040 vs 3,642 bytes per transaction).

### Step 4. `pravaha measure`: measure a program's copy traffic

```
pravaha measure -- python3 my_training.py
pravaha measure --profile 256:h2d:8,64:h2d:async:4 -- ./my_app
pravaha measure --kernels --json out.json -- ./my_app
```

- Without `--profile`: you get transactions and a **bound** (lowest to highest bytes per transaction in the calibration), because pravaha does not know your copy sizes.
- With `--profile size:direction[:async]:count,...`: you tell it what copies the program makes, and it converts with the matching factors.
- `--kernels`: runs your program **a second time** under Nsight Compute and adds kernel read and write bytes (sectors × 32). Separate run, cold caches.
- Every report states the scope: the copy count is for the whole system, not your process.

**Exercise:** measure `./build/mixed` with `--profile 256:h2d:8,256:d2h:8 --kernels` and compare with the bytes it prints at DONE.

### Step 5. `pravaha test`: run the paper's checks

```
pravaha test                      # quick checks
pravaha test --full --json t.json # all 11 checks, about 30 minutes
pravaha test --only cpu_only      # one check (here N4)
```

With sudo (needed for Nsight Compute), keeping your settings:
```
sudo env "PATH=$PATH" PYTHONPATH=$PWD/python PRAVAHA_CONFIG_DIR=$HOME/.config/pravaha python3 -m pravaha.cli test --full --json t.json
```

| Check | Plan ID | Question it answers |
|---|---|---|
| calibration | E5 | Is there a calibration, and does it match this machine? |
| linearity | E4 | Does the count grow in a straight line with bytes? |
| copy_blind_to_kernels | E6 | Does the copy counter ignore kernel traffic? |
| kernels | E7/N2 | Do L2 events match known reads and writes in 5 allocations? |
| mixed | N1 | Do both paths together account for a known workload? |
| repeatability | N3 | Do calibrations agree across reboots? |
| cpu_only | N4 | Does CPU traffic move the copy counter? |
| directions | N5 | Host to device, device to host, device to device |
| async | N6 | Do async copies match their calibration? |
| pattern | N7 | Strided and random reads: sectors vs distinct sectors |
| warm_cold | N8 | Does keeping the cache warm change the count? |

Results: PASS, FAIL (never hidden), INFO (a measurement, no pass rule) or SKIP (with the reason). Pass rules were fixed before the runs.

### Step 6. Use it from Python

```python
import pravaha_mem as pravaha

with pravaha.measure() as m:          # starts and stops the copy counter
    run_my_copies()
print(m.net_txn)                       # background corrected transactions (signed)
print(m.summary(profile=[(256, "h2d", 8)]))

k = pravaha.kernels(["./my_app"])      # separate Nsight Compute run
print(k.read_bytes, k.write_bytes)
```

### Step 7. Use it from C

```c
#include "pravaha.h"
pravaha_copy_open(...); pravaha_copy_start(...);
/* copies */
pravaha_copy_stop(...);
double bpt = pravaha_bytes_per_txn(&cal, 256, PRAVAHA_H2D, 0);  /* NAN if not calibrated */
```
See `examples/copy_counter.c`. The C and Python versions are tested to give the same numbers.

---

## Part 3. The rules to remember (and to defend in review)

1. **No bytes without a calibration.** There is no built in table; each machine calibrates itself.
2. **The copy counter counts the whole system.** Keep the machine quiet; pravaha subtracts the background it measures just before each count.
3. **Bytes per transaction are not fixed.** Small copies: about 3,642 to 3,849. Copies of 64 MiB and above: lower, and dependent on the buffer (2,028 to 2,222 at 256 MiB). Exact bytes need a calibration on matching buffers; otherwise report the range.
4. **Kernel counts come from a separate profiled run** with caches flushed before each kernel.
5. **Pageable writes cost about 15% extra sectors.** Compare workloads only with the same allocation type.
6. **Neither counter reads the C2C link or the LPDDR5X itself.** A system level cache on the CPU die may serve part of the traffic.
