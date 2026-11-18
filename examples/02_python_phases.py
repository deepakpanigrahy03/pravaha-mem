"""Lesson 2: phases inside your own Python code.

A small PyTorch job in three phases: load (host to GPU copy), compute (a GPU
kernel reading the data), save (GPU to host copy). pravaha shows each phase's
traffic, and we compare with what the code should move.

Run: python examples/02_python_phases.py   (needs PyTorch with CUDA)
"""
import torch
import pravaha_mem as pv

GIB = 2 ** 30
N = 2 ** 28                                   # 2**28 float32 = 1 GiB
host = torch.ones(N).pin_memory()

with pv.observe(interval_ms=1) as m:     # 1 ms samples: these phases last only milliseconds
    m.mark("load")                            # 1 GiB host to GPU
    dev = host.cuda(); torch.cuda.synchronize()

    m.mark("compute")                         # 10 full reads of 1 GiB by a kernel
    for _ in range(10):
        s = dev.sum()
    torch.cuda.synchronize()

    m.mark("save")                            # 0.25 GiB GPU to host
    back = dev[: N // 4].cpu()

print(m.summary())

p = m.phases()
print("\nexpected vs measured")
print(f"  load    copy        1.00 GiB   measured {p['load']['copy'] / GIB:.3f}")
print(f"  compute kernel read 10.00 GiB  measured {p['compute']['kernel_read'] / GIB:.3f}")
print(f"  save    copy        0.25 GiB   measured {p['save']['copy'] / GIB:.3f}")
print("\nsave is higher than 0.25 GiB: .cpu() into new (pageable) memory goes through a staging")
print("step inside CUDA and PyTorch, and pravaha counts every byte the copy engines really move.")
print("Copy into pinned memory instead (out.copy_(dev)) and save shows close to 0.25 GiB.")
