"""A tiny GPU program that moves a known number of bytes. Nothing to compile:
it calls the CUDA runtime through ctypes. Used by `pravaha test` and the tutorial.

    python -m pravaha_mem.demo MODE COPIES MIB
    MODE: h2d (host to GPU copy), d2h (GPU to host copy), d2d (GPU to GPU copy,
          done by a kernel), memset (a kernel writing)

With PRAVAHA_DEMO_WINDOW=1 it prints when the work ran, as system clock ns:
    WINDOW t0 t1
PRAVAHA_DEMO_QUIET=S makes it wait S seconds between setup and work (default 0.5, so setup traffic stays apart from the work).
"""
import ctypes
import glob
import os
import sys
import time


def _cudart():
    cuda = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    libs = (glob.glob(cuda + "/targets/*/lib/libcudart.so*") + glob.glob(cuda + "/lib64/libcudart.so*")
            + glob.glob("/usr/local/cuda*/targets/*/lib/libcudart.so*")
            + glob.glob("/usr/local/cuda*/lib64/libcudart.so*"))
    if not libs:
        sys.exit("libcudart not found: set CUDA_HOME to your CUDA installation")
    return ctypes.CDLL(sorted(libs)[0])


def main(argv=None):
    a = sys.argv[1:] if argv is None else argv
    mode, n, mib = (a + ["h2d", "8", "256"][len(a):])[:3]
    n, mib = int(n), int(mib)
    rt = _cudart()
    size = ctypes.c_size_t(mib << 20)
    h, d, d2 = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()

    def ck(r):
        if r:
            sys.exit(f"CUDA error {r}")

    ck(rt.cudaMallocHost(ctypes.byref(h), size))
    ck(rt.cudaMalloc(ctypes.byref(d), size))
    ck(rt.cudaMalloc(ctypes.byref(d2), size))
    ck(rt.cudaMemset(d, 1, size)); ck(rt.cudaMemset(d2, 2, size)); ck(rt.cudaDeviceSynchronize())
    time.sleep(float(os.environ.get("PRAVAHA_DEMO_QUIET", "0.5")))
    t0 = time.time_ns()
    for _ in range(n):
        if mode == "h2d": ck(rt.cudaMemcpy(d, h, size, 1))
        elif mode == "d2h": ck(rt.cudaMemcpy(h, d, size, 2))
        elif mode == "d2d": ck(rt.cudaMemcpy(d2, d, size, 3))
        elif mode == "memset": ck(rt.cudaMemset(d, 3, size))
        else: sys.exit("MODE must be h2d, d2h, d2d or memset")
    ck(rt.cudaDeviceSynchronize())
    t1 = time.time_ns()
    if os.environ.get("PRAVAHA_DEMO_WINDOW"):   # set by pravaha test
        print("WINDOW", t0, t1, flush=True)
    print(f"{mode}: moved {n} x {mib} MiB = {n * mib} MiB", file=sys.stderr)
    time.sleep(0.05)


if __name__ == "__main__":
    main()
