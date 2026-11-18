"""known_bytes: a small GPU program that moves an exact, known amount of data.

Use it to check pravaha (the right answer is known in advance) or to learn what each
column means. It calls NVIDIA's standard CUDA functions directly (through ctypes, so
nothing needs compiling) and prints what it did.

    python -m pravaha_mem.known_bytes DIRECTION TIMES SIZE_MIB
    pravaha run -- python -m pravaha_mem.known_bytes h2d 8 256

DIRECTION  where the data goes, and who moves it:
  h2d     host to device: copy from the computer's memory into the GPU
          (copy engines; pravaha column: copy)              CUDA call: cudaMemcpy
  d2h     device to host: copy from the GPU back to the computer's memory
          (copy engines; pravaha column: copy)              CUDA call: cudaMemcpy
  d2d     device to device: copy from one GPU buffer to another
          (GPU code reads and writes; columns: kread, kwrite) CUDA call: cudaMemcpy
  memset  fill a GPU buffer with a value
          (GPU code writes; column: kwrite)                 CUDA call: cudaMemset
  "Host" is NVIDIA's word for the computer side, "device" for the GPU.
TIMES     how many times to repeat
SIZE_MIB  size of each repeat, in MiB (1 MiB = 1,048,576 bytes)
Data moved = TIMES x SIZE_MIB.

Before the work it sets up its buffers: it reserves one buffer in the computer's
memory and two on the GPU, and clears the two GPU buffers (GPU code writing
2 x SIZE_MIB; pravaha shows this in kwrite). Then it waits 0.5 s, so setup and work
are apart, and does the work. Every CUDA program also moves a small fixed amount
when it starts (about 15 MiB of copies).
"""
import argparse
import ctypes
import glob
import os
import sys
import time

WHAT = {
    "h2d": ("copied", "from the computer's memory into the GPU", "copy engines", "copy"),
    "d2h": ("copied", "from the GPU back to the computer's memory", "copy engines", "copy"),
    "d2d": ("copied", "from one GPU buffer to another", "GPU code reading and writing", "kread and kwrite"),
    "memset": ("wrote", "into a GPU buffer", "GPU code writing", "kwrite"),
}


def _cudart():
    cuda = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    libs = (glob.glob(cuda + "/targets/*/lib/libcudart.so*") + glob.glob(cuda + "/lib64/libcudart.so*")
            + glob.glob("/usr/local/cuda*/targets/*/lib/libcudart.so*")
            + glob.glob("/usr/local/cuda*/lib64/libcudart.so*"))
    if not libs:
        sys.exit("known_bytes: libcudart not found: set CUDA_HOME to your CUDA installation")
    return ctypes.CDLL(sorted(libs)[0])


def _args(argv):
    p = argparse.ArgumentParser(prog="known_bytes", formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__,
                                epilog="Examples (and what pravaha should show):\n"
                                       "  h2d 8 256       copy 2 GiB into the GPU         copy   about 2 GiB\n"
                                       "  d2h 8 256       copy 2 GiB out of the GPU       copy   about 2 GiB\n"
                                       "  memset 16 256   GPU code writes 4 GiB           kwrite about 4 GiB\n"
                                       "  d2d 16 256      GPU code reads 4 GiB            kread  about 4 GiB")
    p.add_argument("direction", choices=list(WHAT), help="h2d: into the GPU; d2h: out of the GPU; "
                   "d2d: GPU buffer to GPU buffer (GPU code); memset: GPU code fills a buffer")
    p.add_argument("times", type=int, help="how many times to repeat")
    p.add_argument("size_mib", type=int, help="size of each repeat in MiB")
    return p.parse_args(argv)


def main(argv=None):
    a = _args(sys.argv[1:] if argv is None else argv)
    rt = _cudart()
    size = ctypes.c_size_t(a.size_mib << 20)
    host, gpu_a, gpu_b = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()

    def ck(r):
        if r:
            sys.exit(f"known_bytes: CUDA error {r}")

    # setup: one buffer in the computer's memory (pinned), two on the GPU, both GPU buffers cleared
    ck(rt.cudaMallocHost(ctypes.byref(host), size))
    ck(rt.cudaMalloc(ctypes.byref(gpu_a), size))
    ck(rt.cudaMalloc(ctypes.byref(gpu_b), size))
    ck(rt.cudaMemset(gpu_a, 1, size)); ck(rt.cudaMemset(gpu_b, 2, size)); ck(rt.cudaDeviceSynchronize())
    time.sleep(float(os.environ.get("PRAVAHA_KNOWN_PAUSE", "0.5")))   # keep setup apart from the work

    # the work: TIMES repeats of SIZE_MIB
    t0 = time.time_ns()
    for _ in range(a.times):
        if a.direction == "h2d":
            ck(rt.cudaMemcpy(gpu_a, host, size, 1))       # 1 = cudaMemcpyHostToDevice
        elif a.direction == "d2h":
            ck(rt.cudaMemcpy(host, gpu_a, size, 2))       # 2 = cudaMemcpyDeviceToHost
        elif a.direction == "d2d":
            ck(rt.cudaMemcpy(gpu_b, gpu_a, size, 3))      # 3 = cudaMemcpyDeviceToDevice
        else:
            ck(rt.cudaMemset(gpu_a, 3, size))
    ck(rt.cudaDeviceSynchronize())                        # wait until the GPU has really finished
    t1 = time.time_ns()

    verb, where, who, col = WHAT[a.direction]
    total = a.times * a.size_mib
    print(f"known_bytes: {verb} {total} MiB ({total / 1024:g} GiB) {where}, "
          f"{a.times} x {a.size_mib} MiB, by {who} (pravaha column: {col}); "
          f"setup also cleared 2 x {a.size_mib} MiB on the GPU (kwrite)", file=sys.stderr)
    if os.environ.get("PRAVAHA_KNOWN_WINDOW"):           # used by pravaha verify: when the work ran
        print("WINDOW", t0, t1, flush=True)
    time.sleep(0.05)
    return 0


if __name__ == "__main__":
    sys.exit(main())
