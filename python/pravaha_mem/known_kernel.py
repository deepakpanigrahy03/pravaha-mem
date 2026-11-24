"""known_kernel: a GPU kernel that reads or writes an exact, known amount of data in a
chosen kind of memory.

    python -m pravaha_mem.known_kernel PLACEMENT OP TIMES SIZE_MIB [--prefetch device|none]
                                       [--init gpu|cpu] [--hold SECONDS] [--query]
    pravaha run -- python -m pravaha_mem.known_kernel managed read 200 16

PLACEMENT  how the buffer is allocated (the allocation API, not a claim where pages sit):
  device    cudaMalloc
  pinned    cudaMallocHost (page locked computer memory)
  managed   cudaMallocManaged (CUDA may move pages between computer and GPU)
  pageable  ordinary aligned_alloc memory, read by the GPU directly
            (only where the GPU supports pageable memory access, for example ATS on GB10)
OP        read: each pass reads the whole buffer and sums it (one 8 byte partial sum
               per thread block is written: aux_write_bytes);
          write: each pass writes the whole buffer with a constant.
TIMES     passes over the buffer;  SIZE_MIB  buffer size in MiB.
Logical bytes = TIMES x SIZE_MIB x 1,048,576 for the operation.

Order: allocate; initialise (--init: gpu kernel or cpu writes, same for every placement
that allows it); apply --prefetch (managed only); synchronise; t0; TIMES passes; synchronise;
t1. --hold S replaces the passes with a held wait of S seconds (TIMES must be 0): the
prepared buffer and the CUDA context stay alive, no kernel runs.

Machine lines on stdout (always):
  ATTRS {json}         device attributes and the managed range settings, before and after
  FIRST t0 t1          end of the first pass (wall clock ns, the clock of time.time_ns())
  WINDOW t0 t1         all passes (or the hold)
  CHECK ok|fail {json} checksum after t1 (outside the window), logical and aux bytes
Exit codes: 0 done, 1 CUDA error (name printed), 2 unsupported placement, 3 check failed.

The kernel is compiled at run time with NVRTC (part of the CUDA toolkit) and launched through
the CUDA driver; nothing needs building. Standard library only.
Part of pravaha-mem. MIT License. Copyright (c) 2026 MANTRA Labs.
"""
import argparse
import ctypes
import glob
import json
import os
import sys
import time

PLACEMENTS = ("device", "pinned", "managed", "pageable")
MIB = 1 << 20
THREADS = 256
BLOCKS_PER_SM = 4
READ_VALUE = 0x01010101      # every 32 bit word after initialisation
WRITE_VALUE = 0x02020202     # every 32 bit word after a write pass

# cudaDeviceAttr values (CUDA runtime API)
ATTR = {"multiprocessor_count": 16, "compute_major": 75, "compute_minor": 76, "managed_memory": 83,
        "pageable_memory_access": 88, "concurrent_managed_access": 89,
        "pageable_memory_access_uses_host_page_tables": 100, "direct_managed_mem_access_from_host": 101}
# cudaMemRangeAttribute: 1 read mostly, 2 preferred location, 4 last prefetch location;
# CUDA 13 adds the location type and id variants (5 to 8). Requests and policy, not residency.
RANGE_ATTR = {"read_mostly": 1, "preferred_location": 2, "last_prefetch_location": 4,
              "preferred_location_type": 5, "preferred_location_id": 6,
              "last_prefetch_location_type": 7, "last_prefetch_location_id": 8}

SOURCE = r"""
extern "C" __global__ void kread(const uint4* p, unsigned long long n, unsigned long long* out) {
    unsigned long long s = 0;
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        uint4 v = p[i];
        s += (unsigned long long)v.x + v.y + v.z + v.w;
    }
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
    __shared__ unsigned long long w[32];
    if ((threadIdx.x & 31) == 0) w[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long t = 0;
        for (int k = 0; k < (blockDim.x + 31) / 32; ++k) t += w[k];
        out[blockIdx.x] += t;
    }
}
extern "C" __global__ void kwrite(uint4* p, unsigned long long n, unsigned int v) {
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x)
        p[i] = make_uint4(v, v, v, v);
}
"""


class CudaError(RuntimeError):
    pass


def _lib(names, env="CUDA_HOME"):
    cuda = os.environ.get(env, "/usr/local/cuda")
    for n in names:
        found = (glob.glob(cuda + "/targets/*/lib/" + n) + glob.glob(cuda + "/lib64/" + n)
                 + glob.glob("/usr/local/cuda*/targets/*/lib/" + n) + glob.glob("/usr/local/cuda*/lib64/" + n))
        if found:
            return ctypes.CDLL(sorted(found)[0])
    raise CudaError("%s not found: set CUDA_HOME to your CUDA installation" % names[0])


class _Cuda:
    """Runtime (allocation, attributes), driver (launch) and NVRTC (compile), through ctypes."""

    def __init__(self, device=0):
        self.rt = _lib(["libcudart.so*"])
        self.nvrtc = _lib(["libnvrtc.so*"])
        self.drv = ctypes.CDLL("libcuda.so.1")
        self.rt.cudaGetErrorName.restype = ctypes.c_char_p
        self.device = device
        self.ck(self.rt.cudaSetDevice(device))
        self.ck(self.rt.cudaFree(None))                        # creates the primary context
        ver = ctypes.c_int()
        self.ck(self.rt.cudaRuntimeGetVersion(ctypes.byref(ver)))
        self.runtime_version = ver.value

    def ck(self, r, what="CUDA"):
        if r:
            name = self.rt.cudaGetErrorName(r)
            raise CudaError("%s error %s (%s)" % (what, r, name.decode() if name else "?"))

    def dck(self, r):
        if r:
            raise CudaError("CUDA driver error %s" % r)

    def attr(self, name):
        v = ctypes.c_int()
        r = self.rt.cudaDeviceGetAttribute(ctypes.byref(v), ATTR[name], self.device)
        return v.value if r == 0 else None

    def compile(self):
        """PTX for this GPU's compute capability; returns (kread, kwrite) driver functions."""
        prog = ctypes.c_void_p()
        self.ck(self.nvrtc.nvrtcCreateProgram(ctypes.byref(prog), SOURCE.encode(), b"known_kernel.cu",
                                              0, None, None), "NVRTC")
        arch = "--gpu-architecture=compute_%d%d" % (self.attr("compute_major"), self.attr("compute_minor"))
        opts = (ctypes.c_char_p * 1)(arch.encode())
        r = self.nvrtc.nvrtcCompileProgram(prog, 1, opts)
        if r:
            n = ctypes.c_size_t()
            self.nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
            log = ctypes.create_string_buffer(n.value)
            self.nvrtc.nvrtcGetProgramLog(prog, log)
            raise CudaError("NVRTC compile failed: %s" % log.value.decode(errors="replace")[-300:])
        n = ctypes.c_size_t()
        self.nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(n))
        ptx = ctypes.create_string_buffer(n.value)
        self.nvrtc.nvrtcGetPTX(prog, ptx)
        self.nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
        self.dck(self.drv.cuInit(0))
        mod = ctypes.c_void_p()
        self.dck(self.drv.cuModuleLoadData(ctypes.byref(mod), ptx))
        fns = []
        for name in (b"kread", b"kwrite"):
            f = ctypes.c_void_p()
            self.dck(self.drv.cuModuleGetFunction(ctypes.byref(f), mod, name))
            fns.append(f)
        self.arch = arch.split("=")[1]
        return fns

    def launch(self, fn, blocks, args):
        params = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in args])
        self.dck(self.drv.cuLaunchKernel(fn, blocks, 1, 1, THREADS, 1, 1, 0, None, params, None))

    def sync(self):
        self.ck(self.rt.cudaDeviceSynchronize())


class _MemLocation(ctypes.Structure):                         # CUDA 13 cudaMemLocation
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


def _attributes(cu, ptr=None, size=0, managed=False):
    """Device capabilities, plus the managed range's requested settings (not residency)."""
    out = {k: cu.attr(k) for k in ATTR if k not in ("multiprocessor_count", "compute_major", "compute_minor")}
    if managed and ptr is not None:
        for name, code in RANGE_ATTR.items():
            v = ctypes.c_int(-99)
            r = cu.rt.cudaMemRangeGetAttribute(ctypes.byref(v), ctypes.c_size_t(4), code, ptr, ctypes.c_size_t(size))
            out[name] = v.value if r == 0 else "unavailable(%d)" % r
    return out


def support(attrs):
    """{placement: {"ok": bool, "reason": str|None}} from the device attributes."""
    s = {"device": {"ok": True, "reason": None}, "pinned": {"ok": True, "reason": None}}
    s["managed"] = ({"ok": True, "reason": None} if attrs.get("managed_memory")
                    else {"ok": False, "reason": "managedMemory attribute is 0"})
    s["pageable"] = ({"ok": True, "reason": None} if attrs.get("pageable_memory_access")
                     else {"ok": False, "reason": "pageableMemoryAccess attribute is 0"})
    return s


def query(device=0):
    """Pinned shape: {"attributes": {...}, "supported": {placement: {"ok", "reason"}}}."""
    cu = _Cuda(device)
    a = _attributes(cu)
    return {"attributes": a, "supported": support(a)}


def expected_checksum(op, times, n_vec):
    """Read checksum after TIMES passes (mod 2^64); for write, one verification read pass."""
    per_pass = n_vec * 4 * (READ_VALUE if op == "read" else WRITE_VALUE)
    return (per_pass * (times if op == "read" else 1)) % (1 << 64)


def _alloc(cu, placement, size):
    p = ctypes.c_void_p()
    if placement == "device":
        cu.ck(cu.rt.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(size)))
    elif placement == "pinned":
        cu.ck(cu.rt.cudaMallocHost(ctypes.byref(p), ctypes.c_size_t(size)))
    elif placement == "managed":
        cu.ck(cu.rt.cudaMallocManaged(ctypes.byref(p), ctypes.c_size_t(size), 1))   # cudaMemAttachGlobal
    else:
        libc = ctypes.CDLL(None)
        libc.aligned_alloc.restype = ctypes.c_void_p
        p = ctypes.c_void_p(libc.aligned_alloc(ctypes.c_size_t(4096), ctypes.c_size_t(size)))
        if not p.value:
            raise CudaError("aligned_alloc failed")
    return p


def _initialise(cu, placement, init, p, size, kwrite, blocks, n_vec):
    if init == "gpu":
        cu.launch(kwrite, blocks, [p, ctypes.c_ulonglong(n_vec), ctypes.c_uint(READ_VALUE)])
    elif placement == "device":                                 # cpu writes a host buffer, then copies
        host = ctypes.create_string_buffer(b"\x01" * size, size)
        cu.ck(cu.rt.cudaMemcpy(p, host, ctypes.c_size_t(size), 1))
    else:
        ctypes.memset(p, 0x01, size)
    cu.sync()


def _prefetch(cu, p, size, target):
    if target != "device":
        return
    if cu.runtime_version >= 13000:
        loc = _MemLocation(1, cu.device)                        # cudaMemLocationTypeDevice
        cu.ck(cu.rt.cudaMemPrefetchAsync(p, ctypes.c_size_t(size), loc, ctypes.c_uint(0), None))
    else:
        cu.ck(cu.rt.cudaMemPrefetchAsync(p, ctypes.c_size_t(size), ctypes.c_int(cu.device), None))


def run_workload(placement, op, times, size_mib, prefetch=None, init="gpu", hold_s=None, device=0, emit=print):
    """Do the work and emit the machine lines; returns the CHECK dict."""
    cu = _Cuda(device)
    attrs = _attributes(cu)
    sup = support(attrs)[placement]
    if not sup["ok"]:
        raise SystemExit(_unsupported(placement, sup["reason"]))
    kread, kwrite = cu.compile()
    size, blocks = size_mib * MIB, cu.attr("multiprocessor_count") * BLOCKS_PER_SM
    n_vec = size // 16
    p = _alloc(cu, placement, size)
    out = ctypes.c_void_p()
    cu.ck(cu.rt.cudaMalloc(ctypes.byref(out), ctypes.c_size_t(blocks * 8)))
    cu.ck(cu.rt.cudaMemset(out, 0, ctypes.c_size_t(blocks * 8)))
    _initialise(cu, placement, init, p, size, kwrite, blocks, n_vec)
    if placement == "managed":
        _prefetch(cu, p, size, prefetch or "none")
    cu.sync()
    emit("ATTRS " + json.dumps({"when": "before", **_attributes(cu, p, size, placement == "managed")}))
    args_r = [p, ctypes.c_ulonglong(n_vec), out]
    args_w = [p, ctypes.c_ulonglong(n_vec), ctypes.c_uint(WRITE_VALUE)]
    t0 = time.time_ns()
    if hold_s is not None:
        time.sleep(hold_s)
    for i in range(times):
        cu.launch(kread if op == "read" else kwrite, blocks, args_r if op == "read" else args_w)
        if i == 0:
            cu.sync()
            emit("FIRST %d %d" % (t0, time.time_ns()))
    cu.sync()
    t1 = time.time_ns()
    emit("WINDOW %d %d" % (t0, t1))
    emit("ATTRS " + json.dumps({"when": "after", **_attributes(cu, p, size, placement == "managed")}))
    return _check(cu, op, times, n_vec, blocks, kread, p, out, hold_s is not None, size, emit)


def _check(cu, op, times, n_vec, blocks, kread, p, out, held, size, emit):
    """Checksum outside the window: read sums for read; one verification read pass for write."""
    if op == "write" and not held:
        cu.ck(cu.rt.cudaMemset(out, 0, ctypes.c_size_t(blocks * 8)))
        cu.launch(kread, blocks, [p, ctypes.c_ulonglong(n_vec), out])
        cu.sync()
    host = (ctypes.c_ulonglong * blocks)()
    cu.ck(cu.rt.cudaMemcpy(host, out, ctypes.c_size_t(blocks * 8), 2))
    got = sum(host) % (1 << 64)
    passes = 0 if held else times
    want = 0 if held or (op == "read" and times == 0) else expected_checksum(op, times, n_vec)
    res = {"ok": got == want, "checksum": got, "expected": want, "logical_bytes": passes * size,
           "aux_write_bytes": passes * blocks * 8 if op == "read" else 0, "blocks": blocks,
           "threads": THREADS, "arch": cu.arch, "runtime_version": cu.runtime_version}
    emit("CHECK %s %s" % ("ok" if res["ok"] else "fail", json.dumps(res)))
    return res


def _unsupported(placement, reason):
    print("known_kernel: placement %s not supported here: %s" % (placement, reason), file=sys.stderr)
    return 2


def run(placement, op, times, size_mib, prefetch=None, init="gpu", hold_s=None, timeout=None, env=None):
    """Run the program as a child process; parsed lines plus returncode and stderr tail."""
    import subprocess
    cmd = [sys.executable, "-m", "pravaha_mem.known_kernel", placement, op, str(times), str(size_mib),
           "--init", init]
    if prefetch:
        cmd += ["--prefetch", prefetch]
    if hold_s is not None:
        cmd += ["--hold", str(hold_s)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    return dict(parse(p.stdout.splitlines()), returncode=p.returncode, stderr_tail=(p.stderr or "")[-300:])


def parse(lines):
    """Machine lines to a dict: window, first, attrs_before, attrs_after, check."""
    r = {"t0": None, "t1": None, "first_t1": None, "attrs_before": None, "attrs_after": None, "check": None}
    for line in lines:
        head, _, rest = line.partition(" ")
        if head == "WINDOW":
            r["t0"], r["t1"] = (int(x) for x in rest.split())
        elif head == "FIRST":
            r["first_t1"] = int(rest.split()[1])
        elif head == "ATTRS":
            d = json.loads(rest)
            r["attrs_" + d.pop("when")] = d
        elif head == "CHECK":
            r["check"] = json.loads(rest.split(" ", 1)[1])
    return r


def _args(argv):
    p = argparse.ArgumentParser(prog="known_kernel", formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("placement", nargs="?", choices=PLACEMENTS)
    p.add_argument("op", nargs="?", choices=("read", "write"))
    p.add_argument("times", nargs="?", type=int)
    p.add_argument("size_mib", nargs="?", type=int)
    p.add_argument("--prefetch", choices=("device", "none"), default=None, help="managed only")
    p.add_argument("--init", choices=("gpu", "cpu"), default="gpu")
    p.add_argument("--hold", type=float, default=None, metavar="SECONDS", help="held control, TIMES must be 0")
    p.add_argument("--query", action="store_true", help="print attributes and supported placements, no work")
    p.add_argument("--device", type=int, default=int(os.environ.get("PRAVAHA_DEVICE", "0")))
    a = p.parse_args(argv)
    if not a.query and None in (a.placement, a.op, a.times, a.size_mib):
        p.error("PLACEMENT OP TIMES SIZE_MIB are required (or --query)")
    if a.hold is not None and a.times:
        p.error("--hold needs TIMES 0")
    if a.hold is None and a.times is not None and a.times < 1:
        p.error("TIMES must be at least 1 (use --hold for a control)")
    if a.prefetch and a.placement != "managed":
        p.error("--prefetch applies to managed only")
    if a.size_mib is not None and a.size_mib <= 0:
        p.error("SIZE_MIB must be positive")
    return a


def main(argv=None):
    a = _args(sys.argv[1:] if argv is None else argv)
    try:
        if a.query:
            print(json.dumps(query(a.device)))
            return 0
        res = run_workload(a.placement, a.op, a.times, a.size_mib, a.prefetch, a.init, a.hold, a.device)
    except CudaError as e:
        print("known_kernel: %s" % e, file=sys.stderr)
        return 1
    print("known_kernel: %s %s, %d x %d MiB in %s memory (init %s, prefetch %s): logical %d bytes, check %s"
          % (a.op, "held" if a.hold is not None else "done", a.times, a.size_mib, a.placement, a.init,
             a.prefetch or "n/a", res["logical_bytes"], "ok" if res["ok"] else "FAILED"), file=sys.stderr)
    sys.stdout.flush()
    return 0 if res["ok"] else 3


if __name__ == "__main__":
    sys.exit(main())
