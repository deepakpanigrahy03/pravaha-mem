"""schedule: run a timestamped schedule of GPU workload slots, for energy experiments.

    pravaha schedule SCHEDULE.json [--json OUT] [--raw] [--device N]
    pravaha schedule --template            (a commented starter file)

One process holds one CUDA context and every buffer for the whole schedule, so slots
differ only in the work they do. Setup (allocation, initialisation, kernel compile)
happens before the first slot and is reported apart. Slot kinds:

  idle      sleep `seconds`
  noop      paced empty kernel launches (`pace_ms`) for `seconds`: the submission loop
            of copy and kernel slots without moving bytes (the sham control)
  kernel    `op` read|write over a `footprint_mib` buffer until `logical_mib` have been
            read or written (passes = logical / footprint), so a cache resident and a
            streaming footprint can issue the same operations
  copy      `direction` h2d|d2h, host memory `host` pinned|pageable, `size_mib` per copy:
            `count` copies back to back, or one copy per `pace_ms` for `seconds` (paced)
  compute   `launches` of an arithmetic kernel doing `iters` fused multiply adds per
            thread on registers (no memory traffic beyond one result per thread)
  overlap   a compute part and a copy or kernel part issued on two streams at once
  square    lock-in modulation: `periods` of `period_s`; the on half runs `on`
            (a copy, kernel or noop spec, paced), the off half runs `off` (idle or noop)

Machine lines on stdout (wall clock ns, the clock of time.time_ns()):
  SETUP {json}   buffers, compile time, device attributes
  SLOT {json}    one per slot: index, name, kind, t0, t1, ops, payload_bytes, check, detail
  HALF {json}    one per square wave half: slot, period, half (on|off), t0, t1, ops, payload_bytes
  DONE {json}    totals
Payload bytes are program counted (what the code asked to move); observed bytes come
from the L2 sampler, never from here. Exit 0 done, 1 CUDA error, 2 bad schedule, 3 check failed.
Part of pravaha-mem. MIT License.
"""
import argparse
import ctypes
import json
import sys
import time

MIB = 1 << 20
KINDS = ("idle", "noop", "kernel", "copy", "compute", "overlap", "square")

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
extern "C" __global__ void knop() {}
extern "C" __global__ void kfma(float* out, unsigned int iters) {
    float a = threadIdx.x * 1e-3f, b = 1.0001f, c = 0.9999f;
    for (unsigned int i = 0; i < iters; ++i) { a = fmaf(a, b, c); c = fmaf(c, b, a); }
    out[blockIdx.x * blockDim.x + threadIdx.x] = a + c;
}
"""
FUNCS = ("kread", "kwrite", "knop", "kfma")
READ_VALUE = 0x01010101


class ScheduleError(ValueError):
    """The schedule is not valid; nothing has run."""


# ---------------------------------------------------------------- validation
def _need(spec, keys, where):
    miss = [k for k in keys if k not in spec]
    if miss:
        raise ScheduleError("%s: missing %s" % (where, ", ".join(miss)))


def validate_part(spec, where):
    """One workload part (used alone, inside overlap, or as a square wave half)."""
    kind = spec.get("kind")
    if kind == "idle":
        return
    if kind == "noop":
        _need(spec, ["pace_ms"], where)
    elif kind == "kernel":
        _need(spec, ["op", "footprint_mib"], where)
        if spec["op"] not in ("read", "write"):
            raise ScheduleError("%s: op must be read or write" % where)
        if spec.get("placement", "device") not in ("device", "pinned", "managed", "pageable"):
            raise ScheduleError("%s: unknown placement" % where)
    elif kind == "copy":
        _need(spec, ["direction", "host", "size_mib"], where)
        if spec["direction"] not in ("h2d", "d2h") or spec["host"] not in ("pinned", "pageable"):
            raise ScheduleError("%s: direction h2d|d2h, host pinned|pageable" % where)
    elif kind == "compute":
        _need(spec, ["iters"], where)
    else:
        raise ScheduleError("%s: unknown part kind %r" % (where, kind))


def validate(schedule):
    """Check every slot before any GPU work; returns the slot list."""
    slots = schedule.get("slots")
    if not slots:
        raise ScheduleError("schedule has no slots")
    for i, s in enumerate(slots):
        where = "slot %d (%s)" % (i, s.get("name", "?"))
        kind = s.get("kind")
        if kind not in KINDS:
            raise ScheduleError("%s: unknown kind %r" % (where, kind))
        if kind in ("idle", "noop"):
            _need(s, ["seconds"], where)
            validate_part(s, where)
        elif kind == "kernel":
            _need(s, ["logical_mib"], where)
            validate_part(s, where)
            if s["logical_mib"] % s["footprint_mib"]:
                raise ScheduleError("%s: logical_mib must be a multiple of footprint_mib" % where)
        elif kind == "copy":
            _need(s, ["seconds"] if s.get("pace_ms") else ["count"], where)
            validate_part(s, where)
        elif kind == "compute":
            _need(s, ["launches"], where)
            validate_part(s, where)
        elif kind == "overlap":
            _need(s, ["compute", "transfer"], where)
            validate_part(dict(s["compute"], kind="compute"), where + " compute")
            validate_part(s["transfer"], where + " transfer")
            if s["transfer"]["kind"] not in ("copy", "kernel"):
                raise ScheduleError("%s: transfer must be copy or kernel" % where)
        elif kind == "square":
            _need(s, ["period_s", "periods", "on", "off"], where)
            validate_part(s["on"], where + " on")
            validate_part(s["off"], where + " off")
            if s["off"]["kind"] not in ("idle", "noop"):
                raise ScheduleError("%s: off half must be idle or noop" % where)
            if s["on"]["kind"] in ("copy", "kernel") and "pace_ms" not in s["on"]:
                raise ScheduleError("%s: on half needs pace_ms (one operation per pace)" % where)
    return slots


# ---------------------------------------------------------------- CUDA backend
class CudaBackend:
    """Real GPU work through ctypes, NVRTC and the driver (pattern of known_kernel)."""

    def __init__(self, device=0):
        from pravaha_mem import known_kernel as kk
        self.kk = kk
        self.cu = kk._Cuda(device)
        self.blocks = self.cu.attr("multiprocessor_count") * kk.BLOCKS_PER_SM
        self.threads = kk.THREADS
        self.fn = self._compile()
        self.streams = [self._stream(), self._stream()]
        self.bufs = {}

    def _compile(self):
        cu = self.cu
        prog = ctypes.c_void_p()
        cu.ck(cu.nvrtc.nvrtcCreateProgram(ctypes.byref(prog), SOURCE.encode(), b"schedule.cu", 0, None, None), "NVRTC")
        arch = "--gpu-architecture=compute_%d%d" % (cu.attr("compute_major"), cu.attr("compute_minor"))
        opts = (ctypes.c_char_p * 1)(arch.encode())
        if cu.nvrtc.nvrtcCompileProgram(prog, 1, opts):
            n = ctypes.c_size_t()
            cu.nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
            log = ctypes.create_string_buffer(n.value)
            cu.nvrtc.nvrtcGetProgramLog(prog, log)
            raise self.kk.CudaError("NVRTC compile failed: %s" % log.value.decode(errors="replace")[-300:])
        n = ctypes.c_size_t()
        cu.nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(n))
        ptx = ctypes.create_string_buffer(n.value)
        cu.nvrtc.nvrtcGetPTX(prog, ptx)
        cu.nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
        cu.arch = arch.split("=")[1]              # read by attributes(); known_kernel sets it in its own compile
        cu.dck(cu.drv.cuInit(0))
        mod = ctypes.c_void_p()
        cu.dck(cu.drv.cuModuleLoadData(ctypes.byref(mod), ptx))
        out = {}
        for name in FUNCS:
            f = ctypes.c_void_p()
            cu.dck(cu.drv.cuModuleGetFunction(ctypes.byref(f), mod, name.encode()))
            out[name] = f
        return out

    def _stream(self):
        s = ctypes.c_void_p()
        self.cu.ck(self.cu.rt.cudaStreamCreate(ctypes.byref(s)))
        return s

    def _launch(self, name, args, stream, blocks=None):
        params = (ctypes.c_void_p * max(1, len(args)))(*[ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in args])
        self.cu.dck(self.cu.drv.cuLaunchKernel(self.fn[name], blocks or self.blocks, 1, 1, self.threads, 1, 1, 0,
                                               stream, params if args else None, None))

    def kernel_buffer(self, placement, footprint_mib):
        key = ("k", placement, footprint_mib)
        if key not in self.bufs:
            size = footprint_mib * MIB
            p = self.kk._alloc(self.cu, placement, size)
            n_vec = size // 16
            self._launch("kwrite", [p, ctypes.c_ulonglong(n_vec), ctypes.c_uint(READ_VALUE)], None)
            out = ctypes.c_void_p()
            self.cu.ck(self.cu.rt.cudaMalloc(ctypes.byref(out), ctypes.c_size_t(self.blocks * 8)))
            self.cu.ck(self.cu.rt.cudaMemset(out, 0, ctypes.c_size_t(self.blocks * 8)))
            self.cu.sync()
            self.bufs[key] = (p, n_vec, out)
        return self.bufs[key]

    def copy_buffers(self, host, size_mib):
        key = ("c", host, size_mib)
        if key not in self.bufs:
            size = size_mib * MIB
            h = self.kk._alloc(self.cu, "pinned" if host == "pinned" else "pageable", size)
            ctypes.memset(h, 0x01, size)
            d = ctypes.c_void_p()
            self.cu.ck(self.cu.rt.cudaMalloc(ctypes.byref(d), ctypes.c_size_t(size)))
            self.cu.ck(self.cu.rt.cudaMemset(d, 1, ctypes.c_size_t(size)))
            self.cu.sync()
            self.bufs[key] = (h, d, size)
        return self.bufs[key]

    def compute_buffer(self):
        if "fma" not in self.bufs:
            out = ctypes.c_void_p()
            self.cu.ck(self.cu.rt.cudaMalloc(ctypes.byref(out), ctypes.c_size_t(self.blocks * self.threads * 4)))
            self.bufs["fma"] = out
        return self.bufs["fma"]

    def prepare(self, part):
        if part["kind"] == "kernel":
            self.kernel_buffer(part.get("placement", "device"), part["footprint_mib"])
        elif part["kind"] == "copy":
            self.copy_buffers(part["host"], part["size_mib"])
        elif part["kind"] == "compute":
            self.compute_buffer()

    def kernel_pass(self, part, stream=0):
        p, n_vec, out = self.kernel_buffer(part.get("placement", "device"), part["footprint_mib"])
        if part["op"] == "read":
            self._launch("kread", [p, ctypes.c_ulonglong(n_vec), out], self.streams[stream])
        else:
            self._launch("kwrite", [p, ctypes.c_ulonglong(n_vec), ctypes.c_uint(READ_VALUE)], self.streams[stream])
        return part["footprint_mib"] * MIB

    def copy_once(self, part, stream=0):
        h, d, size = self.copy_buffers(part["host"], part["size_mib"])
        dst, src, kind = (d, h, 1) if part["direction"] == "h2d" else (h, d, 2)
        self.cu.ck(self.cu.rt.cudaMemcpyAsync(dst, src, ctypes.c_size_t(size), kind, self.streams[stream]))
        return size

    def compute_once(self, part, stream=0):
        self._launch("kfma", [self.compute_buffer(), ctypes.c_uint(int(part["iters"]))], self.streams[stream])
        return 0

    def noop(self, stream=0):
        self._launch("knop", [], self.streams[stream], blocks=1)
        return 0

    def sync(self):
        self.cu.sync()

    def read_checksum(self, part):
        """Sum of the read kernel's per block partials, then reset (outside any window)."""
        p, n_vec, out = self.kernel_buffer(part.get("placement", "device"), part["footprint_mib"])
        host = (ctypes.c_ulonglong * self.blocks)()
        self.cu.ck(self.cu.rt.cudaMemcpy(host, out, ctypes.c_size_t(self.blocks * 8), 2))
        self.cu.ck(self.cu.rt.cudaMemset(out, 0, ctypes.c_size_t(self.blocks * 8)))
        self.cu.sync()
        return sum(host) % (1 << 64), n_vec

    def attributes(self):
        return {"blocks": self.blocks, "threads": self.threads, "arch": self.cu.arch,
                "runtime_version": self.cu.runtime_version}


# ---------------------------------------------------------------- the runner
def _expected_read(n_vec, passes):
    return (n_vec * 4 * READ_VALUE * passes) % (1 << 64)


def _paced(be, part, seconds, clock, sleep):
    """One operation per pace_ms until `seconds` have passed; returns (ops, payload bytes)."""
    pace = part["pace_ms"] / 1000.0
    end = clock() + seconds
    ops = payload = 0
    nxt = clock()
    while clock() < end:
        if part["kind"] == "noop":
            be.noop()
        elif part["kind"] == "copy":
            payload += be.copy_once(part)
        else:
            payload += be.kernel_pass(part)
        be.sync()
        ops += 1
        nxt += pace
        rest = min(nxt, end) - clock()
        if rest > 0:
            sleep(rest)
    return ops, payload


def run_slot(be, i, s, emit, clock, sleep, now_ns):
    """Run one slot; returns the SLOT record."""
    kind, check, detail = s["kind"], None, {}
    t0 = now_ns()
    ops = payload = 0
    if kind == "idle":
        sleep(s["seconds"])
    elif kind == "noop":
        ops, payload = _paced(be, s, s["seconds"], clock, sleep)
    elif kind == "kernel":
        passes = s["logical_mib"] // s["footprint_mib"]
        for _ in range(passes):
            payload += be.kernel_pass(s)
        be.sync()
        ops = passes
    elif kind == "copy":
        if s.get("pace_ms"):
            ops, payload = _paced(be, s, s["seconds"], clock, sleep)
        else:
            for _ in range(int(s["count"])):
                payload += be.copy_once(s)
            be.sync()
            ops = int(s["count"])
    elif kind == "compute":
        for _ in range(int(s["launches"])):
            be.compute_once(s)
        be.sync()
        ops = int(s["launches"])
        detail["fma_per_launch"] = int(s["iters"]) * 2 * be.attributes()["blocks"] * be.attributes()["threads"]
    elif kind == "overlap":
        c, tr = dict(s["compute"], kind="compute"), s["transfer"]
        n_c, n_t = int(c["launches"]), int(tr.get("count") or tr["logical_mib"] // tr["footprint_mib"])
        for k in range(max(n_c, n_t)):
            if k < n_c:
                be.compute_once(c, stream=0)
            if k < n_t:
                payload += be.copy_once(tr, stream=1) if tr["kind"] == "copy" else be.kernel_pass(tr, stream=1)
        be.sync()
        ops = n_c + n_t
        detail.update(compute_launches=n_c, transfer_ops=n_t)
    elif kind == "square":
        ops, payload = _square(be, i, s, emit, clock, sleep, now_ns)
    t1 = now_ns()
    if kind == "kernel" and s["op"] == "read":
        got, n_vec = be.read_checksum(s)
        want = _expected_read(n_vec, ops)
        check = {"ok": got == want, "checksum": got, "expected": want}
    rec = {"index": i, "name": s.get("name", "slot_%d" % i), "kind": kind, "t0": t0, "t1": t1,
           "seconds": (t1 - t0) / 1e9, "ops": ops, "payload_bytes": payload, "check": check, "detail": detail}
    emit("SLOT " + json.dumps(rec))
    return rec


def _square(be, i, s, emit, clock, sleep, now_ns):
    """Square wave halves; each half is timed and reported on its own HALF line."""
    half = s["period_s"] / 2.0
    ops = payload = 0
    for p in range(int(s["periods"])):
        for name, part in (("on", s["on"]), ("off", s["off"])):
            h0 = now_ns()
            if part["kind"] == "idle":
                o, b = 0, 0
                sleep(half)
            else:
                o, b = _paced(be, part, half, clock, sleep)
            h1 = now_ns()
            ops, payload = ops + o, payload + b
            emit("HALF " + json.dumps({"slot": i, "period": p, "half": name, "t0": h0, "t1": h1,
                                       "ops": o, "payload_bytes": b}))
    return ops, payload


def run_schedule(schedule, backend, emit=print, clock=time.monotonic, sleep=time.sleep, now_ns=time.time_ns):
    """Validate, prepare every buffer, then run the slots in order; returns the SLOT records."""
    slots = validate(schedule)
    s0 = now_ns()
    for s in slots:
        for part in _parts(s):
            backend.prepare(part)
    backend.sync()
    emit("SETUP " + json.dumps({"t0": s0, "t1": now_ns(), "attributes": backend.attributes()}))
    pause = float(schedule.get("pause_after_setup_s", 0))
    if pause:
        sleep(pause)
    recs = [run_slot(backend, i, s, emit, clock, sleep, now_ns) for i, s in enumerate(slots)]
    ok = all(r["check"] is None or r["check"]["ok"] for r in recs)
    emit("DONE " + json.dumps({"slots": len(recs), "checks_ok": ok,
                               "payload_bytes": sum(r["payload_bytes"] for r in recs)}))
    return recs


def _parts(s):
    if s["kind"] == "overlap":
        return [dict(s["compute"], kind="compute"), s["transfer"]]
    if s["kind"] == "square":
        return [s["on"], s["off"]]
    return [s]


def parse(lines):
    """Machine lines to {setup, slots, halves, done}."""
    out = {"setup": None, "slots": [], "halves": [], "done": None}
    for line in lines:
        head, _, rest = line.partition(" ")
        if head == "SETUP":
            out["setup"] = json.loads(rest)
        elif head == "SLOT":
            out["slots"].append(json.loads(rest))
        elif head == "HALF":
            out["halves"].append(json.loads(rest))
        elif head == "DONE":
            out["done"] = json.loads(rest)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="schedule", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("schedule", help="schedule JSON file")
    ap.add_argument("--device", type=int, default=0)
    a = ap.parse_args(argv)
    with open(a.schedule) as fh:
        sched = json.load(fh)
    try:
        validate(sched)
    except ScheduleError as e:
        print("schedule: %s" % e, file=sys.stderr)
        return 2
    try:
        recs = run_schedule(sched, CudaBackend(a.device), emit=lambda s: print(s, flush=True))
    except Exception as e:  # CUDA errors from the backend
        print("schedule: %s" % e, file=sys.stderr)
        return 1
    return 0 if all(r["check"] is None or r["check"]["ok"] for r in recs) else 3


if __name__ == "__main__":
    sys.exit(main())


def table(parsed):
    """Readable summary of a parsed run: one line per slot, then the totals."""
    lines = ["%-28s %-8s %9s %8s %11s %9s  %s" % ("slot", "kind", "seconds", "ops", "payload GiB", "GB/s", "check")]
    for s in parsed["slots"]:
        sec = s["seconds"]
        rate = s["payload_bytes"] / sec / 1e9 if sec and s["payload_bytes"] else 0.0
        chk = "" if s.get("check") is None else ("ok" if s["check"]["ok"] else "FAILED")
        lines.append("%-28s %-8s %9.2f %8d %11.3f %9.2f  %s" % (s["name"][:28], s["kind"], sec, s["ops"],
                                                                s["payload_bytes"] / 2 ** 30, rate, chk))
    halves = parsed["halves"]
    if halves:
        on = [h for h in halves if h["half"] == "on"]
        lines.append("square wave: %d halves (%d on), on payload %.3f GiB"
                     % (len(halves), len(on), sum(h["payload_bytes"] for h in on) / 2 ** 30))
    d = parsed["done"] or {}
    lines.append("total payload %.3f GiB, checks %s" % ((d.get("payload_bytes") or 0) / 2 ** 30,
                                                        "ok" if d.get("checks_ok") else "FAILED"))
    lines.append("payload = bytes the program asked to move; bytes that reached memory: pravaha observe")
    return "\n".join(lines)


def cli(path, device=0, raw=False, json_path=None):
    """`pravaha schedule FILE.json`: run it, print a table (or the machine lines with raw)."""
    with open(path) as fh:
        sched = json.load(fh)
    try:
        validate(sched)
    except ScheduleError as e:
        print("pravaha schedule: %s" % e, file=sys.stderr)
        return 2
    lines = []
    emit = (lambda s: (lines.append(s), print(s, flush=True))) if raw else lines.append
    try:
        run_schedule(sched, CudaBackend(device), emit=emit)
    except Exception as e:
        print("pravaha schedule: %s" % e, file=sys.stderr)
        return 1
    parsed = parse(lines)
    if json_path:
        with open(json_path, "w") as fh:
            json.dump(parsed, fh, indent=1)
    if not raw:
        print(table(parsed))
    return 0 if (parsed["done"] or {}).get("checks_ok") else 3


TEMPLATE = r'''{
  "comment": "My first schedule. Edit the slots, then run: pravaha schedule my_test.json. Rules: probe rates first; each measured slot at least 2 s (traffic) or 60 s (energy); idle between slots; begin and end with idle. Guide: docs/MANUAL.md, Designing a schedule.",
  "pause_after_setup_s": 2,
  "slots": [
    {"name": "idle_before", "kind": "idle", "seconds": 10},
    {"name": "my_streaming_read", "kind": "kernel", "op": "read", "footprint_mib": 1024, "logical_mib": 65536},
    {"name": "gap", "kind": "idle", "seconds": 2},
    {"name": "my_copy", "kind": "copy", "direction": "h2d", "host": "pinned", "size_mib": 256, "count": 64},
    {"name": "idle_after", "kind": "idle", "seconds": 10}
  ]
}
'''
