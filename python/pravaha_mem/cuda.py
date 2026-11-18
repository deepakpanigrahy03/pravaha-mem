"""Just enough of the CUDA runtime, through ctypes, to make copies of a known size.

Author: Deepak Panigrahy
Part of pravaha (GB10 GPU memory-traffic measurement). License: MIT.

Used by calibration and by the validation checks, so neither needs PyTorch nor
a compiler: only libcudart, which every CUDA install has.

What it can do:
    Buffers(nbytes, pageable)   one host buffer (pinned or ordinary malloc) and two GPU buffers
    .copy("h2d"|"d2h"|"d2d")    one blocking copy of the whole buffer
    .copy_async(direction, n)   n asynchronous copies spread over 2 streams, then wait
    runtime_version()           CUDA runtime version, e.g. "13.0"
"""

import ctypes
import ctypes.util
import glob
import os

# cudaMemcpyKind values from cuda_runtime_api.h
_KIND = {"h2d": 1, "d2h": 2, "d2d": 3}


class CudaError(RuntimeError):
    """A CUDA runtime call failed; the message is CUDA's own error string."""


def _load_cudart():
    """Find libcudart: $PRAVAHA_CUDART, then the linker's search path, then /usr/local/cuda."""
    names = []
    if os.environ.get("PRAVAHA_CUDART"):
        names.append(os.environ["PRAVAHA_CUDART"])
    found = ctypes.util.find_library("cudart")
    if found:
        names.append(found)
    # newest version first; these are fallbacks after the two choices above
    names += sorted(glob.glob("/usr/local/cuda/lib64/libcudart.so*"), reverse=True)
    names += sorted(glob.glob("/usr/local/cuda/targets/*/lib/libcudart.so*"), reverse=True)
    for n in names:
        try:
            rt = ctypes.CDLL(n)
            rt.cudaGetErrorString.restype = ctypes.c_char_p
            return rt
        except OSError:
            continue
    raise CudaError("libcudart not found; set PRAVAHA_CUDART=/path/to/libcudart.so")


def runtime_version():
    """CUDA runtime version as 'major.minor' (cudaRuntimeGetVersion gives e.g. 13000)."""
    rt = _load_cudart()
    v = ctypes.c_int()
    if rt.cudaRuntimeGetVersion(ctypes.byref(v)) != 0:
        return None
    return f"{v.value // 1000}.{(v.value % 1000) // 10}"


class Buffers:
    """One host buffer (pinned or pageable) and two GPU buffers of the same size.

    The second GPU buffer exists only for device-to-device copies.
    Every page is touched once at allocation, so first-touch costs never
    land inside a measurement.
    """

    def __init__(self, nbytes, pageable=False):
        self.rt = _load_cudart()
        self.nbytes = nbytes
        self.pageable = pageable
        self.host = ctypes.c_void_p()
        self.dev = ctypes.c_void_p()
        self.dev2 = ctypes.c_void_p()
        if pageable:
            # ordinary malloc memory: what most programs use without thinking about it
            self._libc = ctypes.CDLL(None)
            self._libc.malloc.restype = ctypes.c_void_p
            self.host = ctypes.c_void_p(self._libc.malloc(ctypes.c_size_t(nbytes)))
            if not self.host.value:
                raise CudaError("malloc failed")
        else:
            # pinned memory: locked in place, the GPU's copy engine reads it directly
            self._check(self.rt.cudaMallocHost(ctypes.byref(self.host), ctypes.c_size_t(nbytes)))
        self._check(self.rt.cudaMalloc(ctypes.byref(self.dev), ctypes.c_size_t(nbytes)))
        self._check(self.rt.cudaMalloc(ctypes.byref(self.dev2), ctypes.c_size_t(nbytes)))
        ctypes.memset(self.host, 1, nbytes)
        self._check(self.rt.cudaMemset(self.dev, 2, ctypes.c_size_t(nbytes)))
        self._check(self.rt.cudaMemset(self.dev2, 3, ctypes.c_size_t(nbytes)))
        self.streams = []
        self.sync()

    def _check(self, rc):
        if rc != 0:
            raise CudaError(self.rt.cudaGetErrorString(rc).decode())

    def _ends(self, direction):
        """(destination, source) pointers for a direction."""
        return {"h2d": (self.dev, self.host), "d2h": (self.host, self.dev),
                "d2d": (self.dev2, self.dev)}[direction]

    def copy(self, direction):
        """One blocking copy of the whole buffer ('h2d', 'd2h' or 'd2d')."""
        dst, src = self._ends(direction)
        self._check(self.rt.cudaMemcpy(dst, src, ctypes.c_size_t(self.nbytes),
                                       ctypes.c_int(_KIND[direction])))

    def copy_async(self, direction, count, nstreams=2):
        """`count` asynchronous copies, alternating over `nstreams` streams, then wait.

        This is how frameworks usually copy (cudaMemcpyAsync on streams).
        Pinned memory only: pageable async copies are staged by the driver.
        """
        if self.pageable:
            raise CudaError("async copies need pinned host memory")
        while len(self.streams) < nstreams:
            s = ctypes.c_void_p()
            self._check(self.rt.cudaStreamCreate(ctypes.byref(s)))
            self.streams.append(s)
        dst, src = self._ends(direction)
        for i in range(count):
            self._check(self.rt.cudaMemcpyAsync(dst, src, ctypes.c_size_t(self.nbytes),
                                                ctypes.c_int(_KIND[direction]),
                                                self.streams[i % nstreams]))
        self.sync()

    def sync(self):
        self._check(self.rt.cudaDeviceSynchronize())

    def free(self):
        """Release everything; safe to call twice."""
        for s in self.streams:
            self.rt.cudaStreamDestroy(s)
        self.streams = []
        for p in (self.dev, self.dev2):
            if p.value:
                self.rt.cudaFree(p)
        self.dev = self.dev2 = ctypes.c_void_p()
        if self.host.value:
            if self.pageable:
                self._libc.free(self.host)
            else:
                self.rt.cudaFreeHost(self.host)
            self.host = ctypes.c_void_p()
