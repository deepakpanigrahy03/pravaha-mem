// zc.cu (v4): a GPU kernel reads (r) or writes (w) 256 MiB, N times, in one kind of memory.
//
// Author: Deepak Panigrahy
// Part of pravaha (GB10 GPU memory-traffic measurement). License: MIT.
//
// Purpose: give the counters a known amount of kernel traffic.
//   One round = 256 MiB = 8,388,608 sectors of 32 bytes, read or written once.
//
// Memory kinds (first argument):
//   pinned   cudaHostAlloc, mapped: CPU memory locked in place
//   malloc   ordinary (pageable) CPU memory; the GB10 GPU can read it directly
//   device   cudaMalloc: memory allocated for the GPU (on the GB10 it is still
//            in the shared LPDDR5X)
//   managed  cudaMallocManaged: one pointer for CPU and GPU (new in v4); filled
//            by the CPU first; with "prefetch" as 4th argument it is first moved
//            to the GPU with cudaMemPrefetchAsync, so both placements can be tested
//
// Build:  nvcc -O2 -arch=sm_121 zc.cu -o zc
// Run:    ./zc MODE [ROUNDS=8] [r|w] [prefetch]
// Results from v1 to v3 (pinned, malloc, device) are unchanged by v4.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

// Read every 8-byte word once (grid-stride loop, coalesced).
__global__ void readk(const unsigned long long* p, size_t n, unsigned long long* out) {
    unsigned long long s = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) s += p[i];
    atomicAdd(out, s);   // keeps the reads from being optimised away
}

// Write every 8-byte word once.
__global__ void writek(unsigned long long* p, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = i;
}

int main(int argc, char** argv) {
    const char* mode = argc > 1 ? argv[1] : "pinned";   // pinned, malloc, device or managed
    int rounds = argc > 2 ? atoi(argv[2]) : 8;
    char op = argc > 3 ? argv[3][0] : 'r';              // r = read, w = write
    bool prefetch = argc > 4 && !strcmp(argv[4], "prefetch");
    size_t bytes = 256ull << 20, n = bytes / 8;
    unsigned long long *h = nullptr, *dp, *out;

    if (!strcmp(mode, "device")) {
        cudaMalloc((void**)&h, bytes); cudaMemset(h, 1, bytes); dp = h;
    } else if (!strcmp(mode, "managed")) {
        cudaMallocManaged((void**)&h, bytes);
        for (size_t i = 0; i < n; i++) h[i] = i;          // CPU fills it first
        if (prefetch) {                                   // optionally move it to the GPU
            int dev = 0;
            cudaGetDevice(&dev);
#if CUDART_VERSION >= 13000
            cudaMemLocation loc = {};                     // CUDA 13: location is a struct
            loc.type = cudaMemLocationTypeDevice;
            loc.id = dev;
            cudaMemPrefetchAsync(h, bytes, loc, 0, 0);
#else
            cudaMemPrefetchAsync(h, bytes, dev, 0);
#endif
        }
        dp = h;
    } else {
        if (mode[0] == 'p') cudaHostAlloc((void**)&h, bytes, cudaHostAllocMapped);
        else h = (unsigned long long*)malloc(bytes);
        for (size_t i = 0; i < n; i++) h[i] = i;          // CPU fills it, as a real program would
        dp = h;
        if (mode[0] == 'p') cudaHostGetDevicePointer((void**)&dp, h, 0);
    }
    cudaMalloc((void**)&out, 8);
    cudaDeviceSynchronize();
    for (int r = 0; r < rounds; r++) { if (op == 'w') writek<<<1024, 256>>>(dp, n); else readk<<<1024, 256>>>(dp, n, out); }
    cudaError_t e = cudaDeviceSynchronize();
    printf("mode=%s op=%c rounds=%d prefetch=%d bytes=%zu status=%s\n", mode, op, rounds, prefetch,
           (size_t)rounds * bytes, cudaGetErrorString(e));
    return 0;
}
