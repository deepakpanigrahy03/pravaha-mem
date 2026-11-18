// pattern.cu: GPU kernels that read memory in a chosen access pattern, with the
// number of 32-byte sectors they SHOULD touch known in advance.
//
// Author: Deepak Panigrahy
// Part of pravaha (GB10 GPU memory-traffic measurement). License: MIT.
//
// Purpose:
//   N7 (access pattern): when reads do not fill whole sectors, what does the L2
//       event count? We compare measured sectors with the expected number of
//       UNIQUE sectors touched, never with the logical bytes requested.
//   N8 (warm vs cold cache): read a buffer smaller than the 24 MiB L2 several
//       times; under "--cache-control none" later rounds should hit in L2.
//
// Patterns (first argument):
//   seq     every 4-byte word once           logical = MiB,  unique sectors = MiB/32
//   stride  one 4-byte word every S bytes    S >= 32: one sector per read
//   random  one 4-byte word in each of a pseudo-random set of DISTINCT sectors
//           (index = i * P mod nsectors, P prime and coprime with nsectors, so
//           no sector repeats and the expected count is exact)
//
// Build:  nvcc -O2 -arch=sm_121 pattern.cu -o pattern
// Run:    ./pattern seq|stride|random [MiB=256] [ROUNDS=1] [S_or_COUNT]
//           stride: S = stride in bytes (default 128)
//           random: COUNT = sectors to touch (default nsectors/4)
// Prints: pattern, logical_bytes, expected_sectors (per round), rounds.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

static const unsigned long long PRIME = 2654435761ull;  // odd prime; buffer sector counts are powers of 2

// Read one 4-byte word at word index idx(i) for i < count (grid-stride loop).
// kind 0 = seq, 1 = stride (step words), 2 = random distinct sectors.
__global__ void readp(const unsigned* p, size_t count, int kind, size_t step,
                      size_t nsectors, unsigned long long* out) {
    unsigned long long s = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < count;
         i += (size_t)gridDim.x * blockDim.x) {
        size_t w;
        if (kind == 0) w = i;
        else if (kind == 1) w = i * step;
        else w = ((i * PRIME) % nsectors) * 8;     // first word of a distinct sector
        s += p[w];
    }
    atomicAdd(out, s);   // keeps the reads from being optimised away
}

int main(int argc, char** argv) {
    const char* pat = argc > 1 ? argv[1] : "seq";
    size_t bytes = (size_t)(argc > 2 ? atoi(argv[2]) : 256) << 20;
    int rounds = argc > 3 ? atoi(argv[3]) : 1;
    size_t nsectors = bytes / 32, count, step = 0, expected;
    int kind;
    if (!strcmp(pat, "stride")) {
        size_t sb = argc > 4 ? (size_t)atoll(argv[4]) : 128;
        kind = 1; step = sb / 4; count = bytes / sb;
        expected = sb >= 32 ? count : bytes / 32;           // under 32 B every sector is touched
    } else if (!strcmp(pat, "random")) {
        kind = 2; count = argc > 4 ? (size_t)atoll(argv[4]) : nsectors / 4;
        if (count > nsectors) count = nsectors;
        expected = count;
    } else {
        kind = 0; count = bytes / 4; expected = nsectors;
    }
    unsigned* d; unsigned long long* out;
    cudaMalloc((void**)&d, bytes); cudaMemset(d, 1, bytes);
    cudaMalloc((void**)&out, 8);
    cudaDeviceSynchronize();
    for (int r = 0; r < rounds; r++) readp<<<1024, 256>>>(d, count, kind, step, nsectors, out);
    cudaError_t e = cudaDeviceSynchronize();
    printf("pattern=%s bytes=%zu rounds=%d logical_bytes=%zu expected_sectors=%zu status=%s\n",
           pat, bytes, rounds, count * 4, expected, cudaGetErrorString(e));
    return 0;
}
