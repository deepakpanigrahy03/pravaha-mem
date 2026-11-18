// mixed.cu: a workload with KNOWN copy traffic and KNOWN kernel traffic.
//
// Author: Deepak Panigrahy
// Part of pravaha (GB10 GPU memory-traffic measurement). License: MIT.
//
// Purpose (letter, experiment N1): check that the copy counter and the L2
// cache counters together account for a workload whose bytes we know.
//
// What it does, in order:
//   1. set up: allocate a pinned host buffer and a device buffer (COPY_MiB each)
//      and a separate 256 MiB device work buffer; touch everything once;
//   2. print "READY" and wait for a line on standard input;
//   3. the measured work:
//        COPIES copies host-to-device, then
//        ROUNDS kernel passes that READ the work buffer, then
//        ROUNDS kernel passes that WRITE the work buffer, then
//        COPIES copies device-to-host;
//   4. print "DONE" with the known byte counts, wait for another line, exit.
//
// Why the two waits: pravaha starts its counter after READY and stops it after
// DONE, so CUDA start-up and allocation never enter the copy count. Under
// Nsight Compute the program gets two empty lines on standard input.
//
// Build:  nvcc -O2 -arch=sm_121 mixed.cu -o mixed
// Run:    ./mixed [COPY_MiB=256] [COPIES=8] [ROUNDS=4]

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Stop with CUDA's message if a call fails.
#define CHECK(call)                                                            \
    do {                                                                       \
        cudaError_t e_ = (call);                                               \
        if (e_ != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), \
                    __FILE__, __LINE__);                                       \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// Read every 8-byte word once (grid-stride loop: neighbouring threads read
// neighbouring words, so every 32-byte sector is fetched exactly once).
__global__ void readk(const unsigned long long* p, size_t n, unsigned long long* out) {
    unsigned long long s = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        s += p[i];
    atomicAdd(out, s);   // keeps the compiler from removing the reads
}

// Write every 8-byte word once.
__global__ void writek(unsigned long long* p, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        p[i] = i;
}

// Wait for one line on standard input (the measuring side's signal).
static void wait_line() {
    int c;
    while ((c = getchar()) != '\n' && c != EOF) {
    }
}

int main(int argc, char** argv) {
    const size_t copy_bytes = (size_t)(argc > 1 ? atoi(argv[1]) : 256) << 20;
    const int copies = argc > 2 ? atoi(argv[2]) : 8;
    const int rounds = argc > 3 ? atoi(argv[3]) : 4;
    const size_t work_bytes = 256ull << 20;          // about 11x the GB10's 24 MiB L2
    const size_t n = work_bytes / sizeof(unsigned long long);

    // ---- 1. set up (not measured) ----
    void *host = nullptr, *dev = nullptr;
    unsigned long long *work = nullptr, *out = nullptr;
    CHECK(cudaMallocHost(&host, copy_bytes));
    CHECK(cudaMalloc(&dev, copy_bytes));
    CHECK(cudaMalloc((void**)&work, work_bytes));
    CHECK(cudaMalloc((void**)&out, sizeof(unsigned long long)));
    for (size_t i = 0; i < copy_bytes; i += 4096) ((char*)host)[i] = 1;   // touch pages
    CHECK(cudaMemset(dev, 0, copy_bytes));
    CHECK(cudaMemset(work, 1, work_bytes));
    CHECK(cudaDeviceSynchronize());

    printf("READY\n");
    fflush(stdout);
    wait_line();

    // ---- 3. the measured work ----
    for (int i = 0; i < copies; i++) CHECK(cudaMemcpy(dev, host, copy_bytes, cudaMemcpyHostToDevice));
    for (int r = 0; r < rounds; r++) readk<<<1024, 256>>>(work, n, out);
    for (int r = 0; r < rounds; r++) writek<<<1024, 256>>>(work, n);
    for (int i = 0; i < copies; i++) CHECK(cudaMemcpy(host, dev, copy_bytes, cudaMemcpyDeviceToHost));
    CHECK(cudaDeviceSynchronize());

    // ---- 4. report the known bytes ----
    printf("DONE copy_mib=%zu copies_per_direction=%d copy_bytes=%zu read_bytes=%zu write_bytes=%zu\n",
           copy_bytes >> 20, copies, 2 * (size_t)copies * copy_bytes,
           (size_t)rounds * work_bytes, (size_t)rounds * work_bytes);
    fflush(stdout);
    wait_line();
    return 0;
}
