// copy_calib.cu: copies of a known size, for calibrating the GPU's SMMU PMCG counter.
//
//   copy_calib h2d ROUNDS [MiB] [pinned|pageable]   exactly ROUNDS host-to-GPU copies of MiB (default 256)
//   copy_calib d2h ROUNDS [MiB]    exactly ROUNDS GPU-to-host copies
//   copy_calib loop SECONDS [MiB]  alternate both directions for SECONDS (background load for scans)
//
// Both buffers are allocated once and reused, so every round moves exactly MiB bytes. The host
// buffer is pinned (cudaHostAlloc, default) or pageable (malloc; the driver stages it), the
// device buffer is cudaMalloc memory.
// Allocating a new pageable buffer per round changes the counted transactions (see the paper).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); return 1; } } while (0)

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s h2d|d2h ROUNDS [MiB] [pinned|pageable] | loop SECONDS [MiB]\n", argv[0]); return 2; }
    const char* dir = argv[1];
    double arg = atof(argv[2]);
    size_t bytes = (size_t)(argc > 3 ? atoi(argv[3]) : 256) << 20;
    bool pageable = argc > 4 && !strcmp(argv[4], "pageable");
    void *h = nullptr, *d = nullptr;
    if (pageable) { h = malloc(bytes); if (!h) { fprintf(stderr, "malloc failed\n"); return 1; } }
    else CK(cudaHostAlloc(&h, bytes, cudaHostAllocDefault));
    CK(cudaMalloc(&d, bytes));
    memset(h, 1, bytes);
    CK(cudaMemset(d, 2, bytes));
    CK(cudaDeviceSynchronize());
    long copies = 0;
    if (!strcmp(dir, "loop")) {
        auto t0 = std::chrono::steady_clock::now();
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < arg) {
            CK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost));
            copies += 2;
        }
    } else {
        cudaMemcpyKind k = !strcmp(dir, "h2d") ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
        if (strcmp(dir, "h2d") && strcmp(dir, "d2h")) { fprintf(stderr, "direction must be h2d, d2h or loop\n"); return 2; }
        for (long r = 0; r < (long)arg; r++) {
            CK(k == cudaMemcpyHostToDevice ? cudaMemcpy(d, h, bytes, k) : cudaMemcpy(h, d, bytes, k));
            copies++;
        }
    }
    CK(cudaDeviceSynchronize());
    printf("dir=%s host=%s copies=%ld bytes=%zu\n", dir, pageable ? "pageable" : "pinned", copies, (size_t)copies * bytes);
    cudaFree(d);
    if (pageable) free(h); else cudaFreeHost(h);
    return 0;
}
