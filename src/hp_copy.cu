// hp_copy.cu (v1): do 2 MiB host pages change the SMMU translation count of GPU copies?
//
// Author: Deepak Panigrahy
// Date: 2026-11-17
// Part of pravaha-mem (GB10 GPU memory traffic measurement). License: MIT.
//
// Test of the hypothesis that smmuv3_pmcg_13002 'transaction' counts GPU ATS translation
// requests, about one per 4 KiB page on each side of a copy. A 256 MiB host buffer is
// aligned to 2 MiB and advised as either 4 KiB pages (small) or transparent huge pages
// (huge), touched by the CPU, registered with cudaHostRegister, then copied N times to
// cudaMalloc memory. Run with N = 0 and N = 16 and subtract, to remove CUDA start.
// The program prints the process's huge page memory so each run can be checked.
//
// Build:  nvcc -O2 -arch=sm_121 hp_copy.cu -o hp_copy
// Run:    ./hp_copy small|huge N

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <cuda_runtime.h>

static long anon_huge_kb() {
    FILE* f = fopen("/proc/self/smaps_rollup", "r");
    if (!f) return -1;
    char line[256];
    long kb = -1;
    while (fgets(line, sizeof line, f))
        if (sscanf(line, "AnonHugePages: %ld kB", &kb) == 1) break;
    fclose(f);
    return kb;
}

int main(int argc, char** argv) {
    if (argc != 3) { printf("usage: hp_copy small|huge N\n"); return 1; }
    bool huge = !strcmp(argv[1], "huge");
    int n = atoi(argv[2]);
    size_t bytes = 256ull << 20, align = 2ull << 20;

    void* h = nullptr;
    if (posix_memalign(&h, align, bytes)) { printf("status=alloc_failed\n"); return 1; }
    int adv = madvise(h, bytes, huge ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
    memset(h, 1, bytes);                                   // CPU first touch: pages become real
    long hp = anon_huge_kb();

    cudaError_t e = cudaHostRegister(h, bytes, cudaHostRegisterDefault);
    if (e != cudaSuccess) { printf("status=register_failed %s\n", cudaGetErrorString(e)); return 1; }
    void* d = nullptr;
    cudaMalloc(&d, bytes);
    cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice);       // warm up (counted in N = 0 too)
    cudaDeviceSynchronize();

    for (int i = 0; i < n; i++) cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice);
    e = cudaDeviceSynchronize();

    printf("pages=%s copies=%d bytes=%zu madvise=%d anon_huge_kb=%ld status=%s\n",
           argv[1], n, (size_t)n * bytes, adv, hp, cudaGetErrorString(e));
    cudaHostUnregister(h);
    cudaFree(d);
    free(h);
    return 0;
}
