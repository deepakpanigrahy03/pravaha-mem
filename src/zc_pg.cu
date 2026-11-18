// zc_pg.cu (v1): why do GPU writes to pageable memory produce extra sectors?
//
// Author: Deepak Panigrahy
// Date: 2026-11-16
// Part of pravaha (GB10 GPU memory-traffic measurement). License: MIT.
//
// Purpose: tests T1 and T2 of review_v7_tests.md. Written separately so that the
// validated zc.cu stays unchanged. The write kernel is the same as in zc.cu.
//
// One pageable 256 MiB buffer, allocated aligned to 2 MiB in every case, so the
// only difference between runs is the page size and the CPU fill.
//
// Arguments:
//   PAGES   small  normal 4 KiB pages (the setting of the original malloc test)
//           huge   madvise(MADV_HUGEPAGE) before first touch: 2 MiB pages
//   PASSES  number of write kernels on the same buffer (default 2)
//   FILL    fill   CPU writes the buffer before the GPU, as zc.cu does (default)
//           nofill the GPU is the first to touch the buffer
//
// Each pass is one kernel launch, so Nsight Compute reports each pass separately.
// The program prints the huge page memory of this process, so a run can be checked.
//
// Build:  nvcc -O2 -arch=sm_121 zc_pg.cu -o zc_pg
// Run:    ./zc_pg small|huge [PASSES=2] [fill|nofill]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <cuda_runtime.h>

// Write every 8-byte word once (identical to zc.cu).
__global__ void writek(unsigned long long* p, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = i;
}

// Huge page memory of this process in kB, from /proc/self/smaps_rollup.
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
    const char* pages = argc > 1 ? argv[1] : "small";
    int passes = argc > 2 ? atoi(argv[2]) : 2;
    bool fill = !(argc > 3 && !strcmp(argv[3], "nofill"));
    bool huge = !strcmp(pages, "huge");
    size_t bytes = 256ull << 20, n = bytes / 8, align = 2ull << 20;

    unsigned long long* h = nullptr;
    if (posix_memalign((void**)&h, align, bytes) != 0) { printf("status=alloc_failed\n"); return 1; }
    int adv = madvise(h, bytes, huge ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
    if (fill) for (size_t i = 0; i < n; i++) h[i] = i;   // CPU first touch, as zc.cu does
    long huge_kb = anon_huge_kb();

    cudaDeviceSynchronize();
    for (int r = 0; r < passes; r++) writek<<<1024, 256>>>(h, n);
    cudaError_t e = cudaDeviceSynchronize();

    printf("pages=%s passes=%d fill=%d madvise=%d anon_huge_kb_after_fill=%ld "
           "anon_huge_kb_after_gpu=%ld bytes_per_pass=%zu status=%s\n",
           pages, passes, fill, adv, huge_kb, anon_huge_kb(), bytes, cudaGetErrorString(e));
    free(h);
    return 0;
}
