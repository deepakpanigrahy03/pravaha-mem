// zc_time.cu
//
// How long does the GPU take to read or write 256 MiB, depending on
// where the memory came from?
//
//   pinned : cudaHostAlloc(..., cudaHostAllocMapped)  CPU memory, locked in place
//   malloc : plain malloc()                           ordinary (pageable) CPU memory
//   device : cudaMalloc()                             memory owned by the GPU
//
// On the GB10 all three live in the same LPDDR5X. zc.cu counts the traffic
// each one causes; this program measures the time. Same kernels, same launch
// shape, so the two results can be compared directly.
//
// Build:
//   /usr/local/cuda/bin/nvcc -O2 -arch=sm_121 zc_time.cu -o zc_time
//
// Run:
//   ./zc_time <mode> <op> [rounds] [touch]
//     mode   pinned | malloc | device
//     op     r (GPU reads) | w (GPU writes)
//     rounds how many timed rounds (default 20)
//     touch  t = the CPU reads the whole buffer between rounds (not timed),
//            like a real program that uses the GPU's results. Host modes only.
//
// Output: one CSV line per round, then a summary line starting with "summary".
// Do not run this under ncu: ncu replays kernels and the times become meaningless.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

// Stop at the first CUDA error and say where it happened.
#define CHECK(call)                                                          \
    do {                                                                     \
        cudaError_t err_ = (call);                                           \
        if (err_ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error: %s (%s:%d)\n",                      \
                    cudaGetErrorString(err_), __FILE__, __LINE__);           \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

// Each thread walks the buffer with a stride of the whole grid, so neighbouring
// threads touch neighbouring words (coalesced access). Identical to zc.cu.
__global__ void readk(const unsigned long long* p, size_t n, unsigned long long* out) {
    unsigned long long sum = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        sum += p[i];
    atomicAdd(out, sum);  // keeps the compiler from removing the reads
}

__global__ void writek(unsigned long long* p, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        p[i] = i;
}

static void usage(const char* prog) {
    fprintf(stderr, "usage: %s pinned|malloc|device r|w [rounds] [t]\n", prog);
    exit(2);
}

int main(int argc, char** argv) {
    if (argc < 3) usage(argv[0]);
    const char* mode = argv[1];
    const char op = argv[2][0];
    const int rounds = argc > 3 ? atoi(argv[3]) : 20;
    const bool touch = argc > 4 && argv[4][0] == 't';

    const bool is_pinned = strcmp(mode, "pinned") == 0;
    const bool is_malloc = strcmp(mode, "malloc") == 0;
    const bool is_device = strcmp(mode, "device") == 0;
    if (!(is_pinned || is_malloc || is_device) || (op != 'r' && op != 'w') || rounds < 1)
        usage(argv[0]);
    if (touch && is_device) {
        fprintf(stderr, "touch (t) only makes sense for pinned or malloc\n");
        return 2;
    }

    const size_t bytes = 256ull << 20;   // 256 MiB, about 11 times the GPU's L2
    const size_t n = bytes / sizeof(unsigned long long);

    // host_ptr: address the CPU uses; gpu_ptr: address the kernel uses.
    unsigned long long* host_ptr = nullptr;
    unsigned long long* gpu_ptr = nullptr;

    if (is_device) {
        CHECK(cudaMalloc((void**)&gpu_ptr, bytes));
        CHECK(cudaMemset(gpu_ptr, 1, bytes));
    } else {
        if (is_pinned) {
            CHECK(cudaHostAlloc((void**)&host_ptr, bytes, cudaHostAllocMapped));
        } else {
            host_ptr = (unsigned long long*)malloc(bytes);
            if (!host_ptr) { fprintf(stderr, "malloc failed\n"); return 1; }
        }
        // The CPU fills the buffer first, as a real program would. This also
        // makes sure every page exists before the GPU touches it.
        for (size_t i = 0; i < n; i++) host_ptr[i] = i;
        gpu_ptr = host_ptr;
        if (is_pinned) CHECK(cudaHostGetDevicePointer((void**)&gpu_ptr, host_ptr, 0));
    }

    unsigned long long* sum_out = nullptr;
    CHECK(cudaMalloc((void**)&sum_out, sizeof(unsigned long long)));

    auto launch = [&]() {
        if (op == 'w') writek<<<1024, 256>>>(gpu_ptr, n);
        else           readk<<<1024, 256>>>(gpu_ptr, n, sum_out);
    };

    // One untimed round first: settles clocks, page tables and first-touch costs.
    launch();
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    std::vector<float> ms(rounds);
    volatile unsigned long long cpu_sum = 0;  // volatile: keep the CPU reads

    printf("mode,op,touch,round,ms,gb_per_s\n");
    for (int r = 0; r < rounds; r++) {
        CHECK(cudaEventRecord(start));
        launch();
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        CHECK(cudaGetLastError());
        CHECK(cudaEventElapsedTime(&ms[r], start, stop));
        printf("%s,%c,%d,%d,%.3f,%.2f\n", mode, op, touch, r, ms[r],
               bytes / (ms[r] * 1e-3) / 1e9);

        if (touch) {  // CPU uses the result, pulling the lines into its caches
            unsigned long long s = 0;
            for (size_t i = 0; i < n; i++) s += host_ptr[i];
            cpu_sum = s;
        }
    }

    // Median is robust to the odd slow round; min and max show the spread.
    std::vector<float> sorted = ms;
    std::sort(sorted.begin(), sorted.end());
    const float median = sorted[rounds / 2];
    printf("summary,%s,%c,%d,rounds=%d,median_ms=%.3f,min_ms=%.3f,max_ms=%.3f,median_gb_per_s=%.2f\n",
           mode, op, touch, rounds, median, sorted.front(), sorted.back(),
           bytes / (median * 1e-3) / 1e9);

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(sum_out));
    if (is_device)      CHECK(cudaFree(gpu_ptr));
    else if (is_pinned) CHECK(cudaFreeHost(host_ptr));
    else                free(host_ptr);
    return 0;
}
