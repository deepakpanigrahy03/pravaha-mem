// gpu_info.cu: GPU properties that matter for interpreting the traffic counters, as key,value CSV.
// l2_bytes tells whether a test buffer is larger than the L2 (so every round reaches memory);
// pageable_access=1 means kernels can read and write ordinary malloc memory directly.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) { fprintf(stderr, "no CUDA device\n"); return 1; }
    int v = 0;
    printf("key,value\n");
    printf("gpu_name,%s\n", p.name);
    printf("compute_capability,%d.%d\n", p.major, p.minor);
    printf("sm_count,%d\n", p.multiProcessorCount);
    printf("l2_bytes,%d\n", p.l2CacheSize);
    printf("integrated,%d\n", p.integrated);
    cudaDeviceGetAttribute(&v, cudaDevAttrPageableMemoryAccess, 0);
    printf("pageable_access,%d\n", v);
    cudaDeviceGetAttribute(&v, cudaDevAttrPageableMemoryAccessUsesHostPageTables, 0);
    printf("uses_host_page_tables,%d\n", v);
    cudaDeviceGetAttribute(&v, cudaDevAttrConcurrentManagedAccess, 0);
    printf("concurrent_managed_access,%d\n", v);
    int rt = 0, drv = 0;
    cudaRuntimeGetVersion(&rt); cudaDriverGetVersion(&drv);
    printf("cuda_runtime,%d.%d\n", rt / 1000, (rt % 1000) / 10);
    printf("cuda_driver_api,%d.%d\n", drv / 1000, (drv % 1000) / 10);
    return 0;
}
