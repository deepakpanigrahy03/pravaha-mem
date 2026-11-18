// ats_probe.cu (v1): SMMU TCU events around GPU copies only, for one allocation type.
//
// Author: Deepak Panigrahy
// Date: 2026-11-17
// Part of pravaha-mem (GB10 GPU memory traffic measurement). License: MIT.
//
// Counts four events of the TCU PMCG (default smmuv3_pmcg_13002) in a window that
// contains only the copies: CUDA start, allocation, registration and one warm up copy
// happen before the window opens. Prints one CSV row per measured batch.
//
// Events: transaction (0x01), tlb_miss (0x02), trans_table_walk_access (0x04),
//         pcie_ats_trans_rq (0x06)
//
// Allocation types for the host buffer:
//   pinned    cudaHostAlloc (as pravaha's calibration)
//   reg4k     malloc aligned to 2 MiB, madvise(MADV_NOHUGEPAGE), touched, cudaHostRegister
//   regthp    the same with madvise(MADV_HUGEPAGE): transparent 2 MiB pages
//   pageable  plain malloc, not registered (CUDA stages the copy itself)
//
// Directions: h2d, d2h, d2d (d2d uses two cudaMalloc buffers; no host buffer)
//
// Usage:  ats_probe ALLOC DIR MIB COPIES REPEATS FRESH [UNIT]
//   FRESH=0: one buffer set for all repeats; FRESH=1: new buffers for every repeat
// Example: ats_probe pinned h2d 256 8 10 0
// Build:  nvcc -O2 -arch=sm_121 ats_probe.cu -o ats_probe
// Needs read access to the counter (perf_event_paranoid <= 0, cap_perfmon, or sudo).

#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cuda_runtime.h>

#define SYSFS "/sys/bus/event_source/devices"
static const char* EVENTS[] = {"transaction", "pcie_ats_trans_rq", "tlb_miss", "trans_table_walk_access"};
static const int NEV = 4;

static int read_text(const char* unit, const char* rel, char* buf, size_t len) {
    char path[256];
    snprintf(path, sizeof path, SYSFS "/%s/%s", unit, rel);
    FILE* f = fopen(path, "r");
    if (!f) return -1;
    if (!fgets(buf, (int)len, f)) { fclose(f); return -1; }
    fclose(f);
    buf[strcspn(buf, "\n")] = '\0';
    return 0;
}

static int place(const char* unit, const char* name, uint64_t value, perf_event_attr* a) {
    char spec[64], rel[96];
    snprintf(rel, sizeof rel, "format/%s", name);
    if (read_text(unit, rel, spec, sizeof spec)) return -1;
    char* colon = strchr(spec, ':');
    if (!colon) return -1;
    *colon = '\0';
    int lo = 0, hi = 0;
    if (sscanf(colon + 1, "%d-%d", &lo, &hi) == 1) hi = lo;
    uint64_t mask = (hi - lo == 63) ? ~0ULL : (((1ULL << (hi - lo + 1)) - 1) << lo);
    uint64_t bits = (value << lo) & mask;
    if (!strcmp(spec, "config")) a->config |= bits;
    else if (!strcmp(spec, "config1")) a->config1 |= bits;
    else if (!strcmp(spec, "config2")) a->config2 |= bits;
    else return -1;
    return 0;
}

static int open_event(const char* unit, const char* ev) {
    char text[128], rel[96];
    perf_event_attr attr;
    memset(&attr, 0, sizeof attr);
    attr.size = sizeof attr;
    attr.disabled = 1;
    if (read_text(unit, "type", text, sizeof text)) return -1;
    attr.type = (uint32_t)strtoul(text, NULL, 10);
    snprintf(rel, sizeof rel, "events/%s", ev);
    if (read_text(unit, rel, text, sizeof text)) return -1;
    for (char* tok = strtok(text, ","); tok; tok = strtok(NULL, ",")) {
        char* eq = strchr(tok, '=');
        uint64_t v = 1;
        if (eq) { *eq = '\0'; v = strtoull(eq + 1, NULL, 0); }
        if (place(unit, tok, v, &attr)) return -1;
    }
    int cpu = 0;
    if (read_text(unit, "cpumask", text, sizeof text) == 0) cpu = atoi(text);
    return (int)syscall(SYS_perf_event_open, &attr, -1, cpu, -1, PERF_FLAG_FD_CLOEXEC);
}

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

static double now() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

struct Bufs { void* h; void* d; void* d2; bool registered, pinned, plain; };

static int alloc(Bufs* b, const char* kind, const char* dir, size_t bytes) {
    memset(b, 0, sizeof *b);
    if (cudaMalloc(&b->d, bytes) != cudaSuccess) return -1;
    if (!strcmp(dir, "d2d")) return cudaMalloc(&b->d2, bytes) == cudaSuccess ? 0 : -1;
    if (!strcmp(kind, "pinned")) {
        b->pinned = true;
        return cudaHostAlloc(&b->h, bytes, cudaHostAllocDefault) == cudaSuccess ? 0 : -1;
    }
    if (!strcmp(kind, "pageable")) {
        b->plain = true;
        b->h = malloc(bytes);
        if (!b->h) return -1;
        memset(b->h, 1, bytes);
        return 0;
    }
    bool thp = !strcmp(kind, "regthp");
    if (!thp && strcmp(kind, "reg4k")) return -2;
    if (posix_memalign(&b->h, 2u << 20, bytes)) return -1;
    madvise(b->h, bytes, thp ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
    memset(b->h, 1, bytes);
    b->registered = true;
    return cudaHostRegister(b->h, bytes, cudaHostRegisterDefault) == cudaSuccess ? 0 : -1;
}

static void release(Bufs* b) {
    if (b->registered) { cudaHostUnregister(b->h); free(b->h); }
    else if (b->pinned) cudaFreeHost(b->h);
    else if (b->plain) free(b->h);
    if (b->d) cudaFree(b->d);
    if (b->d2) cudaFree(b->d2);
}

static void copy(Bufs* b, const char* dir, size_t bytes) {
    if (!strcmp(dir, "h2d")) cudaMemcpy(b->d, b->h, bytes, cudaMemcpyHostToDevice);
    else if (!strcmp(dir, "d2h")) cudaMemcpy(b->h, b->d, bytes, cudaMemcpyDeviceToHost);
    else cudaMemcpy(b->d2, b->d, bytes, cudaMemcpyDeviceToDevice);
}

int main(int argc, char** argv) {
    if (argc < 7) {
        fprintf(stderr, "usage: ats_probe pinned|reg4k|regthp|pageable h2d|d2h|d2d MIB COPIES REPEATS FRESH [UNIT]\n");
        return 1;
    }
    const char* kind = argv[1];
    const char* dir = argv[2];
    size_t bytes = (size_t)atol(argv[3]) << 20;
    int copies = atoi(argv[4]), repeats = atoi(argv[5]), fresh = atoi(argv[6]);
    const char* unit = argc > 7 ? argv[7] : "smmuv3_pmcg_13002";

    int fd[NEV];
    for (int i = 0; i < NEV; i++) {
        fd[i] = open_event(unit, EVENTS[i]);
        if (fd[i] < 0) {
            fprintf(stderr, "cannot open %s/%s: %s (needs perf permission)\n", unit, EVENTS[i], strerror(errno));
            return 1;
        }
    }

    printf("alloc,dir,mib,copies,repeat,bytes,transaction,ats_rq,tlb_miss,walks,seconds,anon_huge_kb\n");
    Bufs b;
    int rc = 0;
    for (int r = 1; r <= repeats; r++) {
        if (r == 1 || fresh) {
            if (r > 1) release(&b);
            if ((rc = alloc(&b, kind, dir, bytes))) {
                fprintf(stderr, "allocation failed (%s)\n", rc == -2 ? "unknown kind" : "CUDA");
                return 1;
            }
            copy(&b, dir, bytes);                       // warm up, outside the window
            cudaDeviceSynchronize();
        }
        long hp = anon_huge_kb();
        for (int i = 0; i < NEV; i++) { ioctl(fd[i], PERF_EVENT_IOC_RESET, 0); ioctl(fd[i], PERF_EVENT_IOC_ENABLE, 0); }
        double t0 = now();
        for (int c = 0; c < copies; c++) copy(&b, dir, bytes);
        cudaDeviceSynchronize();
        double dt = now() - t0;
        uint64_t v[NEV];
        for (int i = 0; i < NEV; i++) {
            ioctl(fd[i], PERF_EVENT_IOC_DISABLE, 0);
            if (read(fd[i], &v[i], sizeof v[i]) != (ssize_t)sizeof v[i]) v[i] = 0;
        }
        printf("%s,%s,%zu,%d,%d,%zu,%llu,%llu,%llu,%llu,%.4f,%ld\n", kind, dir, bytes >> 20, copies, r,
               (size_t)copies * bytes, (unsigned long long)v[0], (unsigned long long)v[1],
               (unsigned long long)v[2], (unsigned long long)v[3], dt, hp);
        fflush(stdout);
    }
    release(&b);
    for (int i = 0; i < NEV; i++) close(fd[i]);
    return 0;
}
