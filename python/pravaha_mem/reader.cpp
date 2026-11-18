/* pravaha-reader: live, device wide sampler of the GPU's L2 cache counters (NVIDIA GB10).
 * Part of pravaha-mem. MIT License. Copyright (c) 2026 MANTRA Labs.
 * Written from the public CUPTI headers. Runs no GPU work of its own; it only reads counters.
 *
 * Protocol (stdout, version 2), one line each:
 *   V 2                                      protocol version, once
 *   M <name1> ... <name5>                    metric names, once
 *   C <cupti_ns> <realtime_ns> <monotonic_ns> clock pair, at start and about every second
 *   S <idx> <t_start_ns> <t_end_ns> <v1> ... <v5>   one per completed sample (L2 sectors)
 *   E ok <n> | E busy | E error <code>       on exit
 * Stops on SIGINT or SIGTERM, or after --seconds N.
 * Usage: pravaha-reader [--interval-ms 10] [--seconds N] [--device 0]
 */
#include <cuda.h>
#include <cupti_activity.h>
#include <cupti_target.h>
#include <cupti_profiler_target.h>
#include <cupti_profiler_host.h>
#include <cupti_pmsampling.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define NM 5
static const char *METRICS[NM] = {
    "lts__t_sectors_srcunit_ce.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_miss_realtime.sum",
    "lts__t_sectors_srcunit_tex_op_write_realtime.sum",
    "lts__t_sectors_op_read_lookup_miss_realtime.sum",
    "lts__t_sectors_op_write_realtime.sum",
};

static volatile sig_atomic_t g_stop = 0;
static void on_sig(int s) { (void)s; g_stop = 1; }

#define CK(call) do { CUptiResult r_ = (call); \
    if (r_ == CUPTI_ERROR_HARDWARE_BUSY) { \
        fprintf(stderr, "pravaha-reader: GPU counters busy: another profiler is using them " \
                "(another pravaha, Nsight Compute, Nsight Systems GPU metrics or DCGM profiling). " \
                "Stop it and try again.\n"); \
        printf("E busy\n"); fflush(stdout); exit(3); } \
    if (r_ != CUPTI_SUCCESS) { \
        const char *m_ = 0; cuptiGetResultString(r_, &m_); \
        fprintf(stderr, "pravaha-reader: %s failed: %s (%d)\n", #call, m_ ? m_ : "?", (int)r_); \
        printf("E error %d\n", (int)r_); fflush(stdout); exit(2); } } while (0)

#define P(T, v) T v; memset(&v, 0, sizeof v); v.structSize = T##_STRUCT_SIZE

static unsigned long long ns(clockid_t c) {
    struct timespec t; clock_gettime(c, &t);
    return (unsigned long long)t.tv_sec * 1000000000ull + (unsigned long long)t.tv_nsec;
}

static void clock_pair(void) {
    uint64_t g = 0;
    unsigned long long r0 = ns(CLOCK_REALTIME);
    cuptiGetTimestamp(&g);
    unsigned long long r1 = ns(CLOCK_REALTIME), m = ns(CLOCK_MONOTONIC);
    printf("C %llu %llu %llu\n", (unsigned long long)g, (r0 + r1) / 2, m);
}

int main(int argc, char **argv)
{
    uint64_t interval_ms = 10; double seconds = 0; size_t dev = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--interval-ms") && i + 1 < argc) interval_ms = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--seconds") && i + 1 < argc) seconds = atof(argv[++i]);
        else if (!strcmp(argv[i], "--device") && i + 1 < argc) dev = strtoull(argv[++i], 0, 10);
        else { fprintf(stderr, "usage: %s [--interval-ms N] [--seconds S] [--device D]\n", argv[0]); return 1; }
    }
    if (interval_ms < 1 || interval_ms > 1000) { fprintf(stderr, "pravaha-reader: --interval-ms must be 1..1000\n"); return 1; }
    signal(SIGINT, on_sig); signal(SIGTERM, on_sig);

    if (cuInit(0) != CUDA_SUCCESS) { fprintf(stderr, "pravaha-reader: cuInit failed\n"); printf("E error -1\n"); return 2; }
    { P(CUpti_Profiler_Initialize_Params, p); CK(cuptiProfilerInitialize(&p)); }

    P(CUpti_Device_GetChipName_Params, cn); cn.deviceIndex = dev; CK(cuptiDeviceGetChipName(&cn));
    P(CUpti_PmSampling_GetCounterAvailability_Params, ca); ca.deviceIndex = dev;
    CK(cuptiPmSamplingGetCounterAvailability(&ca));
    uint8_t *avail = (uint8_t *)malloc(ca.counterAvailabilityImageSize);
    ca.pCounterAvailabilityImage = avail; CK(cuptiPmSamplingGetCounterAvailability(&ca));

    P(CUpti_Profiler_Host_Initialize_Params, hi);
    hi.profilerType = CUPTI_PROFILER_TYPE_PM_SAMPLING; hi.pChipName = cn.pChipName;
    hi.pCounterAvailabilityImage = avail; CK(cuptiProfilerHostInitialize(&hi));
    CUpti_Profiler_Host_Object *host = hi.pHostObject;
    P(CUpti_Profiler_Host_ConfigAddMetrics_Params, am);
    am.pHostObject = host; am.ppMetricNames = METRICS; am.numMetrics = NM;
    CK(cuptiProfilerHostConfigAddMetrics(&am));
    P(CUpti_Profiler_Host_GetConfigImageSize_Params, cs); cs.pHostObject = host;
    CK(cuptiProfilerHostGetConfigImageSize(&cs));
    uint8_t *cfg = (uint8_t *)malloc(cs.configImageSize);
    P(CUpti_Profiler_Host_GetConfigImage_Params, gi);
    gi.pHostObject = host; gi.configImageSize = cs.configImageSize; gi.pConfigImage = cfg;
    CK(cuptiProfilerHostGetConfigImage(&gi));

    P(CUpti_PmSampling_Enable_Params, en); en.deviceIndex = dev; CK(cuptiPmSamplingEnable(&en));
    CUpti_PmSampling_Object *obj = en.pPmSamplingObject;
    P(CUpti_PmSampling_SetConfig_Params, sc);
    sc.pPmSamplingObject = obj; sc.configSize = cs.configImageSize; sc.pConfig = cfg;
    sc.hardwareBufferSize = 512ull << 20;
    sc.samplingInterval = interval_ms * 1000000ull;
    sc.triggerMode = CUPTI_PM_SAMPLING_TRIGGER_MODE_GPU_TIME_INTERVAL;
    sc.hwBufferAppendMode = CUPTI_PM_SAMPLING_HARDWARE_BUFFER_APPEND_MODE_KEEP_OLDEST;
    CK(cuptiPmSamplingSetConfig(&sc));

    /* two counter data images used in turn: when one is full, decoding continues
       into the other at once, and the full one is reset for the next turn */
    const uint32_t MAXS = 65536;
    P(CUpti_PmSampling_GetCounterDataSize_Params, ds);
    ds.pPmSamplingObject = obj; ds.pMetricNames = METRICS; ds.numMetrics = NM; ds.maxSamples = MAXS;
    CK(cuptiPmSamplingGetCounterDataSize(&ds));
    uint8_t *img[2];
    for (int k = 0; k < 2; k++) {
        img[k] = (uint8_t *)malloc(ds.counterDataSize);
        P(CUpti_PmSampling_CounterDataImage_Initialize_Params, ci);
        ci.pPmSamplingObject = obj; ci.counterDataSize = ds.counterDataSize; ci.pCounterData = img[k];
        CK(cuptiPmSamplingCounterDataImageInitialize(&ci));
    }
    int cur = 0;

    printf("V 2\nM"); for (int k = 0; k < NM; k++) printf(" %s", METRICS[k]); printf("\n");
    clock_pair(); fflush(stdout);

    { P(CUpti_PmSampling_Start_Params, st); st.pPmSamplingObject = obj; CK(cuptiPmSamplingStart(&st)); }
    unsigned long long t0 = ns(CLOCK_MONOTONIC), last_clock = t0;

    size_t done = 0; uint64_t global = 0; double vals[NM];
    for (int last = 0; !last; ) {
        if (g_stop) last = 1;
        if (seconds > 0 && (ns(CLOCK_MONOTONIC) - t0) / 1e9 >= seconds) last = 1;
        if (last) { P(CUpti_PmSampling_Stop_Params, sp); sp.pPmSamplingObject = obj; CK(cuptiPmSamplingStop(&sp)); }
        else usleep(100000);
        if (ns(CLOCK_MONOTONIC) - last_clock >= 1000000000ull) { clock_pair(); last_clock = ns(CLOCK_MONOTONIC); }

        for (;;) {
            uint8_t *cd = img[cur];
            P(CUpti_PmSampling_DecodeData_Params, dd);
            dd.pPmSamplingObject = obj; dd.pCounterDataImage = cd; dd.counterDataImageSize = ds.counterDataSize;
            CK(cuptiPmSamplingDecodeData(&dd));
            if (dd.overflow) fprintf(stderr, "pravaha-reader: hardware buffer overflow\n");

            P(CUpti_PmSampling_GetCounterDataInfo_Params, inf);
            inf.pCounterDataImage = cd; inf.counterDataImageSize = ds.counterDataSize;
            CK(cuptiPmSamplingGetCounterDataInfo(&inf));
            for (; done < inf.numCompletedSamples; done++, global++) {
                P(CUpti_PmSampling_CounterData_GetSampleInfo_Params, si);
                si.pPmSamplingObject = obj; si.pCounterDataImage = cd;
                si.counterDataImageSize = ds.counterDataSize; si.sampleIndex = done;
                CK(cuptiPmSamplingCounterDataGetSampleInfo(&si));
                P(CUpti_Profiler_Host_EvaluateToGpuValues_Params, ev);
                ev.pHostObject = host; ev.pCounterDataImage = cd; ev.counterDataImageSize = ds.counterDataSize;
                ev.rangeIndex = done; ev.ppMetricNames = METRICS; ev.numMetrics = NM; ev.pMetricValues = vals;
                CK(cuptiProfilerHostEvaluateToGpuValues(&ev));
                printf("S %llu %llu %llu", (unsigned long long)global,
                       (unsigned long long)si.startTimestamp, (unsigned long long)si.endTimestamp);
                for (int k = 0; k < NM; k++) printf(" %.0f", vals[k]);
                printf("\n");
            }
            fflush(stdout);
            if (dd.decodeStopReason != CUPTI_PM_SAMPLING_DECODE_STOP_REASON_COUNTER_DATA_FULL) break;
            /* switch to the other image at once, then reset the full one */
            int full = cur; cur ^= 1; done = 0;
            P(CUpti_PmSampling_CounterDataImage_Initialize_Params, ci);
            ci.pPmSamplingObject = obj; ci.counterDataSize = ds.counterDataSize; ci.pCounterData = img[full];
            CK(cuptiPmSamplingCounterDataImageInitialize(&ci));
        }
    }
    clock_pair();
    { P(CUpti_PmSampling_Disable_Params, di); di.pPmSamplingObject = obj; cuptiPmSamplingDisable(&di); }
    { P(CUpti_Profiler_Host_Deinitialize_Params, hd); hd.pHostObject = host; cuptiProfilerHostDeinitialize(&hd); }
    { P(CUpti_Profiler_DeInitialize_Params, pd); cuptiProfilerDeInitialize(&pd); }
    printf("E ok %llu\n", (unsigned long long)global); fflush(stdout);
    free(img[0]); free(img[1]); free(cfg); free(avail);
    return 0;
}
