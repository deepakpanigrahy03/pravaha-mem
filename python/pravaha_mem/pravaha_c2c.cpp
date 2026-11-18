/* pravaha-c2c: live, device wide sampler of GPU L2 traffic over NVLink C2C.
 * Part of pravaha-mem. MIT License. Copyright (c) 2026 Deepak Panigrahy and Sirisha Kota.
 * Written from the public CUPTI headers. Runs no GPU work of its own; it only reads counters.
 *
 * Output (stdout, one line each):
 *   M <name1> ... <name5>                       metric names, once
 *   S <idx> <t_start_ns> <t_end_ns> <v1> ... <v5>  one per completed sample (sectors)
 *   E <reason>                                  on exit
 * Stops on SIGINT or SIGTERM, or after --seconds N.
 * Usage: pravaha-c2c [--interval-ms 10] [--seconds N] [--device 0]
 */
#include <cuda.h>
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
        fprintf(stderr, "pravaha-c2c: GPU counters busy: another profiler is using them " \
                "(another pravaha c2c, Nsight Compute, Nsight Systems GPU metrics or DCGM profiling). " \
                "Stop it and try again.\n"); \
        printf("E busy\n"); fflush(stdout); exit(3); } \
    if (r_ != CUPTI_SUCCESS) { \
    const char *m_ = 0; cuptiGetResultString(r_, &m_); \
    fprintf(stderr, "pravaha-c2c: %s failed: %s (%d)\n", #call, m_ ? m_ : "?", (int)r_); \
    printf("E error %d\n", (int)r_); fflush(stdout); exit(2); } } while (0)

#define P(T, v) T v; memset(&v, 0, sizeof v); v.structSize = T##_STRUCT_SIZE

int main(int argc, char **argv)
{
    uint64_t interval_ms = 10; double seconds = 0; size_t dev = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--interval-ms") && i + 1 < argc) interval_ms = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "--seconds") && i + 1 < argc) seconds = atof(argv[++i]);
        else if (!strcmp(argv[i], "--device") && i + 1 < argc) dev = strtoull(argv[++i], 0, 10);
        else { fprintf(stderr, "usage: %s [--interval-ms N] [--seconds S] [--device D]\n", argv[0]); return 1; }
    }
    signal(SIGINT, on_sig); signal(SIGTERM, on_sig);

    if (cuInit(0) != CUDA_SUCCESS) { fprintf(stderr, "pravaha-c2c: cuInit failed\n"); return 2; }
    { P(CUpti_Profiler_Initialize_Params, p); CK(cuptiProfilerInitialize(&p)); }

    /* chip name and counter availability */
    P(CUpti_Device_GetChipName_Params, cn); cn.deviceIndex = dev; CK(cuptiDeviceGetChipName(&cn));
    P(CUpti_PmSampling_GetCounterAvailability_Params, ca); ca.deviceIndex = dev;
    {
        CUptiResult r = cuptiPmSamplingGetCounterAvailability(&ca);
        if (r == CUPTI_ERROR_HARDWARE_BUSY) {
            fprintf(stderr, "pravaha-c2c: GPU counters busy: another profiler is using them "
                    "(another pravaha c2c, Nsight Compute, Nsight Systems GPU metrics or DCGM profiling). "
                    "Stop it and try again.\n");
            printf("E busy\n"); fflush(stdout); return 3;
        }
        CK(r);
    }
    uint8_t *avail = (uint8_t *)malloc(ca.counterAvailabilityImageSize);
    ca.pCounterAvailabilityImage = avail; CK(cuptiPmSamplingGetCounterAvailability(&ca));

    /* host side config for the 5 metrics */
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

    /* device side: enable, configure, allocate counter data */
    P(CUpti_PmSampling_Enable_Params, en); en.deviceIndex = dev; CK(cuptiPmSamplingEnable(&en));
    CUpti_PmSampling_Object *obj = en.pPmSamplingObject;
    P(CUpti_PmSampling_SetConfig_Params, sc);
    sc.pPmSamplingObject = obj; sc.configSize = cs.configImageSize; sc.pConfig = cfg;
    sc.hardwareBufferSize = 512ull << 20;
    sc.samplingInterval = interval_ms * 1000000ull;
    sc.triggerMode = CUPTI_PM_SAMPLING_TRIGGER_MODE_GPU_TIME_INTERVAL;
    sc.hwBufferAppendMode = CUPTI_PM_SAMPLING_HARDWARE_BUFFER_APPEND_MODE_KEEP_OLDEST;
    CK(cuptiPmSamplingSetConfig(&sc));

    const uint32_t MAXS = 65536;  /* image reset (which loses one sample) about every 16 min */
    P(CUpti_PmSampling_GetCounterDataSize_Params, ds);
    ds.pPmSamplingObject = obj; ds.pMetricNames = METRICS; ds.numMetrics = NM; ds.maxSamples = MAXS;
    CK(cuptiPmSamplingGetCounterDataSize(&ds));
    uint8_t *cd = (uint8_t *)malloc(ds.counterDataSize);
    P(CUpti_PmSampling_CounterDataImage_Initialize_Params, ci);
    ci.pPmSamplingObject = obj; ci.counterDataSize = ds.counterDataSize; ci.pCounterData = cd;
    CK(cuptiPmSamplingCounterDataImageInitialize(&ci));

    printf("M"); for (int k = 0; k < NM; k++) printf(" %s", METRICS[k]); printf("\n"); fflush(stdout);

    { P(CUpti_PmSampling_Start_Params, st); st.pPmSamplingObject = obj; CK(cuptiPmSamplingStart(&st)); }
    struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);

    size_t done = 0;      /* samples already printed from the current image */
    uint64_t global = 0;  /* running sample index across image resets */
    double vals[NM];
    for (int last = 0; !last; ) {
        if (g_stop) last = 1;
        if (seconds > 0) {
            struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
            if ((t.tv_sec - t0.tv_sec) + (t.tv_nsec - t0.tv_nsec) / 1e9 >= seconds) last = 1;
        }
        if (last) { P(CUpti_PmSampling_Stop_Params, sp); sp.pPmSamplingObject = obj; CK(cuptiPmSamplingStop(&sp)); }
        else usleep(100000);

        for (;;) {  /* drain the hardware buffer into the image, reset the image when full */
            P(CUpti_PmSampling_DecodeData_Params, dd);
            dd.pPmSamplingObject = obj; dd.pCounterDataImage = cd; dd.counterDataImageSize = ds.counterDataSize;
            CK(cuptiPmSamplingDecodeData(&dd));
            if (dd.overflow) fprintf(stderr, "pravaha-c2c: hardware buffer overflow\n");

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
            CK(cuptiPmSamplingCounterDataImageInitialize(&ci));
            done = 0;
        }
    }

    { P(CUpti_PmSampling_Disable_Params, di); di.pPmSamplingObject = obj; cuptiPmSamplingDisable(&di); }
    { P(CUpti_Profiler_Host_Deinitialize_Params, hd); hd.pHostObject = host; cuptiProfilerHostDeinitialize(&hd); }
    { P(CUpti_Profiler_DeInitialize_Params, pd); cuptiProfilerDeInitialize(&pd); }
    printf("E ok %llu\n", (unsigned long long)global); fflush(stdout);
    free(cd); free(cfg); free(avail);
    return 0;
}
