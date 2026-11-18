/* cpu_stream.c: memory traffic from the CPU only (no GPU involved).
 *
 * Author: Deepak Panigrahy
 * Part of pravaha (GB10 GPU memory-traffic measurement). License: MIT.
 *
 * Purpose (letter, experiment N4): check whether the GPU's copy counter
 * reacts to memory traffic that only the CPU makes. A memcpy of MiB bytes
 * reads MiB and writes MiB, so each pass moves about 2 x MiB through memory.
 *
 * Build:  cc -O2 -o cpu_stream cpu_stream.c
 * Run:    ./cpu_stream [MiB=1024] [PASSES=8]
 * Prints the known bytes moved (read + write) when done.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    size_t mib = (size_t)(argc > 1 ? atoi(argv[1]) : 1024);
    int passes = argc > 2 ? atoi(argv[2]) : 8;
    size_t bytes = mib << 20;
    char *a = malloc(bytes), *b = malloc(bytes);
    if (!a || !b) { fprintf(stderr, "malloc failed\n"); return 1; }
    memset(a, 1, bytes);              /* touch every page before the passes */
    memset(b, 2, bytes);
    for (int i = 0; i < passes; i++) {
        memcpy(i % 2 ? a : b, i % 2 ? b : a, bytes);   /* alternate direction each pass */
    }
    /* use the result so the copies cannot be optimised away */
    printf("cpu_stream mib=%zu passes=%d bytes_moved=%zu check=%d\n",
           mib, passes, 2 * bytes * (size_t)passes, a[bytes / 2] + b[bytes / 3]);
    free(a);
    free(b);
    return 0;
}
