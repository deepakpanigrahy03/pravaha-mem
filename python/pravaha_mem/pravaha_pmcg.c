/* pravaha-pmcg: read one SMMU PMCG "transaction" counter for pravaha, and nothing else.
 *
 * Author: Deepak Panigrahy
 * Part of pravaha-mem (GB10 GPU memory traffic measurement). License: MIT.
 *
 * Why it exists: reading a system wide perf counter needs either
 * kernel.perf_event_paranoid <= 0 or the capability CAP_PERFMON. An administrator who
 * prefers not to change the system setting can give this one small program that
 * capability, once:
 *
 *     sudo install -m 0755 pravaha-pmcg /usr/local/bin/
 *     sudo setcap cap_perfmon=ep /usr/local/bin/pravaha-pmcg
 *
 * and then every user can measure GPU copies without sudo, also from unattended scripts.
 *
 * What it can do, and why that is safe to grant:
 *   - it opens only perf units named smmuv3_pmcg_<hex>, and only their "transaction" event;
 *   - it runs no other program and takes no paths from the user;
 *   - it answers four one letter commands on standard input:
 *       s  reset and start counting        -> "ok"
 *       p  stop counting                   -> "ok"
 *       r  read the current count          -> the count as a decimal number
 *       q  quit
 *
 * Build: cc -O2 -Wall -Wextra -o pravaha-pmcg pravaha_pmcg.c
 * Usage: pravaha-pmcg smmuv3_pmcg_13002     (pravaha starts it; you rarely run it yourself)
 */

#define _GNU_SOURCE
#include <ctype.h>
#include <errno.h>
#include <linux/perf_event.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#define SYSFS "/sys/bus/event_source/devices"
#define EVENT "transaction"

static int die(const char *msg) {
    fprintf(stdout, "error %s\n", msg);
    fflush(stdout);
    return 1;
}

/* Only smmuv3_pmcg_ followed by hex digits: no paths, no other perf units. */
static int unit_ok(const char *u) {
    const char *p = "smmuv3_pmcg_";
    size_t n = strlen(p);
    if (strncmp(u, p, n) != 0 || u[n] == '\0' || strlen(u) > 40) return 0;
    for (const char *c = u + n; *c; c++)
        if (!isxdigit((unsigned char)*c)) return 0;
    return 1;
}

static int read_text(const char *unit, const char *rel, char *buf, size_t len) {
    char path[256];
    snprintf(path, sizeof path, SYSFS "/%s/%s", unit, rel);
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    if (!fgets(buf, (int)len, f)) { fclose(f); return -1; }
    fclose(f);
    buf[strcspn(buf, "\n")] = '\0';
    return 0;
}

/* Put value into the bits that format/<name> names, e.g. "config:0-15". */
static int place(const char *unit, const char *name, uint64_t value, struct perf_event_attr *a) {
    char spec[64], rel[96];
    snprintf(rel, sizeof rel, "format/%s", name);
    if (read_text(unit, rel, spec, sizeof spec)) return -1;
    char *colon = strchr(spec, ':');
    if (!colon) return -1;
    *colon = '\0';
    int lo = 0, hi = 0;
    if (sscanf(colon + 1, "%d-%d", &lo, &hi) == 1) hi = lo;
    if (lo < 0 || hi > 63 || hi < lo) return -1;
    uint64_t mask = (hi - lo == 63) ? ~0ULL : (((1ULL << (hi - lo + 1)) - 1) << lo);
    uint64_t bits = (value << lo) & mask;
    if (!strcmp(spec, "config")) a->config |= bits;
    else if (!strcmp(spec, "config1")) a->config1 |= bits;
    else if (!strcmp(spec, "config2")) a->config2 |= bits;
    else return -1;
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 2 || !unit_ok(argv[1])) return die("usage: pravaha-pmcg smmuv3_pmcg_<hex>");
    const char *unit = argv[1];

    char text[128];
    struct perf_event_attr attr;
    memset(&attr, 0, sizeof attr);
    attr.size = sizeof attr;
    attr.disabled = 1;

    if (read_text(unit, "type", text, sizeof text)) return die("unit not found");
    attr.type = (uint32_t)strtoul(text, NULL, 10);

    /* events/transaction looks like "event=0x1" or "event=0x1,filter=0x0" */
    if (read_text(unit, "events/" EVENT, text, sizeof text)) return die("unit has no transaction event");
    for (char *tok = strtok(text, ","); tok; tok = strtok(NULL, ",")) {
        char *eq = strchr(tok, '=');
        uint64_t v = 1;
        if (eq) { *eq = '\0'; v = strtoull(eq + 1, NULL, 0); }
        if (place(unit, tok, v, &attr)) return die("cannot read the event format");
    }

    int cpu = 0;
    if (read_text(unit, "cpumask", text, sizeof text) == 0) cpu = atoi(text);

    long fd = syscall(SYS_perf_event_open, &attr, -1, cpu, -1, PERF_FLAG_FD_CLOEXEC);
    if (fd < 0) {
        if (errno == EACCES || errno == EPERM)
            return die("permission denied: this program needs cap_perfmon (see docs/MANUAL.md)");
        return die("perf_event_open failed");
    }

    printf("ready %s\n", unit);
    fflush(stdout);

    char line[16];
    while (fgets(line, sizeof line, stdin)) {
        uint64_t v = 0;
        switch (line[0]) {
        case 's':
            if (ioctl((int)fd, PERF_EVENT_IOC_RESET, 0) || ioctl((int)fd, PERF_EVENT_IOC_ENABLE, 0))
                return die("cannot start");
            puts("ok");
            break;
        case 'p':
            if (ioctl((int)fd, PERF_EVENT_IOC_DISABLE, 0)) return die("cannot stop");
            puts("ok");
            break;
        case 'r':
            if (read((int)fd, &v, sizeof v) != (ssize_t)sizeof v) return die("cannot read");
            printf("%llu\n", (unsigned long long)v);
            break;
        case 'q':
            close((int)fd);
            return 0;
        default:
            puts("error unknown command");
        }
        fflush(stdout);
    }
    close((int)fd);
    return 0;
}
