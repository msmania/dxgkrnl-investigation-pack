// dxgtrace.c - LD_PRELOAD shim that logs /dev/dxg ioctls and dumps the output structures of the
// calls that hand libcuda a fence address, so the monitored-fence CPU VA can be found.
//
//   gcc -O2 -shared -fPIC -o dxgtrace.so dxgtrace.c -ldl
//   DXGTRACE_OUT=/tmp/dxgtrace.log LD_PRELOAD=$PWD/dxgtrace.so ./cusync          # or ./forge
//
// Env:  DXGTRACE_OUT   log file (default: stderr)
//       DXGTRACE_DEV   device path substring to match (default "dxg")
//       DXGTRACE_NRS   comma list of ioctl nr values to hexdump, or "all" (default 0x10,0x18,0x29,0x40)
//
// dxgkrnl ioctl nr (magic 0x47):  0x10 CREATESYNCHRONIZATIONOBJECT  0x18 CREATEHWQUEUE
//   0x29 OPENSYNCOBJECT  0x40 OPENSYNCOBJECTFROMNTHANDLE2  0x0f SUBMITCOMMAND  0x34 SUBMITCOMMANDTOHWQUEUE
//   0x36 SUBMITWAITFORSYNCOBJECTSTOHWQUEUE  0x3a WAITFORSYNCHRONIZATIONOBJECTFROMCPU  0x1d DESTROYSYNCHRONIZATIONOBJECT
// Every 8-byte aligned field of a dumped structure that looks like a user pointer into a mapping of
// the device file is reported as "candidate fence CPU VA". Confirm with gdb: rwatch *(unsigned long*)ADDR
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

static int (*real_ioctl)(int, unsigned long, ...);
static int (*real_close)(int);
static FILE *out;
static const char *devmatch = "dxg";
static int dump_all = 0;
static unsigned dump_nrs[32]; static int n_dump = 0;
static char fdpath[4096][64];   // cache fd -> path ("" unknown, "-" not our device)

static void init(void) {
    if (real_ioctl) return;
    real_ioctl = dlsym(RTLD_NEXT, "ioctl");
    real_close = dlsym(RTLD_NEXT, "close");
    const char *o = getenv("DXGTRACE_OUT");
    out = o ? fopen(o, "a") : NULL; if (!out) out = stderr;
    const char *d = getenv("DXGTRACE_DEV"); if (d && *d) devmatch = d;
    const char *n = getenv("DXGTRACE_NRS");
    if (!n) n = "0x10,0x18,0x29,0x40";
    if (!strcmp(n, "all")) dump_all = 1;
    else { char buf[256]; strncpy(buf, n, 255); buf[255] = 0;
        for (char *t = strtok(buf, ","); t && n_dump < 32; t = strtok(NULL, ",")) dump_nrs[n_dump++] = strtoul(t, NULL, 0); }
}
static const char *path_of(int fd) {
    if (fd < 0 || fd >= 4096) return "-";
    if (!fdpath[fd][0]) {
        char p[64], l[256]; snprintf(p, sizeof p, "/proc/self/fd/%d", fd);
        ssize_t k = readlink(p, l, sizeof l - 1);
        if (k > 0) { l[k] = 0; snprintf(fdpath[fd], 64, "%s", strstr(l, devmatch) ? l : "-"); } else strcpy(fdpath[fd], "-");
    }
    return fdpath[fd];
}
static void stamp(void) {
    struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts); struct tm tm; localtime_r(&ts.tv_sec, &tm);
    fprintf(out, "%02d:%02d:%02d.%06ld tid=%ld ", tm.tm_hour, tm.tm_min, tm.tm_sec, ts.tv_nsec / 1000, syscall(SYS_gettid));
}
static int in_dev_mapping(uint64_t v, char *desc, size_t dl) {   // is v inside a mapping of the device file?
    FILE *m = fopen("/proc/self/maps", "r"); if (!m) return 0;
    char line[512]; int hit = 0;
    while (fgets(line, sizeof line, m)) {
        uint64_t lo, hi; char perm[8], path[256] = "";
        if (sscanf(line, "%lx-%lx %7s %*s %*s %*s %255s", &lo, &hi, perm, path) < 3) continue;
        if (v >= lo && v < hi) { snprintf(desc, dl, "%lx-%lx %s %s", lo, hi, perm, path); hit = strstr(path, devmatch) != NULL; break; }
    }
    fclose(m); return hit;
}
static void hexdump(const char *tag, const unsigned char *p, size_t n) {
    for (size_t i = 0; i < n; i += 16) {
        fprintf(out, "    %s +%03zx:", tag, i);
        for (size_t j = i; j < i + 16 && j < n; j++) fprintf(out, " %02x", p[j]);
        fputc('\n', out);
    }
}
int ioctl(int fd, unsigned long req, ...) {
    init();
    va_list ap; va_start(ap, req); void *arg = va_arg(ap, void *); va_end(ap);
    const char *path = path_of(fd);
    if (path[0] == '-') return real_ioctl(fd, req, arg);
    unsigned nr = _IOC_NR(req), sz = _IOC_SIZE(req), magic = _IOC_TYPE(req);
    int dump = dump_all; for (int i = 0; i < n_dump && !dump; i++) dump = (dump_nrs[i] == nr);
    unsigned char before[4096]; if (dump && arg && sz <= sizeof before) memcpy(before, arg, sz);
    int rc = real_ioctl(fd, req, arg);
    stamp();
    fprintf(out, "ioctl fd=%d %s magic=0x%x nr=0x%02x size=0x%x arg=%p -> %d\n", fd, path, magic, nr, sz, arg, rc);
    if (dump && arg && sz <= sizeof before) {
        hexdump("in ", before, sz);
        hexdump("out", (unsigned char *)arg, sz);
        for (size_t off = 0; off + 8 <= sz; off += 8) {
            uint64_t v; memcpy(&v, (unsigned char *)arg + off, 8);
            if (v < 0x10000 || v > 0x7fffffffffffULL) continue;
            char desc[400] = "not mapped";
            int hit = in_dev_mapping(v, desc, sizeof desc);
            if (hit) fprintf(out, "    candidate fence CPU VA at +%03zx: 0x%lx  (%s)  value now=0x%lx\n", off, v, desc, *(volatile uint64_t *)v);
            else if (strcmp(desc, "not mapped")) fprintf(out, "    pointer at +%03zx: 0x%lx  (%s)\n", off, v, desc);
        }
    }
    fflush(out);
    return rc;
}
int close(int fd) { init(); if (fd >= 0 && fd < 4096) fdpath[fd][0] = 0; return real_close(fd); }
