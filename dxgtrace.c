// dxgtrace.c - LD_PRELOAD shim that logs /dev/dxg ioctls and reports the monitored-fence pages
// that libcuda polls, with the guest-physical address of each.
//
//   gcc -O2 -shared -fPIC -pthread -o dxgtrace.so dxgtrace.c -ldl
//   DXGTRACE_OUT=/tmp/dxg.log LD_PRELOAD=$PWD/dxgtrace.so ./cusync          # or ./forge
//   grep FENCE /tmp/dxg.log ; cat /tmp/dxg.log.maps
//
// Env:  DXGTRACE_OUT      log file (default: stderr). Full /proc/self/maps goes to <OUT>.maps
//       DXGTRACE_DEV      device path substring to match (default "dxg")
//       DXGTRACE_NRS      comma list of ioctl nr values to hexdump, or "all" (default 0x10,0x18,0x29,0x40)
//       DXGTRACE_POLL_MS  if set, a thread samples every known fence at this interval and logs
//                         "FENCEVAL" when a value changed; after a freeze the last line is the stuck value
//       DXGTRACE_QUIET    if set, do not log one line per ioctl (FENCE/FENCEVAL/dumps only)
//
// How the fence is mapped (drivers/hv/dxgkrnl/dxgvmbus.c, dxg_map_iospace): the host returns a
// guest-physical address; the guest driver does vm_mmap(MAP_SHARED|MAP_ANONYMOUS) and
// io_remap_pfn_range()s that page into it, with vm_pgoff = GPA >> PAGE_SHIFT. So in /proc/PID/maps
// the fence page is a one-page "rw-s ... /dev/zero (deleted)" line, NOT a /dev/dxg mapping, and
// its offset column is the guest-physical address of the page.
//
// Structure offsets used (include/uapi/misc/d3dkmthk.h):
//   0x10 CREATESYNCHRONIZATIONOBJECT2 (size 0x60): +0x00 device  +0x08 type (5 monitored, 6 periodic)
//        monitored: +0x10 initial value  +0x18 fence CPU VA  +0x20 fence GPU VA   +0x58 sync object handle
//        periodic:  +0x20 fence CPU VA   +0x28 fence GPU VA
//   0x18 CREATEHWQUEUE (size 0x30): +0x00 context  +0x18 queue  +0x1c progress fence handle
//        +0x20 progress fence CPU VA  +0x28 progress fence GPU VA
// Other nr: 0x0f SUBMITCOMMAND  0x34 SUBMITCOMMANDTOHWQUEUE  0x36 SUBMITWAITFORSYNCOBJECTSTOHWQUEUE
//   0x3a WAITFORSYNCHRONIZATIONOBJECTFROMCPU  0x31-0x33 SIGNALSYNCHRONIZATIONOBJECTFROM{CPU,GPU,GPU2}
//   0x1d DESTROYSYNCHRONIZATIONOBJECT  0x29 OPENSYNCOBJECT...  0x40 OPENSYNCOBJECTFROMNTHANDLE2
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
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
static char maps_path[512];
static const char *devmatch = "dxg";
static int dump_all = 0, quiet = 0, poll_ms = 0;
static unsigned dump_nrs[32]; static int n_dump = 0;
static char fdpath[4096][64];   // cache fd -> path ("" unknown, "-" not our device)

#define MAXF 64
static struct fence { uint64_t va, gpu_va, gpa, last; unsigned handle, owner; const char *kind; } fences[MAXF];
static int n_fences = 0;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

static void stamp(void) {
    struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts); struct tm tm; localtime_r(&ts.tv_sec, &tm);
    fprintf(out, "%02d:%02d:%02d.%06ld pid=%d tid=%ld ", tm.tm_hour, tm.tm_min, tm.tm_sec, ts.tv_nsec / 1000,
            getpid(), syscall(SYS_gettid));
}
// Look v up in /proc/self/maps. Returns 0 if unmapped. *shared: mapping is MAP_SHARED.
// *off: file offset of the page containing v (for a dxg_map_iospace mapping this is its GPA).
static int lookup(uint64_t v, char *desc, size_t dl, int *shared, int *isdev, uint64_t *off) {
    FILE *m = fopen("/proc/self/maps", "r"); if (!m) return 0;
    char line[512]; int hit = 0;
    while (fgets(line, sizeof line, m)) {
        uint64_t lo, hi, o; char perm[8], path[256] = "";
        if (sscanf(line, "%lx-%lx %7s %lx %*s %*s %255[^\n]", &lo, &hi, perm, &o, path) < 4) continue;
        if (v >= lo && v < hi) {
            snprintf(desc, dl, "%lx-%lx %s %08lx %s", lo, hi, perm, o, path);
            *shared = perm[3] == 's'; *isdev = strstr(path, devmatch) != NULL;
            *off = o + ((v - lo) & ~0xfffULL); hit = 1; break;
        }
    }
    fclose(m); return hit;
}
static void save_maps(void) {              // full, unfiltered copy of /proc/self/maps
    if (!maps_path[0]) return;
    FILE *m = fopen("/proc/self/maps", "r"), *o = fopen(maps_path, "w"); char line[512];
    if (m && o) while (fgets(line, sizeof line, m)) fputs(line, o);
    if (m) fclose(m);
    if (o) fclose(o);
}
static void *poller(void *arg) {
    (void)arg;
    for (;;) {
        usleep(poll_ms * 1000);
        pthread_mutex_lock(&lock);
        for (int i = 0; i < n_fences; i++) {
            uint64_t now = *(volatile uint64_t *)fences[i].va;
            if (now == fences[i].last) continue;
            stamp(); fprintf(out, "FENCEVAL %s handle=0x%x cpu_va=0x%lx value=0x%lx (was 0x%lx)\n",
                             fences[i].kind, fences[i].handle, fences[i].va, now, fences[i].last);
            fences[i].last = now;
        }
        fflush(out);
        pthread_mutex_unlock(&lock);
    }
    return NULL;
}
static void init(void) {
    if (real_ioctl) return;
    real_close = dlsym(RTLD_NEXT, "close");
    const char *o = getenv("DXGTRACE_OUT");
    out = o ? fopen(o, "a") : NULL;
    if (out) snprintf(maps_path, sizeof maps_path, "%s.maps", o); else out = stderr;
    const char *d = getenv("DXGTRACE_DEV"); if (d && *d) devmatch = d;
    quiet = getenv("DXGTRACE_QUIET") != NULL;
    const char *p = getenv("DXGTRACE_POLL_MS"); if (p) poll_ms = atoi(p);
    const char *n = getenv("DXGTRACE_NRS");
    if (!n) n = "0x10,0x18,0x29,0x40";
    if (!strcmp(n, "all")) dump_all = 1;
    else { char buf[256]; strncpy(buf, n, 255); buf[255] = 0;
        for (char *t = strtok(buf, ","); t && n_dump < 32; t = strtok(NULL, ",")) dump_nrs[n_dump++] = strtoul(t, NULL, 0); }
    real_ioctl = dlsym(RTLD_NEXT, "ioctl");
    if (poll_ms > 0) { pthread_t t; if (!pthread_create(&t, NULL, poller, NULL)) pthread_detach(t); }
}
static const char *path_of(int fd) {
    if (fd < 0 || fd >= 4096) return "-";
    if (!fdpath[fd][0]) {
        char p[64], l[256]; snprintf(p, sizeof p, "/proc/self/fd/%d", fd);
        ssize_t k = readlink(p, l, sizeof l - 1);
        if (k > 0) { l[k] = 0; snprintf(fdpath[fd], 64, "%.63s", strstr(l, devmatch) ? l : "-"); } else strcpy(fdpath[fd], "-");
    }
    return fdpath[fd];
}
static void hexdump(const char *tag, const unsigned char *p, size_t n) {
    for (size_t i = 0; i < n; i += 16) {
        fprintf(out, "    %s +%03zx:", tag, i);
        for (size_t j = i; j < i + 16 && j < n; j++) fprintf(out, " %02x", p[j]);
        fputc('\n', out);
    }
}
static uint64_t q(const unsigned char *a, size_t off) { uint64_t v; memcpy(&v, a + off, 8); return v; }
static unsigned d(const unsigned char *a, size_t off) { unsigned v; memcpy(&v, a + off, 4); return v; }

// Record and print one fence. Caller holds lock.
static void report_fence(const char *kind, unsigned handle, unsigned owner, uint64_t va, uint64_t gpu_va, uint64_t initial) {
    char desc[400] = "not mapped"; int shared = 0, isdev = 0; uint64_t gpa = 0;
    if (!va || !lookup(va, desc, sizeof desc, &shared, &isdev, &gpa)) {
        fprintf(out, "    FENCE %s handle=0x%x cpu_va=0x%lx NOT MAPPED\n", kind, handle, va); return;
    }
    uint64_t now = *(volatile uint64_t *)va;
    fprintf(out, "    FENCE %s handle=0x%x cpu_va=0x%lx gpu_va=0x%lx gpa=0x%lx initial=0x%lx value_now=0x%lx\n"
                 "          maps: %s\n", kind, handle, va, gpu_va, gpa + (va & 0xfff), initial, now, desc);
    for (int i = 0; i < n_fences; i++) if (fences[i].va == va) return;
    if (n_fences < MAXF) fences[n_fences++] = (struct fence){va, gpu_va, gpa + (va & 0xfff), now, handle, owner, kind};
}

int ioctl(int fd, unsigned long req, ...) {
    init();
    va_list ap; va_start(ap, req); void *arg = va_arg(ap, void *); va_end(ap);
    if (!real_ioctl) return syscall(SYS_ioctl, fd, req, arg);     // re-entered during init
    const char *path = path_of(fd);
    if (path[0] == '-') return real_ioctl(fd, req, arg);
    unsigned nr = _IOC_NR(req), sz = _IOC_SIZE(req), magic = _IOC_TYPE(req);
    int dump = dump_all; for (int i = 0; i < n_dump && !dump; i++) dump = (dump_nrs[i] == nr);
    unsigned char before[4096]; int have = dump && arg && sz <= sizeof before;
    if (have) memcpy(before, arg, sz);
    // The driver unmaps a fence page when its object goes away; stop sampling it first.
    // 0x1d destroy sync object, 0x1b destroy hwqueue, 0x05/0x19 destroy context/device (drop all).
    if (magic == 0x47 && arg && (nr == 0x1d || nr == 0x1b || nr == 0x05 || nr == 0x19)) {
        unsigned h; memcpy(&h, arg, 4);
        pthread_mutex_lock(&lock);
        for (int i = 0; i < n_fences; )
            if (nr == 0x05 || nr == 0x19 || (nr == 0x1d && fences[i].handle == h) || (nr == 0x1b && fences[i].owner == h)) {
                stamp(); fprintf(out, "FENCEGONE %s handle=0x%x cpu_va=0x%lx last=0x%lx (nr=0x%02x)\n",
                                 fences[i].kind, fences[i].handle, fences[i].va, fences[i].last, nr);
                fences[i] = fences[--n_fences];
            } else i++;
        fflush(out);
        pthread_mutex_unlock(&lock);
    }
    int rc = real_ioctl(fd, req, arg);
#ifdef DXGTRACE_TEST
    rc = 0;                                   // self-test against a device that rejects the ioctl
#endif
    const unsigned char *a = arg;
    int is_sync = magic == 0x47 && nr == 0x10 && arg && sz >= 0x60 && (d(a, 8) == 5 || d(a, 8) == 6);
    int is_hwq  = magic == 0x47 && nr == 0x18 && arg && sz >= 0x30;
    if (quiet && !have && !is_sync && !is_hwq) return rc;

    pthread_mutex_lock(&lock);
    stamp();
    fprintf(out, "ioctl fd=%d %s magic=0x%x nr=0x%02x size=0x%x arg=%p -> %d\n", fd, path, magic, nr, sz, arg, rc);
    if (have) { hexdump("in ", before, sz); hexdump("out", a, sz); }
    if (rc == 0 && is_sync) {
        if (d(a, 8) == 5) report_fence("monitored", d(a, 0x58), 0, q(a, 0x18), q(a, 0x20), q(a, 0x10));
        else              report_fence("periodic",  d(a, 0x58), 0, q(a, 0x20), q(a, 0x28), 0);
        save_maps();
    } else if (rc == 0 && is_hwq) {
        fprintf(out, "    hwqueue context=0x%x queue=0x%x\n", d(a, 0), d(a, 0x18));
        report_fence("hwqueue-progress", d(a, 0x1c), d(a, 0x18), q(a, 0x20), q(a, 0x28), 0);
        save_maps();
    } else if (have) {
        // Unknown layout: flag every pointer into a shared mapping (the device, or the anonymous
        // shared pages dxg_map_iospace creates). Private mappings (heap, stack) are only listed.
        for (size_t off = 0; off + 8 <= sz; off += 8) {
            uint64_t v = q(a, off);
            if (v < 0x10000 || v > 0x7fffffffffffULL) continue;
            char desc[400]; int shared = 0, isdev = 0; uint64_t gpa = 0;
            if (!lookup(v, desc, sizeof desc, &shared, &isdev, &gpa)) continue;
            if (shared || isdev)
                fprintf(out, "    candidate fence CPU VA at +%03zx: 0x%lx gpa=0x%lx value_now=0x%lx  (%s)\n",
                        off, v, gpa + (v & 0xfff), *(volatile uint64_t *)v, desc);
            else fprintf(out, "    pointer at +%03zx: 0x%lx  (%s)\n", off, v, desc);
        }
    }
    fflush(out);
    pthread_mutex_unlock(&lock);
    return rc;
}
int close(int fd) {
    init();
    if (fd >= 0 && fd < 4096) fdpath[fd][0] = 0;
    return real_close ? real_close(fd) : (int)syscall(SYS_close, fd);
}
