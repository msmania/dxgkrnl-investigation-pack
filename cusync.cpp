// cusync.cpp - minimal CUDA driver-API program that hammers cuStreamSynchronize.
// Same behaviour as cusync.py.  Builds with only g++ and libcuda (cuda.h optional):
//     g++ -O2 -std=c++17 -pthread cusync.cpp -o cusync -lcuda
//     # bare WSL distro, no toolkit:
//     g++ -O2 -std=c++17 -pthread cusync.cpp -o cusync -L/usr/lib/wsl/lib -lcuda
//     LD_LIBRARY_PATH=/usr/lib/wsl/lib ./cusync [--loops N] [--kernel-ms MS] [--inflight N]
//                                               [--watchdog S] [--legacy-stream] [--exit-after-stuck S]
//                                               [--sched auto|spin|yield|blocking] [--readback none|pageable|pinned]
// Defaults mirror the miner: context created with CU_CTX_SCHED_YIELD (its wait loop calls sched_yield),
// and a cuMemcpyDtoHAsync into pageable host memory before every cuStreamSynchronize.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <thread>
#include <sys/syscall.h>
#include <unistd.h>

#if __has_include(<cuda.h>)
#include <cuda.h>
#else
// Minimal driver-API declarations so the toolkit headers are not required.
extern "C" {
typedef int CUresult; typedef int CUdevice; typedef unsigned long long CUdeviceptr;
typedef struct CUctx_st* CUcontext; typedef struct CUmod_st* CUmodule; typedef struct CUfunc_st* CUfunction;
typedef struct CUstream_st* CUstream; typedef struct CUevent_st* CUevent;
CUresult cuInit(unsigned);
CUresult cuDriverGetVersion(int*);
CUresult cuDeviceGet(CUdevice*, int);
CUresult cuDeviceGetName(char*, int, CUdevice);
CUresult cuDeviceGetAttribute(int*, int, CUdevice);
CUresult cuCtxCreate_v2(CUcontext*, unsigned, CUdevice);
CUresult cuCtxDestroy_v2(CUcontext);
CUresult cuModuleLoadDataEx(CUmodule*, const void*, unsigned, int*, void**);
CUresult cuModuleGetFunction(CUfunction*, CUmodule, const char*);
CUresult cuModuleUnload(CUmodule);
CUresult cuMemAlloc_v2(CUdeviceptr*, size_t);
CUresult cuMemFree_v2(CUdeviceptr);
CUresult cuMemcpyDtoHAsync_v2(void*, CUdeviceptr, size_t, CUstream);
CUresult cuMemHostAlloc(void**, size_t, unsigned);
CUresult cuMemFreeHost(void*);
CUresult cuStreamCreate(CUstream*, unsigned);
CUresult cuStreamDestroy_v2(CUstream);
CUresult cuStreamSynchronize(CUstream);
CUresult cuEventCreate(CUevent*, unsigned);
CUresult cuEventRecord(CUevent, CUstream);
CUresult cuEventElapsedTime(float*, CUevent, CUevent);
CUresult cuEventDestroy_v2(CUevent);
CUresult cuLaunchKernel(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, CUstream, void**, void**);
CUresult cuGetErrorName(CUresult, const char**);
}
#define CU_CTX_SCHED_SPIN 1
#define CU_CTX_SCHED_YIELD 2
#define CU_CTX_SCHED_BLOCKING_SYNC 4
#define CU_STREAM_NON_BLOCKING 1
#define CU_JIT_ERROR_LOG_BUFFER 5
#define CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES 6
#define CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT 16
#endif

static const char* PTX = R"(
.version 7.0
.target sm_75
.address_size 64
.visible .entry spin(.param .u64 spin_param_out, .param .u32 spin_param_iters)
{
    .reg .pred %p<2>; .reg .b32 %r<12>; .reg .b64 %rd<6>;
    ld.param.u64 %rd1, [spin_param_out];
    ld.param.u32 %r1, [spin_param_iters];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r2, %tid.x; mov.u32 %r3, %ctaid.x; mov.u32 %r4, %ntid.x;
    mad.lo.s32 %r5, %r3, %r4, %r2;
    mov.u32 %r6, %r5; mov.u32 %r7, 0; mov.u32 %r8, 1664525; mov.u32 %r9, 1013904223;
    setp.eq.u32 %p1, %r1, 0; @%p1 bra DONE;
LOOP:
    mad.lo.s32 %r6, %r6, %r8, %r9; xor.b32 %r6, %r6, %r5;
    add.u32 %r7, %r7, 1; setp.lt.u32 %p1, %r7, %r1; @%p1 bra LOOP;
DONE:
    mul.wide.u32 %rd3, %r5, 4; add.s64 %rd4, %rd2, %rd3; st.global.u32 [%rd4], %r6; ret;
}
)";

static std::string ts() {
    using namespace std::chrono;
    auto now = system_clock::now(); time_t t = system_clock::to_time_t(now);
    int ms = duration_cast<milliseconds>(now.time_since_epoch()).count() % 1000;
    char b[32]; strftime(b, sizeof b, "%H:%M:%S", localtime(&t));
    char o[48]; snprintf(o, sizeof o, "%s.%03d", b, ms); return o;
}
static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
#define CHECK(call) do { CUresult _rc = (call); if (_rc) { const char* n = "?"; cuGetErrorName(_rc, &n); \
    printf("%s %s FAILED rc=%d %s\n", ts().c_str(), #call, _rc, n); fflush(stdout); exit(2); } } while (0)

static std::atomic<double> sync_started{0.0};
static std::atomic<long> sync_loop{0};
static std::atomic<bool> stop_flag{false};
static double stuck_since = 0;

int main(int argc, char** argv) {
    long loops = 0; double kernel_ms = 100, watchdog = 20, report = 10, exit_after_stuck = 0;
    int inflight = 1, threads = 256, blocks = 0; bool legacy = false; unsigned fixed_iters = 0;
    std::string sched = "yield", readback = "pageable";
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i]; auto next = [&]{ return (i + 1 < argc) ? argv[++i] : "0"; };
        if (a == "--loops") loops = atol(next()); else if (a == "--kernel-ms") kernel_ms = atof(next());
        else if (a == "--inflight") inflight = atoi(next()); else if (a == "--watchdog") watchdog = atof(next());
        else if (a == "--report") report = atof(next()); else if (a == "--exit-after-stuck") exit_after_stuck = atof(next());
        else if (a == "--threads") threads = atoi(next()); else if (a == "--blocks") blocks = atoi(next());
        else if (a == "--iters") fixed_iters = strtoul(next(), nullptr, 10); else if (a == "--legacy-stream") legacy = true;
        else if (a == "--sched") sched = next(); else if (a == "--readback") readback = next();
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 1; }
    }
    pid_t pid = getpid(); long tid = syscall(SYS_gettid);
    printf("%s start pid=%d tid=%ld\n", ts().c_str(), pid, tid); fflush(stdout);

    CHECK(cuInit(0));
    int ver = 0; CHECK(cuDriverGetVersion(&ver));
    CUdevice dev; CHECK(cuDeviceGet(&dev, 0));
    char name[128]; CHECK(cuDeviceGetName(name, sizeof name, dev));
    int sms = 0; CHECK(cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
    printf("%s device: %s sms=%d cuda_api=%d\n", ts().c_str(), name, sms, ver);
    unsigned flags = sched == "spin" ? CU_CTX_SCHED_SPIN : sched == "yield" ? CU_CTX_SCHED_YIELD
                   : sched == "blocking" ? CU_CTX_SCHED_BLOCKING_SYNC : 0;
    CUcontext ctx; CHECK(cuCtxCreate_v2(&ctx, flags, dev));

    char log[8192] = {0}; int opts[2] = {CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
    void* vals[2] = {log, (void*)(size_t)sizeof log};
    CUmodule mod; CUresult rc = cuModuleLoadDataEx(&mod, PTX, 2, opts, vals);
    if (rc) { printf("%s cuModuleLoadDataEx FAILED rc=%d\nJIT log:\n%s\n", ts().c_str(), rc, log); return 2; }
    CUfunction fn; CHECK(cuModuleGetFunction(&fn, mod, "spin"));

    if (!blocks) blocks = sms * 8;
    size_t nbytes = (size_t)blocks * threads * 4;
    CUdeviceptr dptr; CHECK(cuMemAlloc_v2(&dptr, nbytes));
    CUstream stream = nullptr; if (!legacy) CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    CUevent ev0, ev1; CHECK(cuEventCreate(&ev0, 0)); CHECK(cuEventCreate(&ev1, 0));
    void* rb_host = nullptr;
    if (readback == "pinned") CHECK(cuMemHostAlloc(&rb_host, nbytes, 0));
    else if (readback == "pageable") rb_host = malloc(nbytes);
    printf("%s stream=%p sched=%s readback=%s grid=%dx%d inflight=%d\n", ts().c_str(), (void*)stream,
           sched.c_str(), readback.c_str(), blocks, threads, inflight); fflush(stdout);

    unsigned iters = fixed_iters ? fixed_iters : (1u << 16);
    void* params[2] = {&dptr, &iters};

    std::thread([&] {
        while (!stop_flag) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            double t0 = sync_started; if (t0 == 0) continue;
            double waited = now_s() - t0; if (waited < watchdog) continue;
            if (!stuck_since) {
                stuck_since = now_s();
                printf("%s STUCK cuStreamSynchronize has not returned for %.0fs (loop=%ld pid=%d tid=%ld inflight=%d); "
                       "attach: gdb -p %d / cat /proc/%d/task/%ld/{stat,wchan,syscall}\n",
                       ts().c_str(), waited, sync_loop.load(), pid, tid, inflight, pid, pid, tid);
            } else if ((long)waited % (long)(report < 1 ? 1 : report) == 0)
                printf("%s STUCK still waiting %.0fs (pid=%d tid=%ld)\n", ts().c_str(), waited, pid, tid);
            fflush(stdout);
            if (exit_after_stuck && now_s() - stuck_since >= exit_after_stuck) {
                printf("%s STUCK exit-after-stuck reached, _exit(3)\n", ts().c_str()); fflush(stdout); _exit(3);
            }
        }
    }).detach();
    signal(SIGINT, [](int) { stop_flag = true; }); signal(SIGTERM, [](int) { stop_flag = true; });

    auto round = [&](int n, double& wall_ms, float& gpu_ms) {
        CHECK(cuEventRecord(ev0, stream));
        for (int i = 0; i < n; i++) CHECK(cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, stream, params, nullptr));
        if (rb_host) CHECK(cuMemcpyDtoHAsync_v2(rb_host, dptr, nbytes, stream));
        CHECK(cuEventRecord(ev1, stream));
        double t0 = now_s(); sync_started = t0;
        CUresult r = cuStreamSynchronize(stream);
        wall_ms = (now_s() - t0) * 1000; sync_started = 0;
        if (stuck_since) { printf("%s RECOVERED cuStreamSynchronize returned rc=%d after %.1fs\n", ts().c_str(), r, wall_ms / 1000); stuck_since = 0; }
        CHECK(r);
        CHECK(cuEventElapsedTime(&gpu_ms, ev0, ev1));
    };

    if (!fixed_iters) {
        for (int i = 0; i < 6; i++) {
            double w; float g; round(1, w, g); if (g <= 0) break;
            double nv = std::min((double)iters * kernel_ms / g, 4294967295.0);
            if (std::abs(nv - iters) < iters * 0.1) break;
            iters = (unsigned)std::max(nv, 1.0);
        }
    }
    printf("%s iters=%u (~%.0f ms/kernel) looping, watchdog=%.0fs\n", ts().c_str(), iters, kernel_ms, watchdog); fflush(stdout);

    long loop = 0; double t_rep = now_s(), sum_sync = 0, max_sync = 0, sum_gpu = 0; int n = 0;
    while (!stop_flag && (loops == 0 || loop < loops)) {
        loop++; sync_loop = loop;
        double w; float g; round(inflight, w, g);
        n++; sum_sync += w; if (w > max_sync) max_sync = w; sum_gpu += g;
        if (now_s() - t_rep >= report) {
            printf("%s loop=%ld sync avg=%.1fms max=%.1fms gpu avg=%.1fms rate=%.2f loops/s\n",
                   ts().c_str(), loop, sum_sync / n, max_sync, sum_gpu / n, n / (now_s() - t_rep)); fflush(stdout);
            t_rep = now_s(); n = 0; sum_sync = max_sync = sum_gpu = 0;
        }
    }
    stop_flag = true;
    printf("%s done loops=%ld\n", ts().c_str(), loop);
    CHECK(cuEventDestroy_v2(ev0)); CHECK(cuEventDestroy_v2(ev1));
    if (!legacy) CHECK(cuStreamDestroy_v2(stream));
    CHECK(cuMemFree_v2(dptr));
    if (readback == "pinned") CHECK(cuMemFreeHost(rb_host)); else free(rb_host);
    CHECK(cuModuleUnload(mod)); CHECK(cuCtxDestroy_v2(ctx));
    printf("%s CUSYNC_OK\n", ts().c_str());
    return 0;
}
