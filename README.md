# cuStreamSynchronize repro

Small stand-ins for the miner in `../bare-wsl-capture/freeze-probe-bare/harness.sh`.
Both do, in a loop: `cuLaunchKernel` (spin kernel, ~100 ms of GPU time) then
`cuStreamSynchronize` on a created non-blocking stream. In the captured freeze the
miner thread sat in `cuStreamSynchronize` spinning on `sched_yield` (state R); a
watchdog thread here prints a `STUCK` line with pid/tid once a single sync exceeds
`--watchdog` seconds, and keeps the process alive for gdb/strace.

## Matching the miner's wait loop
The miner's context uses CU_CTX_SCHED_YIELD, so its wait inside cuStreamSynchronize calls
sched_yield in a loop (gdb: `b sched_yield` fires constantly).  With the default flag (auto)
a one-context process gets CU_CTX_SCHED_SPIN and the wait is a pure busy-spin with no
syscalls at all, so `b sched_yield` never fires.  Both programs now default to
`--sched yield`, and also enqueue a `cuMemcpyDtoHAsync` into pageable host memory before
each sync (`--readback pageable`), which is the other place the miner's thread was seen
spinning.  Measured here, 30 loops: sched auto = 0 sched_yield calls, sched yield = ~300k.

## Files
- `cusync.py`  – python3 + libcuda.so.1 only (PTX kernel is JIT-compiled by the driver). No toolkit needed.
- `cusync.cpp` – same thing in C++. Builds with just g++ and libcuda; uses cuda.h if present.

## Run on the WSL box (bare distro, driver libs only)
```
export LD_LIBRARY_PATH=/usr/lib/wsl/lib
python3 cusync.py                                   # forever, 1 kernel/sync, 100 ms kernels
python3 cusync.py --kernel-ms 20 --inflight 8       # many short kernels per sync
python3 cusync.py --legacy-stream --sched spin      # variations
python3 cusync.py --watchdog 30 --exit-after-stuck 600   # exit 3 after 10 min stuck

g++ -O2 -std=c++17 -pthread cusync.cpp -o cusync -L/usr/lib/wsl/lib -lcuda
./cusync --kernel-ms 100 --watchdog 30
```
Both print a progress line every `--report` seconds, `STUCK ...` when a sync overruns,
`RECOVERED` if it later returns, and `CUSYNC_OK` on clean exit.

## Hooking into the existing harness
The `capture()` function in harness.sh only needs `$PID`; run cusync in the background,
`tail -F` its log for `STUCK`, and call the same dmesg / threads / gdb / strace steps.
The STUCK line already prints the `gdb -p PID` command and the `/proc/PID/task/TID` path.

## Tested
Local GTX 1660 (driver 580, CUDA 13.0): kernel JIT ok, loop runs, watchdog fires on a
long kernel. The PTX targets sm_75 so it JITs on both Turing and the RTX 5070 Ti.

## Finding the fence address libcuda polls
`dxgtrace.so` (build: `gcc -O2 -shared -fPIC -o dxgtrace.so dxgtrace.c -ldl`) is an LD_PRELOAD
shim that logs every /dev/dxg ioctl and hexdumps the output of the calls that return fence
addresses (create-sync-object 0x10, create-hwqueue 0x18, open-sync-object 0x29/0x40). Any
8-byte field that points into a mapping of /dev/dxg is printed as a candidate fence CPU VA.

    DXGTRACE_OUT=/tmp/dxg.log LD_PRELOAD=$PWD/dxgtrace.so ./cusync      # or the miner
    grep candidate /tmp/dxg.log

Confirm which candidate the spin loop reads (run while cusync is looping):

    gdb -p PID
    (gdb) rwatch *(unsigned long*)0xADDR       # hardware read watchpoint
    (gdb) c                                    # stops inside libcuda's yield loop if ADDR is polled
    (gdb) bt 3 ; x/gx 0xADDR ; info registers rax rcx rdx

Translate it to a guest-physical address for the host side (root, inside WSL):

    python3 va2gpa.py PID 0xADDR

`DXGTRACE_NRS=all` dumps every dxg ioctl (large). Codes seen so far: 0x0f legacy SubmitCommand
(miner on the 5070 Ti), 0x34/0x36 hardware-queue submit/wait (cusync on the 2060 SUPER).
