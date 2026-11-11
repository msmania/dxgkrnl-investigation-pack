#!/usr/bin/env python3
"""va2gpa.py PID VADDR  - translate a guest virtual address of a process to its guest-physical page.

Reads /proc/PID/pagemap (needs root) and prints the PFN, the guest physical address, the mapping
the VA lives in, and the current 8-byte value at the address (via /proc/PID/mem).
Run it inside WSL while cusync/forge is running, with the address printed by dxgtrace.so.
"""
import os, struct, sys
pid, va = int(sys.argv[1]), int(sys.argv[2], 0)
page = os.sysconf("SC_PAGE_SIZE")
with open(f"/proc/{pid}/pagemap", "rb") as f:
    f.seek((va // page) * 8)
    e = struct.unpack("<Q", f.read(8))[0]
present, swapped, exclusive, softdirty = bool(e >> 63 & 1), bool(e >> 62 & 1), bool(e >> 56 & 1), bool(e >> 55 & 1)
pfn = e & ((1 << 55) - 1)
mapping = next((l.strip() for l in open(f"/proc/{pid}/maps") if int(l.split("-")[0], 16) <= va < int(l.split()[0].split("-")[1], 16)), "not mapped")
print(f"va        0x{va:x}")
print(f"mapping   {mapping}")
print(f"pagemap   present={present} swapped={swapped} exclusive={exclusive} raw=0x{e:016x}")
if pfn:
    print(f"pfn       0x{pfn:x}")
    print(f"gpa       0x{pfn * page + (va % page):x}   (page 0x{pfn * page:x})")
else:
    print("pfn       0 (not reported: not present, or no CAP_SYS_ADMIN, or PFNMAP page)")
try:
    with open(f"/proc/{pid}/mem", "rb") as m:
        m.seek(va); print(f"value     0x{struct.unpack('<Q', m.read(8))[0]:x}")
except OSError as ex:
    print(f"value     unreadable: {ex}")
