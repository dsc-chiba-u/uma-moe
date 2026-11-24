# dgx-spark environment (2026-10-21T10:34:23+09:00)

## OS / kernel
Linux dgx-spark 6.17.0-1014-nvidia #14-Ubuntu SMP PREEMPT_DYNAMIC Tue Mar 17 19:01:40 UTC 2026 aarch64 aarch64 aarch64 GNU/Linux
PRETTY_NAME="Ubuntu 24.04.4 LTS"

## CPU
CPU(s):                                  20
On-line CPU(s) list:                     0-19
Model name:                              Cortex-X925
Core(s) per socket:                      10
CPU(s) scaling MHz:                      100%
CPU max MHz:                             3900.0000
Model name:                              Cortex-A725
Core(s) per socket:                      10
CPU(s) scaling MHz:                      100%
CPU max MHz:                             2808.0000
L2 cache:                                25 MiB (20 instances)
L3 cache:                                24 MiB (2 instances)
NUMA node0 CPU(s):                       0-19

## Memory
MemTotal:       127601168 kB
Hugepagesize:       2048 kB
THP: always [madvise] never

## GPU
name, driver_version, compute_cap
NVIDIA GB10, 580.126.09, 12.1
Cuda compilation tools, release 13.0, V13.0.88
Build cuda_13.0.r13.0/compiler.36424714_0

## NVMe
NAME    MODEL                       SIZE
nvme0n1 SAMSUNG MZALC4T0HBL1-00B07  3.7T
max_hw_sectors_kb nr_requests: 128 1023
sudo:no

## Software
Strata: a790805 Tue Sep 29 22:58:59 2026 +0200 Engine 0.1.27: CJK answers faster (the dr (branch spark-aarch64, local aarch64 patch: env/strata-aarch64.patch)
llama.cpp (Strata-bundled): 3cf0325
llama.cpp master: 339f324 Tue Oct 20 17:19:08 2026 -0700

## Addendum (2026-11-10, revision after review)

- Page size at measurement time: 4 KB (`getconf PAGESIZE` = 4096 today; the final runs' engine logs record
  "using 4 KB pages" when the DRAM tier was registered, which is what `paper/check/claims/characterization.py` reads).
- The machine was updated after the measurements: on 2026-11-10 it runs kernel 7.0.0-1024-nvidia and driver
  580.178.04 (the measurements above: 6.17.0-1014-nvidia, 580.126.09). No measurement of the paper was made on the
  updated system.
