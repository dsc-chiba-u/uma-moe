"""Kernel memory-compaction activity every 0.5 s, for the vm.compaction_proactiveness A/B runs (compaction_ab.sh).

    python -I bench/p2/compaction_sampler.py OUT.csv

Columns: time; the sysctl's current value; CPU ticks (user + system) of kcompactd0 and kswapd0; compaction, migration
and reclaim counters from /proc/vmstat (cumulative); free memory; and the Normal zone's fragmentation score for 2 MB
blocks (the percentage of free memory not in blocks of order 9 or more, which proactive compaction compares with
100 - proactiveness + 10).
"""
import os
import sys
import time

VMSTAT = ["compact_daemon_wake", "compact_daemon_migrate_scanned", "compact_daemon_free_scanned", "compact_stall",
          "compact_success", "compact_fail", "compact_isolated", "pgmigrate_success", "pgmigrate_fail",
          "pgscan_kswapd", "pgsteal_kswapd", "pgscan_direct", "allocstall_normal", "allocstall_movable",
          "thp_fault_alloc", "thp_fault_fallback"]


def kthread(name):
    for pid in os.listdir("/proc"):
        if pid.isdigit():
            try:
                if open(f"/proc/{pid}/comm").read().strip() == name:
                    return pid
            except OSError:
                pass
    return None


def ticks(pid):
    s = open(f"/proc/{pid}/stat").read()
    f = s[s.rindex(")") + 2:].split()
    return int(f[11]) + int(f[12])


def main(out):
    kc, ks = kthread("kcompactd0"), kthread("kswapd0")
    with open(out, "a") as f:
        if f.tell() == 0:
            f.write("t,proactiveness,kcompactd_ticks,kswapd_ticks," + ",".join(VMSTAT) +
                    ",memfree_gib,normal_free_gib,frag_score_2m\n")
        while True:
            vm = dict(l.split() for l in open("/proc/vmstat"))
            mem = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
            free = big = 0
            for l in open("/proc/buddyinfo"):
                p = l.split()
                if p[3] == "Normal":
                    c = list(map(int, p[4:]))
                    free = sum(n << o for o, n in enumerate(c))
                    big = sum(n << o for o, n in enumerate(c) if o >= 9)
            score = 100.0 * (free - big) / free if free else 0.0
            f.write(f"{time.time():.3f},{open('/proc/sys/vm/compaction_proactiveness').read().strip()},{ticks(kc)},"
                    f"{ticks(ks)}," + ",".join(vm.get(k, "") for k in VMSTAT) +
                    f",{mem['MemFree'] / 1048576:.2f},{free * 4096 / 2**30:.2f},{score:.1f}\n")
            f.flush()
            time.sleep(0.5)


if __name__ == "__main__":
    main(sys.argv[1])
