"""Page cache at engine start (bench/p2/pagecache_ab.sh): per request, the decode rate, the time to the first token,
the GPU's part of a window, and what the kernel's reclaim did during the decode (kswapd's CPU share, pages it scanned
and reclaimed per second, refaults of evicted file pages, major faults).

    python analysis/pagecache_ab.py   -> results/pagecache/summary.json, prints the table
"""
import bisect
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results/pagecache"
TIMING = re.compile(r"decode timing: (\d+) windows, avg T ([\d.]+), ([\d.]+) tokens/window, ([\d.]+) ms/window = verify "
                    r"[\d.]+ \(GPU-reach wait ([\d.]+)")
HZ = 100


def series(path):
    rows = list(csv.DictReader(open(path)))
    return [float(r["t"]) for r in rows], rows


def at(s, t):
    ts, rows = s
    return rows[max(bisect.bisect_right(ts, t) - 1, 0)]


def rate(s, t0, t1, key):
    return (float(at(s, t1)[key]) - float(at(s, t0)[key])) / (t1 - t0)


def main():
    smp, ref = series(R / "sampler.csv"), series(R / "refaults.csv")
    starts = {m[1]: (float(m[2]), float(m[3])) for m in
              re.finditer(r"(\w+): MemFree: ([\d.]+) GiB  Cached: ([\d.]+) GiB", (R / "runner.log").read_text())}
    out = []
    for c in ("E1", "F1", "E2", "F2"):
        d = ROOT / f"results/pc_{c}_t60_nodraft"
        if not (d / "requests.jsonl").exists():
            continue
        reqs = [json.loads(l) for l in open(d / "requests.jsonl")]
        tim = [m for m in (TIMING.search(l) for l in open(d / "engine.log", errors="replace")) if m]
        for r, m in zip(reqs, tim):
            t0, t1 = r["t_first"], r["t_end"]
            out.append({
                "cond": c, "prompt": r["prompt_id"], "memfree_start_gib": starts[c][0], "cached_start_gib": starts[c][1],
                "tok_s": r["decode_tok_s_client"], "ttft_s": t0 - r["t_send"], "gpu_ms_window": float(m[5]),
                "memfree_gib": float(at(smp, t0)["memfree_gib"]),
                "kswapd_pct": rate(smp, t0, t1, "kswapd_ticks") * 100 / HZ,
                "pgscan_kswapd_s": rate(smp, t0, t1, "pgscan_kswapd"), "pgsteal_kswapd_s": rate(smp, t0, t1, "pgsteal_kswapd"),
                "kcompactd_pct": rate(smp, t0, t1, "kcompactd_ticks") * 100 / HZ,
                "refault_file_s": rate(ref, t0, t1, "workingset_refault_file"), "pgmajfault_s": rate(ref, t0, t1, "pgmajfault"),
            })
    json.dump(out, open(R / "summary.json", "w"), indent=1)
    print(f"{'run':4s} {'prompt':9s} {'start: free/cached GiB':>22s} {'tok/s':>6s} {'ttft s':>6s} {'GPU ms':>7s} {'free':>5s}"
          f" {'kswapd%':>7s} {'scan/s':>8s} {'steal/s':>8s} {'refault/s':>9s} {'majflt/s':>8s} {'kcompactd%':>10s}")
    for o in out:
        print(f"{o['cond']:4s} {o['prompt']:9s} {o['memfree_start_gib']:10.1f} / {o['cached_start_gib']:5.1f}    {o['tok_s']:6.1f}"
              f" {o['ttft_s']:6.1f} {o['gpu_ms_window']:7.1f} {o['memfree_gib']:5.2f} {o['kswapd_pct']:7.0f}"
              f" {o['pgscan_kswapd_s']:8.0f} {o['pgsteal_kswapd_s']:8.0f} {o['refault_file_s']:9.0f} {o['pgmajfault_s']:8.0f}"
              f" {o['kcompactd_pct']:10.0f}")


if __name__ == "__main__":
    main()
