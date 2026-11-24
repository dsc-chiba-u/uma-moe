"""P1: summarize tier runs. Per run: decode tok/s per prompt (server-side), the window shape, the GPU stage split
(waitB = the GPU waiting for the SSD's share), the DRAM-tier share of the layer-windows' experts, SSD bytes and
prefetch counters, MemAvailable after load.

    python analysis/p1_summary.py RUN [RUN ...]       (or a prefix ending in *)
"""
import json
import re
import sys
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "results"


def parse(run):
    log = (RES / run / "engine.log").read_text(errors="replace").splitlines()
    rj = json.load(open(RES / run / "run.json"))
    reqs, cur = [], {}
    for l in log:
        m = re.search(r"decode timing: (\d+) windows, avg T ([\d.]+), ([\d.]+) tokens/window, ([\d.]+) ms/window", l)
        if m:
            cur.update(T=float(m[2]), tpw=float(m[3]), mspw=float(m[4]))
            m2 = re.search(r"SSD wait ([\d.]+) ms/window", l)
            cur["ssd_wait"] = float(m2[1]) if m2 else None
            m3 = re.search(r"VRAM hits ([\d.]+), PCIe [\d.]+; DRAM tier ([\d.]+), SSD ([\d.]+)", l)
            if m3:
                cur.update(vram=float(m3[1]), dram=float(m3[2]), ssd=float(m3[3]))
        m = re.search(r"GPU stages \(ms/window\):(.*)", l)
        if m:
            g = m[1]
            cur["waitB"] = sum(float(x) for x in re.findall(r"waitB ([\d.]+)", g))
            cur["gpu"] = float(re.search(r"total ([\d.]+)", g)[1])
        m = re.search(r"generated in \d+ ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)", l)
        if m:
            cur.update(tps=float(m[1]), acc=int(m[2]), drafts=int(m[3]))
        m = re.search(r"decode expert tier: (\d+) resident \+ (\d+) from the SSD \(([\d.]+)% resident\), ([\d.]+) GB read, "
                      r"waited (\d+) ms in (\d+) layer-windows(?:; prefetched (\d+), used (\d+), evicted unused (\d+), "
                      r"still loading when asked (\d+))?", l)
        if m:
            cur.update(res_pct=float(m[3]), ssd_gb=float(m[4]))
            if m[7] is not None:
                cur.update(pf=int(m[7]), pf_used=int(m[8]), pf_evicted=int(m[9]), late=int(m[10]))
            reqs.append(cur)
            cur = {}
    return rj, reqs


def main():
    runs = []
    for a in sys.argv[1:]:
        if a.endswith("*"):
            runs += sorted(p.name for p in RES.iterdir() if p.name.startswith(a[:-1]) and (p / "run.json").exists())
        else:
            runs.append(a)
    for run in runs:
        rj, reqs = parse(run)
        prompts = [s.get("prompt_id") for s in rj.get("server_stats", [])]
        mem = (rj.get("mem_avail_before_gib") or 0) - (rj.get("mem_avail_min_gib") or 0)
        print(f"{run}  (memory {mem:.1f} GiB, error {rj.get('error')})")
        for pid, r in zip(prompts, reqs):
            pf = ""
            if "pf" in r:
                pf = f" | prefetch {r['pf']} used {r['pf_used']} evicted {r['pf_evicted']} late {r['late']}"
            print(f"  {pid or '?':9s} {r.get('tps', 0):5.1f} tok/s  T {r.get('T', 0):.2f} tok/win {r.get('tpw', 0):.2f}"
                  f" ms/win {r.get('mspw', 0):6.1f} gpu {r.get('gpu', 0):6.1f} waitB {r.get('waitB', 0):5.1f}"
                  f" | VRAM {r.get('vram', 0):5.2f} DRAM {r.get('dram', 0):5.2f} SSD {r.get('ssd', 0):5.2f} /layer-win"
                  f" | resident {r.get('res_pct', 0):5.1f}% SSD {r.get('ssd_gb', 0):6.2f} GB{pf}")


if __name__ == "__main__":
    main()
