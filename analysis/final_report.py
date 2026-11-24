"""P2 final comparison on the 188 GB Q8_0: one row per run (results/f_*): decode tok/s per prompt (median of repeats)
and mean, CPU/GPU utilization while decoding, memory taken (MemAvailable before load minus its minimum), NVMe GB
read per request (sampler), output identity across our runs, window breakdown (decode timing lines).

    python analysis/final_report.py            -> stdout table, results/final_report.json
"""
import csv
import json
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_runs as S  # noqa: E402

RES = S.RES
SHORT = ["code_en", "story_en", "explain_en", "math_en", "essay_ja", "tech_ja"]
ORDER = ["f_ours_t45_nodraft", "f_ours_t45_mtp", "f_ours_t60_nodraft", "f_ours_t60_mtp", "f_ours_t75_nodraft",
         "f_ours_t75_mtp", "f_ours_t60_nopred_nodraft", "f_ours_t60_nopred_mtp", "f_ours_t60_nooverlap_nodraft",
         "f_ours_t60_nooverlap_mtp", "f_ours_t60_hostalloc_nodraft", "f_ours_t60_plemmap_mtp", "f_up_b66_exact_nodraft",
         "f_up_b66_exact_mtp", "f_up_b90_default_mtp", "f_up_b90_exact_mtp", "f_up_b66_default_mtp",
         "f_llama_cpumoeall_nomtp", "f_llama_cpumoeall_mtp", "f_llama_cpumoe20_nomtp",
         # Strata 0.2.6 runs (2026-10-26, bench/p2/upstream026_runs.sh, upstream026_unbuf_runs.sh)
         "f_up026_b66_exact_nodraft", "f_up026_b66_exact_mtp", "f_up026_b90_default_mtp", "f_up026_b90_exact_mtp",
         "f_up026_b66_default_mtp", "f_up026_b66_exact_unbuf_nodraft", "f_up026_b66_exact_unbuf_mtp"]


def texts(run):
    out = {}
    p = RES / run / "requests.jsonl"
    if p.exists():
        for line in open(p):
            r = json.loads(line)
            out[(r["prompt_id"], r["repeat"])] = r.get("text", "")
    return out


def nvme_gb(run):
    """NVMe GB read per request (sampler MB/s integrated over the request), split at the first token:
    {"total", "prompt", "decode"}. The engine's own counter (tier()) covers decoding only."""
    p = RES / run / "samples.csv"
    reqs = [json.loads(l) for l in open(RES / run / "requests.jsonl")] if (RES / run / "requests.jsonl").exists() else []
    if not p.exists() or not reqs:
        return None
    rows = [r for r in csv.DictReader(open(p))]
    t = [float(r["t"]) for r in rows]
    mbs = [float(r["nvme_read_mbs"]) if r["nvme_read_mbs"] not in ("", "[N/A]") else 0.0 for r in rows]

    def integ(a, b):
        return sum(mbs[i] * (t[i] - t[i - 1]) / 1000.0 for i in range(1, len(t)) if a <= t[i] <= b)

    pr = [integ(r["t_send"], r["t_first"]) for r in reqs]
    de = [integ(r["t_first"], r["t_end"]) for r in reqs]
    rate = [d / (r["t_end"] - r["t_first"]) for d, r in zip(de, reqs) if r["t_end"] > r["t_first"]]
    return {"total": st.mean(a + b for a, b in zip(pr, de)), "prompt": st.mean(pr), "decode": st.mean(de),
            "decode_gbs": st.mean(rate) if rate else None}


Q8_BLOB = 5_222_400   # bytes of one Q8_0 expert blob in experts.bin (tools/iq_pack.py; Section 6 of the paper)


def tier(run):
    """Mean per-request DRAM-tier counters while decoding (strata-p1d's 'decode expert tier' lines).

    The engine's line counts, per layer-window, the distinct experts computed from a slot that was ready when the
    layer's plan was published (`resident`) and those still arriving then (`ssd`); `hit` is their ratio. With
    STRATA_TIER_NO_OVERLAP=1 the host waits for the reads before publishing, so every expert counts as resident there.
    Three derived fields describe the state when the layer asked for the experts (TieredExpertSource::begin_layer):
    `demand` = experts read on demand per request (not in any slot when asked: `ssd` minus the prefetched ones found
    still loading, `late`; with overlap off, the bytes read over the blob size, since nothing is prefetched),
    `inflight` = `late`, and `ready` = the rest, as a share of the lookups (`ready_pct`)."""
    p = RES / run / "engine.log"
    if not p.exists():
        return None
    acc = []
    for l in open(p, errors="replace"):
        m = re.search(r"decode expert tier: (\d+) resident \+ (\d+) from the SSD \(([\d.]+)% resident\), ([\d.]+) GB read, "
                      r"waited (\d+) ms in (\d+) layer-windows; prefetched (\d+), used (\d+), evicted unused (\d+), "
                      r"still loading when asked (\d+)", l)
        if m:
            acc.append(dict(resident=int(m[1]), ssd=int(m[2]), hit=float(m[3]), gb=float(m[4]), wait_ms=int(m[5]),
                            wait_layers=int(m[6]), prefetched=int(m[7]), used=int(m[8]), evicted=int(m[9]), late=int(m[10])))
    if not acc:
        return None
    no_overlap = (RES / run / "run.json").exists() and json.load(open(RES / run / "run.json")).get("env", {}).get("STRATA_TIER_NO_OVERLAP") == "1"
    for x in acc:
        look = x["resident"] + x["ssd"]
        x["demand"] = round(x["gb"] * 1e9 / Q8_BLOB) - x["prefetched"] if no_overlap else x["ssd"] - x["late"]
        x["inflight"] = x["late"]
        x["ready_pct"] = 100.0 * (look - x["demand"] - x["inflight"]) / look if look else None
        x["inflight_pct"] = 100.0 * x["inflight"] / look if look else None
        x["demand_pct"] = 100.0 * x["demand"] / look if look else None
    return {k: st.mean(x[k] for x in acc) for k in acc[0]}


def prompt_read(run):
    """Mean prompt-read time (ms) and tokens of the short prompts (Strata's 'prompt N tokens ... read in X ms')."""
    p = RES / run / "engine.log"
    if not p.exists():
        return None
    ms = [(int(m[1]), float(m[2])) for m in re.finditer(r"prompt (\d+) tokens = \d+ reused \+ \d+ read in (\d+) ms",
                                                         open(p, errors="replace").read())]
    return {"tokens": st.mean(a for a, _ in ms), "ms": st.mean(b for _, b in ms)} if ms else None


def windows(run):
    """Mean per-request window breakdown from STRATA_DECODE_TIMING lines (Strata runs)."""
    p = RES / run / "engine.log"
    if not p.exists():
        return None
    acc = []
    for l in open(p, errors="replace"):
        m = re.search(r"decode timing: (\d+) windows, avg T ([\d.]+), ([\d.]+) tokens/window, ([\d.]+) ms/window = verify "
                      r"([\d.]+) \(GPU-reach wait ([\d.]+) \+ per-layer host ([\d.]+).*?stage ([\d.]+)\) \+ commit/emit ([\d.]+) "
                      r"\+ draft ([\d.]+)", l)
        if m:
            d = dict(T=float(m[2]), tpw=float(m[3]), mspw=float(m[4]), host=float(m[7]), stage=float(m[8]),
                     commit=float(m[9]), draft=float(m[10]))
            m2 = re.search(r"SSD wait ([\d.]+) ms/window", l)
            d["ssd_wait"] = float(m2[1]) if m2 else 0.0
            acc.append(d)
    if not acc:
        return None
    return {k: st.mean(x[k] for x in acc) for k in acc[0]}


def main():
    ours = [r for r in ORDER if r.startswith("f_ours") and (RES / r / "run.json").exists()]
    ref = texts(ours[0]) if ours else {}
    rows = []
    for run in ORDER:
        if not (RES / run / "run.json").exists():
            continue
        s = S.summarize(RES / run)
        p = s["prompts"]
        dec = {k: p[k]["decode_tok_s"] for k in SHORT if k in p and p[k]["decode_tok_s"]}
        cpu = [p[k]["cpu_util"] for k in SHORT if k in p and p[k]["cpu_util"] is not None]
        gpu = [p[k]["gpu_util"] for k in SHORT if k in p and p[k]["gpu_util"] is not None]
        t = texts(run)
        same = sum(1 for k, v in t.items() if ref.get(k) == v) if run.startswith("f_ours") else None
        rows.append({"run": run, "decode": dec, "mean": st.mean(dec.values()) if dec else None,
                     "cpu": st.mean(cpu) if cpu else None, "gpu": st.mean(gpu) if gpu else None,
                     "mem_gib": (s["mem_avail_before"] or 0) - (s["mem_avail_min"] or 0), "nvme": nvme_gb(run),
                     "tier": tier(run), "prompt": prompt_read(run),
                     "same": same, "n": len(t), "win": windows(run), "error": s["error"]})
    f = S.fmt
    print(f"{'run':30s} {'mean':>5s} " + " ".join(f"{k[:7]:>7s}" for k in SHORT) + "  cpu%  gpu%  GiB  NVMe GB/req  same")
    for r in rows:
        print(f"{r['run']:30s} {f(r['mean']):>5s} " + " ".join(f"{f(r['decode'].get(k)):>7s}" for k in SHORT) +
              f"  {f(r['cpu']):>4s} {f(r['gpu']):>5s} {r['mem_gib']:5.1f}  {f((r['nvme'] or {}).get('total')):>6s}"
              f"  {'' if r['same'] is None else str(r['same']) + '/' + str(r['n'])}" + (f"  ERROR {r['error']}" if r['error'] else ""))
    json.dump(rows, open(RES / "final_report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
