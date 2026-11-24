"""vm.compaction_proactiveness A/B (bench/p2/compaction_ab.sh): per-request decode speed, the GPU's part of a window,
and kcompactd's CPU time during that request, per phase (A1 B1 A2 B2 A3) and configuration.

    python analysis/compaction_ab.py [run1|run2]   -> results/compaction/<run>/summary.json, prints the tables

run1: the first A/B/A/B/A (runs k_<phase>_<config>); run2: the repeat with the page cache filled before every
configuration (runs k2_<phase>_<config>, bench/p2/compaction_ab2.sh).

A request's kcompactd time is the growth of kcompactd0's CPU ticks (user + system, 100 Hz) between the samples
(results/compaction/<run>/sampler.csv, every 0.5 s) that bracket its decode; "busy" means more than 10% of the decode.
"""
import bisect
import sys
import csv
import json
import re
import statistics as st
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
RUN = sys.argv[1] if len(sys.argv) > 1 else "run1"
OUT = RES / "compaction" / RUN
PREFIX = {"run1": "k", "run2": "k2"}[RUN]
PHASES = ["A1", "B1", "A2", "B2", "A3"]
CFGS = ["t60_hostalloc_nodraft", "t60_mtp", "t75_mtp"]
HZ = 100
TIMING = re.compile(r"decode timing: (\d+) windows, avg T ([\d.]+), ([\d.]+) tokens/window, ([\d.]+) ms/window = verify "
                    r"[\d.]+ \(GPU-reach wait ([\d.]+) .*?SSD wait ([\d.]+) ms/window")


def load_sampler():
    rows = list(csv.DictReader(open(OUT / "sampler.csv")))
    return [float(r["t"]) for r in rows], rows


def at(ts, rows, t):
    """the last sample at or before t (the first one if t precedes all)"""
    i = max(bisect.bisect_right(ts, t) - 1, 0)
    return rows[i]


def delta(ts, rows, t0, t1, key):
    return float(at(ts, rows, t1)[key]) - float(at(ts, rows, t0)[key])


def run_requests(name, ts, rows):
    d = RES / name
    if not (d / "requests.jsonl").exists():
        return None
    reqs = [json.loads(l) for l in open(d / "requests.jsonl")]
    tim = [m for m in (TIMING.search(l) for l in open(d / "engine.log", errors="replace")) if m]
    meta = json.load(open(d / "run.json"))
    out = []
    for r, m in zip(reqs, tim):
        t0, t1 = r["t_first"], r["t_end"]
        kc = delta(ts, rows, t0, t1, "kcompactd_ticks") / HZ
        tpw = float(m[3])
        out.append({
            "prompt": r["prompt_id"], "repeat": r["repeat"], "tok_s": r["decode_tok_s_client"],
            "ms_per_tok": float(m[4]) / tpw, "gpu_ms_window": float(m[5]), "ssd_wait_ms_window": float(m[6]),
            "tokens_per_window": tpw, "decode_s": t1 - t0, "kcompactd_s": kc, "kcompactd_frac": kc / (t1 - t0),
            "daemon_scanned": delta(ts, rows, t0, t1, "compact_daemon_migrate_scanned"),
            "daemon_wakes": delta(ts, rows, t0, t1, "compact_daemon_wake"),
            "migrated": delta(ts, rows, t0, t1, "pgmigrate_success"),
            "stalls": delta(ts, rows, t0, t1, "compact_stall"),
            "proactiveness": sorted({at(ts, rows, t)["proactiveness"] for t in (t0, t1)}),
            "frag_score": float(at(ts, rows, t0)["frag_score_2m"]), "memfree_gib": float(at(ts, rows, t0)["memfree_gib"]),
        })
    start = datetime.fromisoformat(meta["started"]).timestamp()
    load_end = reqs[0]["t_send"] if reqs else start
    run = {"load_s": meta.get("load_s"), "kcompactd_load_s": delta(ts, rows, start, load_end, "kcompactd_ticks") / HZ,
           "kcompactd_run_s": delta(ts, rows, start, reqs[-1]["t_end"], "kcompactd_ticks") / HZ if reqs else None,
           "run_s": (reqs[-1]["t_end"] - start) if reqs else None}
    return out, run


def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    m = st.mean(xs)
    sd = st.stdev(xs) if len(xs) > 1 else 0.0
    return {"n": len(xs), "mean": m, "median": st.median(xs), "sd": sd, "cv_pct": 100 * sd / m if m else None,
            "min": min(xs), "max": max(xs)}


def main():
    ts, rows = load_sampler()
    summary = {"phases": {}, "conditions": {}}
    for ph in PHASES:
        for cfg in CFGS:
            got = run_requests(f"{PREFIX}_{ph}_{cfg}", ts, rows)
            if got is None:
                continue
            reqs, run = got
            busy = [r for r in reqs if r["kcompactd_frac"] > 0.10]
            summary["phases"][f"{ph}/{cfg}"] = {
                "phase": ph, "cfg": cfg, "run": run, "requests": reqs,
                "tok_s": stats([r["tok_s"] for r in reqs]), "gpu_ms_window": stats([r["gpu_ms_window"] for r in reqs]),
                "ms_per_tok": stats([r["ms_per_tok"] for r in reqs]),
                "kcompactd_s": sum(r["kcompactd_s"] for r in reqs), "busy_requests": len(busy),
                "gpu_ms_window_calm": stats([r["gpu_ms_window"] for r in reqs if r["kcompactd_frac"] <= 0.10]),
                "gpu_ms_window_busy": stats([r["gpu_ms_window"] for r in busy]),
            }
    for cond in ("A", "B"):
        for cfg in CFGS:
            reqs = [r for k, v in summary["phases"].items() if v["phase"][0] == cond and v["cfg"] == cfg
                    for r in v["requests"]]
            if not reqs:
                continue
            # per prompt, so that prompt mix does not count as variability: spread of each prompt's repeats
            by_prompt = {}
            for r in reqs:
                by_prompt.setdefault(r["prompt"], []).append(r["tok_s"])
            summary["conditions"][f"{cond}/{cfg}"] = {
                "tok_s": stats([r["tok_s"] for r in reqs]), "gpu_ms_window": stats([r["gpu_ms_window"] for r in reqs]),
                "busy_requests": sum(1 for r in reqs if r["kcompactd_frac"] > 0.10), "n": len(reqs),
                "kcompactd_s_requests": sum(r["kcompactd_s"] for r in reqs),
                "per_prompt_cv_pct": {p: stats(v)["cv_pct"] for p, v in by_prompt.items()},
                "per_prompt_mean": {p: st.mean(v) for p, v in by_prompt.items()},
            }
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
    f = lambda x, nd=1: "--" if x is None else f"{x:.{nd}f}"
    print(f"{'phase/config':34s} {'n':>3s} {'tok/s':>6s} {'cv%':>5s} {'min':>6s} {'max':>6s} {'GPU ms':>7s} {'max':>6s}"
          f" {'busy':>4s} {'kc s':>6s} {'kc load':>7s} {'load s':>6s}")
    for k, v in summary["phases"].items():
        t, g = v["tok_s"], v["gpu_ms_window"]
        print(f"{k:34s} {t['n']:3d} {f(t['mean'])} {f(t['cv_pct']):>5s} {f(t['min']):>6s} {f(t['max']):>6s} "
              f"{f(g['mean']):>7s} {f(g['max']):>6s} {v['busy_requests']:4d} {f(v['kcompactd_s']):>6s} "
              f"{f(v['run']['kcompactd_load_s']):>7s} {f(v['run']['load_s']):>6s}")
    print()
    for k, v in summary["conditions"].items():
        t, g = v["tok_s"], v["gpu_ms_window"]
        print(f"{k:34s} n {v['n']:3d} tok/s {f(t['mean'])} (sd {f(t['sd'], 2)}, cv {f(t['cv_pct'])}%) GPU ms/window "
              f"{f(g['mean'])} (sd {f(g['sd'], 2)}, max {f(g['max'])}) busy {v['busy_requests']} kcompactd "
              f"{f(v['kcompactd_s_requests'])} s; per-prompt cv% " +
              " ".join(f"{p[:4]} {f(c)}" for p, c in v["per_prompt_cv_pct"].items()))


if __name__ == "__main__":
    main()
