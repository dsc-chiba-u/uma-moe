"""Summarize harness runs: per run and prompt, the median over repeats of server-side decode tok/s, prefill tok/s,
draft acceptance, cache hit rate, plus CPU/GPU utilisation and NVMe reads averaged over the decode phase.

    python analysis/summarize_runs.py PREFIX [PREFIX ...]   -> prints a table, writes results/summary_<PREFIX>.json
"""
import csv
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stats_pairing as SP  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def load_samples(path):
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                rows.append({k: float(v) if v not in ("", "[N/A]") else None for k, v in r.items()})
            except ValueError:
                pass
    return rows


def window_mean(samples, a, b, key):
    v = [s[key] for s in samples if a <= s["t"] <= b and s.get(key) is not None]
    return st.mean(v) if v else None


def decode_rate(run, s):
    """The server's own decode tokens/s for one request (Section 7.1 of the paper).

    Strata prints the rate itself (`N generated in M ms (X tok/s)`): `decode_tok_s` is that X. llama.cpp's `eval time`
    line prints `(N-1) / time`: its first output token comes out of the prompt pass, so the server divides the tokens
    generated after it by the generation time (tools/server/server-common.h, n_gen_tps, commit 339f324). Runs recorded
    before 2026-11-22 stored 256 / time in `decode_tok_s` (bench/harness.py then divided the full count); the raw
    run.json files are kept as recorded and the server's definition is applied here, from the recorded `gen_tokens` and
    `gen_ms` (notes/revision-2026-11-22.md, S-4)."""
    if run.get("engine") == "llama" and s.get("gen_tokens") and s.get("gen_ms"):
        return (s["gen_tokens"] - 1) / s["gen_ms"] * 1000.0
    return s.get("decode_tok_s")


def summarize(run_dir):
    run = json.load(open(run_dir / "run.json"))
    reqs = [json.loads(l) for l in open(run_dir / "requests.jsonl")] if (run_dir / "requests.jsonl").exists() else []
    return summarize_records(run, reqs, load_samples(run_dir / "samples.csv"))


def summarize_records(run, reqs, samples):
    """Per-prompt medians of a run from its run.json dict, requests.jsonl rows and samples.

    The server's statistics rows are paired with the requests by stats_pairing.pair_server_stats (the rows belong to
    the requests that finished, in order; a failed request has none), not by position in requests.jsonl and not by the
    prompt_id/repeat labels stored in run.json, which the harness wrote by position before 2026-11-22. When the pairing
    is undetermined, the server-side values (decode rate, prefill, acceptance, hit rate) are reported as missing
    (`server_stats_pairing` says so); the client-side values and the samples are still summarized."""
    stats = run.get("server_stats", [])
    idx, pairing = SP.pair_server_stats(run.get("engine"), reqs, stats)
    stat_of = {i: s for s, i in zip(stats, idx) if i is not None}
    by = {}
    for i, r in enumerate(reqs):
        s = stat_of.get(i, {})
        d = by.setdefault(r["prompt_id"], {"decode": [], "prefill": [], "acc": [], "drafts": [], "gen": [], "hit": [],
                                            "cpu": [], "gpu": [], "nvme": [], "client_decode": []})
        if decode_rate(run, s):
            d["decode"].append(decode_rate(run, s))
        if s.get("prefill_ms") and s.get("prompt_read", 0) > 100:
            d["prefill"].append(s["prompt_read"] / s["prefill_ms"] * 1000)
        if s.get("drafts"):
            d["acc"].append(s["drafts_accepted"] / s["drafts"]); d["drafts"].append((s["drafts_accepted"], s["drafts"], s["gen_tokens"]))
        if s.get("hit_rate") is not None:
            d["hit"].append(s["hit_rate"])
        if r.get("decode_tok_s_client"):
            d["client_decode"].append(r["decode_tok_s_client"])
        if r.get("t_first"):
            d["cpu"].append(window_mean(samples, r["t_first"], r["t_end"], "cpu_all"))
            d["gpu"].append(window_mean(samples, r["t_first"], r["t_end"], "gpu_util"))
            d["nvme"].append(window_mean(samples, r["t_first"], r["t_end"], "nvme_read_mbs"))
    med = lambda v: st.median([x for x in v if x is not None]) if [x for x in v if x is not None] else None
    out = {"name": run["name"], "error": run.get("error"), "server_stats_pairing": pairing,
           "mem_avail_before": run.get("mem_avail_before_gib"),
           "mem_avail_loaded": run.get("mem_avail_loaded_gib"), "mem_avail_min": run.get("mem_avail_min_gib"),
           "load_s": run.get("load_s"), "prompts": {}}
    for pid, d in by.items():
        steps = None
        if d["drafts"]:
            # tokens per verify step: generated / (generated - accepted)
            steps = med([g / max(1, g - a) for a, _, g in d["drafts"]])
        out["prompts"][pid] = {"decode_tok_s": med(d["decode"]), "decode_min": min(d["decode"]) if d["decode"] else None,
                               "decode_max": max(d["decode"]) if d["decode"] else None, "prefill_tok_s": med(d["prefill"]),
                               "accept_rate": med(d["acc"]), "tokens_per_step": steps, "hit_rate": med(d["hit"]),
                               "cpu_util": med(d["cpu"]), "gpu_util": med(d["gpu"]), "nvme_mbs": med(d["nvme"]),
                               "client_decode": med(d["client_decode"]), "n": len(d["decode"])}
    return out


def fmt(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def main():
    prefixes = sys.argv[1:] or ["e2_"]
    for pre in prefixes:
        runs = sorted(p for p in RES.iterdir() if p.is_dir() and p.name.startswith(pre) and (p / "run.json").exists())
        allout = []
        for rd in runs:
            s = summarize(rd); allout.append(s)
            mem = (s["mem_avail_before"] or 0) - (s["mem_avail_min"] or 0)
            print(f"\n{s['name']}  load {fmt(s['load_s'], 0)} s  footprint ~{mem:.1f} GiB  {('ERROR ' + s['error']) if s['error'] else ''}"
                  f"{'  SERVER STATISTICS NOT PAIRED WITH THE REQUESTS (reported as missing)' if s['server_stats_pairing'] == 'undetermined' else ''}")
            for pid, p in s["prompts"].items():
                print(f"  {pid:11s} decode {fmt(p['decode_tok_s']):>6s} [{fmt(p['decode_min'])}-{fmt(p['decode_max'])}]"
                      f"  prefill {fmt(p['prefill_tok_s'], 0):>5s}  acc {fmt(p['accept_rate'], 2):>4s}  tok/step {fmt(p['tokens_per_step'], 2):>4s}"
                      f"  hit {fmt(p['hit_rate'])}  cpu {fmt(p['cpu_util'])}%  gpu {fmt(p['gpu_util'])}%  nvme {fmt(p['nvme_mbs'], 0)}MB/s")
        json.dump(allout, open(RES / f"summary_{pre.rstrip('_')}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
