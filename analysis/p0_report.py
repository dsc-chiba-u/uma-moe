"""P0: Strata 0.1.27 against the fixed miss path, E2 protocol.  One row per run: decode tok/s (median over 3 repeats,
then the mean over the six short prompts and per prompt), prefill tok/s of the three long prompts, CPU and GPU
utilisation while decoding (short prompts), the memory the run took (MemAvailable before load minus its minimum),
and whether each short prompt's text matches the reference run's.

    python analysis/p0_report.py        -> table on stdout, results/p0_report.json
"""
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_runs as S  # noqa: E402

RES = S.RES
SHORT = ["code_en", "story_en", "explain_en", "math_en", "essay_ja", "tech_ja"]
LONG = ["long5k", "long19k", "long26k"]
RUNS = ["p0f_base_auto_nodraft", "p0f_base_auto_mtp", "p0f_base_c4096_nodraft", "p0f_base_c4096_mtp",
        "p0f_base_c4096_pcie1_nodraft", "p0f_base_c4096_pcie1_mtp", "p0f_fix_auto_nodraft", "p0f_fix_auto_mtp",
        "p0f_fix_c1024_nodraft", "p0f_fix_c1024_mtp", "p0f_fix_c24576_mtp"]
REF = "p0f_base_auto_nodraft"


def texts(run):
    out = {}
    p = RES / run / "requests.jsonl"
    if not p.exists():
        return out
    for line in open(p):
        r = json.loads(line)
        if r["prompt_id"] in SHORT:
            out[(r["prompt_id"], r["repeat"])] = r.get("text", "")
    return out


def main():
    ref = texts(REF)
    rows = []
    for run in RUNS:
        if not (RES / run / "run.json").exists():
            continue
        s = S.summarize(RES / run)
        p = s["prompts"]
        dec = {k: p[k]["decode_tok_s"] for k in SHORT if k in p}
        pre = {k: p[k]["prefill_tok_s"] for k in LONG if k in p}
        cpu = [p[k]["cpu_util"] for k in SHORT if k in p and p[k]["cpu_util"] is not None]
        gpu = [p[k]["gpu_util"] for k in SHORT if k in p and p[k]["gpu_util"] is not None]
        t = texts(run)
        same = sum(1 for key, v in t.items() if ref.get(key) == v)
        mine_consistent = all(t.get((k, 0)) == t.get((k, r)) for k in SHORT for r in (1, 2) if (k, r) in t)
        rows.append({"run": run, "decode": dec, "decode_mean": st.mean(dec.values()) if dec else None,
                     "prefill": pre, "cpu": st.mean(cpu) if cpu else None, "gpu": st.mean(gpu) if gpu else None,
                     "footprint_gib": (s["mem_avail_before"] or 0) - (s["mem_avail_min"] or 0),
                     "loaded_gib": (s["mem_avail_before"] or 0) - (s["mem_avail_loaded"] or 0),
                     "same_as_ref": same, "n_texts": len(t), "repeats_identical": mine_consistent,
                     "accept": {k: p[k]["accept_rate"] for k in SHORT if k in p}})
    f = S.fmt
    print(f"{'run':30s} {'mean':>5s} " + " ".join(f"{k[:7]:>7s}" for k in SHORT) + "  prefill 5k/19k/26k   cpu%  gpu%  mem(GiB)  same/ref")
    for r in rows:
        print(f"{r['run']:30s} {f(r['decode_mean']):>5s} " + " ".join(f"{f(r['decode'].get(k)):>7s}" for k in SHORT) +
              "  " + "/".join(f(r["prefill"].get(k), 0) for k in LONG) +
              f"  {f(r['cpu']):>5s} {f(r['gpu']):>5s}  {r['loaded_gib']:5.1f}/{r['footprint_gib']:5.1f}  {r['same_as_ref']}/{r['n_texts']}"
              + ("" if r["repeats_identical"] else " (repeats differ)"))
    json.dump(rows, open(RES / "p0_report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
