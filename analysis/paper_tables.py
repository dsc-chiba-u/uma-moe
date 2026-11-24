"""LaTeX tables for the paper from results/final_report.json (analysis/final_report.py).

    python analysis/paper_tables.py   -> paper/tables/q8_final.tex, paper/tables/q8_ablation.tex
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "tables"
SHORT = ["code_en", "story_en", "explain_en", "math_en", "essay_ja", "tech_ja"]

# (run, engine, memory configuration, MTP)
MAIN = [
    ("f_ours_t45_nodraft", "this work", "DRAM tier 45\\%", False),
    ("f_ours_t45_mtp", "this work", "DRAM tier 45\\%", True),
    ("f_ours_t60_nodraft", "this work", "DRAM tier 60\\%", False),
    ("f_ours_t60_mtp", "this work", "DRAM tier 60\\%", True),
    ("f_ours_t75_nodraft", "this work", "DRAM tier 75\\%", False),
    ("f_ours_t75_mtp", "this work", "DRAM tier 75\\%", True),
    ("f_up_b66_exact_nodraft", "Strata 0.2.2", "budget 66.8\\,GiB, exact", False),
    ("f_up_b66_exact_mtp", "Strata 0.2.2", "budget 66.8\\,GiB, exact", True),
    # release 0.2.6 (2026-10-26 runs; its conversation cache off), with its own choice of file tier and with the
    # unbuffered one forced on; 0.2.2's other three configurations and 0.2.6's other two are in the text
    ("f_up026_b66_exact_nodraft", "Strata 0.2.6", "budget 66.8\\,GiB, exact", False),
    ("f_up026_b66_exact_mtp", "Strata 0.2.6", "budget 66.8\\,GiB, exact", True),
    ("f_up026_b66_exact_unbuf_nodraft", "Strata 0.2.6", "same, unbuffered$^\\S$", False),
    ("f_up026_b66_exact_unbuf_mtp", "Strata 0.2.6", "same, unbuffered$^\\S$", True),
    ("f_up026_b90_default_mtp", "Strata 0.2.6", "budget 90\\,GiB, defaults$^\\ddagger$", True),
    ("f_llama_cpumoeall_nomtp", "llama.cpp", "all experts on CPU", False),
    ("f_llama_cpumoeall_mtp", "llama.cpp", "all experts on CPU", True),
    ("f_llama_cpumoe20_nomtp", "llama.cpp", "20 layers' experts on CPU", False),
]
ABL = [
    ("f_ours_t60_nodraft", "full design"),
    ("f_ours_t60_nopred_nodraft", "no prediction"),
    ("f_ours_t60_nooverlap_nodraft", "no prediction, no overlap"),
    ("f_ours_t60_hostalloc_nodraft", "slots from \\texttt{cudaHostAlloc}"),
    ("f_ours_t60_mtp", "full design"),
    ("f_ours_t60_nopred_mtp", "no prediction"),
    ("f_ours_t60_nooverlap_mtp", "no prediction, no overlap"),
    ("f_ours_t60_plemmap_mtp", "n-gram table via page cache"),
]


def f(x, nd=1):
    return "--" if x is None else f"{x:.{nd}f}"


def no_overlap(run):
    """True when the run was recorded with STRATA_TIER_NO_OVERLAP=1 (results/<run>/run.json env)."""
    p = ROOT / "results" / run / "run.json"
    return p.exists() and json.load(open(p)).get("env", {}).get("STRATA_TIER_NO_OVERLAP") == "1"


def main():
    R = {r["run"]: r for r in json.load(open(ROOT / "results" / "final_report.json"))}
    OUT.mkdir(parents=True, exist_ok=True)
    lines = []
    for run, eng, cfg, mtp in MAIN:
        r = R.get(run)
        if r is None:
            lines.append(f"{eng} & {cfg} & {'\\checkmark' if mtp else '--'} & \\multicolumn{{12}}{{c}}{{\\pending{{(not run)}}}} \\\\")
            continue
        d = r["decode"]
        nv = r.get("nvme") or {}
        same = f"{r['same']}" if r.get("same") is not None else "--"
        lines.append(" & ".join([eng, cfg, "\\checkmark" if mtp else "--", f(r["mean"])] + [f(d.get(k)) for k in SHORT] +
                                [f(r["cpu"], 0), f(r["gpu"], 0), f(r["mem_gib"]), f(nv.get("prompt")) + " / " + f(nv.get("decode")),
                                 same]) + " \\\\")
        if run in ("f_ours_t75_mtp", "f_up_b66_exact_mtp", "f_up026_b90_default_mtp"):
            lines.append("\\midrule")
    head = ["\\begin{tabular}{llcrrrrrrrrrrcc}", "\\toprule",
            "Engine & Expert memory & MTP & mean & code & story & explain & math & essay & tech & CPU\\% & GPU\\% & GiB "
            "& NVMe GB & same \\\\", "\\midrule"]
    tail = ["\\bottomrule",
            "\\multicolumn{15}{p{0.97\\textwidth}}{$^\\ddagger$ skips a missed expert that every token of a window routes "
            "among its 3 lowest-weighted of 10 (\\texttt{STRATA\\_ROUTE\\_TAIL\\_SKIP=7}), which changes the output. "
            "$^\\S$ its unbuffered file tier forced on (\\texttt{STRATA\\_UNBUFFERED\\_LOAD=1}); at 66.8\\,GiB its own "
            "rule reads through the page cache.}",
            "\\end{tabular}"]
    (OUT / "q8_final.tex").write_text("\n".join(head + lines + tail) + "\n")
    lines = []
    for run, what in ABL:
        r = R.get(run)
        if r is None:
            lines.append(f"{what} & \\multicolumn{{6}}{{c}}{{\\pending{{(not run)}}}} \\\\")
            continue
        w, t = r.get("win") or {}, r.get("tier") or {}
        tpw = w.get("tpw") or 1.0
        # hit: the share of the windows' experts whose slot was ready when the layer asked for them (final_report.py's
        # ready_pct; a prefetched expert still arriving counts as not in DRAM); misses: experts read on demand per request.
        # With overlap off (STRATA_TIER_NO_OVERLAP=1) the engine logs no demand-read counter and final_report.py derives
        # the count from the GB read, which the engine prints with two decimals: an estimate, marked with \approx.
        approx = no_overlap(run)
        lines.append(" & ".join([what, f(r["mean"]), f(w.get("mspw", 0) / tpw if w else None),
                                 f(w.get("ssd_wait", 0) / tpw if w else None), f(t.get("ready_pct")),
                                 "--" if t.get("demand") is None else ("$\\approx$" if approx else "") + f"{t['demand']:,.0f}",
                                 f(t.get("gb"))]) + " \\\\")
        if run == "f_ours_t60_hostalloc_nodraft":
            lines.append("\\midrule")
    head = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
            "Configuration & tok/s & ms/tok & wait & hit\\% & misses & GB \\\\", "\\midrule"]
    tail = ["\\bottomrule",
            "\\multicolumn{7}{p{0.95\\columnwidth}}{$\\approx$ with overlap off the engine logs no demand-read counter; the "
            "count is derived from the GB read (printed with two decimals) over the blob size, uncertain by about $\\pm$1.}",
            "\\end{tabular}"]
    (OUT / "q8_ablation.tex").write_text("\n".join(head + lines + tail) + "\n")
    print("wrote", OUT / "q8_final.tex", OUT / "q8_ablation.tex")


if __name__ == "__main__":
    main()
