"""LaTeX tables of the supplementary material (paper/supplement/supplement.tex) from the run records.

    python analysis/supplement_tables.py   -> paper/supplement/tables/*.tex

S1, S2: the decode rate of every request behind Table 5 and Table 7: the server's own tokens/s per prompt and repeat
(run.json server_stats through summarize_runs.decode_rate: Strata's printed rate; for llama.cpp the rate its `eval
time` line prints, (gen_tokens-1)/gen_ms, see summarize_runs.decode_rate). S3: the runs set aside and their repeats.
S4: the slow episodes. S5, S6 (two parts), S7: the engine builds, command lines and model files of the main runs
(run.json, models.sha256: release/files/ here, the repository root in the release). Column type P (a ragged-right p column) is defined in supplement.tex.
"""
import json
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_runs as S  # noqa: E402
import stats_pairing as SP  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = ROOT / "paper" / "supplement" / "tables"
SHORT = ["code_en", "story_en", "explain_en", "math_en", "essay_ja", "tech_ja"]
HEAD = ["code", "story", "explain", "math", "essay", "tech"]

TABLE5 = [  # (run, engine, configuration, MTP) as in analysis/paper_tables.py
    ("f_ours_t45_nodraft", "this work", "DRAM tier 45\\%", False),
    ("f_ours_t45_mtp", "this work", "DRAM tier 45\\%", True),
    ("f_ours_t60_nodraft", "this work", "DRAM tier 60\\%", False),
    ("f_ours_t60_mtp", "this work", "DRAM tier 60\\%", True),
    ("f_ours_t75_nodraft", "this work", "DRAM tier 75\\%", False),
    ("f_ours_t75_mtp", "this work", "DRAM tier 75\\%", True),
    ("f_up_b66_exact_nodraft", "Strata 0.2.2", "66.8\\,GiB, exact", False),
    ("f_up_b66_exact_mtp", "Strata 0.2.2", "66.8\\,GiB, exact", True),
    ("f_up026_b66_exact_nodraft", "Strata 0.2.6", "66.8\\,GiB, exact", False),
    ("f_up026_b66_exact_mtp", "Strata 0.2.6", "66.8\\,GiB, exact", True),
    ("f_up026_b66_exact_unbuf_nodraft", "Strata 0.2.6", "same, unbuffered", False),
    ("f_up026_b66_exact_unbuf_mtp", "Strata 0.2.6", "same, unbuffered", True),
    ("f_up026_b90_default_mtp", "Strata 0.2.6", "90\\,GiB, defaults", True),
    ("f_llama_cpumoeall_nomtp", "llama.cpp", "all experts on CPU", False),
    ("f_llama_cpumoeall_mtp", "llama.cpp", "all experts on CPU", True),
    ("f_llama_cpumoe20_nomtp", "llama.cpp", "20 layers' experts on CPU", False),
]
TABLE7 = [
    ("f_ours_t60_nodraft", "full design", False),
    ("f_ours_t60_nopred_nodraft", "no prediction", False),
    ("f_ours_t60_nooverlap_nodraft", "no prediction, no overlap", False),
    ("f_ours_t60_hostalloc_nodraft", "slots from \\texttt{cudaHostAlloc}", False),
    ("f_ours_t60_mtp", "full design", True),
    ("f_ours_t60_nopred_mtp", "no prediction", True),
    ("f_ours_t60_nooverlap_mtp", "no prediction, no overlap", True),
    ("f_ours_t60_plemmap_mtp", "n-gram table via page cache", True),
]
# runs set aside: (run, what it is, why it is set aside, the run that replaces it). The figures in the reasons are from
# the runs' engine.log (server rates in request order) and samples.csv (CPU: for the t75 run the mean of cpu_all over
# all samples of each run; for the hostalloc run the mean over the samples between each request's first token and end).
SET_ASIDE = [
    ("x_f_ours_t75_mtp_contended", "this work, 75\\%, MTP",
     "other jobs ran on the machine while it ran (noted in \\texttt{bench/p2/extra\\_runs3.sh}): 2.5--13.5 tokens/s on "
     "five requests against 17.8--33.5 on the rest; "
     "CPU 9.7\\% against 5.7\\% in the repeat (means over all samples of each run)", "f_ours_t75_mtp"),
    ("x_f_ours_t60_nooverlap_mtp_contended", "ablation: no prediction, no overlap, MTP",
     "the same: 5.4 tokens/s on the first request against 20--28 on the rest", "f_ours_t60_nooverlap_mtp"),
    ("x_f_ours_t60_hostalloc_nodraft_interfered", "ablation: \\texttt{cudaHostAlloc} slots, drafts off",
     "five requests at 13.1--13.7 tokens/s against 17.1--18.0 on the rest, CPU 10.3--10.5\\% during the slow requests "
     "against 5.7--6.6\\% during the others: the pattern of the two runs above, but no other work on the machine was "
     "recorded and the cause was not established", "f_ours_t60_hostalloc_nodraft"),
    ("x_f_up026_b66_exact_mtp_convcache", "Strata 0.2.6, 66.8\\,GiB, exact, MTP",
     "its conversation cache was on: 0.2.6 sizes it from the free RAM by default (0.2.2 has the option, off by "
     "default); the reported runs turn it off (\\texttt{--conversation-cache-mib 0}); no request reused a parked state",
     "f_up026_b66_exact_mtp"),
    ("x_f_up026_b66_exact_nodraft_convcache", "Strata 0.2.6, 66.8\\,GiB, exact, drafts off",
     "the same; its first request (the code prompt, first pass) produced no tokens: the engine stopped itself after "
     "60\\,s without progress and the server started it again; the other eleven completed and are the ones counted here",
     "f_up026_b66_exact_nodraft"),
    ("x_f_up026_b90_default_mtp_watchdog", "Strata 0.2.6, 90\\,GiB, defaults, MTP",
     "with the conversation cache on, the harness's memory watchdog killed the server during the seventh request "
     "(MemAvailable under 6\\,GiB); six requests completed", "f_up026_b90_default_mtp"),
    ("x_burn_cpu0_t60_mtp", "this work, 60\\%, MTP, a process pinned to core 0",
     "deliberate interference (Section 7.1); three prompts, one repeat; not a Table 5 run", "--"),
    ("x_burn_cpu19_t60_mtp", "this work, 60\\%, MTP, a process pinned to core 19",
     "deliberate interference (Section 7.1); as above", "--"),
]
SLOW = [  # the two slow episodes and the normal run after the second (results/pagecache/README.md)
    ("pagecache/cleanroom_smoke/smoke_t60_nodraft", "clean-room build, drafts off", "slow; reclaim not sampled"),
    ("pagecache/cleanroom_smoke/smoke_t60_mtp", "clean-room build, MTP", "slow; reclaim not sampled"),
    ("pagecache/cleanroom_smoke/smoke2_cleanroom_t60_nodraft", "clean-room build, drafts off",
     "slow; kswapd reclaiming during both requests (sampler)"),
    ("pagecache/cleanroom_smoke/smoke2_paper_t60_nodraft", "strata-p1d (the paper's), drafts off",
     "slow; kswapd reclaiming during both requests (sampler)"),
    ("pagecache/cleanroom_smoke/smoke2_paper_t60_mtp", "strata-p1d (the paper's), MTP", "normal; no reclaim (sampler)"),
]
# (engine as in run.json, binary, source, patches in the release). The binary and the file names in `patches` are
# typeset in \texttt by s5(); `patches` uses {name} placeholders for them. Names as in release/files/README.md (Engines).
ENGINES = [
    ("strata-p1d", "~/src/Strata/engine/strata-p1d",
     "Strata v0.1.27 (99b40973; our clone at a790805, same tree) + the GB10 port + the expert tier",
     "{patches/strata-0.1.27-p1d.diff} applied to the unpatched v0.1.27 tree, then its new-files archive "
     "{patches/strata-0.1.27-p1d-newfiles.tgz} unpacked (the diff is cumulative: it includes the GB10 port)"),
    ("Strata 0.2.2", "~/src/Strata-upstream/build-gb10",
     "Strata v0.2.2 (011c9e60) + the Arm build changes", "{patches/strata-0.2.2-gb10.diff} and its new-files archive"),
    ("Strata 0.2.6", "~/src/Strata-upstream-026/build-gb10",
     "Strata v0.2.6 (4b9e05a2) + the Arm build changes", "{patches/strata-0.2.6-gb10.diff} and its new-files archive"),
    ("llama.cpp", "~/src/llama.cpp-master/build/bin/llama-server",
     "llama.cpp commit 339f324, CUDA, compute capability 12.1", "none"),
]
# Table S6 is typeset in two parts (one table* float each): the Strata runs of Table 5; the llama.cpp runs and the
# Table 7 ablations not already in Table 5.
S6_PARTS = [[t[0] for t in TABLE5 if t[1] != "llama.cpp"],
            [t[0] for t in TABLE5 if t[1] == "llama.cpp"] +
            [t[0] for t in TABLE7 if t[0] not in ("f_ours_t60_nodraft", "f_ours_t60_mtp")]]


def run(name):
    return json.load(open(RES / name / "run.json"))


def requests_of(name):
    p = RES / name / "requests.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


def completed(name):
    """The requests of requests.jsonl that the server finished (usage block received), in order."""
    return [q for q in requests_of(name) if SP.finished(q)]


def rates(name):
    """{(prompt, repeat): the server's own decode tokens/s} (summarize_runs.decode_rate).

    The server reports statistics only for the requests it finished, in order. They are paired with the requests by
    stats_pairing.pair_server_stats (shared with bench/harness.py and summarize_runs.py), not by the prompt_id/repeat
    labels stored in run.json: the harness that wrote those labels before 2026-11-22 zipped the statistics with all
    requests, so after a failed request every label is off by one (x_f_up026_b66_exact_nodraft_convcache, whose first
    request produced 0 tokens). The run.json files are kept as recorded. A run whose pairing is undetermined gets no
    rates (the generator then fails loudly rather than print a guess)."""
    r = run(name)
    reqs = requests_of(name)
    stats = r.get("server_stats", [])
    idx, status = SP.pair_server_stats(r.get("engine"), reqs, stats)
    if stats and status == "undetermined":
        raise RuntimeError(f"{name}: the server statistics cannot be paired with the requests")
    return {(reqs[i]["prompt_id"], reqs[i]["repeat"]): S.decode_rate(r, s) for s, i in zip(stats, idx)}


def f(x, nd=1):
    return "--" if x is None else f"{x:.{nd}f}"


def per_request(rows, what_cols):
    out = []
    for item in rows:
        name = item[0]
        r = rates(name)
        vals = [r.get((p, k)) for p in SHORT for k in (0, 1)]
        med = [st.median([v for v in (r.get((p, 0)), r.get((p, 1))) if v is not None]) for p in SHORT]
        cells = what_cols(item) + [f(v) for v in vals] + [f(st.mean(med))]
        out.append(" & ".join(cells) + " \\\\")
    return out


def tex_escape(s):
    return s.replace("_", "\\_").replace("~", "\\textasciitilde{}").replace("#", "\\#").replace("%", "\\%")


def tt(s, breakable=False):
    """\\texttt{s}; with breakable=True a line may break after '_', '/', '=', ',' and a '-' inside a word (for p{}
    columns; tt fonts allow no other breaks inside a word)."""
    t = tex_escape(s)
    if breakable:
        t = t.replace("\\_", "\\_\\allowbreak{}")
        for c in "/=,":
            t = t.replace(c, c + "\\allowbreak{}")
        t = re.sub(r"(?<=[^\s-])-(?=[^-])", "-\\\\allowbreak{}", t)  # inside a word, not the '--' of an option
    return f"\\texttt{{{t}}}"


def spaced(rows, pt=2):
    """A little space between multi-line rows (booktabs \\addlinespace)."""
    out = []
    for i, row in enumerate(rows):
        if i:
            out.append(f"\\addlinespace[{pt}pt]")
        out.append(row)
    return out


def s1():
    head = ["\\begin{tabular}{lllc" + "r" * 12 + "r}", "\\toprule",
            "Run & Engine & Expert memory & MTP & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{h}}}" for h in HEAD) +
            " & mean \\\\",
            "& & & & " + " & ".join(["1 & 2"] * 6) + " & \\\\", "\\midrule"]
    body = per_request(TABLE5, lambda it: [tt(it[0]), it[1], it[2], "\\checkmark" if it[3] else "--"])
    return head + body + ["\\bottomrule", "\\end{tabular}"]


def s2():
    head = ["\\begin{tabular}{llc" + "r" * 12 + "r}", "\\toprule",
            "Run & Configuration & MTP & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{h}}}" for h in HEAD) + " & mean \\\\",
            "& & & " + " & ".join(["1 & 2"] * 6) + " & \\\\", "\\midrule"]
    body = per_request(TABLE7, lambda it: [tt(it[0]), it[1], "\\checkmark" if it[2] else "--"])
    return head + body + ["\\bottomrule", "\\end{tabular}"]


def mean_rate(name):
    """Mean of the per-prompt medians over the requests with server statistics (Table 5's definition)."""
    r = rates(name)
    by = {}
    for (p, k), v in r.items():
        by.setdefault(p, []).append(v)
    return st.mean(st.median(v) for v in by.values()) if by else None


def cpu(name):
    import csv
    p = RES / name / "samples.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p)))
    return st.mean(float(x["cpu_all"]) for x in rows) if rows else None


def s3():
    head = ["\\begin{tabular}{P{0.17\\textwidth}P{0.13\\textwidth}rrrP{0.28\\textwidth}P{0.17\\textwidth}}", "\\toprule",
            "Run (results/) & Configuration & started & req. & tok/s & why it is set aside & reported run (tok/s) \\\\",
            "\\midrule"]
    body = []
    for name, what, why, rep in SET_ASIDE:
        r = run(name)
        n = len(r.get("server_stats", []))  # the requests the server completed (statistics exist only for those)
        rp = "--" if rep == "--" else f"{tt(rep, True)} ({f(mean_rate(rep))})"
        body.append(" & ".join([tt(name, True), what, r["started"].replace("T", " ")[5:16], str(n),
                                f(mean_rate(name)), why, rp]) + " \\\\")
    return head + spaced(body) + ["\\bottomrule", "\\end{tabular}"]


def s4():
    head = ["\\begin{tabular}{lP{0.16\\textwidth}rllP{0.22\\textwidth}}", "\\toprule",
            "Run (results/) & Engine & started & tok/s (code, essay) & first token, s & note \\\\", "\\midrule"]
    body = []
    for name, eng, note in SLOW:
        r = run(name)
        ss = r.get("server_stats", [])
        reqs = requests_of(name)
        body.append(" & ".join([tt(name.split('/')[-1]), eng, r["started"].replace("T", " ")[5:16],
                                ", ".join(f(S.decode_rate(r, s)) for s in ss),
                                ", ".join(f(q["ttft_s"]) for q in reqs), note]) + " \\\\")
    return head + body + ["\\bottomrule", "\\end{tabular}"]


def s5():
    head = ["\\begin{tabular}{lP{0.2\\textwidth}P{0.3\\textwidth}P{0.32\\textwidth}}", "\\toprule",
            "Engine & Binary & Source & Patches in the release \\\\", "\\midrule"]
    body = []
    for eng, binary, source, patches in ENGINES:
        patches = re.sub(r"\{([^}]*)\}", lambda m: tt(m.group(1), True), patches)
        body.append(" & ".join([eng, tt(binary, True), source, patches]) + " \\\\")
    return head + body + ["\\bottomrule", "\\end{tabular}"]


def s6(names):
    """Command lines of the main runs: the arguments and environment of each Table 5 and Table 7 run (one part)."""
    head = ["\\begin{tabular}{P{0.16\\textwidth}P{0.79\\textwidth}}", "\\toprule",
            "Run & Arguments (run.json) and environment \\\\", "\\midrule"]
    body = []
    for name in names:
        r = run(name)
        args = " ".join(a.replace("~/", "~/") for a in r["args"])
        env = " ".join(f"{k}={v}" for k, v in (r.get("env") or {}).items())
        body.append(f"{tt(name, True)} & {tt(args, True)}" + (f" \\quad [{tt(env, True)}]" if env else "") + " \\\\")
    return head + body + ["\\bottomrule", "\\end{tabular}"]


def models_sha256():
    """The model checksum list: release/files/models.sha256 in the writing repository, models.sha256 at the root of
    the released repository (release/make_release.sh copies it there)."""
    for p in (ROOT / "release" / "files" / "models.sha256", ROOT / "models.sha256"):
        if p.exists():
            return p
    raise FileNotFoundError("models.sha256 (release/files/ or the repository root)")


def s7():
    lines = [l.strip() for l in open(models_sha256()) if l.strip() and not l.startswith("#")]
    head = ["\\begin{tabular}{P{0.4\\textwidth}P{0.28\\textwidth}}", "\\toprule",
            "File (relative to the home directory) & SHA-256 (two lines of 32 hex digits) \\\\", "\\midrule"]
    body = []
    for l in lines:
        h, p = l.split(None, 1)
        body.append(f"{tt(p, True)} & \\texttt{{{h[:32]}\\allowbreak {h[32:]}}} \\\\")
    return head + spaced(body) + ["\\bottomrule", "\\end{tabular}"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("s1_table5_requests", s1), ("s2_table7_requests", s2), ("s3_set_aside", s3), ("s4_slow", s4),
                     ("s5_engines", s5), ("s6_commands_a", lambda: s6(S6_PARTS[0])),
                     ("s6_commands_b", lambda: s6(S6_PARTS[1])), ("s7_models", s7)):
        (OUT / f"{name}.tex").write_text("\n".join(fn()) + "\n")
    old = OUT / "s6_commands.tex"  # replaced by the two parts
    if old.exists():
        old.unlink()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
