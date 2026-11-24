"""P0 before/after: Strata 0.1.27 (strata-base) vs the fixed miss path (strata-p0), E2 protocol (6 short prompts x 512
tokens + 5.4K/18.9K/26K-token prompts x 200 tokens, 3 repeats), drafts off and MTP on."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import harness as H
PROMPTS = list(H.P.SHORT) + ["long5k", "long19k", "long26k"]
E = lambda n: str(H.STRATA / "engine" / n)
RUNS = {}
for mode in ("mtp", "nodraft"):
    m = mode == "mtp"
    RUNS[f"p0f_base_auto_{mode}"] = ("strata-base", H.strata_args("auto", mtp=m))
    RUNS[f"p0f_base_c4096_{mode}"] = ("strata-base", H.strata_args(4096, mtp=m))
    RUNS[f"p0f_base_c4096_pcie1_{mode}"] = ("strata-base", H.strata_args(4096, mtp=m, extra=["--pcie-frac", "1.0"]))
    RUNS[f"p0f_fix_auto_{mode}"] = ("strata-p0", H.strata_args("auto", mtp=m))
    RUNS[f"p0f_fix_c1024_{mode}"] = ("strata-p0", H.strata_args(1024, mtp=m))
RUNS["p0f_fix_c24576_mtp"] = ("strata-p0", H.strata_args(24576, mtp=True))
for name in (sys.argv[1:] or RUNS):
    exe, args = RUNS[name]
    s = H.run(name, "strata", args, PROMPTS, repeats=3, need_gib=85, meta={"exp": "P0"}, exe=E(exe))
    print(name, {k: s.get(k) for k in ("load_s", "requests", "error", "watchdog", "mem_avail_loaded_gib", "mem_avail_min_gib")}, flush=True)
