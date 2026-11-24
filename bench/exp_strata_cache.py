"""E4: Strata expert-cache capacity sweep on UMA: cache slots x drafting, plus the miss path (CPU vs GPU zero-copy)
and static vs adaptive residency at a mid-size cache."""
import sys, harness as H
PROMPTS = ["code_en", "story_en", "essay_ja"]
RUNS = {}
for slots in (1024, 4096, 8192, 12288, 16384, 20480, 24576):
    for mtp in (True, False):
        RUNS[f"e4_strata_c{slots}_{'mtp' if mtp else 'nodraft'}"] = H.strata_args(slots, mtp=mtp)
for slots in (1024, 8192):
    RUNS[f"e4_strata_c{slots}_nodraft_pcie1"] = H.strata_args(slots, mtp=False, extra=["--pcie-frac", "1.0"])
    RUNS[f"e4_strata_c{slots}_nodraft_pcie0"] = H.strata_args(slots, mtp=False, extra=["--pcie-frac", "0.0"])
RUNS["e4_strata_c8192_nodraft_static"] = H.strata_args(8192, mtp=False, extra=["--adapt-every", "0"])
RUNS["e4_strata_c1024_mtp_pcie1"] = H.strata_args(1024, mtp=True, extra=["--pcie-frac", "1.0"])
for name in (sys.argv[1:] or RUNS):
    s = H.run(name, "strata", RUNS[name], PROMPTS, repeats=2, need_gib=85, meta={"exp": "E4"}, max_tokens=256)
    print(name, {k: s.get(k) for k in ("load_s", "requests", "error", "watchdog", "mem_avail_loaded_gib", "mem_avail_min_gib")}, flush=True)
