"""P0 diagnosis: where the MTP verify window's experts are computed (CPU / VRAM / GPU zero-copy) with the miss path set to
GPU zero-copy (--pcie-frac 1) and a small expert cache, drafts off vs MTP on, with Strata's own per-window timing."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import harness as H
PROMPTS = ["code_en", "story_en", "essay_ja"]
ENV = {"STRATA_DECODE_TIMING": "1", "STRATA_VERIFY_PROFILE": "1"}
RUNS = {
    "p0_diag_c1024_nodraft_pcie1": H.strata_args(1024, mtp=False, extra=["--pcie-frac", "1.0"]),
    "p0_diag_c1024_mtp_pcie1": H.strata_args(1024, mtp=True, extra=["--pcie-frac", "1.0"]),
    "p0_diag_c24576_mtp": H.strata_args(24576, mtp=True),
}
for name in (sys.argv[1:] or RUNS):
    s = H.run(name, "strata", RUNS[name], PROMPTS, repeats=1, need_gib=85, meta={"exp": "P0"}, max_tokens=256, env=ENV)
    print(name, {k: s.get(k) for k in ("load_s", "requests", "error", "watchdog", "mem_avail_loaded_gib")}, flush=True)
