"""P1: the expert tier (pinned DRAM slots over experts.bin on the SSD) in Strata.
usage: tier.py <name> <q2|q8> <vram_slots> <tier_gib|0> <mtp|nodraft> [--prompts a,b,c] [--repeats N] [--max-tokens N]
               [--exe strata-p1] [ENV:K=V ...] [extra engine args ...]
tier_gib 0 = no tier (the all-experts arena: Q2_0 only)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import harness as H
CFG = {"q2": None, "q8": os.path.join(os.path.dirname(__file__), "strata-q8_0.json")}
TIER = {"q2": os.path.expanduser("~/src/Strata-data/packs/q2_0-tier/experts.bin"), "q8": "pack"}
name, model, vram, tier_gib, mode = sys.argv[1:6]
rest = sys.argv[6:]
prompts, repeats, max_tokens, exe, env, extra = ["code_en", "story_en", "essay_ja"], 1, 256, "strata-p1", {}, []
i = 0
while i < len(rest):
    a = rest[i]
    if a == "--prompts": prompts = rest[i + 1].split(","); i += 2; continue
    if a == "--repeats": repeats = int(rest[i + 1]); i += 2; continue
    if a == "--max-tokens": max_tokens = int(rest[i + 1]); i += 2; continue
    if a == "--exe": exe = rest[i + 1]; i += 2; continue
    if a.startswith("ENV:"): k, v = a[4:].split("=", 1); env[k] = v; i += 1; continue
    extra.append(a); i += 1
if float(tier_gib) > 0:
    extra = ["--expert-tier", TIER[model], "--expert-tier-gib", tier_gib] + extra
args = H.strata_args(int(vram) if vram != "auto" else "auto", mtp=(mode == "mtp"), extra=extra, cfg_file=CFG[model])
need = 20 + (float(tier_gib) if float(tier_gib) > 0 else 40) + (8 if model == "q8" else 4)
s = H.run(name, "strata", args, prompts, repeats=repeats, need_gib=need, meta={"exp": "P1", "model": model,
          "vram_slots": vram, "tier_gib": float(tier_gib), "mode": mode}, max_tokens=max_tokens, env=env or None,
          exe=str(H.STRATA / "engine" / exe), cfg_file=CFG[model])
print(name, {k: s.get(k) for k in ("load_s", "requests", "error", "watchdog", "mem_avail_loaded_gib", "mem_avail_min_gib")}, flush=True)
