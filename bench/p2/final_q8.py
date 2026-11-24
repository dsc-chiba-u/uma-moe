"""P2: Q8_0 end-to-end, E2 protocol (6 short prompts x 512 tokens, the 5.4K/18.9K/26K-token prompts x 200 tokens, 3
repeats), Strata with the expert tier at a given DRAM size, MTP on/off.
usage: final_q8.py <name> <tier_gib> <mtp|nodraft> [--exe strata-p1] [--vram 1024] [ENV:K=V ...] [extra args]"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import harness as H
CFG = os.path.join(os.path.dirname(__file__), "..", "p1", "strata-q8_0.json")
PROMPTS = list(H.P.SHORT) + ["long5k", "long19k", "long26k"]
name, gib, mode = sys.argv[1:4]
rest = sys.argv[4:]
exe, vram, env, extra, repeats, prompts = "strata-p1", 1024, {}, [], 3, PROMPTS
i = 0
while i < len(rest):
    a = rest[i]
    if a == "--exe": exe = rest[i + 1]; i += 2; continue
    if a == "--vram": vram = int(rest[i + 1]); i += 2; continue
    if a == "--repeats": repeats = int(rest[i + 1]); i += 2; continue
    if a == "--prompts": prompts = rest[i + 1].split(","); i += 2; continue
    if a.startswith("ENV:"): k, v = a[4:].split("=", 1); env[k] = v; i += 1; continue
    extra.append(a); i += 1
args = H.strata_args(vram, mtp=(mode == "mtp"), extra=["--expert-tier", "pack", "--expert-tier-gib", gib] + extra, cfg_file=CFG)
s = H.run(name, "strata", args, prompts, repeats=repeats, need_gib=float(gib) + 30, meta={"exp": "P2", "tier_gib": float(gib),
          "mode": mode, "vram": vram}, env=env or None, exe=str(H.STRATA / "engine" / exe), cfg_file=CFG)
print(name, {k: s.get(k) for k in ("load_s", "requests", "error", "watchdog", "mem_avail_loaded_gib", "mem_avail_min_gib")}, flush=True)
