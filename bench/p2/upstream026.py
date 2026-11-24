"""P2 baseline, latest release: upstream Strata 0.2.6 (built for GB10 from 4b9e05a2, tag v0.2.6 with the aarch64 arrangement of its Metal build) on the
188 GB Q8_0, its RAM-budget mode (experts beyond the budget mapped from the GGUF; the CPU computes the ones the GPU
cache does not hold).
usage: upstream.py <name> <budget_gib> <expert_cache> <mtp|nodraft> [--prompts a,b] [--max-tokens N] [--repeats N] [ENV:K=V]"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import harness as H
CFG = os.path.join(os.path.dirname(__file__), "strata-upstream026-q8_0.json")
name, budget, cache, mode = sys.argv[1:5]
rest = sys.argv[5:]
prompts, max_tokens, repeats, env, extra = ["code_en", "story_en", "essay_ja"], 256, 1, {}, []
i = 0
while i < len(rest):
    a = rest[i]
    if a == "--prompts": prompts = rest[i + 1].split(","); i += 2; continue
    if a == "--max-tokens": max_tokens = int(rest[i + 1]); i += 2; continue
    if a == "--repeats": repeats = int(rest[i + 1]); i += 2; continue
    if a.startswith("ENV:"): k, v = a[4:].split("=", 1); env[k] = v; i += 1; continue
    extra.append(a); i += 1
args = H._home(json.load(open(CFG)))["args"]
args[args.index("--resident-budget-gib") + 1] = budget
args[args.index("--expert-cache") + 1] = cache
if mode != "mtp":
    args += ["--mtp-max-t", "1", "--suffix-draft", "0"]
args += extra
s = H.run(name, "strata", args, prompts, repeats=repeats, need_gib=float(budget) + 18, meta={"exp": "P2", "engine": "upstream-0.2.6",
          "budget": float(budget), "cache": cache, "mode": mode}, max_tokens=max_tokens, env=env or None, cfg_file=CFG,
          server_root=os.path.expanduser("~/src/Strata-upstream-026"))
print(name, {k: s.get(k) for k in ("load_s", "requests", "error", "watchdog", "mem_avail_loaded_gib", "mem_avail_min_gib")}, flush=True)
