"""P0 A/B runs: one config per name, 3 prompts x 1 repeat x 256 tokens, Strata's per-window timing on.
usage: ab.py <name> <exe|base> <expert_cache> <mtp|nodraft> [extra args...]"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import harness as H
name, exe, cache, mode = sys.argv[1:5]
extra = sys.argv[5:]
exe = None if exe == "base" else (exe if "/" in exe else str(H.STRATA / "engine" / exe))
ENV = {"STRATA_DECODE_TIMING": "1", "STRATA_VERIFY_PROFILE": "1"}
for kv in [a for a in extra if a.startswith("ENV:")]:
    k, v = kv[4:].split("=", 1); ENV[k] = v
extra = [a for a in extra if not a.startswith("ENV:")]
args = H.strata_args(int(cache), mtp=(mode == "mtp"), extra=extra)
s = H.run(name, "strata", args, ["code_en", "story_en", "essay_ja"], repeats=1, need_gib=85, meta={"exp": "P0"},
          max_tokens=256, env=ENV, exe=exe)
print(name, {k: s.get(k) for k in ("load_s", "requests", "error", "watchdog", "mem_avail_loaded_gib")}, flush=True)
