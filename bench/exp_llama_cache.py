"""E4b: llama.cpp master with the experts in host memory (--cpu-moe, CPU computes them) and its GPU SLRU expert
cache (--moe-cache-mib), i.e. the discrete-GPU tiering recipe applied to a UMA machine."""
import sys, harness as H
PROMPTS = ["code_en", "story_en", "essay_ja"]
BASE = ["-m", str(H.Q2_GGUF), "-ngl", "999", "-fa", "on", "-c", "8192", "--no-warmup", "--cpu-moe"]
RUNS = {f"e4b_llama_cpumoe_cache{mib}": BASE + (["--moe-cache-mib", str(mib)] if mib else []) for mib in (0, 4096, 8192, 16384, 24576, 32768)}
for name in (sys.argv[1:] or RUNS):
    s = H.run(name, "llama", RUNS[name], PROMPTS, repeats=2, need_gib=85, meta={"exp": "E4b"}, max_tokens=256)
    print(name, {k: s.get(k) for k in ("load_s", "requests", "error", "watchdog", "mem_avail_loaded_gib", "mem_avail_min_gib")}, flush=True)
