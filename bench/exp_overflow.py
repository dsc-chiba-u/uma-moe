"""E5: a model larger than memory. Qwen3.8-Flash-Next Q8_0 (188 GB) on the 128 GB UMA box with llama.cpp master:
experts stay in the mmap'd file (--cpu-moe; the page cache holds what fits, misses fault in from NVMe), the rest is
on the GPU. Every run starts with the model evicted from the page cache."""
import subprocess, sys, glob, harness as H
Q8 = sorted(glob.glob(str(H.Path.home() / "models/unsloth-Qwen3.8-Flash-Next-GGUF/Q8_0/*.gguf")))
# llama.cpp disables mmap on integrated GPUs by default (CUDA reports mmap_support = false for IGPU, PR #28160) and would
# then read the whole 188 GB into RAM: -lm mmap keeps the experts in the page cache and, with the default lazy mode, reads
# the 54 GB per-layer embedding (N-gram) table on demand.
BASE = ["-m", Q8[0], "-ngl", "999", "-fa", "on", "-c", "16384", "--no-warmup", "--cpu-moe", "-lm", "mmap"]
RUNS = {
    "e5_llama_q8_cpumoe": (BASE, ["code_en", "story_en", "code_en"]),
    "e5_llama_q8_cpumoe_cache16g": (BASE + ["--moe-cache-mib", "16384"], ["code_en", "story_en", "code_en"]),
    "e5_llama_q8_cpumoe_mtp2": (BASE + ["-md", str(H.MTP_GGUF), "--spec-type", "draft-mtp", "-ngld", "999", "--spec-draft-n-max", "2"],
                                ["code_en", "story_en", "code_en"]),
    "e5_llama_q8_cpumoe_prefill": (BASE, ["long5k"]),
    "e5_llama_q8_cpumoe_prefill_ub4096": (BASE + ["-ub", "4096", "-b", "4096"], ["long5k"]),
}
for name in (sys.argv[1:] or RUNS):
    args, prompts = RUNS[name]
    for f in Q8:
        subprocess.run([str(H.ROOT / "bench/nvme_bench"), "evict", f], check=True)
    s = H.run(name, "llama", args, prompts, repeats=1, need_gib=100, meta={"exp": "E5"}, max_tokens=128)
    print(name, {k: s.get(k) for k in ("load_s", "requests", "error", "watchdog", "mem_avail_loaded_gib", "mem_avail_min_gib")}, flush=True)
