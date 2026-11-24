"""P2 baseline: llama.cpp master on the 188 GB Q8_0 (does not fit in RAM). Experts beyond what the GPU holds stay in
the mmap'd GGUF (page cache, misses from NVMe) and are computed on the CPU. -lm mmap is required: llama.cpp's default
load mode on this integrated GPU reads the whole file into memory. Every run starts with the model evicted.
usage: llama_q8.py <name> [n_cpu_moe|all] [mtp|nomtp] [--prompts ...] [--repeats N] [--max-tokens N]"""
import subprocess, sys, glob, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import harness as H
Q8 = sorted(glob.glob(str(H.Path.home() / "models/unsloth-Qwen3.8-Flash-Next-GGUF/Q8_0/*.gguf")))
name = sys.argv[1]
ncpu = sys.argv[2] if len(sys.argv) > 2 else "all"
mtp = len(sys.argv) > 3 and sys.argv[3] == "mtp"
rest = sys.argv[4:]
prompts, repeats, max_tokens = ["code_en", "story_en", "essay_ja"], 1, 256
i = 0
while i < len(rest):
    if rest[i] == "--prompts": prompts = rest[i + 1].split(","); i += 2; continue
    if rest[i] == "--repeats": repeats = int(rest[i + 1]); i += 2; continue
    if rest[i] == "--max-tokens": max_tokens = int(rest[i + 1]); i += 2; continue
    i += 1
args = ["-m", Q8[0], "-ngl", "999", "-fa", "on", "-c", "32768", "--no-warmup", "-lm", "mmap"]
args += ["--cpu-moe"] if ncpu == "all" else ["--n-cpu-moe", ncpu]
if mtp:
    args += ["-md", str(H.MTP_GGUF), "--spec-type", "draft-mtp", "-ngld", "999", "--spec-draft-n-max", "3"]
for f in Q8:
    subprocess.run([str(H.ROOT / "bench/nvme_bench"), "evict", f], check=True)
s = H.run(name, "llama", args, prompts, repeats=repeats, need_gib=100, meta={"exp": "P2", "n_cpu_moe": ncpu, "mtp": mtp},
          max_tokens=max_tokens)
print(name, {k: s.get(k) for k in ("load_s", "requests", "error", "watchdog", "mem_avail_loaded_gib", "mem_avail_min_gib")}, flush=True)
