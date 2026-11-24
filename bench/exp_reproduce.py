"""E2: Strata (MTP on / drafts off) and llama.cpp master (no MTP / MTP n=2,3,4) on the same Q2_0 GGUF, same prompts."""
import sys, harness as H
PROMPTS = list(H.P.SHORT) + ["long5k", "long19k", "long26k"]
LLAMA_BASE = ["-m", str(H.Q2_GGUF), "-ngl", "999", "-fa", "on", "-c", "32768", "--no-warmup"]
LLAMA_MTP = ["-md", str(H.MTP_GGUF), "--spec-type", "draft-mtp", "-ngld", "999"]
RUNS = {
    "e2_strata_mtp": ("strata", H.strata_args("auto", mtp=True)),
    "e2_strata_nodraft": ("strata", H.strata_args("auto", mtp=False)),
    "e2_strata_spec3": ("strata", H.strata_args("auto", mtp=True, spec=3)),
    "e2_strata_spec5": ("strata", H.strata_args("auto", mtp=True, spec=5)),
    "e2_llama_nomtp": ("llama", LLAMA_BASE),
    "e2_llama_mtp2": ("llama", LLAMA_BASE + LLAMA_MTP + ["--spec-draft-n-max", "2"]),
    "e2_llama_mtp3": ("llama", LLAMA_BASE + LLAMA_MTP + ["--spec-draft-n-max", "3"]),
    "e2_llama_mtp4": ("llama", LLAMA_BASE + LLAMA_MTP + ["--spec-draft-n-max", "4"]),
    "e2_llama_nomtp_managed": ("llama", LLAMA_BASE),   # all weights in cudaMallocManaged (host page tables)
}
ENV = {"e2_llama_nomtp_managed": {"GGML_CUDA_ENABLE_UNIFIED_MEMORY": "1"}}
for name in (sys.argv[1:] or RUNS):
    engine, args = RUNS[name]
    s = H.run(name, engine, args, PROMPTS, repeats=3, need_gib=85, meta={"exp": "E2"}, env=ENV.get(name))
    print(name, {k: s.get(k) for k in ("load_s", "requests", "error", "watchdog", "mem_avail_loaded_gib", "mem_avail_min_gib")}, flush=True)
