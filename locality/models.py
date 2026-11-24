"""The models of the locality study: GGUF source, routing configuration (from the GGUF header) and chat settings."""
from pathlib import Path

HOME = Path.home()
MROOT = HOME / "models/locality"

MODELS = {
    # key: repo/files on Hugging Face (None = local), gguf path, notes
    # TheBloke's 2023 conversion stores experts as separate tensors, which llama.cpp master no longer loads
    "mixtral-8x7b": {
        "repo": "mradermacher/Mixtral-8x7B-Instruct-v0.1-GGUF", "files": ["Mixtral-8x7B-Instruct-v0.1.Q4_K_M.gguf"],
        "quant": "Q4_K_M", "label": "Mixtral-8x7B-Instruct-v0.1", "experts": 8, "k": 2, "shared": 0,
        "kwargs": {}},
    "gpt-oss-20b": {
        "repo": "ggml-org/gpt-oss-20b-GGUF", "files": ["gpt-oss-20b-MXFP4.gguf"],
        "quant": "MXFP4 (native)", "label": "gpt-oss-20b", "experts": 32, "k": 4, "shared": 0,
        "kwargs": {"reasoning_effort": "low"},
        # raw text is far out of distribution for gpt-oss (NLL rises with position to 4-8 nats/token): documents are
        # teacher-forced as the final-channel answer of an assistant turn (harmony format)
        "doc_wrap": {"split": "<|start|>user", "append": "<|start|>user<|message|>Write a document.<|end|>"
                                                         "<|start|>assistant<|channel|>final<|message|>"}},
    "glm-4.7-flash": {
        "repo": "unsloth/GLM-4.7-Flash-GGUF", "files": ["GLM-4.7-Flash-Q4_K_M.gguf"],
        "quant": "Q4_K_M", "label": "GLM-4.7-Flash", "experts": 64, "k": 4, "shared": 1,
        "kwargs": {"enable_thinking": False},
        # the tokenizer adds no BOS, but every GLM prompt starts with [gMASK]<sop>; without it some sequences that start
        # mid-text are unstable (NLL 5-10 nats/token for two of the nine context probes)
        "doc_prefix": "[gMASK]<sop>"},
    "qwen3-30b-a3b": {
        "repo": "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF", "files": ["Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"],
        "quant": "Q4_K_M", "label": "Qwen3-30B-A3B-Instruct-2507", "experts": 128, "k": 8, "shared": 0,
        "kwargs": {}},
    "qwen3.6-35b-a3b": {
        "repo": "unsloth/Qwen3.6-35B-A3B-GGUF", "files": ["Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"],
        "quant": "UD-Q4_K_M", "label": "Qwen3.6-35B-A3B", "experts": 256, "k": 8, "shared": 1,
        "kwargs": {"enable_thinking": False}},
    "qwen3.8-flash-next": {
        "local": HOME / "src/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf",
        "quant": "Q2_0 (GSQ-RCO)", "label": "Qwen3.8-Flash-Next", "experts": 512, "k": 10, "shared": 1,
        "kwargs": {"enable_thinking": False}},
}


def gguf_path(key):
    m = MODELS[key]
    if "local" in m:
        return Path(m["local"])
    return MROOT / key / m["files"][0]


def file_bytes(key):
    p = gguf_path(key)
    if "local" in MODELS[key]:
        return sum(q.stat().st_size for q in p.parent.glob(p.name.split("-00001-of-")[0] + "-0000*-of-*.gguf"))
    return sum((MROOT / key / f).stat().st_size for f in MODELS[key]["files"])
