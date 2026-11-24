#!/bin/bash
# Downloads used by the UMA MoE study (2026-10-21).
set -e
HF=~/work/uma-moe/.venv/bin/hf
cd ~/models
$HF download unsloth/Qwen3.8-Flash-Next-GGUF --local-dir unsloth-Qwen3.8-Flash-Next-GGUF \
  --include "MTP/README.md" "MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf" "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"
echo "MTP done $(date -Is)"
$HF download unsloth/Qwen3.8-Flash-Next-GGUF --local-dir unsloth-Qwen3.8-Flash-Next-GGUF --include "Q8_0/*" --max-workers 8
echo "Q8_0 done $(date -Is)"
