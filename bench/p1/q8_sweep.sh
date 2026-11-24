#!/bin/bash
# P1: Q8_0 decode against the DRAM tier's size (share of all 24,576 experts resident: VRAM tier 1,024 + DRAM slots of
# 4.98 MiB), drafts off and MTP on.  3 prompts x 1 repeat x 256 tokens; STRATA_DECODE_TIMING on (the window breakdown).
cd ~/work/uma-moe/bench
for pct_gib in "25:24.9" "45:48.8" "60:66.8" "75:84.7"; do
  pct=${pct_gib%%:*}; gib=${pct_gib##*:}
  for m in mtp nodraft; do
    flock ~/work/uma-moe/.gpu.lock ../.venv/bin/python p1/tier.py p1_q8_t${pct}_${m} q8 1024 $gib $m ENV:STRATA_DECODE_TIMING=1 ENV:STRATA_VERIFY_PROFILE=1
  done
done
