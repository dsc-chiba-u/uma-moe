#!/bin/bash
# P1: prediction width (top-k) and depth (layers ahead) at 45% residency, Q8_0, drafts off and MTP on.
cd ~/work/uma-moe/bench
for kd in "16:1" "24:1" "16:2"; do
  k=${kd%%:*}; d=${kd##*:}
  for m in nodraft mtp; do
    flock ~/work/uma-moe/.gpu.lock ../.venv/bin/python p1/tier.py p1c_q8_t45_k${k}d${d}_${m} q8 1024 48.8 $m --exe strata-p1c \
      ENV:STRATA_DECODE_TIMING=1 ENV:STRATA_VERIFY_PROFILE=1 ENV:STRATA_TIER_PREFETCH_K=$k ENV:STRATA_TIER_PREFETCH_DEPTH=$d
  done
done
