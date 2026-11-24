#!/bin/bash
# P2 final decode comparison on the 188 GB Q8_0 (GB10, 128 GB): 6 short prompts x 2 repeats x 256 tokens, temperature 0.
#   ours (strata-p1d): DRAM tier 45 / 60 / 75% of the experts, MTP on/off; ablations at 60%: no next-layer prediction,
#   no intra-layer overlap; upstream Strata 0.2.2 at our 60% memory (exact) and with a 90 GiB budget (its defaults);
#   llama.cpp master: --cpu-moe and --n-cpu-moe 20, -lm mmap, MTP off/on.
cd ~/work/uma-moe/bench
L=~/work/uma-moe/.gpu.lock
P=code_en,story_en,explain_en,math_en,essay_ja,tech_ja
T="--prompts $P --repeats 2 --max-tokens 256"
for pct_gib in "60:66.8" "45:48.8" "75:84.7"; do
  pct=${pct_gib%%:*}; gib=${pct_gib##*:}
  for m in mtp nodraft; do
    flock $L ../.venv/bin/python p1/tier.py f_ours_t${pct}_${m} q8 1024 $gib $m --exe strata-p1d $T ENV:STRATA_DECODE_TIMING=1
  done
done
for m in mtp nodraft; do
  flock $L ../.venv/bin/python p1/tier.py f_ours_t60_nopred_${m} q8 1024 66.8 $m --exe strata-p1d $T ENV:STRATA_DECODE_TIMING=1 ENV:STRATA_TIER_PREFETCH=0
  flock $L ../.venv/bin/python p1/tier.py f_ours_t60_nooverlap_${m} q8 1024 66.8 $m --exe strata-p1d $T ENV:STRATA_DECODE_TIMING=1 ENV:STRATA_TIER_PREFETCH=0 ENV:STRATA_TIER_NO_OVERLAP=1
done
EXACT="ENV:STRATA_ROUTE_TAIL_SKIP=0 ENV:STRATA_PREFILL_CPU_SHARE=0 ENV:STRATA_DECODE_TIMING=1"
for m in mtp nodraft; do
  flock $L ../.venv/bin/python p2/upstream.py f_up_b66_exact_${m} 66.8 1024 $m $T $EXACT
done
flock $L ../.venv/bin/python p2/upstream.py f_up_b90_default_mtp 90 1024 mtp $T ENV:STRATA_DECODE_TIMING=1
for spec in "all:nomtp" "all:mtp" "20:nomtp"; do
  k=${spec%%:*}; m=${spec##*:}
  flock $L ../.venv/bin/python p2/llama_q8.py f_llama_cpumoe${k}_${m} $k $m $T
done
