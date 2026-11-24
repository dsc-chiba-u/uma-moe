#!/bin/bash
# P2: the cudaHostAlloc arm again (the first run had 5 of 12 requests slowed by an unexplained interference, kept as
# x_f_ours_t60_hostalloc_nodraft_interfered). Waits for the process given as $1 (the previous queue) to exit.
cd ~/work/uma-moe/bench
while kill -0 "$1" 2>/dev/null; do sleep 30; done
T="--prompts code_en,story_en,explain_en,math_en,essay_ja,tech_ja --repeats 2 --max-tokens 256"
flock ~/work/uma-moe/.gpu.lock ../.venv/bin/python p1/tier.py f_ours_t60_hostalloc_nodraft q8 1024 66.8 nodraft \
  --exe strata-p1d $T ENV:STRATA_DECODE_TIMING=1 ENV:STRATA_ARENA_HOSTALLOC=1
