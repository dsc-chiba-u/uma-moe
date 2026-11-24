#!/bin/bash
# P1/P2 extra: what backs the DRAM tier's slots (Q8_0, 60% residency, drafts off, 6 prompts x 2 x 256 tokens):
# 4 KB pages registered with cudaHostRegister (the default, f_ours_t60_nodraft), cudaHostAlloc (STRATA_ARENA_HOSTALLOC=1),
# and anonymous memory with transparent 2 MB pages before registration (STRATA_ARENA_THP=1, needs strata-p1e).
cd ~/work/uma-moe/bench
while pgrep -f '[p]2/ple_placement.sh' > /dev/null; do sleep 30; done
T="--prompts code_en,story_en,explain_en,math_en,essay_ja,tech_ja --repeats 2 --max-tokens 256"
flock ~/work/uma-moe/.gpu.lock ../.venv/bin/python p1/tier.py f_ours_t60_hostalloc_nodraft q8 1024 66.8 nodraft \
  --exe strata-p1d $T ENV:STRATA_DECODE_TIMING=1 ENV:STRATA_ARENA_HOSTALLOC=1
(cd ~/src/Strata && nice cmake --build build --target strata -j 8 > ~/work/uma-moe/results/pipeline/build_p1e.log 2>&1 &&
  cp build/strata engine/strata-p1e) || { echo "build failed"; exit 1; }
( while sleep 20; do echo "$(date +%T) $(grep AnonHugePages /proc/meminfo)"; done ) > ~/work/uma-moe/results/pipeline/thp_meminfo.log &
W=$!
flock ~/work/uma-moe/.gpu.lock ../.venv/bin/python p1/tier.py f_ours_t60_thp_nodraft q8 1024 66.8 nodraft \
  --exe strata-p1e $T ENV:STRATA_DECODE_TIMING=1 ENV:STRATA_ARENA_THP=1
kill $W
