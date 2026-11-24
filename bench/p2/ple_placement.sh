#!/bin/bash
# P1/P2: the n-gram (PLE) table's placement for Q8_0 at 60% residency: unbuffered 4 KiB reads (default) against the page
# cache (mmap), MTP on, 6 prompts x 2 x 256 tokens.
cd ~/work/uma-moe/bench
while pgrep -f '[p]2/final_runs.sh' > /dev/null; do sleep 30; done
flock ~/work/uma-moe/.gpu.lock ../.venv/bin/python p1/tier.py f_ours_t60_plemmap_mtp q8 1024 66.8 mtp --exe strata-p1d \
  --prompts code_en,story_en,explain_en,math_en,essay_ja,tech_ja --repeats 2 --max-tokens 256 ENV:STRATA_DECODE_TIMING=1 --ple-io mmap
