#!/bin/bash
# P2: Strata 0.2.2's two departures from our setting, one at a time: its 90 GiB budget with exact settings, and its
# default approximation (STRATA_ROUTE_TAIL_SKIP=7 and the default prompt CPU share) at our 66.8 GiB.
# Waits for the process given as $1 (the previous queue) to exit.
cd ~/work/uma-moe/bench
while kill -0 "$1" 2>/dev/null; do sleep 30; done
L=~/work/uma-moe/.gpu.lock
T="--prompts code_en,story_en,explain_en,math_en,essay_ja,tech_ja --repeats 2 --max-tokens 256"
EXACT="ENV:STRATA_ROUTE_TAIL_SKIP=0 ENV:STRATA_PREFILL_CPU_SHARE=0 ENV:STRATA_DECODE_TIMING=1"
flock $L ../.venv/bin/python p2/upstream.py f_up_b90_exact_mtp 90 1024 mtp $T $EXACT
flock $L ../.venv/bin/python p2/upstream.py f_up_b66_default_mtp 66.8 1024 mtp $T ENV:STRATA_DECODE_TIMING=1
