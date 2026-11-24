#!/bin/bash
# P2 extra: Strata 0.2.2's prompt processing on the Q8_0 at our 60% memory (exact settings), the 5.4K-token prompt.
# Waits for the process given as $1 (the previous queue) to exit.
cd ~/work/uma-moe/bench
while kill -0 "$1" 2>/dev/null; do sleep 30; done
EXACT="ENV:STRATA_ROUTE_TAIL_SKIP=0 ENV:STRATA_PREFILL_CPU_SHARE=0 ENV:STRATA_DECODE_TIMING=1"
flock ~/work/uma-moe/.gpu.lock ../.venv/bin/python p2/upstream.py f_up_b66_exact_prefill 66.8 1024 mtp --prompts long5k \
  --max-tokens 64 $EXACT
