#!/bin/bash
# P2: upstream Strata 0.2.2 on the Q8_0 (GB10), 3 prompts x 256 tokens.
#   same memory as ours at 60% (RAM budget 66.8 GiB + 1,024 GPU-cache slots), exact (no tail skip, no CPU prefill share),
#   MTP on and off; and its own recommendation for this RAM (budget 90 GiB, near its RAM-less-24-GB rule) with its defaults (tail skip on).
cd ~/work/uma-moe/bench
L=~/work/uma-moe/.gpu.lock
EXACT="ENV:STRATA_ROUTE_TAIL_SKIP=0 ENV:STRATA_PREFILL_CPU_SHARE=0 ENV:STRATA_DECODE_TIMING=1"
flock $L ../.venv/bin/python p2/upstream.py p2_up_q8_b66_exact_mtp 66.8 1024 mtp $EXACT
flock $L ../.venv/bin/python p2/upstream.py p2_up_q8_b66_exact_nodraft 66.8 1024 nodraft $EXACT
flock $L ../.venv/bin/python p2/upstream.py p2_up_q8_b90_default_mtp 90 1024 mtp ENV:STRATA_DECODE_TIMING=1
