#!/bin/bash
# P2 (continued): upstream Strata 0.2.2 on the Q8_0, drafts off at the same memory as ours at 60%, and a 90 GiB RAM
# budget (near its RAM-less-24-GB rule) with its defaults (lossy tail skip on).
cd ~/work/uma-moe/bench
L=~/work/uma-moe/.gpu.lock
while pgrep -f '[p]2_up_q8_b66_exact_mtp' > /dev/null; do sleep 10; done
flock $L ../.venv/bin/python p2/upstream.py p2_up_q8_b66_exact_nodraft 66.8 1024 nodraft ENV:STRATA_ROUTE_TAIL_SKIP=0 ENV:STRATA_PREFILL_CPU_SHARE=0 ENV:STRATA_DECODE_TIMING=1
flock $L ../.venv/bin/python p2/upstream.py p2_up_q8_b90_default_mtp 90 1024 mtp ENV:STRATA_DECODE_TIMING=1
