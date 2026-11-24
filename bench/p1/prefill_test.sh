#!/bin/bash
# P1: Q8_0 prompt reading with the tier (60%): the 5.4K and 18.9K-token prompts, bypass (default) vs admission.
cd ~/work/uma-moe/bench
L=~/work/uma-moe/.gpu.lock
flock $L ../.venv/bin/python p1/tier.py p1d_q8_t60_prefill_bypass q8 1024 66.8 mtp --exe strata-p1d --prompts long5k,long19k ENV:STRATA_DECODE_TIMING=1
flock $L ../.venv/bin/python p1/tier.py p1d_q8_t60_prefill_admit q8 1024 66.8 mtp --exe strata-p1d --prompts long5k,long19k ENV:STRATA_DECODE_TIMING=1 ENV:STRATA_TIER_PREFILL_ADMIT=1
