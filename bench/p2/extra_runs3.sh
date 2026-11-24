#!/bin/bash
# P2 re-runs and a check of the host thread's sensitivity to CPU contention. The first f_ours_t75_mtp and
# f_ours_t60_nooverlap_mtp runs had stretches of 300-800 ms windows (GPU busy at half power) while other jobs ran on
# the machine (kept as x_*_contended). Strata pins its driver thread, which spins on the GPU's doorbells, to CPU 0.
# Waits for the process given as $1 (the previous queue) to exit.
cd ~/work/uma-moe/bench
while kill -0 "$1" 2>/dev/null; do sleep 30; done
L=~/work/uma-moe/.gpu.lock
T="--prompts code_en,story_en,explain_en,math_en,essay_ja,tech_ja --repeats 2 --max-tokens 256"
flock $L ../.venv/bin/python p1/tier.py f_ours_t75_mtp q8 1024 84.7 mtp --exe strata-p1d $T ENV:STRATA_DECODE_TIMING=1
flock $L ../.venv/bin/python p1/tier.py f_ours_t60_nooverlap_mtp q8 1024 66.8 mtp --exe strata-p1d $T ENV:STRATA_DECODE_TIMING=1 \
  ENV:STRATA_TIER_PREFETCH=0 ENV:STRATA_TIER_NO_OVERLAP=1
# one busy-looping process pinned to CPU 0 (the driver thread's core), then to CPU 19 (a big core, the control)
S="--prompts code_en,story_en,essay_ja --repeats 1 --max-tokens 256"
for c in 0 19; do
  taskset -c $c python3 -c 'while True: pass' &
  B=$!
  flock $L ../.venv/bin/python p1/tier.py x_burn_cpu${c}_t60_mtp q8 1024 66.8 mtp --exe strata-p1d $S ENV:STRATA_DECODE_TIMING=1
  kill $B
done
