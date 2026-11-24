#!/bin/bash
# After the pipeline: extra microbenchmarks in clean conditions, mmap read-ahead variants, and the E7 profiles.
set -u
ROOT=~/work/uma-moe; R=$ROOT/results; B=$ROOT/bench; P=$R/pipeline
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $P/pipeline.log; }
Q8DIR=~/models/unsloth-Qwen3.8-Flash-Next-GGUF/Q8_0; F3=$(ls $Q8DIR/*00003-of-*.gguf)
Q2=~/src/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf

while pgrep -f 'bench/pipeline.sh|\./pipeline.sh' >/dev/null; do sleep 15; done
log "post: start"
{
  for a in dev pinned registered pageable file:$Q2; do for k in 10 40; do $B/uma_bw gpu-gather $a 16 1350 $k 3000; done; done
  for t in 1 5 10; do $B/uma_bw cpu-seq $t 8 big; done; $B/uma_bw cpu-seq 10 8 little; $B/uma_bw cpu-seq 20 8 all
  $B/uma_bw concurrent none 4 20 all 6; for t in 5 10 20; do $B/uma_bw concurrent dev 4 $t all 6; done
} > $P/post_e1_clean.log 2>&1
log "post: clean E1 rerun done"
{
  for adv in normal seq; do for t in 1 8 32; do $B/nvme_bench_v2 mmap $F3 1700 $t 1500 $adv; done; done
} > $P/post_mmap.log 2>&1
log "post: mmap read-ahead variants done"
bash $B/exp_profile.sh > $P/post_profile.log 2>&1
log "post: E7 profiles done"
log "post finished"
