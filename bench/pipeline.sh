#!/bin/bash
# Runs the remaining experiments one after another, each alone on the machine.
# Progress: results/pipeline/pipeline.log; per-step output: results/pipeline/<step>.log
set -u
ROOT=~/work/uma-moe; R=$ROOT/results; B=$ROOT/bench; PY=$ROOT/.venv/bin/python
P=$R/pipeline; mkdir -p $P
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $P/pipeline.log; }
Q2=~/src/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
Q8DIR=~/models/unsloth-Qwen3.8-Flash-Next-GGUF/Q8_0

log "waiting for the E4 sweep"
while pgrep -f exp_strata_cache.py >/dev/null; do sleep 10; done
log "E4 sweep done"
(cd $B && $PY exp_strata_cache.py e4_strata_c1024_mtp_pcie1) > $P/e4_extra.log 2>&1
log "E4 extra done"

# ---- E1 (GPU idle): launch/doorbell latency, working-set size per allocation path, huge pages
{
  $B/uma_bw launch
  for a in dev pinned registered pageable managed; do for g in 0.25 1 4 16; do $B/uma_bw gpu-seq $a $g; done; done
  $B/uma_bw gpu-seq pageable-thp 4
  $B/uma_bw gpu-seq file:$Q2 4
  $B/uma_bw gpu-seq filethp:$Q2 4
} > $P/e1_gpu.log 2>&1
log "E1 GPU microbenchmarks done"

log "waiting for the Q8_0 download"
while pgrep -f 'hf download' >/dev/null; do sleep 15; done
ls -la $Q8DIR > $P/q8_files.txt
log "download done: $(du -sh $Q8DIR | cut -f1)"

# ---- E1 (NVMe idle): sequential, random O_DIRECT, mmap faults
F3=$(ls $Q8DIR/*00003-of-*.gguf)
{
  for t in 1 4 16; do $B/nvme_bench seq $F3 1024 $t 8; done
  for qd in 1 8 32; do $B/nvme_bench rand $F3 4 $qd 20000; done
  for blk in 128 512 1700 5120; do for qd in 1 8 32 64; do
    n=$(( 6000000 / blk )); $B/nvme_bench rand $F3 $blk $qd $n; done; done
  for t in 1 8 32; do $B/nvme_bench mmap $F3 1700 $t 1500 none; $B/nvme_bench mmap $F3 1700 $t 1500 willneed; done
} > $P/e1_nvme.log 2>&1
log "E1 NVMe microbenchmarks done"

# ---- E1: NVMe DMA vs GPU reads on the same memory controller
{
  echo "== GPU alone"; $B/uma_bw gpu-seq dev 16
  echo "== NVMe alone"; $B/nvme_bench seq $F3 1024 16 30
  echo "== GPU while NVMe streams"; $B/nvme_bench seq $F3 1024 16 50 & NP=$!; sleep 2; $B/uma_bw gpu-seq dev 16; wait $NP
  echo "== NVMe while GPU streams"; $B/uma_bw concurrent dev 4 0 all 25 & GP=$!; sleep 3; $B/nvme_bench seq $F3 1024 16 30; wait $GP
} > $P/e1_contention.log 2>&1
log "E1 contention done"

# ---- E8: trace replay against the Q8_0 experts on NVMe
$PY $ROOT/analysis/expert_map.py $R/maps/q8_0.map $(ls $Q8DIR/*.gguf) > $P/e8_map.log 2>&1
{
  for res in 0.25 0.33 0.45 0.60 0.75; do
    cap=$($PY -c "print(int($res*24576))")
    for pol in lru belady; do $B/replay_io $R/maps/q8_0.map $R/maps/trace_nodraft.i32 1000:1000 $cap $pol token 32; done
    $B/replay_io $R/maps/q8_0.map $R/maps/trace_nodraft.i32 1000:1000 $cap static token 32 $R/maps/rank_oracle.i64
    $B/replay_io $R/maps/q8_0.map $R/maps/trace_nodraft.i32 1000:1000 $cap lru layer 32
  done
  cap=11059
  for qd in 4 8 16 64; do $B/replay_io $R/maps/q8_0.map $R/maps/trace_nodraft.i32 1000:1000 $cap lru token $qd; done
  for res in 0.25 0.33 0.45; do cap=$($PY -c "print(int($res*24576))")
    for pol in lru belady; do $B/replay_io $R/maps/q2_0.map $R/maps/trace_nodraft.i32 1000:1000 $cap $pol token 32; done; done
} > $P/e8_replay.log 2>&1
log "E8 replay done"

# ---- E2, E5, E6, E4b
(cd $B && $PY exp_reproduce.py) > $P/e2.log 2>&1;      log "E2 done"
(cd $B && $PY exp_overflow.py) > $P/e5.log 2>&1;       log "E5 done"
(cd $B && ./exp_quality.sh) > $P/e6.log 2>&1;          log "E6 done"
(cd $B && $PY exp_llama_cache.py) > $P/e4b.log 2>&1;   log "E4b done"
log "pipeline finished"
