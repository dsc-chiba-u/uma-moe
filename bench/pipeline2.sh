#!/bin/bash
# Rerun of the steps the first pipeline could not finish (E5 needed -lm mmap), after post.sh is done.
set -u
ROOT=~/work/uma-moe; R=$ROOT/results; B=$ROOT/bench; PY=$ROOT/.venv/bin/python; P=$R/pipeline
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $P/pipeline.log; }
while pgrep -f '[.]/post[.]sh' >/dev/null; do sleep 15; done
mkdir -p $R/e5_failed_nommap && mv $R/e5_llama_q8_cpumoe* $R/e5_failed_nommap/ 2>/dev/null
(cd $B && $PY exp_overflow.py) > $P/e5.log 2>&1;       log "E5 (mmap) done"
(cd $B && ./exp_quality.sh) > $P/e6.log 2>&1;          log "E6 done"
(cd $B && $PY exp_llama_cache.py) > $P/e4b.log 2>&1;   log "E4b done"
log "pipeline2 finished"
