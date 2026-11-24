#!/bin/bash
# vm.compaction_proactiveness: A/B/A/B/A over three configurations, at the original value (A) and at 0 (B).
# Only compaction_root.sh (run as root) changes the value: it saves the original, writes "armed" here,
# sets 0 when this script writes ready_B<n>, and restores the original when this script writes done_B<n>. This script
# starts measuring once "armed" exists and holds the GPU lock through all five phases, so no other job runs between
# them. Started unpinned, like the paper's runs, so the engine's host thread takes CPU 0 as there; the sampler runs on
# CPU 14.
set -u
cd ~/work/uma-moe/bench
D=~/work/uma-moe/results/compaction
K=/proc/sys/vm/compaction_proactiveness
R=$D/run1
mkdir -p $R
log() { echo "$(date '+%F %T') $*" | tee -a $R/runner.log; }
wait_value() {   # $1 = value, $2 = timeout in s
  local n=0
  until [ "$(cat $K)" = "$1" ]; do sleep 2; n=$((n + 2)); [ $n -ge $2 ] && return 1; done
  return 0
}
phase() {        # $1 = phase name
  log "phase $1 begins: compaction_proactiveness=$(cat $K)"
  ../.venv/bin/python p1/tier.py k_$1_t60_hostalloc_nodraft q8 1024 66.8 nodraft --exe strata-p1d $T --repeats 4 \
    ENV:STRATA_DECODE_TIMING=1 ENV:STRATA_ARENA_HOSTALLOC=1
  ../.venv/bin/python p1/tier.py k_$1_t60_mtp q8 1024 66.8 mtp --exe strata-p1d $T --repeats 2 ENV:STRATA_DECODE_TIMING=1
  ../.venv/bin/python p1/tier.py k_$1_t75_mtp q8 1024 84.7 mtp --exe strata-p1d $T --repeats 2 ENV:STRATA_DECODE_TIMING=1
  log "phase $1 ends: compaction_proactiveness=$(cat $K)"
}

until [ -f $D/armed ]; do sleep 10; done
ORIG=$(cat $D/proactiveness_before.txt)
log "armed by the root script; original value $ORIG"
exec 9>~/work/uma-moe/.gpu.lock
flock 9
log "GPU lock taken"
taskset -c 14 ../.venv/bin/python -I p2/compaction_sampler.py $R/sampler.csv &
SAMPLER=$!
T="--prompts code_en,story_en,explain_en,math_en,essay_ja,tech_ja --max-tokens 256"

phase A1
for b in 1 2; do
  touch $D/ready_B$b
  if wait_value 0 900; then phase B$b; else log "B$b skipped: the value did not change to 0 within 15 min"; fi
  touch $D/done_B$b
  wait_value $ORIG 900 || log "WARNING: the value was not restored to $ORIG within 15 min"
  phase A$((b + 1))
done
kill $SAMPLER
log "all phases done: compaction_proactiveness=$(cat $K)"
