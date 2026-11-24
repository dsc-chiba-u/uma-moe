#!/bin/bash
# vm.compaction_proactiveness A/B/A/B/A, second run. The first run (compaction_ab.sh, results/compaction/run1) saw
# kcompactd only in its first block: that block started with about 16 GiB more page cache, so free memory during the
# cudaHostAlloc decode was about 1 GiB and fragmented (2 MB fragmentation score 97-100, above the proactive-compaction
# threshold of 90); later blocks had 17 GiB free (score 1-4), whatever the setting. Here every configuration starts
# with the page cache filled by reading a 27 GB file. Block A1 is checked first: if kcompactd runs less than 30 s
# during its cudaHostAlloc decode, the state was not reproduced; the script then writes "abort" (the root script exits
# without changing the setting) and stops. Otherwise the same handshake as compaction_ab.sh: ready_B<n> / done_B<n>.
set -u
cd ~/work/uma-moe/bench
D=~/work/uma-moe/results/compaction
R=$D/run2
K=/proc/sys/vm/compaction_proactiveness
FILL=~/models/locality/mixtral-8x7b/Mixtral-8x7B-Instruct-v0.1.Q4_K_M.gguf
mkdir -p $R
log() { echo "$(date '+%F %T') $*" | tee -a $R/runner.log; }
wait_value() {   # $1 = value, $2 = timeout in s
  local n=0
  until [ "$(cat $K)" = "$1" ]; do sleep 2; n=$((n + 2)); [ $n -ge $2 ] && return 1; done
  return 0
}
fill() {
  cat $FILL > /dev/null
  log "  page cache filled: $(awk '/^(MemFree|Cached):/ {printf "%s %.1f GiB  ", $1, $2 / 1048576}' /proc/meminfo)"
}
phase() {        # $1 = phase name
  log "phase $1 begins: compaction_proactiveness=$(cat $K)"
  fill
  ../.venv/bin/python p1/tier.py k2_$1_t60_hostalloc_nodraft q8 1024 66.8 nodraft --exe strata-p1d $T --repeats 4 \
    ENV:STRATA_DECODE_TIMING=1 ENV:STRATA_ARENA_HOSTALLOC=1
  fill
  ../.venv/bin/python p1/tier.py k2_$1_t60_mtp q8 1024 66.8 mtp --exe strata-p1d $T --repeats 2 ENV:STRATA_DECODE_TIMING=1
  fill
  ../.venv/bin/python p1/tier.py k2_$1_t75_mtp q8 1024 84.7 mtp --exe strata-p1d $T --repeats 2 ENV:STRATA_DECODE_TIMING=1
  log "phase $1 ends: compaction_proactiveness=$(cat $K)"
}
kcompactd_in_decode() {   # kcompactd0 CPU seconds between the first token of the first request and the last request's end
  ../.venv/bin/python -I - "$R/sampler.csv" ~/work/uma-moe/results/$1/requests.jsonl <<'EOF'
import csv, json, sys
rows = list(csv.DictReader(open(sys.argv[1])))
reqs = [json.loads(l) for l in open(sys.argv[2])]
t0, t1 = reqs[0]["t_first"], reqs[-1]["t_end"]
w = [int(r["kcompactd_ticks"]) for r in rows if t0 <= float(r["t"]) <= t1]
print(f"{(w[-1] - w[0]) / 100:.0f}" if len(w) > 1 else "0")
EOF
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
KC=$(kcompactd_in_decode k2_A1_t60_hostalloc_nodraft)
log "A1 check: kcompactd ran ${KC} s during the cudaHostAlloc decode"
if [ "$KC" -lt 30 ]; then
  log "the state of the first run was not reproduced; stopping before any B block"
  touch $D/abort
  kill $SAMPLER
  exit 0
fi
for b in 1 2; do
  touch $D/ready_B$b
  if wait_value 0 900; then phase B$b; else log "B$b skipped: the value did not change to 0 within 15 min"; fi
  touch $D/done_B$b
  wait_value $ORIG 900 || log "WARNING: the value was not restored to $ORIG within 15 min"
  phase A$((b + 1))
done
kill $SAMPLER
log "all phases done: compaction_proactiveness=$(cat $K)"
