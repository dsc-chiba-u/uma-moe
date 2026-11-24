#!/bin/bash
# Page cache at engine start: the main configuration (60% residency, drafts off, code_en and essay_ja, 64 tokens) after
# dropping the page cache of the large files (E) and after reading 45 GB of other model files twice (F), as E1 F1 E2 F2.
# Prompted by two clean-room smoke tests that ran at 0.8-1.9 tokens/s while kswapd reclaimed memory during decoding;
# the A/B runs of results/compaction started with 35-37 GiB cached and never saw that. Hold the GPU lock while running.
set -u
cd ~/work/uma-moe/bench
R=~/work/uma-moe/results/pagecache
mkdir -p $R
ROOTS="$HOME/models $HOME/src/Strata-data $HOME/work/uma-moe-release $HOME/work/uma-moe/results"
FILL="$HOME/models/locality/mixtral-8x7b/Mixtral-8x7B-Instruct-v0.1.Q4_K_M.gguf $HOME/models/locality/glm-4.7-flash/GLM-4.7-Flash-Q4_K_M.gguf"
log() { echo "$(date '+%F %T') $*" | tee -a $R/runner.log; }
mem() { awk '/^(MemFree|Cached):/ {printf "%s %.1f GiB  ", $1, $2 / 1048576}' /proc/meminfo; }
taskset -c 14 ../.venv/bin/python -I p2/compaction_sampler.py $R/sampler.csv &
SAMPLER=$!
taskset -c 15 ../.venv/bin/python -I - $R/refaults.csv <<'PY' &
import sys, time
keys = ("workingset_refault_file", "workingset_refault_anon", "pgmajfault", "pgactivate", "pgdeactivate")
with open(sys.argv[1], "w") as f:
    f.write("t," + ",".join(keys) + "\n")
    while True:
        v = dict(l.split() for l in open("/proc/vmstat"))
        f.write(f"{time.time():.3f}," + ",".join(v.get(k, "0") for k in keys) + "\n"); f.flush()
        time.sleep(0.5)
PY
REF=$!
for c in E1 F1 E2 F2; do
  ../.venv/bin/python -I p2/evict_cache.py $ROOTS | tee -a $R/runner.log
  if [ "${c:0:1}" = F ]; then cat $FILL > /dev/null; cat $FILL > /dev/null; fi
  log "$c: $(mem)"
  ../.venv/bin/python p1/tier.py pc_${c}_t60_nodraft q8 1024 66.8 nodraft --exe ~/src/Strata/engine/strata-p1d \
    --prompts code_en,essay_ja --repeats 1 --max-tokens 64 ENV:STRATA_DECODE_TIMING=1
done
kill $SAMPLER $REF
log "done"
