#!/bin/bash
# Rerun the release's clean-room smoke test on a calm machine: the clean-room binary, then the paper's binary, with the
# compaction sampler logging MemFree, kcompactd and kswapd. Run under the GPU lock.
set -u
C=~/work/uma-moe-release/cleanroom
S=/tmp/scratch
taskset -c 14 ~/work/uma-moe/.venv/bin/python -I ~/work/uma-moe/bench/p2/compaction_sampler.py $S/smoke_again_sampler.csv &
SAMPLER=$!
cd $C/uma-moe/bench
for exe in cleanroom:$C/Strata/build/strata paper:$HOME/src/Strata/engine/strata-p1d; do
  for m in nodraft mtp; do
    echo "$(date +%T) ${exe%%:*} $m  $(awk '/^(MemFree|Cached):/ {printf "%s %.1f GiB  ", $1, $2 / 1048576}' /proc/meminfo)"
    $C/venv/bin/python p1/tier.py smoke2_${exe%%:*}_t60_$m q8 1024 66.8 $m --exe ${exe#*:} --prompts code_en,essay_ja \
      --repeats 1 --max-tokens 64 ENV:STRATA_DECODE_TIMING=1 > $S/smoke2_${exe%%:*}_$m.out 2>&1
  done
done
kill $SAMPLER
echo "$(date +%T) done"
