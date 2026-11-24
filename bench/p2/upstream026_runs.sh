#!/bin/bash
# P2 baseline, latest release (2026-10-26): upstream Strata 0.2.6 (tag v0.2.6, 4b9e05a2; its Qwen engine is unchanged since
# 0.2.5) built for GB10 with the same aarch64 arrangement as 0.2.2 (plus stubs for three AVX-512 entry points added after
# 0.2.2), on the 188 GB Q8_0, with the five configurations and the protocol of the paper's 0.2.2 rows: 6 short prompts x
# 2 repeats x 256 tokens, temperature 0; the 0.2.2 pack, MTP head and expert profile (formats unchanged since 0.2.2).
# Its conversation cache is off (--conversation-cache-mib 0): it parks each request's prompt state in RAM for an
# agent's next turn (up to 8 GiB here), which these independent requests never reuse. 0.2.2 has the same option with
# the default 0 (off); 0.2.6 sizes the cache from free RAM by default, and with it the 90 GiB budget ran out of memory
# (x_f_up026_b90_default_mtp_watchdog; the first two runs with the cache on are kept as
# x_f_up026_b66_exact_*_convcache). The runs wait for memory to come back after the previous engine exits.
# Then the unbuffered (O_DIRECT) file tier forced on at our memory: at this budget its rule (#286) picks the file cache,
# as in the 0.2.2 runs; at 90 GiB it switches to unbuffered reads by itself. The nearest the Qwen engine comes to the
# RAM store with direct SSD reads that 0.2.4 added to the DeepSeek engine.
cd ~/work/uma-moe/bench
L=~/work/uma-moe/.gpu.lock
P=code_en,story_en,explain_en,math_en,essay_ja,tech_ja
T="--prompts $P --repeats 2 --max-tokens 256"
EXACT="ENV:STRATA_ROUTE_TAIL_SKIP=0 ENV:STRATA_PREFILL_CPU_SHARE=0 ENV:STRATA_DECODE_TIMING=1"
NOCC="--conversation-cache-mib 0"
wait_mem() {   # until MemAvailable (GiB) reaches $1, at most 10 minutes
  for _ in $(seq 60); do
    [ "$(awk '/^MemAvailable:/ {print int($2 / 1048576)}' /proc/meminfo)" -ge "$1" ] && return 0
    sleep 10
  done
  echo "memory did not come back to $1 GiB"; return 1
}
run() { wait_mem "$1" && flock $L ../.venv/bin/python p2/upstream026.py "${@:2}"; }
for m in mtp nodraft; do
  run 100 f_up026_b66_exact_${m} 66.8 1024 $m $T $EXACT $NOCC
done
for m in mtp nodraft; do
  run 100 f_up026_b66_exact_unbuf_${m} 66.8 1024 $m $T $EXACT ENV:STRATA_UNBUFFERED_LOAD=1 $NOCC
done
run 108 f_up026_b90_default_mtp 90 1024 mtp $T ENV:STRATA_DECODE_TIMING=1 $NOCC
run 108 f_up026_b90_exact_mtp 90 1024 mtp $T $EXACT $NOCC
run 100 f_up026_b66_default_mtp 66.8 1024 mtp $T ENV:STRATA_DECODE_TIMING=1 $NOCC
