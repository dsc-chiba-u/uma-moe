#!/bin/bash
# E6: quality. Perplexity on wikitext-2 (test) for Q2_0 (GSQ-RCO, all on GPU) and Q8_0 (overflow, experts mmap'd),
# and the KL divergence of Q2_0 against Q8_0 on the same tokens. 8 chunks of 2048 tokens. The Q8_0 run computes the
# experts on the CPU straight from the page cache (--no-op-offload): with op offload llama.cpp copies every expert to
# the GPU for each batch through mmap page faults, which ran at ~3 tok/s in E5.
set -e
B=~/src/llama.cpp-master/build/bin
R=~/work/uma-moe/results/e6_quality; mkdir -p $R
W=~/work/uma-moe/bench/data/wikitext/wikitext-2-raw/wiki.test.raw
Q8=$(ls ~/models/unsloth-Qwen3.8-Flash-Next-GGUF/Q8_0/*00001-of-*.gguf)
Q2=~/src/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
COMMON="-f $W -c 2048 --chunks 8 -fa on -b 2048 -ub 2048"
if [ ! -s $R/q8_logits.bin ]; then
  $B/llama-perplexity -m $Q8 $COMMON -ngl 999 --cpu-moe -lm mmap --no-op-offload --kl-divergence-base $R/q8_logits.bin > $R/q8_ppl.log 2>&1
fi
$B/llama-perplexity -m $Q2 $COMMON -ngl 999 --kl-divergence-base $R/q8_logits.bin --kl-divergence > $R/q2_vs_q8_kld.log 2>&1
grep -E 'Final estimate|PPL|KLD|Same top|RMS' $R/q8_ppl.log $R/q2_vs_q8_kld.log | tail -30
