#!/bin/bash
# P1: two cheap knobs at 45% residency (Q8_0): 32 I/O threads; the split verify window (MTP).
cd ~/work/uma-moe/bench
for m in nodraft mtp; do
  flock ~/work/uma-moe/.gpu.lock ../.venv/bin/python p1/tier.py p1d_q8_t45_io32_${m} q8 1024 48.8 $m --exe strata-p1d \
    ENV:STRATA_DECODE_TIMING=1 ENV:STRATA_VERIFY_PROFILE=1 ENV:STRATA_IO_THREADS=32
done
flock ~/work/uma-moe/.gpu.lock ../.venv/bin/python p1/tier.py p1d_q8_t45_split_mtp q8 1024 48.8 mtp --exe strata-p1d \
  ENV:STRATA_DECODE_TIMING=1 --spec-split
flock ~/work/uma-moe/.gpu.lock ../.venv/bin/python p1/tier.py p1d_q8_t45_base_mtp q8 1024 48.8 mtp --exe strata-p1d \
  ENV:STRATA_DECODE_TIMING=1 ENV:STRATA_VERIFY_PROFILE=1
