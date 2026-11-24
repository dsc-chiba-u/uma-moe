#!/bin/bash
# P0: expert-cache size sweep with the fixed miss path (UMA defaults: direct, pcie_frac 1.0)
cd ~/work/uma-moe/bench
for c in 1024 4096 8192 16384; do
  for m in mtp nodraft; do
    ../.venv/bin/python p0/ab.py p0_sweep_c${c}_${m} strata-p0 $c $m
  done
done
