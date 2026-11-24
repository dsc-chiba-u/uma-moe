#!/bin/bash
# P1 preparation (run once): merge the Q8_0 shards into one GGUF, build the Q8_0 Strata pack with experts.bin, and a
# Q2_0 experts.bin for testing the expert tier on the model that fits in RAM.
# The packs are built with the tools/iq_pack.py of the p1d engine tree (README step 3), not with Strata's own: the
# unsloth Q8_0 file stores blk.1.ple_conv1d.weight as F32, the engine reads that tensor as F16, and only the p1d
# packer converts it (its F16_TENSORS rule); the v0.1.27 packer would store it as F32 (index kind 2), which the
# engine would read as F16 halves of F32 words. P1D names the p1d tree (default ~/src/Strata-build/p1d).
set -euo pipefail
M=~/models/unsloth-Qwen3.8-Flash-Next-GGUF
P1D=${P1D:-~/src/Strata-build/p1d}
PY=~/src/Strata/.venv/bin/python
[ -f "$P1D/tools/iq_pack.py" ] || { echo "no packer at $P1D/tools/iq_pack.py: build the p1d tree (README step 3) or set P1D" >&2; exit 1; }
grep -q "F16_TENSORS" "$P1D/tools/iq_pack.py" || { echo "$P1D/tools/iq_pack.py is not the p1d packer (no F16_TENSORS rule)" >&2; exit 1; }
cd ~/src/llama.cpp-master && nice -n 10 cmake --build build --target llama-gguf-split -j 8 2>&1 | tail -1
mkdir -p $M/Q8_0-merged
if [ ! -f $M/Q8_0-merged/Qwen3.8-Flash-Next-Q8_0.gguf ]; then
  /usr/bin/time -v ~/src/llama.cpp-master/build/bin/llama-gguf-split --merge \
    $M/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf $M/Q8_0-merged/Qwen3.8-Flash-Next-Q8_0.gguf 2>&1 | tail -25
fi
cd "$P1D"
/usr/bin/time -v $PY tools/iq_pack.py --gguf $M/Q8_0-merged/Qwen3.8-Flash-Next-Q8_0.gguf \
  --out ~/src/Strata-data/packs/q8_0 --compat-bf16 --experts-bin 2>&1 | tail -40
/usr/bin/time -v $PY tools/iq_pack.py \
  --gguf ~/src/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  --out ~/src/Strata-data/packs/q2_0-tier --experts-bin 2>&1 | tail -25
ls -la ~/src/Strata-data/packs/q8_0 ~/src/Strata-data/packs/q2_0-tier
# the PLE weight must be stored as F16 (index kind 5, 81,920 bytes for 4 x 10,240 values)
grep "blk.1.ple_conv1d.weight" ~/src/Strata-data/packs/q8_0/index.txt
