#!/bin/bash
# Build the locality tools. moe_trace links the snapshot of llama.cpp master 339f324 libraries in llama-bin/.
set -e
cd "$(dirname "$0")"
LL=$HOME/src/llama.cpp-master
nice -n 19 g++ -O2 -std=c++17 -pthread -DLLAMA_COMMIT='"339f324"' -I$LL/include -I$LL/ggml/include moe_trace.cpp \
    -o moe_trace -L$PWD/llama-bin -lllama -lggml -lggml-base -Wl,-rpath,$PWD/llama-bin
nice -n 19 g++ -O3 -march=native -std=c++17 -pthread locsim.cpp -o locsim
