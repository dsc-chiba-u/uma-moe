#!/bin/bash
# Start the Mixtral download only after the Qwen3.6 trace is finished (priority: GLM, Qwen3.6, then Mixtral).
cd ~/work/uma-moe/locality
until grep -q -E "model done: qwen3.6-35b-a3b|trace failed: qwen3.6-35b-a3b|gen failed: qwen3.6-35b-a3b" logs/run_all.log; do sleep 15; done
exec ./dl_all.sh mixtral-8x7b
