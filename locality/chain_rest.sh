#!/bin/bash
# After the Qwen3.6 trace: re-trace GLM-4.7-Flash with its [gMASK]<sop> prefix, then download Mixtral (run_all.sh
# picks Mixtral up once the file exists). Each step is one locked job behind the main-session reservation check.
cd ~/work/uma-moe
until grep -q -E "model done: qwen3.6-35b-a3b|trace failed: qwen3.6-35b-a3b|gen failed: qwen3.6-35b-a3b" locality/logs/run_all.log; do sleep 15; done
while true; do
  echo "[$(date +%T)] waiting for lock: glm-4.7-flash trace (with [gMASK]<sop>)"
  locality/locked_job.sh .venv/bin/python -I locality/run_model.py glm-4.7-flash trace; rc=$?
  echo "[$(date +%T)] released lock (rc=$rc): glm-4.7-flash trace"
  [ $rc -ne 75 ] && break
  sleep 300
done
cd locality && exec ./dl_all.sh mixtral-8x7b
