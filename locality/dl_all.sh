#!/bin/bash
# Download the study's models one at a time, each under the shared lock (large downloads) and only while the
# main session has not reserved the machine (locked_job.sh).
cd ~/work/uma-moe/locality
for key in "$@"; do
  echo "[$(date +%T)] waiting for lock: $key"
  ./locked_job.sh nice -n 10 ~/work/uma-moe/.venv/bin/python -I download.py "$key"
  echo "[$(date +%T)] released lock: $key (rc=$?)"
done
