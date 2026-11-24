#!/bin/bash
# CPU only (nice 19): whenever a model's trace_meta.json is newer than its analysis JSON, run analyze.py for it and
# regenerate results/locality/analysis/tables.md. Exits once no locality GPU job is queued and nothing is left.
cd ~/work/uma-moe
LOG=locality/logs/watch_refresh.log
while true; do
  changed=0
  for meta in results/locality/*/trace/trace_meta.json; do
    key=$(basename "$(dirname "$(dirname "$meta")")")
    out=results/locality/analysis/$key.json
    if [ ! -f "$out" ] || [ "$meta" -nt "$out" ]; then
      echo "[$(date +%T)] analyzing $key" >> $LOG
      LOC_TMP=$HOME/work/uma-moe/locality/tmp nice -n 19 .venv/bin/python -I locality/analyze.py "$key" >> $LOG 2>&1 && changed=1
    fi
  done
  if [ $changed -eq 1 ]; then
    nice -n 19 .venv/bin/python -I locality/report.py > results/locality/analysis/tables.md 2>> $LOG
    echo "[$(date +%T)] tables.md refreshed" >> $LOG
  fi
  if ! pgrep -f 'run_all.sh' > /dev/null && ! pgrep -f 'chain_rest.sh' > /dev/null && [ $changed -eq 0 ]; then
    echo "[$(date +%T)] no queued locality jobs left; watcher exits" >> $LOG
    break
  fi
  sleep 300
done
