#!/bin/bash
# Run the per-model pipeline (gen -> tok -> trace) for the given model keys, one locked GPU job at a time.
# A job that finds too little free memory exits 75; that model is retried in a later round (the others go on);
# rounds without progress sleep 5 minutes without holding the lock.
cd ~/work/uma-moe
PY=~/work/uma-moe/.venv/bin/python
LOCK=~/work/uma-moe/.gpu.lock
R=results/locality
locked() {
  echo "[$(date +%T)] waiting for lock: $*"
  locality/locked_job.sh "$@"; local rc=$?
  echo "[$(date +%T)] released lock (rc=$rc): $*"
  return $rc
}
pending=("$@")
while [ ${#pending[@]} -gt 0 ]; do
  next=(); progressed=0
  for key in "${pending[@]}"; do
    gguf=$($PY -I -c "import sys; sys.path.insert(0, 'locality'); from models import gguf_path; print(gguf_path('$key'))")
    if [ ! -f "$gguf" ]; then next+=("$key"); continue; fi
    if [ ! -f $R/$key/gen/gen.ok ]; then
      locked $PY -I locality/run_model.py $key gen; rc=$?
      if [ $rc -eq 75 ]; then next+=("$key"); continue; fi
      if [ $rc -ne 0 ]; then echo "gen failed: $key"; continue; fi
      progressed=1
    fi
    if [ ! -f $R/$key/tokens/docs.json ] || [ $R/$key/gen/gen.ok -nt $R/$key/tokens/docs.json ]; then
      $PY -I locality/run_model.py $key tok || { echo "tok failed: $key"; continue; }
    fi
    if [ ! -f $R/$key/trace/trace_meta.json ]; then
      locked $PY -I locality/run_model.py $key trace; rc=$?
      if [ $rc -eq 75 ]; then next+=("$key"); continue; fi
      if [ $rc -ne 0 ]; then echo "trace failed: $key"; continue; fi
      progressed=1
    fi
    echo "[$(date +%T)] model done: $key"
  done
  pending=("${next[@]}")
  if [ ${#pending[@]} -gt 0 ] && [ $progressed -eq 0 ]; then
    echo "[$(date +%T)] pending: ${pending[*]} (sleeping 300 s)"; sleep 300
  fi
done
echo "[$(date +%T)] all done"
