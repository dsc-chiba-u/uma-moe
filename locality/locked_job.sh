#!/bin/bash
# Run "$@" under the shared GPU lock, but never while the main session has reserved the machine:
# wait (polling every 60 s) while ~/work/uma-moe/.main-reserved exists, take the lock, and re-check the reservation
# after acquiring it (exit 76 inside the lock = reserved meanwhile -> release and wait again).
RES=$HOME/work/uma-moe/.main-reserved
LOCK=$HOME/work/uma-moe/.gpu.lock
while true; do
  while [ -e "$RES" ]; do sleep 60; done
  flock "$LOCK" bash -c '[ -e "$0" ] && exit 76; exec "$@"' "$RES" "$@"
  rc=$?
  [ $rc -ne 76 ] && exit $rc
  echo "[$(date +%T)] machine reserved by the main session; waiting"
done
