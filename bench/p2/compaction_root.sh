#!/bin/bash
# Root side of the vm.compaction_proactiveness A/B (compaction_ab.sh, compaction_ab2.sh): sets the value 0 for the two
# measurement blocks B1 and B2 and the original value otherwise.
# It saves the current value first and restores it after each block, on exit, on Ctrl-C or a closed terminal, and
# after a timeout. If the measurements report that there is nothing to compare ("abort"), it exits without changing
# anything. Nothing is written to /etc, so a reboot also restores the default. Run it inside tmux and leave it
# open: it may wait several hours for the GPU, then takes about 1.5 hours.
[ "$(id -u)" -eq 0 ] || exec sudo bash "$0" "$@"
set -u
D="$(cd "$(dirname "$0")/../.." && pwd)/results/compaction" || exit 1
K=/proc/sys/vm/compaction_proactiveness
mkdir -p "$D"
ORIG=$(cat "$K")
case "$ORIG" in ''|*[!0-9]*|0) echo "unexpected current value '$ORIG'; nothing changed"; exit 1;; esac
echo "$ORIG" > "$D/proactiveness_before.txt"
restore() { echo "$ORIG" > "$K"; echo "$(date +%T) restored: compaction_proactiveness=$(cat "$K")"; }
trap restore EXIT
trap 'exit 1' INT TERM HUP
waitfile() {     # 0 = the file appeared, 2 = the measurements aborted, 1 = timeout
  local n=0
  until [ -e "$1" ]; do
    [ -e "$D/abort" ] && return 2
    sleep 5; n=$((n + 5)); [ "$n" -ge "$2" ] && return 1
  done
  return 0
}
touch "$D/armed"
chown "${SUDO_USER:-root}": "$D" "$D/armed" "$D/proactiveness_before.txt"
echo "$(date +%T) saved the original value $ORIG to $D/proactiveness_before.txt"
for b in 1 2; do
  echo "$(date +%T) waiting for the measurements to reach block B$b ..."
  waitfile "$D/ready_B$b" 28800
  case $? in
    1) echo "the measurements did not reach B$b within 8 hours"; exit 1;;
    2) echo "$(date +%T) the measurements found nothing to compare; the setting was not changed"; exit 0;;
  esac
  echo 0 > "$K"
  echo "$(date +%T) block B$b: compaction_proactiveness=$(cat "$K")"
  waitfile "$D/done_B$b" 7200 || { echo "block B$b did not finish within 2 hours"; exit 1; }
  restore
done
echo "$(date +%T) finished; this terminal can be closed"
