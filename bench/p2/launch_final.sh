#!/bin/bash
# wait for the queued P2 upstream runs and the P1 prefill test, then reserve the machine and run the final comparison
cd ~/work/uma-moe/bench
while pgrep -f '[u]pstream_runs2.sh' > /dev/null || pgrep -f '[p]1/prefill_test.sh' > /dev/null; do sleep 20; done
touch ~/work/uma-moe/.main-reserved
p2/final_runs.sh
