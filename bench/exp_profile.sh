#!/bin/bash
# E7: where does a decode step's time go? Nsight Systems on llama-server (no MTP and MTP n=3) and on the Strata
# engine (drafts off and MTP), one short request each, then the kernel summary and GPU busy fraction.
set -u
ROOT=~/work/uma-moe; R=$ROOT/results/e7_profile; mkdir -p $R
PY=$ROOT/.venv/bin/python
LB=~/src/llama.cpp-master/build/bin
Q2=~/src/Strata-data/models/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
MTP=~/models/unsloth-Qwen3.8-Flash-Next-GGUF/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf

wait_health() { for i in $(seq 1 300); do curl -sf http://127.0.0.1:8080/health >/dev/null && return 0; sleep 2; done; return 1; }
one_request() {   # $1 engine (strata|llama)
  (cd $ROOT/bench && $PY -c "
import harness as H, prompts as P, json
msgs, _ = P.messages('code_en'); r = H.request('$1', msgs, 256)
print(json.dumps({k: r[k] for k in ('completion_tokens', 'decode_tok_s_client', 'ttft_s')}))")
}

profile_llama() {   # $1 name, rest: llama-server args
  local name=$1; shift
  nsys profile -t cuda,osrt --cuda-graph-trace=node -o $R/$name -f true --wait=primary \
    $LB/llama-server --port 8080 --host 127.0.0.1 -m $Q2 -ngl 999 -fa on -c 8192 --no-warmup "$@" > $R/$name.server.log 2>&1 &
  local pid=$!
  wait_health || { echo "$name: server not ready"; kill $pid; return; }
  one_request llama > $R/$name.request.json   # warm-up, also measured
  one_request llama >> $R/$name.request.json
  kill -INT $pid; wait $pid
}

profile_llama llama_nomtp
profile_llama llama_mtp3 -md $MTP --spec-type draft-mtp -ngld 999 --spec-draft-n-max 3

# Strata: profile the engine process tree through server.py
profile_strata() {   # $1 name, $2 mtp (1|0)
  local name=$1 mtp=$2
  (cd $ROOT/bench && $PY -c "
import harness as H, json
args = H.strata_args('auto', mtp=bool($mtp))
cfg = json.load(open(H.STRATA_CFG)); cfg['args'] = args; cfg['log'] = '$R/$name.engine.log'
json.dump(cfg, open('$R/$name.config.json', 'w'), indent=1)")
  nsys profile -t cuda,osrt --cuda-graph-trace=node -o $R/$name -f true --wait=all \
    ~/src/Strata/.venv/bin/python ~/src/Strata/serve/server.py --engine strata --config $R/$name.config.json --port 8080 \
    > $R/$name.server.log 2>&1 &
  local pid=$!
  for i in $(seq 1 300); do grep -q 'ready: http' $R/$name.server.log 2>/dev/null && break; sleep 2; done
  one_request strata > $R/$name.request.json
  one_request strata >> $R/$name.request.json
  pkill -INT -P $pid; kill -INT $pid; wait $pid
}
profile_strata strata_nodraft 0
profile_strata strata_mtp 1

for f in $R/*.nsys-rep; do
  nsys stats -r cuda_gpu_kern_sum,cuda_api_sum -f csv -o ${f%.nsys-rep} $f > /dev/null 2>&1
done
echo "profiles in $R"
