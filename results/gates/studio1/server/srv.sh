#!/bin/bash
# usage: DRAFT=0|1 TAG=x bash srv.sh   (under gpu-run)
cd ~/.exo-inkling-bench/exact-work/s1/short || exit 1
PY=../../../venv-exact/bin/python; OV=$PWD/../${STK:-stack13}/ov
M=$HOME/.exo-inkling-bench/models/thinkingmachines--Inkling-2-Mini
PORT=${PORT:-52511}; URL=http://127.0.0.1:$PORT
E="MLX_VLM_EXACT_PREFILL_EVAL_EVERY=0 MLX_VLM_INKLING2_PREFILL_FUSED=1 MLX_VLM_INKLING2_DECODE_MIN_KEYS=1024"
if [ "${DRAFT:-1}" = 1 ]; then
  E="$E MLX_VLM_DRAFT_MODEL=$M MLX_VLM_DRAFT_BLOCK_SIZE=8 MLX_VLM_INKLING2_INCREMENTAL_DRAFT_PREFILL=1 MLX_VLM_INKLING2_EXACT_VERIFY=1 MLX_VLM_INKLING2_COMPILED_VERIFY=1 MLX_VLM_INKLING2_DRAFT_CTX_WINDOW=65536 MLX_VLM_INKLING2_DRAFT_CONF=${TAU:-0.5} MLX_VLM_INKLING2_DRAFT_MIN_CTX=1024 MLX_VLM_INKLING2_DRAFT_MAX_CTX=262144"
fi
START="import mlx.core as mx; lim = mx.device_info()['max_recommended_working_set_size']; mx.set_memory_limit(lim); mx.set_wired_limit(lim); mx.set_cache_limit(1024**3); from mlx_vlm.server.cli import main; main()"
LOG=srv-$TAG.log; : > $LOG
env PYTHONPATH=$OV HF_HUB_OFFLINE=1 $E $XENV $PY -u -c "$START" --host 127.0.0.1 --port $PORT --model-dir $M \
  --prefill-step-size 4096 --max-num-seqs 1 --max-tokens 8192 --log-level INFO >> $LOG 2>&1 &
SP=$!
for i in $(seq 600); do curl -sf -o /dev/null $URL/health && break; kill -0 $SP 2>/dev/null || { tail -30 $LOG; exit 1; }; sleep 1; done
for c in ${CTXS:-4096 32768}; do for r in 1 2; do
  $PY ../../../mlxserve/bench_client.py --url $URL/v1/chat/completions --model $M --prompt ../../../mlxserve/prompts/user-$c.txt --max-tokens 256 --tag $TAG-$c-$r --out $PWD/srv-$TAG-$c-$r.json >/dev/null 2>&1
  python3 -c "import json;d=json.load(open('srv-$TAG-$c-$r.json'));print('$TAG',$c,$r,{k:d.get(k) for k in ('prompt_tokens','completion_tokens','prompt_tps','decode_tps','needle_found','error')})"
done; done
grep -iE "tok/s|spec|draft|exact|Traceback|Error" $LOG | tail -20
kill $SP; wait $SP 2>/dev/null
echo SRVDONE
