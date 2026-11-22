#!/bin/bash
# Drafted writing on stack12 (stack8 + 0010), Studio 2. usage: CTXS="4096 32768" KS="4 5 8" STEP=4096 NOHASH=1 ./ds.sh (under ../gpu-run.sh)
cd ~/.exo-inkling-bench/exact-work/short || exit 1
PY=../../venv-exact/bin/python; OV=$PWD/../${STK:-stack13}/ov
E="${BASEENV:-MLX_VLM_EXACT_PREFILL_EVAL_EVERY=0 MLX_VLM_INKLING2_PREFILL_FUSED=1} MLX_VLM_INKLING2_INCREMENTAL_DRAFT_PREFILL=1 MLX_VLM_INKLING2_EXACT_VERIFY=1 MLX_VLM_INKLING2_COMPILED_VERIFY=1 MLX_VLM_INKLING2_DRAFT_CTX_WINDOW=65536 ${XENV}"
for c in ${CTXS:-4096 32768}; do for k in ${KS:-4 5 8}; do
 T=${TAG:-ds}-$c-k$k
 echo "$(date +%T) ctx $c k $k step ${STEP:-4096} $XENV"
 env PYTHONPATH=$OV $E ROOT=$HOME/.exo-inkling-bench/bf16-longctx MODEL=$HOME/.exo-inkling-bench/models/thinkingmachines--Inkling-2-Mini NOHASH=${NOHASH:-1} BLOCK=$k OUT=$PWD/$T.json PLAN="[{\"ctx\":$c,\"step\":${STEP:-4096},\"max_tokens\":256},{\"ctx\":$c,\"step\":${STEP:-4096},\"max_tokens\":256}]" $PY -u ../stack13/exact_check_draft.py 2>&1 | grep -E "Error|error" | cut -c1-240
 python3 -c "
import json; d=json.load(open('$T.json'))
for r in d['runs']: print('$T', 'dtps %.2f ptps %.1f needle %s acc %s rounds %s'%(r['decode_tps'],r['prompt_tps'],r['needle_found'],r.get('accept_rate'),r.get('rounds')), 'misses', r.get('exact_verify_misses'), 'ntok', r['gen_tokens'])"
done; done
echo DSDONE
