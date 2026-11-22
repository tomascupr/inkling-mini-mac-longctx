#!/bin/bash
# Prefill-fused A/B on Studio 2. usage: CFGS="B P" CTXS="4096 32768" ./pf.sh (under ../gpu-run.sh)
cd ~/.exo-inkling-bench/exact-work/short || exit 1
PY=../../venv-exact/bin/python
OV=$PWD/../stack8/ov
for c in ${CTXS:-4096 32768}; do for cfg in ${CFGS:-P B}; do
 OVX=""
 case $cfg in
  B) E="MLX_VLM_EXACT_PREFILL_EVAL_EVERY=0"; S=2048;;
  P) E="MLX_VLM_EXACT_PREFILL_EVAL_EVERY=0 MLX_VLM_INKLING2_PREFILL_FUSED=1"; S=2048;;
  Q) E="MLX_VLM_EXACT_PREFILL_EVAL_EVERY=0 MLX_VLM_INKLING2_PREFILL_FUSED=1"; S=4096;;
  X) E="MLX_VLM_EXACT_PREFILL_EVAL_EVERY=0 MLX_VLM_INKLING2_PREFILL_FUSED=1 PF_AR=0 PF_MG=0"; S=2048;;
  XQ) E="MLX_VLM_EXACT_PREFILL_EVAL_EVERY=0 MLX_VLM_INKLING2_PREFILL_FUSED=1 PF_AR=0 PF_MG=0"; S=4096;;
  Y) E="MLX_VLM_EXACT_PREFILL_EVAL_EVERY=0 MLX_VLM_INKLING2_PREFILL_FUSED=1 PF_AR=0"; S=2048;;
  YQ) E="MLX_VLM_EXACT_PREFILL_EVAL_EVERY=0 MLX_VLM_INKLING2_PREFILL_FUSED=1 PF_AR=0"; S=4096;;
  A) E="MLX_VLM_EXACT_PREFILL_EVAL_EVERY=0 MLX_VLM_INKLING2_PREFILL_FUSED=1 PF_AR=1 PF_ARX=1 PF_MG=0"; S=2048;;
  AQ) E="MLX_VLM_EXACT_PREFILL_EVAL_EVERY=0 MLX_VLM_INKLING2_PREFILL_FUSED=1 PF_AR=1 PF_ARX=1 PF_MG=0"; S=4096;;
  M) E="MLX_VLM_EXACT_PREFILL_EVAL_EVERY=0 MLX_VLM_INKLING2_PREFILL_FUSED=1 PF_AR=1 PF_MG=1"; S=2048;;
  MQ) E="MLX_VLM_EXACT_PREFILL_EVAL_EVERY=0 MLX_VLM_INKLING2_PREFILL_FUSED=1 PF_AR=1 PF_MG=1"; S=4096;;
  N) E="MLX_VLM_EXACT_PREFILL_EVAL_EVERY=0 MLX_VLM_INKLING2_PREFILL_FUSED=1"; S=2048; OVX=$PWD/../stack12/ov;;
  NQ) E="MLX_VLM_EXACT_PREFILL_EVAL_EVERY=0 MLX_VLM_INKLING2_PREFILL_FUSED=1"; S=4096; OVX=$PWD/../stack12/ov;;
  NH) E="NOHASH=1 MLX_VLM_EXACT_PREFILL_EVAL_EVERY=0 MLX_VLM_INKLING2_PREFILL_FUSED=1"; S=2048; OVX=$PWD/../stack12/ov;;
  NQH) E="NOHASH=1 MLX_VLM_EXACT_PREFILL_EVAL_EVERY=0 MLX_VLM_INKLING2_PREFILL_FUSED=1"; S=4096; OVX=$PWD/../stack12/ov;;
  R) E="MLX_VLM_EXACT_PREFILL_EVAL_EVERY=1 MLX_MAX_MB_PER_BUFFER=2 MLX_VLM_INKLING2_PREFILL_FUSED=1"; S=2048;;
 esac
 echo "$(date +%T) ctx $c cfg $cfg step $S env $E"
 if [ -n "$OVX" ]; then PP=$OVX; else PP=$PWD/pf:$OV; fi
 env PYTHONPATH=$PP $E OUT=$PWD/pf-$c-$cfg.json PLAN="[{\"ctx\":$c,\"step\":$S,\"max_tokens\":256},{\"ctx\":$c,\"step\":$S,\"max_tokens\":256}]" $PY -u ../common/exact_check.py 2>&1 | grep -E "Error|error" | cut -c1-300
done; done
python3 - <<"P"
import json,glob
R={4096:"../stack/mc-stock4k.json",32768:"../ref/check-32k-base.json"}
for f in sorted(glob.glob("pf-*-%s.json"%__import__("os").environ.get("ONLY","*"))):
    c=int(f.split("-")[1]); d=json.load(open(f)); b=json.load(open(R[c]))["runs"][0]
    rs=d["runs"]; r=rs[-1]
    agree=sum(x==y for x,y in zip(r["tokens"],b["tokens"]))
    print(f,"ptps",[round(x["prompt_tps"],1) for x in rs],"dtps",[round(x["decode_tps"],2) for x in rs],"peak %.2f needle %s sha_eq %s tok_eq %s first_div %s agree %d/%d"%(r["peak_mlx_GiB"],r["needle_found"],r.get("logprobs_sha256")==b.get("logprobs_sha256"),r["tokens"]==b["tokens"],next((i for i,(x,y) in enumerate(zip(r["tokens"],b["tokens"])) if x!=y),None),agree,len(b["tokens"])))
P
echo PFDONE
