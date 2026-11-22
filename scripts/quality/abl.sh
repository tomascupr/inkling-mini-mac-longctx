#!/bin/bash
cd ~/.exo-inkling-bench/exact-work/short || exit 1
PY=../../venv-exact/bin/python; OV=$PWD/../stack8/ov
for v in ${VARS:-PF_AR PF_MG PF_MOE PF_SC PF_L2 PF_PJ}; do
  echo "== $v=0"
  env PYTHONPATH=$PWD/pf:$OV MODE=test TAG=abl-$v SKIP_GREEDY=1 MLX_VLM_INKLING2_PREFILL_FUSED=1 $v=0 $PY -u top1.py 2>&1 | grep SUMMARY
done
echo "== ONLY each"
for v in PF_AR PF_MG PF_MOE PF_SC PF_L2 PF_PJ; do
  E="PF_AR=0 PF_MG=0 PF_MOE=0 PF_SC=0 PF_L2=0 PF_PJ=0 $v=1"
  echo "== only $v"
  env PYTHONPATH=$PWD/pf:$OV MODE=test TAG=only-$v SKIP_GREEDY=1 MLX_VLM_INKLING2_PREFILL_FUSED=1 $E $PY -u top1.py 2>&1 | grep SUMMARY
done
echo ABLDONE
