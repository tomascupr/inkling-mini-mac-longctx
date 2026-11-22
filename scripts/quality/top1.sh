#!/bin/bash
cd ~/.exo-inkling-bench/exact-work/short || exit 1
PY=../../venv-exact/bin/python; OV=$PWD/../stack8/ov
[ -f top1-ref.json ] || env PYTHONPATH=$PWD/pf:$OV MODE=ref $PY -u top1.py 2>&1 | grep -v Redirects
env PYTHONPATH=$PWD/pf:$OV MODE=test TAG=${TAG:-pf} MLX_VLM_INKLING2_PREFILL_FUSED=1 $PY -u top1.py 2>&1 | grep -v Redirects
echo TOP1DONE
