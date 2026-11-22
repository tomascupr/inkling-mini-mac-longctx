#!/bin/bash
# Fork-vs-fork noise floor for the top-1 gate: fused OFF, other prefill chunk sizes.
cd ~/.exo-inkling-bench/exact-work/short || exit 1
PY=../../venv-exact/bin/python; OV=$PWD/../stack8/ov
for s in 4096 1024; do
  echo "== fork STEP=$s"
  env PYTHONPATH=$OV MODE=test TAG=fork-step$s STEP=$s SKIP_GREEDY=1 $PY -u top1.py 2>&1 | grep SUMMARY
done
echo NOISEDONE
