#!/bin/bash
# Control: the unpatched community fork (same code stack4 patches), GSM8K only, to separate fork-vs-stock drift from our patches.
cd ~/.exo-inkling-bench/exact-work/qbench
echo "$(date "+%F %T %Z") RUN gsm8k fork" >> run.log
env BUILD=fork BENCHES=gsm8k OUT=raw-fork.jsonl PYTHONPATH=$PWD/../common/overlay-base MLX_MAX_MB_PER_BUFFER=2 ../../venv-exact/bin/python -u bench.py >> run-fork.out 2>&1
echo "$(date "+%F %T %Z") DONE fork" >> run.log
