#!/bin/bash
# Run the quality benchmark on both builds, one benchmark at a time (both builds per benchmark),
# under the shared GPU lock, with macmon telemetry. Usage: ../gpu-run.sh ./run.sh
cd ~/.exo-inkling-bench/exact-work/qbench
log(){ echo "$(date '+%F %T %Z') $*" >> run.log; }
(/opt/homebrew/bin/macmon pipe -i 2000 >> telemetry-macmon.jsonl 2>> telemetry-macmon.err &) ; sleep 1
log "START telemetry pid $(pgrep -n macmon)"
E="MLX_VLM_EXACT_PREFILL_EVAL_EVERY=1 MLX_MAX_MB_PER_BUFFER=2"
for b in ${BENCH_LIST:-gsm8k mmlu_pro ruler_niah_32k}; do
  log "RUN $b stock"
  env BUILD=stock BENCHES=$b OUT=raw-stock.jsonl $E ../../venv-dspark/bin/python -u bench.py >> run-stock.out 2>&1
  log "RUN $b stack4"
  env BUILD=stack4 BENCHES=$b OUT=raw-stack4.jsonl PYTHONPATH=$PWD/../stack4/ov $E ../../venv-exact/bin/python -u bench.py >> run-stack4.out 2>&1
done
log "DONE"
pkill -f "macmon pipe" ; true
