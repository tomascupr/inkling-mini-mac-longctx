#!/bin/bash
# Studio 1 same-session 1M head-to-head (Tomas t459u).
# MLX-Serve first, then our final stack (s1/stack32 = S2 h30 overlay),
# serve.sh settings + 0019 + BQ_SWA=64, env unchanged (DRAFT_MAX_CTX=262144).
# Prompt target 1,040,000 tokens (under model max 1,048,576). 2 measured runs
# plus warmup, same harness as h32. Caller holds gpu.lock.
set -uo pipefail
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin
BENCH="$HOME/.exo-inkling-bench"
SHORT="$BENCH/exact-work/s1/short"
MS="$BENCH/mlxserve"
CTX=1040000
export TIMEOUT=21600 HTTP_TIMEOUT=21600 CTX_SIZE=1048576 REPS=2
cd "$SHORT" || exit 1
LOG="$SHORT/h2h34-1m.log"
exec > >(tee -a "$LOG") 2>&1

XENV="MLX_VLM_INKLING2_SPEC_LOG=1 MLX_VLM_BATCH_CACHE_EVAL_INTERVAL=0 PF_MGC=1 MLX_VLM_INKLING2_DRAFT_LEAN_PREFILL=1 MLX_VLM_SERVER_FOLD_LAST=1 CACHE_LIMIT=0 MLX_VLM_EXACT_PREFILL_EVAL_ASYNC=1 MLX_VLM_RELATIVE_ATTN_BQ_SWA=64"

rss_of() {
  local pid="$1"
  ps -o rss=,pid= -p "$pid" 2>/dev/null | awk '{printf "rss_kib=%s pid=%s\n", $1, $2}'
}

echo "=========================================="
echo "h2h34 start $(date '+%F %T %Z')"
echo "host=$(hostname) ctx=$CTX ctx-size=$CTX_SIZE timeout=$TIMEOUT http-timeout=$HTTP_TIMEOUT"
echo "note=524k on this Mac was stopped at 00:06:11 CET because it duplicated Studio 2"
echo "draft_max_ctx is left at 262144 from env.sh; report whether draft is active at 1M"
echo "work=$SHORT"
echo "=========================================="

echo "---- mlxserve ctx=$CTX start $(date '+%T %Z') ----"
mlx_fail=0
if ! bash "$MS/rerun_draft_on_long.sh" "$CTX"; then
  mlx_fail=1
  echo "MLXSERVE_FAIL ctx=$CTX"
fi
L=$(ls -t "$MS/logs"/serve-h34-*-$CTX-draft-on.log "$MS/logs"/serve-h32-*-$CTX-draft-on.log 2>/dev/null | head -1 || true)
# tag is h32-HHMM in rerun script... actually T=h32-$(date +%H%M)-$CTX from the script. I should change that or grep both.
L=$(ls -t "$MS/logs"/serve-*-$CTX-draft-on.log 2>/dev/null | head -1 || true)
echo "mlxserve log ${L:-NONE}"
if [[ -n "${L:-}" ]]; then
  grep -E "tokens streamed|out of memory|OOM|refused|does not fit|Prompt exceeds|SERVER DIED|stall|admission|needed=" "$L" | tail -60
  grep -E "^RSS:|Mem:|peak|wired|ctx-size|Request timeout|Prefill chunk|Context size" "$L" | head -40
fi

echo "---- ours ctx=$CTX start $(date '+%T %Z') ----"
ours_fail=0
tag="h34-$CTX"
# sample memory of whatever python server appears
(
  for i in $(seq 1 720); do
    echo -n "$(date +%T) "
    ps auxww | grep "[v]env-exact/bin/python -u -c" | awk '{print "rss_kib="$6, "pcpu="$3, "pid="$2}'
    sleep 60
  done
) > "$SHORT/h2h34-ours-rss.log" 2>&1 &
RSSPID=$!

if ! TAG="$tag" STK=stack32 STEP=8192 CTXS="$CTX" REPS=2 XENV="$XENV" HTTP_TIMEOUT="$HTTP_TIMEOUT" bash "$SHORT/../stack16/srv3_long.sh" > "srv-${tag}.out" 2>&1; then
  ours_fail=1
  echo "OURS_FAIL ctx=$CTX"
fi
kill $RSSPID 2>/dev/null || true
wait $RSSPID 2>/dev/null || true
echo "== ours $tag $(date '+%T %Z')"
cat "srv-${tag}.out"
grep -h -o "VIOLET-ANCHOR-[0-9]*" srv-${tag}-*.json 2>/dev/null | sort | uniq -c || true
"$BENCH/venv-exact/bin/python" - << PY || true
import json,glob,os
for p in sorted(glob.glob("srv-${tag}-*.json")):
    d=json.load(open(p))
    keys=('prompt_tokens','completion_tokens','needle_found','error','finish_reason','elapsed_s')
    print("JSON", os.path.basename(p), {k:d.get(k) for k in keys})
PY
# Draft-activity hints from our server log (env left unchanged; DRAFT_MAX_CTX=262144)
if [[ -f "srv-${tag}.log" ]]; then
  echo "---- draft / memory hints in ours log ----"
  grep -iE "draft|speculat|Drafter|SPEC|MAX_CTX|out of memory|OOM|peak|wired|MemoryError|does not fit|Prefill progress|Prefill started|Request completed|Traceback" "srv-${tag}.log" | tail -80
fi
echo "---- ours rss samples (tail) ----"
tail -20 "$SHORT/h2h34-ours-rss.log" 2>/dev/null || true
echo "PAIR_DONE ctx=$CTX mlx_fail=$mlx_fail ours_fail=$ours_fail $(date '+%T %Z')"
echo "H2H34DONE $(date '+%F %T %Z')"
