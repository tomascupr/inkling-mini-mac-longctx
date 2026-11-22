#!/bin/bash
# Mac Studio M4 Max 64 GB: Inkling 2 Mini follow-ups on a long conversation,
# on betizzel/mlx-vlm m5pro-inkling2 @55ca2e9e plus patches/0004-prefix-cache.patch.
# Measured on Metal at 131k (2026-11-22): turn 2 restored 131,006 cached tokens,
# first token in 1.54 s vs a 219.5 s cold first turn. See results/.
#
# The venv is left alone (`test` installs pytest if it is missing): the patched
# package is a checkout in $SRC that goes first on PYTHONPATH. Stop the server
# that holds the weights first: stop your own server first
#
#   bash run_mac.sh setup     clone the fork at 55ca2e9e into $SRC and apply the patch
#   bash run_mac.sh test      the prefix-cache tests, the 6 Metal ones included, and the compaction test
#   bash run_mac.sh server    start the server (prefix cache in RAM), log in $RUN/server.log
#   bash run_mac.sh ask       CTX document tokens + a question, then a follow-up; then the cache counters
#   bash run_mac.sh stats     the cache counters and the server's cache log lines
#   bash run_mac.sh stop
#
# The measurement is `ask`: one JSON line per turn. Turn 2 must show
# cached_tokens = turn 1's prompt_tokens - 32, and first_token_s is its wait.
# If cached_tokens is 0, `stats` says why: memory_skips > 0 means the checkpoint
# did not fit APC_GB, exact_stores = 0 that nothing was stored.
#
# Checkpoints on disk, across a restart (each one is a safetensors file of the
# whole cache, about 4.2 GiB at 131k, derived: 32 KiB per token plus the rings):
#
#   DISK=1 bash run_mac.sh server && bash run_mac.sh ask
#   DISK=1 bash run_mac.sh server      (restarts it)
#   bash run_mac.sh again              turn 2 again: cached_tokens = its prompt_tokens - 32, disk_hits = 1
#
# The patch against the unpatched fork, same steps:  PATCH=0 bash run_mac.sh setup
#
# Settings (environment): CTX=131072 document tokens, STEP=2048 prefill chunk,
# EFFORT=none, MAX_TOKENS=300, PORT=52417, APC_GB (RAM for the one checkpoint;
# default 32 KiB x CTX + 2 GiB, derived), DISK_GB=20.
#
# The launch is the fork author's (his render-config.py), with three changes:
#   - no --max-kv-size: with it the server summarises a conversation that has
#     reached 3/4 of it, unless the request carries "context_management": []
#     (followup.py sends that too). The summary replaces the document.
#   - --log-level INFO and APC_TRACE=1, for the cache lines in the log.
#   - APC_GB follows CTX, and DISK=1 turns the disk tier on.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${ROOT:-$HOME/inkling-mini}
PY=${PY:-$ROOT/venv/bin/python}
export MODEL=${MODEL:-$ROOT/models/Inkling-2-Mini}
PATCHFILE=$HERE/../patches/0004-prefix-cache.patch
SRC=${SRC:-$ROOT/mlx-vlm-prefix-cache}
RUN=${RUN:-$ROOT/prefix-cache}
REPO=${REPO:-https://github.com/betizzel/mlx-vlm.git}
BASE=55ca2e9e3df180dc2c46550e44309d6af5489b86
PORT=${PORT:-52417}
export URL=http://127.0.0.1:$PORT
export CTX=${CTX:-131072}
export STATE=$RUN/followup-state.json
APC_GB=${APC_GB:-$((CTX * 32768 / 1073741824 + 2))}

patched() {
    PYTHONPATH=$SRC "$PY" "$@"
}

stop() {
    if [ -f "$RUN/server.pid" ] && kill "$(cat "$RUN/server.pid")" 2>/dev/null; then
        while kill -0 "$(cat "$RUN/server.pid")" 2>/dev/null; do sleep 1; done
    fi
    rm -f "$RUN/server.pid"
}

stats() {
    curl -sf "$URL/v1/cache/stats" | "$PY" -c '
import json, sys
s = json.load(sys.stdin)
keys = ("exact_stores", "exact_hits", "disk_hits", "disk_writes", "disk_write_failures", "memory_skips",
        "memory_evictions", "exact_resident_bytes", "memory_max_bytes", "disk_bytes", "last_reject")
print(json.dumps({k: s[k] for k in keys if k in s}))'
    grep -E "APC_TRACE|APC " "$RUN/server.log" | tail -20 || true
}

mkdir -p "$RUN"
case "${1:-}" in
setup)
    if [ ! -d "$SRC/.git" ]; then
        git clone --quiet --branch m5pro-inkling2 "$REPO" "$SRC"
        git -C "$SRC" checkout --quiet --detach "$BASE"
    fi
    [ "$(git -C "$SRC" rev-parse HEAD)" = "$BASE" ] || { echo "$SRC is not at $BASE" && exit 1; }
    # Only this patch goes on or comes off; a tree with other changes is refused.
    if git -C "$SRC" apply --reverse --check "$PATCHFILE" 2>/dev/null; then
        [ "${PATCH:-1}" = 1 ] || git -C "$SRC" apply --reverse "$PATCHFILE"
    elif [ "${PATCH:-1}" = 1 ]; then
        git -C "$SRC" apply "$PATCHFILE"
    fi
    git -C "$SRC" status --short
    patched -c 'import mlx.core as mx, mlx_vlm; print("mlx-vlm", mlx_vlm.__version__, "from", mlx_vlm.__file__, "MLX", mx.__version__)'
    ;;
test)
    patched -c 'import pytest' 2>/dev/null || "$PY" -m pip install --quiet pytest
    cd "$SRC"
    patched -m pytest -q -rs mlx_vlm/tests/test_inkling2_prefix_cache.py \
        "mlx_vlm/tests/test_server.py::TestCompaction::test_summary_request_renders_the_conversation_reasoning_effort"
    ;;
server)
    stop
    if curl -s -o /dev/null "$URL/health"; then
        echo "port $PORT is taken: stop that server first" && exit 1
    fi
    cd "$RUN"
    # The author's start: MLX's memory and wired limits at the recommended working set.
    START="import mlx.core as mx; lim = mx.device_info()['max_recommended_working_set_size']; \
mx.set_memory_limit(lim); mx.set_wired_limit(lim); mx.set_cache_limit(1024**3); \
from mlx_vlm.server.cli import main; main()"
    env PYTHONPATH="$SRC" HF_HUB_OFFLINE=1 APC_ENABLED=1 APC_TRACE=1 \
        APC_MEMORY_MAX_GB="$APC_GB" APC_MEMORY_RESERVE_GB=2 \
        APC_CHECKPOINT_ENTRIES=1 APC_CHECKPOINT_INTERVAL_TOKENS=0 APC_CHECKPOINT_GUARD_TOKENS=32 \
        APC_DISK_ENABLED="${DISK:-0}" APC_DISK_PATH="$RUN/apc" APC_DISK_MAX_GB="${DISK_GB:-20}" \
        MLX_VLM_TOKEN_QUEUE_TIMEOUT=7200 KV_KEY_SCHEME=uniform KV_VALUE_SCHEME=uniform \
        MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS=1000 \
        nohup "$PY" -u -c "$START" --host 127.0.0.1 --port "$PORT" --model-dir "$MODEL" \
        --prefill-step-size "${STEP:-2048}" --max-num-seqs 1 --max-tokens 8192 --log-level INFO \
        >>server.log 2>&1 &
    echo $! >server.pid
    until curl -sf -o /dev/null "$URL/health"; do
        kill -0 "$(cat server.pid)" 2>/dev/null || { tail -20 server.log && exit 1; }
        sleep 1
    done
    echo "server PID $(cat server.pid) on $URL, APC_MEMORY_MAX_GB=$APC_GB, disk tier ${DISK:-0}, log $RUN/server.log"
    ;;
ask | again)
    patched "$HERE/followup.py" "$1"
    stats
    ;;
stats) stats ;;
stop) stop ;;
*)
    sed -n '2,40p' "$0"
    exit 1
    ;;
esac
