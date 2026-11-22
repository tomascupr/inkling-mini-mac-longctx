#!/bin/bash
# OpenAI-compatible server with the measured-fastest settings (single request, draft head on).
#   bash scripts/serve.sh [port]        ROOT / MODEL as in env.sh
# - prefill chunk 8192 (a 4k prompt is one forward; 32k is four),
# - MLX's buffer cache left at the working-set size: a 1 GiB cache limit made prefill re-allocate its
#   buffers every visit (S2: 32k reading 864 -> 892-900 without it),
# - per-visit prefill evals issued async so the next visit's graph is built while the GPU runs.
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/env.sh"
export MLX_VLM_EXACT_PREFILL_EVAL_ASYNC=1
# Sliding-window attention in 64-query blocks (bitwise equal; 6.9 -> 4.7 ms per 4k visit).
export MLX_VLM_RELATIVE_ATTN_BQ_SWA=64
exec "$ROOT/venv/bin/python" -u -c "import mlx.core as mx; lim = mx.device_info()['max_recommended_working_set_size']; mx.set_memory_limit(lim); mx.set_wired_limit(lim); mx.set_cache_limit(lim); from mlx_vlm.server.cli import main; main()" \
    --host 127.0.0.1 --port "${1:-8080}" --model-dir "$MODEL" \
    --prefill-step-size "${STEP:-8192}" --max-num-seqs 1 --max-tokens 8192
