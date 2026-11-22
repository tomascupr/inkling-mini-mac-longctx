# Default run environment (source it): fused prefill, short-prompt decode kernels,
# and speculative decoding with the official DSpark draft head (exact verify).
#   source scripts/env.sh            ROOT=~/inkling-mini, MODEL=$ROOT/models/Inkling-2-Mini
#   NO_DRAFT=1 source scripts/env.sh  the same without the draft head
ROOT=${ROOT:-$HOME/inkling-mini}
MODEL=${MODEL:-$ROOT/models/Inkling-2-Mini}
export PYTHONPATH=$ROOT/src/stacked
export MLX_VLM_EXACT_PREFILL_EVAL_EVERY=1 MLX_MAX_MB_PER_BUFFER=2
export MLX_VLM_RELATIVE_ATTN_KTILES=3 MLX_VLM_EXACT_HOST_VOCAB=1      # long-context reading (0008, 0009)
export MLX_VLM_INKLING2_PREFILL_FUSED=1                                # fused prefill kernels (0010)
export MLX_VLM_INKLING2_DECODE_MIN_KEYS=1024                           # compiled decode from 1k keys (0011)
if [ "${NO_DRAFT:-0}" != 1 ]; then
    # Draft head (0007 exact verify + 0012 confidence cut). The output is the same
    # tokens as without it. Requests outside 1,024..262,144 context tokens decode
    # without the draft (slower below: short prompts can't use the exact verifier;
    # not measured above).
    export MLX_VLM_DRAFT_MODEL=$MODEL MLX_VLM_DRAFT_BLOCK_SIZE=8
    export MLX_VLM_INKLING2_INCREMENTAL_DRAFT_PREFILL=1 MLX_VLM_INKLING2_EXACT_VERIFY=1
    export MLX_VLM_INKLING2_COMPILED_VERIFY=1 MLX_VLM_INKLING2_DRAFT_CTX_WINDOW=65536
    export MLX_VLM_INKLING2_DRAFT_CONF=0.5
    export MLX_VLM_INKLING2_DRAFT_MIN_CTX=1024 MLX_VLM_INKLING2_DRAFT_MAX_CTX=262144
fi
