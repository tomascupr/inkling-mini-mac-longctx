#!/bin/bash
# Build the recipe on a Mac: the betizzel/mlx-vlm fork from its GitHub zip
# (no git needed), an unpatched copy for reference runs and a patched copy.
#
#   bash scripts/install.sh            ROOT=~/inkling-mini by default
#   PREFIX=1 bash scripts/install.sh   also apply 0004-prefix-cache.patch (+ 0006 eviction fix) to the patched copy
#   NO_DRAFT=1 bash scripts/install.sh  skip 0007/0012 (the draft head is applied by default: same tokens, faster writing)
#   (0008 KT3 attention tiles and 0009 host vocab tables are always applied; both are off unless their env is set)
#   DSPARK=1 bash scripts/install.sh   older 0005 (same tokens, not bitwise) on stacked-v4.patch; not combinable with PREFIX=1
#
# Layout afterwards:
#   $ROOT/venv              Python venv, mlx 0.32.4, the unpatched fork installed (for its deps)
#   $ROOT/src/base          unpatched fork source   -> PYTHONPATH=$ROOT/src/base
#   $ROOT/src/stacked       fork + patches/stacked.patch -> PYTHONPATH=$ROOT/src/stacked
# Weights (official, ~18 GiB):
#   hf download thinkingmachines/Inkling-2-Mini --local-dir $ROOT/models/Inkling-2-Mini
#   (our runs used Hub revision ed40e7da)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${ROOT:-$HOME/inkling-mini}
PYBIN=${PYBIN:-python3}
BASE=55ca2e9e3df180dc2c46550e44309d6af5489b86
mkdir -p "$ROOT/src" && cd "$ROOT/src"
[ -f fork-$BASE.zip ] || curl -fsSL -o fork-$BASE.zip "https://github.com/betizzel/mlx-vlm/archive/$BASE.zip"
rm -rf base stacked "mlx-vlm-$BASE"
unzip -q fork-$BASE.zip && mv "mlx-vlm-$BASE" base && cp -R base stacked
# 0005 (DSpark) was built on the previous stack (stacked-v4.patch); with DSPARK=1 that one is used.
if [ "${DSPARK:-0}" = 1 ]; then STACK=stacked-v4.patch; else STACK=stacked.patch; fi
(cd stacked && patch -p1 --no-backup-if-mismatch < "$HERE/../patches/$STACK")
if [ "${DSPARK:-0}" != 1 ]; then
    # Long-context reading: env MLX_VLM_RELATIVE_ATTN_KTILES=3 and MLX_VLM_EXACT_HOST_VOCAB=1 (bit-identical).
    (cd stacked && patch -p1 --no-backup-if-mismatch < "$HERE/../patches/0008-kt3-attention-tiles.patch")
    (cd stacked && patch -p1 --no-backup-if-mismatch < "$HERE/../patches/0009-host-vocab-prefill.patch")
    # Short-context reading: env MLX_VLM_INKLING2_PREFILL_FUSED=1 (bit-identical fused prefill kernels).
    (cd stacked && patch -p1 --no-backup-if-mismatch < "$HERE/../patches/0010-prefill-fused.patch")
    # Short-prompt writing: env MLX_VLM_INKLING2_DECODE_MIN_KEYS=1024 (on in scripts/env.sh; same answers, quality-gated).
    (cd stacked && patch -p1 --no-backup-if-mismatch < "$HERE/../patches/0011-decode-min-keys.patch")
fi
if [ "${PREFIX:-0}" = 1 ] && [ "${DSPARK:-0}" = 1 ]; then
    echo "PREFIX=1 and DSPARK=1 cannot be combined yet" >&2; exit 1
fi
DRAFT=0
if [ "${NO_DRAFT:-0}" != 1 ] && [ "${DSPARK:-0}" != 1 ]; then
    # Speculative decoding with the official draft head, exact verify (same tokens) + confidence cut.
    (cd stacked && patch -p1 --no-backup-if-mismatch < "$HERE/../patches/0007-exact-dspark.patch")
    (cd stacked && patch -p1 --no-backup-if-mismatch < "$HERE/../patches/0012-draft-confidence-cut.patch")
    DRAFT=1
fi
if [ "${DSPARK:-0}" = 1 ]; then
    (cd stacked && patch -p1 --no-backup-if-mismatch < "$HERE/../patches/0005-dspark-compiled-verify.patch")
fi
if [ "${PREFIX:-0}" = 1 ] && [ "$DRAFT" = 1 ]; then
    # 0004 + 0006 rebased onto 0007 (one hunk overlapped); prefix cache + draft not yet measured end to end.
    (cd stacked && patch -p1 --no-backup-if-mismatch < "$HERE/../patches/0013-prefix-cache-on-exact-dspark.patch")
elif [ "${PREFIX:-0}" = 1 ]; then
    (cd stacked && patch -p1 --no-backup-if-mismatch < "$HERE/../patches/0004-prefix-cache.patch")
    (cd stacked && patch -p1 --no-backup-if-mismatch < "$HERE/../patches/0006-prefix-cache-evict-fix.patch")
fi
if [ "${DSPARK:-0}" != 1 ]; then
    # Server: one greedy request decodes through the same compiled step as generate (same tokens).
    (cd stacked && patch -p1 --no-backup-if-mismatch < "$HERE/../patches/0014-server-compiled-decode.patch")
fi
if [ "$DRAFT" = 1 ]; then
    # Server + draft head: a single request runs the exact-verify round loop (same tokens as no draft).
    (cd stacked && patch -p1 --no-backup-if-mismatch < "$HERE/../patches/0015-server-exact-draft.patch")
    # Draft head proposes from a 16k candidate list (+ prompt ids) with a compiled Markov loop; same tokens.
    (cd stacked && patch -p1 --no-backup-if-mismatch < "$HERE/../patches/0016-draft-vocab-compiled-markov.patch")
    cp "$HERE/../data/draft_vocab_order.json" stacked/mlx_vlm/speculative/drafters/inkling2_dspark/
fi
[ -x "$ROOT/venv/bin/python" ] || "$PYBIN" -m venv "$ROOT/venv"
"$ROOT/venv/bin/pip" install --quiet --upgrade pip
"$ROOT/venv/bin/pip" install --quiet ./base "mlx==0.32.4"
# The official processor needs torch + torchvision (we ran torch 2.15.0, torchvision 0.29.2).
"$ROOT/venv/bin/pip" install --quiet torch torchvision
for v in base stacked; do
    PYTHONPATH="$ROOT/src/$v" "$ROOT/venv/bin/python" -c \
        "import mlx.core as mx, mlx_vlm; print('$v', mlx_vlm.__version__, mlx_vlm.__file__, 'mlx', mx.__version__)"
done
echo "Run env: source scripts/env.sh   (ROOT=$ROOT; NO_DRAFT=1 for no draft head)"
