#!/bin/bash
# Build the recipe on a Mac: the betizzel/mlx-vlm fork from its GitHub zip
# (no git needed), an unpatched copy for reference runs and a patched copy.
#
#   bash scripts/install.sh            ROOT=~/inkling-mini by default
#   PREFIX=1 bash scripts/install.sh   also apply 0004-prefix-cache.patch to the patched copy
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
(cd stacked && patch -p1 --no-backup-if-mismatch < "$HERE/../patches/stacked.patch")
if [ "${PREFIX:-0}" = 1 ]; then
    (cd stacked && patch -p1 --no-backup-if-mismatch < "$HERE/../patches/0004-prefix-cache.patch")
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
echo "Recommended run env: MLX_VLM_EXACT_PREFILL_EVAL_EVERY=1 MLX_MAX_MB_PER_BUFFER=2 PYTHONPATH=$ROOT/src/stacked"
