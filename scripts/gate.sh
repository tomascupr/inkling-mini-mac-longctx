#!/bin/bash
# Exactness + speed gate: greedy needle runs on the unpatched fork and on the
# patched build, then compare token ids and the sha256 over every per-token
# logprob vector. One GPU job at a time; nothing else heavy on the Mac.
#
#   bash scripts/gate.sh                 CTXS="32768 131072", both builds
#   CTXS="262144" SKIP_BASE=1 bash scripts/gate.sh
#
# Needs $ROOT/corpus.txt (any plain text longer than the largest context; we
# used ~6.9 MB of public-domain Project Gutenberg books) and the layout from install.sh.
# Bit-identity only holds against a reference made with the same corpus,
# step and MLX version, so make your own base reference on your machine.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export ROOT=${ROOT:-$HOME/inkling-mini}
PY=${PY:-$ROOT/venv/bin/python}
OUTDIR=${OUTDIR:-$ROOT/gate}; mkdir -p "$OUTDIR"
CTXS=${CTXS:-"32768 131072"}; STEP=${STEP:-2048}; MAXT=${MAXT:-256}
for c in $CTXS; do
    plan="[{\"ctx\":$c,\"step\":$STEP,\"max_tokens\":$MAXT}]"
    if [ "${SKIP_BASE:-0}" != 1 ]; then
        PYTHONPATH="$ROOT/src/base:$HERE" OUT="$OUTDIR/base-$c.json" PLAN="$plan" \
            "$PY" -u "$HERE/exact_check.py" | grep '^{' | cut -c1-240
    fi
    PYTHONPATH="$ROOT/src/stacked:$HERE" MLX_VLM_EXACT_PREFILL_EVAL_EVERY=1 MLX_MAX_MB_PER_BUFFER=2 \
        OUT="$OUTDIR/stacked-$c.json" PLAN="$plan" "$PY" -u "$HERE/exact_check.py" | grep '^{' | cut -c1-240
done
OUTDIR="$OUTDIR" CTXS="$CTXS" "$PY" - <<'PYEOF'
import json, os
d = os.environ["OUTDIR"]
for c in os.environ["CTXS"].split():
    r = json.load(open(f"{d}/stacked-{c}.json"))["runs"][0]
    s = "ctx %s prefill %.1f tok/s decode %.2f tok/s peak %.2f GiB needle %s" % (
        c, r["prompt_tps"], r["decode_tps"], r["peak_mlx_GiB"], r.get("needle_found"))
    try:
        b = json.load(open(f"{d}/base-{c}.json"))["runs"][0]
        s += " | tokens_equal %s logprob_sha_equal %s (base %.1f / %.2f / %.2f)" % (
            r["tokens"] == b["tokens"], r["logprobs_sha256"] == b["logprobs_sha256"],
            b["prompt_tps"], b["decode_tps"], b["peak_mlx_GiB"])
    except FileNotFoundError:
        s += " | no base reference"
    print(s)
PYEOF
