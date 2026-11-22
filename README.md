# Inkling-2-Mini, long context, one Mac

**Inkling-2-Mini with 262k context on one 64 GB Mac Studio: up to 2.2× faster decode and 1.75× faster prefill than stock, bit-exact, full bf16 quality.**

Measured on M4 Max 64 GB, official weights, bf16 KV cache, versus the stock Inkling 2 Mini MLX build
(MLX-VLM 0.8.0+inkling2mini.2 from inkling2-mlx v0.2.0, mlx 0.32.4) on the same Mac:

| Context | Prefill tok/s (this / stock) | Decode tok/s (this / stock) | Speed-up (prefill / decode) |
|---|---|---|---|
| 32k | **863.6** / 667.9 | **39.8** / 27.0 | 1.29× / 1.47× |
| 131k | **623.6** / 401.9 | **31.2** / 16.2 | 1.55× / 1.93× |
| 262k | **448.7** / 256.4 | **23.9** / 10.7 | 1.75× / 2.23× |

Needle-in-a-haystack retrieval passes at every length. Full benchmarks, methodology and raw logs: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Follow-ups in 1.5 s on a 131k document

Read a 131k-token document once (~218 s), then ask follow-up questions that start answering in
**1.46–1.54 s**: the conversation's cache is restored instead of re-reading the document. Saving that cache
to disk, so a document survives a server restart, passes our exactness tests; its Mac timing is not measured yet.

## Why it's fast

- **Compiled decode step:** each token runs through one compiled MLX function instead of a freshly built graph of ~13k nodes, cutting CPU overhead per token.
- **Exact, memory-lean prefill:** logits only for the last token, per-chunk embeddings, KV allocated once, so bigger chunks fit and 32k peak memory drops from 26 to 22 GiB.
- **Attention and MoE kernel tuning:** larger attention tiles and a sort-free expert path, value-identical to the original.
- **Prefix cache:** the fused attention kernel stays on after a cached prefix is restored, so follow-ups skip the long re-read.

## Quickstart

```bash
git clone https://github.com/tomascupr/inkling-mini-mac-longctx && cd inkling-mini-mac-longctx
bash scripts/install.sh                     # PREFIX=1 also adds the prefix-cache patch
hf download thinkingmachines/Inkling-2-Mini --local-dir ~/inkling-mini/models/Inkling-2-Mini

export PYTHONPATH=~/inkling-mini/src/stacked
export MLX_VLM_EXACT_PREFILL_EVAL_EVERY=1 MLX_MAX_MB_PER_BUFFER=2
```

`install.sh` downloads the base code as a zip, applies `patches/stacked.patch` and sets up a venv with
mlx 0.32.4. For fast follow-ups, start the server with the prefix cache on (`APC_ENABLED=1`; see
`scripts/prefix_cache.sh`). Check your own build with `scripts/gate.sh`.

## Quality

Bit-identical outputs: on our gates the patched build produces the same tokens and the same per-token
log-probabilities, bit for bit, as the unpatched code. No quantized KV cache, no sparse or approximate
attention: this is the full bf16 model, just faster.

## What's next

- **1M tokens on one Mac:** the memory budget projects to fit at bf16 under the 51.8 GiB GPU limit; not yet measured.
- **Speculative decoding** with Inkling's own draft head, to push decode further.
- More prefill and decode headroom: we're at ~67–70% of the chip's compute limit on prefill and ~49–62% of its memory bandwidth on decode.

## Credits

Built on [betizzel/mlx-vlm `m5pro-inkling2`](https://github.com/betizzel/mlx-vlm/tree/m5pro-inkling2) (@55ca2e9e), which ports
jundot's `relative_attention` kernel from [ml-explore/mlx-lm#2086](https://github.com/ml-explore/mlx-lm/pull/2086); on
[inkling2-mlx](https://github.com/internet-projects/inkling2-mlx) (NetProjects), [MLX-VLM](https://github.com/Blaizzy/mlx-vlm)
(Prince Canuma and contributors) and Apple's [MLX](https://github.com/ml-explore/mlx). Inkling-2-Mini is by Thinking Machines Lab.
Our patches came from an exact memory track, a prefix-cache track, and two optimization tracks run with AI coding
agents (Claude on decode, Codex on prefill).

MIT licensed (see `LICENSE`); hunks touching the Apache-2.0 `inkling2_loop` files stay Apache-2.0.
