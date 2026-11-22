# Run Inkling-2-Mini faster on a single Mac

We made Thinking Machines' Inkling-2-Mini **write answers up to 2.2× faster** and **read long documents up to 1.75× faster**
on one 64 GB Mac Studio, at full quality: our patches leave the model's output **bit-for-bit unchanged**. The model supports up to 1M tokens of context;
we have measured up to 262k tokens on one Mac so far.

**Before** = the standard MLX build people use to run Inkling-2-Mini on a Mac today (inkling2-mlx v0.2.0).
**After** = this repo: a community fork of that MLX code plus our patches (credits below). Same Mac (M4 Max, 64 GB), same official weights, same prompts.

Speeds are tokens per second. **Reading** = processing your prompt or document (prefill).
**Writing** = generating the answer (decode).

| Prompt length | Reading: before → after | Writing: before → after | Speed-up (reading / writing) |
|---|---|---|---|
| 4k tokens | 758.4 → **951.8** | 33.1 → **50.6** | 1.25× / 1.53× |
| 32k tokens | 667.9 → **865.3** | 27.0 → **47.2** | 1.30× / 1.75× |
| 131k tokens | 401.9 → **624.3** | 16.2 → **35.8** | 1.55× / 2.21× |
| 262k tokens* | 256.4 → **448.7** | 10.7 → **23.9** | 1.75× / 2.23× |

\* 262k was measured with the previous version of our patches (writing has since sped up 15–24% at 4k–131k; 262k not re-run yet).

The longer the prompt, the bigger the gain. At 32k–262k the model found a fact hidden in the middle of the document in every run, before and after.
Full benchmarks, methods and raw logs: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Follow-up questions in 1.5 s

Ask about a 131k-token document once: reading it takes about 3.6 minutes. Ask a follow-up, and the answer starts
in **1.5 s** (1.46–1.54 s measured), because the Mac keeps the document's processed state instead of reading it again.

## Optional: speculative decoding, much faster writing

Inkling-2-Mini ships a small draft head (DSpark) that guesses several words ahead; the big model then checks
all the guesses in one pass, so the answer is the same, just written faster. The standard build already supports it.
Our optional `0005` patch (install with `DSPARK=1`) makes the checking step a compiled program that reads the
long document's memory once per check instead of once per guessed word:

| Writing speed (tok/s) | 4k prompt | 32k prompt | 131k prompt |
|---|---:|---:|---:|
| Standard build, no draft | 33.1 | 27.0 | 16.2 |
| Standard build + draft head | 58.8 | – | – |
| This repo, no draft | 50.6 | 47.2 | 35.8 |
| **This repo + draft head + `0005`** | **65.5** (block 5) | **63.8** (block 8) | **53.5** (block 8) |

Same output tokens as without the draft in every run (checked token by token). The 4k first request is 63.5
(it pays a one-time setup). Details: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## What we changed

- **No more rebuilding the same work for every word.** Each step of writing an answer is compiled once (turned into a
  ready-made GPU program) and reused, instead of being rebuilt from scratch for every new token.
- **Memory-lean reading.** Prompts are read in chunks without keeping unneeded copies around, so long documents fit
  in 64 GB with room to spare.
- **Tuned GPU kernels.** The attention kernel (how each word looks back at the document) and the expert kernels (the
  model's mixture-of-experts layers) are tuned for the M4 Max, with bit-identical results.
- **Instant follow-ups.** The processed state of a document is kept, so follow-up questions start in 1.5 s instead of 3.6 min.

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

Bit-identical outputs: at 4k, 32k and 131k the patched build produces the same tokens and the same per-token
log-probabilities, bit for bit, as the same code without our patches (the community fork we build on). No quantized KV cache, no sparse or approximate
attention: this is the full bf16 model, just faster.

## What's next

- **1M tokens on one Mac:** the model supports 1M tokens; our memory budget projects a 1M context to fit at full bf16 quality (~50.6 of the 51.8 GiB GPU limit). A full 1M run is next; not measured yet.
- **Speculative decoding** at 262k+ and combined with the prefix cache: not measured yet.
- More prefill and decode headroom: we're at ~67–70% of the chip's compute limit on prefill and ~58–65% of its memory bandwidth on decode.

## Credits

Built on [betizzel/mlx-vlm `m5pro-inkling2`](https://github.com/betizzel/mlx-vlm/tree/m5pro-inkling2) (@55ca2e9e), which ports
jundot's `relative_attention` kernel from [ml-explore/mlx-lm#2086](https://github.com/ml-explore/mlx-lm/pull/2086); on
[inkling2-mlx](https://github.com/internet-projects/inkling2-mlx) (NetProjects), [MLX-VLM](https://github.com/Blaizzy/mlx-vlm)
(Prince Canuma and contributors) and Apple's [MLX](https://github.com/ml-explore/mlx). Inkling-2-Mini is by Thinking Machines Lab.
Our patches came from an exact memory track, a prefix-cache track, and two optimization tracks run with AI coding
agents (Claude on decode, Codex on prefill).

MIT licensed (see `LICENSE`); hunks touching the Apache-2.0 `inkling2_loop` files stay Apache-2.0.
