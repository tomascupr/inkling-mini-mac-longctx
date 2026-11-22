# Run Inkling-2-Mini faster on a single Mac

We made Thinking Machines' Inkling-2-Mini **write answers up to 2.8× faster** and **read long documents up to 1.8× faster**
on one 64 GB Mac Studio, at full quality: our patches leave the model's output **bit-for-bit unchanged**. The model supports up to 1M tokens of context,
and we measured it at the full 1M on one Mac (the hidden fact was found; outputs are checked bit-for-bit through 262k).

**Before** = the standard MLX build people use to run Inkling-2-Mini on a Mac today (inkling2-mlx v0.2.0).
**After** = this repo: a community fork of that MLX code plus our patches (credits below). Same Mac (M4 Max, 64 GB), same official weights, same prompts.

Speeds are tokens per second. **Reading** = processing your prompt or document (prefill).
**Writing** = generating the answer (decode).

| Prompt length | Reading: before → after | Writing: before → after | Speed-up (reading / writing) |
|---|---|---|---|
| 4k tokens | 758.4 → **952.4** | 33.1 → **56.9** | 1.26× / 1.72× |
| 32k tokens | 667.9 → **867.3** | 27.0 → **53.7** | 1.30× / 1.99× |
| 131k tokens | 401.9 → **625.2** | 16.2 → **40.2** | 1.56× / 2.48× |
| 262k tokens | 256.4 → **453.9** | 10.7 → **29.8** | 1.77× / 2.78× |
| 524k tokens | not measured* → **292.6** | not measured* → **17.9** | – |
| 1M tokens | not measured* → **156.0** | not measured* → **10.0** | – |

The 524k and 1M rows were measured with the previous build (writing ~10–12% slower than today's at 4k–262k); they will be re-run.

\* We have no 524k or 1M run on the standard build, so there is no "before" number (an earlier 524k run on the unpatched fork was stopped after 14 minutes, still reading). Ours read the 524k-token document in 30 minutes (peak 35.3 GiB of GPU memory) and the full 1M-token document (1,048,558 tokens) in 1 h 52 min (peak 50.0 of the 51.8 GiB GPU limit), and found the hidden fact both times.

The longer the prompt, the bigger the gain. At 32k–1M the model found a fact hidden in the middle of the document in every run, before and after.
Full benchmarks, methods and raw logs: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Follow-up questions in 1.5 s

Ask about a 131k-token document once: reading it takes about 3.6 minutes. Ask a follow-up, and the answer starts
in **1.5 s** (1.46–1.54 s measured), because the Mac keeps the document's processed state instead of reading it again.

## Optional: speculative decoding, much faster writing

Inkling-2-Mini ships a small draft head (DSpark) that guesses several words ahead; the big model then checks
all the guesses in one pass, so the answer is the same, just written faster. The standard build already supports it.
Our optional `0005` patch (install with `DSPARK=1`; it applies to the previous build, `stacked-v4.patch`, until it is ported) makes the checking step a compiled program that reads the
long document's memory once per check instead of once per guessed word:

| Writing speed (tok/s) | 4k prompt | 32k prompt | 131k prompt |
|---|---:|---:|---:|
| Standard build, no draft | 33.1 | 27.0 | 16.2 |
| Standard build + draft head | 58.8 | – | – |
| This repo (previous build `stacked-v4`), no draft | 50.6 | 47.2 | 35.8 |
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

Bit-identical outputs: at 4k, 32k, 131k and 262k the patched build produces the same tokens and the same per-token
log-probabilities, bit for bit, as the same code without our patches (the community fork we build on). No quantized KV cache, no sparse or approximate
attention: this is the full bf16 model, just faster.

Standard benchmarks (greedy, same prompts, 2026-11-22): **GSM8K** (first 100) 97% standard build vs **99%** this repo;
**MMLU-Pro** (140, 10 per subject) 78.6% vs **77.9%**; **RULER needle-in-a-haystack at 32k** (12) 12/12 both. The paired
differences are within noise (2 and 13 items differ; McNemar p = 0.5 and 1.0). This repo's output was token-for-token identical
to the community fork on all 100 GSM8K items; it differs from the standard build's wording by design (different
code), not in accuracy. Same run: 1.3× faster reading and 1.8× faster writing on the 32k items. Details:
[docs/BENCHMARKS.md](docs/BENCHMARKS.md#standard-quality-benchmarks).

## What's next

- **1M tokens on one Mac:** measured (above). Next: speculative decoding and the prefix cache at 1M.
- **Speculative decoding** at 262k+ and combined with the prefix cache: not measured yet.
- More prefill and decode headroom: we're at ~67–70% of the chip's compute limit on prefill and ~66–78% of its memory bandwidth on decode (32k–262k).

## Credits

Built on [betizzel/mlx-vlm `m5pro-inkling2`](https://github.com/betizzel/mlx-vlm/tree/m5pro-inkling2) (@55ca2e9e), which ports
jundot's `relative_attention` kernel from [ml-explore/mlx-lm#2086](https://github.com/ml-explore/mlx-lm/pull/2086); on
[inkling2-mlx](https://github.com/internet-projects/inkling2-mlx) (NetProjects), [MLX-VLM](https://github.com/Blaizzy/mlx-vlm)
(Prince Canuma and contributors) and Apple's [MLX](https://github.com/ml-explore/mlx). Inkling-2-Mini is by Thinking Machines Lab.
Our patches came from an exact memory track, a prefix-cache track, and two optimization tracks run with AI coding
agents (Claude on decode, Codex on prefill).

MIT licensed (see `LICENSE`); hunks touching the Apache-2.0 `inkling2_loop` files stay Apache-2.0.
