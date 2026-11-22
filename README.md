# Inkling 2 Mini on a Mac, fast

**Full quality on the official weights (no requant) · fastest Inkling 2 Mini prompt reading on an M4 Max · 1M-token
context on a 64 GB Mac.**

Thinking Machines' Inkling-2-Mini, on one Mac Studio (M4 Max, 64 GB): **writing up to 2.8× faster** and **reading up to
1.8× faster** than the standard MLX build, the full 1M-token context on one machine, and the same answers. No
re-quantized weights, no FP8 KV cache, no approximate attention.

**Before** = inkling2-mlx v0.2.0, the standard MLX build people run today.
**After** = this repo: the community fork by betizzel (with jundot's attention kernel) plus our patches (credits below).
Same Mac, same official weights, same prompts.

### Head to head with MLX-Serve (same Mac, same official weights)

Measured on one M4 Max 64 GB (our second test Mac; MLX-Serve's own server-log numbers). Tokens per second.

| | MLX-Serve | This repo | |
|---|---:|---:|---|
| Reading, 4k prompt | 1010–1012 | **1013–1037** | **fastest** |
| Reading, 32k prompt | 877 | **884–888** | **fastest** |
| Writing, 32k, no draft | 50.2 | 48.9 | on par (−3%) |
| Writing, 4k, no draft | 56.4 | 55.1–55.9 | on par (−1 to −2%) |
| Writing, 4k, draft head (default) | 66.1 | **74.1** (block 8; 74.8–76.3 at block 5) | **fastest** |
| Writing, 32k, draft head (default) | 74.8 | 72.3 | on par (−3%), in progress |

Reading rows: prefill step 4096 (the reference fork's own chunking option, which passes our quality gate); at step
2048 the same build is bit-identical to the reference fork and reads 987 / 872. Writing with the draft head (on by
default) gives the same tokens as writing without it. Writing rows use `MLX_VLM_INKLING2_DECODE_MIN_KEYS=1024` (patch
0011, on by default: on the fixed 8-prompt quality set its answers are identical to the reference fork's, 8/8;
log-probabilities are not bit-identical). Raw runs: [docs/BENCHMARKS.md](docs/BENCHMARKS.md#short-context-vs-mlx-serve-studio-2-second-m4-max-64-gb-2026-11-22).

### Before → after (standard build → this repo)

Measured on our first test Mac (same model of Mac; compare speeds within one table). Speeds are tokens per second. **Reading** = processing your prompt or document (prefill).
**Writing** = generating the answer (decode).

| Prompt length | Reading: before → after | Writing: before → after | Speed-up (reading / writing) |
|---|---|---|---|
| 4k tokens | 758.4 → **952.4** | 33.1 → **56.9** | 1.26× / 1.72× |
| 32k tokens | 667.9 → **867.3** | 27.0 → **53.7** | 1.30× / 1.99× |
| 131k tokens | 401.9 → **625.2** | 16.2 → **40.2** | 1.56× / 2.48× |
| 262k tokens | 256.4 → **453.9** | 10.7 → **29.8** | 1.77× / 2.78× |
| 524k tokens | not measured* → **292.6** | not measured* → **17.9** | – |
| 1M tokens | not measured* → **175.6** | not measured* → **11.4** | – |

The 1M row is today's build at prefill step 2048 with `MLX_VLM_RELATIVE_ATTN_KTILES=3 MLX_VLM_EXACT_HOST_VOCAB=1` (patches 0008/0009; same tokens and log-probabilities as the 163.3 tok/s run without them; was 156.0 / 10.0 with the previous build at step 512). The 524k row is still the previous build (writing ~10–12% slower than today's at 4k–262k); it will be re-run.

\* We have no 524k or 1M run on the standard build, so there is no "before" number (an earlier 524k run on the unpatched fork was stopped after 14 minutes, still reading). Ours read the 524k-token document in 30 minutes (peak 35.3 GiB of GPU memory) and the full 1M-token document (1,048,558 tokens) in 1 h 40 min (peak 50.1 of the 51.8 GiB GPU limit), and found the hidden fact both times.

The longer the prompt, the bigger the gain. At 32k–1M the model found a fact hidden in the middle of the document in every run, before and after.
Full benchmarks, methods and raw logs: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Follow-up questions in 1.5 s

Ask about a 131k-token document once: reading it takes about 3.6 minutes. Ask a follow-up, and the answer starts
in **1.5 s** (1.46–1.54 s measured), because the Mac keeps the document's processed state instead of reading it again.

## Speculative decoding, on by default: faster writing, same tokens

Inkling-2-Mini ships a small draft head (DSpark) that guesses several words ahead; the big model then checks
all the guesses in one pass. Usually that check is computed differently from normal one-word-at-a-time writing, so
the result is "the same words, almost always" (rounding can flip a near-tie). **Our check computes every guessed
position with exactly the same GPU arithmetic as normal writing** (same kernels, same order of additions, each
weight and each block of the document's memory read once for all guesses), so the output is **the same tokens as
writing without the draft**. The draft also reports how sure it is of each guess, and the check stops at the first
unlikely one instead of paying for guesses that would be thrown away (block 8, cut at 50% cumulative confidence).

| Writing speed (tok/s, second test Mac) | 4k prompt | 32k prompt | 131k prompt |
|---|---:|---:|---:|
| This repo, no draft | 55.6 | 48.9 | 36.4 |
| **This repo, default (draft head)** | **74.1** | **72.3** | **50.4** |

Every draft run gave exactly the no-draft tokens and found the hidden fact (131k: single cold run). The draft head
switches itself off for contexts under 1,024 tokens (the exact check can't run there yet, so drafting would be
slower: 33.9 vs 42.0 tok/s at 600 tokens) and over 262,144 tokens (not yet measured at 524k/1M, where memory is
tight; it adds ~1.1 GiB). `NO_DRAFT=1` turns it off. Env and details: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## What we changed

- **No more rebuilding the same work for every word.** Each step of writing an answer is compiled once (turned into a
  ready-made GPU program) and reused, instead of being rebuilt from scratch for every new token.
- **Memory-lean reading.** Prompts are read in chunks without keeping unneeded copies around, so long documents fit
  in 64 GB with room to spare.
- **Tuned GPU kernels.** The attention kernel (how each word looks back at the document) and the expert kernels (the
  model's mixture-of-experts layers) are tuned for the M4 Max, with bit-identical results.
- **Vocabulary tables off the GPU while reading huge prompts.** At 1M tokens the weights plus the reserved KV cache sit
  right at MLX's buffer-cache limit (0.95 × the recommended working set), and past it MLX's allocator frees its buffer
  cache on every miss, so every op gets fresh Metal buffers (+20–30% per layer). Keeping the two 0.86 GiB vocabulary
  tables (input embeddings and output head) in host memory during the chunked prefill, gathering each chunk's
  embedding rows on the CPU, and moving both back before the last token keeps the GPU under that line. Same outputs.
- **Instant follow-ups.** The processed state of a document is kept, so follow-up questions start in 1.5 s instead of 3.6 min.

## Quickstart

```bash
git clone https://github.com/tomascupr/inkling-mini-mac-longctx && cd inkling-mini-mac-longctx
bash scripts/install.sh                     # PREFIX=1 also adds the prefix-cache patch
hf download thinkingmachines/Inkling-2-Mini --local-dir ~/inkling-mini/models/Inkling-2-Mini

source scripts/env.sh                       # default env: fused prefill, draft head on (NO_DRAFT=1 for off)
```

`install.sh` downloads the base code as a zip, applies `patches/stacked.patch` and our patches (draft head
included; `NO_DRAFT=1` skips it) and sets up a venv with mlx 0.32.4. The server picks the draft head up from
`MLX_VLM_DRAFT_MODEL` (set by `scripts/env.sh`). For fast follow-ups, start the server with the prefix cache on (`APC_ENABLED=1`; see
`scripts/prefix_cache.sh`). Check your own build with `scripts/gate.sh`.

## Quality

- **Official weights**, exactly as Thinking Machines ships them (NVFP4 + MXFP8). No re-quantization, bf16 KV cache,
  no FP8 KV, no sparse or approximate attention.
- **Checked against the reference fork:** the default path is bit-identical to the unpatched community fork (same
  tokens and the same per-token log-probabilities, sha256) at 4k, 32k, 131k and 262k; at 1M the hidden fact is found and
  outputs match our earlier 1M run bit for bit. The new prefill kernels (attention residuals, message gates, MoE
  combine) reproduce MLX's own arithmetic order, so they are bit-identical too.
- **The draft head** (on by default) gives exactly the tokens of writing without it. **Not bit-identical, gated:**
  0011's decode kernels below 4,096 keys (on by default; same answers, 8/8), prefill step 4096 and the older
  approximate fusion kernels (opt-in). Each must pass our gate: needle found at 32k/131k/1M, the same greedy answers, ≥99% top-1 token agreement with the fork on a fixed prompt
  set (`scripts/quality/top1.py`).

Standard benchmarks (greedy, same prompts, 2026-11-22): **GSM8K** (first 100) 97% standard build vs **99%** this repo;
**MMLU-Pro** (140, 10 per subject) 78.6% vs **77.9%**; **RULER needle-in-a-haystack at 32k** (12) 12/12 both. The paired
differences are within noise (2 and 13 items differ; McNemar p = 0.5 and 1.0). This repo's output was token-for-token identical
to the community fork on all 100 GSM8K items; it differs from the standard build's wording by design (different
code), not in accuracy. Same run: 1.3× faster reading and 1.8× faster writing on the 32k items. Details:
[docs/BENCHMARKS.md](docs/BENCHMARKS.md#standard-quality-benchmarks).

## What's next

- **1M tokens on one Mac:** measured (above). Next: speculative decoding and the prefix cache at 1M.
- **Draft head at 524k/1M** (synthetic tests: 1M about 11 → 17 tok/s, memory is tight; off above 262k for now), in the server's batched path and with the prefix cache: not measured end to end yet.
- More prefill and decode headroom: we're at ~67–70% of the chip's compute limit on prefill and ~66–78% of its memory bandwidth on decode (32k–262k).

## Credits

Built on [betizzel/mlx-vlm `m5pro-inkling2`](https://github.com/betizzel/mlx-vlm/tree/m5pro-inkling2) (@55ca2e9e), which ports
jundot's `relative_attention` kernel from [ml-explore/mlx-lm#2086](https://github.com/ml-explore/mlx-lm/pull/2086); on
[inkling2-mlx](https://github.com/internet-projects/inkling2-mlx) (NetProjects), [MLX-VLM](https://github.com/Blaizzy/mlx-vlm)
(Prince Canuma and contributors) and Apple's [MLX](https://github.com/ml-explore/mlx). Inkling-2-Mini is by Thinking Machines Lab.
Our patches came from an exact memory track, a prefix-cache track, and two optimization tracks run with AI coding
agents (Claude on decode, Codex on prefill).

MIT licensed (see `LICENSE`); hunks touching the Apache-2.0 `inkling2_loop` files stay Apache-2.0.
