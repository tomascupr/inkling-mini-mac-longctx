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
| Writing, 4k, draft head | 66.1 | **74.8–76.3** | **fastest** |
| Writing, 32k, draft head | 74.8 | 67.7 | behind (−9%), in progress |

Reading rows: prefill step 4096 (the reference fork's own chunking option, which passes our quality gate); at step
2048 the same build is bit-identical to the reference fork and reads 987 / 872. Writing with the draft head gives
the same tokens as writing without it. The 4k writing rows use opt-in `MLX_VLM_INKLING2_DECODE_MIN_KEYS=1024` (patch
0011: same tokens as the reference fork on this run, not bit-identical log-probabilities; quality gate in progress). Raw runs: [docs/BENCHMARKS.md](docs/BENCHMARKS.md#short-context-vs-mlx-serve-studio-2-second-m4-max-64-gb-2026-11-22).

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

## Optional: speculative decoding, much faster writing, still bit-identical

Inkling-2-Mini ships a small draft head (DSpark) that guesses several words ahead; the big model then checks
all the guesses in one pass. Usually that check is computed differently from normal one-word-at-a-time writing, so
the result is "the same words, almost always" (rounding can flip a near-tie). **Our check computes every guessed
position with exactly the same GPU arithmetic as normal writing** (same kernels, same order of additions, each
weight and each block of the document's memory read once for all guesses), so the output is **bit-for-bit the
same as writing without the draft**: same tokens and the same per-token log-probabilities. Install with
`EXACT_DSPARK=1` (patch `0007`):

| Writing speed (tok/s) | 32k prompt | 131k prompt | 262k prompt |
|---|---:|---:|---:|
| Standard build, no draft | 27.0 | 16.2 | 10.7 |
| This repo, no draft | 53.7 | 40.2 | 29.8 |
| **This repo + draft head (block 4), bit-identical** | **66.0** | **52.7** | **35.7** |

Each draft run matched the unpatched code's no-draft output exactly (tokens and log-probability sha256) and found
the hidden fact. The draft helps most on long documents, where every written word otherwise re-reads the whole
document's memory. Env and details: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

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

export PYTHONPATH=~/inkling-mini/src/stacked
export MLX_VLM_EXACT_PREFILL_EVAL_EVERY=1 MLX_MAX_MB_PER_BUFFER=2
export MLX_VLM_INKLING2_PREFILL_FUSED=1      # bit-identical fused prefill kernels (patch 0010)
```

`install.sh` downloads the base code as a zip, applies `patches/stacked.patch` and sets up a venv with
mlx 0.32.4. For fast follow-ups, start the server with the prefix cache on (`APC_ENABLED=1`; see
`scripts/prefix_cache.sh`). Check your own build with `scripts/gate.sh`.

## Quality

- **Official weights**, exactly as Thinking Machines ships them (NVFP4 + MXFP8). No re-quantization, bf16 KV cache,
  no FP8 KV, no sparse or approximate attention.
- **Checked against the reference fork:** the default path is bit-identical to the unpatched community fork (same
  tokens and the same per-token log-probabilities, sha256) at 4k, 32k, 131k and 262k; at 1M the hidden fact is found and
  outputs match our earlier 1M run bit for bit. The new prefill kernels (attention residuals, message gates, MoE
  combine) reproduce MLX's own arithmetic order, so they are bit-identical too.
- **Opt-in options** that are not bit-identical (prefill step 4096, older approximate fusion kernels) must pass our
  gate: needle found at 32k/131k/1M, the same greedy answers, ≥99% top-1 token agreement with the fork on a fixed prompt
  set (`scripts/quality/top1.py`).

Standard benchmarks (greedy, same prompts, 2026-11-22): **GSM8K** (first 100) 97% standard build vs **99%** this repo;
**MMLU-Pro** (140, 10 per subject) 78.6% vs **77.9%**; **RULER needle-in-a-haystack at 32k** (12) 12/12 both. The paired
differences are within noise (2 and 13 items differ; McNemar p = 0.5 and 1.0). This repo's output was token-for-token identical
to the community fork on all 100 GSM8K items; it differs from the standard build's wording by design (different
code), not in accuracy. Same run: 1.3× faster reading and 1.8× faster writing on the 32k items. Details:
[docs/BENCHMARKS.md](docs/BENCHMARKS.md#standard-quality-benchmarks).

## What's next

- **1M tokens on one Mac:** measured (above). Next: speculative decoding and the prefix cache at 1M.
- **Speculative decoding at 524k/1M** (synthetic tests: 1M about 11 → 17 tok/s, memory is tight) and combined with the prefix cache: not measured end to end yet.
- More prefill and decode headroom: we're at ~67–70% of the chip's compute limit on prefill and ~66–78% of its memory bandwidth on decode (32k–262k).

## Credits

Built on [betizzel/mlx-vlm `m5pro-inkling2`](https://github.com/betizzel/mlx-vlm/tree/m5pro-inkling2) (@55ca2e9e), which ports
jundot's `relative_attention` kernel from [ml-explore/mlx-lm#2086](https://github.com/ml-explore/mlx-lm/pull/2086); on
[inkling2-mlx](https://github.com/internet-projects/inkling2-mlx) (NetProjects), [MLX-VLM](https://github.com/Blaizzy/mlx-vlm)
(Prince Canuma and contributors) and Apple's [MLX](https://github.com/ml-explore/mlx). Inkling-2-Mini is by Thinking Machines Lab.
Our patches came from an exact memory track, a prefix-cache track, and two optimization tracks run with AI coding
agents (Claude on decode, Codex on prefill).

MIT licensed (see `LICENSE`); hunks touching the Apache-2.0 `inkling2_loop` files stay Apache-2.0.
