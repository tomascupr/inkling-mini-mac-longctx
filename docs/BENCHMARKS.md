# Benchmarks, methodology and caveats

All numbers measured on one Mac Studio M4 Max (40-core GPU, 64 GB), 2026-11-21/22,
mlx 0.32.4, official `thinkingmachines/Inkling-2-Mini` weights (rev `ed40e7da`, NVFP4 + MXFP8), bf16 KV.

**Baseline ("stock")** = the Inkling 2 Mini MLX path people run today and that we served live:
MLX-VLM `0.8.0+inkling2mini.2` from [internet-projects/inkling2-mlx](https://github.com/internet-projects/inkling2-mlx) v0.2.0,
mlx 0.32.4, same weights, prompts and Mac (`results/runs/stock-v0.2.0-baseline-20261121.json`).

## Measured results (full table)

Single Mac Studio M4 Max 64 GB, official weights, bf16 KV, batch 1, greedy,
needle (`VIOLET-ANCHOR-4729`) inserted mid-document in public-domain Project
Gutenberg text, prefill step 2048, 256 max new tokens. Needle found in every row.

| Context (prompt tokens) | Prefill tok/s | Decode tok/s | Peak MLX GiB | Stock prefill / decode / peak | Speed-up (prefill / decode) | Exactness |
|---|---:|---:|---:|---|---|---|
| 4k (4,078) | **952.4** | **56.85** | **19.46** | 758.4 / 33.09 / 25.58 | 1.26× / 1.72× | bit-identical to unpatched code |
| 32k (32,750) | **867.3** | **53.73** | **20.34** | 667.9 / 27.0 / 28.27 | 1.30× / 1.99× | bit-identical to unpatched code |
| 131k (131,054) | **625.2** | **40.16** | **23.36** | 401.9 / 16.2 / 33.58 | 1.56× / 2.48× | bit-identical to unpatched code |
| 262k (262,126) | **453.9** | **29.76** | **27.35** | 256.4 / 10.7 / 35.38 | 1.77× / 2.78× | bit-identical to unpatched code |
| 524k (524,270), previous build | **292.6** | **17.86** | **35.35** | not measured on the standard build | – | needle found (no 524k unpatched reference run) |
| 1M (1,048,558), step 2048, KT=3 + host vocab, text-only | **175.6** | **11.38** | **50.08** | not measured on the standard build | – | needle found; tokens + logprob sha equal to the 163.3 run (no 1M unpatched reference run) |
| 1M, step 2048, text-only, without 0008/0009 env | 163.3 | 11.36 | 50.76 | | | needle found |
| 1M, step 512, text-only, previous build | 156.0 | 10.04 | 50.04 | | | needle found |

## Short context vs MLX-Serve (Studio 2, second M4 Max 64 GB, 2026-11-22)

Same official weights. Measured on Studio 2 only (it runs ~5% slower than Studio 1 on the same code, so compare
within one machine). MLX-Serve figures are its own server-log numbers on Studio 2 (prefill chunk 2048, block-8 DSpark
draft). Ours: `MLX_VLM_INKLING2_PREFILL_FUSED=1` (patch 0010), `scripts/exact_check.py`, 256 max new tokens, warm run.

| | MLX-Serve | Ours, step 2048 (bit-identical to fork) | Ours, step 4096 (top-1 ≥99%) |
|---|---:|---:|---:|
| Reading 4k | 1010–1012 | 987.0 | **1013–1037** (5 warm runs) |
| Reading 32k | 877 | 872.4 | **884–888** |
| Writing 4k, no draft | 56.4 | 52.6 | 52.9; **55.1–55.9** with `DECODE_MIN_KEYS=1024` (0011) |
| Writing 32k, no draft | 50.2 | 48.6 | 48.9 |
| Writing 4k, draft | 66.1 (block 8) | | 62.6 (block 5); **74.8–76.3** with 0011 (k6 66.5, k8 62.0) |
| Writing 32k, draft | 74.8 (block 8) | | 67.7 (block 5; k4 64.6, k8 59.4) |

Step 2048 rows: tokens and logprob sha equal to the unpatched fork, needle found (`results/gates/studio2/p10b-*.json`).
Draft rows: patch 0007 (exact DSpark verify) + 0010, tokens identical to the no-draft run, `results/gates/studio2/draft/`.
0011 rows: tokens identical to the unpatched fork (4k needle run) and on the fixed 8-prompt set (greedy 8/8, teacher-forced top-1 100% with step 2048; `results/gates/studio2/top1/top1-mk1024.*` vs `top1-s13gate.*`). Still behind MLX-Serve on no-draft writing and on drafted writing at 32k; work in progress.
Step 4096 changes only the prefill chunking (fork vs itself at step 4096: teacher-forced top-1 99.6% on the fixed
8-prompt set, needle answers identical); the fused kernels are bit-identical on top of it (since 17:30 also on short chunks: 8/8 greedy, 100% teacher-forced top-1 at step 2048, `top1-s13gate.*`).

**Bit-identical speculative decoding (`0007`, 2026-11-22 11:53–12:08 CET).** `stacked.patch` + `0007-exact-dspark.patch`,
env `MLX_VLM_INKLING2_INCREMENTAL_DRAFT_PREFILL=1 MLX_VLM_INKLING2_EXACT_VERIFY=1 MLX_VLM_INKLING2_COMPILED_VERIFY=1
MLX_VLM_INKLING2_DRAFT_CTX_WINDOW=65536` plus the recommended env, draft block 4, `scripts/exact_check_draft.py`
(`BLOCK=4`), one process; log `results/gates/gate-stack7-draft-20261122-1153.log`, runs `results/gates/s7-draft-k4-*.json`:

| Context | Reading tok/s | Writing tok/s (no draft → draft) | Accepted / drafted | Peak GiB | Tokens and logprob sha vs unpatched fork (no draft) |
|---|---:|---:|---:|---:|---|
| 32k | 864.9 | 53.73 → **65.99** | 77% | 21.41 | identical |
| 131k | 622.7 | 40.16 → **52.67** | 77% | 24.41 | identical |
| 262k | 452.2 | 29.76 → **35.72** | 59% | 28.41 | identical |

How: the verifier computes each of the k block rows with the single-token decode step's own kernels and reduction
order (multi-row kernels that share weight, expert and KV-block reads across rows but keep each row's arithmetic),
writes only accepted rows into the caches, and keeps the drafter's view of the document in a fixed 64k-token window
(the stock drafter copied its whole context every round: 4 GiB at 1M). Built by the decode track on top of the DSpark
track's compiled verify (0005). Synthetic-depth tests (zero KV at depth, short prompt on top; not end to end): 524k
19.5 → 28.8 tok/s, 1M 11.2 → 17.4 tok/s with peak 51.30 of 51.84 GiB.

**Speculative decoding (DSpark) is not in the table above.** Every row compares the standard build and our build
*without* a draft model. The standard build can already use Inkling's DSpark draft head: on the same Mac, 4k prompt,
it writes at **58.8 tok/s** with draft block 5 (40.8 with block 3, 56.3 with block 7;
`results/runs/stock-v0.2.0-dspark-4k-20261122.json`).

With the optional `patches/0005-dspark-compiled-verify.patch` on top of `stacked.patch` and
`MLX_VLM_INKLING2_INCREMENTAL_DRAFT_PREFILL=1 MLX_VLM_INKLING2_COMPILED_VERIFY=1 MLX_VLM_INKLING2_SPLITK_VERIFY=shared`
(plus the recommended env), one fresh process, runs in this order
(`results/gates/dspark-compiled-verify-4k-32k-131k.json`, `results/gates/gate-dspark-20261122-0412.log`):

| Run | Draft block | Writing tok/s | Same tokens as no-draft run |
|---|---:|---:|---|
| 4k, no draft | – | 50.07 | (reference; logprob sha also equal to `stacked.patch`) |
| 4k, first draft request (includes one-time draft-head setup) | 5 | 63.53 | yes |
| 4k | 5 | **65.52** | yes |
| 32k, no draft | – | 47.30 | (reference; logprob sha also equal) |
| 32k | 5 | 59.90 | yes |
| 32k | 8 | **63.83** | yes |
| 131k | 8 | **53.52** | yes |

Without `MLX_VLM_INKLING2_SPLITK_VERIFY=shared` the 131k draft run was *slower* than no draft (24.2 tok/s), so set it.
What 0005 does: the draft head's view of the prompt is captured incrementally per prefill chunk instead of as four
full-prompt histories joined at the end; the big model's check of a draft block runs as one compiled graph with
offset-only rollback of the global KV cache; and the global-attention check reads the long KV once per block with a
shared split-K kernel for all block rows, instead of once per row. With 0005 applied and no draft model, outputs stay
bit-identical (tokens + logprob sha at 4k/32k/131k). Speculative runs are checked on output tokens only.
0005 does not combine with 0004 yet (one hunk conflicts).

| Prefix cache, 131k document + follow-up question | Time to first token |
|---|---:|
| Turn 1, cold (whole document prefilled) | 219.5 s |
| Turn 2, follow-up (131,006 tokens restored from the cache) | **1.54 s** |

- "Stock" = `internet-projects/inkling2-mlx` v0.2.0 (mlx-vlm `0.8.0+inkling2mini.2`) on the
  same Mac, same weights and prompts; it needed smaller prefill steps (1024 / 512 / 256) to fit.
- Starting code (unpatched betizzel fork) at step 2048, the reference for the bit-identity check: 845.5 / 33.46 / 26.19 GiB at 32k and 605.1 / 27.13 / 29.62 GiB at 131k.
- 4k–262k rows (updated 2026-11-22 11:08–11:24 CET): current `stacked.patch` (= 0001 + the decode track's final 0002 + 0003),
  recommended env, one gate run `results/gates/gate-stack6-20261122-1108.log`, runs `results/gates/s6-*.json`; tokens
  AND logprob sha256 identical to the unpatched fork at all four lengths (references below). Reading is unchanged from
  the previous build (within 0.3%); writing is +10–14% (previous build: 50.64 / 47.23 / 35.80 / 27.01).
  The 524k row is the previous build (`stacked-v4.patch`). The 1M row is the current `stacked.patch` at step 2048 (below).
- Previous build's 4k, 32k and 131k rows: `stacked-v4.patch` (2026-11-22 03:20 CET), recommended env, gate log
  `results/gates/gate-20261122-0320.log`, runs `results/gates/gate-*.json`. The 4k reference is the unpatched fork at 4k
  (941.5 / 35.27 / 25.10 GiB). The 262k and 524k rows are the headline runs
  (`results/gates/headline-20261122.log`, `results/gates/h-524k.json`, `results/gates/h-262k.json`); the 262k
  reference is the unpatched fork at 262k, step 2048 (439.0 / 21.81 / 34.18 GiB, `results/gates/h-262k-stock.json`). The prefix-cache row is the fork's server
  with APC on plus `0004-prefix-cache.patch` (the unpatched fork also hits the cache,
  in about 2.0 s, once APC is on). 0004 has not yet been measured stacked with the speed patches.
- Raw data: `results/gates/` (patched runs), `results/reference/` (unpatched references),
  `results/runs/` (stock and unpatched-fork benchmark runs), `results/prefix-cache-131k-server.log`
  (server log: cold `time_to_first_token=218.274s`, follow-up `cached_tokens=131006`, `time_to_first_token=1.459s`;
  the table uses the client-side times), `results/LEADERBOARD.tsv` (every experiment).

1M, current build + 0008/0009: measured 2026-11-22 14:30–16:10 CET on Studio 1 (`results/gates/h7-1m-kt3-hostvocab.json`, `.log`):
1,048,558 prompt tokens read at 175.6 tok/s (+7.5% vs 163.3), 95 tokens written at 11.38 tok/s, needle found, peak 50.08 GiB.
Env: recommended env + `MLX_VLM_RELATIVE_ATTN_KTILES=3 MLX_VLM_EXACT_HOST_VOCAB=1`. Tokens and logprob sha256 identical to
the 163.3 run below (same step), so both patches are bit-identical at 1M too.

1M, current build without 0008/0009: measured 2026-11-22 12:20–14:07 CET (`results/gates/h6-1m-step2048.json`, `results/gates/headline-h6-20261122.log`):
1,048,558 prompt tokens read in 6,420 s (163.3 tok/s, +4.7% vs 156.0), 95 tokens written at 11.36 tok/s (+13% vs 10.04), needle found,
peak 50.76 GiB against the 51.84 GiB cap. Settings: current `stacked.patch`, recommended env, **step 2048** (fits since the
conv-state fix; synthetic-depth probe predicted 50.58 GiB), vision tower dropped, default max context, no sysctl. It wrote the
same answer as the step-512 run, but the tokens differ from position 9 (both 95 tokens) and so does the logprob sha: the
fork itself is not chunk-size invariant (step 512 vs 2048 also changes the fork's own logprobs), and there is no unpatched 1M reference at either step.

1M, previous build: measured 2026-11-22 05:15–07:07 CET (`results/gates/h-1m.json`, `results/gates/headline-20261122.log`):
1,048,558 prompt tokens read in 6,721 s (156.0 tok/s), 95 tokens written at 10.04 tok/s, needle found, peak 50.04 GiB
against the 51.84 GiB cap (projection was ~50.0). Settings: recommended env, step 512, vision tower dropped
(`scripts/exact_text_only.py`, text logits unchanged), default max context, no sysctl.

4k row: "before" measured 2026-11-22 02:25 CET (Europe/Prague) on the unmodified standard build with the same
`bench_longctx.py` and settings as the other baseline runs (bf16 KV, no draft, step 2048, 256 max tokens, needle found;
`results/runs/stock-v0.2.0-4k-20261122.json`). "After" (948.0 / 40.98) is a single patched run at 4k from our
optimization log, step 2048; it was not part of the exactness gate.

## Patches

| File | What | Author / track |
|---|---|---|
| `patches/stacked.patch` | **Use this.** 0001 + 0002 + 0003 combined, against the fork. Gated above. | |
| `patches/stacked-v4.patch` | The previous combined build (2026-11-22 03:20; behind the 524k/1M rows and the quality benchmark). `install.sh` uses it with `DSPARK=1`, because 0005 was built on it. | |
| `patches/0001-exact-longctx-memory.patch` | Memory: last-token-only head (`logits_to_keep`), per-chunk text embedding, one-shot KV reservation (no concat growth), optional per-visit eval in prefill (`MLX_VLM_EXACT_PREFILL_EVAL_EVERY=1`). Switches: `MLX_VLM_EXACT_CHUNK_EMBED=0`, `MLX_VLM_EXACT_KV_RESERVE=0`. 32k peak 26.2 → 22.2 GiB. | memory track |
| `patches/0002-compiled-decode.patch` | Decode: one `mx.compile`d decode step (fewer graph nodes and kernels per token), wired into `generate_step`; v4 (2026-11-22 08:16, current) adds a fused MoE decode (router → top-k + gate/up + SwiGLU, then down + combine), o_proj with the attention gate and tail, fused message gates, multi-query attention residuals, a custom bf16 GEMV for the projections and an MLX-replica sliding-window attention with the bias read in place; v3 (03:20) added about a dozen fused kernels that copy MLX's own arithmetic order (attention-residual mixing, router, MoE output combine with its short convolution, l2-clips, norms, conv-state updates, merged attention projections), a decode attention kernel whose cost follows the live context length rather than the reserved KV size, and a fix so the saved conv state no longer keeps a whole prefill chunk alive (prefill working memory 0.43 GiB per 512-token step; projected 1M peak ~50.0 GiB). All bit-identical. Applies on top of 0001. | Claude track |
| `patches/0003-prefill-attn-moe.patch` | Prefill: relative-attention kernel tiles (BQ=64, fast mask, global visits only) and a sort-free inverse permutation in the MoE (SwitchGLU) path, value-identical. Applies on top of 0001 + 0002. (Codex's compiled loop message gate, +0.6% prefill, is left out: it disabled the compiled decode path.) | Codex track |
| `patches/0004-prefix-cache.patch` | Prefix cache: keep the fused attention kernel after a cached prefix is restored (no full mask over the whole cache), keep the conversation's reasoning effort in compaction so it hits the checkpoint, tests. Applies to the fork, and on top of `stacked.patch` (with fuzz). | prefix-cache track |
| `patches/0007-exact-dspark.patch` | On top of `stacked.patch`: bit-identical speculative decoding with the DSpark draft head (exact multi-row verifier, windowed draft context; includes the 0005 compiled-verify port). Without a draft model, output is bit-identical to `stacked.patch`. | decode track (Claude) + DSpark track (Codex) |
| `patches/0006-prefix-cache-evict-fix.patch` | Prefix cache, on top of 0004: port of betizzel/mlx-vlm 295f5b21. When a follow-up's checkpoint extends the one it continues, storing it no longer also reserves room for the next prefill, so it can replace the old checkpoint instead of being dropped (upstream saw the 2nd follow-up at 125k re-read the whole conversation: 295.8 s). Optional `APC_EVICT_SUPERSEDED=1` evicts superseded turns. Bookkeeping only, no math. Confirmed on our base: the upstream tests fail without it (3 of 726 in `test_cache.py`) and all 726 pass with it. Not yet re-measured end to end on a 3-turn 131k conversation. | betizzel (ported) |
| `patches/0005-dspark-compiled-verify.patch` | Optional, on top of `stacked.patch`: faster speculative decoding with the DSpark draft head (incremental draft-prompt capture, compiled block verify, offset-only rollback, shared split-K verify attention). Opt-in via the three env vars above; without a draft model, outputs are bit-identical to `stacked.patch`. Not yet combinable with 0004. | DSpark track (Codex) |

All patches are `patch -p1` from the fork's source root (paths `a/mlx_vlm/...`).
Applying 0001 → 0002 → 0003 in order gives a tree identical to `stacked.patch` (checked).

## Scripts

| Script | Purpose |
|---|---|
| `scripts/install.sh` | Fork zip → unpatched + patched source trees and a venv. |
| `scripts/gate.sh` | Exactness + speed gate: unpatched vs patched, token ids and logprob sha, at 32k / 131k (set `CTXS`). |
| `scripts/exact_check.py` | One greedy needle run: tokens, sha256 over all logprob vectors, prefill/decode tok/s, MLX peak. |
| `scripts/probe_exact.py` | Synthetic-depth memory probe (pre-sized zero KV, then a real short prefill + decode). Not a quality test. |
| `scripts/memprof.py` | Per-chunk / per-visit working-memory profiler. |
| `scripts/exact_text_only.py` | Drops the vision tower for text-only runs (0.59 GiB; text logits unchanged). |
| `scripts/prefix_cache.sh`, `scripts/followup.py` | Prefix-cache server launch and the two-turn follow-up measurement. |
| `scripts/test_inkling2_prefix_cache.py` | Unit tests for 0004 (also added by the patch under `mlx_vlm/tests/`). |

Paths default to `~/inkling-mini` (`ROOT`), with the model in `$ROOT/models/Inkling-2-Mini`
and the corpus in `$ROOT/corpus.txt` (any plain text longer than the context; we used
~6.9 MB of public-domain Project Gutenberg books). Run one GPU job at a time.

## Roofline method

Machine limits: **14.2 TFLOP/s** fp16/bf16 FMA (measured on M4 Max by third parties; there is
no 2× fp16 matrix rate on M4; 16.2 TFLOP/s on paper) and **546 GB/s** DRAM bandwidth (spec;
~444–458 GB/s achievable in our own read benchmarks).

**Prefill (corrected FLOP model).** Per prompt token, counting a multiply-add as 2 FLOPs:
dense work over the 48 layer visits (24 layers, middle 12 looped 3×) is **8.84 GFLOP**
(attention projections, router, 8-of-128 routed experts, loop message gates); the 40
sliding-window visits add **0.50 GFLOP** (1,024 keys each); the 8 global visits add
`8 × 4 × Hq × D × S_avg` = 98,304 × S_avg FLOP, where **S_avg = N/2** for a causal prompt of
N tokens. (An earlier count charged every token the last chunk's depth, which overstated
131k efficiency as ~95%; that was wrong.) Ceiling tok/s = 14.2e12 / FLOP per token:

| Context | GFLOP/token | Ceiling tok/s | Measured | % of ceiling |
|---|---:|---:|---:|---:|
| 32k | 10.95 | 1,297 | 865.3 | 66.7% |
| 131k | 15.78 | 900 | 624.3 | 69.4% |
| 262k | 22.22 | 639 | 453.6 | 71.0% |
| 524k | 35.11 | 404 | 292.6 | 72.4% |
| 1M (step 2048, KT=3 + host vocab) | 60.88 | 233 | 175.6 | 75.4% |
| 1M (step 2048) | 60.88 | 233 | 163.3 | 70.0% |
| 1M (step 512, previous build) | 60.88 | 233 | 156.0 | 67.0% |

At 524k this is 18.4 PFLOP for the whole prompt (73% global attention), a floor of
~21.6 min at peak; measured 29.9 min. Where prefill time goes now: big matmuls run at ~96% of peak and
expert GEMMs ~79%, but small ops (router ~29%, message gate ~14%) pull a whole chunk to ~66%.
Details: `docs/trackA-prefill-notes.md`, `docs/PREFILL_TECHNIQUES.md` §7.

**Decode (bandwidth floor).** Each token must read the active weights once — 5.50 GB
(bf16 attention projections 2.58, 8 routed experts per visit 1.84, head 0.93, the rest
0.15; `results/decode-weight-bytes.json`) — plus 0.17 GB of sliding KV and
32 KiB × S of global KV. Floor = bytes / 546 GB/s:

| Context | GB/token | Floor ms | Ceiling tok/s | Measured | % of ceiling |
|---|---:|---:|---:|---:|---:|
| 32k | 6.74 | 12.34 | 81.0 | 53.73 | 66% |
| 131k | 9.96 | 18.24 | 54.8 | 40.16 | 73% |
| 262k | 14.26 | 26.11 | 38.3 | 29.76 | 78% |
| 524k (previous build) | 22.85 | 41.85 | 23.9 | 17.86 | 75% |
| 1M (KT=3 + host vocab) | 40.03 | 73.3 | 13.6 | 11.38 | 84% |
| 1M | 40.03 | 73.3 | 13.6 | 11.36 | 83% |
| 1M (previous build) | 40.03 | 73.3 | 13.6 | 10.04 | 74% |

The remaining decode gap is mostly CPU-side graph build/encode (~17 ms/token) and CPU↔GPU
coupling, not GPU work: with large command buffers the GPU alone needs ~19.8 ms/token at 32k. Details: `docs/trackB-decode-notes.md`.

## Exactness methodology

1. Fixed prompt: Project Gutenberg text truncated to the target context, needle sentence
   inserted at the first paragraph break past the midpoint, question at the end.
2. Greedy decoding (`temperature=0`), fixed `prefill_step_size` (2048), 256 max tokens.
3. For every generated token, record the token id and hash the full float32 logprob vector
   into one running sha256 (`scripts/exact_check.py`).
4. Run the **unpatched fork** and the **patched build** in the same venv (only `PYTHONPATH`
   differs) and require identical token lists **and** identical logprob sha. Needle must be found.
5. Promotion ladder: kernel microbenchmarks must be bitwise equal (or max-abs-diff 0) on
   random inputs → 32k e2e gate → 131k e2e gate. A variant that changes any bit is rejected
   or reported as non-exact (see `exact` column in `results/LEADERBOARD.tsv`).

Caveat: the fork itself is not chunk-size invariant (at step 512 vs 2048 tokens match but the
logprob sha differs), so references must use the same step. Environment knobs used here
(`MLX_MAX_MB_PER_BUFFER`, `MLX_VLM_EXACT_PREFILL_EVAL_EVERY`) only change scheduling and were
gated bit-identical.

## Repo hygiene
Result files had local usernames in paths replaced with `~`; machine-specific fields
(process lists, network details) were removed from the run JSONs. Numbers are unchanged.

## Standard quality benchmarks

Run 2026-11-22 08:59–11:09 CET on Studio 1 under the GPU lock (nothing else on the GPU). Script, sample IDs, raw
per-item JSONL (tokens, text, timings), telemetry and summary: `results/quality-20261122/`.

**Builds.** *Standard* = inkling2-mlx v0.2.0 (MLX-VLM `0.8.0+inkling2mini.2`). *This repo* = the `stack4` build
(the one behind the README table: fork + 0001 + 0002 + 0003), recommended env (`MLX_VLM_EXACT_PREFILL_EVAL_EVERY=1
MLX_MAX_MB_PER_BUFFER=2`). *Fork control* (GSM8K only) = the community fork with our exact memory patch 0001.
Same weights (rev `ed40e7da`), same chat template (`apply_chat_template`, `reasoning_effort="low"`), greedy
(temperature 0), prefill step 2048, batch 1, one request at a time via `stream_generate`.

**Data** (public HF datasets, pinned revisions in `datasets.json`; prompts built by `prep.py`, sha256 of the prompt
file `0fc9d478…`):
- GSM8K `openai/gsm8k@740312ad` test, first 100. Prompt asks for `Final answer: <number>`; exact numeric match. max 2048 tokens.
- MMLU-Pro `TIGER-Lab/MMLU-Pro@b189ec76` test, 10 per category × 14 (seed 0). Last `Answer: <letter>`. max 2048 tokens.
- RULER `SaylorTwift/RULER-32768-llama-3.2-tokenizer@e748e0cd`, niah_single_1/2/3, 4 each (cut from 20 to 12 for
  wall time). ~34.4k Inkling tokens each. Correct if every gold string is in the final answer. max 512 tokens.

| Benchmark | n | Standard build | This repo | Paired (std only / repo only) | Token-identical items | Truncated (std / repo) |
|---|---:|---:|---:|---|---:|---|
| GSM8K | 100 | 97 (97.0%, 95% CI 91.5–99.0) | **99** (99.0%, 94.6–99.8) | 0 / 2, McNemar p = 0.50 | 23 / 100 | 0 / 0 |
| MMLU-Pro | 140 | 110 (78.6%, 71.1–84.6) | 109 (77.9%, 70.3–83.9) | 7 / 6, p = 1.00 | 11 / 140 | 11 / 8 |
| RULER NIAH 32k | 12 | 12 (100%, 75.7–100) | 12 (100%, 75.7–100) | 0 / 0 | 4 / 12 | 0 / 0 |

CIs are Wilson 95%. Neither difference is significant; the samples are small (a 1-item swing on GSM8K is 1 point).

**Why tokens differ from the standard build.** The standard build is different code (a different MLX-VLM build), so
greedy decoding drifts apart by rounding at the first near-tie: first divergence at median token 67 (GSM8K) / 40
(MMLU-Pro), as early as token 2. After that the two builds write different but equally valid text (97/100 GSM8K
items still get the same answer and score). **Against the community fork (our base), this repo was token-for-token
identical on all 100 GSM8K items** (fork control, 99/100 correct, same as ours), which is the bit-exactness claim.

**MMLU-Pro truncation caveat.** With low reasoning effort the model sometimes keeps reasoning past 2048 tokens
(11 standard / 8 repo; engineering 6 / 2 and law 2 / 2 are the most common). Truncated items mostly have no parseable
answer and count as wrong for both builds: 10 / 7 unparsed.

**Speed in the same run** (medians per request, `stream_generate`'s own counters):

| | GSM8K (~108-token prompts) | MMLU-Pro (~219) | RULER 32k (~34.4k) |
|---|---|---|---|
| Reading, tok/s (std → repo) | 354 → 386 | 535 → 565 | 649 → **865** |
| Writing, tok/s | 37.1 → 42.9 | 36.1 → 42.5 | 26.5 → **47.3** |
| Time to first token | 0.28 → 0.26 s | 0.36 → 0.34 s | 53.5 → 40.1 s |
| Peak MLX memory | 20.3 → 18.5 GiB | 21.3 → 18.8 GiB | 32.7 → 20.5 GiB |
| Wall time (all items) | 803 → 698 s | 2,370 → 1,851 s | 699 → 511 s |

No errors, no stalls (largest gap between consecutive telemetry samples 2.1 s at a 2 s interval; no gaps between
requests), no swap use. GPU telemetry from `macmon` (no sudo), every 2 s: GPU busy ~98–99% throughout; GPU power
~14–18 W while writing short answers vs ~51–55 W while reading 32k prompts; max GPU temperature 84 °C.

**Improvement leads seen in this run** (ranked):
1. *Short-prompt writing is slower than 32k writing.* This repo writes 42.5–43.0 tok/s after 100–200-token prompts
   but 47.3 tok/s after 34k-token prompts (and 50.6 at 4k in `exact_check`). The rate doesn't depend on answer
   length (42.1 for answers over 1,000 tokens), so it isn't warm-up. The likely cause is a different decode path when
   the global KV is short (the split-K decode attention is used from S >= 4096). Worth up to ~+15% on chat-style use.
2. *Writing is latency-bound, not bandwidth-bound.* "GPU busy 98%" at only ~14 W while writing (vs ~55 W reading)
   means the GPU waits on many small dependent kernels: the same gap as the 58–66%-of-bandwidth decode numbers.
   Fewer, fused kernels per token (the decode track's work) is the lever.
3. *Short-prompt reading is overhead-bound*: 386 tok/s at ~108 tokens vs 865 at 32k; time to first token is ~0.26 s
   either way, so it barely matters for chat.
4. *MMLU-Pro truncations* are a model/effort setting, not a speed issue: medium effort or a larger token budget
   would remove most of them in a future run.
