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
| 4k (4,078) | **951.8** | **50.64** | **19.46** | 758.4 / 33.09 / 25.58 | 1.25× / 1.53× | bit-identical to unpatched code |
| 32k (32,750) | **865.3** | **47.23** | **20.34** | 667.9 / 27.0 / 28.27 | 1.30× / 1.75× | bit-identical to unpatched code |
| 131k (131,054) | **624.3** | **35.80** | **23.34** | 401.9 / 16.2 / 33.58 | 1.55× / 2.21× | bit-identical to unpatched code |
| 262k (262,126), default env, previous patch version | **448.7** | **23.86** | **33.06** | 256.4 / 10.7 / 35.38 | 1.75× / 2.23× | needle found (no 262k unpatched reference run) |

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
- 4k, 32k and 131k rows: current `stacked.patch` (2026-11-22 03:20 CET), recommended env, gate log
  `results/gates/gate-20261122-0320.log`, runs `results/gates/gate-*.json`. The 4k reference is the unpatched fork at 4k
  (941.5 / 35.27 / 25.10 GiB). The 262k row is the previous patch version (before decode v2) in the default MLX env
  (`results/gates/gate-default-env.log`). The prefix-cache row is the fork's server
  with APC on plus `0004-prefix-cache.patch` (the unpatched fork also hits the cache,
  in about 2.0 s, once APC is on). 0004 has not yet been measured stacked with the speed patches.
- Raw data: `results/gates/` (patched runs), `results/reference/` (unpatched references),
  `results/runs/` (stock and unpatched-fork benchmark runs), `results/prefix-cache-131k-server.log`
  (server log: cold `time_to_first_token=218.274s`, follow-up `cached_tokens=131006`, `time_to_first_token=1.459s`;
  the table uses the client-side times), `results/LEADERBOARD.tsv` (every experiment).

Not yet measured with the patches: 524k and 1M. *Estimate:* with the recommended env
and step 512, a 1M-token context projects to ~50.6 GiB peak against the 51.84 GiB cap
(32 GiB of bf16 global KV + 17.6 GiB text weights + ~1.0 GiB per-step working memory).

4k row: "before" measured 2026-11-22 02:25 CET (Europe/Prague) on the unmodified standard build with the same
`bench_longctx.py` and settings as the other baseline runs (bf16 KV, no draft, step 2048, 256 max tokens, needle found;
`results/runs/stock-v0.2.0-4k-20261122.json`). "After" (948.0 / 40.98) is a single patched run at 4k from our
optimization log, step 2048; it was not part of the exactness gate.

## Patches

| File | What | Author / track |
|---|---|---|
| `patches/stacked.patch` | **Use this.** 0001 + 0002 + 0003 combined, against the fork. Gated above. | |
| `patches/0001-exact-longctx-memory.patch` | Memory: last-token-only head (`logits_to_keep`), per-chunk text embedding, one-shot KV reservation (no concat growth), optional per-visit eval in prefill (`MLX_VLM_EXACT_PREFILL_EVAL_EVERY=1`). Switches: `MLX_VLM_EXACT_CHUNK_EMBED=0`, `MLX_VLM_EXACT_KV_RESERVE=0`. 32k peak 26.2 → 22.2 GiB. | memory track |
| `patches/0002-compiled-decode.patch` | Decode: one `mx.compile`d decode step (fewer graph nodes and kernels per token), wired into `generate_step`; v3 (2026-11-22 03:20) adds about a dozen fused kernels that copy MLX's own arithmetic order (attention-residual mixing, router, MoE output combine with its short convolution, l2-clips, norms, conv-state updates, merged attention projections), a decode attention kernel whose cost follows the live context length rather than the reserved KV size, and a fix so the saved conv state no longer keeps a whole prefill chunk alive (prefill working memory 0.43 GiB per 512-token step; projected 1M peak ~50.0 GiB). All bit-identical. Applies on top of 0001. | Claude track |
| `patches/0003-prefill-attn-moe.patch` | Prefill: relative-attention kernel tiles (BQ=64, fast mask, global visits only) and a sort-free inverse permutation in the MoE (SwitchGLU) path, value-identical. Applies on top of 0001 + 0002. (Codex's compiled loop message gate, +0.6% prefill, is left out: it disabled the compiled decode path.) | Codex track |
| `patches/0004-prefix-cache.patch` | Prefix cache: keep the fused attention kernel after a cached prefix is restored (no full mask over the whole cache), keep the conversation's reasoning effort in compaction so it hits the checkpoint, tests. Applies to the fork, and on top of `stacked.patch` (with fuzz). | prefix-cache track |
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
| 262k | 22.22 | 639 | 448.7 | 70.2% |

At 524k this is 18.4 PFLOP for the whole prompt (73% global attention), a floor of
~19–22 min (*estimate*). Where prefill time goes now: big matmuls run at ~96% of peak and
expert GEMMs ~79%, but small ops (router ~29%, message gate ~14%) pull a whole chunk to ~66%.
Details: `docs/trackA-prefill-notes.md`, `docs/PREFILL_TECHNIQUES.md` §7.

**Decode (bandwidth floor).** Each token must read the active weights once — 5.50 GB
(bf16 attention projections 2.58, 8 routed experts per visit 1.84, head 0.93, the rest
0.15; `results/decode-weight-bytes.json`) — plus 0.17 GB of sliding KV and
32 KiB × S of global KV. Floor = bytes / 546 GB/s:

| Context | GB/token | Floor ms | Ceiling tok/s | Measured | % of ceiling |
|---|---:|---:|---:|---:|---:|
| 32k | 6.74 | 12.34 | 81.0 | 47.23 | 58% |
| 131k | 9.96 | 18.24 | 54.8 | 35.80 | 65% |
| 262k | 14.26 | 26.11 | 38.3 | 23.86 (previous version) | 62% |

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
