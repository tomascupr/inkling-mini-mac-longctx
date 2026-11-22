# Inkling-2-Mini long context on one Mac (M4 Max, 64 GB)

An **exact, full-quality** recipe for running Thinking Machines' Inkling-2-Mini
(official NVFP4 + MXFP8 weights, **bf16 KV cache**) at long context on a single
Mac Studio M4 Max with 64 GB, using MLX. It is a set of small patches on top of
an existing community fork, plus the scripts and logs that measure them.

"Exact" means: greedy output is **bit-identical** to the unpatched fork (same
token ids and the same sha256 over every per-token logprob vector). No FP8 KV,
no sparse or approximate attention, no relaxed Metal math.

Status: work in progress. Numbers below are measured unless marked *estimate*.

## Base

- **betizzel/mlx-vlm**, branch `m5pro-inkling2`, commit
  [`55ca2e9e3df180dc2c46550e44309d6af5489b86`](https://github.com/betizzel/mlx-vlm/tree/55ca2e9e3df180dc2c46550e44309d6af5489b86)
  (mlx-vlm `0.8.0+inkling2mini.1`), which carries the fused `relative_attention`
  prefill kernel from ml-explore/mlx-lm#2086.
- **mlx 0.32.4** / mlx-metal 0.32.4, Python 3.13.
- Weights: `thinkingmachines/Inkling-2-Mini` (Hub revision `ed40e7da`), loaded natively.
- Hardware: Mac Studio `Mac16,9`, M4 Max (40-core GPU), 64 GB. MLX working-set
  cap 51.84 GiB (`max_recommended_working_set_size`); `iogpu.wired_limit_mb` left at 0.

## Quickstart

```bash
git clone <this repo> && cd inkling-mini-mac-longctx
bash scripts/install.sh            # fork zip -> ~/inkling-mini/src/{base,stacked}, venv with mlx 0.32.4
#   PREFIX=1 bash scripts/install.sh   to also apply the prefix-cache patch (0004)
hf download thinkingmachines/Inkling-2-Mini --local-dir ~/inkling-mini/models/Inkling-2-Mini

# Recommended runtime env (scheduling only; output is unchanged)
export PYTHONPATH=~/inkling-mini/src/stacked
export MLX_VLM_EXACT_PREFILL_EVAL_EVERY=1 MLX_MAX_MB_PER_BUFFER=2
```

By hand, the same thing is:

```bash
curl -L -o fork.zip https://github.com/betizzel/mlx-vlm/archive/55ca2e9e3df180dc2c46550e44309d6af5489b86.zip
unzip fork.zip && cd mlx-vlm-55ca2e9e3df180dc2c46550e44309d6af5489b86
patch -p1 < /path/to/patches/stacked.patch
pip install "mlx==0.32.4" . torch torchvision   # torch/torchvision: needed by the official processor
```

**Prefix cache (follow-up turns in ~1.5 s):** the fork's server has an automatic
prefix cache (APC) that is **off by default**. Turn it on with `APC_ENABLED=1`
and give it room for one checkpoint (`APC_MEMORY_MAX_GB` ≈ 32 KiB × context + 2 GiB;
about 6 GiB at 131k). `APC_DISK_ENABLED=1` persists checkpoints to disk so a long
document survives a server restart. `scripts/prefix_cache.sh` has the full,
measured server launch and a two-turn test (`setup`, `server`, `ask`, `again`).
Don't pass `--max-kv-size`: with it the server summarises a conversation that
reaches 3/4 of it, replacing the document.

`install.sh` is assembled from the exact steps we ran (zip install, overlay on
`PYTHONPATH`), but has not yet been re-run end to end on a clean Mac.

## Measured results

Single Mac Studio M4 Max 64 GB, official weights, bf16 KV, batch 1, greedy,
needle (`VIOLET-ANCHOR-4729`) inserted mid-document in public-domain Project
Gutenberg text, prefill step 2048, 256 max new tokens. Needle found in every row.

| Context (prompt tokens) | Prefill tok/s | Decode tok/s | Peak MLX GiB | Stock prefill / decode / peak | Speed-up (prefill / decode) | Exact vs fork |
|---|---:|---:|---:|---|---|---|
| 32k (32,750) | **863.6** | **39.80** | **22.78** | 667.9 / 27.0 / 28.27 | 1.29× / 1.47× | tokens + logprob sha identical |
| 131k (131,054) | **623.6** | **31.20** | **25.78** | 401.9 / 16.2 / 33.58 | 1.55× / 1.93× | tokens + logprob sha identical |
| 262k (262,126), default env | **448.7** | **23.86** | **33.06** | 256.4 / 10.7 / 35.38 | 1.75× / 2.23× | needle found (no 262k fork reference run) |

| Prefix cache, 131k document + follow-up question | Time to first token |
|---|---:|
| Turn 1, cold (whole document prefilled) | 219.5 s |
| Turn 2, follow-up (131,006 tokens restored from the cache) | **1.54 s** |

- "Stock" = `internet-projects/inkling2-mlx` v0.2.0 (mlx-vlm `0.8.0+inkling2mini.2`) on the
  same Mac, same weights and prompts; it needed smaller prefill steps (1024 / 512 / 256) to fit.
- Unpatched betizzel fork at step 2048: 845.5 / 33.46 / 26.19 GiB at 32k and 605.1 / 27.13 / 29.62 GiB at 131k.
- 32k and 131k rows use the recommended env; the 262k row is the default MLX env
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

## Patches

| File | What | Author / track |
|---|---|---|
| `patches/stacked.patch` | **Use this.** 0001 + 0002 + 0003 combined, against the fork. Gated above. | |
| `patches/0001-exact-longctx-memory.patch` | Memory: last-token-only head (`logits_to_keep`), per-chunk text embedding, one-shot KV reservation (no concat growth), optional per-visit eval in prefill (`MLX_VLM_EXACT_PREFILL_EVAL_EVERY=1`). Switches: `MLX_VLM_EXACT_CHUNK_EMBED=0`, `MLX_VLM_EXACT_KV_RESERVE=0`. 32k peak 26.2 → 22.2 GiB. | memory track |
| `patches/0002-compiled-decode.patch` | Decode: one `mx.compile`d decode step (fewer graph nodes and kernels per token), wired into `generate_step`. Applies on top of 0001. | Claude track |
| `patches/0003-prefill-attn-moe.patch` | Prefill: relative-attention kernel tiles (BQ=64, fast mask) and a sort-free inverse permutation in the MoE (SwitchGLU) path, value-identical. Applies on top of 0001 + 0002. | Codex track |
| `patches/0004-prefix-cache.patch` | Prefix cache: keep the fused attention kernel after a cached prefix is restored (no full mask over the whole cache), keep the conversation's reasoning effort in compaction so it hits the checkpoint, tests. Applies to the fork, and on top of `stacked.patch` (with fuzz). | prefix-cache track |

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
| 32k | 10.95 | 1,297 | 863.6 | 66.6% |
| 131k | 15.78 | 900 | 623.6 | 69.3% |
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
| 32k | 6.74 | 12.34 | 81.0 | 39.80 | 49% |
| 131k | 9.96 | 18.24 | 54.8 | 31.20 | 57% |
| 262k | 14.26 | 26.11 | 38.3 | 23.86 | 62% |

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

## Future work

- Fuse the small prefill ops (router, message gate) and cut the decode CPU overhead further.
- Measured 524k and a true 1M needle run on this one Mac in the recommended config.
- FP8 global KV is parked until the full-quality bf16 path is exhausted.

## Credits

- **jundot** — the `relative_attention` fused prefill kernel and looped-model support,
  [ml-explore/mlx-lm#2086](https://github.com/ml-explore/mlx-lm/pull/2086).
- **betizzel** — the [mlx-vlm `m5pro-inkling2` fork](https://github.com/betizzel/mlx-vlm/tree/m5pro-inkling2)
  this recipe builds on (the #2086 kernel port, split-K decode, prefix-cache admission).
- **NetProjects / internet-projects** — [inkling2-mlx](https://github.com/internet-projects/inkling2-mlx), Inkling 2 Mini support for MLX-VLM.
- **Prince Canuma and contributors** — [MLX-VLM](https://github.com/Blaizzy/mlx-vlm); **Apple** — [MLX](https://github.com/ml-explore/mlx).
- **Thinking Machines Lab** — Inkling-2-Mini (Apache 2.0 + their acceptable use policy).
- Our patches: the exact memory patch (0001), the prefix-cache fix (0004), and two parallel
  optimization tracks run with AI coding agents — a Claude decode track (0002) and a Codex
  prefill track (0003). Prior art we surveyed: `docs/PRIOR_ART.md`.

## License

MIT (see `LICENSE`), matching MLX-VLM and the fork. Hunks that modify
`mlx_vlm/models/inkling2_loop/` touch files the fork ships under Apache 2.0 (adapted from
Hugging Face Transformers); those changes are offered under Apache 2.0 like the files they modify.
