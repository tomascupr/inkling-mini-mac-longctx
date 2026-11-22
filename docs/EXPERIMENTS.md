# Experiment log (newest first)

Plain log of what we tried after the 1M headline, including what failed. Times are CET (Europe/Prague).
Gates: bit-identical tokens + logprob sha256 vs the unpatched fork at 4k/32k/131k/262k; needle at 524k/1M;
with a draft model, the same tokens as without it.

## 2026-11-22

- **18:05–18:45 — the server path was not using any of the decode work; fixed (patches 0014, 0015).** End-to-end check
  of the defaults through `mlx_vlm.server` (Studio 1, its own numbers; client-side decode tok/s, 4k / 32k needle
  prompts, `--max-num-seqs 1`): the CLI/Python `generate` path on the same Mac does 60.1 / 53.4 without the draft and
  80.0 / 77.4 with it, but the server did **44.0 / 39.9** without and **31.1 / 29.0** with the draft (prefill also fell
  to 535–740 / 553 with the draft). Causes: (1) the server's `BatchGenerator` never calls the compiled decode step (it
  decodes eagerly, one sync per token), (2) with a draft it builds batch caches and runs the continuous-batching
  speculative state machine, which has no exact verifier, and a multi-token last prompt chunk leaves the sliding
  windows in concat layout so the verifier's layout check fails every round. Fixes: **0014** — one greedy row on plain
  caches decodes through `fast_decode.compiled_decode` (the same graph `generate_step` uses): 44.0 → **60.1–60.6** (4k),
  39.9 → **53.4–53.7** (32k), same text. **0015** — one greedy row with the draft head gets plain caches + the KV
  reserve, full sliding rings after the prompt, the round loop `_dflash_rounds` (not the batching state machine; the
  server then serves drafted requests one at a time, `MLX_VLM_INKLING2_SERVER_PLAIN_SPEC=0` restores the old path) and
  the 1,024–262,144 draft context window: 31.1 → **77.5–81.6** (4k), 29.0 → **74.1–78.8** (32k), prefill back to
  1040 / 947, same text in every run (all 16 answers hash-equal to the no-draft server's). Intermediate steps: plain
  caches only 36 / 33; + full rings 35.9 / 33.1; + round loop 77.5–81.6 / 74.1–78.8. Files:
  `results/gates/studio1/server/`.
- **18:10–18:40 — 32k drafted, per-round overhead (Studio 2, block 8, τ 0.5, 3 runs per config, tokens identical to
  the no-draft run in every run).** Round profile (ms): draft build 3.7, confidence wait 4.6, verify build 2.0, verify
  53.5 + 6.2, commit 0.5 (≈70.7 per round, 5.04 tokens). Compiling the draft head's Markov loop into one graph
  (`MLX_VLM_INKLING2_DRAFT_COMPILE=1`): draft build 3.6 → 2.5 ms, 71.2 → 71.6–72.6 tok/s. Forcing the full block every
  round: no change (the DSpark drafter already requests its full block). τ 0.4 67.6–67.9, τ 0.6 71.3–72.2 (with
  compile). **The draft's two vocabulary-sized reads** (the target's 201k-row head for the block and the folded
  Markov bias per step, ~1.6 GB per round) are most of the confidence wait: proposing from a candidate subset cut it
  5.4 → 1.7 ms, but taking ids < N as the subset dropped acceptance 88% → 71% (this tokenizer's common English words
  sit around ids 149k–156k), 62–64 tok/s. Next: a frequency-ordered candidate list (below).
- **17:25–17:55 — draft-confidence verify cut (patch 0012) and draft head on by default.** The draft head returns
  each proposal's softmax probability; the round verifies only up to the first position where the cumulative
  probability drops below τ (exact verifier, so tokens are unchanged). Studio 2, 32k, block 8: τ 0.3 66.9 / 0.5
  **72.3** / 0.6 72.6–70.9 / 0.7 71.6 (no cut 59.4; k5 67.7). The cut needs a CPU read of the draft before the verify
  graph is built: τ = 0.0001 (sync, never cuts) costs 5% (56.6 vs 59.4). Pre-building the verify for the likeliest
  cut while the GPU drafts (the exact verifier is pure until commit) did not help (71.5). Also tried: minimum 2
  rows (70.8), draft context window 8k (acceptance drops, 56.9). Studio 1 (same sweep, own numbers): τ 0.4 73.0 /
  0.45 75.3 / 0.5 77.3 at block 8; k6/k7 ceilings 67.4–73.0. Defaults (Studio 2): 4k 74.1 vs 55.6 no draft, 131k
  50.4 vs 36.4, 2048 75.8 vs 55.7; 600-token prompt 33.9 vs 42.0 (exact verifier can't serve <1,024 keys) → draft
  off below 1,024 and above 262,144 context tokens (`MLX_VLM_INKLING2_DRAFT_{MIN,MAX}_CTX`). Draft adds 1.1 GiB peak.
  All runs: tokens identical to the no-draft run. 0007 rebased onto 0008–0011 (one hunk); 0013 = prefix cache
  (0004+0006) on top of 0007 (one overlapping hunk resolved; not measured end to end).
- **17:15–17:30 — 0010 was not bit-identical on short chunks; fixed (shape gate).** The 8-prompt quality set
  (greedy 64 tokens + teacher-forced top-1, prompts of 36–14,692 tokens, step 2048) showed stack13 + 0010 at 4/8
  greedy-identical and 99.04% teacher-forced top-1, while the same tree without `PREFILL_FUSED` was 8/8 / 100%
  (stack12 + 0010: same 4/8, so 0010 itself). Per-part logit diff at 36 / 100 / 151 / 645 / 2048-row chunks: only
  the attention-residual kernels differed, and not at 2048. Cause: the fork's scores are one `(rows × states, D) @
  (D, q)` product and MLX picks its reduction order by shape — gemv's bn=8 split when D ≥ 16·M (one query) and
  steel split-K when D ≥ M (2–4 queries). The 4k/32k sha runs only had ≥2030-row chunks. Fix: the exact kernels
  serve only the shapes whose order they reproduce; the rest keep the fork's path. After: logits equal at all
  chunk sizes tested (36–2048), 8/8 greedy-identical and 100% teacher-forced on the prompt set. Reading is
  unchanged (4k/32k chunks are all ≥4078 rows). Patch 0011 (`DECODE_MIN_KEYS=1024`) gave output identical to its
  control on all 8 prompts, so it adds no divergence. Files: `results/gates/studio2/top1/top1-{s13ctl,s13nof,
  s12f,mk1024,s13gate}.*`, `iso2.py`, `iso{2,3}.log`.
- **17:22 — non-exact compiled verify for drafting (`EXACT_VERIFY=0`, `COMPILED_VERIFY=1`), 32k: slower.** k5 26.0 /
  40.5, k8 46.0 / 47.1 tok/s (exact verify: 67.7 / 59.4), and k5 acceptance moved (69.5% vs 76%: not exact). Every
  round "missed" the compiled path and fell to the generic verify. Dropped.
- **17:00–17:15 — short-prompt writing: the first ~18 tokens after a 4k prompt were decoded eagerly (patch 0011).**
  Decode profile on Studio 2 (stack13, 4k needle prompt): the compiled decode step costs 1.5 ms of CPU, `async_eval`
  16.5 ms wall of which 6.7 ms CPU, so the step is GPU-bound at ~18 ms — but 17 of 123 steps returned "not served"
  and ran the eager graph (~28 ms). Cause: the fork decodes global attention with the stock path below 4096 keys
  (`_DECODE_KERNEL_MIN_KEYS`), and both the compiled step and the exact draft verifier only cover the split-K kernel
  path; a 4078-token prompt sits under the line for its first 17 tokens (the same thing caused the 4–5 "misses" per
  answer in the 4k draft runs). Opt-in `MLX_VLM_INKLING2_DECODE_MIN_KEYS=1024`: no-draft writing 4k 52.6 → **55.1–55.9**,
  drafted 4k k5 62.6 → **74.8 / 76.3** (k6 66.5, k8 62.0), misses 0; tokens identical to the unpatched fork's no-draft
  output in every run (not bit-identical in log-probabilities: split-K attention adds in a different order). Prompts
  ≥4096 tokens are unaffected. Round profile before the fix (32k, k8): draft 7.7 ms, verify graph build 1.9 ms,
  `async_eval` 68 ms, final wait 9 ms per round; verify is GPU-bound (8 rows touch up to 8× the experts).
- **16:45–16:57 — drafted writing on Studio 2 (stack13 = stack7 with 0007 exact DSpark + new 0010; prefill step 4096,
  `NOHASH=1` speed runs; tokens identical to the no-draft run in every case).** Warm tok/s, 4k: k4 59.8 / **k5 62.6** /
  k6 57.7 / k8 53.4 (acceptance 75 / 77 / 63 / 54%); 32k: k4 64.6 / **k5 67.7** / k6 61.4 / k8 59.4 (77 / 76 / 64 /
  58%). MLX-Serve on the same Mac: 66.1 (4k) / 74.8 (32k), block 8. Gaps: −5% at 4k, −9.5% at 32k. Rounds cost ~3
  decode steps at k5 (60 ms at 32k); acceptance falls off after the 5th draft token. At 4k, 4–5 rounds per answer miss
  the exact verifier (sliding-window cache not yet full) and decode token by token. Note: the stack8/stack12 trees have
  no 0007, so the exact-verify env is ignored there (drafting at 4k: 45.9 / 54.1, slower than no draft) — use a tree
  with 0007. No-draft writing, same runs: 4k 52.6, 32k 48.9 (MLX-Serve 56.4 / 50.2); the per-token logprob hash in
  exact_check costs nothing measurable (52.62 with `NOHASH=1`).
- **Reading, repeated runs on Studio 2 (patch 0010, step 4096):** 4k 1013.1 / 1015.1–1037.2 / 1023.3 tok/s warm
  across five runs, 32k 883.5–888.2 (MLX-Serve 1010–1012 / 877). Gate status: step 2048 = bit-identical to the fork
  (sha); step 4096 = the fork's own chunk change, top-1 99.6% vs step 2048 on the 8-prompt set, needle answers
  identical (`results/gates/studio2/top1/top1-fork-step4096.json`).
- **Quality-gate harness (`scripts/quality/top1.py`):** greedy 64-token answers on 8 fixed prompts (6 corpus slices
  3k–60k chars with questions, 2 chat prompts) from the fork at step 2048 as reference, then teacher-forced top-1
  agreement of a candidate on those continuations. Results in `results/gates/studio2/top1/`. The ablation logs
  (`abl.log`, `top1-arx.*`, `top1-only-*`) were measured with the first, inexact kernels (PF_AR 98.65%, PF_MG 99.62%,
  PF_PJ 98.46% alone; `top1-arx` = the exact kernel before the precise-exp fix, 98.46%); the fork vs itself at
  step 1024 is 98.85%, at step 4096 99.62%. The final kernels are bit-identical, so they need no top-1 margin.
- **16:20–16:45 — bit-identical attention-residual and message-gate kernels (patch 0010 update; now on by default
  with `MLX_VLM_INKLING2_PREFILL_FUSED=1`).** Studio 2, applied to a fresh copy of the stack8 tree, eval-every 0:
  step 2048: 4k 987.0, 32k 872.4 tok/s reading, **tokens AND logprob sha equal to the fork** at both, needle found
  (was 904 / 804 with the first 0010). Step 4096: 4k **1013.1** / 32k **888.2** (MLX-Serve there: 1010–1012 / 877);
  step 4096 changes the fork's own chunking (fork vs fork at step 4096: top-1 99.6%, greedy needle answers identical),
  the fused paths add no difference on top. Writing unchanged (52.9 / 48.9 no draft). How the residual kernel matches
  MLX bit for bit: rms_norm with rms_single_row's tree (D/4 threads × 4 columns, simd_sum, threadgroup simd_sum,
  precise rsqrt); one-query scores with gemv's lane order (32 lanes × 4 columns per 128-column block, shuffle-down);
  2–4-query scores as a sequential FMA chain over D (that is what the steel GEMM computes per element, checked);
  softmax over the states axis is MLX's composite op (not the softmax kernel): precise exp(x − max), sequential sum,
  divide; weighted sum as gemv_t (states 0–3, then 4–7, added) or an FMA chain over states. Two misses on the way:
  the fused softmax kernel's fast::exp + reciprocal (wrong op), and metal::exp vs precise::exp (only visible with the
  real queries, |q| up to 8; random test queries at 0.05 hid it). Message gate: rms_norm + concat + cast in one kernel
  with the same tree, the fork's three matmuls (the merged forget/gain matmul was the inexact part), compiled tail.
  Microbench: residual sites 2–3× faster (Q=1: 0.35 vs 0.84 ms at 2 states, 2048 rows; Q=3/4: 0.8–1.7 vs 1.8–3.0 ms).
  Checked in the model per site (PF_CHECK) before the end-to-end sha gate.
- **16:25 — patch 0010 gated on Studio 2 (applied to the stack8 tree, `MLX_VLM_INKLING2_PREFILL_FUSED=1`, step 2048,
  eval-every 0): tokens AND logprob sha equal to the fork at 4k and 32k, needle found.** Reading 4k 879.6 / 901.0 (cold /
  warm), 32k 803.6 / 803.5 (Studio 2 baseline same settings: 870.2 warm at 4k, 773.0 at 32k); writing unchanged
  (52.3 / 49.0). `results/gates/studio2/p10-prefill-fused-4k32k.json`.
- **16:10 — 1M with KT=3 + host vocab (Studio 1): reading 163.3 → 175.6 tok/s (+7.5%), writing 11.38, peak 50.08 GiB,
  needle found; tokens and logprob sha equal to the 163.3 run** (`results/gates/h7-1m-kt3-hostvocab.json`). The probe
  projected 170–178. Patches 0008/0009 committed; `install.sh` applies both (off unless their env is set).
- **15:30–16:10 — short context vs MLX-Serve on Studio 2 (relaxed gate, same official weights).** MLX-Serve (0.32.5
  libmlx, prefill chunk 2048) self-reports 1010/1012 tok/s reading at 4k and 877 at 32k there; we measured 870 / 773
  (step 2048, warm). Profile at depth 0 (barriered, 4k prompt): MoE 52% (gather_qmm NVFP4 at ~10.5 TFLOP/s, i.e. compute
  bound; fusing gate+up into one gather_qmm saved nothing), attention projections 18% (bf16 GEMMs, also compute bound),
  attention core 7%, message gates 6%, attention residuals 6%, short convs / l2 clips / norms ~8%. Env-only sweeps (eval
  cadence, step 2048–8192) gave ≤2%. New prefill-only paths (`MLX_VLM_INKLING2_PREFILL_FUSED=1`, patch 0010), measured:
  MoE route-combine fused into the un-sort (one kernel; bit-identical with safe math, −1.7 ms per visit), the fork's
  virtual-concat short-conv kernel for any length (bit-identical), compiled l2 clips (bit-identical): 4k 870 → 904
  (+3.9%), logprob sha equal to the fork. Adding a fused message gate: 933 (top-1 99.6% alone). Adding a fused
  attention-residual kernel (no stack): 986 at step 2048, 1008 at step 4096 (4k), 865 / 883 at 32k; greedy tokens on
  the 4k/32k needle prompts identical to the fork, but teacher-forced top-1 on an 8-prompt set is 98.5% (fork vs itself
  at step 1024: 98.85%; step 4096: 99.6%) — below the 99% gate, so the residual kernel stays off by default while a
  bit-exact version (replicating MLX's rms_norm / gemv / softmax / gemv_t order) is being tested. Failed: merged q/k/v/r/gate
  projections (bit-exact at 2048 rows only; top-1 98.5% across row counts) — dropped.

- **14:11–14:30 — per-visit profile at 1M depth: the "depth-growing" non-attention cost is MLX allocator thrash, not
  depth** (`results/profile-20261122/`). At 1M depth the 40 non-global visits of a 2048-token chunk took 54.9 ms each
  vs 42.7 ms at depth 0. The same slowdown appears at depth 0 when 32 GiB of unrelated MLX arrays are held (51.6 ms)
  or with `mx.set_cache_limit(0)` (51.5 ms); 28–30 GiB of ballast costs nothing, 31 GiB +14%, 32 GiB +22%. Cause
  (MLX 0.32.4 `metal/allocator.cpp`): once active + cache + request ≥ `gc_limit` = 0.95 × recommended working set
  (49.25 GiB here), every buffer-cache miss releases the cache, so ops allocate fresh Metal buffers. At 1M the weights
  (17.6 GiB) + reserved KV (32 GiB) sit at 49.6 GiB for the whole prefill. `gc_limit` can't be raised from user space
  (it's capped by the recommended working set, i.e. the wired-limit sysctl). The 8 global attention visits run at the
  microbench rate in the model (2.49 s vs 2.47 s), so there's no hidden attention overhead.
- **Fix (exact): keep the two vocabulary tables in host memory during the chunked prefill** (`MLX_VLM_EXACT_HOST_VOCAB=1`,
  `mkhost.py` on `generate/ar.py`). `embed_tokens` and `lm_head` (2 × 0.86 GiB bf16) move into numpy before the chunk
  loop; chunk embeddings are a host row gather (a copy) + the same `embed_norm`; chunk graphs never evaluate logits; both
  tables go back to MLX before the last token's step. At 1M depth: non-global visits 54.9 → 42.6 ms, 2048-token chunk
  22.15 → 21.57 s (−2.6%); with KT=3 20.83 s (−6.0%). Tokens + logprob sha equal to the fork at 4k and 32k (Studio 2).
  1M end-to-end run with both started 14:30 CET.
- **14:07 — 1M on the current build at step 2048: reading 156.0 → 163.3 tok/s (+4.7%), writing 10.04 → 11.36 tok/s
  (+13%), peak 50.76 of 51.84 GiB, needle found** (`results/gates/h6-1m-step2048.json`). Gains come from the bigger
  prefill chunk (fits since the conv-state fix; the synthetic-depth probe predicted +7.6% and 50.58 GiB) and the decode
  track's final 0002. 70% of the 233 tok/s arithmetic ceiling; decode 83% of its bandwidth ceiling.
- **13:00–13:55 — second test rig (Studio 2, same M4 Max / 64 GB) for exactness gates.** KT=3 (three BK16 tiles staged
  per barrier pair in the global relative-attention prefill kernel; per-row online-softmax order unchanged) is
  bit-identical to the unpatched fork at 32k / 131k / 262k (tokens + logprob sha, needle found). End to end it is
  within noise at 32k (786.7 vs 789.7) and +0.6–1.1% at 131k (572.5 / 575.9 vs 569.4); the kernel gain (+3.9% at 1M
  keys, T=2048) only shows at 0.5–1M depth. Still to measure at 1M.
- **13:20–13:55 — DSpark block size (exact verify, all runs bit-identical to the fork):** warm writing tok/s on Studio 2,
  4k: k3 56.5 / k4 56.7 / k5 58.1; 32k: k3 58.8 / k4 62.5 / k5 63.9 / k6 57.7 / k8 56.1; 131k (cold): k4 48.5 / k5 47.9
  / k6 44.7. The adaptive selector (`MLX_VLM_INKLING2_ADAPT_K`) settles on its largest candidate in these short
  (~125-token) answers, so it equals fixed k (3,4,5 → 63.7 at 32k) and adds nothing. Block 4 stays; k5 is +1–2% at
  ≤32k only, within run-to-run noise. Acceptance (77% at k4) is the limit, not block size.
- **12:08 — bit-identical speculative decoding verified and pushed (`0007`).** Draft block 4 at 32k / 131k / 262k:
  65.99 / 52.67 / 35.72 tok/s writing (no draft 53.73 / 40.16 / 29.76), tokens AND logprob sha equal to the unpatched
  fork's no-draft run at all three. Earlier the DSpark port without the exact verifier was slower than no draft on
  stack6 at 131k–262k (26.2 / 23.9 vs 39.9 / 29.6 tok/s, block 8), because its verify ran on the prefill path.
- **11:50 — prefix-cache eviction fix ported (`0006`, from betizzel 295f5b21).** Our base had the bug: on fork + 0004,
  the upstream tests for it fail (3 of 726 in `test_cache.py`: a continuation checkpoint is refused because the store
  also reserves room for the next prefill); with the port all 726 pass. Two of six hunks were re-applied by hand
  (our base has no `owned` argument). Bookkeeping only. End-to-end 3-turn check at 131k still to do.
- **11:31 — betizzel 5a3c6132, router half (cached FP32 router weight) on stack6: no gain, dropped.** 4k 952.5 vs 952.4
  reading, 57.04 vs 56.85 writing; 32k 867.1 vs 867.3 / 53.71 vs 53.73; bit-identical. The compiled-message-gate half
  doesn't apply to our loop code (our decode path has its own fused gates) and was not tried.
- **11:37 — exact DSpark (decode track, in progress):** a verify path whose rows are bitwise the decode rows, so the
  logprob sha matches the no-draft run (not just the tokens). Synthetic depth: 524k 19.46 → 28.76 tok/s with block 4.
- **11:24 — stack6 gate passed; pushed as `stacked.patch`.** Decode track's final 0002 (fused MoE decode, o_proj tail,
  fused message gates, multi-query attention residual, custom bf16 GEMV, MLX-replica sliding attention) stacked on
  0001 + 0003. Writing 56.85 / 53.73 / 40.16 / 29.76 tok/s at 4k / 32k / 131k / 262k (was 50.64 / 47.23 / 35.80 / 27.01),
  reading unchanged, peak unchanged, bit-identical at all four lengths. 0005 (DSpark) no longer applies cleanly (1 hunk
  in `inkling2/language.py`); kept on `stacked-v4.patch` until ported.
- **11:09 — quality benchmark** (GSM8K / MMLU-Pro / RULER NIAH 32k, see BENCHMARKS.md): no quality change;
  token-identical to the fork on 100/100 GSM8K items.
- **Lead found in the quality run:** chat-length prompts (< 4,096 tokens of context) never take the compiled decode
  step: `fast_decode._layout` requires every global cache to hold >= 4,095 keys and every sliding ring to be full,
  so short chats decode eagerly at ~43 tok/s instead of ~57. Fix needs a bitwise compiled path for short contexts.

## Open now
- Reading at 1M (156.0 tok/s, 67% of the 233 tok/s ceiling): ~85% of 1M prefill FLOPs are the 8 global attention
  visits, so the global relative-attention prefill kernel at 0.5–1M keys (now ~68–76% of 14.2 TFLOP/s depending on
  chunk size) and the chunk size affordable at 1M (step 512 used; 1024 may fit after the conv-state fix) are the levers.
- Writing at 1M: KV-read bound (32 GiB per token); speculative decoding amortizes it across a verified block.

## 2026-11-22 19:05–19:20 CET: wrap-up (Tomas 19:04: only necessary checks)

- **0016 committed** (draft head proposes from the first 16,384 ids of `data/draft_vocab_order.json` + the prompt's ids,
  compiled Markov loop; defaults `DRAFT_CONF=0.65 DRAFT_VOCAB=16384 DRAFT_COMPILE=1`). List provenance and rebuild:
  `scripts/draft_vocab.py` (model's own GSM8K/MMLU-Pro answers without RULER rows, then Wikipedia frequencies);
  rebuilt byte-identical. The repo patches reproduce the deployed Studio 2 `dspark.py` exactly.
- **Quality, defaults, stack16 (= install.sh output):** 8-prompt set greedy 8/8 identical to the reference, through the
  draft head (`top1.py DRAFT=8`).
- **Server head-to-head, Studio 2:** reading 979.6 vs 1012.4 (4k), 861.5 vs 877.3 (32k); writing 68.6 vs 66.1 (4k),
  63.2 vs 74.8 (32k). Prefill step 4096 in the server is slower (946 / 839 reading). Only 4k writing wins in the
  server. The 32k server writing (63) is far below the CLI drafted runs (72–76): not profiled (out of scope).
- Stopped at wrap-up: vg/dec3 sweeps on Studio 2 (no-draft tuning, vocab-size sweep) and the 524k draft + cv runs on
  Studio 1, unfinished; no results recorded from them.
- README rewritten as charts + short text (`scripts/charts.py` → `docs/img/*.svg`); old tables moved to BENCHMARKS.
  "Fastest" dropped from the headline: the server head-to-head does not support it.

## 2026-11-22 19:27– CET: server rows (Tomas 19:27: beat MLX-Serve on all four Studio 2 server rows)

- **Same-prompt CLI vs server (Studio 1, stack16, defaults):** CLI drafted on MLX-Serve's own prompts: 4k 84.8, 32k
  84.8 tok/s (prefill 1020 / 936); server: 4k ~72, 32k ~66 (scaled from Studio 2's 68.9 / 63.1). Same per-round
  time; the server simply ran more rounds.
- **Root cause 1 (fixed, 0017):** the server's single-request draft path never handed the prompt's token ids to the
  draft head, so 0016's candidate list lacked the prompt's vocabulary (generate_step sets `_lookup_prompt_ids`).
  0017 sets it in the plain-spec branch. Studio 1 server: 4k 83.4–83.8, 32k 78.1–78.3 (a17). Studio 2 server (g17):
  4k 80.2 / 80.6 (was 68.2 / 68.9; MLX-Serve 66.1), 32k 74.6 then 55.6 (the last 32k run had prefill 745 tok/s:
  another job on the GPU; needs a clean re-run). Reading unchanged (977–985 / 865).
- Server kill bug in our bench script on Studio 1: `kill $SP` left the server alive, the next run's health check hit
  the old server. Fixed (wait for exit, kill -9 fallback, wait for the port to be free); rb2/rb4 re-run.
- **Server reading sweep (Studio 1, stack17, warm run):** step 2048: 1064.0 / 938.7; step 4096: 1043.4 / 921.1;
  without `MLX_MAX_MB_PER_BUFFER=2`: 1034.8 / 920.9 (2048), 995.8 / 901.0 (4096); 4096 + `EVAL_EVERY=0`: 1033.8 /
  935.2. Keep the defaults (2048, MB 2, eval every chunk).
- **MLX-Serve re-run on Studio 2, same session (19:52, same binary and script, draft on, warm-up + 3 runs):** reading
  1042.1 / 1038.8 / 1034.2 (4k), 898.8 / 898.3 / 897.6 (32k); writing 75.3 / 76.2 / 77.2 (4k, 122 tokens),
  70.6 / 72.2 / 71.5 (32k, 125 tokens). Its published 15:08 numbers (1012.4 / 877.3 / 66.1 / 74.8) were single runs;
  today it reads ~2.5% faster and writes 4k faster, 32k slower. `results/h2h-20261122/mlxserve-rerun-1952-draft-on.log`.
- **Ours right after it (g17b, stack17 = 0017, env.sh defaults):** reading 983.3 / 984.2, 865.1 / 866.2; writing
  79.4 / 80.1 (4k), 74.9 / 74.8 (32k). Same session: writing wins both rows (+5% / +5%), reading is −5% / −4%.
- tau sweep (Studio 1 server, stack17): 0.35: 76.2 / 73.5 (drafted 143 / 133 for 98 / 101 accepted); 0.5: 82.4 /
  78.8; 0.65 (default): 83.4–86.1 / 78.1–80.6. Rounds stay 25–26: the cut only trims wasted drafts. 0.8: lost to a
  bench-script bug (an overwritten running script left a stale server on the port; its requests hit the 0.5 server).
  Fixed with versioned scripts and a random port per run.
- Draft-head prefill cost (Studio 1 server, step 4096, no MB): draft off 1044.2 / 941.3 vs on 1033.8 / 935.2 (−1.0% / −0.6%).

### 2026-11-22 20:00– CET: fast prototyping (Tomas 20:11: single-shot server runs, gate at the end)

One line per attempt. Server log, warm run, 4k / 32k; S1 = Studio 1, S2 = Studio 2.
- Opus #5 env parity (S1, stack18): `EVAL_EVERY=0` at step 2048: reading 1053.6 / 931.3 vs 1064.2 / 939.4 (worse);
  also without `MB=2`: 1036 / 922 (worse); step 4096 + `EVAL_EVERY=0`: 1042 / 925.5 (worse). env.sh stays. FAIL.
- Opus #2a `MLX_VLM_BATCH_CACHE_EVAL_INTERVAL=0` (no cache clear inside the answer), S1: 32k writing 80.5 -> 81.2,
  4k 86.5 -> 86.9. Small WIN, kept.
- No 1 GiB cache limit + no per-chunk `clear_cache` in server prefill (S1, with `EVAL_EVERY=0`): 1062.8 / 940.3, i.e.
  recovers what `EVAL_EVERY=0` lost, no gain over defaults. Re-testing with defaults.
- Opus #2b + #1 bucket: build the draft candidate list during prefill (async), JSON parsed once, size bucket 4096
  (S1 f1): 4k 87.4, 32k 81.8 (from 86.9 / 81.2); S2 g21a: 81.8 / 75.5 (from 80.1 / 74.8). Small WIN, kept.
- Message-gate tail casts inside the compiled kernel (`PF_MGC`): microbench bitwise equal at 36–4,096 rows,
  0.40 -> 0.30 ms per gate at 2,048 rows; S2 server with lean prefill: 983.4 / 868.3 vs 978.2 / 865.5. Small WIN.
- Opus #4 lean draft prefill (only the global draft layer per chunk, async; sliding layers once from the last 1,024
  rows): S1 1064.1 / 941.5 vs 1061.6 / 939.3, tokens and acceptance unchanged. Small WIN.
- Fused gate+up expert GEMM (one `gather_qmm`, N = 2,304; the checkpoint's w13 shares one global scale): S1 1062.3 /
  940.2 vs 1063.2 / 941.7, same answers. No gain (the expert GEMM is compute-bound; N doesn't help). FAIL, off.
- Server step 3072 / 2560 for every prompt (S1, cache limit kept): 4k 1041 / 1054.5 (worse than 2048), 32k 941.5 / 947.9.
  A long-prompt-only override (`MLX_VLM_SERVER_STEP_LONG=2560` above 8k tokens) then read 942.0 vs 941.9 at 32k on S1
  and 864.4 vs 864.4 on S2: no gain. Opt-in, off.
- Default cache limit kept, no per-chunk `clear_cache` in server prefill, `EVAL_EVERY=1` (S1): 1066.1 / 943.2 vs
  1064 / 941.5. Within noise. Off.
- Last-chunk fold (`MLX_VLM_SERVER_FOLD_LAST=1`): the server ran the last prompt token as its own forward after the
  last chunk; now the last chunk and the last token run in one forward (as `generate` does). S2: 4k 983.4 -> 987.2;
  S1: 1064 -> 1068. Small WIN (+0.4% at 4k, 32k unchanged), default on in 0018.
- Draft `CONF_MIN=0` (allow a verify of the bonus token alone when the drafter is unsure, S1): 4k writing 87.7 vs 87.4,
  32k 80.9 vs 81.8, more rounds. With tau 0.8: 87.1 / 76.8. tau 0.8 alone: 86.3 / 78.6. FAIL; tau 0.65 + at least one
  draft stays.
- Approximate fusions back on (S2): merged forget/gain matmul (`PF_MGX=0`) 988.5 / 870.7, plus the older attention-
  residual kernel (`PF_ARX=0`) 993.9 / 872.4 vs 987.2 / 867.7. +0.7% / +0.5%, but `PF_ARX=0` already failed the 99%
  top-1 gate (98.5%) earlier today. Not used.
- Fused message-gate tail (`PF_MGF=1`): forget/gain f32 K=64 products with the same 8x8 simdgroup MMA k-order as MLX's
  GEMM plus tanh/combine in one kernel. Bitwise equal to the reference at 2, 7, 30, 2,030, 2,048 and 4,078 rows;
  0.46 -> 0.37 ms per gate at 2,048 rows in isolation. Server (S1): 1068.1 / 941.9 vs 1067.7 / 942.1; S2: 984.7 vs
  985.0. No end-to-end gain (the gate is overlapped). Opt-in, off.
- What MLX-Serve does differently (read from github.com/ddalcu/mlx-serve, `src/plugins/inkling2`): its prefill runs each
  visit as "a few kernels around the matmuls" (`pre.metal`, `mid.metal`, its own rel-attention), and it loads the
  NVFP4 experts without the per-expert global scale inside the GEMM (`scale / (6 x 448)` applied to the output). Its
  binary ships libmlx 0.32.5 (unreleased main), which has MLX PR #4709 (fp4 -> half dequant fast path, only when no
  global scale is passed).
- Same expert layout here (`PF_POSTSCALE=1`: plain NVFP4 gathers, scale applied in the SwiGLU and folded into the routing
  weights): on mlx 0.32.4 S1 1062.6 / 938.2 vs 1068.1 / 942.2 (the extra scale pass costs); on mlx 0.32.4 + PR #4709
  built from source (S2): 973.2 / 863.3 vs 976.5 / 865.7 with the old layout and 981.0 / 865.9 on stock 0.32.4. The
  half-loader fast path doesn't reach the sorted gather at our shapes. FAIL; answers still found the needle.
- **21:00 same-session head-to-head, Studio 2, final stack (0018 defaults), server log, warm runs.** MLX-Serve
  (`mlxserve-rerun-2055-draft-on.log`, 3 warm): reading 1031.0 / 1036.8 / 1029.7 at 4k, 892.7 / 894.4 / 893.6 at 32k;
  writing 73.9 / 74.5 / 75.0 at 4k, 72.7 / 72.7 / 71.3 at 32k. Ours (`srv-h26.log`, 2 warm): reading 983.0 / 982.2 at 4k,
  864.2 / 863.4 at 32k; writing 78.0 / 79.3 at 4k, 77.1 / 76.8 at 32k. Writing wins both rows (+5% / +7%); reading is
  4.9% / 3.4% behind. Needle found in all 6 of our answers.
- 21:00 gate on the final stack (Studio 1, `top1-g26.json`): greedy answers through the draft head 8/8 identical to the
  reference; teacher-forced top-1 520/520 = 100% vs the fork's teacher-forced argmax (96.5% vs the fork's greedy text,
  same as before).

### 2026-11-22 21:03– CET: reading push (fast prototyping, single warm run unless noted)
- CPU vs GPU in prefill (S1, CLI path, eval per visit): time outside `mx.eval` is 6% of wall at 4k (warm) and 1.1% at
  32k, so prefill is GPU-bound. Per-op profile (barriered, 4k): MoE 55.6% (23.6 ms per 2,048-row visit), attention
  projections 20.8%, attention core 9.8%, message gates 3.8%, l2 clips 2.5%, short convs 2.5%, residual glue ~4%.
- Async per-visit eval (`MLX_VLM_EXACT_PREFILL_EVAL_ASYNC=1`): S2 983.7 / 869.0 and 987.3 / 871.5 vs 981.7 / 863.9
  (+0.3% / +0.7%); every 2 visits: 980.5 / 863.1 (S2), 1070 / 943 (S1); every 4: 1062 / 938 (S1). Small WIN for every 1.
- Step 4096 with the 1 GiB server cache limit (S1, async): 1043.9 / 924.7. Still worse, as before.
- **The server's 1 GiB MLX buffer-cache limit was throttling prefill.** No limit (`CACHE_LIMIT=0`, i.e. the working-set
  size): step 2048 S2 994.0 / 871.1 (S1 1079.7 / 952.1); step 4096 S2 1004.9 / 892.3 (S1 1090.8 / 968.2). At 4096
  the 32k answer's draft acceptance changed (27 rounds / 95 accepted vs 25 / 102: chunking changes the hidden states'
  last bits, so the drafter sees slightly different inputs; same answer text), and 32k writing fell to 78.8 (S1) /
  71.4 (S2).
- 32k step sweep without the limit (S1, async): 3072 960.3 / 83.0 writing, 6144 967.9 / 81.6, 8192 972.6 / 83.3.
- 4k single chunk variants (S2, step 4096, no limit): eval every 2 1000.2, no per-visit eval 1003.7, `PF_MGF=1` + 4 MB
  buffers 1003.3. Plateau ~1,000-1,005 vs MLX-Serve 1,030-1,037.
- **Step 8192, no cache limit, async eval (S2 `srv-gi1`, then `srv-gk` with 2 warm runs):** reading 1004.7 / 1005.1 at
  4k (one forward) and 899.3 / 900.4 / 899.2 at 32k (four chunks); writing 79.6 / 79.6 and 76.1 / 76.0. Against MLX-Serve's
  20:55 run on the same Mac (1029.7–1036.8 / 892.7–894.4 reading, 73.9–75.0 / 71.3–72.7 writing): **32k reading +0.6%,
  both writing rows ahead; 4k reading still 2.6% behind.** S1: 32k 972.6 (8192) / 976.4 (16384, writing 80.4).
  Launcher: `scripts/serve.sh`. Needle found in all 18 answers.
- 4k single-forward knobs (S1, step 8192, no limit): 8 MB command buffers 1088.1, 200 ops per buffer 1088.5, merged
  forget/gain matmul (`PF_MGX=0`) 1090.5 vs 1085.7. Within noise. FAIL.

### 2026-11-22 21:50– CET: 4k reading push
- **21:51 same-session confirm, Studio 2** (MLX-Serve `mlxserve-rerun-2150-draft-on.log`, then ours `srv-h27`, the
  `scripts/serve.sh` settings): MLX-Serve reading 1030.9 / 1036.6 / 1026.4 at 4k and 896.6 / 896.4 / 897.0 at 32k, writing
  75.9 / 76.1 / 77.0 and 70.7 / 70.6 / 71.7. Ours reading 1003.1 / 1008.8 and 901.3 / 899.3, writing 79.3 / 80.6 and
  75.0 / 75.8. The 32k reading lead holds (+0.4%, thin); 4k is −2.4%. Needle 18/18.
- Prefill profile at 4k, one 4,077-row chunk (S1, barriered): expert GEMMs 52.7% (14.7 ms per call), attention
  projections 21.2%, relative-attention kernel 9.6% (7.4 ms per visit), message gates 3.9%, short convs 2.4%, l2 clips
  2.3%, residual glue ~4%, MoE glue (route, sort, SwiGLU, un-sort) 3.4%.
- Looped core (layers 6–18 × 3): the passes are sequential (each pass reads the one before through the message
  gates) and a visit's expert GEMM already reads each expert's weights once for all 4k rows (compute bound, ~12.8
  TFLOP/s), so keeping weights hot across passes has nothing to reuse. Not prototyped.
- **Relative-attention kernel tiles at 4k (microbench, 4,077 rows, bitwise equal to the default in every config):**
  sliding visits (1,024-key window) 6.9–7.0 ms with the defaults (KT=3, 32-query blocks); 64-query blocks 4.7–4.8
  ms; KT=1 4.7 ms; 16-query blocks 13.5; prefetch 12.7; head grouping 5.5. Global visits: defaults best (9.7 ms).
  -> `MLX_VLM_RELATIVE_ATTN_BQ_SWA=64` (env only, same arithmetic).
- Expert gather without the sorted copy of x (`gather_qmm` with `lhs_indices`, microbench): bitwise equal, 66.6 vs
  70.5 ms in a synthetic bench. Not wired yet.
- **`MLX_VLM_RELATIVE_ATTN_BQ_SWA=64` in the server (single warm run):** S1 4k 1089.0 -> 1115.2 (+2.4%), 32k 972.5 ->
  994.8 (+2.3%); S2 1026.2 / 1032.2 at 4k (MLX-Serve 1026.4–1036.6 at 21:51) and 917.0 at 32k (+2.2% over MLX-Serve).
  Writing unchanged. WIN; default in `env.sh` and `serve.sh`.
- Expert GEMM reading x rows in place via `lhs_indices` (`PF_LHS=1`) in the server: 4k 354.9, 32k 341.1. MLX takes a
  much slower gather kernel when `lhs_indices` is not the identity. FAIL.
- Global-visit attention tiles at 4k (`attnbench3`, all bitwise): base 9.70 ms, BQ128 10.99, BQ128+KT1 11.96, KT2 9.83,
  FAST_MASK=0 9.76, K_ROWMAJOR 9.86, PIPELINE_QK 9.62, PAD0 9.95; PAD16 fails to load. No change worth taking. FAIL.
- `PF_MGF=1` on top of BQ_SWA64 (S1, 4k): 1114.4 -> 1116.3. Noise. FAIL.
- **S2 same session h28 (22:10–22:16 CET), MLX-Serve then ours (serve.sh settings + BQ_SWA64), warm runs:** reading 4k
  MLX-Serve 1040.5 / 1033.9 / 1033.6 vs ours 1040.1 / 1019.8; 32k 895.5 / 896.7 / 895.6 vs 919.5 / 918.2 / 919.2
  (+2.6%). Writing 4k 74.8–76.7 vs 79.4 / 80.1; 32k 71.7–72.4 vs 76.5 / 76.4. Needle 18/18. 4k reading is within
  about 0.6%, still slightly behind on average.
