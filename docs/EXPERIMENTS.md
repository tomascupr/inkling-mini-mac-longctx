# Experiment log (newest first)

Plain log of what we tried after the 1M headline, including what failed. Times are CET (Europe/Prague).
Gates: bit-identical tokens + logprob sha256 vs the unpatched fork at 4k/32k/131k/262k; needle at 524k/1M;
with a draft model, the same tokens as without it.

## 2026-11-22

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
