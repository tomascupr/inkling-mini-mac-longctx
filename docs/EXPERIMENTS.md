# Experiment log (newest first)

Plain log of what we tried after the 1M headline, including what failed. Times are CET (Europe/Prague).
Gates: bit-identical tokens + logprob sha256 vs the unpatched fork at 4k/32k/131k/262k; needle at 524k/1M;
with a draft model, the same tokens as without it.

## 2026-11-22

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
- **betizzel 5a3c6132** (cached FP32 router weight + compiled loop message gates, +3% prefill on M5 Pro): applies
  with fuzz; an earlier compiled message gate disabled our compiled decode path, so it is being re-tested stacked.
- **betizzel 295f5b21** (prefix cache: a continuation turn evicted the checkpoint it extended): 2 of 6 hunks don't
  apply on fork + 0004; whether our build has the bug is being checked before porting.

## Open now
- Reading at 1M (156.0 tok/s, 67% of the 233 tok/s ceiling): ~85% of 1M prefill FLOPs are the 8 global attention
  visits, so the global relative-attention prefill kernel at 0.5–1M keys (now ~68–76% of 14.2 TFLOP/s depending on
  chunk size) and the chunk size affordable at 1M (step 512 used; 1024 may fit after the conv-state fix) are the levers.
- Writing at 1M: KV-read bound (32 GiB per token); speculative decoding amortizes it across a verified block.
