# Second-opinion ideas: beat MLX-Serve on the server (2026-11-22)

Review only: nothing here was run on Apple hardware. **Every gain below is an estimate** worked out from the logs and
code in this repo; treat each one as a hypothesis to measure.

Code references point at the tree `scripts/install.sh` builds (fork `55ca2e9e` + `stacked.patch` + 0007–0016; every
patch applied cleanly), with line numbers from that tree. MLX-Serve references point at
[ddalcu/MLX-Serve](https://github.com/ddalcu/MLX-Serve) tag `v26.11.4-pre-release.1` (commit `2fe9be8`). Its MLX is the
submodule pin `c69303b` ("0.32.5", 77 commits after v0.32.4, no Metal patches).

## Targets (server log, Studio 2, `results/h2h-20261122/`)

| tok/s | MLX-Serve | ours | needed |
|---|---:|---:|---:|
| Reading 4k | 1012.4 | 979.6 | +3.4% |
| Reading 32k | 877.3 | 861.5 | +1.8% |
| Writing 4k (draft) | 66.1 | 68.5 | ahead |
| Writing 32k (draft) | 74.8 | 63.2 | +18% |

## What the logs and code show

**1. The server's draft head cannot propose 13 of the 119 tokens of the 32k answer.** With 0016, the drafter
proposes only from the first 16,384 ids of `data/draft_vocab_order.json`, plus control ids, plus the prompt's ids
(`_lookup_prompt_ids`). Only `generate_step` sets those prompt ids (`generate/ar.py:477`). The server's
`PromptProcessingBatch` never does, so in the server the list is a fixed 16,439 ids. I tokenized the server's own answer
(`srv-f2k-32768-1.json`) with the official `tokenizer.json`. These tokens are outside the list, and every one of them
is in the prompt:

| position | token | | position | token |
|---:|---|---|---:|---|
| 14, 47, 63 | `Ġarchiv` | | 83, 99, 119 | `LET` (VIOLET) |
| 38 | `ĠReply` | | 85, 101, 121 | `ANCH` (ANCHOR) |
| | | | 88, 104, 124 | `472` (4729) |

Each one is a guaranteed rejection that ends its round early. I compared the server's 10-token progress lines against
Studio 1's pre-0016 server run (`results/gates/studio1/server/srv-p4.out`, 75.9 tok/s). The head-to-head run is 334 ms
slower in total, and 322 ms of that falls in the segments that contain these tokens. The machine difference (Studio 2
is ~5% slower) and 0016's faster drafting roughly cancel out. EXPERIMENTS 18:10–18:40 measured a harsher version of
this: an ids < N subset, which also lacked common words, dropped acceptance from 88% to 71% and gave
**62–64 tok/s**. The server's 63.2 sits in that band.

**2. Other differences between the CLI and server decode paths:**

| | CLI (`generate_step` → `run_speculative_rounds`) | server (0014/0015) |
|---|---|---|
| prompt ids in the draft list | yes | no |
| work per round | the round loop only | each token goes through a full server iteration (`SpeculativeGenerationBatch.next()` drains a round only if `round_meta` is a dict; `_dflash_rounds` yields `None`/a logprob row) |
| streams | one (`@on_generation_stream`, `ar.py:276`) | batch on `mx.default_stream` (`server/generation.py:2161`); verify/commit on `_generation_stream.generation_stream`, a separate stream |
| MLX buffer-cache clears during decode | none | every 50 server steps = every 50 tokens (`ar.py:3749–3758`) |
| env | `EVAL_EVERY=0`, no `MLX_MAX_MB_PER_BUFFER` | `scripts/env.sh`: `EVAL_EVERY=1`, `MB=2` |

Calibration: before 0016 on Studio 1, these differences cost only about 2% (server 75.9 vs CLI 77.4 at 32k). Most of
today's gap is item 1.

**3. Reading.** Four contributors:
- **Draft-head prefill.** Studio 1 server with the draft off vs on (`srv-c0` vs `srv-p4`, warm) read 1079.1 → 1060.9
  at 4k (−1.7%) and 956.4 → 950.9 at 32k (−0.6%). MLX-Serve with its draft on vs off reads 1012.4 vs 1009.9.
- **Env.** The CLI baselines (987 / 872 at step 2048, 1013 / 888 at step 4096; `results/gates/studio2/p10b-*`) used
  `EVAL_EVERY=0`. The head-to-head server used `EVAL_EVERY=1`, which syncs after each of the 48 visits of every chunk.
- **Step 4096.** In the server, step 4096 is about 3–4% slower than step 2048 (`srv-f4k.log`: the first 4,096-token
  chunk takes 4.34 s vs 4.18 s for two 2,048-token chunks; whole reads 952 / 838 vs 979 / 861), while the CLI is
  2.6% / 1.8% faster at 4096.
- **Last prompt token.** The separate last-token step takes 43–45 ms (the gap between `4077/4078` and `Prefill
  completed`), which is 1.0% of the 4k read.

**4. What MLX-Serve does differently.**
- **Prefill.**
  - Same MLX expert kernel (`fp_gather_qmm_rhs`, 16×32 tiles; unchanged in its MLX pin). It fuses gate_up into one
    gather with N=2304 and passes **no** global scale; it multiplies the per-expert factor afterwards in f32
    (`layers.zig` `expertMm`, 403–428).
  - The glue between matmuls is a few kernels per visit (`kernels.zig`: `pre`, `prep`, `mid`, `norm`, `topk`, `act`,
    `sum`, `state`).
  - Per chunk it does `mlx_eval` + `mlx_clear_cache`, like our CLI (`engines/mlx/generate/init.zig:555–558`). The last
    prompt token also runs alone (`init.zig:349`), but as a one-row fused visit.
  - For the draft, prefill keeps only the last `window − 1` = 1,023 rows of the trunk's taps (`plugin.zig` `Spec.push`,
    91–107, `async_eval`). The drafter's context K/V are projected lazily in rounds (`draft.zig` `ContextKv`).
- **DSpark.**
  - A round drafts 7 and verifies while the running probability is ≥ cut. The cut is computed online:
    `clamp(1.34·√(row cost / plain step), 0.3, 0.8)` (`plugin.zig` `RowCost`, 287–342; log: "cut 0.61 (a verified row
    0.21 of a plain step)").
  - With 0 drafts it runs a plain step. The verify appends nothing, then commits the accepted rows.
  - The engine skips drafting on bad stretches and probes again later (`plain_ticks=12 skipped=12 probes=5` at 32k;
    `plain_ticks=49` at 4k).
  - It clears the allocator cache every 256 generated tokens (`prefill_chunk.zig:423`, `generator.zig` `advanceStep`),
    so never inside a 125-token answer.
- **Its 32k log in numbers.**
  - 26 attempts with 83 accepted drafts and 4.85 rows on average, plus 12 plain steps at 19.68 ms, for 125 tokens in
    1.671 s.
  - That is ≈ 55 ms per attempt including drafting, and 4.19 tokens per attempt (5.15 per attempt that verified at
    least one draft).
  - Ours (CLI, Studio 2): 70.7 ms per round and 5.04 tokens per round.
  - So per round we accept more and pay ~28% more time. Per extra verified row the cost comes out similar for both
    (≈0.4 plain step; my derivation, assuming ~5 ms of drafting per MLX-Serve attempt). The real gap is the
    server-only losses above, plus MLX-Serve choosing plain steps where drafting doesn't pay.

## Ranked ideas

### Server writing at 32k (top 3)

#### 1. Give the server's draft head the prompt's token ids

- **Hypothesis:** see finding 1. Restoring the prompt ids returns the server to the CLI's acceptance (~0.88 and
  ~25 rounds for 126 tokens, as in the CLI runs).
- **Change:**
  - `generate/ar.py`, `PromptProcessingBatch.__init__` (0015's block, lines 2204–2209): when
    `draft_model is not None and len(input_ids) == 1`, set
    `draft_model.__dict__["_lookup_prompt_ids"] = mx.array(ids, mx.int32)`. Use `apc_meta[0]["full_input_ids"]` when a
    warm prefix is used, else `input_ids[0]`.
  - `speculative/drafters/inkling2_dspark/dspark.py`, `_build_vocab_subset` (362–405): pad the subset to a coarse
    bucket (e.g. a multiple of 4,096 instead of 256). The compiled Markov loop is keyed by subset size (line 686), and
    with prompt ids every new prompt would otherwise retrace it. Optionally pre-warm the 16k–28k buckets at server start.
- **Expected gain (estimate):** 32k writing 63.2 → **69–73 tok/s** (+6 to +10). Taking back all ~0.32 s lost in
  the segments with unproposable tokens would give ~75 (126 tokens in ~1.67 s); I discount that because progress
  lines blur round boundaries. 4k writing gains too (same 13 tokens): 68.5 → ~72–75.
- **Measure (≤15 min):** server A/B with the same env and client, 3 requests per prompt (the first is a warm-up). Log
  `len(accept_lens)` and `sum(accept_lens)` at request end; expect ~25 rounds and ~0.88 acceptance at 32k.
- **Quality risk:** none for tokens: the drafter only proposes and the exact verifier decides every token. Check that
  answers hash-equal the no-draft server's, as in the 0015 runs. Watch for a compile stall the first time a bucket is
  used.

#### 2. Keep allocator resets and per-request setup out of the decode window

- **Hypothesis:**
  - (a) `_decode_step` runs `mx.eval(cache_states)` + `mx.clear_cache()` every 50 server steps (`ar.py:3749–3758`;
    `DEFAULT_CACHE_EVAL_INTERVAL = 50`, line 65). With one token per step (idea 3), that is at tokens 50 and 100 of a
    126-token answer. After each clear, the next verify round allocates its few thousand intermediate buffers fresh.
    The CLI's drafted path never clears (`generate_step`'s every-256 clear sits in the non-speculative loop), and
    MLX-Serve clears every 256 tokens. In both server runs, the slowest 10-token segment is 50→60 (297 ms in the
    head-to-head, 213 ms in Studio 1's run, against an average of ~130 ms), and that text has no hard-to-draft tokens.
  - (b) 0016's `_build_vocab_subset` runs in `reset()` (dspark.py 407–411), which `_dflash_rounds` calls when decode
    starts (dflash.py:679). It re-parses the 524 KB JSON, runs numpy set operations, gathers 16.6k head rows and does a
    blocking `mx.eval`, all inside the server's decode timer on every request. The first 10 tokens took 184–186 ms in
    the head-to-head vs 131 ms in Studio 1's pre-0016 run.
- **Change:**
  - (a) In `_decode_step`, skip `mx.clear_cache()` (keep the eval) when the generation batch is a
    `SpeculativeGenerationBatch` with `can_extend == False`. For the A/B, just set
    `MLX_VLM_BATCH_CACHE_EVAL_INTERVAL=0`.
  - (b) Load the JSON once at module import. Build the subset in `begin_target_prefill` (dspark.py:419, which runs
    during the first prefill chunk) with `mx.async_eval`. `reset()` skips the build if this request already has it.
- **Expected gain (estimate):** +1 to +3 tok/s at 32k (≈20–40 ms for the subset build, 20–80 ms per cache clear, on a
  ~2.0 s decode).
- **Measure (≤15 min):** (a) is an env-only A/B. For (b), compare the time from `Decode started` to the first
  progress line, and total decode.
- **Quality risk:** none (scheduling and allocation only). Check peak memory: MLX's buffer cache may grow to its limit,
  and 32k peaks around 22 GiB today.

#### 3. Make the server's round loop work like the CLI's

- **Hypothesis:** see the table in finding 2.
  - Each token currently passes through a whole server iteration: continuations, cancellations, the request queue,
    `_step_group` and response queues.
  - The next round's draft is only built after the last token of the round has been handed to the server. Only
    `prepare_target_hidden` is scheduled before the yields, and the Inkling drafter doesn't define it.
  - The draft runs on another stream than the verify and commit, so every round crosses streams twice.
  - Together this is the structural ~2% server-vs-CLI gap seen on Studio 1, plus the chance to overlap token emission
    with the draft's GPU time.
- **Change:**
  - `speculative/dflash.py` `_dflash_rounds` (645–878): add `per_round=False`. When true:
    - After the commit (line 850), unless the round hit `max_tokens` or a stop id, build the next round's draft and
      `mx.async_eval` it (tokens and confidence), then stash it.
    - Yield the round's tokens once, as `(new_tokens, {"round_len": n})`.
    - At the top of the loop, use the stash; drop it if the thinking budget forced a different token.
  - `speculative/utils.py` `run_speculative_server_rounds` (dflash branch, 339–357): pass `per_round=True` and stop at
    the first stop token.
  - `generate/ar.py` `SpeculativeGenerationBatch.next()` (2125–2154): append every token of the round in one call.
  - `server/generation.py:2161`: use `_generation_stream.generation_stream` instead of `mx.default_stream(...)`, so the
    batch and the round loop share one stream.
- **Expected gain (estimate):** +1 to +2.5 tok/s at 32k (≈1.5–2 ms of GPU idle per round × ~25 rounds).
- **Measure (≤15 min):** server A/B. Add `perf_counter` stamps and log the mean gap between the walk's sync and the
  next draft's `async_eval`.
- **Quality risk:** none for tokens (same proposals and verifier). A cancelled request must discard its stashed draft.

### Server reading

#### 4. Lean, asynchronous draft-head prefill

- **Hypothesis:** with `INCREMENTAL_DRAFT_PREFILL=1`, every chunk runs `_project_target_hidden` (dspark.py 468–473).
  For all three draft layers it mixes 4 taps in f32 over all 2,048 rows, projects the context K/V and calls
  `mx.eval(entry.state)`. That makes the server's otherwise asynchronous chunk (`prompt_step`, `ar.py:2635`)
  synchronous.
  - Two of the three layers (local layers 0 and 2, window 1,024) keep only the last 1,024 rows.
  - The finish adds three more blocking evals and a 1,025-row re-projection (438–466).
  - `DraftWindowCache._alloc` zero-fills a 65,544-row K and V buffer (268 MB) on every request (77–82).
  - Measured cost (finding 3): −1.7% at 4k and −0.6% at 32k. MLX-Serve pays nothing for its draft during prefill.
- **Change:**
  - Per chunk, project only the global draft layer (index 1) and use `mx.async_eval`.
  - In `finish_target_hidden`, project local layers 0 and 2 once from `_prefill_global_tail` (last 1,024 rows) plus
    the final token, after setting their offsets to `prompt_len − tail_len` (rope positions unchanged). Use async evals.
  - Size `DraftWindowCache` to `min(window, prompt + max_tokens) + block`, passing the length in via
    `begin_target_prefill`, and reuse the buffer across requests.
- **Expected gain (estimate):** reading +1 to +1.5% at 4k (≈ +10 to +15 tok/s) and +0.4 to +0.6% at 32k.
- **Measure (≤15 min):** server reading with the draft on vs off (`MLX_VLM_DRAFT_MODEL` unset) at 4k and 32k. Target:
  draft on within 0.3% of draft off.
- **Quality risk:** none for output tokens (this only touches the draft's context). Proposals could change in the last
  bit if MLX picks a different GEMM tile for 1,025 rows than for 2,048; compare acceptance in the same run.

#### 5. Server env parity (`EVAL_EVERY`, `MLX_MAX_MB_PER_BUFFER`)

- **Hypothesis:** `EVAL_EVERY=1` and `MB=2` are the 1M-memory settings. The head-to-head server is the only run that
  used them with the draft head on.
  - `EVAL_EVERY=1` does a blocking `mx.eval(x, c.state)` after each of the 48 visits of a chunk
    (`inkling2_loop/language.py:320–335`): 96 syncs at 4k and 768 at 32k, each leaving the GPU idle while Python builds
    the next visit.
  - `MB=2` commits a Metal command buffer after almost every kernel.
  - The CLI baselines and Studio 1 server runs used `EVAL_EVERY=0` and no MB setting. Studio 1's server also launched
    through a wrapper with `set_wired_limit(max)` and `set_cache_limit(1 GiB)` (`srv.sh`).
  - This is my best guess for why step 4096 is slower only in the server.
- **Change:** in `scripts/env.sh`, set both only behind a `LONGCTX=1` (≥524k) flag. Alternatively keep `EVAL_EVERY=1`
  and add the existing `MLX_VLM_EXACT_PREFILL_EVAL_ASYNC=1` (`language.py:33`). Launch the server the way Studio 1 did.
- **Expected gain (estimate):** reading 0 to +1.5%. The LEADERBOARD row "full-chunk eval sync vs async" (1014.5 →
  1022.5, +0.8%) is the per-chunk version of this effect; per-visit syncs cost more. Writing ±2 tok/s; the sign is
  unknown, because MB size changes how early the GPU starts each draft and verify graph.
- **Measure (≤15 min):** three server sessions: env.sh as is; `EVAL_EVERY=0`; `EVAL_EVERY=0` with MB unset.
- **Quality risk:** none (scheduling only; already gated bit-identical). Memory at ≤262k is far below the cap.

#### 6. Prefill step 4096 in the server, once 4 and 5 are in

- **Hypothesis:** the CLI reads 2.6% faster at 4k and 1.8% faster at 32k with step 4096 (`p10b-pf-*-NQ` vs `-N`).
  The server loses that, plausibly because of the per-visit and draft-prefill syncs (ideas 4 and 5).
- **Change:** `--prefill-step-size 4096` for the server only (peak memory +3.4 GiB at 32k). MLX-Serve uses 2048.
- **Expected gain (estimate):** +1.5 to +2.5% at 4k and +1 to +1.5% at 32k, on top of ideas 4 and 5.
- **Measure (≤15 min):** server at both step sizes with the same env, plus `scripts/quality/top1.py DRAFT=8` on the
  8-prompt set.
- **Quality risk:** gated, not bit-identical: it changes the fork's chunking. Fork vs fork at step 4096 gives
  teacher-forced top-1 99.6% and identical needle answers (`results/gates/studio2/top1/top1-fork-step4096.json`).
  The CLI default can stay at 2048.

#### 7. Message gates: batch the three stream gates, fuse the K=64 projections with the tail

- **Hypothesis:** each loop visit applies 4 message gates (96 per chunk). `message_gate_exact` (prefill_fused.py
  495–504) runs:
  - a norm + concat kernel;
  - `down_proj`, a 2048×4608 @ 4608×64 matmul whose N=64 gives only 32 output tiles for a 40-core GPU;
  - two f32 matmuls with K=64 that write 2048×2304 f32 forget and gain logits (2 × 18.9 MB);
  - a compiled tail that reads them back along with f32 casts of hidden and message.

  The LEADERBOARD measures the message gate at **2.0 TFLOP/s**, vs 11–13.7 for the projections. That puts 0.23 TFLOP
  per chunk at ≈110 ms, about 5% of a 2.1 s chunk (the pre-0010 profile said "message gates 6%").
- **Change:**
  - (1) The q, k and v gates of a visit have the same shapes. Run their `down_proj` as one batched matmul
    `[3, rows, 4608] @ [3, 4608, 64]` (stack the weights once), and their forget/gain matmuls as batched
    `[3, rows, 64] @ [3, 64, 2304]`.
  - (2) A kernel that computes forget and gain for a row tile with simdgroup 8×8 MMAs in steel GEMM's K order, then
    applies `(1 − tanh f)·h + tanh g·m` in registers, reading h and m as bf16. The f32 logits never touch memory.
- **Expected gain (estimate):** +1.5 to +3% reading at both lengths.
- **Measure (≤15 min for step 1):** per-site microbench in the PF_CHECK style: max |Δ| = 0 against
  `message_gate_exact` at 36–2,048 rows, plus time per gate; then the sha gate at 4k and 32k.
- **Quality risk:** each row count must be proven bitwise. The earlier inexact attempt merged forget and gain along N;
  batching keeps M, N and K, but MLX might pick a different tile for a batched call. Keep 0010's shape gate.

#### 8. A faster last-prompt-token step

- **Hypothesis:** after the last chunk, `PromptProcessingBatch.generate()` (`ar.py:2674`) runs:
  - the last prompt token through the eager model call, with tap capture;
  - the draft finish: three blocking evals and a 1,025-row re-projection;
  - `_full_rings`;
  - a cache eval.

  That is 43 ms at 4k (1.0% of the read). MLX-Serve also runs the last token alone, but as a one-row fused visit.
- **Change:** run the last token through `fast_decode.compiled_decode`, extended to return the four tap captures
  (the drafter's `target_layer_ids` 5, 23, 41, 48). Compiled and eager are bitwise equal (0002). Make the draft finish asynchronous
  (idea 4).
- **Expected gain (estimate):** +0.4 to +0.6% at 4k (−15 to −25 ms); negligible at 32k.
- **Measure (≤15 min):** the gap between `4077/4078` and `Prefill completed` in the server log.
- **Quality risk:** none if the captures equal the eager path's (sha check at 4k and 32k).

#### 9. Bigger tiles for the prefill expert GEMM (`gather_qmm_rhs`)

- **Hypothesis:** the expert GEMMs (about half of prefill) run at 79–82% of peak (LEADERBOARD `op-switchglu-nvfp4`
  11.7 TFLOP/s), vs ~96% for dense bf16. MLX 0.32.4 fixes `bm=16, bn=32, bk=32, wm=1, wn=2` in `gather_qmm_rhs`
  (`mlx/backend/metal/quantized.cpp:1749–1750`, "TODO: Tune the block sizes").
  - At BM=16, the NVFP4 dequant of each 32×32 weight tile (with the global scale) is shared by only 16 rows. By my
    count that adds 20–30% ALU work per MAC, which matches the gap.
  - BM=32 halves that but pads more: an expert's last row tile is half empty on average, about 12% of 16,384 routed
    rows at BM=32 vs 6% at BM=16 (3% vs 6% at step 4096).
  - MLX-Serve runs the same 16×32 kernel, so this would be an edge rather than catching up.
- **Change:** build MLX 0.32.4 from source with a row-count-dependent tile (BM=32, WM=2 when an expert averages ≥64
  rows) and the matching `fp_gather_qmm_rhs` instantiation. Keep BK=32 and the `QuantizedBlockLoader` with the global
  scale, so each output's K order is unchanged.
- **Expected gain (estimate):** 0 to +3% reading (more at step 4096).
- **Measure (≤15 min, after a ~15–25 min MLX build):** microbench `gather_qmm` at the real shapes (16,384 sorted rows
  over 128 experts; 2304→1152 and 1152→2304; NVFP4 with global scale). Compare bitwise against the stock tiles and
  time both.
- **Quality risk:** must be bitwise; steel's K loop doesn't depend on BM or BN, but verify it. If it isn't bitwise,
  drop it.

### More writing ideas

#### 10. Compile the draft head's three layers

- **Hypothesis:** in the Studio 2 round profile, draft build drops from 3.7 ms to 2.5 ms with 0016's compiled Markov
  loop. The rest is the three draft layers run eagerly: `draft_block` (641–673), where `_call_window` (168–182) slices
  the window to `[:end]`, a shape that changes every round. The GPU is idle while this graph is built.
- **Change:** attend over the full fixed `DraftWindowCache` buffer with an additive mask for unwritten slots (fixed
  shapes; the local rings are already a fixed 1,024). Compile `draft_block` per block size, as the verifier is.
- **Expected gain (estimate):** −1 to −2 ms per round, so +1 to +2 tok/s at 32k.
- **Measure (≤15 min):** 0016's `MLX_VLM_INKLING2_DRAFT_TIMERS=1` before and after, plus a server A/B.
- **Quality risk:** none for tokens. Masked attention over more keys may change proposal bits; check that acceptance
  is unchanged.

#### 11. Adopt MLX-Serve's draft policy: measured cut, plain steps, skip and probe

- **Hypothesis:** see finding 4. We always verify at least one draft (`_conf_min = 1`, dflash.py:715/775) and always
  draft, even through stretches where the drafter can't help.
- **Change:** in `_dflash_rounds`:
  - Allow `keep = 0` and run a 1-row exact verify (same tap capture) instead of a 2-row one.
  - Derive τ from measured round time per row count (rounds are already timed for `_AdaptiveBlock`), using
    MLX-Serve's `1.34·√(row/plain)` as the starting curve.
  - After k rounds with 0 accepted, skip drafting for m ticks, then probe.
- **Expected gain (estimate):** 0 to +2 tok/s at 32k (smaller once idea 1 lands).
- **Measure (≤15 min):** server A/B on both prompts plus one chat prompt from the 8-prompt set.
- **Quality risk:** none for tokens.

#### 12. Profile the verify itself before touching its kernels

- **Hypothesis:** at 32k a ~5.5-row exact verify costs 54–60 ms (≈2.9 plain steps). Its bandwidth floor is ~30 ms:
  attention weights 2.6 GB + ~38 distinct experts per visit ≈ 8.8 GB + head 0.9 GB + KV 1.2 GB ≈ 13.7 GB at
  ~450 GB/s. That is ~50–55% of bandwidth, vs ~75% for a plain step. MLX-Serve's attempt costs about the same per row,
  so this is headroom rather than a gap.
- **Change:** measure first. Run timing-only ablations inside `ExactVerifier._step_batched` (exact_verify.py 547–607),
  the way trackB did for decode: experts (`moe_rows`), projections (`gemv_m`), global and sliding attention,
  attention-residual mixing, head. Then attack the largest, most likely the multi-row MoE (one pass per expert tile for
  all rows that picked it).
- **Expected gain (estimate):** if the verify reached 65% of bandwidth: −10 ms per round, so +8 to +10 tok/s at 32k.
  This is high effort.
- **Measure (≤15 min):** the ablation run on a synthetic 32k cache (`dbench`-style).
- **Quality risk:** any kernel change must keep every row bitwise equal to the decode step (the existing promotion
  ladder).

## Rough outcome if the estimates hold

These are estimates; they assume each step's gain mostly survives the ones before it.
- **Writing at 32k:** 63.2 → 69–73 after idea 1 → 72–77 after ideas 2 and 3 → +1 to +3 more from ideas 10 and 11.
- **Reading at 4k:** 979.6 → ~990–995 after idea 4 → ~995–1010 after ideas 5 and 8 → ~1010–1035 after idea 6, with
  idea 7 as margin.
- **Reading at 32k:** 861.5 → ~866–875 after ideas 4 and 5 → ~875–885 after idea 6 → ~888–905 after idea 7.

## Checked and not recommended

- **Moving to MLX-Serve's MLX pin (`c69303b`).**
  - The prefill MoE kernel `fp_gather_qmm_rhs` and its tiles are unchanged.
  - #4709 (fast fp4 dequant) only serves dense `quantized_matmul` without a global scale. Our dense projections are
    bf16, and our experts go through gather with a global scale.
  - #4712 and #4723 speed up MLX's few-row qmv and SDPA, which our exact verifier doesn't use.
  - #4568 (cached custom-kernel source hash) saves ~0.4 µs per custom-kernel dispatch, roughly 0.1–0.2 ms per round.
  - #4769 changes `addmm` rounding, which the draft head's Markov step uses.
  - Not worth a rebuild on its own.
- **A non-exact verify:** it was slower here (EXPERIMENTS 17:22), and it gives up "same tokens as no draft".
- **MLX-Serve's 1,023-row draft context:** we measured an 8k window: acceptance dropped and writing fell to 56.9.
- **Folding the last prompt token into the last chunk:** MLX-Serve doesn't do it either. It changes the fork's
  numerics for the last token and saves little more than idea 8.

## Shared measurement protocol (Studio 2, ≤15 min per A/B)

1. Start `mlx_vlm.server` with one env, the same launch wrapper every time, `--prefill-step-size 2048` and
   `--max-num-seqs 1`. Run the head-to-head client three times per prompt; the first request is a warm-up.
2. Read the server's own `Request completed` lines (the same numbers as the charts). For writing, also log rounds and
   accepted drafts per request.
3. Check that the answer text hashes match a no-draft server run. Ideas 1–5 and 8–12 must not change tokens; idea 6
   (and 7 or 9 if they aren't bitwise) go through `top1.py` and the needle check.
4. Run at most two or three configurations per session. Studio 1 is ~5% faster, so compare within one machine.
5. Caveat when comparing against MLX-Serve: its decode rate may exclude the first token (one token is ≈0.8% of 126).
