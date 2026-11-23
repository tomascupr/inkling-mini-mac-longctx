# Benchmarks and audit

The companion to the README. Every number here comes from a file in this repo, named next to it. The tables between
`<!-- gen:... -->` markers are written by `python scripts/charts.py` straight from the logs, never by hand.

**Scope.** All claims are for the M4 Max Mac Studio (40-core GPU, 64 GB) and nothing else. We have two of them:
*Studio 1* and *Studio 2*. Studio 2 runs about 5% slower than Studio 1 on the same code, so numbers are only compared
within one machine. The 4k, 32k, 131k and 524k head-to-head vs MLX-Serve is on Studio 2; the 262k session is on Studio 1.

**Model.** Official `thinkingmachines/Inkling-2-Mini` weights (Hub revision `ed40e7da`, NVFP4 + MXFP8) as shipped, bf16
KV cache. No requantizing, no FP8 KV, no sparse or approximate attention, anywhere in this document.

Contents: [the claim](#the-claim) · [head-to-head vs MLX-Serve](#head-to-head-vs-mlx-serve) ·
[quality](#quality-what-is-verified-on-what) · [settings](#settings-and-environment-variables) ·
[what did not work](#what-did-not-work) · [long context](#long-context-studio-1-command-line) ·
[not measured](#not-measured-yet) · [scripts](#scripts)

## The claim

| README says | Backed by |
|---|---|
| +5.4% faster writing at 4k, +4.3% at 32k | Sessions h29 and h30 below; our slowest counted run beats MLX-Serve's fastest in both sessions |
| +9.6% faster writing at 131k | Session h31 (one session, 2 counted runs each side); our slowest counted run beats MLX-Serve's fastest |
| +18.3% faster writing at 262k | Session h32 on Studio 1. Headline is every run (32.1 → 38.0), not the drop-first 30.4 → 38.7 (+27.3%), because dropping first leaves MLX-Serve with one run and its first run was faster than its second. Our slowest of the three still beats MLX-Serve's fastest of the two. |
| +7.4% faster writing at 524k | Session h33 on Studio 2. Headline is our drop-first run (17.3) against MLX-Serve's only counted run (16.1). All-runs on our side would be +8.7%; the headline is more conservative. See the 524k notes below. |
| +3.2% faster reading at 32k, +4.7% at 131k, +6.2% at 262k, +2.6% at 524k | Same tests: h29 and h30 at 32k, h31 at 131k, h32 at 262k (drop-first: 477.9 → 507.9), h33 at 524k (drop-first ours 311.7 vs MLX-Serve's only run 303.8) |
| Reading at 4k is level | Pooled +0.36%; in h30 the ranges overlap, so this is a tie, not a lead |
| 8/8 greedy answers identical to the reference, 520/520 teacher-forced top-1 | Gate g26 on the stack through 0018 (not the exact final stack, see [quality](#quality-what-is-verified-on-what)) |
| Hidden fact found in 20/20 of our head-to-head answers | `needle_found: True` on every request in `srv-h29.out` (6), `srv-h30.out` (6) and `srv-h31.out` (3); h32 session log (`h2h32-262k.log`) records true/true/true on our three 262k answers; h33 session log (`h2h33-524k.log`) records true/true on our two 524k answers |

Not claimed: anything on other Macs; concurrent or batched requests (both servers ran one request at a time); prompts
under 1,024 tokens (our draft head is off there); 1M against MLX-Serve; the first request at a new prompt
length (usually slower on writing: ours at 4k and 32k, both servers' at 131k and ours at 262k; MLX-Serve's first 262k writing run was faster than its second; [disclosed below](#first-request-at-each-prompt-length)).

## Head-to-head vs MLX-Serve

### Setup

- **Machine and order.** Both servers in the same session on the same Mac, one at a time: MLX-Serve first,
  stopped, then ours. 4k and 32k (h29, h30), 131k (h31) and 524k (h33) ran on Studio 2; 262k (h32) ran on Studio 1. On 22 Nov
  2026: h29 started 22:33 CET, h30 22:42 CET (4k and 32k), h31 23:10–23:29 CET (131k), h32 23:15–23:59 CET (262k),
  h33 23:38 CET–23 Nov 01:16 CET (524k). The order was the same in every session (not alternated).
- **Same model and prompts.** Both load the same weights directory (`thinkingmachines--Inkling-2-Mini`, about 18.19 GB)
  with bf16 KV (MLX-Serve logs `kv-quant: off`). Prompts: MLX-Serve's own needle prompts
  `mlxserve/prompts/user-{4096,32768,131072,524288}.txt` and the 262k needle prompt in h32 (4,078, 32,750, 131,054, 262,126 and 524,270 prompt tokens), greedy (temperature 0),
  `max_tokens` 256, streaming, one request at a time. In h29 and h30 each server gets 4k three times, then 32k three
  times; in h31, 131k three times; in h32, MLX-Serve ran 262k twice after a 4k 16-token warm-up, and ours ran 262k three
  times; in h33, MLX-Serve ran 524k twice after a 4k 16-token warm-up (the second request was refused; see below), and ours ran 524k twice.
  MLX-Serve also gets a 16-token warm-up at 4k first in the Studio 2 sessions.
- **MLX-Serve.** `mlx-serve 26.11.4-pre-release.1` (ships MLX 0.32.5), its DSpark draft head (block 8), prefill chunk
  4096 (its own memory-sized choice; in h32, memory-sized down from 8192), context 65,536 (163,840 in h31, 294,912 in h32, 557,056 in h33),
  `--max-concurrent=1` (h32 also `--timeout 7200`; h33 `--timeout 0`). Rate from its log line
  `<- P+G tokens streamed [prefill: X tok/s, decode: Y tok/s]`. Full logs: `mlxserve-rerun-2233-draft-on.log` (h29),
  `mlxserve-rerun-2242-draft-on.log` (h30), `mlxserve-rerun-2310-draft-on.log` (h31); the request lines are copied
  into `h2h29.log`, `h2h30.log`, `h2h31-131k.log`, `h2h32-262k.log` and `h2h33-524k.log` (h32's and h33's MLX-Serve full logs were not committed).
- **This repo.** `mlx_vlm.server` with every patch through 0019 (`scripts/install.sh`), mlx 0.32.4, the
  `scripts/serve.sh` settings: `scripts/env.sh` (draft head block 8, exact verify, confidence cut 0.65, 16,384-id
  candidate list), prefill step 8192, MLX buffer cache at the working-set size, async per-visit prefill eval,
  sliding-window attention in 64-query blocks; `--max-num-seqs 1`. Rate from the server's own log line
  (`LOG prompt_tokens=P ... prefill=X tok/s decode=Y tok/s`), copied into the session logs. Client records with the
  needle check: `srv-h29.out`, `srv-h30.out`, `srv-h31.out` (h32 and h33 have no separate client record in the repo; the session
  logs themselves record the needle check). The configuration is the one recorded in
  `docs/EXPERIMENTS.md` (22:33, 22:42, 23:10, 23:15 and 23:38 entries); no environment snapshot or full server log was committed for
  these sessions. h32 and h33 used `scripts/serve.sh` plus the patch 0019 overlay copied from the Studio 2 h30 stack, with `BQ_SWA=64`.
- **Token counts.** Each server counts its own: MLX-Serve logs 122 generated tokens at 4k, 125 at 32k and 131k, 128 at 262k, and 131 at 524k; ours
  123, 126, 133, 129 and 132. Rates are each server's tokens over its own time.
- All files are in `results/h2h-20261122/`.

### Counting and rounding

- **Both sides:** every request except the first one at each prompt length in the session, so 2 per prompt length per
  session on our side, and 3 at 4k and 2 at 32k and 131k on MLX-Serve's. At 262k MLX-Serve ran twice, so dropping first
  leaves it with one run. MLX-Serve's first 4k request is its own 16-token warm-up; at 32k, 131k and 262k its first
  request is a full one and is dropped like ours.
  At 524k MLX-Serve completed only one request, so that sole run is counted; our first 524k request is still dropped.
  If a drop-first would leave one side with a single run, and the mean of every run at that length gives a smaller
  1-decimal delta, the headline, README table and charts use the all-runs figures (262k writing only). The 524k
  headline stays on drop-first ours vs MLX-Serve's only run, which is more conservative than counting our first too.
  [The first requests and what counting them changes are below](#first-request-at-each-prompt-length).
- Means are plain means of the server-reported rates. Shown to 0.1 with **ours rounded down and MLX-Serve's rounded
  up**. Differences (Δ) are computed from the unrounded means and rounded down. So no rounding favours this repo.
- **Verdict.** *Faster* only if, in every session, our slowest counted run beats MLX-Serve's fastest. *Behind* if
  MLX-Serve's slowest beats our fastest in every session. Anything else is *level*. Per session the same test reads
  ahead / level / behind.

### Results, pooled over sessions (h29 and h30 at 4k and 32k, h31 at 131k, h32 at 262k, h33 at 524k)

<!-- gen:h2h-pooled -->
| Prompt | | MLX-Serve mean (n, range) | This repo mean (n, range) | Δ | Verdict |
|---|---|---:|---:|---:|---|
| 4k (4,078) | reading | 1,036.2 (6, 1,027.6–1,042.1) | 1,039.8 (4, 1,036.3–1,042.0) | +0.36% | level (+0.36%) |
| 4k (4,078) | writing | 76.3 (6, 75.6–76.8) | 80.4 (4, 80.1–81.1) | +5.49% | +5.4% faster |
| 32k (32,750) | reading | 898.4 (4, 897.8–899.0) | 927.8 (4, 927.3–928.5) | +3.28% | +3.2% faster |
| 32k (32,750) | writing | 72.6 (4, 71.6–74.2) | 75.7 (4, 75.1–76.6) | +4.37% | +4.3% faster |
| 131k (131,054) | reading | 674.0 (2, 673.9–674.0) | 706.2 (2, 706.2–706.2) | +4.78% | +4.7% faster |
| 131k (131,054) | writing | 51.7 (2, 51.6–51.7) | 56.6 (2, 56.6–56.7) | +9.68% | +9.6% faster |
| 262k (262,126) | reading | 477.9 (1, 477.9–477.9) | 507.9 (2, 507.8–508.0) | +6.27% | +6.2% faster |
| 262k (262,126) | writing | 32.1 (2, 30.4–33.8) all runs | 38.0 (3, 36.6–38.7) all runs | +18.38% | +18.3% faster |
| 524k (524,270) | reading | 303.8 (1, 303.8–303.8) | 311.7 (1, 311.7–311.7) | +2.60% | +2.6% faster |
| 524k (524,270) | writing | 16.1 (1, 16.1–16.1) | 17.3 (1, 17.3–17.3) | +7.45% | +7.4% faster |

262k writing: the headline is the mean of every run, not the drop-first mean (30.4 → 38.7, +27.30%). Dropping the first request leaves one side with a single run, and all-runs is more conservative.
524k reading: MLX-Serve completed one request; that sole run is counted. The headline drops our first request (303.8 → 311.7, +2.60%). Counting our first too would be +4.34%.
524k writing: MLX-Serve completed one request; that sole run is counted. The headline drops our first request (16.1 → 17.3, +7.45%). Counting our first too would be +8.69%.
<!-- /gen:h2h-pooled -->

### Every run, per session

<!-- gen:h2h-sessions -->
**h29**, started 2026-11-22 22:33 CET (`results/h2h-20261122/h2h29.log`)

| Prompt | | MLX-Serve first request (not counted) | MLX-Serve counted runs | Mean | Our first request (not counted) | Our counted runs | Mean | Δ | Same-session |
|---|---|---:|---|---:|---:|---|---:|---:|---|
| 4k | reading | 1,023.5 (16-token warm-up) | 1,039.3, 1,027.6, 1,036.6 | 1,034.5 | 893.2 | 1,041.5, 1,039.7 | 1,040.6 | +0.58% | ahead |
| 4k | writing | 101.4 (16-token warm-up) | 76.5, 76.6, 76.1 | 76.4 | 70.2 | 80.1, 80.5 | 80.3 | +5.10% | ahead |
| 32k | reading | 898.3 | 898.1, 897.8 | 898.0 | 928.8 | 927.4, 928.2 | 927.8 | +3.32% | ahead |
| 32k | writing | 74.1 | 72.3, 74.2 | 73.3 | 69.0 | 75.1, 75.7 | 75.4 | +2.93% | ahead |

**h30**, started 2026-11-22 22:42 CET (`results/h2h-20261122/h2h30.log`)

| Prompt | | MLX-Serve first request (not counted) | MLX-Serve counted runs | Mean | Our first request (not counted) | Our counted runs | Mean | Δ | Same-session |
|---|---|---:|---|---:|---:|---|---:|---:|---|
| 4k | reading | 1,025.6 (16-token warm-up) | 1,037.0, 1,034.2, 1,042.1 | 1,037.8 | 855.4 | 1,042.0, 1,036.3 | 1,039.1 | +0.13% | level |
| 4k | writing | 102.0 (16-token warm-up) | 75.6, 76.1, 76.8 | 76.2 | 69.2 | 80.2, 81.1 | 80.6 | +5.88% | ahead |
| 32k | reading | 898.9 | 899.0, 898.4 | 898.7 | 926.7 | 927.3, 928.5 | 927.9 | +3.24% | ahead |
| 32k | writing | 70.9 | 72.0, 71.6 | 71.8 | 70.2 | 75.4, 76.6 | 76.0 | +5.84% | ahead |

**h31**, started 2026-11-22 23:10 CET (`results/h2h-20261122/h2h31-131k.log`)

| Prompt | | MLX-Serve first request (not counted) | MLX-Serve counted runs | Mean | Our first request (not counted) | Our counted runs | Mean | Δ | Same-session |
|---|---|---:|---|---:|---:|---|---:|---:|---|
| 131k | reading | 674.0 | 674.0, 673.9 | 674.0 | 703.5 | 706.2, 706.2 | 706.2 | +4.78% | ahead |
| 131k | writing | 45.0 | 51.7, 51.6 | 51.7 | 52.3 | 56.6, 56.7 | 56.6 | +9.68% | ahead |

**h32**, started 2026-11-22 23:15 CET (`results/h2h-20261122/h2h32-262k.log`)

| Prompt | | MLX-Serve first request (not counted) | MLX-Serve counted runs | Mean | Our first request (not counted) | Our counted runs | Mean | Δ | Same-session |
|---|---|---:|---|---:|---:|---|---:|---:|---|
| 262k | reading | 478.0 | 477.9 | 477.9 | 507.4 | 508.0, 507.8 | 507.9 | +6.27% | ahead |
| 262k | writing | 33.8 | 30.4 | 30.4 | 36.6 | 38.7, 38.7 | 38.7 | +27.30% | ahead |

**h33**, started 2026-11-22 23:38 CET (`results/h2h-20261122/h2h33-524k.log`)

| Prompt | | MLX-Serve first request (not counted) | MLX-Serve counted runs | Mean | Our first request (not counted) | Our counted runs | Mean | Δ | Same-session |
|---|---|---:|---|---:|---:|---|---:|---:|---|
| 524k | reading | 303.8 (only run, counted) | 303.8 | 303.8 | 322.3 | 311.7 | 311.7 | +2.60% | ahead |
| 524k | writing | 16.1 (only run, counted) | 16.1 | 16.1 | 17.7 | 17.3 | 17.3 | +7.45% | ahead |
<!-- /gen:h2h-sessions -->

Notes on these numbers:

- `docs/EXPERIMENTS.md` gives h29 as +0.6% (4k reading) and +3.4% (32k reading), the pooled 4k reading as +0.4%, and
  h31 as +4.8% (131k reading) and +9.7% (131k writing). Those are medians or means of all runs, rounded to the nearest
  0.1. The figures here are re-derived from warm runs and rounded down: +0.58%, +3.32%, +0.36%, +4.78% (+4.7%) and
  +9.68% (+9.6%).
- The h29 4k reading passes the per-session test (our 1,039.7 vs MLX-Serve's best 1,039.3) by 0.4 tok/s; h30 does not.
  4k reading is level.
- 131k rests on one session (h31) with 2 counted runs per side. The gaps are wide against the run-to-run spread:
  reading 706.2 vs at most 674.0, writing at least 56.6 vs at most 51.7.
- 262k rests on one session (h32, Studio 1). Prefill is steady. Drop-first reading is 477.9 → 507.9 (+6.27%, +6.2%).
  Drop-first writing is 30.4 → 38.7 (+27.30%), but that 30.4 is MLX-Serve's only remaining run, and its first run
  (33.8) was faster than its second. The headline therefore uses every run: 32.1 → 38.0 (+18.38%, +18.3%). Our
  slowest of the three (36.6) still beats MLX-Serve's fastest of the two (33.8). `docs/EXPERIMENTS.md` reports
  medians of all runs as +6.2% reading and +20.6% writing.

### First request at each prompt length

The first request at a new prompt length is usually slower on writing, and the default headline counts neither side's.
262k writing is the exception on MLX-Serve, and its headline is handled as noted below.

- **4k.** Ours is also the first request after server start: reading 893.2 and 855.4 tok/s, writing 70.2 and 69.2,
  against 1,036–1,042 and 80–81 afterwards. MLX-Serve's first 4k request is its own 16-token warm-up (its writing rate
  over 16 tokens is not comparable).
- **32k.** Reading is steady on both sides. **Our writing is soft: 69.0 tok/s in h29 and 70.2 in h30**, against
  75.1–76.6 afterwards (68.6 in h28). MLX-Serve's first 32k writing is 74.1 in h29 (its fastest of three) and 70.9 in
  h30 (its slowest).
- **131k (h31).** Reading is steady (MLX-Serve 674.0, then 674.0 and 673.9; ours 703.5, then 706.2 twice). **Writing is
  soft on both sides: MLX-Serve 45.0, then 51.7 and 51.6; ours 52.3, then 56.6 and 56.7.** Our server was started fresh
  for h31, so our first 131k request is also its cold start.
- **262k (h32, Studio 1).** Reading is steady (MLX-Serve 478.0, then 477.9; ours 507.4, then 508.0 and 507.8).
  **Our writing shows the usual first-use dip: 36.6, then 38.7 and 38.7. MLX-Serve writing went the other way: 33.8,
  then 30.4.** Because dropping first leaves MLX-Serve with only that 30.4 run, the headline uses every run
  (32.1 → 38.0, +18.3%) rather than 30.4 → 38.7 (+27.3%).

It looks like a one-time cost of a new prompt length (kernel compilation for new shapes, or KV and buffer growth), not
steady-state decoding; it has not been investigated further. Counting the first requests changes the results as
follows:

<!-- gen:h2h-first -->
| Prompt | | First requests: MLX-Serve → ours | Drop first request | Our first request counted, MLX-Serve's not | All runs counted | Headline |
|---|---|---|---|---|---|---|
| 4k | reading | 16-token warm-up → 893.2 (h29), 16-token warm-up → 855.4 (h30) | +0.36%, level | −4.97%, level | – (MLX-Serve's is its 16-token warm-up) | drop first |
| 4k | writing | 16-token warm-up → 70.2 (h29), 16-token warm-up → 69.2 (h30) | +5.49%, faster | +0.78%, level | – (MLX-Serve's is its 16-token warm-up) | drop first |
| 32k | reading | 898.3 → 928.8 (h29), 898.9 → 926.7 (h30) | +3.28%, faster | +3.28%, faster | +3.27%, faster | drop first |
| 32k | writing | 74.1 → 69.0 (h29), 70.9 → 70.2 (h30) | +4.37%, faster | +1.57%, level | +1.58%, level | drop first |
| 131k | reading | 674.0 → 703.5 (h31) | +4.78%, faster | +4.65%, faster | +4.64%, faster | drop first |
| 131k | writing | 45.0 → 52.3 (h31) | +9.68%, faster | +6.87%, faster | +11.66%, faster | drop first |
| 262k | reading | 478.0 → 507.4 (h32) | +6.27%, faster | +6.24%, faster | +6.23%, faster | drop first |
| 262k | writing | 33.8 → 36.6 (h32) | +27.30%, faster | +25.00%, faster | +18.38%, faster | all runs |
| 524k | reading | 303.8 (only run, counted) → 322.3 (h33) | +2.60%, faster | +4.34%, faster | +4.34%, faster | drop first |
| 524k | writing | 16.1 (only run, counted) → 17.7 (h33) | +7.45%, faster | +8.69%, faster | +8.69%, faster | drop first |
<!-- /gen:h2h-first -->

Counting our first request but not MLX-Serve's is the harshest reading for us. It makes 4k level on both metrics (our
cold start against MLX-Serve's runs after its warm-up) and 32k writing level; 32k reading and both 131k rows stay
faster. Counting both sides' first requests leaves 32k writing level and widens 131k writing, because MLX-Serve's first
131k request is softer than ours; the headline keeps the warm-run +9.6%. At 262k, counting both sides' first requests
leaves reading at +6.2% (same 1-decimal headline as drop-first) and cuts writing from +27.3% to +18.3%, because
MLX-Serve's first writing run was faster than its second; that +18.3% is the README and chart figure. 262k reading
stays on the drop-first means (477.9 → 507.9). At 524k, MLX-Serve completed only one request, so that sole run is
counted and our first request is still dropped (303.8 → 311.7 reading, 16.1 → 17.3 writing). Counting our first too
would be +4.3% / +8.7%; the headline keeps the more conservative drop-first figures.

**524k notes (h33, Studio 2).** MLX-Serve's second request was memory-refused (needed 25,399 MB, 18,496 MB available).
Ours decoded without the draft (it switches off above 262,144 tokens); MLX-Serve drafted. An earlier attempt at our
524k runs in the same session is discarded: the 600s default `MLX_VLM_TOKEN_QUEUE_TIMEOUT` aborted the client stream
while prefill was still running, and a second request was started on top of it. Those numbers are not used. The
counted runs are from a clean rematch with the timeout set to 3,600s.

To add a session, list its log under `h2h_sessions` in `results/h2h-20261122/charts.json` and run
`python scripts/charts.py`: the README table, the charts, the X renders and the tables above gain the new rows under
the same counting, rounding and verdict rules.

### How we got here (earlier runs on Studio 2)

Δ of our counted mean vs MLX-Serve's mean in each same-session pair, same counting as above; MLX-Serve chose prefill
chunk 4096 in all of them. The earlier comparison of ours at 19:09 against MLX-Serve's 15:08 runs
(`mlxserve-draft-on.log`, `srv-f2k.log`) is not listed: it is not same-session, and with one MLX-Serve run at 32k
nothing is left after its first request.

<!-- gen:h2h-history -->
| Session (Studio 2) | Our stack | Reading 4k | Reading 32k | Writing 4k | Writing 32k | Logs |
|---|---|---:|---:|---:|---:|---|
| g17b, 19:52 CET | + 0017 (server passes the prompt ids to the draft head) | −5.3% | −3.6% | +4.6% | +4.1% | `mlxserve-rerun-1952-draft-on.log`, `srv-g17b.log` |
| h26, 20:55 CET | + 0018 (last-chunk fold, lean draft prefill, early candidate list) | −4.9% | −3.4% | +5.6% | +6.8% | `mlxserve-rerun-2055-draft-on.log`, `srv-h26.log` |
| h27, 21:50 CET | + serve.sh (step 8192, no 1 GiB cache limit, async per-visit eval) | −2.5% | +0.4% | +4.7% | +5.9% | `mlxserve-rerun-2150-draft-on.log`, `srv-h27.log` |
| h28, 22:10 CET | + sliding-window attention in 64-query blocks (BQ_SWA=64) | −0.6% | +2.5% | +5.3% | +6.1% | `h2h28.log` |
| h29, 22:33 CET | + 0019 (l2 clips as one kernel per call): final stack | +0.5% | +3.3% | +5.1% | +2.9% | `h2h29.log` |
| h30, 22:42 CET | final stack, repeat | +0.1% | +3.2% | +5.8% | +5.8% | `h2h30.log` |
<!-- /gen:h2h-history -->

## Quality: what is verified on what

**Gate g26** (22 Nov 2026 21:00 CET, Studio 1; `results/h2h-20261122/gate26.log`, `top1-g26.json`). Stack: every patch
through 0018, `scripts/env.sh` defaults with the draft head on. Run with `scripts/quality/top1.py` (Python
`stream_generate`, prefill step 2048) on a fixed set of 8 prompts (6 corpus slices of 1,347 to 14,692 tokens with
questions, 2 chat prompts of 36 and 37 tokens):

- greedy 64-token answers through the draft head identical to the reference on **8/8** prompts;
- teacher-forced top-1 equal to the reference's teacher-forced argmax on **520/520** tokens (100%);
- 502/520 (96.54%) against the reference's greedy text, the same figure as in earlier gates. Because our teacher-forced
  argmax equals the reference's on all 520 tokens, the reference scores the same 502/520 against its own greedy text:
  the gap is between reading the answer back as a prefill and decoding it token by token, not this repo.

The reference is the unpatched community fork (`betizzel/mlx-vlm` at `55ca2e9e`) at prefill step 2048.

**Needle.** The harness line reads `18 VIOLET-ANCHOR-4729` in both `h2h29.log` and `h2h30.log` (logged as "needle
18/18" in `docs/EXPERIMENTS.md`) and `9 VIOLET-ANCHOR-4729` in `h2h31-131k.log` ("all 6 answers" in
`docs/EXPERIMENTS.md`). Our client records show `needle_found: True` on every one of our requests: 6 in `srv-h29.out`,
6 in `srv-h30.out`, 3 in `srv-h31.out`. For h32, `h2h32-262k.log` records `needle_found true/true` (MLX-Serve) and
`true/true/true` (ours), all `VIOLET-ANCHOR-4729` (5/5 in that session). For h33, `h2h33-524k.log` records
`needle_found` true on MLX-Serve's completed request and true/true on our two 524k answers (the refused MLX-Serve
request has no answer). No `srv-h32.out` or `srv-h33.out` was committed, so the README's 20/20 is our 15 from the
client records plus our 3 from the h32 session log plus our 2 from the h33 session log.

**What changed after gate g26, and how each change was checked:**

| Change | Check | Source |
|---|---|---|
| Patch 0019: l2 clips as one Metal kernel per call (`PF_L2K`) | q/k/rel clips `array_equal` to the default at 4,077 rows on both Studios; server A/B output text identical in every run | `docs/EXPERIMENTS.md` (prefill l2 clips entry) |
| `MLX_VLM_RELATIVE_ATTN_BQ_SWA=64` (sliding-window attention in 64-query blocks) | bitwise equal to the default in every tile configuration tested (4,077 rows); same arithmetic, different tiling | `docs/EXPERIMENTS.md` (relative-attention tiles entry) |
| Async per-visit prefill eval, buffer cache at the working-set size | scheduling and memory allocation only; not separately gated | `scripts/serve.sh` |
| Prefill step 8192 in `serve.sh` (the gate uses 2048) | not gated (see below) | |

**Not done: the full gate (8-prompt greedy + teacher-forced top-1) has not been re-run on the exact final stack
(0019 + `BQ_SWA=64`) or with the `serve.sh` settings.** The README says so.

Two caveats beyond that:

- **Prefill chunk size.** The reference fork is not chunk-size invariant: changing the prefill step changes the last
  bits of the hidden states (fork against itself, step 4096 vs 2048: teacher-forced top-1 99.6% on the 8 prompts,
  needle answers identical, `results/gates/studio2/top1/top1-fork-step4096.json`). `serve.sh` uses step 8192, which
  has not been through the 8-prompt gate. MLX-Serve uses its own chunk (4096).
- **Patch 0011** (compiled decode from 1,024 keys, `MLX_VLM_INKLING2_DECODE_MIN_KEYS=1024`, on by default). Below
  4,096 keys it uses split-K attention, which adds in a different order: tokens identical on the gate (it is part of
  g26), log-probabilities not bit-identical. Prompts of 4,096 tokens or more are unaffected.

**The draft head** never changes the output: its guesses are verified with the single-token decode step's own kernels
and reduction order (patch 0007), so the tokens are the same as decoding without it. Checked in g26 (greedy through the
draft head) and in every draft run since 0007 (tokens identical to the no-draft run).

### Earlier exactness results (bit level)

- Stack6 (`stacked.patch` = 0001 + 0002 + 0003): tokens **and** the sha256 of every per-token log-probability vector
  identical to the unpatched fork at 4k, 32k, 131k and 262k (`results/gates/gate-stack6-20261122-1108.log`,
  `results/gates/s6-*.json`; references in `results/reference/`).
- 0008 and 0009 at 1M: tokens and log-probability sha identical to the run without them (`results/gates/h7-1m-kt3-hostvocab.json`).
- 0010 at step 2048: tokens and log-probability sha equal to the fork at 4k and 32k (`results/gates/studio2/p10b-*.json`).
- 0007 (exact draft verify, block 4): tokens and log-probability sha equal to the fork's no-draft run at 32k, 131k
  and 262k (`results/gates/gate-stack7-draft-20261122-1153.log`, `results/gates/s7-draft-k4-*.json`).

### Exactness methodology

1. Fixed prompt: Project Gutenberg text truncated to the target context, needle sentence inserted at the first
   paragraph break past the midpoint, question at the end.
2. Greedy decoding (`temperature=0`), fixed `prefill_step_size` (2048), 256 max tokens.
3. For every generated token, record the token id and hash the full float32 log-probability vector into one running
   sha256 (`scripts/exact_check.py`).
4. Run the unpatched fork and the patched build in the same venv (only `PYTHONPATH` differs) and require identical
   token lists and identical sha. The needle must be found.
5. Promotion ladder: kernel microbenchmarks bitwise equal (or max-abs-diff 0) on random inputs, then a 32k end-to-end
   gate, then 131k. A variant that changes any bit is rejected or reported as non-exact.
6. Relaxed gate for changes that cannot be bitwise (chunk size, 0011): `scripts/quality/top1.py`, the same greedy
   answers on the 8 prompts and at least 99% teacher-forced top-1.

### Standard quality benchmarks (older build)

Run 22 Nov 2026 08:59–11:09 CET on Studio 1. Script, sample ids, raw per-item JSONL, telemetry and summary:
`results/quality-20261122/`. *Standard* = inkling2-mlx v0.2.0 (MLX-VLM `0.8.0+inkling2mini.2`). *This repo* = the
`stack4` build (fork + 0001 + 0002 + 0003), no draft head. Same weights, same chat template (`reasoning_effort="low"`),
greedy, prefill step 2048, one request at a time.

| Benchmark | n | Standard build | This repo | Paired (std only / repo only) | Token-identical items |
|---|---:|---:|---:|---|---:|
| GSM8K (first 100) | 100 | 97 (97.0%, 95% CI 91.5–99.0) | 99 (99.0%, 94.6–99.8) | 0 / 2, McNemar p = 0.50 | 23 / 100 |
| MMLU-Pro (10 per subject) | 140 | 110 (78.6%, 71.1–84.6) | 109 (77.9%, 70.3–83.9) | 7 / 6, p = 1.00 | 11 / 140 |
| RULER NIAH 32k | 12 | 12 (100%) | 12 (100%) | 0 / 0 | 4 / 12 |

Neither difference is significant. Against the community fork (our base), this repo was token-for-token identical on
all 100 GSM8K items; tokens differ from the standard build because it is different code (greedy decoding drifts at the
first near-tie). MMLU-Pro truncations at 2,048 tokens: 11 standard / 8 repo, counted as wrong.

## Settings and environment variables

### Patches applied by `scripts/install.sh`

All patches are `patch -p1` from the fork's source root (`betizzel/mlx-vlm` at `55ca2e9e`).

| Patch | What | Exact? |
|---|---|---|
| `stacked.patch` | 0001 + 0002 + 0003 combined: memory-lean long context (last-token head, per-chunk embedding, one-shot KV reservation), one compiled decode step with fused MoE / attention / gate kernels, prefill attention tiles and a sort-free MoE permutation | bitwise |
| 0008 | Global attention tiles stage three key blocks per barrier (`MLX_VLM_RELATIVE_ATTN_KTILES=3`) | bitwise |
| 0009 | Vocabulary tables in host memory during long prefill (`MLX_VLM_EXACT_HOST_VOCAB=1`) | bitwise |
| 0010 | Fused prefill kernels: attention residuals, message gates, MoE route-combine, short conv, l2 clips (`MLX_VLM_INKLING2_PREFILL_FUSED=1`) | bitwise |
| 0011 | Compiled decode from 1,024 keys (`MLX_VLM_INKLING2_DECODE_MIN_KEYS=1024`) | same tokens on the gate; log-probs differ below 4,096 keys |
| 0007 | Draft head with an exact verifier, windowed draft context | same tokens as no draft |
| 0012 | Draft confidence cut, draft context limits | same tokens |
| 0014 | Server: one greedy request decodes through the compiled step | same tokens |
| 0015 | Server: one drafted request runs the exact-verify round loop | same tokens |
| 0016 | Draft proposes from a 16,384-id candidate list (`data/draft_vocab_order.json`) + the prompt's ids, compiled Markov loop | same tokens |
| 0017 | Server passes the prompt ids to the draft head | same tokens |
| 0018 | Server: last chunk and last token in one forward, lean draft prefill, candidate list built during prefill, message-gate casts inside the compiled tail | bitwise (casts); same tokens |
| 0019 | Prefill l2 clips as one Metal kernel per call | bitwise in its own tests |

Optional: `PREFIX=1` adds the prefix cache (0004 + 0006, or 0013 on top of the draft head; 0013 not measured end to
end). `NO_DRAFT=1` skips the draft-head patches. `DSPARK=1` builds the older 0005 compiled verify on
`stacked-v4.patch` (same tokens, not bitwise). Per-patch history: `docs/EXPERIMENTS.md`.

### Environment

Set by `scripts/env.sh` (the defaults):

| Variable | Value | Effect |
|---|---|---|
| `MLX_VLM_EXACT_PREFILL_EVAL_EVERY` | 1 | evaluate after every prefill layer visit (memory; scheduling only) |
| `MLX_MAX_MB_PER_BUFFER` | 2 | MLX command-buffer size (scheduling only) |
| `MLX_VLM_RELATIVE_ATTN_KTILES` | 3 | 0008 global attention tiles |
| `MLX_VLM_EXACT_HOST_VOCAB` | 1 | 0009 host vocabulary tables |
| `MLX_VLM_RELATIVE_ATTN_BQ_SWA` | 64 | sliding-window attention in 64-query blocks (default 32) |
| `MLX_VLM_INKLING2_PREFILL_FUSED` | 1 | 0010 fused prefill kernels |
| `MLX_VLM_INKLING2_DECODE_MIN_KEYS` | 1024 | 0011 (default 4096) |
| `MLX_VLM_DRAFT_MODEL`, `MLX_VLM_DRAFT_BLOCK_SIZE` | model dir, 8 | draft head on, block 8 |
| `MLX_VLM_INKLING2_INCREMENTAL_DRAFT_PREFILL` | 1 | draft head's view of the prompt captured per chunk |
| `MLX_VLM_INKLING2_EXACT_VERIFY`, `MLX_VLM_INKLING2_COMPILED_VERIFY` | 1, 1 | 0007 exact verifier |
| `MLX_VLM_INKLING2_DRAFT_CTX_WINDOW` | 65536 | draft head context window |
| `MLX_VLM_INKLING2_DRAFT_CONF` | 0.65 | 0012 confidence cut |
| `MLX_VLM_INKLING2_DRAFT_COMPILE`, `MLX_VLM_INKLING2_DRAFT_VOCAB` | 1, 16384 | 0016 compiled loop, candidate list size |
| `MLX_VLM_INKLING2_DRAFT_MIN_CTX`, `MLX_VLM_INKLING2_DRAFT_MAX_CTX` | 1024, 262144 | draft head off outside this context range |
| `MLX_VLM_BATCH_CACHE_EVAL_INTERVAL` | 0 | no cache clears inside an answer (server) |
| `MLX_VLM_SERVER_FOLD_LAST`, `MLX_VLM_INKLING2_DRAFT_LEAN_PREFILL` | 1, 1 | 0018 |

`NO_DRAFT=1` leaves out the draft and server-draft variables. `scripts/serve.sh` adds
`MLX_VLM_EXACT_PREFILL_EVAL_ASYNC=1` (async per-visit eval), `--prefill-step-size 8192` (`STEP` overrides),
`--max-num-seqs 1`, and sets MLX's memory, wired and buffer-cache limits to the recommended working-set size (the
server's default 1 GiB cache limit made prefill re-allocate its buffers on every visit).

On by default inside the patches (set to `0` to switch off): `PF_L2K` and `PF_L2KS` (0019), `PF_MGC` (0018),
`PF_AR`, `PF_MG`, `PF_MOE`, `PF_SC`, `PF_L2` (0010 parts, with `PREFILL_FUSED=1`), `MLX_VLM_INKLING2_SERVER_COMPILED_DECODE`
(0014), `MLX_VLM_INKLING2_SERVER_PLAIN_SPEC` (0015; `0` restores the batching path), `MLX_VLM_INKLING2_DRAFT_VOCAB_EARLY`
(0018), `MLX_VLM_SERVER_PREFILL_CLEAR_CACHE`.

Opt-in experiments, off, not used for any number in the README: `PF_MGF=1` (fused message-gate tail; bitwise, no
end-to-end gain), `PF_GU=1` (fused gate+up expert GEMM; no gain), `PF_POSTSCALE=1` (MLX-Serve's expert scale layout;
slower), `PF_ARX=0` and `PF_MGX=0` (older approximate kernels; not bitwise, `PF_ARX=0` failed the 99% top-1 gate),
`PF_PJ=1` (merged projections; not bitwise), `MLX_VLM_SERVER_STEP_LONG` (long-prompt step override; no gain),
`MLX_VLM_INKLING2_DRAFT_VOCAB_FILE` (another candidate list), `MLX_VLM_INKLING2_DRAFT_TIMERS` and
`MLX_VLM_INKLING2_SPEC_LOG` (logging), `APC_ENABLED=1` (prefix cache, needs `PREFIX=1`).

## What did not work

Short version of `docs/EXPERIMENTS.md`, which has every attempt. S1 / S2 = Studio 1 / Studio 2; server log, warm run.

**Reading (prefill)**

- Fused gate+up expert GEMM (`PF_GU`): 1,062.3 / 940.2 vs 1,063.2 / 941.7 tok/s (S1, 4k / 32k). The expert GEMM is
  compute-bound; a wider N doesn't help.
- Attention projections as one GEMM over concatenated weights (bitwise): 10.51 → 10.50 ms at 4,077 rows. Compute-bound;
  saves one launch.
- Merged q/k/v/r/gate projections (`PF_PJ`): bitwise at 2,048 rows only, teacher-forced top-1 98.5% across row counts.
  Dropped.
- The older approximate fusions (`PF_ARX=0`, `PF_MGX=0`): +0.7% / +0.5% on S2, but `PF_ARX=0` was at 98.5% top-1,
  below the 99% gate. Not used.
- MLX-Serve's expert layout (plain NVFP4 gathers, scale applied afterwards, `PF_POSTSCALE=1`): slower on mlx 0.32.4
  (1,062.6 / 938.2 vs 1,068.1 / 942.2, S1) and on mlx 0.32.4 + MLX PR #4709 built from source (973.2 / 863.3 vs
  976.5 / 865.7, S2).
- Expert GEMM reading rows in place via `lhs_indices` (`PF_LHS=1`): 4k 354.9, 32k 341.1. MLX picks a much slower
  gather kernel when `lhs_indices` isn't the identity.
- Fused message-gate tail (`PF_MGF=1`): bitwise, 0.46 → 0.37 ms per gate in isolation, no end-to-end gain (overlapped).
- MoE softcapped SwiGLU as one kernel: bitwise version 1.59–1.78 → 0.86–0.88 ms in the microbench, no server gain
  (1,126.3 / 1,001.1 vs 1,126.4 / 1,002.8, S1). Not kept.
- Value norm and casts through `rms_norm_cast` (`PF_VN`): 0.50 → 0.56 ms per visit. Not kept.
- Global-visit attention tile variants at 4k (BQ128, KT1/KT2, pipelined QK, padding): base 9.70 ms was best or level.
- Server prefill step 3072 / 2560 / 4096 with the 1 GiB cache limit: slower than 2048; a long-prompt-only step
  override: no gain. `EVAL_EVERY=0` or no `MLX_MAX_MB_PER_BUFFER=2`: slower. 8 MB command buffers, 200 ops per
  buffer: noise.

**Writing (decode and draft head)**

- Non-exact compiled verify (`EXACT_VERIFY=0`): slower (32k block 8: 46.0 / 47.1 vs 59.4 tok/s) and acceptance moved,
  so not exact. Dropped.
- First draft-head port without the exact verifier: slower than no draft at 131k–262k (26.2 / 23.9 vs 39.9 / 29.6).
- 0005 without `SPLITK_VERIFY=shared` at 131k: 24.2 tok/s, slower than no draft.
- Draft candidates as "ids below N": acceptance 88% → 71%, 62–64 tok/s (this tokenizer's common English words sit
  around ids 149k–156k). Replaced by the frequency-ordered list (0016).
- Draft context window 8k: acceptance drops, 56.9 tok/s. Pre-building the verify for the likeliest cut: 71.5 (no help).
- Adaptive block-size selector: settles on its largest candidate in ~125-token answers, adds nothing.
- `CONF_MIN=0` (allow verifying only the bonus token), τ 0.8: 32k 80.9 vs 81.8, τ 0.8 86.3 / 78.6 vs 87.4 / 81.8 (S1).
- betizzel 5a3c6132 router half (cached FP32 router weight): 952.5 vs 952.4 reading, no gain.

## Long context (Studio 1, command line)

![Reading and writing speed from 4k to 1M tokens of context](img/long-context.svg)

This repo without the draft head, one Mac Studio M4 Max 64 GB (Studio 1), bf16 KV, batch 1, greedy, needle
(`VIOLET-ANCHOR-4729`) inserted mid-document in public-domain Project Gutenberg text, prefill step 2048, 256 max new
tokens, `scripts/exact_check.py`. Needle found in every row. These runs predate patches 0010–0019.

| Context (prompt tokens) | Reading tok/s | Writing tok/s | Peak MLX GiB | Standard build: reading / writing / peak | Speed-up (reading / writing) | Exactness |
|---|---:|---:|---:|---|---|---|
| 4k (4,078) | 952.4 | 56.85 | 19.46 | 758.4 / 33.09 / 25.58 | 1.25× / 1.71× | bit-identical to the fork |
| 32k (32,750) | 867.3 | 53.73 | 20.34 | 667.9 / 27.0 / 28.27 | 1.29× / 1.99× | bit-identical to the fork |
| 131k (131,054) | 625.2 | 40.16 | 23.36 | 401.9 / 16.2 / 33.58 | 1.55× / 2.47× | bit-identical to the fork |
| 262k (262,126) | 453.9 | 29.76 | 27.35 | 256.4 / 10.7 / 35.38 | 1.77× / 2.78× | bit-identical to the fork |
| 524k (524,270), previous build | 292.6 | 17.86 | 35.35 | not measured | – | needle found |
| 1M (1,048,558), KT=3 + host vocab, text-only | 175.6 | 11.38 | 50.08 | not measured | – | needle found; tokens + sha equal to the 163.3 run |
| 1M, without 0008/0009 env | 163.3 | 11.36 | 50.76 | | | needle found |
| 1M, step 512, previous build | 156.0 | 10.04 | 50.04 | | | needle found |

Speed-ups are the measured ratios rounded down. *Standard build* = inkling2-mlx v0.2.0 (MLX-VLM `0.8.0+inkling2mini.2`)
with mlx 0.32.4 on the same Mac, weights and prompts (`results/runs/stock-v0.2.0-*.json`); it needed smaller prefill
steps (1024 / 512 / 256) to fit. There is no 524k or 1M standard-build run (an earlier 524k run on the unpatched fork
was stopped after 14 minutes, still reading).

Sources: 4k–262k `results/gates/gate-stack6-20261122-1108.log` and `results/gates/s6-*.json` (stack6, recommended env);
524k `results/gates/h-524k.json` (`stacked-v4.patch`); 1M `results/gates/h7-1m-kt3-hostvocab.json` and `.log`
(14:30–16:10 CET; env + `MLX_VLM_RELATIVE_ATTN_KTILES=3 MLX_VLM_EXACT_HOST_VOCAB=1`), `results/gates/h6-1m-step2048.json`
(163.3), `results/gates/h-1m.json` (156.0). The 1M run read 1,048,558 tokens in about 1 h 40 min at 175.6 tok/s, peak
50.08 of the 51.84 GiB GPU limit, vision tower dropped (`scripts/exact_text_only.py`, text logits unchanged). The 1M
step-2048 and step-512 runs wrote the same answer, but their tokens differ from position 9: the fork itself is not
chunk-size invariant, and there is no unpatched 1M reference at either step.

**Why 1M fits.** At 1M the weights (17.6 GiB) plus the reserved KV (32 GiB) sit at MLX's buffer-cache line
(0.95 × the recommended working set, 49.25 GiB); past it MLX's allocator frees its cache on every miss and every op
gets fresh Metal buffers (+20–30% per layer). Keeping the two 0.86 GiB vocabulary tables in host memory during the
chunked prefill and moving them back before the last token keeps the GPU under that line (`docs/EXPERIMENTS.md`,
14:11–14:30).

**Draft head at long context** (patch 0007, block 4, `scripts/exact_check_draft.py`,
`results/gates/gate-stack7-draft-20261122-1153.log`, `results/gates/s7-draft-k4-*.json`):

| Context | Reading tok/s | Writing tok/s (no draft → draft) | Accepted / drafted | Peak GiB | vs unpatched fork (no draft) |
|---|---:|---:|---:|---:|---|
| 32k | 864.9 | 53.73 → 65.99 | 77% | 21.41 | tokens and sha identical |
| 131k | 622.7 | 40.16 → 52.67 | 77% | 24.41 | tokens and sha identical |
| 262k | 452.2 | 29.76 → 35.72 | 59% | 28.41 | tokens and sha identical |

**Prefix cache** (fork server with APC + 0004, 131k document and a follow-up, client-side times,
`results/prefix-cache-131k-server.log`): turn 1 cold 219.5 s to first token, turn 2 (131,006 tokens restored) 1.54 s.
Not yet measured stacked with the speed patches.

### Roofline

Machine limits: 14.2 TFLOP/s fp16/bf16 FMA (measured on M4 Max by third parties; 16.2 on paper) and 546 GB/s DRAM
bandwidth (spec; 444–458 GB/s achievable in our read benchmarks).

Prefill: 8.84 GFLOP per prompt token for the dense work over the 48 layer visits, 0.50 GFLOP for the 40
sliding-window visits, plus 98,304 × S_avg FLOP for the 8 global visits (S_avg = N/2).

| Context | GFLOP/token | Ceiling tok/s | Measured | % of ceiling |
|---|---:|---:|---:|---:|
| 32k | 10.95 | 1,297 | 865.3 | 66.7% |
| 131k | 15.78 | 900 | 624.3 | 69.4% |
| 262k | 22.22 | 639 | 453.6 | 71.0% |
| 524k | 35.11 | 404 | 292.6 | 72.4% |
| 1M (KT=3 + host vocab) | 60.88 | 233 | 175.6 | 75.4% |

Decode: each token reads 5.50 GB of active weights (`results/decode-weight-bytes.json`) plus 0.17 GB of sliding KV and
32 KiB × S of global KV.

| Context | GB/token | Ceiling tok/s | Measured (no draft) | % of ceiling |
|---|---:|---:|---:|---:|
| 32k | 6.74 | 81.0 | 53.73 | 66% |
| 131k | 9.96 | 54.8 | 40.16 | 73% |
| 262k | 14.26 | 38.3 | 29.76 | 78% |
| 1M (KT=3 + host vocab) | 40.03 | 13.6 | 11.38 | 84% |

Details: `docs/trackA-prefill-notes.md`, `docs/trackB-decode-notes.md`, `docs/PREFILL_TECHNIQUES.md`.

## Not measured yet

- A second 131k session, and a 1M head-to-head vs MLX-Serve.
- The full quality gate on the exact final stack and the `serve.sh` settings (prefill step 8192).
- Why the first request at a new prompt length writes more slowly (on both servers).
- Concurrent requests: both servers were measured one request at a time.
- Prompts under 1,024 tokens: the draft head is off there, and chat-length writing takes a slower decode path.
- The draft head at 524k and 1M (it switches itself off above 262,144 tokens), and the prefix cache combined with the
  draft head (0013).
- 1M with patches 0010–0019.
- Any Mac other than the M4 Max Mac Studio 64 GB.

## Scripts

| Script | Purpose |
|---|---|
| `scripts/install.sh` | Fork zip → unpatched and patched source trees and a venv |
| `scripts/env.sh`, `scripts/serve.sh` | Default environment; OpenAI-compatible server with the benchmarked settings |
| `scripts/charts.py` | Charts (SVG for the docs, 1600×900 PNG renders `docs/img/x-*.png` for posts), README table and the generated tables here, from the logs |
| `scripts/gate.sh`, `scripts/exact_check.py`, `scripts/exact_check_draft.py` | Exactness and speed gate: token ids and log-probability sha, unpatched vs patched |
| `scripts/quality/top1.py` | 8-prompt greedy + teacher-forced top-1 gate |
| `scripts/probe_exact.py`, `scripts/memprof.py` | Synthetic-depth memory probe; per-chunk working-memory profiler |
| `scripts/exact_text_only.py` | Drops the vision tower for text-only runs (text logits unchanged) |
| `scripts/prefix_cache.sh`, `scripts/followup.py`, `scripts/test_inkling2_prefix_cache.py` | Prefix-cache server, two-turn measurement, tests |
| `scripts/draft_vocab.py` | Rebuilds `data/draft_vocab_order.json` (byte-identical) |

Paths default to `~/inkling-mini` (`ROOT`). Run one GPU job at a time. Some older run JSONs had local usernames in
paths replaced with `~` and machine-specific fields (process lists, network details) removed; numbers are unchanged.
