## STEER 01:46 (from supervisor) -- READ FIRST
Good: attention work is done (bq64+fastmask, 871 tok/s @32k exact). Stay on the NON-attention ~83% of 32k prefill: MoE/dense matmuls (gather_qmm NVFP4 expert GEMMs, MXFP8 attention projections, shared/dense MLP), expert batching/sorting, NVFP4 dequant cost, CPU/GPU overlap and graph build per chunk.
Measure per-op share at 32k step 2048 first (dense matmul TFLOP/s vs 14.2 peak), then attack the largest. Target: 32k e2e prefill >= 75% of the 1,297 tok/s ceiling (>= 973 tok/s).
Note: your best.patch is now stacked with Claude's (which also edits kernels.py and switch_layers.py); keep your diffs minimal and rebased on ../common/overlay-base.
Rules unchanged: exact (tokens + sha), no FP8 compute shortcuts/sparse/approx, no 'relaxed'/'fast' Metal math, GPU only via ../gpu-run.sh.

# Track A — exact prefill optimization notes

## SUPERVISOR STEER — 2026-11-22 01:35 CET (read this; it overrides earlier priorities)
Corrected prefill roofline (whole-prompt average, not the last chunk): global causal attention averages depth N/2, so
GF/token = 8.84 dense + 0.50 SWA + 98304*N/2/1e9 global = **10.95 at 32k (ceiling 1,297 tok/s)** and **15.79 at 131k (ceiling 899 tok/s)**.
Measured e2e: 849 tok/s at 32k = **65%**, 605 tok/s at 131k = **67%** of the 14.2 TFLOP/s roofline. (The "639 ceiling / 95%" figure used the depth-131k last chunk for every token — wrong.)
Time split implied by measurements: at 32k, dense/MoE/projections take ~83% of prefill time and run at only ~9.1 TFLOP/s (~64%); global attention runs at 72-77% and is <15% of 32k time. At 131k dense is ~60% of time (~8.8 TFLOP/s, 62%), global attention ~36% (kernel 76%).
- **Track A (Codex): PIVOT NOW** from attention tiles (they give 1-4%) to the NON-ATTENTION share of prefill at 32k: MoE SwitchGLU path (gather_qmm NVFP4 for T*8 routed rows: sort/unsort cost, expert grouping, tile shapes, dequant cost), MXFP8 dense projections (q/k/v/gate/o/r fused into fewer GEMMs where exact), router/softcap/conv/attention-residual glue kernels, Python graph build per chunk (~48 visits), CPU/GPU overlap between chunks (async_eval), command-buffer settings. Microbench one full layer visit and one 2048-token chunk at depth 0 (memprof-style, ~975 tok/s baseline = ~9.1 TFLOP/s on 9.34 GF/token); target ≥12 TFLOP/s. Exactness rules unchanged (bit-exact or report diffs).
- **Track B (Claude): wire the compiled decode step into real generation now** (generate_step decode loop / language-model call path, env switch default on), pass the bitwise check vs the eager path, then run e2e exact_check at 32k and at 131k (`PLAN='[{"ctx":131072,"step":2048,"max_tokens":256}]'`) via ../gpu-run.sh and compare against ../ref/check-32k-base.json and ../ref/check-131k-base.json (tokens + logprobs_sha256). Write NOTES.md (required by the brief) and best.patch.


## Roofline derivation

Machine ceilings from `../BRIEF.md`: 14.2 TFLOP/s bf16/fp16 FMA and 546 GB/s
specified DRAM bandwidth.  The checkpoint dimensions are `H=2304`, `Hq=24`,
`Hkv=8`, `D=128`, relative width `Dr=16`, `E=128` experts, top `K=8`, expert
intermediate `I=1152`, and loop-gate rank 64.  The 24 physical layers plus two
extra visits of layers 6--17 give 48 visits: 40 sliding and 8 global.  There are
24 revisits with four loop message gates each.

Counting one multiply-add as two FLOPs, dense work per ordinary layer visit is:

- Attention projections (Q, K, V, relative, output, sigmoid gate, and sink):
  `2*H*(Hq*D + 2*Hkv*D + Hq*Dr + 2*Hq*D + Hq)` =
  53,784,576 FLOP.
- Router: `2*H*E` = 589,824 FLOP.
- Routed expert gate, up, and down projections:
  `K*2*(H*I + H*I + I*H)` = 127,401,984 FLOP.
- Thus 48 visits cost `(53,784,576 + 589,824 + 127,401,984)*48` =
  8.725266432 GFLOP/token.
- Each message gate has a `2H -> rank` projection and two `rank -> H`
  projections, or `8*H*rank` FLOP.  Four gates on 24 revisits add
  `24*4*8*2304*64` = 0.113246208 GFLOP/token.
- The two dynamic K/V short-convolution weight projections on eight global
  visits add `8*2*2*H*(Hkv*4)` = 0.002359296 GFLOP/token.

Total non-attention-core dense work is therefore **8.840871936
GFLOP/token**, reproducing the contract's rounded 8.84.  The final vocabulary
head, evaluated for only the retained last row, is
`2*2304*201024` = **0.926318592 GFLOP/chunk**.

At steady state, each sliding query sees exactly 1024 keys.  Its 40 visits cost
`40*4*1024*Hq*D` = **0.503316480 GFLOP/token** for QK and PV.

For a global call whose `T` queries are the last `T` positions of a cache of
length `S`, the exact number of causally visible query/key pairs is

`P(T,S) = sum(i=0..T-1, S-T+i+1) = T*S - T*(T-1)/2`.

Useful attention work per call is `Fcall=4*Hq*D*P(T,S)`.  Across all eight
global visits, the per-token contribution is `8*Fcall/T`.  This excludes the
upper causal triangle even when a physical 8x8/16-key tile executes it.

| S | T | visible pairs/call | useful TFLOP/call | global GF/token | total GF/token incl. last-row head |
|---:|---:|---:|---:|---:|---:|
| 32,768 | 512 | 16,646,400 | 0.204551 | 3.196109 | 12.542106 |
| 32,768 | 1,024 | 33,030,656 | 0.405881 | 3.170943 | 12.516036 |
| 32,768 | 2,048 | 65,012,736 | 0.798876 | 3.120611 | 12.465252 |
| 131,072 | 512 | 66,978,048 | 0.823026 | 12.859785 | 22.205783 |
| 131,072 | 1,024 | 133,693,952 | 1.642831 | 12.834619 | 22.179712 |
| 131,072 | 2,048 | 266,339,328 | 3.272778 | 12.784288 | 22.128928 |
| 524,288 | 512 | 268,304,640 | 3.296927 | 51.514491 | 60.860489 |
| 524,288 | 1,024 | 536,347,136 | 6.590634 | 51.489325 | 60.834418 |
| 524,288 | 2,048 | 1,071,645,696 | 13.168382 | 51.438993 | 60.783634 |

For measured chunk time `t`, end-to-end achieved throughput and roofline share
are `T*total_GF/t` and `(T*total_GF/t)/14.2e3`, respectively.  The corresponding
ideal token rates span 1132--1139 tok/s at 32k, 639--642 tok/s at 131k, and
233.3--233.6 tok/s at 524k for these chunk sizes.

## Experiments

- Baseline smoke, S=4096/T=64: 0.9385 ms median, 3.4058 useful TFLOP/s,
  23.98% of compute roof.  This deliberately tiny shape is launch/occupancy
  dominated and is not part of the target matrix.
- Baseline short timing, S=32768/T=512: 20.7627 ms, 9.8518 TFLOP/s, 69.38%.
- Baseline 10 s timing, S=32768/T=512: **20.7341 ms, 9.8654 TFLOP/s,
  69.47%** (480 measured calls).  Use this 10 s result in comparisons.
- Baseline S=32768/T=1024: 39.3606 ms, 10.3118 TFLOP/s, 72.62%.
- Baseline S=32768/T=2048: 74.3306 ms, 10.7476 TFLOP/s, 75.69%.
- Baseline S=131072/T=512: 85.1644 ms, 9.6640 TFLOP/s, 68.06%.
- Baseline S=131072/T=1024: 164.4030 ms, 9.9927 TFLOP/s, 70.37%.
- Baseline S=131072/T=2048: 316.2836 ms, 10.3476 TFLOP/s, 72.87%.
- Baseline S=524288/T=512: 341.1444 ms, 9.6643 TFLOP/s, 68.06%.
- Baseline S=524288/T=1024: 657.7935 ms, 10.0193 TFLOP/s, 70.56%.
- Baseline S=524288/T=2048: 1281.9481 ms, 10.2722 TFLOP/s, 72.34%.

Current candidate: group all three query heads that share a KV head into one
threadgroup, with BQ selectable from 8/16/24/32.  Each head retains the base
kernel's 8-row simdgroup and identical MMA/softmax instruction sequence; only
the K/V staging is shared.  BQ=16 has six simdgroups (192 threads) and reduces
K/V tile loads by 1/3 versus base.  BQ=24 and 32 reduce them by 5/9 and 2/3 but
use 288 and 384 threads, respectively, so they need occupancy measurement.

- Grouped-GQA BQ=16/BK=16 is bitwise equal on randomized global and sliding
  cases, but reaches only 8.8098 TFLOP/s (62.04%) at S=32768/T=512.  BQ=8 is
  8.9336, BQ=24 is 7.8631, and BQ=32 is 7.4342 TFLOP/s.  The larger
  threadgroups lose more occupancy than sharing K/V saves; discard grouping.
- Single-head query-block sweep at S=32768/T=512 (BK=16): BQ 16/32/48/64/96/128
  gives 8.3501/9.8606/8.2469/**10.1820**/7.3977/7.8789 TFLOP/s.  BQ=64
  is +3.2% versus the 10 s base and remains bitwise equal on a randomized
  non-aligned case.  Its 256 threads also exactly match the 256 eight-element
  K/V staging chunks per BK=16 tile.
- BK=32 at BQ=32/64 gives 9.3536/9.8547 TFLOP/s; BK=8 gives
  9.4232/9.0678.  Both changed 2--3% of randomized BF16 elements by at most
  0.0001220703125 (one small BF16 ULP here) because the online-softmax update
  interval changed.  They are slower as well, so retain exact BK=16.
- Corrected fast-mask + BQ=64/BK=16/PAD=8 is bitwise equal on randomized
  global and sliding tails.  Global matrix (TFLOP/s, percent of 14.2):

  | S | T=512 | T=1024 | T=2048 |
  |---:|---:|---:|---:|
  | 32,768 | 10.2873, 72.45% | 10.7236, 75.52% | 10.9170, 76.88% |
  | 131,072 | 10.4023, 73.26% | 10.6780, 75.20% | 10.8674, 76.53% |
  | 524,288 | 10.3508, 72.89% | 10.6565, 75.05% | 10.8344, 76.30% |

  Relative to the matching baseline points, gains are 4.3/4.0/1.6% at 32k,
  7.6/6.9/5.0% at 131k, and 7.1/6.4/5.5% at 524k.  The fast mask alone is
  about +0.9% at 32k/T=512; its benefit grows with far-tile count.  PAD 0/4/8/16
  produced 10.0302/10.2583/10.2851/9.6407 TFLOP/s, confirming PAD=8.
- Sliding steady-state calls prefer the base BQ=32 and general mask.  At
  T=512/1024/2048 base gives 6.9585/8.2220/9.2581 TFLOP/s, while BQ=64 gives
  6.6191/7.6995/8.4885.  BQ=32 with the fast path also regresses to 6.8021 at
  T=512.  Final dispatch therefore uses BQ=64 + fast mask only for global, and
  BQ=32 + original mask for sliding.
- 32k e2e (`step=2048`, 32,750 prompt tokens): **870.570 prefill tok/s**,
  32.895 decode tok/s, peak 22.782 GiB.  This is +2.96% versus the checked-in
  reference run's 845.533 tok/s.  All 126 token IDs and the full logprob SHA256
  match exactly (`c4176a...01a1e`).  Exact prompt work is 358.485852 TFLOP;
  achieved prefill is **9.5294 TFLOP/s = 67.11%** of compute roof.
- 131k e2e (`step=2048`, 131,054 prompt tokens): **627.365 prefill tok/s**,
  27.742 decode tok/s, peak 25.783 GiB.  This is +3.67% versus the 605.133
  tok/s unpatched reference and +1.77% versus the 616.459 tok/s exact-memory
  reference at the same step.  All token IDs and the full logprob SHA256 match
  both references exactly (`7aeec7...004e2`).  Exact prompt work is
  2068.536208 TFLOP; achieved prefill is **9.9022 TFLOP/s = 69.73%** of the
  compute roof.
- Measured BF16 copy-like DRAM bandwidth (512 MiB input plus 512 MiB output):
  **433.36 GB/s**, 79.37% of the 546 GB/s specification (2.4777 ms median).
- MoE prefill route-sort inversion at 2048 tokens/16,384 routes: the original
  second argsort takes 0.158917 ms median; collision-free permutation scatter
  takes 0.112917 ms, **29.0% faster**, with exactly equal inverse indices.  The
  scatter is now used by `_gather_sort`; its absolute 48-visit chunk saving is
  small but consistent.
- Metal `relaxed` and `fast` math modes each changed 14/420,864 randomized
  BF16 outputs by at most 0.00012207 for only about 0.1% speed.  Retain `safe`.
- Final patch-of-record 32k e2e after the mask branch hoist, padding-free caller
  specialization, and MoE scatter: **871.087 prefill tok/s**, 33.482 decode
  tok/s, 22.782 GiB peak.  This is +3.02% over reference; token IDs and logprob
  SHA remain exactly equal.  Achieved prefill is **9.5350 TFLOP/s = 67.15%**.
- Chunk scheduling check at 32k: step 4096 reaches 875.749 tok/s, only +0.54%
  over step 2048, while peak rises from 22.782 to 26.209 GiB and the logprob
  SHA changes (all 126 greedy token IDs still match).  Retain step 2048 as the
  exact patch-of-record schedule; the marginal throughput is not worth the
  extra working set or loss of bitwise logprob parity.
