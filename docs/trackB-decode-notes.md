## STEER 01:46 (from supervisor) -- READ FIRST
Verified by supervisor under gpu-run.sh: your compiled decode (ovB2 snapshot) is bit-exact at 32k AND 131k (tokens + logprob sha equal); decode 32k 33.46->40.16 tok/s, 131k 27.13->31.29. 131k e2e is DONE, do not re-run it.
Next target: the ~17 ms/token CPU graph build + encode. Goal: decode >= 65% of the 546 GB/s bandwidth floor
(32k floor 12.34 ms -> need <= 19.0 ms/token = >= 52.7 tok/s; 131k floor 18.2 ms -> <= 28.0 ms = >= 35.7 tok/s).
Ideas: keep ONE compiled graph across tokens with zero retrace (shapes fixed: offsets/ring index/length as arrays -- verify no recompile per token via timing/trace counts);
pipeline: async_eval token t+1 graph while t runs (build next step before syncing .item()); avoid any per-step Python-side mx.array creation, dict/list rebuilds, or host syncs;
cut node count (5536 kernels: 1067 AsType, 384 Concatenate, 118x4 decomposed softmax -> fuse into custom Metal kernels or mx.fast ops where bitwise identical);
consider an mx.compile'd multi-layer block, or capturing the whole 48-visit step in fewer custom kernels.
Rules unchanged: bitwise-exact greedy (tokens + logprob sha vs ../ref), no FP8/sparse/approx, no sudo/sysctl/push, stay in your dir, GPU only via ../gpu-run.sh.
Append results to LEADERBOARD.tsv and keep best.patch (diff vs ../common/overlay-base) current.

# Track B (decode + memory) — NOTES

All numbers measured on the M4 Max (40-core GPU) in this session unless marked "derived".
GPU runs all via ../gpu-run.sh. Python: venv-exact (mlx 0.32.4), overlay = ./overlay.

## 1. Roofline (derived from the in-memory parameter census, census.py -> census.json)

Correction to the brief: the attention projections are **bf16** in memory (hf_quant_config "cast"
group). MXFP8 is used only for the **experts of layers 20-23**; experts of layers 0-19 are NVFP4
(group 16, fp8 scales, + per-expert fp32 global scale). Census total language params 17.59 GiB (matches).

Bytes read per decode token (48 layer visits: layers 0-5 and 18-23 once, loop core 6-17 three
times; 24 of the loop visits carry message gates):

| part | MB / token | how |
|---|---:|---|
| attention projections q,k,v,r,o,gate,sink (bf16) | 2581.7 | 48 x 53.78 MB |
| experts, 8 of 128 per visit | 1839.4 | 44 NVFP4 visits x 8 x 4.479 MB + 4 MXFP8 visits x 8 x 8.211 MB (weights+scales) |
| message gates (loop passes 1,2) | 114.1 | 24 x 4 gates x (64x4608 + 2x2304x64) bf16 + queries |
| router, norms, convs, rel, attn-res queries | 36.7 | |
| head (bf16, full vocab 201024 x 2304) | 926.3 | read once per token |
| embedding row | 0.01 | |
| **weights total** | **5498.2** | |
| sliding KV (40 visits x 1024 keys x 8 heads x 128 x K,V x bf16) | 167.8 | once S >= 1024 |
| global KV (8 visits x S x 32 KiB) | S x 0.032768 | |

Bandwidth: spec 546 GB/s. Measured achievable: big bf16 GEMV (the head, 0.93 GB) 458 GB/s;
read-only row reduction 444 GB/s; copy 370 GB/s (read+write counted); NVFP4/MXFP8 8-route
gather_qmm at real shapes 437 / 452 GB/s; split-K decode attention (marginal, 32k->131k) ~445 GB/s.

| depth S | bytes/token | ms @546 | ceiling tok/s @546 | ceiling @458 (measured) |
|---:|---:|---:|---:|---:|
| 0 | 5.498 GB | 10.07 | 99.3 | 83.3 |
| 32k | 6.740 GB | 12.34 | 81.0 | 68.0 |
| 131k | 9.961 GB | 18.24 | 54.8 | 46.0 |
| 262k | 14.256 GB | 26.11 | 38.3 | 32.1 |
| 524k | 22.846 GB | 41.84 | 23.9 | 20.0 |
| 1M | 40.026 GB | 73.31 | 13.6 | 11.4 |

Baseline vs ceiling @546: 34.7 tok/s at 32k e2e = 43%; 23.0 at 262k = 60%; 6.5 at 1M = 48%.

## 2. Where decode time goes (baseline, synthetic decode bench dbench.py)

dbench.py: synthetic caches at depth S (random bf16 global K/V, full 1024-key sliding rings,
random conv states), decode loop identical to generate_step (async_eval pipelining, greedy).

Baseline eager decode (ms/token): S=1024 27.57 (36.3 tok/s, 38% of roofline), S=32k 29.78
(33.6 tok/s, 41%), S=131k 35.90 (27.9 tok/s, 51%).
- one decode step = **12,951 graph nodes**, ~7,000 real kernels (1609 AsType, 976 Matmul, 514 RMSNorm, ...).
- CPU per step: Python graph build 4.9 ms + C++ encode (async_eval) ~15-17 ms CPU.
- dependent tiny-kernel latency on the GPU: 2.4 us.
- single command buffer per step (MLX_MAX_OPS_PER_BUFFER=20000): eval wall - encode CPU => ~25 ms of GPU time per step at 32k.
- MLX_MAX_OPS_PER_BUFFER=1000 + MB=1000 alone: 32k 29.6 -> 26.3 ms (+12.6%) on the eager path; only ~0.2 ms once compiled (GPU-bound then).
- kernel efficiency (kbench.py, DRAM-resident weights): q/o GEMV 3072x2304 31.5 us (450 GB/s) independent / 36.8 us chained;
  k 1024x2304 13 / 20 us; r 384x2304 7.6 / 15 us; sink 24x2304 4.6 / 8.8 us (12-24 GB/s); concatenated 8576x2304 86 / 93 us (459 / 426 GB/s).

## 3. Experiments (all exactness checks are bitwise unless noted)

| # | what | result | % roofline @546 | exact |
|---|---|---|---|---|
| 0 | baseline eager (synthetic) | 32k 29.78 ms, 131k 35.90 ms | 41.4 / 50.8 | ref |
| 1 | **whole decode step as ONE mx.compile graph** (fast_decode.py + step_cache.py: functional cache views, offsets/ring index as arrays, global split-K kernel reads the reserved buffer with an array length; KV written in place by slice_update with array start, buffers donated) | 32k 25.87 ms, 131k 33.09 ms; Python build 4.9 -> 1.9 ms | 47.7 / 55.1 | bitwise (logprobs max|d|=0, 4 steps x 2 depths) |
| 2 | + prepare_decode: fp32 router/loop-gate weights stored once, NVFP4 kernel factor stored once, constant gather_qmm lhs indices + aranges, same-input projections [q;r;gate;k;v] as one GEMV (originals are row views: memory neutral) | 32k 24.86 ms (40.2 tok/s), 131k 32.10 ms (31.2 tok/s) | 49.7 / 56.8 | bitwise |
| 2b | same without catproj | 32k 25.00, 131k 32.23 | | bitwise |
| 2c | same + MLX_MAX_OPS_PER_BUFFER=1000/MB=1000 | 32k 24.67, 131k 32.15 | | bitwise |
| E1 | **e2e 32k exact check** (generate_step uses the compiled step for greedy decode) | decode 40.15 tok/s (ref run 33.46), prefill 851 tok/s, peak 26.06 GiB (ref 26.19) | 49.6 | **126/126 tokens, logprobs sha256 identical** |

Compiled graph after #2: 9551 nodes, 5536 non-view kernels per step (1067 AsType, 972 Matmul, 449 RMSNorm,
384 Concatenate, 358 Sum, 193 custom, 144 GatherQMM, 118x4 decomposed softmax (axis=-2) ...).

| 3 | + q/k/v message gates of a loop visit as ONE batched chain (stacked weights, views: memory neutral) | 32k 24.93, 131k 32.19: GPU time unchanged, C++ encode CPU 12.3 -> 11.2 ms | 49.5 / 56.7 | bitwise |

### Component costs (timing-only ablations in the compiled step, S=32k, 24.82 ms total)
decode attention 5.31 ms (8 global split-K ~2.4-2.9 + 40 sliding ~2.9), experts 5.09, same-input projections 4.60,
attention-residual mixing 2.70, router 0.88, short convs 0.69, l2 clips 0.39, head 0.42 (*).
Message gates: ablation said 12.2 ms but that is an artefact (without messages pass p no longer depends on
pass p-1 and compile CSE merges/overlaps the passes); isolated cost measured 36 us x 96 calls = 3.4 ms.
(*) the head GEMV alone is 1.86 ms (497 GB/s, hbench.py); ablating it saved only 0.42 ms => the step has
GPU idle time, i.e. it is not purely GPU-bound.
Isolated kernels (hbench.py): sliding SDPA (1024 keys, mask+sinks) 22-29 us/visit (KV re-read per query head);
global split-K at 32k 368 us/visit (364 GB/s).

### Free-running test (fixed input token, CPU allowed to queue 3 steps ahead) => pure GPU time per step
| config | token-dependent pipeline (as generate_step) | free-running |
|---|---|---|
| default MLX env, 32k | 25.62 ms | 25.62 ms |
| default, 131k | 32.34 | 32.38 |
| MLX_MAX_OPS_PER_BUFFER=1000 MLX_MAX_MB_PER_BUFFER=1000, 32k | 24.79 | **19.76** |
| same, 131k | 32.56 | **26.80** |
=> with bigger command buffers the GPU needs only ~19.8 ms/token at 32k (26.8 at 131k): already inside the
65%-of-roofline targets (19.0 / 28.0). ~5-6 ms/token is lost to CPU<->GPU coupling: async_eval of step t+1
blocks (21.8 ms wall vs 10.4 ms CPU) until step t is nearly done, so the GPU starves while t+1 finishes encoding.
