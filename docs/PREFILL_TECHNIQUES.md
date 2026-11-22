# Inkling-2-Mini long-prompt prefill: techniques, hardware floor, and what to try next

Research memo, written 2026-11-22 ~00:40 CET (Europe/Prague), before the patch work. Its 'fork' figures are the unpatched betizzel fork; current measured numbers are in the README.
Scope: one M4 Max 64 GB Mac Studio (Studio 1) running betizzel/mlx-vlm with the fused relative-attention kernel.
Labels: **measured** = our own runs on Studio 1. **paper / author** = third-party claim, not reproduced. **estimate** = my arithmetic. Each one is marked.

## 0. TL;DR

- **Yes, "25–40 min for a 524k prompt" is what's happening.** It's an estimate: the fork's 524k run hadn't finished when this was written. The fork has held **about 10 TFLOP/s effective** from 32k through 262k. At that rate 524k is **about 31 min**.
- **The hardware floor is about 19–22 min.** That's how long the arithmetic takes at 100% of the M4 Max GPU peak. So the fork is already within roughly **1.4–1.6× of the hardware limit**, and no exact kernel trick can make 524k "massively" faster on one M4 Max.
- **Only three kinds of change are big:** (a) **skip work** with block-sparse prefill on the 8 global attention visits; (b) **add compute** with a second Studio over Thunderbolt RDMA, or prefill on a GPU; (c) **don't redo work**, by persisting the prefix KV so follow-up questions are seconds.
- **Ranked next tries:**
  1. Block-sparse global attention: measure sparsity first, then build the kernel.
  2. Two Studios with tensor parallelism (TP=2), using the 2-way shard already in the stack. Exact, and makes 1M fit.
  3. Persistent prefix KV cache plus FP8 global KV.
  - Quick extra win: skip the full-chunk logits computation (§4.1).

## 1. Where the time goes, from first principles

### 1.1 Assumptions (all from `model/config.json` and `mlx_vlm/models/inkling2/language.py` in this pack)

- hidden 2304, 24 Q heads, 8 KV heads, head_dim 128, d_rel 16, MoE with 128 experts, top-8, expert intermediate 1152, vocab 201,024.
- **Loop:** `core_start=6, core_end=18, num_passes=3`. *Note: the brief said 2 loop passes; the config says 3.* Visits: layers 0–5, then 6–17 ×3, then 18–23 = **48 layer visits**.
  - 8 are global visits: layer 5 ×1, 11 ×3, 17 ×3, 23 ×1.
  - 40 are SWA-1024 visits.
  - Either way, 8 global caches, as the brief said.
- **Active matmul parameters per visit:**
  - Attention projections: q, gate, o (3 × 2304×3072), k, v (2 × 2304×1024), r (2304×384), sink (2304×24) = **26.9 M**.
  - MoE: 8 × 3 × 2304×1152 + router = **64.0 M**.
  - Plus loop MessageControllers (24 visits × 2.4 M).
  - Total **4.42 B active params/token → 8.84 GFLOP/token** (2 FLOP per MAC).
- **SWA attention:** 40 visits × 4 × 1024 × 128 × 24 = **0.50 GFLOP/token** (constant in L).
- **Global attention, causal:** 8 caches × 4·(L²/2)·H·d, with H = 24 query heads, d = 128.
  - This assumes the kernel skips fully masked causal tiles. Chunked prefill does that by construction, because each chunk only attends to the keys before it.
- The LM head (0.93 GFLOP/token) is excluded from the floor. See §4.1: stock v0.2.0 computes it for every token in each chunk anyway.
- **M4 Max 40-core GPU peak:**
  - Theoretical FP32 is 5,120 ALUs × 2 × 1.578 GHz = **16.2 TFLOP/s** ([nanoreview](https://nanoreview.net/en/gpu/apple-m4-max-gpu-40-core); [eatyourbytes](https://www.eatyourbytes.com/applecpu-detail/apple-m4-max-40-core-gpu/) says 16.18).
  - Measured FMA peak is **14.2 TFLOP/s**, and "fp16 inputs run at the f32 rate" ([vdeviatkov/hpc-math-core](https://github.com/vdeviatkov/hpc-math-core)). Best measured fp16 GEMM there: 12.7–13.7 TFLOP/s (MPS).
  - Another measurement: Metal `matmul2d` fp16 tops out at **14.8 TFLOP/s** on M4 Max, and fp8 is *not* faster (0.94× fp16) because M4 has no matrix unit ([ramchandk.com](https://ramchandk.com/blog/rigel-apple-m4-metal-tensor)).
  - Apple doesn't publish an official GPU TFLOPS figure. Some spec sites list 17.0 FP32 / 34 "FP16", but the measured sources above don't support a 2× fp16 rate on M4.

### 1.2 Result (estimate; script logic reproduced in §7)

| Prompt | Dense + SWA PFLOP | Global attn PFLOP | Attn share | Floor @16.2 TF | Floor @14.2 TF (measured peak) | Measured (fork) | Effective TF (fork) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 32k | 0.31 | 0.05 | 15% | 0.4 min | 0.4 min | 38 s | 9.5 (67% of 14.2) |
| 131k | 1.22 | 0.84 | 41% | 2.1 min | 2.4 min | 208 s | 9.9 (70%) |
| 262k | 2.45 | 3.38 | 58% | 6.0 min | 6.8 min | 580 s | 10.1 (71%) |
| **524k** | **4.90** | **13.51** | **73%** | **18.9 min** | **21.6 min** | running | at 10 TF: **~31 min** (estimate) |
| 1M | 9.80 | 54.04 | 85% | 65.7 min | 74.9 min | not run | at 10 TF: **~106 min** (estimate) |

The stock v0.2.0 path, for comparison: 7.3 / 6.3 / 5.7 / **4.5** effective TF at 32k / 131k / 262k / 524k. That's why stock 524k took 67.5 min.

**What this means:**
- The fork is a good kernel. It holds ~70% of the measured FMA peak, flat across context lengths.
- The best any exact kernel could do at 524k is ~1.4× more, which would bring it to ~21 min. Expect less, because softmax exp and the rel-bias don't run on FMA units.
- Attention is 73% of the work at 524k and 85% at 1M. **Skipping attention work is the only single-Mac lever with a large payoff:**

| Global-attention blocks kept | 524k @ 10 TF (estimate) | 1M @ 10 TF (estimate) |
|---:|---:|---:|
| 100% (dense) | 30.7 min | 106 min |
| 30% | 14.9 min | — |
| 20% | 12.7 min | 34 min |
| 10% | 10.4 min | 25 min |
| 5% | 9.3 min | — |

The dense MoE and projection part, 4.9 PFLOP ≈ **8 min at 10 TF**, is the floor that sparse attention can't touch at 524k.

## 2. Technique-by-technique

The Inkling-specific facts that matter below, read from the code:
- Global layers use **no RoPE**. They use a relative bias with extent 1024 (with clipped distance, so beyond 1024 the bias is a per-query constant), a log-position query scale `tau`, L2-clipped q and k, and a **query-dependent sink logit**.
- So outside the 1024-token diagonal band, a global score is just `q·k` plus a per-query constant. That makes block-sparse kernels *easier* to port than for RoPE models, because the band, where the rel-bias matters, is always kept dense.

| # | Technique | Measured speedup (source) | Quality impact (source) | Applies to Inkling? | Available in MLX? | Effort (estimate) |
|---|---|---|---|---|---|---|
| A1 | **MInference** (vertical-slash / block-sparse, per-head offline pattern) | Prefill 1.8× @100k, 4.1× @300k, **6.8× @500k**, 10× @1M on A100, Llama-3-8B-1M ([arXiv 2407.02490](https://arxiv.org/abs/2407.02490)) | NIAH all-green; matches full attention on RULER / InfiniteBench in the paper. FlexPrefill's table shows MInference dropping to 58 vs 78 at 128k on Llama ([2502.20766](https://arxiv.org/abs/2502.20766)) | Yes, on the 8 global visits only. Needs an offline per-head pattern search on Inkling. | No. CUDA/Triton ([microsoft/MInference](https://github.com/microsoft/MInference)) | High |
| A2 | **FlexPrefill** (query-aware, cumulative-attention budget γ) | Attention **3.49×** (γ=0.9) / 2.43× (γ=0.95) @128k, A100 ([ICLR'25 paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/03645743ea35690f30d795d6bac149a5-Paper-Conference.pdf)). One GitHub issue reports no gain in their setup ([#12](https://github.com/bytedance/FlexPrefill/issues/12)) | RULER avg 89.2 vs 88.7 full (Llama-3.1-8B, 4k–128k) | Yes (adaptive per head, which suits unknown patterns) | No (Triton) | High |
| A3 | **XAttention** (antidiagonal block scoring) | Attention up to **13.5× @256k** vs FlashInfer. Block selection 24.9× cheaper than MInference, 5.9× cheaper than FlexPrefill ([MIT Han lab](https://hanlab.mit.edu/projects/xattention), [arXiv 2503.16428](https://arxiv.org/abs/2503.16428)) | "Comparable to full attention" on RULER / LongBench / VideoMME | Yes. Cheap scorer, good fit for a Metal port | No (Triton / FlashInfer) | Medium–high |
| A4 | **FlashPrefill V2 on MLX** (pooled-key block selection plus mean correction for pruned blocks) | **On Apple Silicon:** M4 Pro, Qwen3-4B. Attention 2.9× @16k, 4.3× @32k, 5.6× @64k. **End-to-end prefill 1.36× @16k, 1.78× @32k, 2.58× @64k** (64k: 347 s → 134 s), at 73–89% block sparsity ([Sevren blog](https://sevren.ai/blog/flashprefillv2-mlx), code [sevren-ai/FlashPrefillv2-MLX](https://github.com/sevren-ai)) | Paper reports accuracy; Sevren measured speed only (4 NIAH prompts) | **Yes. The closest existing code to what we need.** Must add the rel-bias band, per-query far-field constant, query sink, and `tau` | **Yes. Custom Metal kernel, MLX 0.32.2** | Medium (2–4 days to adapt into betizzel's kernel, estimate) |
| A5 | **SpargeAttn** (training-free, any model) | Llama-3.1 @128k end-to-end 52 s → 29.98 s (~1.73×) ([arXiv 2502.18137](https://arxiv.org/html/2502.18137), [thu-ml/SpargeAttn](https://github.com/thu-ml/spargeattn)) | Near-lossless claimed | Yes | No (CUDA) | High |
| A6 | **SampleAttention** | TTFT vs FA2: 1.62× (α=.95) / 2.28× (α=.80) @96k. 2.27× / 4.62× @1M. MLSys'25 says up to 5.29× ([arXiv 2406.15486](https://arxiv.org/abs/2406.15486v2), [MLSys](https://proceedings.mlsys.org/paper_files/paper/2025/hash/2d04d97593c8c33d415337f408ed0e1b-Abstract-Conference.html)) | "Near-lossless" at α=.95 | Yes | No | High |
| A7 | **Star Attention** (blockwise-local plus anchor on hosts, then global for the query) | Up to **11×** vs Ring Attention, retaining 97–100% accuracy ([arXiv 2411.17116](https://arxiv.org/html/2411.17116v2), [NVIDIA/Star-Attention](https://github.com/nvidia/Star-Attention)) | 97–100% of baseline accuracy | Partly. Also works on **one** device as an approximation (context blocks attend only to an anchor plus themselves) | No | Medium–high |
| A8 | **APB** (sequence-parallel plus passing compressed blocks between hosts) | Prefill up to **9.2×** vs FlashAttention, 4.2× vs Ring, 1.6× vs Star ([ACL'25](https://aclanthology.org/2025.acl-long.525/), [thunlp/APB](https://github.com/thunlp/apb)) | "No observable degradation" (their tasks) | Multi-device only. Would need 2+ Studios | No | High |
| B1 | **Chunked-prefill size tuning** | Ours (measured): stock had to shrink the step to 128 at 524k (dense `[H,step,S]` mask). The fork runs 4096–8192 | Exact | Already applied by the fork. Remaining headroom is bounded at ~1.4× by §1 | Yes (`prefill_step_size`) | Trivial. Expect ≤5–10% (estimate) |
| B2 | **Flash-attention Metal kernels** (mlx `fast.scaled_dot_product_attention`, philipturner's MFA, mlx-mfa) | MFA forward reaches **85–86% ALU utilization** on M1 Max at D=128 ([metal-flash-attention](https://github.com/philipturner/metal-flash-attention)). Stock mlx SDPA with a materialized `[H,LQ,S]` mask was our 4.5–7.3 TF baseline; the fused rel-attention kernel ([mlx-lm#2086](https://github.com/ml-explore/mlx-lm/pull/2086) → betizzel) reaches ~10 TF | Exact | **Done.** This is the fork. Squeezing 70% → 85% of FMA peak would be ~1.2× (estimate) | mlx SDPA; mlx-mfa ([PyPI](https://pypi.org/project/mlx-mfa/2.60.1/)) has a block-sparse `flash_attention_sparse` (STEEL kernel that skips masked tiles on M1–M4) but no rel-bias or sink | Medium for +~20% |
| C1 | **FP8 KV** (E4M3) | razorback16 inkling-infer runs FP8 global KV by default. 262k: KVCache 8.0 GiB, peak MLX 28.5 GiB (author, M3 Ultra) ([PRIOR_ART.md](PRIOR_ART.md)). oMLX #5511: 98k KV 3.56 → 1.78 GB | Small. razorback16 ships exactness guards against bf16; omlx claims cosine-level parity. **Not measured on Mini** | Yes. Halves the 8 global caches: 1M = 16 GiB instead of 32 GiB. **Does not speed up prefill on M4** (fp8 compute is 0.94× fp16, [ramchandk](https://ramchandk.com/blog/rigel-apple-m4-metal-tensor)). Helps decode bandwidth | razorback16 (built on mlx-lm#2086); mlx_vlm has `KV_BITS` / TurboQuant / Hybrid quant code paths, **but Inkling's attention expects dense k/v (concat plus SDPA), so they likely don't apply as-is (unverified)** | Low–medium |
| C2 | **KIVI** (2-bit asymmetric K per-channel / V per-token) | 2.6× less peak memory, up to 4× batch, 2.35–3.47× throughput ([ICML'24](https://proceedings.mlr.press/v235/liu24bz.html)) | Small on LongBench (paper) | Memory and decode only. No prefill gain | No Inkling path | Medium |
| C3 | **KVQuant** (per-channel pre-RoPE K, non-uniform, outlier-aware) | 3-bit with <0.1 PPL loss. LLaMA-7B **1M context on one A100-80GB**, 10M on 8 GPUs ([arXiv 2401.18079](https://arxiv.org/abs/2401.18079)) | <0.1 PPL at 3-bit | Memory only. "Pre-RoPE" is moot because Inkling has no RoPE | No | High |
| D1 | **SnapKV** | 3.6× generation speed, 8.2× memory @16k. 380k on one A100-80GB ([arXiv 2404.14469](https://arxiv.org/abs/2404.14469)) | Small on LongBench | **No prefill saving.** It compresses *after* prefill. Would cut decode KV for 1M | No | Medium |
| D2 | **PyramidKV** | Matches full KV on LongBench keeping ~12% of KV ([arXiv 2406.02069](https://arxiv.org/abs/2406.02069)) | Lossy at low budgets | Same as SnapKV: decode-side only | No | Medium |
| D3 | **StreamingLLM** (sinks plus recent window) | Up to 22.2× vs sliding-window recompute, stable to 4M tokens ([arXiv 2309.17453](https://arxiv.org/abs/2309.17453)) | **Destroys long-range retrieval:** RULER 128k ≈ 31 vs 78 full on Llama ([FlexPrefill table](https://arxiv.org/abs/2502.20766)) | **No.** Inkling already *is* this on 40 of 48 visits (SWA-1024 plus sink). Applying it to the 8 global visits removes exactly what makes needle retrieval work | — | Don't |
| E1 | **Prefix KV cache to disk** (LMCache; mlx-lm `cache_prompt`) | LMCache doc example: 15k tokens 6.31 s cold → 0.148 s warm ([LMCache docs](https://docs.lmcache.ai/kv_cache/local_storage.html)). mlx-lm: `mlx_lm.cache_prompt … --prompt-cache-file x.safetensors`, reused by `mlx_lm.generate` ([mlx-lm README](https://github.com/ml-explore/mlx-lm)). betizzel claims a ~2 s follow-up at 125k, but **it did not reproduce on Studio 1 (measured: >5.5 min, stopped)** | Exact | **Yes, and it's the biggest user-visible win for "ask many questions about one 524k doc."** The cache holds 8 global KVCaches (16 GiB bf16 @524k, 8 GiB FP8) plus 40 × 1024-token rings plus conv states. Reload from internal SSD in seconds (estimate) | Pattern exists in mlx-lm. mlx_vlm has a `checkpoint_store` / sessions but no proven Inkling path | Low–medium |
| F1 | **Ring attention / context parallel** | Generic; MInference shows 1M → 22 s with TP plus CP on 8×A100 ([2407.02490](https://arxiv.org/abs/2407.02490)) | Exact | Works for 2 Studios, but needs custom online-softmax merge kernels plus collectives | **No ready MLX implementation** (only pipeline and tensor parallel exist, per [Apple ML research](https://machinelearning.apple.com/research/mlx-distributed-inference)) | High |
| F2 | **Tensor parallel over Thunderbolt-5 RDMA (MLX JACCL)** | Apple: TP cuts TTFT at 8k from 24.6 → 14.4 s (Mistral 128B, 2× M5 Ultra) and 4.1 → 3.1 s (Qwen 27B). Pipeline-parallel prefill up to ~1.8× @65k. "Tensor communication overhead can limit MoE prefill gains" at short context ([Apple ML research](https://machinelearning.apple.com/research/mlx-distributed-inference), [JACCL README](https://github.com/ml-explore/mlx/blob/main/mlx/distributed/jaccl/lib/README.md)) | Exact | **Yes, and it's already in the code.** `Attention.shard` / `LanguageModel.shard` in inkling2 support a 2-way head plus expert split. At long context the attention share (73–85%) splits cleanly by KV head (8 → 4 per Mac). Caveat: `_project` zero-pads full projection weights (for bitwise determinism), so the projection FLOPs are *not* halved; MoE and attention are | Yes: `mlx.launch --backend jaccl`, with a second M4 Max Studio over TB5. Needs `rdma_ctl enable` on both Macs | Low code, but **needs machine changes** (§5) |
| F3 | **exo** | 3,620 prefill / 210 decode tok/s on Inkling, M5 Ultra, short context (exo tweet). Pipeline parallel | Exact | #2086 says pipeline parallel doesn't work with the looped layers. No long-context numbers | App | — |
| G1 | **Disaggregated prefill on a GPU** (exo DGX Spark → Mac) | Llama-3.1-8B, 8k prompt: Mac 6.42 s, Spark 4.34 s, **both 2.32 s (2.8×)**, with KV streamed layer by layer over 10 GbE ([exo FAQ](https://exolabs.net/faq/can-i-mix-mac-and-nvidia/), [Simon Willison](https://simonwillison.net/2025/Oct/16/nvidia-dgx-spark-apple-mac-studio/)). exo 1.0 KV streaming isn't public | Exact, if the GPU kernel is exact | A datacenter GPU prefills far faster than an M4 Max, but shipping the KV at 524k is **16 GiB bf16 / 8 GiB FP8** plus small SWA and conv state, so the link bandwidth decides it (estimate: ~2.3 min at 1 Gb/s) | No. Needs a custom vLLM-side KV export of per-visit caches, a format converter, and an MLX loader | High (1–2 weeks, estimate). Out of scope for a one-Mac recipe |
| H1 | **Speculative / MTP decode** (DSpark) | Ours (measured): stock +22–24% decode @32k–131k. **On the fork it hurts: 28.0 → 10.4 tok/s, +3 GiB** at 131k | Exact | Decode only, zero prefill effect | Built in | Leave off on the fork |
| H2 | Hardware: **M5 Neural Accelerators** | Apple: TTFT up to 4.06× vs M4 (4k prompts, Qwen3-14B-4bit) ([Apple ML research](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)) | — | Explains why M5 Pro/Ultra community numbers beat ours | — | Buying hardware; not a software path |

## 3. Why sparse prefill should suit Inkling (with one caveat)

- **What makes it a good fit:**
  - The rel-bias is only non-constant within 1024 tokens, and every block-sparse method keeps the diagonal band dense anyway. The far field is plain `q·k·tau` plus a per-query constant.
  - The query-dependent sink gives each query an explicit "attend nowhere" outlet, so it doesn't have to spread probability mass across the whole context. That should make far-field attention sharper and sparser (hypothesis; §5 Step 1 measures it).
  - The SWA layers already give dense local coverage on 40 of 48 visits.
- **Caveat:** the 8 global visits are the model's *only* long-range path. In Llama-type models every layer can reach far, so pruning is spread out. Here it lands entirely on these 8 visits, and the needle depends on them. The quality gate has to be stricter: multi-depth needles plus a RULER subset, not one mid-document needle.
- **Loop revisits** (layers 11 and 17 ×3) may have different patterns per pass. Calibrate per visit, not per layer.

## 4. Quick, nearly free wins found in the code

1. **Full-chunk LM head during prefill (stock v0.2.0).**
   - What happens: `generate/ar.py` passes `logits_to_keep=1`, but `inkling2/language.py LanguageModel.__call__` swallows it in `**kwargs` and computes logits for *every token in the chunk*.
   - Cost: 0.93 GFLOP/token, about 9% of FLOPs at 32k and ~3% at 524k. Plus a `[step, 201,024]` transient, about 1.6 GB bf16 at step 4096, more if fp32.
   - Fix: slice `pre_norm[:, -1:]` before `_logits_from_norm` when not verifying.
   - Gain (estimate): +3–9% speed, and the freed memory allows a larger prefill step. **I haven't checked whether betizzel's fork already fixes this.**
2. **Prefill step.** The fork used 8192 at 32k/131k and 4096 at 262k. Keep the largest step that stays under the 51.84 GiB cap at each length, since bigger GEMM tiles mean better MoE efficiency. Expect ≤5–10% (estimate).
3. **Pass/fail mechanics to keep:** needle at several depths, no swap growth, `peak_mlx_GiB` reported.

## 5–6. (Internal planning sections removed.)

Since this memo was written we chose to exhaust **exact, full-quality, single-Mac** work first: no sparse or approximate attention and no FP8 KV yet. The prefix cache (item #3's first half) is done and measured; see the README.

## 7. Arithmetic (reproducible)

```python
H=2304; nq=24; nkv=8; d=128; drel=16
attn = H*nq*d*2 + nq*d*H + H*nkv*d*2 + H*nq*drel + H*nq      # q,gate,o,k,v,r,sink = 26.9M
moe  = 8*3*H*1152 + 128*H                                     # 64.0M active
ctrl = 4*(64*2*H + 2*H*64)                                    # loop MessageControllers, 24 visits
params = 48*(attn+moe) + 24*ctrl + 8*2*H*32                   # 4.42B active/token
dense = 2*params                                              # 8.84 GFLOP/token
swa   = 40*4*1024*d*nq                                        # 0.50 GFLOP/token
glob  = lambda L: 8*2*L*L*d*nq                                # 8 caches x 4*(L^2/2)*H*d
# 524,288: dense+swa 4.90 PF, global 13.51 PF -> 18.41 PF; /16.2e12 = 18.9 min, /14.2e12 = 21.6 min, /10e12 = 30.7 min
```
