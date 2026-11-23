# Inkling 2 Mini quality evals — 23 Nov 2026

Host: Studio 1 (`Tomass-Mac-Studio-6.local`). Endpoint: `127.0.0.1:8080` via `scripts/serve.sh 8080`, draft on, TK=4, `MLX_VLM_TOKEN_QUEUE_TIMEOUT=3600`. Recipe commit: `fd26d84`. Started 08:24 CET, finished 11:30 CET.

Sampling (from `meta.json`): temperature 0.0, top_p 1.0, max_tokens 4096, reasoning_effort high (0.9), thinking on, n_samples 1, greedy, seed 20261123. Scoring uses the assistant answer content, not the reasoning trace. The official HF model card does not report these benchmarks.

| Task | Score | Truncated | Errors | Time |
| --- | --- | --- | --- | --- |
| GSM8K (random 200 of 1319) | 96.5% (193/200) | 4 | 0 | 2470 s |
| MMLU-Pro (random 200) | 69.0% (138/200) | 52 | 0 | 6866 s |
| GPQA Diamond (full 198) | SKIPPED (dataset unavailable, see `cache/*_subset.json`) | — | — | — |
| AIME 2025 (full 30) | 36.7% (11/30) | 19 | 0 | 1820 s |

`max_tokens` 4096 truncated 26% of MMLU-Pro and 63% of AIME 2025 (truncated = scored wrong), so scores are a floor. GPQA needs Hugging Face access to the gated `Idavidrein/gpqa` dataset.
