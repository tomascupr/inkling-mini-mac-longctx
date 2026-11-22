# Per-visit prefill profile at 1M depth (2026-11-22 14:11–14:30 CET, Studio 1)

`mkprof.py` adds env-gated wall-clock timers around the existing per-visit `mx.eval` in
`inkling2_loop/language.py` (timing only). Runs use `common/probe_exact.py` (synthetic depth: global KV pre-sized with
zeros), prompt 2,551 tokens, step 2048 → one 2048-token chunk + one 502-token chunk.
Columns: step, layer, cache index, pass, cache type, offset, chunk tokens, seconds. Global visits are steps 5, 11, …, 47.

| file | setup |
|---|---|
| visits.tsv | stack6, depth 0 (chunks 1–2) then depth 1,044,480 (chunks 3–4) |
| visits-kt3.tsv | stack6 + KT=3, depth 1,044,480 |
| visits-bal{0,28,29,30,31,32}.tsv, visits-b*.tsv | depth 0 with N GiB of extra MLX arrays held (memory-pressure control) |
| visits-c0.tsv | depth 0, `mx.set_cache_limit(0)` |
| visits-hv-kt{1,3}.tsv | `mkhost.py` (host vocab tables during prefill), depth 1,044,480, KT1 / KT3 |
