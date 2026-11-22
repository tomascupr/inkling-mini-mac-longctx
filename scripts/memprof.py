#!/usr/bin/env python3
"""Per-step working-memory profiler for Inkling-2-Mini on the betizzel fork.
Pre-sizes the 8 global KVCaches to S keys (zeros, bf16, like probe_cache.py),
then runs ONE prefill chunk of T real tokens and records MLX peak minus the
active memory before the step. MODE=chunk: one graph per chunk (as generate()).
MODE=layer: also mx.eval after every layer visit and record the peak per visit.
Output text/values are irrelevant here; this measures memory only."""
import gc, json, os, sys, time
from pathlib import Path
import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.models import cache as cache_mod
GiB = 1024**3
ROOT = Path(os.environ.get("ROOT", Path.home() / "inkling-mini"))
MODEL = os.environ.get("MODEL", str(ROOT / "models/Inkling-2-Mini"))
PLAN = json.loads(os.environ.get("PLAN", '[{"S":0,"T":4096}]'))
MODE = os.environ.get("MODE", "chunk")
OUT = os.environ.get("OUT", "memprof.json")
if os.environ.get("CACHE_LIMIT"):
    mx.set_cache_limit(int(float(os.environ["CACHE_LIMIT"]) * GiB))
if os.environ.get("PATCH"):
    import importlib; importlib.import_module(os.environ["PATCH"]).apply()

def nbytes(tree):
    from mlx.utils import tree_flatten
    return sum(v.nbytes for _, v in tree_flatten(tree) if isinstance(v, mx.array))

model, proc = load(MODEL)
mx.eval(model.parameters())
res = {"active_after_load_GiB": mx.get_active_memory() / GiB, "mode": MODE, "runs": []}
for name, mod in model.children().items():
    res[f"params_{name}_GiB"] = nbytes(mod) / GiB if hasattr(mod, "parameters") else None
lm = model.language_model
res["lm_head_GiB"] = nbytes(lm.lm_head.parameters()) / GiB
res["embed_GiB"] = nbytes(lm.model.embed_tokens.parameters()) / GiB
print(json.dumps(res), flush=True)
tok = proc.tokenizer if hasattr(proc, "tokenizer") else proc
corpus = (ROOT / "corpus.txt").read_text()[:400000]
ids_all = tok.encode(corpus)

visit_peaks = []
if MODE == "layer":
    import mlx_vlm.models.inkling2.language as L2
    import mlx_vlm.models.inkling2_loop.language as LL
    for cls in (L2.DecoderLayer, LL.LoopDecoderLayer):
        orig = cls.__call__
        def wrapped(self, x, blocks, cache=None, *a, _orig=orig, **k):
            mx.reset_peak_memory(); before = mx.get_active_memory()
            out = _orig(self, x, blocks, cache, *a, **k)
            mx.eval(out, blocks, cache.state if cache is not None else [])
            visit_peaks.append(((mx.get_peak_memory() - before) / GiB, self.is_sliding))
            return out
        cls.__call__ = wrapped

for p in PLAN:
    S, T = p["S"], p["T"]
    gc.collect(); mx.clear_cache()
    cache = lm.make_cache(); cap = ((S + T + 255) // 256) * 256
    for cl in cache:
        kv = cl[0]
        if type(kv) is cache_mod.KVCache:
            kv.keys = mx.zeros((1, 8, cap, 128), mx.bfloat16)
            kv.values = mx.zeros((1, 8, cap, 128), mx.bfloat16)
            kv.offset = S
            mx.eval(kv.keys, kv.values)
    mx.eval([c.state for c in cache])
    gc.collect(); mx.clear_cache()
    ids = mx.array(ids_all[:T])[None]
    before = mx.get_active_memory(); mx.reset_peak_memory(); visit_peaks.clear()
    t0 = time.perf_counter()
    out = lm(ids, cache=cache, logits_to_keep=1)
    if p.get("logits"):
        mx.eval(out.logits[:, -1, :])
    targets = [c.state for c in cache]
    if p.get("delout"):
        del out; out = None
    mx.eval(targets)
    del targets
    dt = time.perf_counter() - t0
    peak = mx.get_peak_memory()
    del out
    rec = {"S": S, "T": T, "cap": cap, "before_GiB": before / GiB, "peak_GiB": peak / GiB,
           "working_GiB": (peak - before) / GiB, "after_GiB": mx.get_active_memory() / GiB,
           "cache_GiB": mx.get_cache_memory() / GiB, "s": dt, "tok_s": T / dt, "logits": bool(p.get("logits")), "delout": bool(p.get("delout"))}
    if visit_peaks:
        rec["visit_max_GiB"] = max(v for v, _ in visit_peaks)
        rec["visit_glob_max_GiB"] = max([v for v, s in visit_peaks if not s] or [0])
        rec["visit_swa_max_GiB"] = max([v for v, s in visit_peaks if s] or [0])
        rec["visits"] = len(visit_peaks)
        rec["visit_list"] = [round(v, 3) for v, _ in visit_peaks]
    res["runs"].append(rec); print(json.dumps(rec), flush=True)
    del cache; gc.collect(); mx.clear_cache()
Path(OUT).write_text(json.dumps(res, indent=1))
print("DONE")
