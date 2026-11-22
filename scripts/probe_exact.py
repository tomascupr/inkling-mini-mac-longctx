#!/usr/bin/env python3
"""1M fit probe (synthetic depth). Pre-sizes the 8 global KVCaches to `cap`
keys (bf16 zeros) with offset S, then runs a real ~2.5k-token chunked prefill +
decode through stream_generate on top. Measures MLX peak at that total depth.
Values are meaningless (zero K/V); this is a memory/speed probe only.
env PLAN='[{"S":1044480,"step":512,"max_tokens":128}]' TEXT_ONLY=1 OUT=..."""
import gc, json, os, time
from datetime import datetime
from pathlib import Path
import mlx.core as mx
import mlx_vlm
from mlx_vlm import apply_chat_template, load, stream_generate
from mlx_vlm.models import cache as cache_mod
GiB = 1024**3
ROOT = Path(os.environ.get("ROOT", Path.home() / "inkling-mini"))
D = ROOT  # corpus.txt lives here (any plain text longer than the context)
MODEL = os.environ.get("MODEL", str(ROOT / "models/Inkling-2-Mini"))
PLAN = json.loads(os.environ["PLAN"]); OUT = Path(os.environ.get("OUT", "probe-exact.json"))
lim = mx.device_info()["max_recommended_working_set_size"] / GiB
res = {"started": datetime.now().isoformat(timespec="seconds"), "mlx_vlm_file": mlx_vlm.__file__, "cap_GiB": lim,
       "env": {k: v for k, v in os.environ.items() if k.startswith("MLX")}, "runs": []}
model, processor = load(MODEL)
if os.environ.get("TEXT_ONLY") == "1":
    from exact_text_only import drop_vision
    res["dropped_vision_GiB"] = drop_vision(model)
mx.eval(model.parameters()); res["weights_active_GiB"] = mx.get_active_memory() / GiB
prompt = apply_chat_template(processor, model.config, "Summarise this text in three sentences:\n\n"
                             + (D / "corpus.txt").read_text()[:6000], reasoning_effort="none")
tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
n_prompt = len(tok.encode(prompt))
lm = model.language_model
for p in PLAN:
    S, step, mt = p["S"], p["step"], p.get("max_tokens", 128)
    rec = {**p, "prompt_tokens": n_prompt}
    try:
        gc.collect(); mx.clear_cache(); mx.reset_peak_memory()
        cache = lm.make_cache(); cap = ((S + n_prompt + mt + 255) // 256) * 256
        for cl in cache:
            kv = cl[0]
            if type(kv) is cache_mod.KVCache:
                kv.keys = mx.zeros((1, 8, cap, 128), mx.bfloat16); kv.values = mx.zeros((1, 8, cap, 128), mx.bfloat16)
                kv.offset = S; mx.eval(kv.keys, kv.values)
        rec.update(cap=cap, after_alloc_GiB=mx.get_active_memory() / GiB)
        print("alloc", json.dumps(rec), flush=True)
        if rec["after_alloc_GiB"] > lim - 0.3:
            raise RuntimeError("pre-check: alloc too close to cap")
        t = time.perf_counter(); last = None; n = 0
        for r in stream_generate(model, processor, prompt, prompt_cache=cache, max_tokens=mt, temperature=0.0,
                                 prefill_step_size=step):
            last = r; n += 1
            if n == 1:
                rec["prefill_peak_GiB"] = mx.get_peak_memory() / GiB
                rec["prefill_working_GiB"] = rec["prefill_peak_GiB"] - rec["after_alloc_GiB"]
                mx.reset_peak_memory()
        rec["decode_peak_GiB"] = mx.get_peak_memory() / GiB
        peak = max(rec["prefill_peak_GiB"], rec["decode_peak_GiB"])
        rec.update(ok=True, elapsed_s=time.perf_counter() - t, prompt_tps=last.prompt_tps, decode_tps=last.generation_tps,
                   gen_tokens=n, peak_mlx_GiB=peak, working_GiB=peak - rec["after_alloc_GiB"], headroom_GiB=lim - peak)
    except Exception as e:
        rec.update(ok=False, error=f"{type(e).__name__}: {e}", peak_mlx_GiB=mx.get_peak_memory() / GiB)
    res["runs"].append(rec); OUT.write_text(json.dumps(res, indent=1)); print(json.dumps(rec), flush=True)
    cache = None; gc.collect(); mx.clear_cache()
    if not rec.get("ok"):
        break
res["finished"] = datetime.now().isoformat(timespec="seconds"); OUT.write_text(json.dumps(res, indent=1)); print("DONE")
