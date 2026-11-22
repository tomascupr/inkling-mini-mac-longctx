"""Short quality benchmark for Inkling-2-Mini on MLX: greedy, fixed prompts, per-request timing.
Run once per build with the build's python / PYTHONPATH:
  BUILD=stack4|stock BENCHES=gsm8k,mmlu_pro,ruler_niah_32k OUT=raw-<build>.jsonl python bench.py
Each output line = one item: tokens, text, timing (TTFT, prefill/decode tok/s), truncation, score."""
import json, os, re, sys, time, hashlib, traceback
from pathlib import Path
import mlx.core as mx
import mlx_vlm
from mlx_vlm import load, apply_chat_template, stream_generate

MODEL = str(Path.home() / ".exo-inkling-bench/models/thinkingmachines--Inkling-2-Mini")
BUILD = os.environ["BUILD"]; OUT = Path(os.environ["OUT"])
BENCHES = os.environ.get("BENCHES", "gsm8k,mmlu_pro,ruler_niah_32k").split(",")
LIMIT = int(os.environ.get("LIMIT", "0"))
REASONING = os.environ.get("REASONING", "low")
STEP = int(os.environ.get("STEP", "2048"))
GiB = 1024 ** 3

def final_text(t):
    return t.split("<|content_text|>")[-1] if "<|content_text|>" in t else t

def score(item, text):
    ft = final_text(text)
    if item["bench"] == "gsm8k":
        m = re.findall(r"Final answer:\s*\$?\s*(-?[\d,]*\.?\d+)", ft) or re.findall(r"(-?[\d,]*\.?\d+)", ft)
        if not m: return None, False
        pred = m[-1].replace(",", "")
        try: ok = abs(float(pred) - float(item["gold"])) < 1e-6
        except ValueError: ok = False
        return pred, ok
    if item["bench"] == "mmlu_pro":
        m = re.findall(r"Answer:\s*\(?\*{0,2}([A-J])\b", ft) or re.findall(r"\b([A-J])\b", ft.strip()[-20:])
        pred = m[-1] if m else None
        return pred, pred == item["gold"]
    if item["bench"] == "ruler_niah_32k":
        return None, all(g in ft for g in item["gold"])

items = [json.loads(l) for l in open("prompts.jsonl")]
items = [i for i in items if i["bench"] in BENCHES]
RULER_PER_TASK = int(os.environ.get("RULER_PER_TASK", "4"))  # scope cut for wall time: 4 per niah task = 12 items
if RULER_PER_TASK:
    seen = {}
    keep = []
    for i in items:
        if i["bench"] == "ruler_niah_32k":
            t = i["id"].rsplit("-", 1)[0]; seen[t] = seen.get(t, 0) + 1
            if seen[t] > RULER_PER_TASK: continue
        keep.append(i)
    items = keep
if LIMIT: items = [i for b in BENCHES for i in [x for x in items if x["bench"] == b][:LIMIT]]
done = set()
if OUT.exists():
    done = {json.loads(l)["id"] for l in open(OUT)}
t0 = time.perf_counter()
model, processor = load(MODEL)
mx.eval(model.parameters())
meta = {"type": "meta", "build": BUILD, "mlx_vlm_version": mlx_vlm.__version__, "mlx_vlm_file": mlx_vlm.__file__,
        "mlx": mx.__version__, "reasoning_effort": REASONING, "prefill_step_size": STEP, "temperature": 0.0,
        "env": {k: v for k, v in os.environ.items() if k.startswith("MLX")}, "load_s": time.perf_counter() - t0,
        "active_after_load_GiB": mx.get_active_memory() / GiB}
print(json.dumps(meta), flush=True)
with open(OUT.with_suffix(".meta.json"), "w") as f: json.dump(meta, f, indent=1)
for it in items:
    if it["id"] in done: continue
    prompt = apply_chat_template(processor, model.config, it["user"], reasoning_effort=REASONING)
    rec = {"build": BUILD, "bench": it["bench"], "id": it["id"], "max_tokens": it["max_tokens"],
           "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()[:16], "started": time.time()}
    try:
        mx.clear_cache(); mx.reset_peak_memory()
        toks, text, last, ttft = [], "", None, None
        ts = time.perf_counter()
        for r in stream_generate(model, processor, prompt, max_tokens=it["max_tokens"], temperature=0.0,
                                 prefill_step_size=STEP):
            if ttft is None: ttft = time.perf_counter() - ts
            toks.append(int(r.token)); text += r.text; last = r
        el = time.perf_counter() - ts
        pred, ok = score(it, text)
        rec.update({"ok": True, "elapsed_s": el, "ttft_s": ttft, "prompt_tokens": last.prompt_tokens,
                    "prompt_tps": last.prompt_tps, "gen_tokens": len(toks), "decode_tps": last.generation_tps,
                    "peak_mlx_GiB": mx.get_peak_memory() / GiB,
                    "finish_reason": getattr(last, "finish_reason", None),
                    "truncated": len(toks) >= it["max_tokens"], "gold": it["gold"], "pred": pred, "correct": bool(ok),
                    "tokens": toks, "text": text})
    except Exception as e:
        rec.update({"ok": False, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-2000:]})
    with open(OUT, "a") as f: f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k not in ("tokens", "text", "traceback")}), flush=True)
print("BENCH_DONE", BUILD, time.perf_counter() - t0, flush=True)
