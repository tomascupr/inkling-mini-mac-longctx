#!/usr/bin/env python3
"""Greedy token-identity check (optionally with the DSpark draft head: BLOCK=4) + speed/peak for Inkling-2-Mini long-context runs.
Same prompt construction as bench/bench_longctx.py (PG19 corpus + needle in the
middle). Records the full generated token-id list so two runs (unpatched fork vs
patched overlay) can be compared token for token.
env: PLAN='[{"ctx":32768,"step":2048,"max_tokens":256}]' OUT=... TAG=...
     TEXT_ONLY=1 drops the vision tower after load (exact for text prompts)."""
import gc, hashlib, json, os, time
from datetime import datetime
from pathlib import Path
import mlx.core as mx
import mlx_vlm
from mlx_vlm import apply_chat_template, load, stream_generate
GiB = 1024**3
ROOT = Path(os.environ.get("ROOT", Path.home() / "inkling-mini"))
D = ROOT  # corpus.txt lives here
MODEL = os.environ.get("MODEL", str(ROOT / "models/Inkling-2-Mini"))
PLAN = json.loads(os.environ["PLAN"]); OUT = Path(os.environ.get("OUT", "exact-check.json"))
NEEDLE_CODE = "VIOLET-ANCHOR-4729"
NEEDLE = f"\n\nAn archivist's note: the catalogue code of the Prague observatory archive box is {NEEDLE_CODE}.\n\n"
QUESTION = ("\n\nQuestion: According to the archivist's note in the document above, what is the catalogue code "
            "of the Prague observatory archive box? Reply with the code only.")
res = {"started": datetime.now().isoformat(timespec="seconds"), "tag": os.environ.get("TAG"),
       "mlx_vlm_file": mlx_vlm.__file__, "env": {k: v for k, v in os.environ.items() if k.startswith("MLX")}, "runs": []}
model, processor = load(MODEL)
if os.environ.get("TEXT_ONLY") == "1":
    import sys; sys.path.insert(0, str(Path(__file__).resolve().parent)); from exact_text_only import drop_vision
    res["dropped_vision_GiB"] = drop_vision(model)
BLOCK = int(os.environ.get("BLOCK", "0"))
draft = kind = None
if BLOCK > 1:
    from mlx_vlm.speculative.drafters import load_drafter
    from mlx_vlm.speculative.utils import speculative_stats_since, speculative_stats_snapshot
    draft, kind = load_drafter(MODEL)
    mx.eval(draft.parameters())
    t = time.perf_counter(); draft.bind(model); res["bind_s"] = time.perf_counter() - t
res["block"] = BLOCK
mx.eval(model.parameters()); res["active_after_load_GiB"] = mx.get_active_memory() / GiB
tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
ids = tok.encode((D / "corpus.txt").read_text()[: 8 * max(p["ctx"] for p in PLAN)])
def build(ctx):
    doc = tok.decode(ids[: ctx - 120])
    cut = doc.find("\n\n", len(doc) // 2); cut = cut if cut > 0 else len(doc) // 2
    doc = doc[:cut] + NEEDLE + doc[cut:]
    user = "Below is a long document. Read all of it, then answer the question at the end.\n\n" + doc + QUESTION
    return apply_chat_template(processor, model.config, user, reasoning_effort=os.environ.get("REASONING", "low"))
for p in PLAN:
    prompt = build(p["ctx"])
    gc.collect(); mx.clear_cache(); mx.reset_peak_memory()
    toks, text, last = [], "", None
    h = hashlib.sha256()
    t0 = time.perf_counter()
    kw = {}
    if draft is not None:
        kw = dict(draft_model=draft, draft_kind=kind, draft_block_size=BLOCK)
        snap = speculative_stats_snapshot(draft)
    n_lp = 0
    for r in stream_generate(model, processor, prompt, max_tokens=p.get("max_tokens", 256), temperature=0.0,
                             prefill_step_size=p["step"], **kw):
        toks.append(int(r.token)); text += r.text; last = r
        if r.logprobs is not None:
            lp = r.logprobs if isinstance(r.logprobs, mx.array) else mx.array(r.logprobs)
            h.update(memoryview(lp.astype(mx.float32))); n_lp += 1
    el = time.perf_counter() - t0
    rec = {**p, "elapsed_s": el, "prompt_tokens": last.prompt_tokens, "prompt_tps": last.prompt_tps,
           "decode_tps": last.generation_tps, "gen_tokens": len(toks), "peak_mlx_GiB": mx.get_peak_memory() / GiB,
           "needle_found": NEEDLE_CODE in text, "tokens": toks, "logprobs_sha256": h.hexdigest(), "n_logprobs": n_lp, "text_tail": text[-300:]}
    if draft is not None:
        rounds, accepted, drafted = speculative_stats_since(draft, snap)
        rec.update(rounds=rounds, accepted=accepted, drafted=drafted,
                   accept_rate=accepted / drafted if drafted else None,
                   mean_emitted=(accepted + rounds) / rounds if rounds else None)
    rec["exact_verify_misses"] = model.language_model.__dict__.get("_exact_verify_misses", 0)
    if draft is not None:
        lk = draft.__dict__.pop("_lookup", None)
        rec["lookup_rounds"] = None if lk is None else lk.used
        pr = draft.__dict__.pop("_round_profile", None)
        if pr:
            import statistics as _st
            rec["profile_ms"] = {k: {"first": round(v[0] * 1e3, 1), "median_rest": round(_st.median(v[1:]) * 1e3, 2) if len(v) > 1 else None}
                                 for k, v in pr.items() if k != "bs"}
        ad = draft.__dict__.pop("_adapt", None)
        if ad is not None:
            from collections import Counter
            rec["k_hist"] = dict(Counter(ad.history))
            rec["k_time_ms"] = {k: round(v * 1e3, 2) for k, v in ad.time.items()}
    ref = os.environ.get("REF")
    if ref:
        b = json.load(open(ref))["runs"][0]
        rec.update(tokens_equal=toks == b["tokens"][: len(toks)] and len(toks) == len(b["tokens"]),
                   sha_equal=rec["logprobs_sha256"] == b["logprobs_sha256"],
                   first_mismatch=next((i for i, (a, c) in enumerate(zip(toks, b["tokens"])) if a != c), None))
    res["runs"].append(rec); OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k != "tokens"}), flush=True)
res["finished"] = datetime.now().isoformat(timespec="seconds"); OUT.write_text(json.dumps(res, indent=1)); print("DONE")
