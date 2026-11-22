"""Summarise raw-stock.jsonl / raw-stack4.jsonl (+ telemetry) into summary.json, divergence.jsonl and a markdown table."""
import json, math, statistics as st, datetime as dt
from collections import defaultdict
def load(p):
    return {r["id"]: r for r in map(json.loads, open(p))}
A, B = load("raw-stock.jsonl"), load("raw-stack4.jsonl")
def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)
def pct(xs, q):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None
summary, div = {}, []
benches = sorted(b for b in {r["bench"] for r in A.values()} if any(i in B for i in A if A[i]["bench"] == b))
for b in benches:
    ids = sorted(i for i in A if A[i]["bench"] == b and i in B)
    s = {"n_paired": len(ids)}
    for name, R in (("stock", A), ("stack4", B)):
        rs = [R[i] for i in ids]; ok = [r for r in rs if r.get("ok")]
        k = sum(r["correct"] for r in ok); lo, hi = wilson(k, len(rs))
        s[name] = {"correct": k, "n": len(rs), "acc": k / max(1, len(rs)), "ci95": [lo, hi],
                   "errors": len(rs) - len(ok), "truncated": sum(r["truncated"] for r in ok),
                   "no_parse": sum(r["pred"] is None for r in ok) if b != "ruler_niah_32k" else None,
                   "prompt_tokens_mean": st.mean(r["prompt_tokens"] for r in ok),
                   "gen_tokens_mean": st.mean(r["gen_tokens"] for r in ok),
                   "gen_tokens_total": sum(r["gen_tokens"] for r in ok),
                   "ttft_s_median": st.median(r["ttft_s"] for r in ok), "ttft_s_p95": pct([r["ttft_s"] for r in ok], .95),
                   "prefill_tps_median": st.median(r["prompt_tps"] for r in ok),
                   "decode_tps_median": st.median(r["decode_tps"] for r in ok),
                   "decode_tps_min": min(r["decode_tps"] for r in ok),
                   "decode_tps_max": max(r["decode_tps"] for r in ok),
                   "peak_mlx_GiB_max": max(r["peak_mlx_GiB"] for r in ok),
                   "wall_s": sum(r["elapsed_s"] for r in ok)}
    both = [(A[i]["correct"], B[i]["correct"]) for i in ids if A[i].get("ok") and B[i].get("ok")]
    s["paired"] = {"both_right": sum(a and b_ for a, b_ in both), "both_wrong": sum((not a) and (not b_) for a, b_ in both),
                   "stock_only": sum(a and not b_ for a, b_ in both), "stack4_only": sum(b_ and not a for a, b_ in both)}
    n01, n10 = s["paired"]["stack4_only"], s["paired"]["stock_only"]
    # exact two-sided McNemar (binomial on discordant pairs)
    m = n01 + n10
    s["paired"]["mcnemar_p"] = min(1.0, 2 * sum(math.comb(m, j) for j in range(0, min(n01, n10) + 1)) / 2 ** m) if m else 1.0
    ident, fd, agree_tok, tot_tok, same_ans = 0, [], 0, 0, 0
    for i in ids:
        ta, tb = A[i].get("tokens", []), B[i].get("tokens", [])
        k = next((j for j in range(min(len(ta), len(tb))) if ta[j] != tb[j]), min(len(ta), len(tb)))
        same = ta == tb; ident += same
        agree_tok += k; tot_tok += max(len(ta), len(tb))
        same_ans += (A[i].get("pred"), A[i].get("correct")) == (B[i].get("pred"), B[i].get("correct"))
        if not same: fd.append(k)
        div.append({"bench": b, "id": i, "identical": same, "first_divergence_token": None if same else k,
                    "len_stock": len(ta), "len_stack4": len(tb),
                    "stock_ctx": None if same else A[i]["text"][:0], "pred_stock": A[i].get("pred"), "pred_stack4": B[i].get("pred"),
                    "correct_stock": A[i].get("correct"), "correct_stack4": B[i].get("correct")})
    s["agreement"] = {"items_token_identical": ident, "items": len(ids),
                      "token_agreement_prefix_rate": agree_tok / max(1, tot_tok),
                      "same_answer_and_score": same_ans,
                      "first_divergence_median": st.median(fd) if fd else None,
                      "first_divergence_min": min(fd) if fd else None}
    summary[b] = s
# telemetry
tel = []
try:
    for l in open("telemetry-macmon.jsonl"):
        try: tel.append(json.loads(l))
        except Exception: pass
except FileNotFoundError: pass
if tel:
    g = [t["gpu_usage"][1] for t in tel]; gp = [t["gpu_power"] for t in tel]; sp = [t.get("sys_power", 0) for t in tel]
    ram = [t["memory"]["ram_usage"] / 2**30 for t in tel]; sw = [t["memory"]["swap_usage"] / 2**30 for t in tel]
    summary["telemetry"] = {"samples": len(tel), "interval_ms": 2000, "gpu_util_mean": st.mean(g), "gpu_util_p50": pct(g, .5),
        "frac_samples_gpu_lt_50pct": sum(x < .5 for x in g) / len(g), "gpu_power_W_mean": st.mean(gp), "gpu_power_W_max": max(gp),
        "sys_power_W_mean": st.mean(sp), "ram_used_GiB_max": max(ram), "swap_used_GiB_max": max(sw),
        "gpu_temp_max": max(t["temp"]["gpu_temp_avg"] for t in tel)}
    # per-phase telemetry: join samples to requests by time window
    def ts(t): return dt.datetime.fromisoformat(t["timestamp"]).timestamp()
    tt = [(ts(t), t) for t in tel]
    for name, R in (("stock", A), ("stack4", B)):
        for b in benches:
            us, ws = [], []
            for r in R.values():
                if r["bench"] != b or not r.get("ok"): continue
                s0, s1 = r["started"], r["started"] + r["elapsed_s"]
                for x, t in tt:
                    if s0 <= x <= s1: us.append(t["gpu_usage"][1]); ws.append(t["gpu_power"])
            if us: summary[b][name]["gpu_util_mean_during"] = st.mean(us); summary[b][name]["gpu_power_W_mean_during"] = st.mean(ws)
    # stalls: gaps between consecutive requests and between samples
    gaps = [b_ - a for (a, _), (b_, _) in zip(tt, tt[1:])]
    summary["telemetry"]["max_sample_gap_s"] = max(gaps) if gaps else None
json.dump(summary, open("summary.json", "w"), indent=1)
with open("divergence.jsonl", "w") as f:
    for d in div: d.pop("stock_ctx", None); f.write(json.dumps(d) + "\n")
for b in benches:
    s = summary[b]; a, c = s["stock"], s["stack4"]
    print(f"{b}: n={s['n_paired']} stock {a['correct']}/{a['n']} {a['acc']:.1%} [{a['ci95'][0]:.1%},{a['ci95'][1]:.1%}] | stack4 {c['correct']}/{c['n']} {c['acc']:.1%} [{c['ci95'][0]:.1%},{c['ci95'][1]:.1%}] | paired {s['paired']} | agree {s['agreement']}")
    for k in ("truncated", "errors", "no_parse", "prompt_tokens_mean", "gen_tokens_mean", "ttft_s_median", "prefill_tps_median", "decode_tps_median", "decode_tps_min", "peak_mlx_GiB_max", "wall_s", "gpu_util_mean_during", "gpu_power_W_mean_during"):
        print(f"   {k}: stock {a.get(k)}  stack4 {c.get(k)}")
print(summary.get("telemetry"))
