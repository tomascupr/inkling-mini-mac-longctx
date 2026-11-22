"""Draw the README charts (docs/img/*.svg) from the measured result files. No dependencies.

Head-to-head (Studio 2, one M4 Max 64 GB): MLX-Serve's own server log vs this repo's server log
(results/h2h-20261122/). Every request's decode/prefill rate is the server-reported one; the first request per
context after server start is a warm-up and is not charted. Long context (Studio 1): results/gates/*.json.
usage: python scripts/charts.py
"""
import json, os, re, statistics

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
H2H = os.path.join(ROOT, "results", "h2h-20261122")
IMG = os.path.join(ROOT, "docs", "img")
LABEL = "M4 Max 64 GB, same official weights"
INK, OURS, THEIRS, GRID, MUTED = "#111827", "#2563eb", "#9ca3af", "#e5e7eb", "#6b7280"


def mlxserve(log):
    """MLX-Serve '<- P+G tokens streamed [prefill: X tok/s, decode: Y tok/s]' lines; skips its 16-token warm-up."""
    rows = {}
    for m in re.finditer(r"<- (\d+)\+(\d+) tokens streamed \[prefill: ([\d.]+) tok/s, decode: ([\d.]+) tok/s\]",
                         open(log).read()):
        p, g, pre, dec = int(m[1]), int(m[2]), float(m[3]), float(m[4])
        if g > 16:
            rows.setdefault(p, []).append((pre, dec))
    return rows


def ours(log):
    """mlx-vlm 'Request completed: ... prompt_tokens=P ... prefill=X tok/s decode=Y tok/s' lines; first per P is warm-up."""
    rows = {}
    for m in re.finditer(r"Request completed:.*?prompt_tokens=(\d+).*?prefill=([\d.]+) tok/s decode=([\d.]+) tok/s",
                         open(log).read()):
        rows.setdefault(int(m[1]), []).append((float(m[2]), float(m[3])))
    return {p: v[1:] for p, v in rows.items()}


def bars(path, title, subtitle, groups, unit="tok/s"):
    """groups: [(label, [(series, value, note), ...])]"""
    W, H, top, bottom, left = 720, 360, 92, 64, 24
    vmax = max(v for _, s in groups for _, v, _ in s) * 1.18
    gw = (W - 2 * left) / len(groups)
    bw = min(110, gw / (len(groups[0][1]) + 1.2))
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
           f'font-family="Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif">',
           f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
           f'<text x="{left}" y="34" font-size="20" font-weight="700" fill="{INK}">{title}</text>',
           f'<text x="{left}" y="58" font-size="13" fill="{MUTED}">{subtitle}</text>']
    base = H - bottom
    for gi, (glabel, series) in enumerate(groups):
        x0 = left + gi * gw + (gw - bw * len(series) - 12 * (len(series) - 1)) / 2
        for si, (name, v, note) in enumerate(series):
            h = (base - top) * v / vmax
            x = x0 + si * (bw + 12)
            col = OURS if name == "This repo" else THEIRS
            out.append(f'<rect x="{x:.1f}" y="{base - h:.1f}" width="{bw:.1f}" height="{h:.1f}" rx="4" fill="{col}"/>')
            out.append(f'<text x="{x + bw / 2:.1f}" y="{base - h - 8:.1f}" font-size="15" font-weight="700" '
                       f'text-anchor="middle" fill="{INK}">{v:,.1f}</text>')
            out.append(f'<text x="{x + bw / 2:.1f}" y="{base + 18:.1f}" font-size="12" text-anchor="middle" '
                       f'fill="{INK}">{name}</text>')
            if note:
                out.append(f'<text x="{x + bw / 2:.1f}" y="{base - h + 18:.1f}" font-size="11" text-anchor="middle" '
                           f'fill="#ffffff">{note}</text>')
        cx = left + gi * gw + gw / 2
        out.append(f'<text x="{cx:.1f}" y="{base + 42:.1f}" font-size="14" font-weight="600" text-anchor="middle" '
                   f'fill="{INK}">{glabel}</text>')
    out.append(f'<line x1="{left}" y1="{base}" x2="{W - left}" y2="{base}" stroke="{GRID}"/>')
    out.append(f'<text x="{W - left}" y="{H - 8}" font-size="11" text-anchor="end" fill="{MUTED}">{LABEL} · {unit}, higher is better</text>')
    out.append("</svg>")
    open(path, "w").write("\n".join(out))


def pct(a, b):
    return f"{100 * (a / b - 1):+.0f}%"


def main():
    os.makedirs(IMG, exist_ok=True)
    cfg = json.load(open(os.path.join(H2H, "charts.json")))
    them = mlxserve(os.path.join(H2H, cfg["mlxserve_log"]))
    read_ours = ours(os.path.join(H2H, cfg["ours_reading_log"]))
    write_ours = ours(os.path.join(H2H, cfg["ours_writing_log"]))
    summary = {}
    rg, wg = [], []
    for p, name in ((4078, "4k prompt"), (32750, "32k prompt")):
        t_pre = statistics.median(x[0] for x in them[p])
        t_dec = statistics.median(x[1] for x in them[p])
        o_pre = statistics.median(x[0] for x in read_ours[p])
        o_dec = statistics.median(x[1] for x in write_ours[p])
        summary[name] = dict(mlxserve_prefill_decode=them[p],
                             ours_prefill=[x[0] for x in read_ours[p]], ours_decode=[x[1] for x in write_ours[p]])
        rg.append((name, [("MLX-Serve", t_pre, ""), ("This repo", o_pre, pct(o_pre, t_pre))]))
        wg.append((name, [("MLX-Serve", t_dec, ""), ("This repo", o_dec, pct(o_dec, t_dec))]))
    bars(os.path.join(IMG, "reading-vs-mlxserve.svg"), "Reading a prompt (prefill)",
         cfg["reading_subtitle"], rg)
    bars(os.path.join(IMG, "writing-vs-mlxserve.svg"), "Writing the answer (decode), draft head on",
         cfg["writing_subtitle"], wg)
    # Long context, this repo only (Studio 1, no draft).
    runs = []
    for f in cfg["long_context_runs"]:
        r = json.load(open(os.path.join(ROOT, f)))["runs"][0]
        runs.append(r)
    W, H, top, bottom, left = 720, 380, 92, 70, 24
    gw = (W - 2 * left) / len(runs)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
           f'font-family="Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif">',
           f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
           f'<text x="{left}" y="34" font-size="20" font-weight="700" fill="{INK}">Up to 1M tokens of context on one 64 GB Mac</text>',
           f'<text x="{left}" y="58" font-size="13" fill="{MUTED}">{cfg["long_subtitle"]}</text>']
    base = H - bottom
    pmax = max(r["prompt_tps"] for r in runs) * 1.2
    for i, r in enumerate(runs):
        lab = cfg["long_context_labels"][i]
        bw = gw * 0.34
        x = left + i * gw + gw / 2 - bw - 3
        for j, (v, col) in enumerate(((r["prompt_tps"], OURS), (r["decode_tps"] * 10, "#93c5fd"))):
            h = (base - top) * v / pmax
            xx = x + j * (bw + 6)
            txt = f'{r["prompt_tps"]:.0f}' if j == 0 else f'{r["decode_tps"]:.1f}'
            out.append(f'<rect x="{xx:.1f}" y="{base - h:.1f}" width="{bw:.1f}" height="{h:.1f}" rx="3" fill="{col}"/>')
            out.append(f'<text x="{xx + bw / 2:.1f}" y="{base - h - 6:.1f}" font-size="12" font-weight="700" '
                       f'text-anchor="middle" fill="{INK}">{txt}</text>')
        out.append(f'<text x="{left + i * gw + gw / 2:.1f}" y="{base + 18:.1f}" font-size="14" font-weight="600" '
                   f'text-anchor="middle" fill="{INK}">{lab}</text>')
        out.append(f'<text x="{left + i * gw + gw / 2:.1f}" y="{base + 34:.1f}" font-size="11" text-anchor="middle" '
                   f'fill="{MUTED}">{r["peak_mlx_GiB"]:.0f} GiB · needle {"✓" if r.get("needle_found") else "✗"}</text>')
    out.append(f'<line x1="{left}" y1="{base}" x2="{W - left}" y2="{base}" stroke="{GRID}"/>')
    out.append(f'<rect x="{left}" y="70" width="10" height="10" fill="{OURS}"/><text x="{left + 14}" y="79" font-size="11" fill="{INK}">reading, tok/s</text>')
    out.append(f'<rect x="{left + 110}" y="70" width="10" height="10" fill="#93c5fd"/><text x="{left + 124}" y="79" font-size="11" fill="{INK}">writing, tok/s (bar ×10)</text>')
    out.append(f'<text x="{W - left}" y="{H - 8}" font-size="11" text-anchor="end" fill="{MUTED}">{LABEL} · peak GPU memory · hidden fact found</text>')
    out.append("</svg>")
    open(os.path.join(IMG, "long-context.svg"), "w").write("\n".join(out))
    json.dump(summary, open(os.path.join(H2H, "summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
