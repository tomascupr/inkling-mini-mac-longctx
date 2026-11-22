"""Draw the charts and write the result tables straight from the measured logs (needs matplotlib).

Head-to-head vs MLX-Serve: every session listed under "h2h_sessions" in results/h2h-20261122/charts.json. A session
log holds MLX-Serve's own log lines ('<- P+G tokens streamed [prefill: X tok/s, decode: Y tok/s]') and this repo's
server log lines ('LOG prompt_tokens=P ... prefill=X tok/s decode=Y tok/s'), same Mac, same session, MLX-Serve first.
Each prompt length found in the logs is one row of the tables and one group in the charts, so a new length is added
by listing its session log in charts.json and running this script again.

Counting: on both sides, every request except the first one at each prompt length in a session (reported
separately). MLX-Serve's first 4k request is its own 16-token warm-up.
Rounding never favours this repo: its means are rounded down, MLX-Serve's up, differences (from unrounded means) down.
Verdict: "faster" only if, in every session, our slowest counted run beats MLX-Serve's fastest one; "behind" if
MLX-Serve's slowest beats our fastest in every session; otherwise "level".

Writes docs/img/{reading,writing}-vs-mlxserve.svg, docs/img/long-context.svg, the 1600x900 PNGs for X
(docs/img/x-vs-mlxserve.png with both metrics, docs/img/x-{reading,writing}-vs-mlxserve.png),
results/h2h-20261122/summary.json and the generated blocks (between <!-- gen:NAME --> and <!-- /gen:NAME -->) in
README.md and docs/BENCHMARKS.md.
usage: python scripts/charts.py
"""
import json, logging, math, os, re, sys
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
H2H = os.path.join(ROOT, "results", "h2h-20261122")
IMG = os.path.join(ROOT, "docs", "img")
INK, OURS, THEIRS, GRID, MUTED, LOSS = "#111827", "#2563eb", "#9ca3af", "#e5e7eb", "#6b7280", "#b91c1c"
METRICS = (("reading", "Reading a prompt (prefill)"), ("writing", "Writing the answer (decode), draft head on"))
X_BG, X_INK, X_MUTED, X_THEIRS, X_OURS, X_GAIN, X_LOSS = (
    "#0b1220", "#f8fafc", "#94a3b8", "#4b5563", "#3b82f6", "#60a5fa", "#f87171")
X_DPI, X_SIZE = 100, (16, 9)

MS_RE = re.compile(r"<- (\d+)\+(\d+) tokens streamed \[prefill: ([\d.]+) tok/s, decode: ([\d.]+) tok/s\]")
OURS_RE = re.compile(r"(?:Request completed:.*?|LOG )prompt_tokens=(\d+).*?prefill=([\d.]+) tok/s decode=([\d.]+) tok/s")

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "svg.fonttype": "none",
    "svg.hashsalt": "inkling-mini-mac-longctx",
    "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED,
    "xtick.color": INK,
    "ytick.color": MUTED,
})


def mean(xs):
    return sum(xs) / Decimal(len(xs))


def down(x, places=1):
    return x.quantize(Decimal(1).scaleb(-places), rounding=ROUND_FLOOR)


def up(x, places=1):
    return x.quantize(Decimal(1).scaleb(-places), rounding=ROUND_CEILING)


def pct(ours, theirs):
    return (mean(ours) / mean(theirs) - 1) * 100


def fmt(x, places=1):
    return f"{x:,.{places}f}"


def fmt_pct(p, places):
    s = f"{down(p, places):+.{places}f}%"
    return s.replace("-", "\u2212")


def label(p):
    return f"{2 ** round(math.log2(p)) // 1000}k"


def session_names(sessions):
    """'h29 (22:33) and h30 (22:42) CET, 2026-11-22', with the date per session when they differ."""
    dates = {s["started"][:10] for s in sessions}
    if len(dates) == 1:
        times = [f"{s['name']} ({s['started'][11:16]})" for s in sessions]
        return f"{', '.join(times[:-1])} and {times[-1]} CET, {dates.pop()}" if len(times) > 1 else f"{times[0]} CET, {dates.pop()}"
    return ", ".join(f"{s['name']} ({s['started']})" for s in sessions)


def read_mlxserve(text):
    rows = {}
    for m in MS_RE.finditer(text):
        rows.setdefault(int(m[1]), []).append((Decimal(m[3]), Decimal(m[4]), int(m[2])))
    return rows


def read_ours(text):
    rows = {}
    for m in OURS_RE.finditer(text):
        rows.setdefault(int(m[1]), []).append((Decimal(m[2]), Decimal(m[3])))
    return rows


def load_pair(mlxserve_log, ours_log):
    """{prompt_tokens: {"mlxserve_first": (pre, dec, generated), "mlxserve": [(pre, dec)], "first": (pre, dec),
    "ours": [(pre, dec)]}}"""
    them = read_mlxserve(open(os.path.join(H2H, mlxserve_log)).read())
    us = read_ours(open(os.path.join(H2H, ours_log)).read())
    out = {}
    for p in sorted(set(them) & set(us)):
        if len(us[p]) < 2 or len(them[p]) < 2:
            print(f"warning: {ours_log}: {len(them[p])} MLX-Serve and {len(us[p])} own request(s) at {p} tokens, "
                  "nothing left on one side after its first request")
            continue
        out[p] = {"mlxserve_first": them[p][0], "mlxserve": [x[:2] for x in them[p][1:]],
                  "first": us[p][0], "ours": us[p][1:]}
    return out


def session_verdict(ours, theirs):
    if min(ours) > max(theirs):
        return "ahead"
    if max(ours) < min(theirs):
        return "behind"
    return "level"


def overall(verdicts):
    if all(v == "ahead" for v in verdicts):
        return "faster"
    if all(v == "behind" for v in verdicts):
        return "behind"
    return "level"


def analyse(sessions):
    """One entry per (prompt length, metric), pooled over the sessions that measured that length."""
    rows = []
    for p in sorted({p for s in sessions for p in s["data"]}):
        for k, (metric, _) in enumerate(METRICS):
            per = []
            for s in sessions:
                d = s["data"].get(p)
                if not d:
                    continue
                t = [x[k] for x in d["mlxserve"]]
                o = [x[k] for x in d["ours"]]
                with_first = [d["first"][k]] + o
                ms_first, warmup = d["mlxserve_first"][k], d["mlxserve_first"][2] <= 16
                both = None if warmup else ([ms_first] + t, with_first)
                per.append(dict(session=s["name"], started=s["started"], mlxserve=t, ours=o, first=d["first"][k],
                                mlxserve_first=ms_first, mlxserve_first_is_warmup=warmup,
                                delta=pct(o, t), verdict=session_verdict(o, t),
                                delta_with_first=pct(with_first, t), verdict_with_first=session_verdict(with_first, t),
                                delta_both=both and pct(both[1], both[0]),
                                verdict_both=both and session_verdict(both[1], both[0])))
            t_all = [x for r in per for x in r["mlxserve"]]
            o_all = [x for r in per for x in r["ours"]]
            f_all = [r["first"] for r in per] + o_all
            full = all(not r["mlxserve_first_is_warmup"] for r in per)
            rows.append(dict(prompt=p, label=label(p), metric=metric, sessions=per,
                             mlxserve=t_all, ours=o_all, mlxserve_mean=mean(t_all), ours_mean=mean(o_all),
                             delta=pct(o_all, t_all), verdict=overall([r["verdict"] for r in per]),
                             delta_with_first=pct(f_all, t_all),
                             verdict_with_first=overall([r["verdict_with_first"] for r in per]),
                             delta_both=pct(f_all, [r["mlxserve_first"] for r in per] + t_all) if full else None,
                             verdict_both=overall([r["verdict_both"] for r in per]) if full else None))
    return rows


def verdict_text(r):
    return {"faster": f"{fmt_pct(r['delta'], 1)} faster", "behind": f"{fmt_pct(r['delta'], 1)} behind",
            "level": f"level ({fmt_pct(r['delta'], 2)})"}[r["verdict"]]


def verdict_cell(row, bold=True):
    if row["verdict"] == "faster":
        s = f"{fmt_pct(row['delta'], 1)} faster"
        return f"**{s}**" if bold else s
    if row["verdict"] == "behind":
        return f"{fmt_pct(row['delta'], 1)} behind"
    return f"level ({fmt_pct(row['delta'], 2)})"


def readme_table(rows, sessions):
    by = {(r["prompt"], r["metric"]): r for r in rows}
    lines = ["| Prompt | Reading (prefill), tok/s<br>MLX-Serve → this repo | | "
             "Writing (decode), tok/s<br>MLX-Serve → this repo | |",
             "|---|---:|:--|---:|:--|"]
    for p in sorted({r["prompt"] for r in rows}):
        rd, wr = by[(p, "reading")], by[(p, "writing")]
        lines.append(f"| {label(p)} ({p:,} tokens) "
                     f"| {fmt(up(rd['mlxserve_mean']))} → {fmt(down(rd['ours_mean']))} | {verdict_cell(rd)} "
                     f"| {fmt(up(wr['mlxserve_mean']))} → {fmt(down(wr['ours_mean']))} | {verdict_cell(wr)} |")
    lines += ["", f"<sub>Same-session runs {session_names(sessions)}. Warm runs, server-reported rates: on both sides the first "
                  f"request at each prompt length is not counted (MLX-Serve's at 4k is its own 16-token warm-up). "
                  f"Rounding never favours this repo. "
                  f"Method and every raw run: [docs/BENCHMARKS.md](docs/BENCHMARKS.md#head-to-head-vs-mlx-serve).</sub>"]
    return "\n".join(lines)


def runs(xs):
    return ", ".join(fmt(x) for x in xs)


def bench_pooled(rows):
    lines = ["| Prompt | | MLX-Serve mean (n, range) | This repo mean (n, range) | Δ | Verdict |",
             "|---|---|---:|---:|---:|---|"]
    for r in rows:
        lines.append(f"| {r['label']} ({r['prompt']:,}) | {r['metric']} "
                     f"| {fmt(up(r['mlxserve_mean']))} ({len(r['mlxserve'])}, {fmt(min(r['mlxserve']))}–{fmt(max(r['mlxserve']))}) "
                     f"| {fmt(down(r['ours_mean']))} ({len(r['ours'])}, {fmt(min(r['ours']))}–{fmt(max(r['ours']))}) "
                     f"| {fmt_pct(r['delta'], 2)} | {verdict_cell(r, bold=False)} |")
    return "\n".join(lines)


def bench_sessions(rows, sessions):
    out = []
    for s in sessions:
        out += [f"**{s['name']}**, started {s['started']} (`results/h2h-20261122/{s['log']}`)", "",
                "| Prompt | | MLX-Serve first request (not counted) | MLX-Serve counted runs | Mean "
                "| Our first request (not counted) | Our counted runs | Mean | Δ | Same-session |",
                "|---|---|---:|---|---:|---:|---|---:|---:|---|"]
        for r in rows:
            for x in r["sessions"]:
                if x["session"] != s["name"]:
                    continue
                ms_first = fmt(x["mlxserve_first"]) + (" (16-token warm-up)" if x["mlxserve_first_is_warmup"] else "")
                out.append(f"| {r['label']} | {r['metric']} | {ms_first} | {runs(x['mlxserve'])} "
                           f"| {fmt(up(mean(x['mlxserve'])))} | {fmt(x['first'])} | {runs(x['ours'])} "
                           f"| {fmt(down(mean(x['ours'])))} | {fmt_pct(x['delta'], 2)} | {x['verdict']} |")
        out.append("")
    return "\n".join(out).rstrip()


def bench_first(rows):
    lines = ["| Prompt | | First requests: MLX-Serve → ours | Warm runs only (headline) | "
             "Our first request counted, MLX-Serve's not | Both first requests counted |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        firsts = ", ".join(("16-token warm-up" if x["mlxserve_first_is_warmup"] else fmt(x["mlxserve_first"]))
                           + f" → {fmt(x['first'])} ({x['session']})" for x in r["sessions"])
        both = "– (MLX-Serve's is its 16-token warm-up)" if r["delta_both"] is None else \
            f"{fmt_pct(r['delta_both'], 2)}, {r['verdict_both']}"
        lines.append(f"| {r['label']} | {r['metric']} | {firsts} | {fmt_pct(r['delta'], 2)}, {r['verdict']} "
                     f"| {fmt_pct(r['delta_with_first'], 2)}, {r['verdict_with_first']} | {both} |")
    return "\n".join(lines)


def bench_history(history):
    lines = ["| Session (Studio 2) | Our stack | Reading 4k | Reading 32k | Writing 4k | Writing 32k | Logs |",
             "|---|---|---:|---:|---:|---:|---|"]
    for h in history:
        d = load_pair(h["mlxserve_log"], h["ours_log"])
        cells = []
        for k, p in ((0, 4078), (0, 32750), (1, 4078), (1, 32750)):
            o, t = [x[k] for x in d[p]["ours"]], [x[k] for x in d[p]["mlxserve"]]
            cells.append(fmt_pct(pct(o, t), 1))
        logs = f"`{h['mlxserve_log']}`" + ("" if h["ours_log"] == h["mlxserve_log"] else f", `{h['ours_log']}`")
        lines.append(f"| {h['name']}, {h['started'][11:]} | {h['ours']} | " + " | ".join(cells) + f" | {logs} |")
    return "\n".join(lines)


def replace_block(path, name, body):
    text = open(path).read()
    start, end = f"<!-- gen:{name} -->", f"<!-- /gen:{name} -->"
    i, j = text.find(start), text.find(end)
    if i < 0 or j < i:
        sys.exit(f"{path}: missing {start} ... {end}")
    open(path, "w").write(text[: i + len(start)] + "\n" + body + "\n" + text[j:])


def h2h_chart(path, metric, title, rows, sessions):
    rows = [r for r in rows if r["metric"] == metric]
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    fig.subplots_adjust(left=0.1, right=0.98, top=0.74, bottom=0.27)
    w, gap = 0.34, 0.03
    top = max(max(r["mlxserve"] + r["ours"]) for r in rows)
    for i, r in enumerate(rows):
        for xc, vals, m, col, rounder in ((i - w / 2 - gap, r["mlxserve"], r["mlxserve_mean"], THEIRS, up),
                                          (i + w / 2 + gap, r["ours"], r["ours_mean"], OURS, down)):
            ax.bar(xc, float(m), w, color=col, zorder=2)
            n = len(vals)
            xs = [xc + (j - (n - 1) / 2) * (w * 0.6 / max(n - 1, 1)) for j in range(n)]
            ax.scatter(xs, [float(v) for v in vals], s=14, color=INK, edgecolor="white", linewidth=0.6, zorder=3)
            ax.text(xc, float(m) * 0.93, fmt(rounder(m)), ha="center", va="top", fontsize=10, color="white",
                    fontweight="bold", zorder=4)
        ax.text(i, float(top) * 1.06, verdict_text(r), ha="center", va="bottom", fontsize=11.5, fontweight="bold",
                color={"faster": OURS, "behind": LOSS, "level": MUTED}[r["verdict"]])
    ax.set_xticks(range(len(rows)), [f"{r['label']} prompt\n({r['prompt']:,} tokens)" for r in rows], fontsize=10.5)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylim(0, float(top) * 1.2)
    ax.set_ylabel("tokens per second (server log)", fontsize=9.5)
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.tick_params(axis="y", labelsize=9, length=0)
    ax.tick_params(axis="x", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=THEIRS), plt.Rectangle((0, 0), 1, 1, color=OURS),
               plt.Line2D([], [], marker="o", ls="", color=INK, markeredgecolor="white", markersize=5)]
    ax.legend(handles, ["MLX-Serve", "This repo", "one run"], loc="upper right", bbox_to_anchor=(1.0, 1.17), ncol=3,
              frameon=False, fontsize=9.5)
    fig.text(0.02, 0.945, title, fontsize=15, fontweight="bold", color=INK)
    fig.text(0.02, 0.86, "M4 Max Mac Studio · same official weights and prompts · same session, MLX-Serve run first\n"
                         f"Sessions {session_names(sessions)} · server-reported rates", fontsize=9, color=MUTED,
             linespacing=1.45)
    firsts = "; ".join(f"{r['label']} " + ", ".join(fmt(x["first"]) for x in r["sessions"]) for r in rows)
    fig.text(0.02, 0.03, "Bars: mean of counted runs (MLX-Serve rounded up, ours down). Faster only where every counted "
                         "run of ours beats\nevery MLX-Serve run in each session. Not counted on either side: the first "
                         "request at each prompt length (MLX-Serve's\nat 4k is its 16-token warm-up; ours, tok/s: "
                         f"{firsts}). Raw runs and method: docs/BENCHMARKS.md", fontsize=7.8, color=MUTED,
             linespacing=1.4)
    save(fig, path)


def fit(text, px, largest):
    """Largest font size (pt, at most `largest`) at which `text` in bold Inter fits in `px` pixels at X_DPI."""
    em = sum(0.64 if c.isdigit() else 0.62 if c.isalpha() or c in "%+−" else 0.3 for c in text)
    return min(largest, px / em * 72 / X_DPI)


def x_panel(ax, rows, title=None):
    w, gap = 0.36, 0.025
    top = max(float(max(r["mlxserve_mean"], r["ours_mean"])) for r in rows)
    ax.set_facecolor(X_BG)
    ax.set_xlim(-0.55, len(rows) - 0.45)
    ax.set_ylim(0, top * 1.38)
    unit = ax.get_window_extent().width / (len(rows) + 0.1)
    for i, r in enumerate(rows):
        for xc, m, col, txt in ((i - w / 2 - gap, r["mlxserve_mean"], X_THEIRS, fmt(up(r["mlxserve_mean"]))),
                                (i + w / 2 + gap, r["ours_mean"], X_OURS, fmt(down(r["ours_mean"])))):
            ax.bar(xc, float(m), w, color=col, zorder=2)
            ax.text(xc, float(m) + top * 0.025, txt, ha="center", va="bottom", color=X_INK, fontweight="bold",
                    fontsize=fit(txt, unit * (w + 2 * gap) * 0.85, 30))
        txt = verdict_text(r)
        ax.text(i, top * 1.2, txt, ha="center", va="bottom", fontweight="bold", fontsize=fit(txt, unit * 0.92, 32),
                color={"faster": X_GAIN, "behind": X_LOSS, "level": "#cbd5e1"}[r["verdict"]])
    ticks = [f"{r['label']} prompt" for r in rows]
    ax.set_xticks(range(len(rows)), ticks, fontsize=min(fit(t, unit * 0.95, 26) for t in ticks), color=X_INK)
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0, pad=14)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(X_MUTED)
    if title:
        ax.set_title(title, loc="left", fontsize=26, color=X_INK, fontweight="bold", pad=18)


def x_chart(path, panels, title, subtitle, sessions):
    """1600x900 PNG for X: one panel per (rows, panel title), bars from zero, values on the bars, no y axis."""
    fig, axes = plt.subplots(1, len(panels), figsize=X_SIZE, dpi=X_DPI, facecolor=X_BG, squeeze=False)
    fig.subplots_adjust(left=0.035, right=0.975, top=0.64 if len(panels) > 1 else 0.7, bottom=0.23, wspace=0.1)
    for ax, (rows, panel_title) in zip(axes[0], panels):
        x_panel(ax, rows, panel_title)
    fig.text(0.035, 0.9, title, fontsize=40, fontweight="bold", color=X_INK)
    fig.text(0.035, 0.835, subtitle, fontsize=20, color=X_MUTED)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (X_THEIRS, X_OURS)]
    fig.legend(handles, ["MLX-Serve", "Ours"], loc="upper right", bbox_to_anchor=(0.975, 0.8), ncol=2,
               frameon=False, fontsize=22, labelcolor=X_INK, handlelength=1.2, columnspacing=1.5)
    fig.text(0.035, 0.03, f"Sessions {session_names(sessions)}.\n"
                          "Server-reported rates, mean of warm runs: on both sides the first request at each prompt "
                          "length is not counted.\nOur means rounded down, MLX-Serve's up. Faster = our slowest "
                          "run beat MLX-Serve's fastest in every session. Method: docs/BENCHMARKS.md",
             fontsize=14, color=X_MUTED, linespacing=1.5)
    fig.savefig(path, format="png", facecolor=X_BG, metadata={"Software": None})
    plt.close(fig)


def x_charts(rows, sessions):
    by = {m: [r for r in rows if r["metric"] == m] for m, _ in METRICS}
    what = "Inkling 2 Mini, same official weights · M4 Max Mac Studio, same session · tokens per second"
    x_chart(os.path.join(IMG, "x-vs-mlxserve.png"),
            [(by["reading"], "Reading a prompt (prefill)"), (by["writing"], "Writing the answer (decode)")],
            "Inkling 2 Mini vs MLX-Serve",
            "Same official weights · M4 Max Mac Studio, same session · tokens per second · draft heads on", sessions)
    x_chart(os.path.join(IMG, "x-reading-vs-mlxserve.png"), [(by["reading"], None)],
            "Reading speed vs MLX-Serve", what, sessions)
    x_chart(os.path.join(IMG, "x-writing-vs-mlxserve.png"), [(by["writing"], None)],
            "Writing speed vs MLX-Serve", what + " · draft heads on", sessions)


def long_chart(path, cfg):
    runs_ = [json.load(open(os.path.join(ROOT, f)))["runs"][0] for f in cfg["long_context_runs"]]
    labels = cfg["long_context_labels"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.3))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.76, bottom=0.2, wspace=0.18)
    for ax, key, name, col, f in ((axes[0], "prompt_tps", "Reading, tok/s", OURS, "{:,.1f}"),
                                  (axes[1], "decode_tps", "Writing, tok/s", "#60a5fa", "{:.2f}")):
        vals = [r[key] for r in runs_]
        ax.bar(range(len(vals)), vals, 0.62, color=col, zorder=2)
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.02, f.format(v), ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_xticks(range(len(vals)), [f"{l}\n{r['peak_mlx_GiB']:.2f} GiB" for l, r in zip(labels, runs_)], fontsize=9)
        ax.set_ylim(0, max(vals) * 1.15)
        ax.set_title(name, loc="left", fontsize=11, color=INK, fontweight="bold")
        ax.yaxis.grid(True, color=GRID, zorder=0)
        ax.tick_params(axis="y", labelsize=8.5, length=0)
        ax.tick_params(axis="x", length=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    found = all(r.get("needle_found") for r in runs_)
    fig.text(0.02, 0.94, "Up to 1M tokens of context on one 64 GB Mac Studio", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.02, 0.875, cfg["long_subtitle"] + ("; hidden fact found at every length" if found else ""),
             fontsize=9, color=MUTED)
    fig.text(0.02, 0.03, "Under each bar: peak MLX GPU memory. Sources: " + ", ".join(
        os.path.basename(f) for f in cfg["long_context_runs"]) + " (results/gates/)", fontsize=7.5, color=MUTED)
    save(fig, path)


def save(fig, path):
    fig.savefig(path, format="svg", metadata={"Date": None})
    plt.close(fig)


def as_json(x):
    if isinstance(x, Decimal):
        return float(x)
    if isinstance(x, dict):
        return {k: as_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [as_json(v) for v in x]
    return x


def main():
    os.makedirs(IMG, exist_ok=True)
    cfg = json.load(open(os.path.join(H2H, "charts.json")))
    sessions = [dict(s, data=load_pair(s["log"], s["log"])) for s in cfg["h2h_sessions"]]
    rows = analyse(sessions)
    for metric, title in METRICS:
        h2h_chart(os.path.join(IMG, f"{metric}-vs-mlxserve.svg"), metric, title, rows, sessions)
    x_charts(rows, sessions)
    long_chart(os.path.join(IMG, "long-context.svg"), cfg)
    replace_block(os.path.join(ROOT, "README.md"), "h2h-table", readme_table(rows, sessions))
    bench = os.path.join(ROOT, "docs", "BENCHMARKS.md")
    replace_block(bench, "h2h-pooled", bench_pooled(rows))
    replace_block(bench, "h2h-sessions", bench_sessions(rows, sessions))
    replace_block(bench, "h2h-first", bench_first(rows))
    replace_block(bench, "h2h-history", bench_history(cfg["h2h_history"]))
    summary = [{k: v for k, v in r.items()} for r in rows]
    json.dump(as_json(summary), open(os.path.join(H2H, "summary.json"), "w"), indent=1)
    for r in rows:
        print(f"{r['label']:>5} {r['metric']:<8} MLX-Serve {fmt(up(r['mlxserve_mean']))}  ours {fmt(down(r['ours_mean']))}  "
              f"{fmt_pct(r['delta'], 2)}  {r['verdict']}  (our first counted: {fmt_pct(r['delta_with_first'], 2)}, "
              f"{r['verdict_with_first']}; both counted: "
              + ("-" if r["delta_both"] is None else f"{fmt_pct(r['delta_both'], 2)}, {r['verdict_both']}") + ")")


if __name__ == "__main__":
    main()
