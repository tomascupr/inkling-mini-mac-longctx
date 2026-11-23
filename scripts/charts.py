"""Draw the charts and write the result tables straight from the measured logs (needs matplotlib).

Head-to-head vs MLX-Serve: every session listed under "h2h_sessions" in results/h2h-20261122/charts.json. A session
log holds MLX-Serve's own log lines ('<- P+G tokens streamed [prefill: X tok/s, decode: Y tok/s]') and this repo's
server log lines ('LOG prompt_tokens=P ... prefill=X tok/s decode=Y tok/s'), same Mac, same session, MLX-Serve first.
Each prompt length found in the logs is one row of the tables and one group in the charts, so a new length is added
by listing its session log in charts.json and running this script again.

Counting: on both sides, every request except the first one at each prompt length in a session (reported
separately). MLX-Serve's first 4k request is its own 16-token warm-up.
If dropping the first request leaves one side with a single run, and counting every run at that length yields a
smaller (more conservative) delta, the headline, README table and charts use the all-runs figures; both variants
are in docs/BENCHMARKS.md.
If MLX-Serve completed only one full request at a length, that sole run is counted and our first request is still
dropped; that is the headline when it is more conservative than counting our first too.
Rounding never favours this repo: its means are rounded down, MLX-Serve's up, differences (from unrounded means) down.
Verdict: "faster" only if, in every session, our slowest counted run beats MLX-Serve's fastest one; "behind" if
MLX-Serve's slowest beats our fastest in every session; otherwise "level".

Writes docs/img/{reading,writing}-vs-mlxserve.svg, docs/img/long-context.svg, the 1600x900 PNGs for X
(docs/img/x-vs-mlxserve.png with both metrics, docs/img/x-{reading,writing}-vs-mlxserve.png),
results/h2h-20261122/summary.json and the generated blocks (between <!-- gen:NAME --> and <!-- /gen:NAME -->) in
README.md and docs/BENCHMARKS.md. The charts themselves carry values and deltas only; counting, rounding and raw
runs live in docs/BENCHMARKS.md.
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
METRICS = (("reading", "Reading, tok/s"), ("writing", "Writing, tok/s"))
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
        if len(us[p]) < 2:
            print(f"warning: {ours_log}: {len(them[p])} MLX-Serve and {len(us[p])} own request(s) at {p} tokens, "
                  "nothing left on one side after its first request")
            continue
        if len(them[p]) == 1:
            # Sole MLX-Serve request at this length is counted (cannot drop first).
            out[p] = {"mlxserve_first": them[p][0], "mlxserve": [them[p][0][:2]],
                      "first": us[p][0], "ours": us[p][1:], "mlxserve_keep_first": True}
        elif len(them[p]) >= 2:
            out[p] = {"mlxserve_first": them[p][0], "mlxserve": [x[:2] for x in them[p][1:]],
                      "first": us[p][0], "ours": us[p][1:], "mlxserve_keep_first": False}
        else:
            print(f"warning: {ours_log}: {len(them[p])} MLX-Serve and {len(us[p])} own request(s) at {p} tokens, "
                  "nothing left on one side after its first request")
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
                keep = d.get("mlxserve_keep_first", False)
                if keep:
                    both = (t, with_first)
                elif warmup:
                    both = None
                else:
                    both = ([ms_first] + t, with_first)
                per.append(dict(session=s["name"], started=s["started"], mlxserve=t, ours=o, first=d["first"][k],
                                mlxserve_first=ms_first, mlxserve_first_is_warmup=warmup,
                                mlxserve_keep_first=keep,
                                delta=pct(o, t), verdict=session_verdict(o, t),
                                delta_with_first=pct(with_first, t), verdict_with_first=session_verdict(with_first, t),
                                delta_both=both and pct(both[1], both[0]),
                                verdict_both=both and session_verdict(both[1], both[0])))
            t_all = [x for r in per for x in r["mlxserve"]]
            o_all = [x for r in per for x in r["ours"]]
            f_all = [r["first"] for r in per] + o_all
            full = all(not r["mlxserve_first_is_warmup"] for r in per)
            t_both = []
            for r in per:
                if r["mlxserve_keep_first"] or r["mlxserve_first_is_warmup"]:
                    t_both.extend(r["mlxserve"])
                else:
                    t_both.append(r["mlxserve_first"])
                    t_both.extend(r["mlxserve"])
            row = dict(prompt=p, label=label(p), metric=metric, sessions=per,
                       mlxserve=t_all, ours=o_all, mlxserve_mean=mean(t_all), ours_mean=mean(o_all),
                       delta=pct(o_all, t_all), verdict=overall([r["verdict"] for r in per]),
                       delta_with_first=pct(f_all, t_all),
                       verdict_with_first=overall([r["verdict_with_first"] for r in per]),
                       delta_both=pct(f_all, t_both) if full else None,
                       verdict_both=overall([r["verdict_both"] for r in per]) if full else None)
            # Headline is the drop-first mean, unless that leaves one side with a single run
            # and counting every run is more conservative (a smaller delta).
            # Only switch when the 1-decimal headline would actually fall (262k writing,
            # not 262k reading, where both variants already display as +6.2%).
            if (full and (len(t_all) == 1 or len(o_all) == 1)
                    and down(pct(f_all, t_both), 1) < down(row["delta"], 1)):
                row.update(headline_from="all_runs", headline_mlxserve=t_both, headline_ours=f_all,
                           headline_mlxserve_mean=mean(t_both), headline_ours_mean=mean(f_all),
                           headline_delta=pct(f_all, t_both),
                           headline_verdict=overall([r["verdict_both"] for r in per]))
            else:
                row.update(headline_from="warm", headline_mlxserve=t_all, headline_ours=o_all,
                           headline_mlxserve_mean=row["mlxserve_mean"], headline_ours_mean=row["ours_mean"],
                           headline_delta=row["delta"], headline_verdict=row["verdict"])
            rows.append(row)
    return rows


def verdict_text(r):
    return {"faster": f"{fmt_pct(r['delta'], 1)} faster", "behind": f"{fmt_pct(r['delta'], 1)} behind",
            "level": f"level ({fmt_pct(r['delta'], 2)})"}[r["verdict"]]


def chart_delta(r):
    """Short delta for the charts: +5.4% when we win, 'level' otherwise, no method text."""
    if r["headline_verdict"] == "level":
        return "level"
    if r["headline_verdict"] == "faster":
        return fmt_pct(r["headline_delta"], 1)
    return f"{fmt_pct(r['headline_delta'], 1)} behind"


def verdict_cell(row, bold=True):
    if row["verdict"] == "faster":
        s = f"{fmt_pct(row['delta'], 1)} faster"
        return f"**{s}**" if bold else s
    if row["verdict"] == "behind":
        return f"{fmt_pct(row['delta'], 1)} behind"
    return f"level ({fmt_pct(row['delta'], 2)})"


def readme_delta(row):
    if row["headline_verdict"] == "faster":
        return f"**{fmt_pct(row['headline_delta'], 1)}**"
    if row["headline_verdict"] == "behind":
        return fmt_pct(row["headline_delta"], 1)
    return "level"


def readme_table(rows, sessions):
    by = {(r["prompt"], r["metric"]): r for r in rows}
    lines = ["| Prompt | Reading, tok/s<br>MLX-Serve → ours | Writing, tok/s<br>MLX-Serve → ours |",
             "|---|---:|---:|"]
    for p in sorted({r["prompt"] for r in rows}):
        rd, wr = by[(p, "reading")], by[(p, "writing")]
        lines.append(f"| {label(p)} ({p:,} tokens) "
                     f"| {fmt(up(rd['headline_mlxserve_mean']))} → {fmt(down(rd['headline_ours_mean']))}, {readme_delta(rd)} "
                     f"| {fmt(up(wr['headline_mlxserve_mean']))} → {fmt(down(wr['headline_ours_mean']))}, {readme_delta(wr)} |")
    lines += ["", "<sub>Same-session runs on an M4 Max Mac Studio. Method and every raw run: "
              "[docs/BENCHMARKS.md](docs/BENCHMARKS.md#head-to-head-vs-mlx-serve).</sub>"]
    return "\n".join(lines)


def runs(xs):
    return ", ".join(fmt(x) for x in xs)


def bench_pooled(rows):
    lines = ["| Prompt | | MLX-Serve mean (n, range) | This repo mean (n, range) | Δ | Verdict |",
             "|---|---|---:|---:|---:|---|"]
    for r in rows:
        t, o = r["headline_mlxserve"], r["headline_ours"]
        note = " all runs" if r["headline_from"] == "all_runs" else ""
        lines.append(f"| {r['label']} ({r['prompt']:,}) | {r['metric']} "
                     f"| {fmt(up(r['headline_mlxserve_mean']))} ({len(t)}, {fmt(min(t))}–{fmt(max(t))}){note} "
                     f"| {fmt(down(r['headline_ours_mean']))} ({len(o)}, {fmt(min(o))}–{fmt(max(o))}){note} "
                     f"| {fmt_pct(r['headline_delta'], 2)} | {verdict_cell({'delta': r['headline_delta'], 'verdict': r['headline_verdict']}, bold=False)} |")
    extras = [r for r in rows if r["headline_from"] == "all_runs"]
    if extras:
        lines.append("")
        for r in extras:
            lines.append(
                f"{r['label']} {r['metric']}: the headline is the mean of every run, not the drop-first mean "
                f"({fmt(up(r['mlxserve_mean']))} → {fmt(down(r['ours_mean']))}, {fmt_pct(r['delta'], 2)}). "
                f"Dropping the first request leaves one side with a single run, and all-runs is more conservative.")
    keeps = [r for r in rows if any(x.get("mlxserve_keep_first") for x in r["sessions"])]
    if keeps:
        if not extras:
            lines.append("")
        for r in keeps:
            lines.append(
                f"{r['label']} {r['metric']}: MLX-Serve completed one request; that sole run is counted. "
                f"The headline drops our first request "
                f"({fmt(up(r['headline_mlxserve_mean']))} → {fmt(down(r['headline_ours_mean']))}, "
                f"{fmt_pct(r['headline_delta'], 2)}). Counting our first too would be "
                f"{fmt_pct(r['delta_with_first'], 2)}.")
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
                if x["mlxserve_first_is_warmup"]:
                    ms_first = fmt(x["mlxserve_first"]) + " (16-token warm-up)"
                elif x.get("mlxserve_keep_first"):
                    ms_first = fmt(x["mlxserve_first"]) + " (only run, counted)"
                else:
                    ms_first = fmt(x["mlxserve_first"])
                out.append(f"| {r['label']} | {r['metric']} | {ms_first} | {runs(x['mlxserve'])} "
                           f"| {fmt(up(mean(x['mlxserve'])))} | {fmt(x['first'])} | {runs(x['ours'])} "
                           f"| {fmt(down(mean(x['ours'])))} | {fmt_pct(x['delta'], 2)} | {x['verdict']} |")
        out.append("")
    return "\n".join(out).rstrip()


def bench_first(rows):
    lines = ["| Prompt | | First requests: MLX-Serve → ours | Drop first request | "
             "Our first request counted, MLX-Serve's not | All runs counted | Headline |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        def _first_ms(x):
            if x["mlxserve_first_is_warmup"]:
                return "16-token warm-up"
            lab = fmt(x["mlxserve_first"])
            return lab + " (only run, counted)" if x.get("mlxserve_keep_first") else lab
        firsts = ", ".join(_first_ms(x) + f" → {fmt(x['first'])} ({x['session']})" for x in r["sessions"])
        both = "– (MLX-Serve's is its 16-token warm-up)" if r["delta_both"] is None else \
            f"{fmt_pct(r['delta_both'], 2)}, {r['verdict_both']}"
        headline = "all runs" if r["headline_from"] == "all_runs" else "drop first"
        lines.append(f"| {r['label']} | {r['metric']} | {firsts} | {fmt_pct(r['delta'], 2)}, {r['verdict']} "
                     f"| {fmt_pct(r['delta_with_first'], 2)}, {r['verdict_with_first']} | {both} | {headline} |")
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
    fig, ax = plt.subplots(figsize=(8.8, 4.35))
    fig.subplots_adjust(left=0.04, right=0.98, top=0.70, bottom=0.14)
    w, gap = 0.32, 0.04
    top = max(float(max(r["headline_mlxserve_mean"], r["headline_ours_mean"])) for r in rows)
    for i, r in enumerate(rows):
        pair = ((i - w / 2 - gap, r["headline_mlxserve_mean"], THEIRS, fmt(up(r["headline_mlxserve_mean"]))),
                (i + w / 2 + gap, r["headline_ours_mean"], OURS, fmt(down(r["headline_ours_mean"]))))
        # figsize 8.8 in at 100 dpi, axes ~0.94 of the width
        unit = 8.8 * X_DPI * 0.94 / (len(rows) + 0.2)
        close = labels_overlap(pair[0][3], pair[1][3], 11, pair[0][0], pair[1][0], unit)
        for j, (xc, m, col, txt) in enumerate(pair):
            ax.bar(xc, float(m), w, color=col, zorder=2)
            lift = top * (0.10 if close and j == 1 else 0.025)
            ax.text(xc, float(m) + lift, txt, ha="center", va="bottom", fontsize=11,
                    color=INK, fontweight="bold", zorder=4)
        d = chart_delta(r)
        ax.text(i, top * (1.18 if close else 1.14), d, ha="center", va="bottom", fontsize=13, fontweight="bold",
                color={"faster": OURS, "behind": LOSS, "level": MUTED}[r["headline_verdict"]])
    ax.set_xticks(range(len(rows)), [r["label"] for r in rows], fontsize=13)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylim(0, top * 1.36)
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0, pad=8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    handles = [plt.Rectangle((0, 0), 1, 1, color=THEIRS), plt.Rectangle((0, 0), 1, 1, color=OURS)]
    ax.legend(handles, ["MLX-Serve", "Ours"], loc="upper right", bbox_to_anchor=(1.0, 1.24), ncol=2,
              frameon=False, fontsize=11)
    fig.text(0.05, 0.935, title, fontsize=16, fontweight="bold", color=INK)
    fig.text(0.05, 0.865, "M4 Max Mac Studio · same session vs MLX-Serve", fontsize=10.5, color=MUTED)
    save(fig, path)


def fit(text, px, largest):
    """Largest font size (pt, at most `largest`) at which `text` in bold Inter fits in `px` pixels at X_DPI."""
    em = sum(0.64 if c.isdigit() else 0.62 if c.isalpha() or c in "%+−" else 0.3 for c in text)
    return min(largest, px / em * 72 / X_DPI)


def text_em(text):
    return sum(0.64 if c.isdigit() else 0.62 if c.isalpha() or c in "%+−" else 0.3 for c in text)


def labels_overlap(txt0, txt1, fs, xc0, xc1, unit):
    """True if two centred labels at fontsize `fs` would overlap, given x-units in `unit` pixels."""
    return (text_em(txt0) + text_em(txt1)) * fs * X_DPI / 72 / 2 >= abs(xc1 - xc0) * unit * 0.92


def x_val(x):
    """Value label for the X renders: no thousands separator, so 4k reading stays larger."""
    return f"{x:.1f}"


def x_panel(ax, rows, title=None):
    w, gap = 0.30, 0.05
    top = max(float(max(r["headline_mlxserve_mean"], r["headline_ours_mean"])) for r in rows)
    ax.set_facecolor(X_BG)
    ax.set_xlim(-0.58, len(rows) - 0.42)
    ax.set_ylim(0, top * 1.42)
    unit = ax.get_window_extent().width / (len(rows) + 0.16)
    _closes = []
    for i, r in enumerate(rows):
        pair = ((i - w / 2 - gap, r["headline_mlxserve_mean"], X_THEIRS, x_val(up(r["headline_mlxserve_mean"]))),
                (i + w / 2 + gap, r["headline_ours_mean"], X_OURS, x_val(down(r["headline_ours_mean"]))))
        # Use a large side-by-side size when it stays readable; otherwise stagger
        # and keep the larger type (needed for 4k reading on the combined X chart).
        ems = text_em(pair[0][3]) + text_em(pair[1][3])
        dist = abs(pair[1][0] - pair[0][0]) * unit * 0.92
        fs_side = min(22, dist * 2 * 72 / X_DPI / ems) if ems else 22
        if fs_side >= 14:
            fs, close = fs_side, False
        else:
            fs, close = min(fit(txt, unit * 0.50, 22) for *_, txt in pair), True
        for j, (xc, m, col, txt) in enumerate(pair):
            ax.bar(xc, float(m), w, color=col, zorder=2)
            lift = top * (0.12 if close and j == 1 else 0.03)
            ax.text(xc, float(m) + lift, txt, ha="center", va="bottom", color=X_INK, fontweight="bold",
                    fontsize=fs)
        # Delta labels are drawn once below, at a shared size that keeps neighbours apart.
        _closes.append(close)
    deltas = [chart_delta(r) for r in rows]
    d_fs = min((fit(txt, unit * 0.80, 30) for txt in deltas), default=30)
    for a, b in zip(deltas, deltas[1:]):
        ems = text_em(a) + text_em(b)
        if ems:
            d_fs = min(d_fs, unit * 0.88 * 2 * 72 / X_DPI / ems)
    for i, r in enumerate(rows):
        ax.text(i, top * (1.22 if _closes[i] else 1.16), deltas[i], ha="center", va="bottom",
                fontweight="bold", fontsize=d_fs,
                color={"faster": X_GAIN, "behind": X_LOSS, "level": "#cbd5e1"}[r["headline_verdict"]])
    ticks = [r["label"] for r in rows]
    ax.set_xticks(range(len(rows)), ticks, fontsize=min(fit(t, unit * 0.7, 26) for t in ticks), color=X_INK)
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0, pad=14)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(X_MUTED)
    if title:
        ax.set_title(title, loc="left", fontsize=28, color=X_INK, fontweight="bold", pad=22)


def x_chart(path, panels, title, subtitle=None):
    """1600x900 PNG for X: one panel per (rows, panel title), values and deltas only, no footnotes."""
    fig, axes = plt.subplots(1, len(panels), figsize=X_SIZE, dpi=X_DPI, facecolor=X_BG, squeeze=False)
    fig.subplots_adjust(left=0.04, right=0.975, top=0.66 if subtitle else 0.72, bottom=0.14, wspace=0.13)
    fig.canvas.draw()
    for ax, (rows, panel_title) in zip(axes[0], panels):
        x_panel(ax, rows, panel_title)
    fig.text(0.04, 0.915, title, fontsize=36, fontweight="bold", color=X_INK)
    if subtitle:
        fig.text(0.04, 0.855, subtitle, fontsize=18, color=X_MUTED)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (X_THEIRS, X_OURS)]
    fig.legend(handles, ["MLX-Serve", "Ours"], loc="upper right", bbox_to_anchor=(0.975, 0.97), ncol=2,
               frameon=False, fontsize=20, labelcolor=X_INK, handlelength=1.2, columnspacing=1.4)
    fig.text(0.975, 0.035, "github.com/tomascupr/inkling-mini-mac-longctx", ha="right", va="center",
             fontsize=11, color=X_MUTED)
    fig.savefig(path, format="png", facecolor=X_BG, metadata={"Software": None})
    plt.close(fig)


def x_charts(rows, sessions):
    by = {m: [r for r in rows if r["metric"] == m] for m, _ in METRICS}
    x_chart(os.path.join(IMG, "x-vs-mlxserve.png"),
            [(by["writing"], "Writing, tok/s"), (by["reading"], "Reading, tok/s")],
            "Inkling 2 Mini vs MLX-Serve", "M4 Max Mac Studio")
    x_chart(os.path.join(IMG, "x-writing-vs-mlxserve.png"), [(by["writing"], None)],
            "Writing, tok/s", "Inkling 2 Mini vs MLX-Serve  ·  M4 Max Mac Studio")
    x_chart(os.path.join(IMG, "x-reading-vs-mlxserve.png"), [(by["reading"], None)],
            "Reading, tok/s", "Inkling 2 Mini vs MLX-Serve  ·  M4 Max Mac Studio")


def long_chart(path, cfg):
    runs_ = [json.load(open(os.path.join(ROOT, f)))["runs"][0] for f in cfg["long_context_runs"]]
    labels = cfg["long_context_labels"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.76, bottom=0.18, wspace=0.16)
    for ax, key, name, col, f in ((axes[0], "prompt_tps", "Reading, tok/s", OURS, "{:,.1f}"),
                                  (axes[1], "decode_tps", "Writing, tok/s", "#60a5fa", "{:.2f}")):
        vals = [r[key] for r in runs_]
        ax.bar(range(len(vals)), vals, 0.62, color=col, zorder=2)
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.03, f.format(v), ha="center", va="bottom", fontsize=9, color=INK,
                    fontweight="bold")
        ax.set_xticks(range(len(vals)), [f"{l}\n{r['peak_mlx_GiB']:.2f} GiB" for l, r in zip(labels, runs_)], fontsize=9)
        ax.set_ylim(0, max(vals) * 1.18)
        ax.set_title(name, loc="left", fontsize=12, color=INK, fontweight="bold")
        ax.tick_params(axis="y", labelsize=8.5, length=0)
        ax.tick_params(axis="x", length=0)
        ax.yaxis.grid(True, color=GRID, zorder=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    fig.text(0.05, 0.93, "Up to 1M tokens of context on one 64 GB Mac Studio", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.05, 0.86, "Studio 1 · command line · draft head off", fontsize=10, color=MUTED)
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
        extra = ""
        if r["headline_from"] != "warm":
            extra = (f"  HEADLINE all-runs {fmt(up(r['headline_mlxserve_mean']))} → "
                     f"{fmt(down(r['headline_ours_mean']))}  {fmt_pct(r['headline_delta'], 2)}  {r['headline_verdict']}")
        print(f"{r['label']:>5} {r['metric']:<8} MLX-Serve {fmt(up(r['mlxserve_mean']))}  ours {fmt(down(r['ours_mean']))}  "
              f"{fmt_pct(r['delta'], 2)}  {r['verdict']}  (our first counted: {fmt_pct(r['delta_with_first'], 2)}, "
              f"{r['verdict_with_first']}; both counted: "
              + ("-" if r["delta_both"] is None else f"{fmt_pct(r['delta_both'], 2)}, {r['verdict_both']}")
              + extra + ")")


if __name__ == "__main__":
    main()
