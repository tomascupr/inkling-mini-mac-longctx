# X post draft (not posted)

For Tomas Cupr's account. Post once the 131k row is in the README. Every claim here is a README claim; the sources are
in [BENCHMARKS.md](BENCHMARKS.md#the-claim). The repo is private: replace `[REPO_LINK]` before posting (X counts any
link as 23 characters).

Images: `python scripts/charts.py --png` writes `docs/img/reading-vs-mlxserve.png` and
`docs/img/writing-vs-mlxserve.png`. Attach both to the main post.

## Main post

```text
Official Inkling 2 Mini weights on an M4 Max Mac Studio. No requantizing, no FP8 KV, no approximate attention.

Same session vs MLX-Serve:
writing 5.4% faster at 4k, 4.3% at 32k[W131]
reading 3.2% faster at 32k, level at 4k[R131]

[REPO_LINK]
```

242/280 characters with the 131k slots empty.

### The 131k slots

Fill `[W131]` and `[R131]` from the README's 131k row after `python scripts/charts.py` has generated it. Use the
verdict as written and the README's figures unchanged (`W` and `R` below). Say faster only where the README does.

| README verdict at 131k | `[W131]` (writing) | `[R131]` (reading) |
|---|---|---|
| faster | `, W% at 131k` | `; R% faster at 131k` |
| level | `, level at 131k` | ` and 131k` |
| behind | `; W% behind at 131k` | `; R% behind at 131k` |

With percentages like `12.3%` every combination fits in 280 (at most 279) except writing behind with reading faster
or behind (286): in those cases leave both slots empty and give the 131k results a thread post of their own. A 262k row goes in only if it is in the README, under
the same rules, and most likely in the thread rather than the main post.

## Thread (optional)

**2/**

```text
The speed is all engineering:
- Metal kernels that add in exactly MLX's order, so results are bit for bit the unfused code's, in fewer GPU passes
- a server that takes the compiled decode path and checks draft guesses exactly
- a draft head that now actually gets the prompt
```

**3/**

```text
Quality: 8/8 greedy answers identical to the reference, 520/520 teacher-forced top-1 tokens on our gate.

The gate predates the last two kernel changes (bitwise-equal in their own tests) and the server settings. Re-running it on the final stack is still to do.
```

**4/**

```text
Method: warm runs, each server's own log, rounding never in our favour. Our first request at a new prompt length is slower; the repo shows it.

Built on jundot's kernel (mlx-lm#2086), betizzel's mlx-vlm fork, mlx-vlm and MLX. MLX-Serve by ddalcu is a strong baseline. Thanks.
```
