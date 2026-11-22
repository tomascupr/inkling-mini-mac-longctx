#!/usr/bin/env python3
"""A long document, a question, then a follow-up question, against the running server.

    python followup.py ask     turn 1 (document + question), then turn 2 (the follow-up)
    python followup.py again   turn 2 once more, as sent by `ask` (after a server restart)

Prints one JSON line per turn. `cached_tokens` is what the server restored from
its prefix cache, `first_token_s` the wait for the first generated token: these
are the measurements. Turn 2 should show cached_tokens = turn 1's prompt_tokens
minus the checkpoint guard (32). `again` after a restart, with the disk tier on,
should show its own prompt_tokens minus 32: turn 2's checkpoint, read from disk.

Environment: URL, MODEL (the model directory, as the server knows it), CORPUS
(a text file longer than the context), CTX (document tokens, default 131072),
EFFORT (reasoning_effort, default none), MAX_TOKENS (default 300), STATE.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(os.environ.get("ROOT", Path.home() / "inkling-mini"))
URL = os.environ.get("URL", "http://127.0.0.1:52417")
MODEL = os.environ.get("MODEL", str(ROOT / "models/Inkling-2-Mini"))
CORPUS = Path(os.environ.get("CORPUS", ROOT / "corpus.txt"))
CTX = int(os.environ.get("CTX", 131072))
STATE = Path(os.environ.get("STATE", "followup-state.json"))
NEEDLE = "VIOLET-ANCHOR-4729"
NOTE = f"\n\nAn archivist's note: the catalogue code of the Prague observatory archive box is {NEEDLE}.\n\n"
QUESTION = "\n\nQuestion: what is the catalogue code of the Prague observatory archive box? Reply with the code only."
FOLLOW_UP = "Which city's observatory does that archive box belong to? Reply with the city only."


def document():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    ids = tokenizer.encode(CORPUS.read_text(), add_special_tokens=False)
    if len(ids) < CTX:
        sys.exit(f"{CORPUS} has {len(ids)} tokens, fewer than CTX={CTX}")
    text = tokenizer.decode(ids[: CTX - 100])
    middle = text.find("\n\n", len(text) // 2)
    middle = middle if middle > 0 else len(text) // 2
    return text[:middle] + NOTE + text[middle:] + QUESTION


def ask(turn, messages):
    body = {
        "model": MODEL,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0,
        "max_tokens": int(os.environ.get("MAX_TOKENS", 300)),
        # The same effort every turn: the template writes it at the start of the
        # prompt, so a change is a different prompt and nothing is restored.
        "reasoning_effort": os.environ.get("EFFORT", "none"),
        # No server-side summary of older turns: it would replace the document.
        "context_management": [],
    }
    request = urllib.request.Request(
        URL + "/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    start, first, content, usage = time.perf_counter(), None, "", {}
    try:
        response = urllib.request.urlopen(request, timeout=7200)
    except urllib.error.HTTPError as error:
        sys.exit(f"turn {turn}: HTTP {error.code} {error.read().decode()[:500]}")
    with response:
        for line in response:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("error"):
                sys.exit(f"turn {turn}: {event['error']}")
            usage = event.get("usage") or usage
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                if first is None and (delta.get("content") or delta.get("reasoning_content")):
                    first = time.perf_counter() - start
                content += delta.get("content") or ""
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    print(
        json.dumps(
            {
                "turn": turn,
                "prompt_tokens": usage.get("prompt_tokens"),
                "cached_tokens": cached,
                "completion_tokens": usage.get("completion_tokens"),
                "first_token_s": first and round(first, 2),
                "total_s": round(time.perf_counter() - start, 2),
                "content": content.strip()[:200],
            }
        ),
        flush=True,
    )
    return content


if sys.argv[1:] == ["ask"]:
    messages = [{"role": "user", "content": document()}]
    answer = ask(1, messages)
    messages += [{"role": "assistant", "content": answer}, {"role": "user", "content": FOLLOW_UP}]
    STATE.write_text(json.dumps(messages))
    ask(2, messages)
elif sys.argv[1:] == ["again"]:
    ask("2 again", json.loads(STATE.read_text()))
else:
    sys.exit(__doc__)
