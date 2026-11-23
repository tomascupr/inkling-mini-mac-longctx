#!/usr/bin/env python3
"""Quality evals against the local Inkling OpenAI-compatible server.

Fixed settings (also written to meta.json):
  seed=20261123
  temperature=0.0, top_p=1.0, max_tokens=4096
  reasoning_effort=high (0.9, the model default; thinking on)
  n: GSM8K 200 / MMLU-Pro 200 / GPQA Diamond 198 (full) / AIME 2025 30 (full)
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

SEED = 20261123
N_GSM8K = 200
N_MMLU = 200
URL = os.environ.get("EVAL_URL", "http://127.0.0.1:8080/v1/chat/completions")
MODEL = os.environ.get(
    "EVAL_MODEL", "/Users/tomascupr/inkling-mini/models/Inkling-2-Mini"
)
TEMPERATURE = 0.0
TOP_P = 1.0
MAX_TOKENS = 4096
REASONING_EFFORT = "high"
HTTP_TIMEOUT = 900
OUTDIR = Path(os.environ.get("EVAL_OUT", ".")).resolve()
CACHE = Path(os.environ.get("EVAL_CACHE", str(OUTDIR / "cache"))).resolve()

CET = timezone(timedelta(hours=1), name="CET")


def now_cet() -> str:
    return datetime.now(CET).strftime("%Y-%m-%d %H:%M:%S CET")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def append_jsonl(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def extract_boxed(text: str) -> str | None:
    if not text:
        return None
    matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if matches:
        return matches[-1].strip()
    # nested or leftover
    matches = re.findall(r"\\boxed\{(.+?)\}", text, re.S)
    if matches:
        return re.sub(r"\s+", " ", matches[-1]).strip()
    return None


def last_number(text: str) -> str | None:
    if not text:
        return None
    nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text.replace(",", ""))
    # the replace above already dropped commas inside; redo on original commas
    nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if not nums:
        return None
    return nums[-1].replace(",", "")


def gsm8k_gold(answer: str) -> str:
    if "####" in answer:
        return answer.split("####")[-1].strip().replace(",", "")
    return (last_number(answer) or "").strip()


def extract_gsm8k(text: str) -> str | None:
    if not text:
        return None
    boxed = extract_boxed(text)
    if boxed:
        n = last_number(boxed)
        if n is not None:
            return n
    m = re.findall(r"####\s*([-\d][-\d,.]*)", text)
    if m:
        return m[-1].replace(",", "").strip()
    m = re.findall(r"(?:final answer|the answer is)\s*[:\s]*([-\d][-\d,.]*)", text, re.I)
    if m:
        return m[-1].replace(",", "").strip()
    return last_number(text)


def extract_letter(text: str, letters: str = "ABCDEFGHIJ") -> str | None:
    if not text:
        return None
    boxed = extract_boxed(text)
    if boxed:
        m = re.search(rf"\b([{letters}])\b", boxed.strip(), re.I)
        if m:
            return m.group(1).upper()
        if boxed.strip().upper() in letters:
            return boxed.strip().upper()
    patterns = [
        rf"(?:final answer|answer|ans)\s*[:is]*\s*\**([{letters}])\b",
        rf"(?:the correct (?:choice|option) is)\s*([{letters}])\b",
        rf"^\s*([{letters}])\s*[\).:]",
    ]
    for pat in patterns:
        ms = re.findall(pat, text, re.I | re.M)
        if ms:
            return ms[-1].upper()
    # last standalone letter line
    ms = re.findall(rf"^\s*([{letters}])\s*$", text, re.I | re.M)
    if ms:
        return ms[-1].upper()
    return None


def extract_aime(text: str) -> str | None:
    if not text:
        return None
    boxed = extract_boxed(text)
    if boxed:
        n = last_number(boxed)
        if n is not None:
            try:
                v = int(float(n))
                if 0 <= v <= 999:
                    return str(v)
            except ValueError:
                pass
    # prefer "answer is NNN"
    ms = re.findall(r"(?:final answer|the answer is|answer)\s*[:\s]*([0-9]{1,3})\b", text, re.I)
    if ms:
        return str(int(ms[-1]))
    n = last_number(text)
    if n is None:
        return None
    try:
        v = int(float(n))
    except ValueError:
        return None
    if 0 <= v <= 999:
        return str(v)
    return None


def numbers_equal(a: str | None, b: str | None) -> bool:
    if a is None or b is None or a == "" or b == "":
        return False
    try:
        return float(a) == float(b)
    except ValueError:
        return a.strip() == b.strip()


def chat(prompt: str) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "reasoning_effort": REASONING_EFFORT,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    ttft = None
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage = None
    finish = None
    err = None
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line or line.startswith(":") or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if ttft is None:
                    ttft = time.perf_counter() - t0
                ch = (obj.get("choices") or [{}])[0]
                delta = ch.get("delta") or {}
                if delta.get("content"):
                    content_parts.append(delta["content"])
                if delta.get("reasoning_content") or delta.get("reasoning"):
                    reasoning_parts.append(delta.get("reasoning_content") or delta.get("reasoning"))
                if ch.get("finish_reason"):
                    finish = ch.get("finish_reason")
                if obj.get("usage"):
                    usage = obj["usage"]
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        try:
            if hasattr(e, "read"):
                err += " body=" + e.read().decode("utf-8", "replace")[:2000]
        except Exception:
            pass
    elapsed = time.perf_counter() - t0
    return {
        "elapsed_s": elapsed,
        "ttft_s": ttft,
        "content": "".join(content_parts),
        "reasoning_content": "".join(reasoning_parts),
        "finish_reason": finish,
        "usage": usage,
        "error": err,
    }


def prepare_datasets() -> dict[str, list[dict]]:
    from datasets import load_dataset

    CACHE.mkdir(parents=True, exist_ok=True)
    out: dict[str, list[dict]] = {}

    # GSM8K test
    gs = load_dataset("openai/gsm8k", "main", split="test")
    rows = []
    for i, ex in enumerate(gs):
        rows.append(
            {
                "id": f"gsm8k-{i}",
                "src_index": i,
                "question": ex["question"],
                "gold": gsm8k_gold(ex["answer"]),
                "gold_raw": ex["answer"],
            }
        )
    rng = random.Random(SEED)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    pick = sorted(idx[:N_GSM8K])  # keep stable file order by original index after shuffle-select
    # "fixed random subset": shuffle with seed, take first n, then run in that shuffled order
    order = idx[:N_GSM8K]
    out["gsm8k"] = [rows[i] for i in order]
    write_json(CACHE / "gsm8k_subset.json", {"seed": SEED, "n": len(out["gsm8k"]), "total": len(rows), "ids": [r["id"] for r in out["gsm8k"]]})

    # MMLU-Pro test
    mp = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    rows = []
    letters = "ABCDEFGHIJ"
    for i, ex in enumerate(mp):
        opts = list(ex["options"])
        labeled = [f"{letters[j]}. {opts[j]}" for j in range(len(opts))]
        ans = ex.get("answer")
        if isinstance(ans, str) and ans:
            gold = ans.strip().upper()
        else:
            gold = letters[int(ex["answer_index"])]
        rows.append(
            {
                "id": f"mmlu-pro-{ex.get('question_id', i)}",
                "src_index": i,
                "question": ex["question"],
                "options": labeled,
                "gold": gold,
                "category": ex.get("category"),
            }
        )
    rng = random.Random(SEED)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    out["mmlu_pro"] = [rows[i] for i in idx[:N_MMLU]]
    write_json(CACHE / "mmlu_pro_subset.json", {"seed": SEED, "n": len(out["mmlu_pro"]), "total": len(rows), "ids": [r["id"] for r in out["mmlu_pro"]]})

    # GPQA Diamond full. Official dataset is gated; skip cleanly if this account has no access.
    try:
        gp = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    except Exception as e:
        print("GPQA_SKIP", type(e).__name__, str(e)[:400], flush=True)
        write_json(CACHE / "gpqa_diamond_subset.json", {
            "skipped": True,
            "reason": "official Idavidrein/gpqa is gated on Hugging Face; this login does not have access",
            "error_type": type(e).__name__,
        })
        out["gpqa_diamond"] = []
        gp = None
    rows = []
    if gp is not None:
      for i, ex in enumerate(gp):
        correct = ex["Correct Answer"].strip()
        choices = [
            correct,
            ex["Incorrect Answer 1"].strip(),
            ex["Incorrect Answer 2"].strip(),
            ex["Incorrect Answer 3"].strip(),
        ]
        # reproducible shuffle per item
        item_rng = random.Random(f"{SEED}:gpqa:{i}")
        order = [0, 1, 2, 3]
        item_rng.shuffle(order)
        letters = "ABCD"
        labeled = []
        gold = None
        for j, src in enumerate(order):
            labeled.append(f"{letters[j]}. {choices[src]}")
            if src == 0:
                gold = letters[j]
        q = ex.get("Question") or ex.get("question")
        rows.append(
            {
                "id": f"gpqa-diamond-{i}",
                "src_index": i,
                "question": q,
                "options": labeled,
                "gold": gold,
                "permutation": order,
            }
        )
      out["gpqa_diamond"] = rows  # full 198, original order
      write_json(CACHE / "gpqa_diamond_subset.json", {"seed": SEED, "n": len(rows), "total": len(rows), "shuffle": "per-item choice order from seed, items not subsampled"})

    # AIME 2025 full
    am = load_dataset("math-ai/aime25", split="test")
    rows = []
    for i, ex in enumerate(am):
        rows.append(
            {
                "id": f"aime2025-{ex.get('id', i)}",
                "src_index": i,
                "question": ex["problem"],
                "gold": str(int(str(ex["answer"]).strip())),
            }
        )
    out["aime2025"] = rows
    write_json(CACHE / "aime2025_subset.json", {"seed": SEED, "n": len(rows), "total": len(rows), "subsampled": False})
    return out


def prompt_for(task: str, item: dict) -> str:
    if task == "gsm8k":
        return (
            "Solve the following grade-school math problem. "
            "Write your reasoning, then put the final numeric answer on its own line after ####.\n\n"
            f"Problem: {item['question']}"
        )
    if task in {"mmlu_pro", "gpqa_diamond"}:
        opts = "\n".join(item["options"])
        last = "J" if task == "mmlu_pro" else "D"
        return (
            "Answer the multiple-choice question. Think step by step, then put the "
            f"final answer letter (A-{last}) on its own line as Answer: <letter>.\n\n"
            f"Question: {item['question']}\n\nOptions:\n{opts}"
        )
    if task == "aime2025":
        return (
            "Solve the following AIME problem. The answer is an integer from 0 to 999. "
            "Write your reasoning, then put the final answer in \\boxed{}.\n\n"
            f"{item['question']}"
        )
    raise ValueError(task)


def score_item(task: str, item: dict, content: str) -> tuple[str | None, bool]:
    if task == "gsm8k":
        pred = extract_gsm8k(content)
        return pred, numbers_equal(pred, item["gold"])
    if task in {"mmlu_pro", "gpqa_diamond"}:
        letters = "ABCDEFGHIJ" if task == "mmlu_pro" else "ABCD"
        pred = extract_letter(content, letters)
        gold = item["gold"]
        return pred, (pred is not None and pred == gold)
    if task == "aime2025":
        pred = extract_aime(content)
        return pred, (pred is not None and pred == item["gold"])
    raise ValueError(task)


def run_task(task: str, items: list[dict]) -> dict:
    pred_path = OUTDIR / task / "predictions.jsonl"
    done_ids = {r["id"] for r in load_jsonl(pred_path) if not r.get("error")}
    print(f"=== {task} n={len(items)} already_ok={len(done_ids)} {now_cet()}", flush=True)
    t0 = time.time()
    for k, item in enumerate(items, 1):
        if item["id"] in done_ids:
            continue
        prompt = prompt_for(task, item)
        resp = chat(prompt)
        content = resp.get("content") or ""
        # if the model only put the answer in reasoning, still score content first
        pred, correct = score_item(task, item, content)
        rec = {
            "id": item["id"],
            "task": task,
            "src_index": item.get("src_index"),
            "gold": item["gold"],
            "pred": pred,
            "correct": correct,
            "finish_reason": resp.get("finish_reason"),
            "elapsed_s": resp.get("elapsed_s"),
            "ttft_s": resp.get("ttft_s"),
            "usage": resp.get("usage"),
            "error": resp.get("error"),
            "content": content,
            "reasoning_content": resp.get("reasoning_content") or "",
            "ts": now_cet(),
        }
        if task in {"mmlu_pro", "gpqa_diamond"}:
            rec["category"] = item.get("category")
            rec["permutation"] = item.get("permutation")
        append_jsonl(pred_path, rec)
        status = "ERR" if rec["error"] else ("OK" if correct else "NO")
        usage = rec.get("usage") or {}
        print(
            f"{task} {k}/{len(items)} {item['id']} {status} pred={pred} gold={item['gold']} "
            f"{rec['elapsed_s']:.1f}s tokens={usage.get('completion_tokens')} finish={rec['finish_reason']}",
            flush=True,
        )
        if rec["error"]:
            time.sleep(2)
    rows = load_jsonl(pred_path)
    ok_rows = [r for r in rows if not r.get("error")]
    n_ok = len(ok_rows)
    n_correct = sum(1 for r in ok_rows if r.get("correct"))
    n_err = sum(1 for r in rows if r.get("error"))
    n_trunc = sum(1 for r in ok_rows if r.get("finish_reason") not in (None, "stop"))
    acc = (n_correct / n_ok) if n_ok else None
    summary = {
        "task": task,
        "n_scheduled": len(items),
        "n_scored": n_ok,
        "n_correct": n_correct,
        "n_error": n_err,
        "n_truncated": n_trunc,
        "accuracy": acc,
        "accuracy_pct": None if acc is None else round(100.0 * acc, 1),
        "seed": SEED,
        "elapsed_s": time.time() - t0,
    }
    write_json(OUTDIR / task / "summary.json", summary)
    print(f"=== {task} done scored={n_ok}/{len(items)} acc={summary['accuracy_pct']}% errors={n_err} {now_cet()}", flush=True)
    return summary


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "started": now_cet(),
        "host": os.uname().nodename if hasattr(os, "uname") else "",
        "endpoint": URL,
        "model": MODEL,
        "recipe_commit": "fd26d84c1edb5b499792965981fbdcd8fcce2599",
        "serve": "bash scripts/serve.sh 8080, MLX_VLM_TOKEN_QUEUE_TIMEOUT=3600, draft on, default TK=4",
        "sampling": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": MAX_TOKENS,
            "reasoning_effort": REASONING_EFFORT,
            "reasoning_effort_numeric": 0.9,
            "thinking_mode": "on; reasoning_effort=high (model default 0.9)",
            "n_samples": 1,
            "greedy": True,
        },
        "seed": SEED,
        "subsets": {
            "gsm8k": f"openai/gsm8k main test, random {N_GSM8K} of 1319, seed {SEED}",
            "mmlu_pro": f"TIGER-Lab/MMLU-Pro test, random {N_MMLU}, seed {SEED}",
            "gpqa_diamond": "Idavidrein/gpqa gpqa_diamond, full set (198); choices shuffled per item with seed",
            "aime2025": "math-ai/aime25 test, full set (30), not subsampled",
        },
        "scoring": {
            "gsm8k": "numeric equality after extracting #### / \\boxed{} / last number from the assistant content (not the reasoning trace)",
            "mmlu_pro": "letter A-J extracted from assistant content",
            "gpqa_diamond": "letter A-D extracted from assistant content",
            "aime2025": "integer 0-999 from \\boxed{} or answer line in assistant content",
        },
        "hf_model_card": {
            "url": "https://huggingface.co/thinkingmachines/Inkling-2-Mini",
            "note": "The official card table does not report GSM8K, MMLU-Pro, GPQA Diamond, or AIME 2025, so no card scores are cited.",
        },
    }
    write_json(OUTDIR / "meta.json", meta)
    print("preparing datasets", now_cet(), flush=True)
    data = prepare_datasets()
    print({k: len(v) for k, v in data.items()}, flush=True)
    summaries = {}
    for task in ("gsm8k", "mmlu_pro", "gpqa_diamond", "aime2025"):
        items = data.get(task) or []
        if not items:
            summaries[task] = {
                "task": task,
                "skipped": True,
                "reason": "dataset unavailable (see cache/*_subset.json)",
                "n_scheduled": 0,
                "n_scored": 0,
                "n_correct": 0,
                "accuracy": None,
            }
            write_json(OUTDIR / task / "summary.json", summaries[task])
            print(f"=== {task} SKIPPED", flush=True)
        else:
            summaries[task] = run_task(task, items)
        write_json(OUTDIR / "summary.json", {"updated": now_cet(), "tasks": summaries})
    meta["finished"] = now_cet()
    write_json(OUTDIR / "meta.json", meta)
    write_json(OUTDIR / "summary.json", {"updated": now_cet(), "finished": True, "tasks": summaries})
    print("ALL_DONE", json.dumps(summaries, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
