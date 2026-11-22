"""Build the fixed prompt sets (JSONL) for the short quality benchmark. Deterministic."""
import json, random, pandas as pd
REV = {"gsm8k": "openai/gsm8k@740312add88f781978c0658806c59bc2815b9866 (main/test)",
       "mmlu_pro": "TIGER-Lab/MMLU-Pro@b189ec765aa7ed75c8acfea42df31fdae71f97be (test)",
       "ruler_niah_32k": "SaylorTwift/RULER-32768-llama-3.2-tokenizer@e748e0cd1872b4bbaa6d5ed9c6fbcf6068951c42 (niah_single_1/2/3)"}
out = []
g = pd.read_parquet("data/gsm8k-test.parquet")
for i in range(100):  # first 100 of the test split
    q, a = g.iloc[i]["question"], g.iloc[i]["answer"]
    gold = a.split("####")[-1].strip().replace(",", "")
    user = (q + "\n\nSolve the problem step by step. End with the final answer as a single number on the last line, "
            "in the form: Final answer: <number>")
    out.append({"bench": "gsm8k", "id": f"gsm8k-test-{i}", "user": user, "gold": gold, "max_tokens": 2048})
m = pd.read_parquet("data/mmlupro-test.parquet")
rng = random.Random(0)
for cat in sorted(m["category"].unique()):
    ids = sorted(m[m["category"] == cat]["question_id"].tolist())
    for qid in sorted(rng.sample(ids, 10)):
        r = m[m["question_id"] == qid].iloc[0]
        opts = list(r["options"]); letters = "ABCDEFGHIJ"[: len(opts)]
        user = (r["question"] + "\n\nOptions:\n" + "\n".join(f"{l}. {o}" for l, o in zip(letters, opts)) +
                "\n\nChoose the single best option. End with the letter on the last line, in the form: Answer: <letter>")
        out.append({"bench": "mmlu_pro", "id": f"mmlu_pro-{int(qid)}", "category": cat, "user": user,
                    "gold": r["answer"], "n_options": len(opts), "max_tokens": 2048})
for task, rows in [("niah_single_1", range(0, 7)), ("niah_single_2", range(0, 7)), ("niah_single_3", range(0, 6))]:
    d = pd.read_parquet(f"data/r32k-{task}.parquet")
    for i in rows:
        r = d.iloc[i]
        out.append({"bench": "ruler_niah_32k", "id": f"{task}-{int(r['index'])}", "user": r["input"],
                    "gold": [str(x) for x in r["outputs"]], "max_tokens": 512, "llama_len": int(r["length"])})
with open("prompts.jsonl", "w") as f:
    for o in out: f.write(json.dumps(o) + "\n")
json.dump(REV, open("datasets.json", "w"), indent=1)
from collections import Counter; print(Counter(o["bench"] for o in out), Counter(o.get("category") for o in out if o["bench"]=="mmlu_pro").most_common(3))
