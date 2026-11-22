"""Rebuild data/draft_vocab_order.json: the draft head's candidate token ids, in priority order (patch 0016).

Provenance (no benchmark prompt, needle or RULER text or output is used):
  1. token ids of the model's own answers on GSM8K (test, first 100) and MMLU-Pro (140 questions), from
     results/quality-20261122/raw-{fork,stack4}.jsonl.gz, RULER rows excluded, most frequent first;
  2. then English Wikipedia (wikimedia/wikipedia 20231101.en, train-00000-of-00041.parquet, first ~60M
     characters) tokenized with the model's tokenizer, most frequent first.
The draft head proposes from the first MLX_VLM_INKLING2_DRAFT_VOCAB ids of this list, the request's prompt ids and
the control tokens (>= 199998). The exact verifier decides every token, so the list only affects speed.

usage: python scripts/draft_vocab.py TOKENIZER_JSON WIKIPEDIA_PARQUET  (needs `tokenizers`, `pyarrow`)
"""
import collections, gzip, json, os, sys

import pyarrow.parquet as pq
from tokenizers import Tokenizer

here = os.path.dirname(os.path.abspath(__file__))
tok = Tokenizer.from_file(sys.argv[1])
style = collections.Counter()
for name in ("raw-fork.jsonl.gz", "raw-stack4.jsonl.gz"):
    for line in gzip.open(os.path.join(here, "..", "results", "quality-20261122", name), "rt"):
        d = json.loads(line)
        if d.get("bench", "").startswith("ruler"):
            continue
        style.update(d.get("tokens") or [])
texts, n = [], 0
for batch in pq.ParquetFile(sys.argv[2]).iter_batches(columns=["text"], batch_size=2000):
    for t in batch.column(0).to_pylist():
        texts.append(t)
        n += len(t)
    if n > 60_000_000:
        break
wiki = collections.Counter()
for enc in tok.encode_batch(texts, add_special_tokens=False):
    wiki.update(enc.ids)
order, seen = [], set()
for t in [t for t, _ in style.most_common()] + [t for t, _ in wiki.most_common()]:
    if t not in seen:
        seen.add(t)
        order.append(t)
json.dump(order, open(os.path.join(here, "..", "data", "draft_vocab_order.json"), "w"))
print(len(order), "ids")
