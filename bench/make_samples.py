"""Build the fixed throughput sample: 5,000 val requests (seed 0), pre-tokenized with the training encoder.

  .venv/bin/python bench/make_samples.py

Writes to data/processed/bench/ (not committed):
  sample5000.jsonl   the requests (+ val row index and meta)
  tok_all.jsonl      token ids + joint-schema spans, as used in training (same for 0.8B and 4B: same tokenizer)
  tok_abs.jsonl      the records of tok_all that have an abstract
  tok_title.jsonl    every record with its abstract removed (title-only, as records without abstract appear)
Spans are counted from the end of the prompt, as vLLM's joint-schema pooler expects.
"""
import json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQ = ROOT / "data/processed/training/requests/review-criteria__record-title+abstract__label-abstract_else_final__train-neg1"
OUT = ROOT / "data/processed/bench"
OUT.mkdir(parents=True, exist_ok=True)

from transformers import AutoTokenizer  # noqa: E402
from unsloth.models.clef import encode_record  # noqa: E402
from unsloth.models.decision import _clef_question  # noqa: E402

tok = AutoTokenizer.from_pretrained("unsloth/Qwen3.5-0.8B", revision="23c69c53358a07516b5827588b3fdb12ae78fd65")
rows = [json.loads(l) for l in open(REQ / "val.jsonl")]
meta = [json.loads(l) for l in open(REQ / "val.meta.jsonl")]
random.seed(0)
idx = sorted(random.sample(range(len(rows)), 5000))
sample = [{"row": i, "meta": meta[i], **rows[i]} for i in idx]


def encoded(r, state):
    q = {n: _clef_question({k: v for k, v in qq.items() if k != "label"}) for n, qq in r["questions"].items()}
    e = encode_record(tok, {"state": state, "questions": q}, max_length=8192)
    L = len(e.input_ids)
    schema = {"question_types": [x.question_type for x in e.questions],
              "question_spans": [[x.question_span[0] - L, x.question_span[1] - L] for x in e.questions],
              "option_spans": [[s - L, t - L] for x in e.questions for s, t in x.option_spans],
              "option_counts": [len(x.option_spans) for x in e.questions]}
    return {"row": r["row"], "ids": list(e.input_ids), "schema": schema}


def write(name, items):
    with open(OUT / name, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")


write("sample5000.jsonl", sample)
full = [encoded(r, r["state"]) for r in sample]
write("tok_all.jsonl", full)
write("tok_abs.jsonl", [t for r, t in zip(sample, full) if r["state"]["record"].get("abstract")])
write("tok_title.jsonl", [encoded(r, {"review": r["state"]["review"], "record": {"title": r["state"]["record"]["title"]}})
                          for r in sample])
print("wrote", OUT)
