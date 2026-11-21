"""Review-first variant of the throughput sample, for prefix caching, with no retraining.

  .venv/bin/python bench/make_review_first.py

Training renders the state with sorted keys, so "record" comes before "review". Here the state is passed as a
pre-rendered string with "review" first; everything else (inner key order, separators, schema, spans) is unchanged.
Writes tok_{all,abs,title}_rf.jsonl next to the sample files in data/processed/bench/ (same rows, same order).
"""
import json
from pathlib import Path

from transformers import AutoTokenizer
from unsloth._vendor.clef.joint_schema_model import render
from unsloth.models.clef import encode_record
from unsloth.models.decision import _clef_question

OUT = Path(__file__).resolve().parent.parent / "data/processed/bench"
tok = AutoTokenizer.from_pretrained("unsloth/Qwen3.5-0.8B", revision="23c69c53358a07516b5827588b3fdb12ae78fd65")
sample = [json.loads(l) for l in open(OUT / "sample5000.jsonl")]


def encoded(r, review, record):
    state = '{"review":' + render(review) + ',"record":' + render(record) + "}"
    q = {n: _clef_question({k: v for k, v in qq.items() if k != "label"}) for n, qq in r["questions"].items()}
    e = encode_record(tok, {"state": state, "questions": q}, max_length=8192)
    L = len(e.input_ids)
    schema = {"question_types": [x.question_type for x in e.questions],
              "question_spans": [[x.question_span[0] - L, x.question_span[1] - L] for x in e.questions],
              "option_spans": [[s - L, t - L] for x in e.questions for s, t in x.option_spans],
              "option_counts": [len(x.option_spans) for x in e.questions]}
    return {"row": r["row"], "review_id": r["meta"]["review_id"], "ids": list(e.input_ids), "schema": schema}


def write(name, items):
    with open(OUT / name, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")


full = [encoded(r, r["state"]["review"], r["state"]["record"]) for r in sample]
write("tok_all_rf.jsonl", full)
write("tok_abs_rf.jsonl", [t for r, t in zip(sample, full) if r["state"]["record"].get("abstract")])
write("tok_title_rf.jsonl", [encoded(r, r["state"]["review"], {"title": r["state"]["record"]["title"]}) for r in sample])
print("wrote", OUT)
