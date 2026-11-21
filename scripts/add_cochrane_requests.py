"""Add the Cochrane training set to the train split of an existing request set.

  python scripts/add_cochrane_requests.py
  python scripts/add_cochrane_requests.py --source <request set> --cochrane data/processed/cochrane/train_neg1

Writes data/processed/training/requests/<source>__plus-cochrane-<cochrane dir name>/:
  train.jsonl / train.meta.jsonl   the source's train requests plus one request per Cochrane record, shuffled (--seed)
  val / test files                 copied unchanged, so results stay comparable with the source
  config.json                      the source config plus what was added; extra_train_data names the Cochrane dir,
                                   so train_decision_model.py can find the Cochrane reviews and records for its
                                   leakage check
Requests are built exactly as in build_requests.py: review state = the criteria (sections joined as there), record
= title + abstract, the same `include` question. Cochrane labels: included study = 1, sampled similar article = 0
(label_source "cochrane"; these are decisions after full text and unscreened negatives, not title/abstract
decisions). The Cochrane build already removed CLEF TAR reviews and every paper of a val/test record; the training
script checks the paper overlap with val again.
"""
import argparse, json, random, shutil, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from screening import QUESTIONS  # noqa: E402
from build_requests import criteria_text  # noqa: E402

REQ = ROOT / "data" / "processed" / "training" / "requests"

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="review-criteria__record-title+abstract__label-abstract_else_final__train-neg1")
ap.add_argument("--cochrane", default="data/processed/cochrane/train_neg1")
ap.add_argument("--seed", type=int, default=17)
args = ap.parse_args()

src, coch = REQ / args.source, ROOT / args.cochrane
cfg = json.loads((src / "config.json").read_text())
assert cfg["review_fields"] == "criteria" and cfg["record_fields"] == "title,abstract", "only the default fields"
out = REQ / f"{args.source}__plus-cochrane-{coch.name}"
assert not out.exists(), f"{out} exists"

reviews = {r["review_id"]: r for r in map(json.loads, (coch / "reviews.jsonl").open())}
src_reviews = {json.loads(l)["review_id"] for l in (src / "train.meta.jsonl").open()}
held = set()
for f in src.glob("*.meta.jsonl"):
    if not f.name.startswith("train."):
        held |= {json.loads(l)["review_id"] for l in f.open()}
assert not set(reviews) & (src_reviews | held), "Cochrane review id clashes with a source review"

rows = [(json.loads(b), json.loads(m)) for b, m in zip((src / "train.jsonl").open(), (src / "train.meta.jsonl").open())]
n_src = len(rows)
added = Counter()
for rec in map(json.loads, (coch / "records.jsonl").open()):
    rev = reviews[rec["review_id"]]
    assert rec["abstract"], "Cochrane train records all have an abstract"
    body = {"state": {"review": {"criteria": criteria_text(rev["criteria"])},
                      "record": {f: rec[f] for f in ("title", "abstract") if rec[f]}},
            "questions": {"include": {**QUESTIONS["include"], "label": bool(rec["label_final"])}}}
    rows.append((body, {"review_id": rec["review_id"], "record_id": rec["record_id"], "label_source": "cochrane"}))
    added[rec["label_final"]] += 1
random.Random(args.seed).shuffle(rows)

out.mkdir(parents=True)
with (out / "train.jsonl").open("w") as fb, (out / "train.meta.jsonl").open("w") as fm:
    for body, meta in rows:
        fb.write(json.dumps(body, ensure_ascii=False) + "\n")
        fm.write(json.dumps(meta) + "\n")
for f in src.iterdir():
    if f.suffix == ".jsonl" and not f.name.startswith("train."):
        shutil.copyfile(f, out / f.name)
cfg.update(name=out.name, plus_cochrane_of=args.source, extra_train_data=[args.cochrane], plus_cochrane_seed=args.seed,
           plus_cochrane={"reviews": len(reviews), "requests": sum(added.values()), "positives": added[1],
                          "negatives": added[0], "source_train_requests": n_src, "train_requests": len(rows),
                          "cochrane_share": round(sum(added.values()) / len(rows), 3)})
(out / "config.json").write_text(json.dumps(cfg, indent=2))
print(out.name, json.dumps(cfg["plus_cochrane"]))
