"""Final Cochrane training set: included studies as positives, sampled similar articles as negatives, 1:1 per review.

  python scripts/build_cochrane_train.py                   # 1 negative per positive
  python scripts/build_cochrane_train.py --neg-per-pos 2

Inputs (from prepare_cochrane_data.py and cochrane_negatives.py) in data/processed/cochrane/:
  reviews.jsonl, records.jsonl, negatives.jsonl
Writes data/processed/cochrane/train_neg<k>/ (same schema as data/processed/training/):
  reviews.jsonl   reviews that have at least one sampled negative
  records.jsonl   positives and negatives of those reviews
  summary.json    counts, and every review or record left out with the reason

Positives: included studies with an abstract (label_final 1; label_abstract set to 1, an included study passed
title/abstract screening). Excluded studies are left out: they passed title/abstract screening, so at abstract
level they are includes, and as label 0 they would contradict that.
Negatives: per review the first round(k * positives) records of negatives.jsonl, which cochrane_negatives.py wrote
in a seeded random order, so this is a random subsample of the 3-per-positive sample. Reviews with fewer
negatives keep all they have (counted as short). Reviews without any negative are left out.
"""
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "data" / "processed" / "cochrane"


def read(path):
    with path.open() as f:
        return [json.loads(line) for line in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg-per-pos", type=float, default=1.0)
    args = ap.parse_args()
    out = DIR / f"train_neg{args.neg_per_pos:g}"
    out.mkdir(exist_ok=True)

    reviews = read(DIR / "reviews.jsonl")
    pos, left_out = defaultdict(list), Counter()
    for r in read(DIR / "records.jsonl"):
        if r["cochrane_list"] != "included":
            left_out["excluded study"] += 1
        elif not r["abstract"]:
            left_out["included study without abstract"] += 1
        else:
            pos[r["review_id"]].append(dict(r, label_abstract=1))
    neg = defaultdict(list)
    for r in read(DIR / "negatives.jsonl"):
        neg[r["review_id"]].append(r)

    kept_reviews, records, dropped_reviews, short = [], [], {}, {}
    for rv in reviews:
        rid = rv["review_id"]
        if not neg[rid]:
            dropped_reviews[rid] = "no sampled negative"
            continue
        if not pos[rid]:
            dropped_reviews[rid] = "no included study with abstract"
            continue
        need = round(args.neg_per_pos * len(pos[rid]))
        take = neg[rid][:need]
        left_out["sampled negative beyond ratio"] += len(neg[rid]) - len(take)
        if len(take) < need:
            short[rid] = need - len(take)
        kept_reviews.append(rv)
        records += pos[rid] + take
    for rid in dropped_reviews:
        left_out["record of a left-out review"] += len(pos[rid]) + len(neg[rid])

    with (out / "reviews.jsonl").open("w") as f:
        for rv in kept_reviews:
            f.write(json.dumps(rv, ensure_ascii=False) + "\n")
    with (out / "records.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_pos = sum(r["label_final"] == 1 for r in records)
    summary = {
        "neg_per_pos": args.neg_per_pos,
        "reviews_in": len(reviews), "reviews_kept": len(kept_reviews),
        "reviews_left_out": dict(Counter(dropped_reviews.values())), "reviews_left_out_ids": dropped_reviews,
        "records": len(records), "positives": n_pos, "negatives": len(records) - n_pos,
        "reviews_short_of_negatives": len(short), "negatives_missing": sum(short.values()),
        "records_left_out": dict(left_out),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "reviews_left_out_ids"}, indent=2))


if __name__ == "__main__":
    main()
