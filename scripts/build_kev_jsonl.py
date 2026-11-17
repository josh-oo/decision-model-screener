"""Turn data/processed/training/{reviews,records}.jsonl into Kev fine-tuning files (JSONL, one request per line).

  python scripts/build_kev_jsonl.py                         # selection criteria, record title + abstract
  python scripts/build_kev_jsonl.py --review-fields criteria,title   # criteria are always required

Writes data/processed/training/kev/<name>/:
  train.jsonl, val.jsonl, test_tar2019.jsonl, test_synergy.jsonl   Kev requests with a label per question
  <same>.meta.jsonl                                                  review_id, record_id, label source; same line order
  config.json                                                        the options used

Each request: state = {"review": {...}, "record": {...}}, one `noul` question "include" with the
instruction used in the Jev runs. Label = title/abstract decision where the source has one, else the
decision after full text (`--label` changes this). The label source is written to the meta file.
The selection criteria are always part of the review information; reviews without criteria are
skipped and listed in config.json.
"""
import argparse, json, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from screening import QUESTIONS  # noqa: E402

DATA = ROOT / "data" / "processed" / "training"
SECTION_NAMES = {"studies": "Types of studies", "participants": "Types of participants",
                 "interventions": "Types of interventions", "outcomes": "Types of outcome measures"}


def criteria_text(criteria):
    """CLEF TAR criteria come as sections, SYNERGY+ as free text; both become one string."""
    if isinstance(criteria, dict):
        return "\n".join(f"{SECTION_NAMES[k]}: {v}" for k, v in criteria.items() if v)
    return criteria


def norm_title(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def paper_keys(rec):
    """Ways to recognize the same paper in another review: PMID, DOI, title (titles under 30 characters are too generic)."""
    keys = set()
    if rec["pmid"]:
        keys.add("pmid:" + rec["pmid"])
    if rec["doi"]:
        keys.add("doi:" + re.sub(r"^https?://(dx\.)?doi\.org/", "", rec["doi"].strip().lower()))
    title = norm_title(rec["title"])
    if len(title) >= 30:
        keys.add("title:" + title)
    return keys


def label_of(rec, mode):
    if mode == "abstract_else_final":
        return (rec["label_abstract"], "abstract") if rec["label_abstract"] is not None else (rec["label_final"], "final")
    key = f"label_{mode}"
    return (rec[key], mode) if rec[key] is not None else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review-fields", default="criteria",
                    help="comma list of: criteria (always required), title, objectives")
    ap.add_argument("--record-fields", default="title,abstract", help="comma list of: title, abstract")
    ap.add_argument("--label", default="abstract_else_final", choices=["abstract_else_final", "abstract", "final"])
    ap.add_argument("--train-neg-per-pos", type=float, default=None,
                    help="optional: subsample train negatives per review to this ratio (val/test never subsampled)")
    ap.add_argument("--max-overlap", type=float, default=0.2,
                    help="drop train reviews sharing more than this fraction of a val/test review's records")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    review_fields = args.review_fields.split(",")
    if "criteria" not in review_fields:
        ap.error("the selection criteria are always part of the review information: include 'criteria'")
    record_fields = args.record_fields.split(",")
    name = args.name or f"review-{'+'.join(review_fields)}__record-{'+'.join(record_fields)}__label-{args.label}"
    out = DATA / "kev" / name
    out.mkdir(parents=True, exist_ok=True)

    reviews = {r["review_id"]: r for r in map(json.loads, (DATA / "reviews.jsonl").open())}
    by_review = defaultdict(list)
    for rec in map(json.loads, (DATA / "records.jsonl").open()):
        by_review[rec["review_id"]].append(rec)

    # Leakage guard, train side only (val/test stay as published). A paper is matched by PMID, DOI or normalized title.
    # 1. Drop a train review with the same title as a val/test review (Parodis_2023a/b) or sharing more than
    #    --max-overlap of a val/test review's records (Kerschbaumer_2024a and Webers_2022 vs Kerschbaumer_2024b).
    # 2. Drop every remaining train record that is the same paper as any val/test record, so no val/test paper is
    #    seen in training, with either label (e.g. 81 of Lewowski_2021's includes are includes of a train review).
    held_out = [rid for rid, r in reviews.items() if r["split"] != "train"]
    held_keys = {h: set().union(*map(paper_keys, by_review[h])) for h in held_out}
    all_held_keys = set().union(*held_keys.values())
    held_titles = {norm_title(reviews[h]["title"]): h for h in held_out}
    overlap_dropped = {}
    for rid, rev in reviews.items():
        if rev["split"] != "train":
            continue
        if norm_title(rev["title"]) in held_titles:
            overlap_dropped[rid] = f"same review title as {held_titles[norm_title(rev['title'])]}"
            continue
        mine = set().union(*map(paper_keys, by_review[rid]))
        for h in held_out:
            share = sum(bool(paper_keys(r) & mine) for r in by_review[h]) / max(1, len(by_review[h]))
            if share > args.max_overlap:
                overlap_dropped[rid] = f"{share:.0%} of {h}"
                break
    shared_with_held_out = Counter()

    files = {"train": [], "val": [], "test_tar2019": [], "test_synergy": []}
    skipped_reviews, unlabeled, subsampled = [], 0, 0
    rng = random.Random(args.seed)
    for rid, rev in reviews.items():
        if rid in overlap_dropped:
            continue
        target = rev["split"] if rev["split"] != "test" else f"test_{rev['source']}"
        review_state = {}
        for f in review_fields:
            value = criteria_text(rev[f]) if f == "criteria" else rev[f]
            if value:
                review_state[f] = value
        if "criteria" not in review_state:  # no selection criteria: review cannot be used
            skipped_reviews.append(rid)
            continue
        recs = by_review[rid]
        if target == "train":
            kept = [r for r in recs if not paper_keys(r) & all_held_keys]
            if len(kept) < len(recs):
                shared_with_held_out[rid] = len(recs) - len(kept)
            recs = kept
        if target == "train" and args.train_neg_per_pos:
            pos = [r for r in recs if label_of(r, args.label)[0] == 1]
            neg = [r for r in recs if label_of(r, args.label)[0] == 0]
            keep = max(1, int(len(pos) * args.train_neg_per_pos))
            if len(neg) > keep:
                subsampled += len(neg) - keep
                neg = rng.sample(neg, keep)
            recs = pos + neg
        for rec in recs:
            label, source = label_of(rec, args.label)
            if label is None:
                unlabeled += 1
                continue
            body = {"state": {"review": review_state,
                              "record": {f: rec[f] for f in record_fields if rec[f]}},
                    "questions": {"include": {**QUESTIONS["include"], "label": bool(label)}}}
            meta = {"review_id": rid, "record_id": rec["record_id"], "label_source": source}
            files[target].append((body, meta))

    rng.shuffle(files["train"])
    counts = {}
    for split, rows in files.items():
        with (out / f"{split}.jsonl").open("w") as fb, (out / f"{split}.meta.jsonl").open("w") as fm:
            for body, meta in rows:
                fb.write(json.dumps(body, ensure_ascii=False) + "\n")
                fm.write(json.dumps(meta) + "\n")
        counts[split] = {"requests": len(rows),
                         "positives": sum(b["questions"]["include"]["label"] for b, _ in rows),
                         "reviews": len({m["review_id"] for _, m in rows}),
                         "label_source": dict(Counter(m["label_source"] for _, m in rows))}
    config = dict(vars(args), name=name, train_reviews_dropped_for_overlap=overlap_dropped,
                  train_records_dropped_same_paper_as_val_test=sum(shared_with_held_out.values()),
                  train_records_dropped_by_review=dict(shared_with_held_out.most_common()),
                  skipped_reviews_missing_review_fields=skipped_reviews,
                  records_without_requested_label=unlabeled, train_negatives_subsampled_away=subsampled,
                  counts=counts)
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
