"""Zero-shot Jev on a request file from scripts/build_requests.py, as a baseline for the trained models.

  python scripts/jev_on_requests.py --estimate                       # size and cost, no API call
  python scripts/jev_on_requests.py --reviews CD010355               # one small review first
  python scripts/jev_on_requests.py                                  # every review of the split
  python scripts/jev_on_requests.py --report                         # metrics from the cache only

Jev gets exactly the request the trained model gets (same state, same `include` question); only the
label is removed and the pinned model is added. Zero-shot: labels are never sent.
Raw responses are cached in results/requests__<config>__<split>/typesafe_jev-1.14/<review>.jsonl;
finished records are not called again. Metrics are computed from the cache. A record without a score
(failed call) is ranked last and counted, never dropped. Ties (Jev returns two decimals) are broken by
BM25 with the review title as query, then record id, as in FINDINGS.md.
"""
import argparse, json, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import screening as S  # noqa: E402

DATA = ROOT / "data" / "processed" / "training"


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def body_of(row):
    questions = {name: {k: v for k, v in q.items() if k != "label"} for name, q in row["questions"].items()}
    return {"model": S.MODEL, "state": row["state"], "questions": questions}


def run_review(review, items, out, workers):
    path = out / f"{review}.jsonl"
    done = set()
    if path.exists():
        done = {r["record_id"] for r in map(json.loads, path.open()) if "error" not in r["response"]}
    todo = [(rid, body) for rid, body in items if rid not in done]
    if not todo:
        return
    with ThreadPoolExecutor(workers) as pool, path.open("a") as f:
        futures = {pool.submit(S.call, body): rid for rid, body in todo}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=review, leave=False):
            row = {"record_id": futures[fut], "response": fut.result()}
            if row["response"].get("error", "").startswith(("401", "402", "403")):
                pool.shutdown(cancel_futures=True)
                raise RuntimeError(row["response"]["error"])
            f.write(json.dumps(row) + "\n")
            f.flush()


def report(meta, out, reviews):
    rows, cost, versions = {}, 0.0, set()
    for path in out.glob("*.jsonl"):
        for r in map(json.loads, path.open()):
            p = r["response"].get("answers", {}).get("include", {}).get("noul")
            if p is not None or (path.stem, r["record_id"]) not in rows:
                rows[path.stem, r["record_id"]] = p
            cost += r["response"].get("usage", {}).get("cost") or 0
            versions.add(r["response"].get("model"))
    recs = {(r["review_id"], r["record_id"]): r for r in read_jsonl(DATA / "records.jsonl")}
    df = pd.DataFrame([{**m, "jev": rows.get((m["review_id"], m["record_id"]))} for m in meta])
    df = df[df.review_id.isin(reviews)]
    for k in ("title", "abstract", "label_abstract", "label_final"):
        df[k] = [recs[r, i][k] for r, i in zip(df.review_id, df.record_id)]
    titles = {r["review_id"]: r["title"] for r in read_jsonl(DATA / "reviews.jsonl")}
    df["bm25"] = 0.0
    for rid, g in df.groupby("review_id"):
        bm = BM25Okapi([S.tokens(f"{t} {a or ''}") or ["_"] for t, a in zip(g.title, g.abstract)])
        df.loc[g.index, "bm25"] = bm.get_scores(S.tokens(titles[rid]))
    missing = int(df.jev.isna().sum())
    print(f"scored {len(df) - missing} of {len(df)} records in {df.review_id.nunique()} reviews | "
          f"without a score (ranked last): {missing} | model: {versions} | cost so far: ${cost:.2f}")
    df["jev"] = df.jev.fillna(-1.0)
    for level in ("abstract", "final"):
        per = []
        for _, g in df.groupby("review_id"):
            ranked = g.sort_values(["jev", "bm25", "record_id"], ascending=[False, False, True])
            m = S.metrics(ranked[f"label_{level}"].fillna(0).astype(int))
            if m:
                per.append(m)
        if per:
            mean = pd.DataFrame(per).mean() * 100
            print(f"{level}-level labels, {len(per)} reviews: " +
                  " ".join(f"{k}={mean[k]:.1f}" for k in ("MAP", "R@10%", "WSS@95%", "TNR@95%")))
    df[["review_id", "record_id", "jev", "bm25", "label_abstract", "label_final"]].to_json(
        out.parent / "scores.jsonl", orient="records", lines=True)   # outside the cache folder
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", default="review-criteria__record-title+abstract__label-abstract_else_final")
    ap.add_argument("--split", default="val")
    ap.add_argument("--reviews", default=None, help="comma list; default: every review of the split")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    req = DATA / "requests" / args.requests
    rows, meta = read_jsonl(req / f"{args.split}.jsonl"), read_jsonl(req / f"{args.split}.meta.jsonl")
    reviews = sorted({m["review_id"] for m in meta})
    if args.reviews:
        reviews = args.reviews.split(",")
    out = S.out_dir(f"requests__{args.requests}__{args.split}")

    if args.report:
        return report(meta, out, reviews)
    by_review = {}
    for row, m in zip(rows, meta):
        if m["review_id"] in reviews:
            by_review.setdefault(m["review_id"], []).append((m["record_id"], body_of(row)))
    if args.estimate:
        chars = np.array([len(json.dumps(b)) for items in by_review.values() for _, b in items])
        price = S.price_per_input_token()
        print(f"{len(chars)} requests in {len(by_review)} reviews | mean chars {chars.mean():.0f} | "
              f"est. input tokens {chars.sum() / 3.06 / 1e6:.1f}M | est. cost ${chars.sum() / 3.06 * price:.2f}")
        return
    for review in tqdm(reviews, desc="reviews"):
        run_review(review, by_review[review], out, args.workers)
    report(meta, out, reviews)


if __name__ == "__main__":
    main()
