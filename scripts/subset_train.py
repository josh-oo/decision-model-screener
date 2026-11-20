"""Copy a request set, keeping only k of its train reviews (for the learning curve over the number of reviews).

  python scripts/subset_train.py --reviews 12 --seed 1
  python scripts/subset_train.py --source <request set name> --reviews 23 --seed 2
  python scripts/subset_train.py --label-source abstract     # all train reviews with title/abstract labels
  python scripts/subset_train.py --reviews 46 --seed 1 --complement   # the other train reviews (for 2-fold teachers)

With --label-source, the subset is every train review whose labels come from that source (no random choice; writes
<source>__labels-<label source>/). With --complement, the subset is every train review that --reviews/--seed would NOT
choose (writes <source>__k<k>-s<seed>-rest/), so the two subsets split the train reviews into two disjoint folds.

Writes data/processed/training/requests/<source>__k<k>-s<seed>/ with the same files as the source, but train.jsonl
and train.meta.jsonl contain only the chosen reviews (all of their requests, so a balanced source stays balanced).
val and test files are copied unchanged, so every subset is evaluated on the same records.

The k reviews are stratified by domain and by size: each domain gets a share of k proportional to its number of
train reviews (largest remainder), its reviews are sorted by number of requests and cut into that many groups of
neighbours, and one review is drawn at random from each group. Subsets with the same seed are not nested.
"""
import argparse, json, random, shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQ = ROOT / "data" / "processed" / "training" / "requests"
REVIEWS = ROOT / "data" / "processed" / "training" / "reviews.jsonl"


def allocate(sizes, k):
    """Split k over groups proportionally to their sizes (largest remainder)."""
    total = sum(sizes.values())
    exact = {g: k * n / total for g, n in sizes.items()}
    alloc = {g: int(x) for g, x in exact.items()}
    for g in sorted(exact, key=lambda g: exact[g] - alloc[g], reverse=True)[:k - sum(alloc.values())]:
        alloc[g] += 1
    return alloc


def choose(review_sizes, domain_of, k, seed):
    rng = random.Random(seed)
    by_domain = defaultdict(list)
    for rid in sorted(review_sizes):
        by_domain[domain_of[rid]].append(rid)
    chosen = []
    for domain, n in allocate({d: len(r) for d, r in by_domain.items()}, k).items():
        ordered = sorted(by_domain[domain], key=lambda r: (review_sizes[r], r))
        for i in range(n):   # n groups of neighbours by size, one review from each
            group = ordered[i * len(ordered) // n:(i + 1) * len(ordered) // n]
            chosen.append(rng.choice(group))
    return sorted(chosen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="review-criteria__record-title+abstract__label-abstract_else_final__train-neg1")
    ap.add_argument("--reviews", type=int, help="number of train reviews to keep")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--label-source", choices=["abstract", "final"], help="keep every review with these labels instead")
    ap.add_argument("--complement", action="store_true", help="keep the train reviews that --reviews/--seed leaves out")
    args = ap.parse_args()
    if (args.label_source is None) == (args.reviews is None or args.seed is None):
        ap.error("give either --reviews and --seed, or --label-source")
    if args.complement and args.label_source:
        ap.error("--complement goes with --reviews and --seed")
    src = REQ / args.source
    out = REQ / (f"{args.source}__labels-{args.label_source}" if args.label_source
                 else f"{args.source}__k{args.reviews}-s{args.seed}" + ("-rest" if args.complement else ""))

    meta = [json.loads(line) for line in (src / "train.meta.jsonl").open()]
    sizes = Counter(m["review_id"] for m in meta)
    domain_of = {r["review_id"]: r["domain"] for r in map(json.loads, REVIEWS.open())}
    if args.label_source:
        sources = {}
        for m in meta:
            sources.setdefault(m["review_id"], set()).add(m["label_source"])
        assert all(len(v) == 1 for v in sources.values()), "a review mixes label sources"
        chosen = {r for r, v in sources.items() if v == {args.label_source}}
    else:
        if not 0 < args.reviews <= len(sizes):
            ap.error(f"--reviews must be between 1 and {len(sizes)}")
        chosen = set(choose(sizes, domain_of, args.reviews, args.seed))
        if args.complement:
            chosen = set(sizes) - chosen

    out.mkdir(parents=True, exist_ok=True)
    kept = 0
    positives = 0
    with (src / "train.jsonl").open() as fb, (out / "train.jsonl").open("w") as ob, \
         (out / "train.meta.jsonl").open("w") as om:
        for body_line, m in zip(fb, meta):   # same line order as the source, which the build already shuffled
            if m["review_id"] in chosen:
                ob.write(body_line)
                om.write(json.dumps(m) + "\n")
                kept += 1
                positives += json.loads(body_line)["questions"]["include"]["label"]
    for f in src.iterdir():
        if f.suffix == ".jsonl" and not f.name.startswith("train."):
            shutil.copyfile(f, out / f.name)

    config = json.loads((src / "config.json").read_text())
    config.update(name=out.name, subset_of=args.source, subset_reviews=len(chosen), subset_seed=args.seed, subset_complement=args.complement,
                  subset_label_source=args.label_source,
                  subset_review_ids=sorted(chosen),
                  subset_domains=dict(Counter(domain_of[r] for r in chosen)),
                  subset_train={"requests": kept, "positives": positives, "reviews": len(chosen),
                                "share_of_source_requests": round(kept / len(meta), 3)})
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(out.name, json.dumps(config["subset_train"]), config["subset_domains"])


if __name__ == "__main__":
    main()
