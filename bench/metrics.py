import json, sys, numpy as np, collections
meta = {}
import pathlib
for l in open(pathlib.Path(__file__).resolve().parent.parent / "data/processed/bench/sample5000.jsonl"):
    r = json.loads(l); meta[r["row"]] = (r["meta"]["review_id"], int(r["questions"]["include"]["label"]))
def ap(y, p):
    o = np.argsort(-p, kind="stable"); y = y[o]; h = np.cumsum(y)
    return (h[y == 1] / (np.flatnonzero(y) + 1)).mean()
def wss95(y, p):
    o = np.argsort(-p, kind="stable"); y = y[o]; k = np.searchsorted(np.cumsum(y), np.ceil(0.95 * y.sum())) + 1
    return (len(y) - k) / len(y) - 0.05
print("file | MAP abs | WSS@95 abs  (sample of 5,000 val records, CLEF TAR abstract-level labels, macro over reviews with >=1 include)")
for f in sys.argv[1:]:
    p = {json.loads(l)["row"]: json.loads(l)["p"] for l in open(f)}
    by = collections.defaultdict(list)
    for row, pp in p.items(): by[meta[row][0]].append((meta[row][1], pp))
    A, W = [], []
    for v in by.values():
        y = np.array([a for a, _ in v]); s = np.array([b for _, b in v])
        if y.sum(): A.append(ap(y, s)); W.append(wss95(y, s))
    print(f"{f} | {100*np.mean(A):.2f} | {100*np.mean(W):.2f}  (reviews={len(A)}, n={len(p)})")
