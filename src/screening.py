"""Shared code for the screening notebooks: data, BM25, Jev calls with an on-disk cache, metrics.

The notebooks import this module and only define what differs between configurations
(usually `build_body`).
"""
import json, math, os, re, time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

MODEL = "typesafe/jev-1.14"   # pinned, no "latest" alias
RAW = ROOT / "data" / "raw"
CRIT = ROOT / "data" / "criteria"
RESULTS = ROOT / "results"
for p in (RAW, CRIT):
    p.mkdir(parents=True, exist_ok=True)

TAR = "https://raw.githubusercontent.com/CLEF-TAR/tar/master/2019-TAR/Task2/Testing/Intervention/qrels/"
LGAR = "https://raw.githubusercontent.com/XITASO/lgar/main/implementation/data/tar2019/intervention/info.json"

# The one yes/no question used in every configuration unless the experiment is about the question.
QUESTIONS = {
    "include": {
        "type": "noul",
        "instructions": "The record should be included in the systematic review: it meets all inclusion criteria and none of the exclusion criteria.",
    }
}


def out_dir(run):
    """Cache directory of one configuration, e.g. results/baseline/typesafe_jev-1.14."""
    path = RESULTS / run / MODEL.replace("/", "_")
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------- data

def download(url, path):
    if not path.exists():
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        path.write_bytes(r.content)
    return path


def read_qrels(level):
    path = download(TAR + f"full.test.intervention.{level}.2019.qrels", RAW / f"intervention.{level}.qrels")
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["review", "_", "pmid", "label"], dtype={"pmid": str})
    return df.drop(columns="_")


def parse_pubmed_xml(content):
    """{pmid: {"pmid", "title", "abstract"}} from an efetch XML response."""
    found = {}
    for art in ET.fromstring(content):
        pmid = art.findtext(".//PMID")
        title = art.find(".//ArticleTitle")
        if title is None:
            title = art.find(".//BookTitle")
        parts = []
        for a in art.findall(".//Abstract/AbstractText"):
            text = "".join(a.itertext()).strip()
            parts.append(f"{a.get('Label')}: {text}" if a.get("Label") else text)
        found[pmid] = {"pmid": pmid,
                       "title": "".join(title.itertext()).strip() if title is not None else "",
                       "abstract": "\n".join(p for p in parts if p)}
    return found


def fetch_pubmed(pmids, path=RAW / "pubmed.jsonl", batch=200):
    """Title and abstract per PMID from NCBI E-utilities, cached as JSON lines."""
    have = {}
    if path.exists():
        have = {r["pmid"]: r for r in map(json.loads, path.open())}
    todo = [p for p in pmids if p not in have]
    with path.open("a") as f:
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            for attempt in range(5):
                r = requests.post("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                                  data={"db": "pubmed", "id": ",".join(chunk), "retmode": "xml"}, timeout=120)
                if r.ok:
                    break
                time.sleep(2 * (attempt + 1))
            r.raise_for_status()
            found = parse_pubmed_xml(r.content)
            for p in chunk:  # PMIDs PubMed no longer returns are kept, with empty text
                rec = found.get(p, {"pmid": p, "title": "", "abstract": "", "missing": True})
                have[p] = rec
                f.write(json.dumps(rec) + "\n")
            time.sleep(0.4)
    return have


def load_criteria():
    return json.loads(download(LGAR, CRIT / "tar2019_intervention.json").read_text())


def load_records():
    """All (review, pmid) pairs with both label levels, record title and abstract. Prints a summary."""
    records = read_qrels("abs").rename(columns={"label": "label_abs"})
    records = records.merge(read_qrels("content").rename(columns={"label": "label_content"}), on=["review", "pmid"])
    records[["label_abs", "label_content"]] = (records[["label_abs", "label_content"]] > 0).astype(int)
    criteria = load_criteria()
    assert set(records.review) == set(criteria), "reviews in qrels and criteria differ"

    pubmed = fetch_pubmed(sorted(records.pmid.unique()))
    records["title"] = records.pmid.map(lambda p: pubmed[p]["title"])
    records["abstract"] = records.pmid.map(lambda p: pubmed[p]["abstract"])
    print(len(criteria), "reviews,", len(records), "records,", records.pmid.nunique(), "unique PMIDs")
    print("not returned by PubMed:", sum(1 for r in pubmed.values() if r.get("missing")),
          "| no abstract:", int((records.abstract == "").sum()), "of", len(records))
    return records, criteria


def fetch_review_abstracts(criteria, path=RAW / "tar2019_intervention_review_abstracts.json"):
    """Title and abstract of each Cochrane review, via the PMC link in the LGAR criteria. Cached.

    The PMC link points to the review version LGAR took the criteria from, which can be a later
    update than the one the CLEF TAR labels were made for.
    """
    if path.exists():
        return json.loads(path.read_text())
    pmcids = {review: c["link"].rstrip("/").rsplit("/", 1)[-1] for review, c in criteria.items()}
    r = requests.get("https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/",
                     params={"ids": ",".join(pmcids.values()), "format": "json"}, timeout=60)
    r.raise_for_status()
    pmid_of = {rec["pmcid"]: str(rec["pmid"]) for rec in r.json()["records"] if "pmid" in rec}
    missing = [pmc for pmc in pmcids.values() if pmc not in pmid_of]
    assert not missing, f"no PMID for {missing}"

    r = requests.post("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                      data={"db": "pubmed", "id": ",".join(pmid_of.values()), "retmode": "xml"}, timeout=120)
    r.raise_for_status()
    found = parse_pubmed_xml(r.content)
    out = {review: {"pmcid": pmc, **found[pmid_of[pmc]]} for review, pmc in pmcids.items()}
    path.write_text(json.dumps(out, indent=1))
    return out


def drop_sections(abstract, drop):
    """Structured abstract without the sections whose label is in `drop`, e.g. {"MAIN RESULTS"}.
    Each section is one line "LABEL: text", as written by `parse_pubmed_xml`."""
    labels = [line.split(":", 1)[0] for line in abstract.split("\n")]
    missing = set(drop) - set(labels)
    assert not missing, f"sections not found: {missing}"
    return "\n".join(line for line, label in zip(abstract.split("\n"), labels) if label not in drop)


# ---------------------------------------------------------------- BM25 and metrics

def tokens(s):
    return re.findall(r"\w+", s.lower())


def add_bm25(records, criteria):
    """BM25 score per record, query = review title (as in LGAR). Also the tie-breaker for all rankings."""
    records["bm25"] = 0.0
    for review, df in records.groupby("review"):
        bm25 = BM25Okapi([tokens(t + " " + a) or ["_"] for t, a in zip(df.title, df.abstract)])
        records.loc[df.index, "bm25"] = bm25.get_scores(tokens(criteria[review]["title"]))
    return records


def metrics(y):
    """y: 0/1 labels in ranked order (best first)."""
    y = np.asarray(y)
    n, pos = len(y), int(y.sum())
    if pos == 0:
        return None
    hits = np.cumsum(y)
    out = {"MAP": float((hits / np.arange(1, n + 1) * y).sum() / pos)}
    for k in (1, 5, 10, 20, 50):
        out[f"R@{k}%"] = hits[math.ceil(n * k / 100) - 1] / pos
    for r in (0.95, 1.0):
        read = int(np.argmax(hits >= math.ceil(r * pos))) + 1   # records read to reach recall r
        tn = (n - read) - (pos - hits[read - 1])
        out[f"WSS@{int(r * 100)}%"] = (n - read) / n - (1 - r)
        if r == 0.95:
            out["TNR@95%"] = tn / (n - pos)
    return out


def evaluate(records, score_col):
    """Macro-average over reviews, for CLEF TAR abstract-level and content-level labels.
    Ties are broken by BM25 score, then PMID. Reviews without includes are skipped."""
    rows = {}
    for label in ("label_abs", "label_content"):
        per_review = []
        for _, df in records.groupby("review"):
            ranked = df.sort_values([score_col, "bm25", "pmid"], ascending=[False, False, True])
            m = metrics(ranked[label])
            if m:
                per_review.append(m)
        rows[label] = (pd.DataFrame(per_review).mean() * 100).round(1) if per_review else pd.Series(dtype=float)
        rows[label]["reviews"] = len(per_review)
    return pd.DataFrame(rows).T


def sens_spec(df, score_col, threshold):
    """Sensitivity, specificity and share flagged for one label column per call; see `threshold_table`."""
    out = {}
    for label in ("label_abs", "label_content"):
        pred, y = df[score_col] >= threshold, df[label] == 1
        out[(label, "sensitivity")] = (pred & y).sum() / y.sum() if y.any() else np.nan
        out[(label, "specificity")] = (~pred & ~y).sum() / (~y).sum()
        out[(label, "flagged")] = pred.mean()
    return pd.Series(out)


def threshold_table(records, score_col, threshold):
    """(summary, per_review) in percent at a fixed score threshold.
    Mean over reviews skips reviews without includes for sensitivity."""
    per_review = pd.DataFrame({r: sens_spec(df, score_col, threshold) for r, df in records.groupby("review")}).T
    summary = pd.DataFrame({"mean over reviews": per_review.mean(),
                            "all records pooled": sens_spec(records, score_col, threshold)}).T
    return (summary * 100).round(1), (per_review * 100).round(1)


# ---------------------------------------------------------------- Jev calls

def price_per_input_token():
    ep = requests.get(f"https://openrouter.ai/api/v1/models/{MODEL}/endpoints", timeout=30).json()["data"]["endpoints"][0]
    print(ep["name"], "| USD per 1M input tokens:", float(ep["pricing"]["prompt"]) * 1e6, "| output:", ep["pricing"]["completion"])
    return float(ep["pricing"]["prompt"])


def estimate_cost(records, build_body, chars_per_token=3.06):
    """Cost estimate from request size, no API call. 3.06 chars/token was measured in the baseline smoke test."""
    price = price_per_input_token()
    chars = pd.Series([len(json.dumps(build_body(r, t, a))) for r, t, a in zip(records.review, records.title, records.abstract)],
                      index=records.index)
    est = chars.groupby(records.review).agg(records="size", chars="sum")
    est["tokens_M"] = est.chars / chars_per_token / 1e6
    est["usd"] = est.chars / chars_per_token * price
    print(f"requests: {len(records)} | mean chars per request: {chars.mean():.0f} | max: {chars.max()}")
    print(f"estimated input tokens: {est.tokens_M.sum():.1f}M | estimated cost: ${est.usd.sum():.2f}")
    return est.round(3)


def call(body, retries=6):
    """One Decisions API call. Returns the response JSON, or {"error": ...} after the retries are used up."""
    err = None
    for attempt in range(retries):
        try:
            r = requests.post("https://openrouter.ai/api/alpha/decisions", json=body, timeout=120,
                              headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
            if r.ok:
                return r.json()
            err = f"{r.status_code}: {r.text[:300]}"
            if r.status_code in (400, 401, 402, 403, 404, 413):   # retrying will not help
                break
        except requests.RequestException as e:
            err = repr(e)
        time.sleep(min(60, 2 ** attempt))
    return {"error": err}


def run_review(records, review, build_body, out, workers=8):
    """Score every record of a review; finished records are read from the cache file out/<review>.jsonl."""
    path = out / f"{review}.jsonl"
    done = {}
    if path.exists():
        done = {r["pmid"]: r for r in map(json.loads, path.open()) if "error" not in r["response"]}
    df = records[records.review == review]
    todo = df[~df.pmid.isin(done)]
    if todo.empty:
        return done
    with ThreadPoolExecutor(workers) as pool, path.open("a") as f:
        futures = {pool.submit(call, build_body(review, t, a)): p for p, t, a in zip(todo.pmid, todo.title, todo.abstract)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=review, leave=False):
            row = {"pmid": futures[fut], "response": fut.result()}
            if row["response"].get("error", "").startswith(("401", "402", "403")):
                pool.shutdown(cancel_futures=True)
                raise RuntimeError(row["response"]["error"])
            f.write(json.dumps(row) + "\n")
            f.flush()
            done[row["pmid"]] = row
    return done


def run_reviews(records, reviews, build_body, out, workers=8):
    for review in tqdm(sorted(reviews), desc="reviews"):
        done = run_review(records, review, build_body, out, workers)
        failed = sum("error" in r["response"] for r in done.values())
        if failed:
            tqdm.write(f"{review}: {failed} of {len(done)} calls failed")


def add_scores(records, out, col, question="include"):
    """Merge cached Jev probabilities into `records` as `col`. Records without a score (failed call,
    refusal, not run) get -1, i.e. ranked last, and are counted, not dropped."""
    rows, cost, versions = [], 0.0, set()
    for path in out.glob("*.jsonl"):
        for r in map(json.loads, path.open()):
            resp = r["response"]
            rows.append({"review": path.stem, "pmid": r["pmid"],
                         col: resp.get("answers", {}).get(question, {}).get("noul")})
            cost += resp.get("usage", {}).get("cost") or 0
            versions.add(resp.get("model"))
    scores = pd.DataFrame(rows, columns=["review", "pmid", col]).drop_duplicates(["review", "pmid"], keep="last")
    records = records.drop(columns=[col], errors="ignore").merge(scores, on=["review", "pmid"], how="left")
    print(f"{col}: scored {int(records[col].notna().sum())} of {len(records)} records | model: {versions} | "
          f"cost: ${cost:.2f} | without a score (ranked last): {int(records[col].isna().sum())}")
    records[col] = records[col].fillna(-1.0)
    return records
