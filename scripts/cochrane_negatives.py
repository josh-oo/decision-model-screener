"""Negatives for the Cochrane extension from PubMed "similar articles" of each review's included studies.

  python scripts/cochrane_negatives.py                  # neighbours, filters, sampling, PubMed text
  python scripts/cochrane_negatives.py --per-positive 3 --rank 10

Needs data/processed/cochrane/{reviews,records,references}.jsonl from prepare_cochrane_data.py.
Writes data/processed/cochrane/negatives.jsonl (records schema, label_abstract = label_final = 0,
cochrane_list = "similar") and negatives_summary.json.

A similar article is a stand-in for a title/abstract exclusion. It is left out of the pool when it could
be eligible or is already a decision of the review:
  listed      PMID, DOI or title (>= 30 characters) in ANY reference list of the review (included, excluded,
              awaiting classification, ongoing, other versions, additional references, included/excluded reviews)
  fuzzy_title title nearly the same as a listed one (British/US spelling, hyphens, truncation; many listed
              reports have no PMID, e.g. "randomised" vs "randomized")
  same_authors shares >= 2 authors with one listed report: usually another report of the same study (design
              paper, secondary analysis); also drops some unrelated papers of the same group, the safe direction
  registry_id a trial registration number in the abstract that appears on the review's pages
  version     another version of the review itself (Cochrane record with the review's title)
  after       entered PubMed in or after the month of the review's last search (the search could not see it)
  commentary  comment, erratum, letter, editorial, news, retraction notice: often attached to a listed study
  no_abstract no abstract in PubMed (train records need an abstract)
Every count is reported in the summary; nothing is dropped silently.
"""
import argparse, json, os, random, re, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from screening import RAW, fetch_pubmed  # noqa: E402
import fulltext_criteria as F  # noqa: E402

load_dotenv(ROOT / ".env")
API_KEY = os.getenv("NCBI_API_KEY")
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
DIR = ROOT / "data" / "processed" / "cochrane"
FULL = RAW / "cochrane_full"
NEIGHBORS = RAW / "pubmed_neighbors.jsonl"
SUMMARY = RAW / "pubmed_summary.jsonl"
AUTHORS = RAW / "pubmed_authors.jsonl"
REGISTRY = re.compile(r"\b(NCT\d{8}|ISRCTN\d{8}|ACTRN\d{14}|ChiCTR[-A-Z0-9]{6,}|DRKS\d{8}|NTR\d{3,5}|"
                      r"CTRI/\d{4}/\d+/\d+|UMIN\d{9}|IRCT\d{6,}N?\d*|\d{4}-\d{6}-\d{2})\b")
KEEP_NEIGHBORS = 50
COMMENTARY = {"Comment", "Published Erratum", "Erratum", "Retraction of Publication", "Retracted Publication",
              "Expression of Concern", "Editorial", "News", "Letter", "Newspaper Article"}
MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov",
                                     "dec"], 1)}
MAX_GAP = 48  # months from last search to publication; Cochrane median is 8, 99th percentile about 40
# "13 August 2018", "August 2018", "August 13, 2018", "November week 2, 2015", "Aug 2018"
DATE = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
                  r"(?:\s+(?:week\s+)?\d{1,2}(?:st|nd|rd|th)?)?,?\s+((?:19|20)\d\d)\b", re.I)
YEAR = re.compile(r"\b((?:19|20)\d\d)\b")


def post(endpoint, data, tries=6):
    if API_KEY:
        data = dict(data, api_key=API_KEY)
    for attempt in range(tries):
        try:
            r = requests.post(EUTILS + endpoint, data=data, timeout=120)
            if r.ok:
                time.sleep(0.15 if API_KEY else 0.4)
                return r.json()
        except (requests.RequestException, ValueError):  # dropped connection, timeout, broken JSON
            pass
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{endpoint} failed {tries} times")


def cached(path):
    """Cache as {pmid: row}; a half-written last line (run killed mid-write) is ignored and fetched again."""
    out = {}
    if path.exists():
        for line in path.open():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[r["pmid"]] = r
    return out


def _elink(chunk):
    d = post("elink.fcgi", {"dbfrom": "pubmed", "db": "pubmed", "linkname": "pubmed_pubmed",
                            "cmd": "neighbor_score", "retmode": "json", "id": chunk})
    got = {}
    for ls in d.get("linksets", []):
        links = (ls.get("linksetdbs") or [{}])[0].get("links", [])
        got[ls["ids"][0]] = [l["id"] for l in links if l["id"] != ls["ids"][0]][:KEEP_NEIGHBORS]
    return chunk, got


def neighbors(seeds, batch=20, workers=6):
    """Similar articles per seed PMID, best first (cached, top KEEP_NEIGHBORS). elink takes about 0.2 s per ID and
    fails on 100-ID requests, so small batches run in parallel; only this thread writes the cache."""
    have = cached(NEIGHBORS)
    todo = [p for p in seeds if p not in have]
    chunks = [todo[i:i + batch] for i in range(0, len(todo), batch)]
    with NEIGHBORS.open("a") as f, ThreadPoolExecutor(workers) as ex:
        for n, (chunk, got) in enumerate(ex.map(_elink, chunks), 1):
            for p in chunk:
                rec = {"pmid": p, "links": got.get(p, [])}
                have[p] = rec
                f.write(json.dumps(rec) + "\n")
            f.flush()
            if n % 100 == 0:
                print(f"  elink {n * batch}/{len(todo)}", flush=True)
    return have


def summaries(pmids, batch=200):
    """Entrez date, publication types, DOI, title, journal per PMID (esummary, cached)."""
    have = cached(SUMMARY)
    todo = [p for p in pmids if p not in have]
    with SUMMARY.open("a") as f:
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            res = post("esummary.fcgi", {"db": "pubmed", "retmode": "json", "id": ",".join(chunk)}).get("result", {})
            for p in chunk:
                s = res.get(p, {})
                hist = {h["pubstatus"]: h["date"][:10] for h in s.get("history", [])}
                doi = next((a["value"] for a in s.get("articleids", []) if a["idtype"] == "doi"), None)
                rec = {"pmid": p, "entrez": hist.get("entrez") or hist.get("pubmed"),
                       "sortpubdate": (s.get("sortpubdate") or "")[:10] or None, "pubtype": s.get("pubtype", []),
                       "doi": doi, "title": s.get("title"), "journal": s.get("source"), "missing": not s or "error" in s}
                have[p] = rec
                f.write(json.dumps(rec) + "\n")
            if i // batch % 100 == 0:
                print(f"  esummary {i + len(chunk)}/{len(todo)}", flush=True)
    return have


def authors(pmids, batch=200):
    """Author names per PMID (esummary, cached separately from the summaries)."""
    have = cached(AUTHORS)
    todo = [p for p in pmids if p not in have]
    with AUTHORS.open("a") as f:
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            res = post("esummary.fcgi", {"db": "pubmed", "retmode": "json", "id": ",".join(chunk)}).get("result", {})
            for p in chunk:
                rec = {"pmid": p, "authors": [a["name"] for a in res.get(p, {}).get("authors", [])
                                              if a.get("authtype") == "Author"]}
                have[p] = rec
                f.write(json.dumps(rec) + "\n")
            f.flush()
            if i // batch % 100 == 0:
                print(f"  authors {i + len(chunk)}/{len(todo)}", flush=True)
    return have


def author_key(name):
    """'Biondi‐Zoccai G' / 'Biondi-Zoccai GG' -> 'biondizoccai g' (surname + first initial)."""
    parts = name.strip().split()
    if len(parts) < 2:
        return None
    sur, ini = re.sub(r"[^a-z]", "", " ".join(parts[:-1]).lower()), re.sub(r"[^A-Za-z]", "", parts[-1])[:1].lower()
    return f"{sur} {ini}" if sur and ini else None


def title_tokens(t):
    """Tokens for fuzzy matching: lower case, z -> s (randomized/randomised), no punctuation."""
    return norm_title(t).replace("z", "s").split()


def fuzzy_same(a, b):
    """Nearly the same title: equal after normalisation, one a prefix of the other (truncation), or token Jaccard >= .85."""
    if len(a) < 5 or len(b) < 5:
        return False
    if a == b or a[:len(b)] == b or b[:len(a)] == a:
        return True
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) >= 0.85


def norm_title(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def norm_doi(d):
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", (d or "").strip().lower()) or None


def latest(text):
    """Latest (year, month) in a text; bare years count as January (earlier cutoff = conservative)."""
    dates = [(int(y), MONTHS[m.lower()]) for m, y in DATE.findall(text)]
    if dates:
        return max(dates)
    years = [int(y) for y in YEAR.findall(text)]
    return (max(years), 1) if years else None


def search_date(doi):
    """(year, month) of the review's last search, where it came from, and the version's online publication
    (year, month). The latest date in the abstract's Search methods, else in Electronic searches; a date is only
    accepted if it is not after publication and at most MAX_GAP months before it. None if no date is plausible."""
    html = (FULL / (re.sub(r"[^a-zA-Z0-9]", "_", doi) + ".html")).read_text(errors="replace")
    pub = re.search(r'name="citation_online_date" content="(\d{4})/(\d{2})', html)
    pub = (int(pub.group(1)), int(pub.group(2))) if pub else None
    if pub is None:
        return None, "no publication date", None
    lines = F.to_lines_from_text(html, ".html")
    tried = []
    for head, src in (("### Search methods", "abstract"), ("#### Electronic searches", "electronic searches")):
        i = next((i for i, l in enumerate(lines) if l == head), None)
        if i is None:
            continue
        j = next((j for j in range(i + 1, len(lines)) if lines[j].startswith(("## ", "### ", "#### "))), len(lines))
        d = latest(" ".join(lines[i + 1:j]))
        if d and d <= pub and (pub[0] - d[0]) * 12 + pub[1] - d[1] <= MAX_GAP:
            return d, src, pub
        tried.append(f"{src}: {d}")
    return None, "no plausible date (" + "; ".join(tried) + ")", pub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=10, help="use the top-N similar articles of each included record")
    ap.add_argument("--per-positive", type=float, default=3, help="negatives sampled per included record")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    reviews = [json.loads(l) for l in (DIR / "reviews.jsonl").open()]
    refs = {r["doi"]: r for r in map(json.loads, (DIR / "references.jsonl").open())}
    pos = defaultdict(list)
    for r in map(json.loads, (DIR / "records.jsonl").open()):
        if r["label_final"] == 1:
            pos[r["review_id"]].append(r["pmid"])

    # Blacklist per review from every reference list.
    black = {}
    for rv in reviews:
        keys = {"title:" + norm_title(rv["title"])}
        for kind, studies in refs[rv["doi"]].items():
            if kind == "doi":
                continue
            for st in studies:
                for rep in st["reports"]:
                    if rep["pmid"]:
                        keys.add("pmid:" + rep["pmid"])
                    if rep["doi"]:
                        keys.add("doi:" + norm_doi(rep["doi"]))
                    if len(norm_title(rep["title"])) >= 30:
                        keys.add("title:" + norm_title(rep["title"]))
        black[rv["review_id"]] = keys
    listed_titles, listed_authors = {}, {}
    for rv in reviews:
        tl, al = [], []
        for kind, studies in refs[rv["doi"]].items():
            if kind == "doi":
                continue
            for st in studies:
                for rep in st["reports"]:
                    if rep.get("title"):
                        tl.append(title_tokens(rep["title"]))
                    ks = {k for k in map(author_key, rep.get("authors", [])) if k}
                    if ks:
                        al.append(ks)
        listed_titles[rv["review_id"]], listed_authors[rv["review_id"]] = tl, al
    safe = lambda d: re.sub(r"[^a-zA-Z0-9]", "_", d) + ".html"
    registry = {rv["review_id"]: set(REGISTRY.findall((RAW / "cochrane_html" / "html" / safe(rv["doi"])).read_text(errors="replace")))
                | set(REGISTRY.findall((FULL / safe(rv["doi"])).read_text(errors="replace"))) for rv in reviews}

    print("search dates ...", flush=True)
    sdate, ssrc, pubdate, no_date = {}, Counter(), {}, {}
    for rv in reviews:
        d, src, pub = search_date(rv["doi"])
        sdate[rv["review_id"]], pubdate[rv["review_id"]] = d, pub
        ssrc[src if d else "no plausible date"] += 1
        if d is None:
            no_date[rv["review_id"]] = src
    print(" ", dict(ssrc))

    # Plausibility of the search date: (a) months between last search and online publication (Cochrane: usually
    # < 1-2 years); (b) share of the review's included records that entered PubMed in or after the search month
    # (should be small; later reports of an included trial can be added, most cannot).
    seed_sm = summaries(sorted({p for v in pos.values() for p in v}))
    ym = lambda d: tuple(map(int, d.split("/")[:2])) if d else None
    months = lambda a, b: (a[0] - b[0]) * 12 + a[1] - b[1]
    plaus, implausible = {}, {}
    for rid, d in sdate.items():
        if d is None:
            continue
        gap = months(pubdate[rid], d) if pubdate[rid] else None
        dates = [ym(seed_sm[p].get("entrez") or seed_sm[p].get("sortpubdate")) for p in pos[rid]]
        dates = [x for x in dates if x]
        after = sum(x >= d for x in dates) / max(1, len(dates))
        plaus[rid] = {"gap_months": gap, "included_after_search": round(after, 3)}
        flags = []
        if after > 0.2:
            flags.append("over 20% of included records entered PubMed after the search date")
        if flags:
            implausible[rid] = flags
    gaps = sorted(v["gap_months"] for v in plaus.values() if v["gap_months"] is not None)
    print(f"  gap search->publication, months: median {gaps[len(gaps) // 2]}, 90th pct {gaps[int(len(gaps) * .9)]}, "
          f"max {gaps[-1]}; implausible: {len(implausible)}")
    # An implausible date is not used: the review gets no negatives (the conservative choice) and is listed.
    for rid in implausible:
        sdate[rid] = None

    seeds = sorted({p for v in pos.values() for p in v})
    print(f"elink for {len(seeds)} included records ...", flush=True)
    nb = neighbors(seeds)
    cand = {rid: {} for rid in pos}  # review -> {pmid: best rank}
    for rid, ps in pos.items():
        for p in ps:
            for rank, q in enumerate(nb[p]["links"][:args.rank], 1):
                cand[rid][q] = min(rank, cand[rid].get(q, rank))
    allc = sorted({q for c in cand.values() for q in c})
    print(f"esummary for {len(allc)} candidates ...", flush=True)
    sm = summaries(allc)

    # Filters, with counts by reason and by neighbour rank (rank shows how dense the neighbourhood is in listed papers).
    reasons, by_rank, kind_hits = Counter(), defaultdict(Counter), Counter()
    listed_kind = {}
    for rv in reviews:
        for kind, studies in refs[rv["doi"]].items():
            if kind != "doi":
                for st in studies:
                    for rep in st["reports"]:
                        if rep["pmid"]:
                            listed_kind.setdefault((rv["review_id"], rep["pmid"]), kind)
    pool = defaultdict(list)
    for rv in reviews:
        rid = rv["review_id"]
        cut = sdate[rid]
        if cut is None:  # no search or publication date: no safe cutoff, so no negatives for this review
            reasons["review_without_search_date"] += len(cand.get(rid, {}))
            continue
        for q, rank in cand.get(rid, {}).items():
            s = sm[q]
            keys = {"pmid:" + q}
            if s.get("doi"):
                keys.add("doi:" + norm_doi(s["doi"]))
            if len(norm_title(s.get("title"))) >= 30:
                keys.add("title:" + norm_title(s["title"]))
            hit = keys & black[rid]
            if hit:
                why = "version" if "title:" + norm_title(rv["title"]) in hit else \
                      "listed_pmid" if "pmid:" + q in hit else "listed_doi_or_title"
                if why == "listed_pmid":
                    kind_hits[listed_kind.get((rid, q), "?")] += 1
            elif s.get("missing"):
                why = "not_in_pubmed"
            elif tuple(map(int, (s.get("entrez") or s.get("sortpubdate") or "9999/12").split("/")[:2])) >= cut:
                why = "after_search"
            elif COMMENTARY & set(s["pubtype"]):
                why = "commentary"
            else:
                why = "candidate"
                pool[rid].append((q, rank))
            reasons[why] += 1
            by_rank[rank][why] += 1

    # Same study under another record: fuzzy title, shared authors (needs author names for the pool).
    print(f"authors for {sum(map(len, pool.values()))} pool records ...", flush=True)
    au = authors(sorted({q for v in pool.values() for q, _ in v}))
    for rid in list(pool):
        keep = []
        for q, rank in pool[rid]:
            tt = title_tokens(sm[q].get("title"))
            ka = {k for k in map(author_key, au[q]["authors"]) if k}
            if any(fuzzy_same(tt, lt) for lt in listed_titles[rid]):
                why = "fuzzy_title"
            elif any(len(ka & la) >= min(2, len(la)) for la in listed_authors[rid]):
                why = "same_authors"
            else:
                keep.append((q, rank))
                continue
            reasons["candidate"] -= 1
            reasons[why] += 1
            by_rank[rank]["candidate"] -= 1
            by_rank[rank][why] += 1
        pool[rid] = keep

    # Sample, then fetch title/abstract; records without abstract are replaced from the same pool.
    rng = random.Random(args.seed)
    for rid in pool:
        rng.shuffle(pool[rid])
    need = {rid: round(args.per_positive * len(pos[rid])) for rid in pool}
    # Fetch in rounds: about 1.3x what each review still needs, then top up the reviews that are still short.
    got = defaultdict(list)
    used = Counter()  # how far into its shuffled pool each review is
    no_abs, rnd = 0, 0
    while True:
        want = {rid: pool[rid][used[rid]:used[rid] + int((need[rid] - len(got[rid])) * 1.3) + 1]
                for rid in pool if len(got[rid]) < need[rid] and used[rid] < len(pool[rid])}
        if not want:
            break
        rnd += 1
        print(f"  abstracts round {rnd}: {sum(map(len, want.values()))} records for {len(want)} reviews", flush=True)
        text = fetch_pubmed(sorted({q for v in want.values() for q, _ in v}))
        for rid, batch in want.items():
            used[rid] += len(batch)
            for q, rank in batch:
                if len(got[rid]) == need[rid]:
                    break
                t = text[q]
                if not t["abstract"]:
                    no_abs += 1
                    continue
                if set(REGISTRY.findall(t["abstract"])) & registry[rid]:
                    reasons["registry_id"] += 1
                    continue
                got[rid].append({"review_id": rid, "record_id": "pmid:" + q, "pmid": q, "doi": sm[q].get("doi"),
                                 "title": t["title"], "abstract": t["abstract"], "missing": False,
                                 "label_abstract": 0, "label_final": 0, "cochrane_list": "similar",
                                 "neighbor_rank": rank, "pubtype": sm[q]["pubtype"]})
    short = Counter({rid: need[rid] - len(got[rid]) for rid in pool if len(got[rid]) < need[rid]})
    rows = [r for rv in reviews for r in got.get(rv["review_id"], [])]
    with (DIR / "negatives.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "rank": args.rank, "per_positive": args.per_positive, "seed": args.seed,
        "search_date_source": dict(ssrc),
        "search_date_none": no_date,
        "search_date_implausible": implausible,
        "search_date_gap_months_quantiles": {q: gaps[int(len(gaps) * q / 100)] for q in (10, 50, 90, 99)},
        "candidates_by_reason": dict(reasons),
        "listed_pmid_hits_by_list": dict(kind_hits),
        "share_listed_by_rank": {r: round((c["listed_pmid"] + c["listed_doi_or_title"] + c["fuzzy_title"]
                                           + c["same_authors"]) / sum(c.values()), 3)
                                 for r, c in sorted(by_rank.items())},
        "negatives": len(rows), "left_out_without_abstract": no_abs,
        "reviews_short_of_negatives": len(short), "negatives_missing": sum(short.values()),
        "pubtype_of_negatives": dict(Counter(p for r in rows for p in r["pubtype"]).most_common(15)),
    }
    (DIR / "negatives_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
