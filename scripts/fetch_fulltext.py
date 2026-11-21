"""Full texts for records whose label is a full-text decision.

  python scripts/fetch_fulltext.py --subset pilot   # CLEF TAR val, records included at title/abstract (427)
  python scripts/fetch_fulltext.py --subset core    # every target below (~141k papers; ask before running)

Targets (one line per paper; a paper can be in several reviews and groups):
  ta_include_final_exclude  label_abstract 1, label_final 0                   (CLEF TAR, SYNERGY+)
  ta_include_final_include  label_abstract 1, label_final 1                   (positives for a full-text model)
  only_final_include        label_abstract null, label_final 1                (SYNERGY+ without abstract labels)
  cochrane_included         Cochrane extension, included studies             (label_final 1)
  cochrane_excluded         Cochrane extension, excluded studies             (label_final 0, passed title/abstract)
Left out: label_abstract null and label_final 0 (SYNERGY+), where the screening stage is unknown.

Sources, in order; the first that gives a full text wins:
  1. PMC open-access bucket on AWS (s3://pmc-oa-opendata, PMC<id>.<version>/PMC<id>.<version>.xml, latest
     version, plus its .json with the licence); PMCID from the NCBI ID converter or OpenAlex
  2. Europe PMC fullTextXML (JATS) by PMCID
  3. NCBI E-utilities efetch db=pmc (body only for open-access articles)
  4. PDF from OpenAlex open-access locations (best first); kept only if the file starts with %PDF.
     At most one request per host every 2 s; a host with 8 failures and no success is skipped from then on.
Paywalled papers are not fetched. Papers are fetched by 8 threads.

Everything is cached in data/raw/fulltext/ (not committed): resolve.jsonl (ID converter + OpenAlex per paper),
xml/<key>.xml, xml/<key>.meta.json (S3 licence), pdf/<key>.pdf, fetch.jsonl (one line per paper and attempt outcome). Re-running skips cached papers.
Outputs in data/processed/fulltext/: targets.jsonl, and summary_<subset>.json (coverage per group, split, label).
Downloaded files are data only: nothing here executes or imports them.
"""
import argparse, json, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "fulltext"
OUT = ROOT / "data" / "processed" / "fulltext"
ENV = dotenv_values(ROOT / ".env")
UA = {"User-Agent": "decision-model-screener/0.1 (research; full-text availability study)"}
MAX_PDF = 60 * 2**20


def norm_doi(d):
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", d.strip().lower()) if d else None


def paper_key(pmid, doi):
    return f"pmid_{pmid}" if pmid else "doi_" + re.sub(r"[^a-z0-9._-]", "_", doi) if doi else None


def build_targets():
    split = {}
    with open(ROOT / "data/processed/training/reviews.jsonl") as f:
        for line in f:
            r = json.loads(line)
            split[r["review_id"]] = f"{r['split']}/{r['source']}"
    papers = {}

    def add(r, group, split_name):
        pmid, doi = r.get("pmid"), norm_doi(r.get("doi"))
        key = paper_key(pmid, doi)
        if key is None:  # no identifier: nothing to resolve; counted in the summary
            key = "noid_" + r["record_id"]
        p = papers.setdefault(key, {"key": key, "pmid": pmid, "doi": doi, "title": r.get("title"), "groups": [],
                                    "reviews": [], "splits": [], "split_groups": []})
        p["doi"] = p["doi"] or doi
        for field, value in (("groups", group), ("reviews", r["review_id"]), ("splits", split_name),
                             ("split_groups", f"{split_name} | {group}")):
            if value not in p[field]:
                p[field].append(value)

    with open(ROOT / "data/processed/training/records.jsonl") as f:
        for line in f:
            r = json.loads(line)
            a, fin = r["label_abstract"], r["label_final"]
            group = {(1, 0): "ta_include_final_exclude", (1, 1): "ta_include_final_include",
                     (None, 1): "only_final_include"}.get((a, fin))
            if group:
                add(r, group, split[r["review_id"]])
    with open(ROOT / "data/processed/cochrane/records.jsonl") as f:
        for line in f:
            r = json.loads(line)
            add(r, "cochrane_" + r["cochrane_list"], "train/cochrane")
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "targets.jsonl", "w") as f:
        for p in papers.values():
            f.write(json.dumps(p) + "\n")
    return list(papers.values())


def load_jsonl(path):
    if not path.exists():
        return {}
    with open(path) as f:
        return {d["key"]: d for d in map(json.loads, f)}


def get(url, timeout=60, **kw):
    for attempt in range(4):
        try:
            r = requests.get(url, headers=UA, timeout=timeout, **kw)
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
            continue
        if r.status_code in (429, 500, 502, 503, 504) and attempt < 3:
            time.sleep(2 ** (attempt + 1))
            continue
        return r


def resolve_chunk(chunk):
    res = {p["key"]: {"key": p["key"], "pmcid": None, "openalex": None} for p in chunk}
    pmids = [p["pmid"] for p in chunk if p["pmid"]]
    if pmids:
        j = get("https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/",
                params={"ids": ",".join(pmids), "format": "json", "tool": "decision-model-screener"}).json()
        for rec in j.get("records", []):
            key = paper_key(str(rec.get("pmid")), None)
            if rec.get("pmcid") and key in res:
                res[key]["pmcid"] = rec["pmcid"]
    # OpenAlex: by PMID where there is one, else by DOI; 50 values per filter
    sel = "id,doi,ids,open_access,best_oa_location,locations"
    for field, values in (("pmid", pmids), ("doi", [p["doi"] for p in chunk if not p["pmid"] and p["doi"]])):
        for k in range(0, len(values), 50):
            j = get("https://api.openalex.org/works", params={
                "filter": f"{field}:" + "|".join(values[k:k + 50]), "per-page": 50, "select": sel,
                "api_key": ENV.get("OPENALEX_API_KEY")}).json()
            for w in j.get("results", []):
                pmid = ((w.get("ids") or {}).get("pmid") or "").rstrip("/").rsplit("/", 1)[-1] or None
                key = paper_key(pmid, None) if field == "pmid" else paper_key(None, norm_doi(w.get("doi")))
                if key in res:
                    res[key]["openalex"] = w
                    pmc = (w.get("ids") or {}).get("pmcid")
                    if pmc and not res[key]["pmcid"]:
                        res[key]["pmcid"] = pmc.rstrip("/").rsplit("/", 1)[-1]
    return list(res.values())


def resolve(papers):
    """PMCID (NCBI ID converter) and OpenAlex open-access info per paper, 200 papers per chunk. Cached in
    resolve.jsonl. A chunk that fails after retries is not written and is retried on the next run."""
    path = RAW / "resolve.jsonl"
    cache = load_jsonl(path)
    todo = [p for p in papers if p["key"] not in cache and (p["pmid"] or p["doi"])]
    print(f"resolve: {len(papers) - len(todo)} cached, {len(todo)} to do", flush=True)
    failed = 0
    with open(path, "a") as out:
        for i in range(0, len(todo), 200):
            try:
                rows = resolve_chunk(todo[i:i + 200])
            except (requests.RequestException, ValueError) as e:
                failed += 1
                print(f"  chunk {i} failed: {type(e).__name__}", flush=True)
                continue
            for d in rows:
                out.write(json.dumps(d) + "\n")
                cache[d["key"]] = d
            out.flush()
            if (i // 200) % 25 == 0:
                print(f"  resolved {min(i + 200, len(todo))}/{len(todo)}", flush=True)
    if failed:
        print(f"resolve: {failed} chunks failed, re-run to retry them", flush=True)
    return cache


S3 = "https://pmc-oa-opendata.s3.amazonaws.com/"


def try_s3(pmcid, path):
    """Latest version of the article in the PMC open-access bucket: XML, plus its JSON metadata (licence)."""
    r = get(S3, params={"list-type": "2", "prefix": pmcid + ".", "delimiter": "/"})
    if r.status_code != 200:
        return f"http_{r.status_code}"
    versions = re.findall(rf"<Prefix>({re.escape(pmcid)}\.(\d+))/</Prefix>", r.text)
    if not versions:
        return "not_in_bucket"
    name = max(versions, key=lambda v: int(v[1]))[0]
    x = get(f"{S3}{name}/{name}.xml", timeout=120)
    if x.status_code != 200 or b"<body" not in x.content:
        return f"http_{x.status_code}" if x.status_code != 200 else "no_body"
    path.write_bytes(x.content)
    m = get(f"{S3}{name}/{name}.json")
    if m.status_code == 200:
        path.with_suffix(".meta.json").write_bytes(m.content)
    return "ok"


def try_europepmc(pmcid, path):
    r = get(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML")
    if r.status_code == 200 and b"<body" in r.content:
        path.write_bytes(r.content)
        return "ok"
    return f"http_{r.status_code}" if r.status_code != 200 else "no_body"


def try_efetch(pmcid, path):
    params = {"db": "pmc", "id": pmcid.removeprefix("PMC"), "retmode": "xml"}
    if ENV.get("NCBI_API_KEY"):
        params["api_key"] = ENV["NCBI_API_KEY"]
    with NCBI_LOCK:  # NCBI allows 10 requests/s with a key
        time.sleep(0.12)
        r = get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi", params=params, timeout=120)
    if r.status_code == 200 and b"<body" in r.content:
        path.write_bytes(r.content)
        return "ok"
    return f"http_{r.status_code}" if r.status_code != 200 else "no_body"


NCBI_LOCK = threading.Lock()
HOSTS = defaultdict(lambda: {"lock": threading.Lock(), "last": 0.0, "ok": 0, "fail": 0})
HOSTS_LOCK = threading.Lock()
BLOCK_AFTER = 8  # failures without a single success


def try_pdf(url, path):
    host = urlparse(url).netloc
    with HOSTS_LOCK:
        h = HOSTS[host]
    if h["ok"] == 0 and h["fail"] >= BLOCK_AFTER:
        return "skipped_host_blocked"
    with h["lock"]:  # at most one request per host every 2 s
        wait = 2.0 - (time.time() - h["last"])
        if wait > 0:
            time.sleep(wait)
        h["last"] = time.time()
        try:
            r = requests.get(url, headers={**UA, "Accept": "application/pdf"}, timeout=60, stream=True,
                             allow_redirects=True)
            head = r.raw.read(5, decode_content=True)
            if r.status_code != 200:
                outcome = f"http_{r.status_code}"
            elif not head.startswith(b"%PDF"):
                outcome = "not_pdf"
            else:
                path.write_bytes(head + r.raw.read(MAX_PDF, decode_content=True))
                outcome = "ok"
        except requests.RequestException as e:
            outcome = "error_" + type(e).__name__
    h["ok" if outcome == "ok" else "fail"] += 1
    return outcome


def fetch_one(p, res):
    w = res.get("openalex") or {}
    attempts, source, file = [], None, None
    if res.get("pmcid"):
        xml = RAW / "xml" / f"{p['key']}.xml"
        for name, fn in (("s3", try_s3), ("europepmc", try_europepmc), ("efetch", try_efetch)):
            try:
                outcome = fn(res["pmcid"], xml)
            except requests.RequestException as e:
                outcome = "error_" + type(e).__name__
            attempts.append([name, outcome])
            if outcome == "ok":
                source, file = name, str(xml.relative_to(ROOT))
                break
    if source is None:
        urls = []
        for loc in [w.get("best_oa_location")] + (w.get("locations") or []):
            if loc and loc.get("pdf_url") and loc["pdf_url"] not in urls:
                urls.append(loc["pdf_url"])
        for url in urls[:4]:
            pdf = RAW / "pdf" / f"{p['key']}.pdf"
            outcome = try_pdf(url, pdf)
            attempts.append([url, outcome])
            if outcome == "ok":
                source, file = "pdf:" + urlparse(url).netloc, str(pdf.relative_to(ROOT))
                break
    oa = w.get("open_access") or {}
    return {"key": p["key"], "pmcid": res.get("pmcid"), "in_openalex": bool(w),
            "oa_status": oa.get("oa_status"), "license": (w.get("best_oa_location") or {}).get("license"),
            "source": source, "file": file, "attempts": attempts}


def fetch(papers, resolved, workers=8):
    (RAW / "xml").mkdir(parents=True, exist_ok=True)
    (RAW / "pdf").mkdir(parents=True, exist_ok=True)
    path = RAW / "fetch.jsonl"
    done = load_jsonl(path)
    # papers that resolve could not look up yet are left for the next run
    todo = [p for p in papers if p["key"] not in done and (p["key"] in resolved or p["key"].startswith("noid_"))]
    print(f"fetch: {len(papers) - len(todo)} cached or unresolved, {len(todo)} to do", flush=True)
    write_lock, n = threading.Lock(), [0]
    with open(path, "a") as out:
        def work(p):
            d = fetch_one(p, resolved.get(p["key"]) or {})
            with write_lock:
                out.write(json.dumps(d) + "\n")
                out.flush()
                done[p["key"]] = d
                n[0] += 1
                if n[0] % 500 == 0:
                    ok = sum(1 for x in done.values() if x["source"])
                    print(f"  fetched {n[0]}/{len(todo)}, full texts so far {ok}", flush=True)
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(work, todo))
    return done


def summarize(papers, done, subset):
    c = defaultdict(Counter)
    for p in papers:
        d = done.get(p["key"]) or {}
        src = d.get("source") or ""
        kind = "no_identifier" if p["key"].startswith("noid_") else \
            "pmc_xml" if src in ("s3", "europepmc", "efetch") else "oa_pdf" if src.startswith("pdf:") else \
            "oa_but_failed" if d.get("oa_status") not in (None, "closed") or d.get("pmcid") else "closed_or_unknown"
        for sg in p["split_groups"]:
            c[sg][kind] += 1
            c[sg]["papers"] += 1
        c["all"][kind] += 1
        c["all"]["papers"] += 1
        if src.startswith("pdf:"):
            c["pdf_hosts"][src[4:]] += 1
        for url, outcome in d.get("attempts", []):
            if url not in ("s3", "europepmc", "efetch") and outcome != "ok":
                c["pdf_failures"][outcome] += 1
    summary = {k: dict(v.most_common()) for k, v in c.items()}
    (OUT / f"summary_{subset}.json").write_text(json.dumps(summary, indent=2))
    for k, v in summary.items():
        if k not in ("pdf_hosts",):
            print(k, v)
    print("top pdf hosts", dict(Counter(summary.get("pdf_hosts", {})).most_common(10)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["pilot", "core"], required=True)
    args = ap.parse_args()
    papers = build_targets()
    print(f"targets: {len(papers)} papers -> {OUT / 'targets.jsonl'}")
    if args.subset == "pilot":
        papers = [p for p in papers if "val/tar2019" in p["splits"]
                  and set(p["groups"]) & {"ta_include_final_exclude", "ta_include_final_include"}]
    print(f"subset {args.subset}: {len(papers)} papers")
    RAW.mkdir(parents=True, exist_ok=True)
    resolved = resolve(papers)
    done = fetch(papers, resolved)
    summarize(papers, done, args.subset)


if __name__ == "__main__":
    sys.exit(main())
