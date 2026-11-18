"""Download and prepare title/abstract screening data for training and evaluating a decision model.

Runs from a fresh clone (`python scripts/prepare_training_data.py`). Downloads go to `data/raw/`,
outputs to `data/processed/training/`. Neither is committed; re-running reuses what is downloaded.

Splits (no Cochrane mining yet):
  train  SYNERGY+ official train split
  val    CLEF TAR 2019 Task 2 Intervention training reviews, minus reviews that are also test reviews
  test   CLEF TAR 2019 Task 2 Intervention test reviews ("tar2019"); SYNERGY+ official test split ("synergy")

Outputs:
  reviews.jsonl  one line per review: id, source, split, title, objectives, criteria
  records.jsonl  one line per (review, record): ids, title, abstract, labels
  summary.json   counts per split and source

Labels: `label_abstract` is the title/abstract decision where the source has one (CLEF TAR
abstract-level qrels; SYNERGY+ `label_abstract_included` for the datasets that have it), else null.
`label_final` is the decision after full text (CLEF TAR content-level qrels; SYNERGY+ `label_included`).
No record is dropped: records without abstract, or not found in the source, are kept with empty text.
"""
import csv, glob, json, os, sys, time, xml.etree.ElementTree as ET, zipfile
from collections import Counter
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from screening import RAW, download, fetch_pubmed  # noqa: E402

OUT = ROOT / "data" / "processed" / "training"
TAR = "https://raw.githubusercontent.com/CLEF-TAR/tar/master/2019-TAR/"
TAR_RAW = RAW / "tar2019"
SYNERGY_RAW = RAW / "synergy"

TAR_TRAIN = ["CD005139", "CD005253", "CD006715", "CD007868", "CD008018", "CD008170", "CD008201",
             "CD010019", "CD010355", "CD010526", "CD010778", "CD011380", "CD011436", "CD011571",
             "CD012120", "CD012164", "CD012223", "CD012347", "CD012521", "CD012930"]
TAR_TEST = ["CD000996", "CD001261", "CD004414", "CD006468", "CD007867", "CD009069", "CD009642",
            "CD010038", "CD010239", "CD010558", "CD010753", "CD011140", "CD011571", "CD011768",
            "CD011977", "CD012069", "CD012164", "CD012342", "CD012455", "CD012551"]
# Task 1 has the criteria (Objectives + Types of ...) per review; the test folder is spelled "protocals".
TAR_SPLITS = {
    "val": dict(reviews=[r for r in TAR_TRAIN if r not in TAR_TEST],
                qrels="Task2/Training/Intervention/qrels/full.train.int.{level}.2019.qrels",
                topics="Task2/Training/Intervention/topics/", criteria="Task1/Training/Intervention/protocols/"),
    "test": dict(reviews=TAR_TEST,
                 qrels="Task2/Testing/Intervention/qrels/full.test.intervention.{level}.2019.qrels",
                 topics="Task2/Testing/Intervention/topics/", criteria="Task1/Testing/Intervention/protocals/"),
}
SECTIONS = {"TypesofStudies": "studies", "TypesofParticipants": "participants",
            "TypesofInterventions": "interventions", "TypesofOutcomeMeasures": "outcomes"}


def get(path, local, optional=False):
    """Download a CLEF TAR file once. With optional=True, None if it does not exist upstream."""
    local.parent.mkdir(parents=True, exist_ok=True)
    try:
        return download(TAR + path, local)
    except requests.HTTPError as e:  # e.g. 404 for a review without criteria file
        if not optional:
            raise
        print(f"  not available: {path} ({e})")
        return None


def read_qrels(path):
    out = {}
    for line in path.read_text().split("\n"):
        if line.strip():
            review, _, pmid, label = line.split()
            out[(review, pmid)] = int(label) > 0
    return out


def tar_reviews_and_records(split, cfg):
    d = TAR_RAW / split
    labels = {lvl: read_qrels(get(cfg["qrels"].format(level=lvl), d / f"{lvl}.qrels")) for lvl in ("abs", "content")}
    assert labels["abs"].keys() == labels["content"].keys()
    reviews, pairs = [], []
    for rid in cfg["reviews"]:
        topic = get(cfg["topics"] + rid, d / "topics" / rid).read_text()
        title = topic.split("Title:", 1)[1].split("Query:", 1)[0].strip()
        crit_file = get(cfg["criteria"] + rid + ".xml", d / "criteria" / f"{rid}.xml", optional=True)
        objectives, criteria = None, None
        if crit_file is not None:
            root = ET.parse(crit_file).getroot()
            objectives = (root.findtext("Objectives") or "").strip() or None
            criteria = {name: (root.findtext(tag) or "").strip() for tag, name in SECTIONS.items()}
        reviews.append(dict(review_id=rid, source="tar2019", split=split, title=title,
                            objectives=objectives, criteria=criteria, domain="Health Sciences",
                            review_type="Cochrane intervention review"))
        pairs += [(rid, pmid) for (r, pmid) in labels["abs"] if r == rid]
    pubmed = fetch_pubmed(sorted({p for _, p in pairs}))
    records = []
    for rid, pmid in pairs:
        pm = pubmed[pmid]
        records.append(dict(review_id=rid, record_id=f"pmid:{pmid}", pmid=pmid, doi=None,
                            title=pm["title"], abstract=pm["abstract"], missing=bool(pm.get("missing")),
                            label_abstract=int(labels["abs"][(rid, pmid)]),
                            label_final=int(labels["content"][(rid, pmid)])))
    return reviews, records


def synergy_reviews_and_records():
    """All SYNERGY+ datasets, without the package's open-access/abstract filter (no record dropped)."""
    from synergy_dataset.base import download_raw_dataset_plus, download_reviews_csv, Dataset
    from synergy_dataset.extractors import _reconstruct_abstract
    from synergy_dataset.splits import TEST_SPLIT

    download_raw_dataset_plus(path=SYNERGY_RAW)
    download_reviews_csv(path=SYNERGY_RAW)
    with (SYNERGY_RAW / "reviews.csv").open(encoding="utf-8-sig", newline="") as f:
        review_meta = {r["Key"]: r for r in csv.DictReader(f)}
    root = SYNERGY_RAW / "synergy-dataset-plus"
    reviews, records = [], []
    for meta_path in sorted(root.glob("*/metadata.json")):
        name = meta_path.parent.name
        ds = Dataset(name, path=meta_path.parent)
        pub = ds.metadata["publication"]
        rm = review_meta.get(name, {})
        reviews.append(dict(review_id=name, source="synergy", split="test" if name in TEST_SPLIT else "train",
                            title=pub.get("title") or pub.get("display_name"),
                            objectives=(rm.get("Research Question") or "").strip() or None,  # research question(s)
                            criteria=pub.get("eligibility_criteria"),
                            domain=((pub.get("primary_topic") or {}).get("domain") or {}).get("display_name"),
                            review_type=rm.get("Review type")))
        works = {}
        for zf in glob.glob(str(meta_path.parent / "works_*.zip")):
            with zipfile.ZipFile(zf) as z:
                for member in z.namelist():
                    for w in json.loads(z.read(member)):
                        works[w["id"].lower()] = w
        for oid, lab in ds.labels.items():
            w = works.get(oid)
            aii = (w or {}).get("abstract_inverted_index_cleaned") or (w or {}).get("abstract_inverted_index")
            pmid = (lab["pmid"] or "").rstrip("/").rsplit("/", 1)[-1]  # stored as https://pubmed.ncbi.nlm.nih.gov/<id>
            records.append(dict(review_id=name, record_id=oid, pmid=pmid if pmid.isdigit() else None, doi=lab["doi"],
                                title=((w or {}).get("title") or "").strip(),
                                abstract=_reconstruct_abstract(aii) if aii else "",
                                abstract_source="synergy" if aii else None,
                                missing=w is None,
                                label_abstract=lab["label_abstract_included"],
                                label_final=lab["label_included"]))
    return reviews, records


def fill_from_pubmed(records):
    """SYNERGY+ ships abstracts only for open-access works. Fill the rest from PubMed where a PMID is known."""
    todo = [r for r in records if not r["abstract"] and r["pmid"] and not r["record_id"].startswith("pmid:")]
    pubmed = fetch_pubmed(sorted({r["pmid"] for r in todo}))
    for r in todo:
        pm = pubmed[r["pmid"]]
        if pm["abstract"]:
            r["abstract"], r["abstract_source"] = pm["abstract"], "pubmed"
        r["title"] = r["title"] or pm["title"]
    print(f"  PubMed: {sum(r['abstract_source'] == 'pubmed' for r in todo)} of {len(todo)} filled")


def norm_doi(doi):
    return (doi or "").lower().replace("https://doi.org/", "").replace("http://dx.doi.org/", "").strip() or None


def pubmed_by_doi(dois, path=RAW / "pubmed_by_doi.jsonl", batch=100):
    """{doi: {"pmid", "title", "abstract"} or None} via PubMed: search a batch of DOIs, fetch the hits and
    match them back by the DOI in the PubMed record (never by position). Cached, also for DOIs not in PubMed.
    NCBI allows 3 requests/s without a key; set NCBI_API_KEY in .env for 10/s."""
    have = {}
    if path.exists():
        have = {r["doi"]: r["hit"] for r in map(json.loads, path.open())}
    todo = sorted({d for d in dois if d} - set(have))
    key = {"api_key": os.getenv("NCBI_API_KEY")} if os.getenv("NCBI_API_KEY") else {}
    pause = 0.12 if key else 0.4
    eutils = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    def post(endpoint, data):
        err = None
        for attempt in range(5):
            try:
                r = requests.post(eutils + endpoint, data={**data, **key}, timeout=120)
                if r.ok:
                    time.sleep(pause)
                    return r
                err = f"{r.status_code} {r.text[:200]}"
            except requests.RequestException as e:  # dropped connection, timeout, truncated response
                err = repr(e)
            time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"NCBI {endpoint} failed: {err}")

    with path.open("a") as f:
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            term = " OR ".join(f'"{d}"[doi]' for d in chunk if '"' not in d)
            pmids = post("esearch.fcgi", {"db": "pubmed", "term": term, "retmax": 1000, "retmode": "json"}).json()["esearchresult"]["idlist"]
            found = {}
            if pmids:
                root = ET.fromstring(post("efetch.fcgi", {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}).content)
                for art in root:
                    rec = _parse_one(art)
                    for el in art.iter():
                        if (el.tag == "ArticleId" and el.get("IdType") == "doi") or (el.tag == "ELocationID" and el.get("EIdType") == "doi"):
                            found[norm_doi(el.text)] = rec
            for d in chunk:
                have[d] = found.get(d)
                f.write(json.dumps({"doi": d, "hit": have[d]}) + "\n")
            print(f"\r  PubMed by DOI: {min(i + batch, len(todo))}/{len(todo)}", end="", flush=True)
    print()
    return have


def _parse_one(art):
    """pmid/title/abstract of one PubmedArticle element (same text format as screening.parse_pubmed_xml)."""
    title = art.find(".//ArticleTitle")
    parts = []
    for a in art.findall(".//Abstract/AbstractText"):
        text = "".join(a.itertext()).strip()
        parts.append(f"{a.get('Label')}: {text}" if a.get("Label") else text)
    return {"pmid": art.findtext(".//PMID"), "title": "".join(title.itertext()).strip() if title is not None else "",
            "abstract": "\n".join(p for p in parts if p)}


def fill_from_pubmed_by_doi(records):
    """Records still without abstract and without PMID: look the DOI up in PubMed."""
    todo = [r for r in records if not r["abstract"] and not r["pmid"] and norm_doi(r["doi"])
            and not r["record_id"].startswith("pmid:")]
    hits = pubmed_by_doi([norm_doi(r["doi"]) for r in todo])
    n = 0
    for r in todo:
        hit = hits.get(norm_doi(r["doi"]))
        if hit:
            r["pmid"] = hit["pmid"]
            if hit["abstract"]:
                r["abstract"], r["abstract_source"] = hit["abstract"], "pubmed"
                n += 1
            r["title"] = r["title"] or hit["title"]
    print(f"  PubMed by DOI: {n} of {len(todo)} filled")


def fill_from_openalex(records, path=SYNERGY_RAW / "openalex_abstracts.jsonl", batch=100):
    """Fallback for works not in PubMed (e.g. software engineering, education): fill remaining SYNERGY+
    abstracts from the OpenAlex API. Cached and resumable: OpenAlex allows
    1,000 calls/day without a key; set OPENALEX_API_KEY in .env for more. Stops at the limit and
    reports how many are left; re-run the script to continue."""
    from synergy_dataset.extractors import _reconstruct_abstract

    have = {}
    if path.exists():
        have = {r["id"]: r["abstract"] for r in map(json.loads, path.open())}
    todo_ids = sorted({r["record_id"].rsplit("/", 1)[-1].upper() for r in records
                       if not r["abstract"] and not r["record_id"].startswith("pmid:")} - set(have))
    params = {"select": "id,abstract_inverted_index", "per-page": batch}
    if os.getenv("OPENALEX_API_KEY"):
        params["api_key"] = os.getenv("OPENALEX_API_KEY")
    stopped = None
    with path.open("a") as f:
        for i in range(0, len(todo_ids), batch):
            chunk = todo_ids[i:i + batch]
            for attempt in range(4):
                try:
                    resp = requests.get("https://api.openalex.org/works", timeout=120,
                                        params={**params, "filter": "openalex_id:" + "|".join(chunk)})
                except requests.RequestException:  # dropped connection, timeout
                    if attempt == 3:
                        raise
                    time.sleep(5 * (attempt + 1))
                    continue
                if resp.status_code != 429 and resp.status_code < 500:
                    break
                if resp.status_code == 429 and int(resp.headers.get("x-ratelimit-remaining", 1)) <= 0:
                    break  # daily limit, not a burst limit
                time.sleep(5 * (attempt + 1))
            if resp.status_code == 429:
                stopped = len(todo_ids) - i
                break
            resp.raise_for_status()
            got = {w["id"].rsplit("/", 1)[-1].upper(): w.get("abstract_inverted_index") for w in resp.json()["results"]}
            for wid in chunk:  # works without abstract are cached as "" so they are not asked again
                have[wid] = _reconstruct_abstract(got[wid]) if got.get(wid) else ""
                f.write(json.dumps({"id": wid, "abstract": have[wid]}) + "\n")
            time.sleep(0.15)
    n = 0
    for r in records:
        if not r["abstract"] and not r["record_id"].startswith("pmid:"):
            text = have.get(r["record_id"].rsplit("/", 1)[-1].upper())
            if text:
                r["abstract"], r["abstract_source"] = text, "openalex"
                n += 1
    print(f"  OpenAlex: {n} filled" + (f"; daily limit reached, {stopped} works not yet asked. "
                                         "Re-run later or set OPENALEX_API_KEY." if stopped else ""))
    return stopped


def write_jsonl(path, rows):
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reviews, records = [], []
    for split, cfg in TAR_SPLITS.items():
        print(f"CLEF TAR 2019 {split} ...")
        rv, rc = tar_reviews_and_records(split, cfg)
        reviews += rv
        records += rc
    print("SYNERGY+ ...")
    rv, rc = synergy_reviews_and_records()
    fill_from_pubmed(rc)
    fill_from_pubmed_by_doi(rc)
    openalex_left = fill_from_openalex(rc)
    reviews += rv
    records += rc
    for r in records:
        if r["abstract"] and not r.get("abstract_source"):
            r["abstract_source"] = "pubmed"  # CLEF TAR records

    split_of = {r["review_id"]: (r["split"], r["source"]) for r in reviews}
    assert len(split_of) == len(reviews), "review id used twice"
    write_jsonl(OUT / "reviews.jsonl", reviews)
    write_jsonl(OUT / "records.jsonl", records)

    summary = {}
    for (split, source) in sorted(set(split_of.values())):
        ids = {r for r, v in split_of.items() if v == (split, source)}
        rec = [r for r in records if r["review_id"] in ids]
        summary[f"{split}/{source}"] = {
            "reviews": len(ids),
            "reviews_without_criteria": sorted(r["review_id"] for r in reviews if r["review_id"] in ids and not r["criteria"]),
            "reviews_with_abstract_labels": len({r["review_id"] for r in rec if r["label_abstract"] is not None}),
            "records": len(rec),
            "included_abstract": sum(r["label_abstract"] or 0 for r in rec),
            "included_final": sum(r["label_final"] for r in rec),
            "no_abstract": sum(not r["abstract"] for r in rec),
            "no_abstract_among_included_final": sum(not r["abstract"] for r in rec if r["label_final"]),
            "abstract_source": dict(Counter(r.get("abstract_source") for r in rec)),
            "not_found_in_source": sum(r["missing"] for r in rec),
        }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("excluded from val because they are test reviews:", sorted(set(TAR_TRAIN) & set(TAR_TEST)))
    if openalex_left:
        print(f"INCOMPLETE: {openalex_left} SYNERGY+ works not yet asked at OpenAlex (daily limit). Re-run to continue.")


if __name__ == "__main__":
    main()
