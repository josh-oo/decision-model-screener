"""Cochrane training extension: selection criteria plus included / excluded studies of Cochrane reviews.

  python scripts/prepare_cochrane_data.py              # select reviews, parse references, fetch PubMed
  python scripts/prepare_cochrane_data.py --no-fetch   # selection and counts only

Inputs (made by notebooks/data/CochraneDatasetParser.ipynb, not committed):
  data/raw/cochrane_html/html/*.html          /references page per review (latest version)
  data/processed/cochrane_criteria.jsonl      criteria section from the /full page
Outputs in data/processed/cochrane/ (same schema as data/processed/training/, split "train"):
  references.jsonl   parsed reference lists of every page (cache)
  reviews.jsonl      selected reviews: id, title, objectives, criteria as the four sections
  records.jsonl      one line per (review, PubMed report of an included or excluded study)
  summary.json       selection counts and reasons

Labels: included study -> label_final 1, excluded study -> label_final 0 (both are full-text decisions;
excluded studies passed title/abstract screening). label_abstract is null: Cochrane does not report
title/abstract decisions. Easy negatives (records excluded at title/abstract) are not in here yet.
"""
import argparse, json, re, sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from screening import RAW, fetch_pubmed  # noqa: E402

REFS_DIR = RAW / "cochrane_html" / "html"
CRITERIA = ROOT / "data" / "processed" / "cochrane_criteria.jsonl"
OUT = ROOT / "data" / "processed" / "cochrane"

# All CLEF TAR 2019 Task 2 Intervention reviews (val and test), every version is left out.
TAR = set("""CD005139 CD005253 CD006715 CD007868 CD008018 CD008170 CD008201 CD010019 CD010355 CD010526
CD010778 CD011380 CD011436 CD011571 CD012120 CD012164 CD012223 CD012347 CD012521 CD012930 CD000996 CD001261
CD004414 CD006468 CD007867 CD009069 CD009642 CD010038 CD010239 CD010558 CD010753 CD011140 CD011768 CD011977
CD012069 CD012342 CD012455 CD012551""".split())
SECTIONS = {"Types of studies": "studies", "Types of participants": "participants",
            "Types of interventions": "interventions", "Types of outcome measures": "outcomes"}
# Sentences written after screening (see TRAINING_NOTES.md, "Leave out").
POST_HOC = re.compile(r"differences between (the )?protocol and (the )?review|this update|updated review|"
                      r"previous (version|update)|original (review|version)|post[\s‐-]?hoc", re.I)
# Every reference list on the page: included, excluded, awaitingAssessment, ongoing, otherVersions,
# additionalReferences, includedReviews, excludedReviews. Records come from included/excluded only; all lists
# are used to keep listed papers out of the sampled negatives.
REF_LIST = re.compile(r'class="bibliographies references_(\w+)"')
KIND = {"includedStudies": "included", "excludedStudies": "excluded"}


def parse_references(path):
    """All reference lists of one /references page; per study its reports (PMID, DOI, title)."""
    text = path.read_text(errors="replace")
    doi = re.search(r'name="citation_doi" content="([^"]+)"', text).group(1)
    soup = BeautifulSoup(text, "html.parser")
    out = {"doi": doi}
    for cls in sorted(set(REF_LIST.findall(text))):
        studies = []
        for st in soup.find_all("div", class_="references_" + cls):
            h = st.find("h4")
            reports = []
            for b in st.find_all("div", class_="bibliography-section"):
                pm = [re.search(r"pubmed/(\d+)", a["href"]).group(1)
                      for a in b.find_all("a", href=re.compile(r"ncbi\.nlm\.nih\.gov/pubmed/\d+"))]
                dois = [a["href"].split("doi.org/")[-1] for a in b.find_all("a", href=re.compile(r"doi\.org/10\."))]
                t = b.find("span", class_="citation-title") or b.find("span", class_="citation")
                # Authors: the citation text before the title, e.g. "Gutin B, Yin Z, Johnson M, Barbeau P."
                cit = b.find("div", class_=re.compile(r"^citation-"))
                authors = ""
                if cit is not None:
                    for node in cit.children:
                        if getattr(node, "name", None) == "span":
                            break
                        authors += node.get_text(" ") if hasattr(node, "get_text") else str(node)
                reports.append({"pmid": pm[0] if pm else None, "doi": dois[0] if dois else None,
                                "title": t.get_text(" ", strip=True) if t else None,
                                "authors": [a.strip() for a in authors.strip().rstrip(".").split(",")
                                            if a.strip() and a.strip() != "et al"]})
            studies.append({"study": h.get_text(" ", strip=True) if h else None, "reports": reports})
        out[KIND.get(cls, cls)] = studies
    return out


def references():
    path = OUT / "references.jsonl"
    if not path.exists():
        with Pool() as pool:
            rows = pool.map(parse_references, sorted(REFS_DIR.glob("*.html")), chunksize=20)
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return {r["doi"]: r for r in map(json.loads, path.open())}


def split_sections(criteria):
    """'#### Types of ...' headings -> four sections; deeper headings become 'Heading: text' lines.
    None if the headings are not exactly the four standard intervention-review sections."""
    blocks, cur = {}, None
    for line in criteria.split("\n"):
        if line.startswith("### "):
            continue
        if line.startswith("#### "):
            cur = line[5:].strip()
            if cur in blocks:
                return None
            blocks[cur] = []
        elif cur is None:
            return None
        elif line.startswith("#####"):
            blocks[cur].append(line.lstrip("#").strip() + ":")
        elif line.strip():
            blocks[cur].append(line.strip())
    if list(blocks) != list(SECTIONS):
        return None
    out = {}
    for name, lines in blocks.items():
        text = ""
        for l in lines:  # glue "Heading:" onto the next line
            text += (" " if text.endswith(":") else "\n" if text else "") + l
        out[SECTIONS[name]] = text.strip()
    return out


def quality(row):
    """Reason the criteria are not usable, or None."""
    if not row["review"].startswith("CD"):
        return "not a CD review (methodology)"
    if row["review"] in TAR:
        return "CLEF TAR review (val/test)"
    if row["criteria_source"] != "full":
        return "criteria from abstract only"
    sec = split_sections(row["criteria"])
    if sec is None:
        return "not the four standard sections (e.g. DTA, extra subsections)"
    if any(len(sec[k]) < 20 for k in ("studies", "participants", "interventions")):
        return "a core section shorter than 20 characters"
    n = sum(map(len, sec.values()))
    if n < 300 or n > 8000:
        return "criteria length outside 300-8000 characters"
    if POST_HOC.search(row["criteria"]):
        return "post-hoc notes in the criteria"
    if not row.get("objectives"):
        return "no objectives"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true", help="selection and counts only, no PubMed calls")
    ap.add_argument("--min-included", type=int, default=3,
                    help="minimum included studies with a PubMed ID (to have positives at all)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    crit = [json.loads(l) for l in CRITERIA.open()]
    refs = references()
    reasons, selected = Counter(), []
    for row in crit:
        why = quality(row)
        if why is None:
            ref = refs.get(row["doi"])
            n_inc = sum(any(r["pmid"] for r in s["reports"]) for s in ref.get("included", [])) if ref else 0
            if ref is None:
                why = "no /references page"
            elif n_inc < args.min_included:
                why = f"fewer than {args.min_included} included studies with PMID"
        reasons[why or "selected"] += 1
        if why is None:
            selected.append(row)
    print(f"{len(selected)} of {len(crit)} reviews selected")
    for k, v in reasons.most_common():
        print(f"  {v:6d}  {k}")

    # One record per (review, PubMed report). A PMID listed under both included and excluded of the same
    # review is ambiguous and left out (counted).
    pairs, conflicts, no_pmid = [], 0, Counter()
    for row in selected:
        ref = refs[row["doi"]]
        seen = {}
        for kind in ("included", "excluded"):
            for st in ref.get(kind, []):
                for rep in st["reports"]:
                    if not rep["pmid"]:
                        no_pmid[kind] += 1
                        continue
                    seen.setdefault(rep["pmid"], []).append((kind, st["study"], rep["doi"]))
        for pmid, hits in seen.items():
            kinds = {h[0] for h in hits}
            if len(kinds) > 1:
                conflicts += 1
                continue
            pairs.append((row, pmid, hits[0]))
    print(f"{len(pairs)} (review, PMID) pairs; {conflicts} left out as both included and excluded; "
          f"reports without PMID: {dict(no_pmid)}")

    summary = {"reviews_in": len(crit), "reviews_selected": len(selected), "selection": dict(reasons),
               "min_included": args.min_included, "pairs": len(pairs), "pmid_in_both_lists": conflicts,
               "reports_without_pmid": dict(no_pmid),
               "pairs_by_kind": dict(Counter(h[0] for _, _, h in pairs))}
    if args.no_fetch:
        print(json.dumps(summary, indent=2))
        return

    pubmed = fetch_pubmed(sorted({p for _, p, _ in pairs}))
    with (OUT / "reviews.jsonl").open("w") as f:
        for row in selected:
            f.write(json.dumps({"review_id": row["review"], "source": "cochrane", "split": "train",
                                "title": row["title"], "objectives": row["objectives"],
                                "criteria": split_sections(row["criteria"]), "doi": row["doi"],
                                "access_status": row["access_status"]}, ensure_ascii=False) + "\n")
    stats = Counter()
    with (OUT / "records.jsonl").open("w") as f:
        for row, pmid, (kind, study, doi) in pairs:
            p = pubmed[pmid]
            rec = {"review_id": row["review"], "record_id": "pmid:" + pmid, "pmid": pmid, "doi": doi,
                   "title": p["title"], "abstract": p["abstract"], "missing": p.get("missing", False),
                   "label_abstract": None, "label_final": int(kind == "included"),
                   "cochrane_list": kind, "study": study}
            stats[kind, "records"] += 1
            stats[kind, "no_abstract"] += not p["abstract"]
            stats[kind, "missing_in_pubmed"] += bool(p.get("missing"))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    summary["records"] = {f"{k}/{s}": v for (k, s), v in sorted(stats.items())}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
