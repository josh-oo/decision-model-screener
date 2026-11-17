"""Extract "Criteria for considering studies for this review" from the full text of each review.

Sources, per review, in this order:
  1. a file you saved by hand in data/raw/fulltext/: CDxxxxxx.html (browser "Save Page As"), .txt or .xml
  2. PMC full-text XML via E-utilities, which PMC only returns for open-access articles (cached as CDxxxxxx.pmc.xml)

Output: data/processed/tar2019_intervention_fulltext_criteria.json, {review: {"source", "doi", "criteria"}}.
Reviews without a usable source are listed at the end.

    python scripts/fulltext_criteria.py              # extract from what is on disk
    python scripts/fulltext_criteria.py --download   # first fetch missing review pages from the Cochrane Library
    python scripts/fulltext_criteria.py --open       # or: open the missing pages in your browser to save by hand

--download sends one plain request per missing review, 5 s apart, and stops at the first response
that is not the review page (e.g. a cookie or bot check). It does not try to get around such checks.
Check the Cochrane Library / Wiley terms of use before using it.
"""
import html, json, re, sys, time, webbrowser
from html.parser import HTMLParser
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import screening as S

FULLTEXT = S.RAW / "fulltext"
OUT = ROOT / "data" / "processed" / "tar2019_intervention_fulltext_criteria.json"
START = re.compile(r"^criteria for considering studies", re.I)
END = re.compile(r"^(search methods for identification of studies|search methods)\b", re.I)
BLOCK = {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "section", "br", "tr", "ul", "ol", "title", "sec", "list-item", "label"}
HEAD = {"h1", "h2", "h3", "h4", "h5", "h6", "title"}


class TextLines(HTMLParser):
    """HTML or JATS XML to text lines; headings are prefixed with '#' by level."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.lines, self.buf, self.skip, self.head = [], [], 0, None

    def flush(self):
        text = " ".join("".join(self.buf).split())
        if text:
            self.lines.append(("#" * (int(self.head[1]) if self.head and self.head[1:].isdigit() else 3) + " " if self.head else "") + text)
        self.buf = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "nav", "header", "footer"):
            self.skip += 1
        if tag in BLOCK:
            self.flush()
            if tag in HEAD:
                self.head = tag

    def handle_endtag(self, tag):
        if tag in ("script", "style", "nav", "header", "footer"):
            self.skip = max(0, self.skip - 1)
        if tag in BLOCK:
            self.flush()
            if tag in HEAD:
                self.head = None

    def handle_data(self, data):
        if not self.skip:
            self.buf.append(data)


def to_lines(path):
    return to_lines_from_text(path.read_text(errors="replace"), path.suffix)


def to_lines_from_text(text, suffix):
    if suffix == ".txt":
        return [l.strip() for l in text.splitlines() if l.strip()]
    if suffix == ".xml":   # JATS: keep only the body, the abstract also has a "Selection criteria" part
        m = re.search(r"<body[\s>].*</body>", text, re.S)
        if not m:
            return []
        text = m.group(0)
    p = TextLines()
    p.feed(text)
    p.flush()
    return [l for l in p.lines if l not in UI_LINES]


UI_LINES = {"Open in table viewer"}   # Cochrane page controls that end up in the text


def cut(lines):
    """Lines from the criteria heading up to the search-methods heading."""
    bare = [l.lstrip("# ") for l in lines]
    starts = [i for i, l in enumerate(bare) if START.match(l)]
    for s in starts:   # the first match can be a table-of-contents entry; take the first with real content
        ends = [i for i in range(s + 1, len(bare)) if END.match(bare[i])]
        if ends and ends[0] - s > 3:
            return "\n".join(lines[s:ends[0]])
    return None


def pmc_xml(review, pmcid):
    path = FULLTEXT / f"{review}.pmc.xml"
    if not path.exists():
        r = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                         params={"db": "pmc", "id": pmcid.removeprefix("PMC"), "retmode": "xml"}, timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)
        time.sleep(0.4)
    return path


def dois(abstracts):
    r = requests.get("https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/",
                     params={"ids": ",".join(a["pmcid"] for a in abstracts.values()), "format": "json"}, timeout=60)
    r.raise_for_status()
    by_pmc = {rec["pmcid"]: rec.get("doi") for rec in r.json()["records"]}
    return {review: by_pmc.get(a["pmcid"]) for review, a in abstracts.items()}


COCHRANE = FULLTEXT / "cochrane"   # pages saved by hand from the browser, any file name
CANONICAL = re.compile(r'<meta[^>]+name="citation_doi"[^>]+content="(10\.1002/14651858\.(CD\d+)\.pub\d+)"')


def saved_pages():
    """{review: (path, doi of the page)} for the pages in data/raw/fulltext/cochrane/, matched by the
    page's canonical DOI, not by file name."""
    pages = {}
    for path in sorted(COCHRANE.glob("*.htm*")):
        m = CANONICAL.search(path.read_text(errors="replace"))
        if not m:
            print(f"WARNING {path.name}: no Cochrane DOI found in the page, ignored")
            continue
        if m.group(2) in pages:
            print(f"WARNING {path.name}: second file for {m.group(2)}, ignored")
            continue
        pages[m.group(2)] = (path, m.group(1))
    return pages


def has_saved(review):
    return review in saved_pages() or any((FULLTEXT / f"{review}{ext}").exists() for ext in (".html", ".htm", ".txt", ".xml"))


FAILED = FULLTEXT / "_failed"   # responses that were not the review page, kept for inspection


def describe(r):
    """Log details of one response: status, redirects, headers, page title, start of the visible text."""
    title = re.search(r"<title[^>]*>(.*?)</title>", r.text, re.S | re.I)
    text = " ".join(re.sub(r"<script.*?</script>|<style.*?</style>|<[^>]+>", " ", r.text, flags=re.S | re.I).split())
    return {
        "status": r.status_code,
        "redirects": " -> ".join(f"{h.status_code} {h.headers.get('location', '')}" for h in r.history) or "-",
        "final_url": r.url,
        "content_type": r.headers.get("content-type", ""),
        "server": r.headers.get("server", ""),
        "bytes": len(r.content),
        "sent_user_agent": r.request.headers.get("user-agent", ""),
        "title": " ".join(title.group(1).split())[:120] if title else "",
        "text_start": text[:300],
    }


def download(doi, stop_on_failure=True, verbose=True):
    """Fetch the Cochrane Library /full page of each review that has no saved file yet.
    Returns one log row per review (a list of dicts). Responses that are not the review page are
    saved to data/raw/fulltext/_failed/ for inspection; with stop_on_failure the loop ends there."""
    session = requests.Session()
    log = []
    for review in sorted(doi):
        row = {"review": review, "doi": doi[review]}
        log.append(row)
        if has_saved(review):
            row["result"] = "skipped (already saved)"
            continue
        url = f"https://www.cochranelibrary.com/cdsr/doi/{doi[review]}/full"
        row["url"] = url
        try:
            r = session.get(url, timeout=60)
        except requests.RequestException as e:
            row["result"] = f"request error: {e!r}"
            print(f"{review}: {row['result']}")
            if stop_on_failure:
                break
            continue
        row.update(describe(r))
        if verbose:
            print(f"--- {review}  {url}")
            for k in ("status", "redirects", "final_url", "content_type", "server", "bytes", "sent_user_agent", "title", "text_start"):
                print(f"    {k:13s} {row[k]}")
        if not r.ok or doi[review] not in r.text:   # not the review page, e.g. a cookie or bot check
            FAILED.mkdir(parents=True, exist_ok=True)
            (FAILED / f"{review}.html").write_text(r.text)
            row["result"] = f"not the review page (saved to {FAILED.relative_to(ROOT)}/{review}.html)"
            print(f"    -> {row['result']}")
            if stop_on_failure:
                print("    Stopping. Set stop_on_failure=False to try the remaining reviews.")
                break
            time.sleep(5)
            continue
        (FULLTEXT / f"{review}.html").write_text(r.text)
        row["result"] = "saved, criteria found" if cut(to_lines_from_text(r.text, ".html")) else "saved, NO criteria found (parser needs a look)"
        print(f"    -> {row['result']}")
        time.sleep(5)
    return log


def open_missing(doi):
    """Open the Cochrane page of each review without a saved file in the default browser."""
    for review in sorted(doi):
        if not has_saved(review):
            webbrowser.open(f"https://www.cochranelibrary.com/cdsr/doi/{doi[review]}/full")
            time.sleep(1)
    print("Save each page as data/raw/fulltext/<CD number>.html (Save Page As, 'Web Page, HTML only').")


def setup():
    """(abstracts, doi) for the 20 reviews."""
    FULLTEXT.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    abstracts = S.fetch_review_abstracts(S.load_criteria())
    doi = dois(abstracts)
    missing_doi = [r for r, d in doi.items() if not d]
    assert not missing_doi, f"no DOI for {missing_doi}"
    return abstracts, doi


def extract(abstracts, doi):
    """Criteria section per review from saved files, else PMC open-access XML. Writes OUT; returns (out, missing)."""
    out, missing = {}, []
    pages = saved_pages()
    for review in sorted(abstracts):
        candidates = []
        if review in pages:
            path, page_doi = pages[review]
            if page_doi == doi[review]:
                candidates.append(path)
            else:
                print(f"WARNING {review}: {path.name} is {page_doi}, expected {doi[review]}; not used")
        candidates += [p for p in (FULLTEXT / f"{review}{ext}" for ext in (".html", ".htm", ".txt", ".xml")) if p.exists()]
        candidates.append(pmc_xml(review, abstracts[review]["pmcid"]))
        for path in candidates:
            criteria = cut(to_lines(path))
            if criteria:
                if path.suffix in (".html", ".htm") and doi[review] not in path.read_text(errors="replace"):
                    print(f"WARNING {review}: {path.name} does not mention {doi[review]}; check the version")
                out[review] = {"source": path.name, "doi": doi[review], "criteria": criteria}
                break
        else:
            missing.append(review)

    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"criteria found for {len(out)} of {len(abstracts)} reviews -> {OUT.relative_to(ROOT)}")
    for review, v in out.items():
        print(f"  {review}  {len(v['criteria']):6d} chars  from {v['source']}")
    if missing:
        print("missing (save the page by hand, or run with --open):")
        for review in missing:
            print(f"  {review}  https://doi.org/{doi[review]}")
    return out, missing


PROTOCOLS = S.RAW / "protocols" / "cochrane"
PROTOCOL_OUT = ROOT / "data" / "processed" / "tar2019_intervention_protocol_criteria.json"


def extract_protocols(folder=PROTOCOLS, out=PROTOCOL_OUT):
    """Criteria section of the protocol (or, where no protocol exists, the previous review version) of
    each review, from pages saved by hand in data/raw/protocols/cochrane/. Pages are matched by their
    citation_doi; the page type (Protocol / Review) and date are recorded."""
    res = {}
    for path in sorted(folder.glob("*.htm*")):
        text = path.read_text(errors="replace")
        doi = re.search(r'name="citation_doi"[^>]+content="([^"]+)"', text).group(1)
        date = re.search(r'name="citation_(?:publication_)?date"[^>]+content="([^"]+)"', text).group(1)
        plain = " ".join(re.sub(r"<script.*?</script>|<style.*?</style>|<[^>]+>", " ", html.unescape(text), flags=re.S).split())
        kind = re.search(r"Cochrane Database of Systematic reviews (\w+(?: \w+)?) - Intervention", plain)
        review = re.search(r"CD\d+", doi).group(0)
        assert review not in res, f"two pages for {review}"
        criteria = cut(to_lines(path))
        assert criteria, f"no criteria section in {path.name}"
        res[review] = {"source": path.name, "doi": doi, "date": date, "type": kind.group(1) if kind else "?", "criteria": criteria}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"{len(res)} reviews -> {out.relative_to(ROOT)}")
    return res


PROTOCOL_TEXT_OUT = ROOT / "data" / "processed" / "tar2019_intervention_protocol_fulltext.json"
_BODY_END = re.compile(r"^#{2,3} (Results|Discussion|Authors' conclusions|Acknowledgements|Appendices|References|"
                       r"Institutional users|Other access options|Purchase instant access|Contributions of authors|"
                       r"Declarations of interest|Sources of support|Notes|What's new|History)\b", re.I)


def extract_protocol_text(folder=PROTOCOLS, out=PROTOCOL_TEXT_OUT):
    """Background + Objectives + Methods of each protocol page (same pages as extract_protocols).
    For the two previous review versions this also stops before Results, so no included studies leak in."""
    res = {}
    for path in sorted(folder.glob("*.htm*")):
        text = path.read_text(errors="replace")
        doi = re.search(r'name="citation_doi"[^>]+content="([^"]+)"', text).group(1)
        review = re.search(r"CD\d+", doi).group(0)
        lines = to_lines(path)
        start = next(i for i, l in enumerate(lines) if l.startswith("## Background"))
        end = next((i for i in range(start + 1, len(lines)) if _BODY_END.match(lines[i])), len(lines))
        body = "\n".join(lines[start:end])
        assert "## Objectives" in body and "## Methods" in body, f"incomplete protocol text for {review}"
        res[review] = {"source": path.name, "doi": doi, "text": body}
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"{len(res)} reviews -> {out.relative_to(ROOT)}")
    return res


def main():
    abstracts, doi = setup()
    if "--download" in sys.argv:
        download(doi)
    if "--open" in sys.argv:
        open_missing(doi)
        return
    extract(abstracts, doi)


if __name__ == "__main__":
    main()
