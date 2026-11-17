# Notes for training a screening decision model

What the Jev experiments and our discussions imply for training a new decision model later:
inputs, data sources, labels, evaluation and pitfalls. Evidence and exact numbers are in
[FINDINGS.md](FINDINGS.md) (entry numbers in brackets).

All evidence so far comes from one zero-shot model (`typesafe/jev-1.14`) on 20 Cochrane intervention
reviews from CLEF TAR 2019, often only the 10-review subset. It shows what information helps and how
an untrained model behaves. It does not prove what a trained model will learn; the checks at the end
are meant to settle that.

## Target use

- Screen titles and abstracts of new records against eligibility criteria, in unseen reviews and
  unseen domains.
- Inputs should be minable at scale from Cochrane reviews and available **before** screening.
- Also for study-based registers, where there is often no review title or abstract, only criteria
  or a scope description.

## Input design

### Recommended

| Part | Use | Evidence |
|---|---|---|
| Topic description: title, objectives, or a short background (1–2 paragraphs) | Always provide some topic text; treat title, objectives, register scope and a topic line derived from the criteria as the same input slot | Short review description is the best input so far (entry 3); per-section questions are weak without topic context and strong with it (entry 8, full-protocol and title runs) |
| Criteria: types of studies, participants, interventions | Core decision signal; interventions carries most of it, study type matters for content-level labels | Ablations (entry 8) |
| Criteria: outcomes | Leave out, or keep with low weight / as a "not reported" target | Hurts with final criteria, confirmed on held-out reviews (entry 8); neutral with protocol criteria |
| One question per criteria section | Useful for criterion-level labels and analysis, if every question also sees the topic (and ideally all criteria) | Without context a section is often unanswerable (e.g. study type gives a review of RCTs 0.92) |

Target size of the review information: about 2–3k characters, like the review abstract without
results. Total input per record then about 1.5–2k tokens.

### Leave out

| Part | Why |
|---|---|
| Search methods | Describes the candidate set, not what separates includes from excludes |
| Data collection and analysis | 44% of a protocol's tokens; risk of bias, statistics, GRADE; no screening value |
| Results, conclusions, *Characteristics of included studies*, references | Not available before screening; leaks information about included studies |
| Post-hoc notes inside the criteria ("See Differences between protocol and review", "In this updated review …") | Written after screening |
| Inline citations "(Author 2005)" in background text | No screening value, 5–15% of the text |

### What did not help (zero-shot)

- Full protocol as context (27k characters median): +0.4 / +2.3 MAP over criteria only, 5× the cost.
  Its value is the topic framing, not the length.
- Criteria in the question instead of the state: no consistent effect.
- A general "title/abstract stage, keep when in doubt" instruction: scores rise, order barely changes.
- Per-section weights tuned on 10 reviews: not stable; only "outcomes low or zero" held.
- Detailed full-text criteria instead of the short abstract criteria: worse (coarse criteria fit
  title/abstract screening better).

## Data sources

### Final review criteria vs protocols

| | Final review (*Criteria for considering studies*) | Protocol |
|---|---|---|
| Matches the labels | Yes, inclusion decisions were made against them | Not always (criteria changed in most reviews; text similarity to final 0.06–0.84) |
| Realistic as input at screening time | Partly; can carry hindsight (e.g. CD012069 added RCT arms after seeing the studies) | Yes |
| Availability | Every review | Most; not for old reviews (CD000996 and CD001261 started as reviews in 2000 / 1998) |
| Zero-shot result, same format | Similar to protocol (+0.9 / −1.6 MAP with all sections), but per-review swings both ways | Same; rules tuned on final criteria did not transfer |

**Recommendation:** train mainly on cleaned final criteria (labels fit), add the protocol as a second
input variant of the same review where available (augmentation), and **evaluate on protocol
criteria** (realistic input).

For updated reviews (`.pub3` and later) the screeners worked from the previous review version, not the
original protocol. Either use the previous version's Background/Objectives/Methods (never its
Results), or the original protocol; keep the choice consistent and documented.

### Extraction (what worked)

- Every Cochrane intervention review and protocol in our set has *Criteria for considering studies*
  with exactly four subsections: types of studies / participants / interventions / outcome measures.
  Exclusions are sentences inside the subsections. Split by heading; no model needed.
- Match pages by the `citation_doi` meta tag, not by file name; check the version suffix (`.pubN`).
  The version without suffix is usually the protocol, but not always (CD009069's protocol is `.pub2`).
  Check the page type ("Protocol – Intervention" vs "Review").
- Code: [scripts/fulltext_criteria.py](scripts/fulltext_criteria.py) (`extract`, `extract_protocols`,
  `extract_protocol_text`).
- Short background by rule (proposed, not yet run): sentences from the start of *Description of the
  condition* and *Description of the intervention* until about 500 characters each (max about 800),
  fall back to the Background intro when there are no subsections, strip citations. "First
  paragraph" alone fails (list lead-ins of 17 characters, paragraphs of 2,000+).

### Access and licensing

- The Cochrane Library blocks scripted access (Cloudflare; robots.txt). PMC's API returns full text
  for only 2 of our 20 reviews. Mass mining needs a text-and-data-mining agreement with
  Wiley/Cochrane; clarify before building the pipeline around it.
- The CLEF TAR "protocol" files (Task 1) are **not protocols**: they are the published review's
  criteria (94–100% overlap), with review-time traces and two extraction errors. Do not use them as
  protocols.
- Checked against Cochrane (2026-11-17): for 18 of 20 test reviews the Task 1 criteria are word for
  word the criteria of the review version we have (`.pub2`–`.pub6`). The 17 validation reviews (Task 1
  training files) agree with the "Selection criteria" in the Cochrane abstracts (PubMed) of the 2018/2019
  versions; their full text is not in PMC, so they are not checked word for word. **Three files have
  shifted sections** (subsections moved into the next field, so the real section is missing):
  CD010558 (interventions = participants' *Diagnosis*, outcomes = *Comorbidities*), CD012551
  (outcomes = interventions' *Comparisons*), CD012347 (interventions = participants' *Comorbidities*,
  outcomes = *Setting*). Left unchanged on purpose: CD010558 and CD012551 are in the test set, which
  stays the published benchmark, and CD012347 is in validation. The training data (SYNERGY+) is not
  affected. Keep this in mind when reading per-review results for these three.
- LGAR's criteria are hand-condensed from the full text; at least one omission found (CD012069 RCT
  arms). Do not treat them as ground truth.

## Labels

- **Review-level:** included studies (positive); *Characteristics of excluded studies* (negative).
  References map to PubMed via DOI/PMID for record titles and abstracts.
- **Criterion-level, almost free:** exclusion reasons ("not an RCT", "wrong population", "no CVC")
  map onto the four sections. Included studies are positive for all sections. Only meaningful if the
  model also sees the topic (see input design).
- **Label level:** excluded studies with reasons are **full-text** exclusions, i.e. hard negatives
  that passed title/abstract screening. Title/abstract exclusions are not documented. Easy negatives
  must come from elsewhere (re-running the documented search, or related PubMed records); otherwise
  the model only learns the fine full-text boundary, which is too strict for title/abstract screening.
- **"Not reported" target:** with full texts available, label whether the abstract shows the deciding
  information at all. Separates "fails the criterion" from "cannot be judged from the abstract"; this
  is the cause of the outcomes problem.
- **Label noise to expect:** in CLEF TAR, 427 of 937 abstract-level includes were excluded at full
  text; 2,183 of 41,996 records have no abstract.

## Behaviour vs information

Assessment from the experiments (see the discussion in FINDINGS.md, entry 8 and later):

- **Mostly behaviour and format, so trainable:** the criteria contain nearly all the information the
  better review abstract has; both approaches separate includes from excludes about equally well
  (mean AUC per review about 98); the loss is in ordering the top places. Jev applies detailed
  criteria too literally for the title/abstract stage, and an instruction cannot change that norm.
  A model trained on title/abstract labels should learn which criteria count at this stage.
- **Genuine information limits, not trainable:** abstracts that do not report outcomes or details;
  criteria that changed after the protocol (labels reflect decisions not in the text); label noise;
  records without an abstract.

## Register setting (no review title)

- Replace the title by whatever topic text exists: register scope, objectives, or a topic line
  derived from the criteria (first sentence of participants + interventions).
- **Context dropout in training:** randomly include or drop title, objectives, background and topic
  line per example; always keep the criteria.
- Train on (record, criteria set) pairs; at inference a record can be checked against many topics.
- Zero-shot cost of a missing title in the split format: about 1–2 MAP. Criteria only vs criteria +
  title + research questions in one question: about 4–5 MAP.

## Evaluation design

- **Remove the CLEF TAR reviews and all their versions and updates** from the training data.
- Hold out by Cochrane review group (unseen domains) and by time (train older, test newer).
- Test inputs: protocol criteria (realistic). Report by information level:

| Level | Context | Corresponds to |
|---|---|---|
| A | Criteria only | Register without scope text |
| B | Criteria + topic line derived from them | Register, automatic |
| C | + scope / objectives | Register with scope, or review question |
| D | + title + short background | Classic review screening |

- Metrics as in LGAR (MAP, recall@k%, WSS@95/100%, TNR@95%), macro over reviews, both label levels,
  with per-review spread; reviews with very few includes swing strongly.
- Combination rules (e.g. dropping outcomes) must be fixed or learned on the training source, then
  tested on held-out reviews; they did not transfer from final to protocol criteria.

## Checks to run once a pipeline exists

1. Criteria only vs criteria + short background/objectives, both trained. If the gap disappears,
   it was behaviour; if it stays, the background carries information.
2. Train on final criteria vs final + protocol, both tested on protocols.
3. Context dropout vs fixed context, tested at levels A–D.
4. Manual check of a sample of top false positives and missed includes: could criteria + record
   abstract have allowed the right decision? Gives the information ceiling.
5. Outcomes as a separate "not reported"-aware target vs dropped.
