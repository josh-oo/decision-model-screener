# Findings

Running record of the experiments: what was run, on which data, what came out, and what it means.
Newest entry at the bottom. Numbers are computed from the response caches in `results/`.

## Summary so far (2026-11-17)

MAP, macro over reviews. "All 20" = full benchmark; "subset" = the 10 reviews with under 1,000
records (3,637 records), used for cheap first tests. Abstract-level / content-level labels.

| Entry | Review information / how it is asked | All 20 | Subset | Full-run cost |
|---|---|---|---|---|
| 1 | LGAR: title, research questions, inclusion + exclusion criteria (baseline) | 57.9 / 60.7 | 70.7 / 77.2 | $1.80 |
| 2 | Title + full review abstract | 61.5 / 71.3 | 73.8 / 84.4 | $3.61 |
| 3 | **Title + review abstract without Main results / Conclusions** | **61.2 / 70.6** | **74.4 / 85.4** | **$2.06** |
| 5 | Title + research questions + full-text criteria | – | 72.0 / 82.1 | ($4.7 est.) |
| 5 | Full-text criteria only | – | 68.2 / 77.0 | ($4.5 est.) |
| 6 | Full-text criteria only, in the question instead of the state | – | 70.0 / 72.7 | ($4.5 est.) |
| 7 | Full-text criteria only + general task description | – | 64.8 / 77.6 | ($4.8 est.) |
| 8 | Full-text criteria, one question per section, product without outcomes | 57.0 / 65.1 | 69.9 / 77.1 | $3.47 |

Main points:

1. **Best input so far: review title + review abstract without results (entry 3).** Content-level
   +9.9 MAP over the LGAR input (95% CI +4.7 to +16.2, 15 of 18 reviews better), abstract-level +3.3
   (CI includes 0). It uses only text available before screening and costs about the same as the
   baseline.
2. **Coarse criteria work better than detailed ones** at the title/abstract stage. The review's own
   full-text criteria beat LGAR's condensed ones, but lose to the short abstract (entries 4–5).
3. **Outcome criteria hurt.** The outcomes question is the lowest-scoring section for most true
   includes; leaving it out helps, confirmed on 10 reviews not used to find this (entry 8).
4. **How the criteria are asked matters less than what the review information contains.**
   Placement (state vs question), a general task description, and splitting into questions did not
   beat one question with a short review description (entries 6–8).
5. **Weights per section cannot be tuned reliably on 10 reviews**; only "outcomes low or zero" is
   stable (entry 8).

## Common setup

Applies to every entry unless the entry says otherwise.

- **Benchmark:** CLEF TAR 2019 Task 2, Intervention test split: 20 Cochrane reviews, 41,996
  (review, record) pairs, 41,183 unique PMIDs. Record title and abstract come from PubMed. 2,183 pairs have
  no abstract and are scored on the title alone. 0 PMIDs are missing from PubMed.
- **Labels:** CLEF TAR abstract-level (`label_abs`, 20 reviews) and content-level (`label_content`,
  18 reviews; CD011140 and CD012342 have no content-level includes and are skipped).
- **Model:** `typesafe/jev-1.14`; every response reports `typesafe/jev-1.14-20261014`.
- **Question:** one `noul` question: *"The record should be included in the systematic review: it
  meets all inclusion criteria and none of the exclusion criteria."* The probability is the ranking
  score. Entries 1–5 use exactly this; entries 6–8 change how the question is asked and say so.
- **Subset runs:** new configurations are first run on the 10 reviews with under 1,000 records
  (CD000996, CD001261, CD004414, CD007867, CD010239, CD011140, CD011571, CD011977, CD012164,
  CD012551; 3,637 records; 9 with content-level includes). Comparisons on the subset always use
  the same records for every configuration. The subset has only small reviews and does not include
  CD012069 or CD006468.
- **Ranking:** score descending; ties broken by BM25 (query = review title), then PMID.
- **Metrics:** as in LGAR, macro-averaged over reviews: MAP, recall@k%, WSS@95/100%, TNR@95%. Also
  sensitivity/specificity at a fixed threshold of 0.05.
- **Zero-shot:** no labels are used to build any input.
- **Code:** [src/screening.py](src/screening.py); one notebook per configuration.

## Where the LGAR criteria and research questions come from

From the LGAR paper ([arXiv 2505.24757](https://arxiv.org/abs/2505.24757), Section 3.1):

- **Manual extraction from the full text** of each review, by the authors. No LLM was used.
- **Research questions:** only the primary ones. If a review has no explicit question, it was
  *inferred from the objectives* or taken from the protocol (protocols were available for TAR2019).
  This explains why `research_questions` and `objectives` are often almost the same (e.g. CD000996).
- **Criteria were edited:** separated from the explanations around them, *reformulated as key
  points*, several study-type or participant criteria merged into one, references to figures and
  tables removed. Criteria on metadata (e.g. publication dates) were left out because LGAR handles
  them at retrieval.
- **Agreement check:** a second author extracted 15 of the 57 reviews independently. All 23 research
  questions and 97 of 99 criteria matched.
- **Not reported:** which review version was used (the `link` points to a PMC version), and how the
  Guo et al. criteria were made. For all 20 TAR reviews, the PMC version is the same Cochrane
  version (`.pubN`) we use for the review abstracts and full texts. For CD006468 its search date
  (14 May 2018) equals the date limit of the CLEF TAR query, so there the version matches the labels;
  the other reviews were not checked this way.
- **LGAR prompt:** system message = review title + research questions + definition of relevant (all
  inclusion criteria met and no exclusion criterion met). User message = task, 0–19 scale, record
  title + abstract, inclusion criteria, exclusion criteria. We put everything in `state` instead.

**Related comparison in the paper:** the QA baseline of Akinseloyin et al. (2024) was run with two
sets of criteria on TAR2019: the *Selection criteria* section of the review's structured abstract
("theirs", no research questions) and LGAR's full-text criteria ("ours"). On the Intervention split,
full-text criteria were somewhat better: MAP 47.1 → 50.4, TNR@95% 67.1 → 68.4 (Table 7). This tests
the QA method, not LGAR or Jev, and only the criteria section, not the whole abstract. Its direction
is opposite to our entry 2. Entry 3 shows that *Main results* / *Conclusions* are not the reason; the
pattern analysis and entry 4 suggest the coarser criteria in the abstract fit title/abstract screening
better than detailed ones.

The paper also reports that research questions help LGAR (TAR2019 MAP 48.4 → 50.6; SYNERGY 36.8 →
40.7). It removes no criteria and does not remove the title.

## 1. Baseline: LGAR input (2026-11)

Notebook: [notebooks/baseline_experiments/01_lgar_baseline.ipynb](notebooks/baseline_experiments/01_lgar_baseline.ipynb). Cache: `results/baseline/`.

`state.review` = title, research questions, inclusion criteria, exclusion criteria (from LGAR
`info.json`). `state.record` = title, abstract.

0 failed calls. Cost $1.80 (42.9M input tokens).

| | Labels | MAP | R@5% | R@10% | R@20% | WSS@95% | TNR@95% | WSS@100% |
|---|---|---|---|---|---|---|---|---|
| BM25 | abs | 22.3 | 34.4 | 46.2 | 60.7 | 33.3 | 39.5 | 28.7 |
| BM25 | content | 17.0 | 30.2 | 47.0 | 65.2 | 43.6 | 49.5 | 44.6 |
| Jev, LGAR input | abs | 57.9 | 65.0 | 77.8 | 86.8 | 66.3 | 74.7 | 62.1 |
| Jev, LGAR input | content | 60.7 | 84.2 | 90.9 | 94.9 | 81.0 | 88.3 | 81.4 |

LGAR paper, same 20 reviews: LGAR MAP 56.8, fine-tuned BioBERT MAP 45.9. On all 31 TAR2019 test reviews:
LGAR MAP 50.6, TNR@95% 76.5. LGAR's own record files and labels may differ slightly from ours.

**Takeaway:** zero-shot Jev with the LGAR input matches LGAR's MAP (57.9 vs 56.8) and costs about $2
for the whole benchmark.

## 2. Review information: title + review abstract instead of research questions and criteria (2026-11)

Notebook: [notebooks/baseline_experiments/02_review_title_abstract.ipynb](notebooks/baseline_experiments/02_review_title_abstract.ipynb). Cache:
`results/review_title_abstract/`.

**One factor changed:** `state.review` = review title + the Cochrane review's own PubMed abstract
(found via the PMC link in the LGAR file, PMC ID → PMID). Records, record input, question and model
are the same as entry 1.

0 failed calls. Cost $3.61 (85.8M input tokens), twice the baseline: review abstracts are 4–6k
characters.

### Ranking (macro over reviews)

| | Labels | MAP | R@1% | R@5% | R@10% | R@20% | WSS@95% | TNR@95% | WSS@100% |
|---|---|---|---|---|---|---|---|---|---|
| LGAR input (entry 1) | abs | 57.9 | 31.5 | 65.0 | 77.8 | 86.8 | 66.3 | 74.7 | 62.1 |
| Title + abstract | abs | 61.5 | 32.7 | 65.4 | 78.9 | 88.8 | 70.1 | 78.8 | 66.3 |
| **Difference** | abs | **+3.6** | +1.2 | +0.4 | +1.1 | +2.0 | +3.8 | +4.1 | +4.2 |
| LGAR input (entry 1) | content | 60.7 | 45.3 | 84.2 | 90.9 | 94.9 | 81.0 | 88.3 | 81.4 |
| Title + abstract | content | 71.3 | 48.9 | 84.9 | 92.8 | 96.2 | 83.5 | 91.1 | 85.2 |
| **Difference** | content | **+10.6** | +3.6 | +0.7 | +1.9 | +1.3 | +2.5 | +2.8 | +3.8 |

Per-review change in MAP:

| Labels | Better / worse | Sign test (two-sided) | Mean ΔMAP, 95% bootstrap CI over reviews |
|---|---|---|---|
| abs | 16 / 4 | p = 0.012 | +3.5 [−1.9, +8.1] |
| content | 15 / 2 (1 tie) | p = 0.002 | +10.6 [+3.9, +17.5] |

Largest changes in MAP (abs / content):

- Gains: CD006468 +12.3 / +48.0, CD012455 +13.1 / +34.0, CD001261 +22.9 / +10.1, CD011977
  +18.5 / +18.9, CD010558 +0.9 / +20.8.
- Losses: CD011140 −36.2 / – (4 abstract-level includes), CD012342 −8.8 / – (6), CD004414 +1.7 / −21.2,
  CD010038 −1.5 / −4.3.

The reviews that got worse mostly have very few includes, where a single record moving changes MAP
a lot.

### At the fixed threshold 0.05

| | Labels | Sensitivity (mean / pooled) | Specificity (mean / pooled) | Flagged (mean / pooled) |
|---|---|---|---|---|
| LGAR input | abs | 92.3 / 89.9 | 75.6 / 77.7 | 27.8 / 23.8 |
| Title + abstract | abs | 97.0 / 98.8 | 63.3 / 69.0 | 39.6 / 32.5 |
| LGAR input | content | 98.0 / 91.8 | 74.0 / 77.1 | 27.8 / 23.8 |
| Title + abstract | content | 100.0 / 99.8 | 61.9 / 68.3 | 39.6 / 32.5 |

Scores are higher overall with the abstract (median 0.03 vs 0.02; mean of abstract-level includes 0.42
vs 0.37). At the same cut-off it therefore flags more records: higher sensitivity, lower
specificity. A fixed threshold does not carry over between input configurations; compare ranking
metrics, or pick the threshold per configuration.

### Caveats

- **The criteria are not really removed.** A Cochrane abstract has a *Selection criteria* section,
  plus *Objectives*, which is nearly the LGAR research question. This entry compares "LGAR criteria +
  research questions" with "the review's own summary, which includes a short version of the
  criteria". It does not compare "with criteria" with "without criteria".
- **Information from after screening.** *Main results* and *Authors' conclusions* report what the
  review found in its included studies (number of trials, participants, effects). No individual
  study is named and no labels are used, so the run is still zero-shot. But a screener would not
  have this text at screening time. *Resolved by entry 3: removing these sections does not lose the
  gain.*
- **Review version.** The PMC link points to the review version LGAR used. See "Where the LGAR
  criteria come from" for what was checked.
- **Longer input.** The abstract adds context and length together; this run cannot separate the two.
- **Small sample.** 20 reviews (18 with content-level labels). On abstract-level labels the bootstrap CI of
  ΔMAP includes 0.

### Conclusion

Jev ranks better with the review's own abstract than with the structured LGAR criteria and research
questions. The gain is clear on content-level labels (+10.6 MAP, 15 of 17 reviews better) and smaller
on abstract-level labels (+3.6 MAP, 16 of 20 better, CI includes 0). So Jev does not need criteria
written out as a structured list: free-text context about the review works at least as well.

This does not show that Jev can screen without criteria, because the abstract contains them.

Cost: twice the baseline ($3.61 vs $1.80).

## 3. Review abstract without Main results and Authors' conclusions (2026-11)

Notebook: [notebooks/baseline_experiments/03_review_title_abstract_no_results.ipynb](notebooks/baseline_experiments/03_review_title_abstract_no_results.ipynb).
Cache: `results/review_title_abstract_no_results/`.

**One factor changed relative to entry 2:** the review abstract keeps only Background, Objectives,
Search methods, Selection criteria, and Data collection and analysis. It drops *MAIN RESULTS* and
*AUTHORS' CONCLUSIONS*, the text a screener would not have. The section labels are the same in all
20 abstracts (CD007867 has no Search methods). The kept text is 1.3–2.9k characters, about 35% of
the full abstract.

**Full run (all 20 reviews):** 0 failed calls, $2.06 in total (subset first, then the other 10).

| | Labels | MAP | R@1% | R@5% | R@10% | R@20% | WSS@95% | TNR@95% | WSS@100% |
|---|---|---|---|---|---|---|---|---|---|
| LGAR input (entry 1) | abs | 57.9 | 31.5 | 65.0 | 77.8 | 86.8 | 66.3 | 74.7 | 62.1 |
| Full abstract (entry 2) | abs | 61.5 | 32.7 | 65.4 | 78.9 | 88.8 | 70.1 | 78.8 | 66.3 |
| Abstract w/o results | abs | 61.2 | 34.2 | 64.8 | 78.3 | 88.1 | 68.1 | 76.8 | 64.8 |
| LGAR input (entry 1) | content | 60.7 | 45.3 | 84.2 | 90.9 | 94.9 | 81.0 | 88.3 | 81.4 |
| Full abstract (entry 2) | content | 71.3 | 48.9 | 84.9 | 92.8 | 96.2 | 83.5 | 91.1 | 85.2 |
| Abstract w/o results | content | 70.6 | 50.9 | 85.9 | 92.3 | 95.7 | 82.9 | 90.4 | 83.5 |

Per-review change in MAP (bootstrap CI over reviews, two-sided sign test):

| Comparison | Labels | Mean ΔMAP [95% CI] | Better / worse | p |
|---|---|---|---|---|
| w/o results vs LGAR input | abs | +3.3 [−1.1, +7.5] | 13 / 7 | 0.26 |
| w/o results vs LGAR input | content | +9.9 [+4.7, +16.2] | 15 / 3 | 0.008 |
| w/o results vs full abstract | abs | −0.3 [−2.2, +1.8] | 9 / 11 | 1.0 |
| w/o results vs full abstract | content | −0.7 [−3.4, +2.0] | 9 / 8 | 1.0 |

On the 10 reviews outside the subset alone (run only after the subset result): +2.8 (abs, 7 of 10
better) and +11.6 (content, 7 of 9 better) against the LGAR input.

At threshold 0.05: sensitivity 97.7% / 100% (abs / content, mean over reviews), specificity 58.9% /
57.6%, 43.9% of records flagged.

**Conclusion:** dropping *Main results* and *Authors' conclusions* costs nothing measurable
(differences to the full abstract are within ±1 MAP, CIs around 0) and saves 43% of the cost. The
gain over the LGAR input therefore does not come from text a screener would not have. This is the
best configuration so far.

**Subset** (first run, for reference; the other subset entries are compared against these numbers):

| | Labels | MAP | R@10% | WSS@95% | TNR@95% |
|---|---|---|---|---|---|
| LGAR input | abs | 70.7 | 73.7 | 60.5 | 71.1 |
| Full abstract (entry 2) | abs | 73.8 | 76.9 | 64.0 | 75.0 |
| Abstract w/o results | abs | 74.4 | 75.8 | 63.8 | 74.8 |
| LGAR input | content | 77.2 | 88.6 | 79.1 | 88.0 |
| Full abstract (entry 2) | content | 84.4 | 92.0 | 83.3 | 92.5 |
| Abstract w/o results | content | 85.4 | 91.8 | 83.0 | 92.0 |

Abstract without results vs the LGAR input, MAP per review: better in 6 of 10 (abs) and 8 of 9 (content).

## Pattern analysis: why the review abstract beats the LGAR input (entries 1–2, all 20 reviews)

No new API calls. PubMed publication types fetched for all records (`data/raw/pubmed_meta.jsonl`).

- **The LGAR criteria are sometimes incomplete.** CD012069 (methylphenidate, adverse events in
  non-randomised studies, 320 abstract-level includes): LGAR lists only non-randomised designs, but
  the review also included the methylphenidate arms of RCTs (stated in both the abstract and the full
  text). With the LGAR input the 84 RCT includes had a median score of 0.05 (around the 60th
  percentile of the ranking); with the review abstract about 0.45 (top 13%).
- **Some LGAR criteria are full-text-level detail that the title/abstract screeners did not apply.**
  CD001261 excludes "trials that reported only on immunogenicity" (this is in the review itself);
  the screeners passed many such trials at abstract level and Jev scored them 0.04 with the LGAR
  input. CD011977 requires a specific outcome (visual acuity change at 12 months) that most abstracts
  cannot show. The abstract's *Selection criteria* omit such details.
- **The longer and more detailed the LGAR text, the smaller the gain from the abstract** (rank
  correlation −0.59 abs, −0.46 content, n = 20, descriptive).
- **The reviews that got worse are mostly noise.** The two largest losses (CD011140, CD012342) have
  no content-level includes; their 4 and 6 abstract-level includes are ranked in the top 3% in both
  configurations.
- **CD006468** (+48 MAP content): with the LGAR input the top 5 records were earlier versions of the
  same Cochrane review and other systematic reviews; with the abstract they dropped below the trials.
  Across all reviews this effect is not general: same-topic reviews are top-ranked false positives in
  both configurations.
- **Records without an abstract** rank lower with the review abstract (51 abstract-level includes,
  median rank from the top 13% to the top 33%; 37 of them in CD012069).

## 4. Full-text criteria: how they were obtained (2026-11)

Notebook: [notebooks/baseline_experiments/04_fulltext_criteria_extraction.ipynb](notebooks/baseline_experiments/04_fulltext_criteria_extraction.ipynb). Code:
[scripts/fulltext_criteria.py](scripts/fulltext_criteria.py). Output:
`data/processed/tar2019_intervention_fulltext_criteria.json` (not committed).

- Section *Criteria for considering studies for this review*, taken as written from the Cochrane full
  text of each review.
- PMC's API gives full text for only 2 of the 20 reviews (CD001261, CD010038). The Cochrane Library
  blocks scripted requests (Cloudflare; its robots.txt also disallows automated agents), so the 20
  pages were saved by hand from the browser into `data/raw/fulltext/cochrane/`.
- Each page is matched to its review by its `citation_doi` tag; all 20 are the version in the DOI
  (same as the PMC link in the LGAR file). CD001261 matches the PMC full text apart from whitespace.
- All 20 have the same four subsections: *Types of studies / participants / interventions / outcome
  measures*. Length 1.5–6.4k characters, CD010038 12.2k; the LGAR criteria are 0.4–1.6k.
- Exclusions are written as sentences inside the subsections ("We excluded …"). All 19 LGAR
  exclusion points are contained in the full text; the full text has some more (e.g. CD010038
  excludes studies at high risk of bias on random sequence generation).
- A few sections contain notes written after screening (references to *Differences between protocol
  and review* in CD012069 and CD012164, to summary-of-findings tables in 4 reviews). These are about
  reporting outcomes, not about which studies were included.

## 5. Review information: full-text criteria (2026-11, subset)

Notebook: [notebooks/baseline_experiments/05_fulltext_criteria.ipynb](notebooks/baseline_experiments/05_fulltext_criteria.ipynb).
Caches: `results/review_fulltext_criteria/` (with research questions),
`results/review_fulltext_criteria_only/` (criteria only).

| `state.review` | MAP abs | MAP content | WSS@95% content | TNR@95% content | Cost (subset) |
|---|---|---|---|---|---|
| LGAR: title, research questions, criteria (entry 1) | 70.7 | 77.2 | 79.1 | 88.0 | $0.14 |
| Title + LGAR research questions + full-text criteria | 72.0 | 82.1 | 81.9 | 91.2 | $0.23 |
| Full-text criteria only | 68.2 | 77.0 | 83.0 | 92.2 | $0.22 |
| Title + abstract w/o results (entry 3) | 74.4 | 85.4 | 83.0 | 92.0 | $0.16 |

0 failed calls in both runs.

- The review's own criteria are better than LGAR's condensed version (+1.3 / +4.9 MAP with title and
  research questions), but worse than the short review abstract.
- **CD001261 shows why:** with the full-text criteria (which keep the "immunogenicity only"
  exclusion) abstract-level MAP is 51.6, below the LGAR input (56.1); with the abstract, which leaves
  this exclusion out, 82.4.
- Title + research questions add about 4–5 MAP on top of the criteria (72.0 vs 68.2; 82.1 vs 77.0).
  Both were removed together, so which of the two matters is open.
- CD012069, where LGAR's omission matters most, is not in the subset.

Discarded: `results/review_title_fulltext_criteria_partial/` holds 1,926 records of a mistaken
configuration (title + criteria, interrupted); not used anywhere.

## 6. How it is asked: full-text criteria in the question instead of the state (2026-11, subset)

Notebook: [notebooks/baseline_experiments/06_criteria_in_question.ipynb](notebooks/baseline_experiments/06_criteria_in_question.ipynb).
Cache: `results/fulltext_criteria_in_question/`.

**One factor changed relative to "full-text criteria only" (entry 5):** the criteria move word for word
from `state.review` into the `instructions` of the question; `state` holds only the record.

| | MAP abs | MAP content | WSS@95% abs / content | TNR@95% abs / content |
|---|---|---|---|---|
| Criteria in state | 68.2 | 77.0 | 61.2 / 83.0 | 72.1 / 92.2 |
| Criteria in question | 70.0 | 72.7 | 66.2 / 81.9 | 77.7 / 91.1 |

Better in 6 of 10 reviews (abs) and 3 of 9 (content). Single reviews swing strongly in both
directions (CD011140 51.0 → 81.7, CD012164 56.0 → 36.2 abs). Same cost (1,412 vs 1,424 tokens per
call). **No consistent effect of placement.**

Discarded: `results/lgar_criteria_in_question_with_title_rq_partial/` (3,062 records, a mistaken
configuration with title + research questions in the state; interrupted, not used).

## 7. How it is asked: general title/abstract task description (2026-11, subset)

Notebook: [notebooks/baseline_experiments/07_task_description.ipynb](notebooks/baseline_experiments/07_task_description.ipynb).
Cache: `results/fulltext_criteria_task_description/`.

**One factor changed relative to "full-text criteria only" (entry 5):** after the fixed question
sentence, the instructions add a review-independent task description: title/abstract screening, keep
when in doubt, criteria the abstract cannot show (outcomes, follow-up, risk of bias) do not count
against a record, exclude when a criterion is clearly failed.

| | MAP abs | MAP content | WSS@95% abs / content | TNR@95% abs / content |
|---|---|---|---|---|
| Criteria only | 68.2 | 77.0 | 61.2 / 83.0 | 72.1 / 92.2 |
| + task description | 64.8 | 77.6 | 60.6 / 80.8 | 71.0 / 89.7 |

Better in 3 of 10 (abs) and 4 of 9 (content). Jev follows the instruction (median score of includes
0.31 → 0.53, of the rest 0.03 → 0.05; share ≥ 0.05 from 42% to 55%), but the order barely changes
(rank correlation 0.92). **More lenient, not better at ranking.** +7% tokens.

## 8. How it is asked: one question per criteria section (2026-11)

Notebook: [notebooks/baseline_experiments/08a_split_questions.ipynb](notebooks/baseline_experiments/08a_split_questions.ipynb).
Cache: `results/fulltext_criteria_split_questions/`. Follow-ups in the same entry:
[08b](notebooks/baseline_experiments/08b_split_questions_with_title.ipynb) (+ review title),
[08c](notebooks/baseline_experiments/08c_split_questions_protocol_criteria.ipynb) (protocol criteria),
[08d](notebooks/baseline_experiments/08d_split_questions_protocol_context.ipynb) (full protocol as context).

**One factor changed relative to entry 6** (all criteria in one question, `state` = record only): four
`noul` questions in one call, one per section (`studies`, `participants`, `interventions`,
`outcomes`), each with its section word for word. Combinations of the four probabilities were fixed
before looking at results: product (main), minimum.

### Subset

| | MAP abs | MAP content | WSS@95% content | TNR@95% content |
|---|---|---|---|---|
| One question (entry 6) | 70.0 | 72.7 | 81.9 | 91.1 |
| Split, product of all 4 | 68.5 | 71.4 | 82.5 | 91.8 |
| Split, minimum | 57.6 | 56.6 | 74.0 | 82.6 |
| Ablation: product without outcomes | 69.9 | 77.1 | 82.5 | 91.6 |
| Ablation: interventions only | 60.7 | 63.8 | 75.7 | 84.6 |
| Ablation: outcomes only | 29.6 | 19.3 | 40.6 | 47.5 |

- Interventions carries most of the signal; without it MAP drops to 49.2 / 45.1. Study type matters at
  content level (without it 59.5). Participants adds little.
- **Outcomes hurts:** it is the lowest-scoring section for 98 of 129 content-level includes (median
  0.37 vs 0.63–0.85 for the other sections).
- **Weights per section, leave-one-review-out** (weighted product, grid {0, 0.25, 0.5, 1, 1.5, 2} per
  section, chosen on the other 9 reviews by mean MAP): 68.8 abs, 77.7 content, i.e. no better than
  equal weights (abs) or simply dropping outcomes (content). In-sample upper bound 71.7 / 80.7. The
  only stable choice in all 19 folds: outcomes weight 0 or 0.25. Other weights contradict each other
  between label levels (interventions high for abs, low for content). Note: weights fitted on other
  reviews' labels are cross-review tuning, not strictly zero-shot.
- Cost: 1,501 vs 1,412 tokens per call (+6%); 72 output tokens per call, currently free.

### All 20 reviews and held-out test

Before the other 10 reviews were run, the rule "product without outcomes" was fixed in the notebook.
0 failed calls, 0 missing answers, $3.47 in total (1,968 tokens per call).

| | Reviews | MAP abs | MAP content | WSS@95% abs / content | TNR@95% abs / content |
|---|---|---|---|---|---|
| LGAR input (entry 1) | all 20 | 57.9 | 60.7 | 66.3 / 81.0 | 74.7 / 88.3 |
| Split, product of all 4 | all 20 | 55.7 | 59.6 | 68.9 / 84.4 | 77.6 / 92.1 |
| Split, without outcomes | all 20 | 57.0 | 65.1 | 68.3 / 84.5 | 76.8 / 92.0 |
| Title + abstract w/o results (entry 3) | all 20 | 61.2 | 70.6 | 68.1 / 82.9 | 76.8 / 90.4 |
| LGAR input | held-out 10 | 45.2 | 44.2 | 72.2 / 82.9 | 78.4 / 88.6 |
| Split, product of all 4 | held-out 10 | 42.9 | 47.8 | 76.8 / 86.3 | 83.2 / 92.3 |
| Split, without outcomes | held-out 10 | 44.2 | 53.1 | 74.7 / 86.5 | 81.1 / 92.3 |
| Title + abstract w/o results | held-out 10 | 48.0 | 55.8 | 72.4 / 82.9 | 78.8 / 88.7 |

- **The outcomes rule holds on the held-out reviews at content level:** +5.3 MAP vs all 4 sections (6
  of 9 better), +9.0 vs the LGAR input (8 of 9 better, sign test p = 0.039). At abstract level only
  +1.4 vs all 4 (5 of 10 better). Outcomes is again the lowest section for 212 of 381 content-level
  includes.
- **The split approach stays behind entry 3 on MAP** (−4.2 abs, −5.5 content over all 20; entry 3
  better in 16 of 20 and 10 of 15 reviews) but is about 1.5 points better on WSS@95% / TNR@95%
  (content), i.e. at the end of the ranking. It costs 70% more.

### Why the review abstract beats the split questions (qualitative, 4 reviews in detail)

- **Both separate includes from excludes equally well overall:** mean AUC per review 97.8 (entry 3)
  vs 98.0 (split without outcomes); 341 of 510 content-level includes get almost the same rank. The
  MAP gap comes from the first 10–20 places of a few reviews (CD004414 +35, CD001261 +19, CD012455,
  CD010038, CD012164 +13–14 content).
- **One weak section decides the product.** CD004414: the interventions question gets 0.09–0.20 for
  every record, includes too (the section is a list of examples Jev cannot match to one record). The
  ranking then follows noise in that value; a narrative review comes 3rd. The single question with
  the review title and abstract scores the actual trials 0.66–0.77.
- **Vague sections add uncertainty to clear cases.** CD001261: participants = "Adults and
  children." gets only 0.47–0.61 for clear field trials, pushing them down (e.g. 0.89 → 0.33 for a
  Ty21a field trial). Two copies of the review itself rank 5th and 9th: the study-type question gives
  them 0.92 because the criterion is about RCTs and the record is a review of RCTs.
- **The single question judges the record as a whole** ("a typhoid field trial", "a melatonin RCT in
  the ICU"); the split questions judge each aspect without that context.
- **Where detail helps (CD010753, split +19.5 content):** the full-text interventions section names
  every antidepressant; with it the split recognises includes such as mianserin or trimipramine
  trials that the abstract approach ranks lower, while the abstract approach puts a review and a
  mechanism paper into its top 10.
- Limits: patterns from 4 reviews; review-type false positives at the top are not more common for
  the split approach overall (19 vs 23 across all top places).

## Open questions / next runs

1. **Fix the split-questions notebook:** its saved tables compare against the one-question and
   criteria-in-state runs, which exist only for the subset; on the held-out reviews those columns
   are wrong. Compare against entries 1 and 3 instead (free).
2. **Combination:** title + abstract without results in the state, plus the split questions without
   outcomes as a second signal. Tests whether the review description (good top of the ranking) and
   per-section questions (good end of the ranking) add up. Subset first.
3. **Title vs research questions:** full-text criteria + title only, to see which of the two
   carries the 4–5 MAP in entry 5. Subset, about $0.23.
4. **Review abstract without *Selection criteria*.** Tests whether Jev needs criteria at all when
   it has a short description of the review.
5. **Review title only** (README plan). Lower bound for review information.
6. **Full-text criteria on CD012069 alone** (about $0.35): does fixing LGAR's omission recover the
   84 RCT includes?
7. Records without an abstract are ranked lower with the review abstract; a record-information
   experiment could address this.
