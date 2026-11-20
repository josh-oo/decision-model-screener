# Findings

Running record of the experiments: what was run, on which data, what came out, and what it means.
Newest entry at the bottom. Numbers are computed from the response caches in `results/`.

## Summary so far (2026-11-20)

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
6. **A fine-tuned small decision model is far behind zero-shot Jev.** On the validation reviews, Laya
   after 3 epochs on balanced SYNERGY+ data reaches MAP 24.2 vs Jev 55.1 (abstract level; entry 9). Jev scores
   about the same on val as on test, so val is not harder for it.
7. **Fine-tuned Qwen3.5 decision models come close to Jev, but more train reviews do not help (entry 11).** 0.8B on
   all 92 SYNERGY+ train reviews reaches MAP 54.4 abstract-level (Jev 55.1, 4B 60.4) and 49.4 content-level (Jev 59.5);
   46 reviews give about the same. Training can collapse at the start (an earlier 0.8B run reached only 21.4), so
   single runs are unreliable.
8. **The fine-tuned 4B model reads the criteria (entry 12).** With another review's criteria or none, its MAP falls
   from 60 to about 10 (random order 6.5), on every val review.
9. **Single 0.8B checkpoints are noisy; compare means over seeds and steps (entry 13).** With warmup no run collapses,
   but checkpoints still differ by about 4 MAP; averaging the scores of 3 seeds lifts MAP abs from 48 to 55.
10. **Keep the train reviews that have only final labels (entry 14).** Without them every metric is worse.
11. **Distillation from 4B helps the 0.8B model (entry 15):** +2.6 MAP at both label levels, +1 WSS@95%, and more
    stable runs. The recipe for later runs is fixed in "Recipe for later decision-model runs".

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

## 9. Validation reviews: zero-shot Jev vs a fine-tuned decision model (2026-11)

Notebook: [notebooks/training/jev_val_baseline.ipynb](notebooks/training/jev_val_baseline.ipynb); calls via
[scripts/jev_on_requests.py](scripts/jev_on_requests.py). Cache:
`results/requests__review-criteria__record-title+abstract__label-abstract_else_final__val/`. Training runs:
[notebooks/training/train_decision_model.ipynb](notebooks/training/train_decision_model.ipynb), outputs in `runs/`
(not committed).

**Differs from the common setup:** the 17 CLEF TAR 2019 Intervention **training** reviews used as validation for
training (27,870 records; without CD011571 and CD012164, which are test reviews, and CD008201, which has no
criteria). Labels: abstract-level (17 reviews) and content-level (16 reviews). Every row uses the same records
and the same request: `state.review` = the review's full-text selection criteria only (one string, outcomes
included), `state.record` = title + abstract, the usual `include` question. Jev gets this request with the label
removed; zero-shot. Jev ties broken by BM25, then record id; trained models by record id.

0 failed calls. Cost $1.45 (34.5M input tokens).

| | Labels | MAP | R@10% | WSS@95% | TNR@95% |
|---|---|---|---|---|---|
| Random order (mean of 50 shuffles) | abs | 6.5 | 9.8 | 8.0 | 13.5 |
| BM25 (query = review title) | abs | 16.4 | 36.0 | 41.3 | 47.6 |
| Laya, untrained (step 0) | abs | 7.5 | 19.0 | 13.0 | 18.5 |
| Laya, 300 steps, all train data (pilot) | abs | 11.1 | 30.6 | 25.1 | 31.3 |
| Laya, 3 epochs, balanced train data | abs | 24.2 | 50.2 | 47.0 | 54.3 |
| **Jev zero-shot** | abs | **55.1** | **81.5** | **75.2** | **83.9** |
| Random order | content | 4.7 | 10.2 | 12.0 | 17.4 |
| BM25 | content | 16.8 | 44.0 | 50.1 | 56.4 |
| Laya, untrained (step 0) | content | 5.8 | 22.7 | 19.9 | 25.2 |
| Laya, 300 steps, all train data (pilot) | content | 9.0 | 36.1 | 32.9 | 38.7 |
| Laya, 3 epochs, balanced train data | content | 25.0 | 58.1 | 55.2 | 61.6 |
| **Jev zero-shot** | content | **59.5** | **95.4** | **85.1** | **92.8** |

Laya = `convaiinnovations/laya-multilingual`, LoRA adapters only (r 16), pretrained decision head frozen, lr 2e-4
cosine, trained on the SYNERGY+ train split (criteria + record title/abstract, label abstract-level where
available, else final). Pilot: effective batch 32, 300 steps (9,600 requests, 4% of an epoch). Balanced: one
random exclude per include per review (27,004 requests), effective batch 64, 3 epochs = 1,266 steps; the reported
step is the best on val by abstract-level MAP (also the last), so slightly optimistic.

Per-review difference to Jev (balanced Laya − Jev), 95% bootstrap CI over reviews:

| Labels | ΔMAP | Better / worse | ΔWSS@95% | Better / worse |
|---|---|---|---|---|
| abs | −30.9 [−42.6, −18.4] | 2 / 15 | −28.1 [−38.2, −18.4] | 1 / 16 |
| content | −34.5 [−49.3, −18.1] | 3 / 13 | −29.9 [−39.7, −19.8] | 1 / 15 |

Balanced Laya vs BM25: +7.8 MAP abs (CI −1.7 to +16.7, 12 of 17 better), WSS@95% +5.8 (9 better / 8 worse).

- **Val is not harder than test for Jev.** Jev reaches MAP 55.1 / 59.5 here and 57.9 / 60.7 on the test reviews
  (entry 1, LGAR input; full-text criteria only were close to that on the test subset, entry 5). BM25 is lower on
  val (16.4 vs 22.3) because the review titles are weaker queries there, and val has fewer includes (1.5% vs
  2.2% pooled; 4 reviews with 1–5 includes), which lowers MAP for every ranker. Val numbers should therefore only
  be compared within val.
- **Untrained Laya cannot screen with this input** (step 0 ≈ random order); all of its ranking ability comes from
  fine-tuning.
- **The fine-tuned model is far behind zero-shot Jev** (−31 MAP, worse on 15 of 17 reviews) and only on par with
  BM25 at the end of the ranking. Training loss falls at each epoch boundary and val loss rises from step 1000 to
  1266, so more epochs on the balanced data are unlikely to close the gap.
- **Reviews hard for both:** CD008170 (Jev 17.6), CD005139 (16.7), CD012930 (35.1); large reviews with many
  includes. Candidates for label noise or criteria that cannot be judged from an abstract.
- **Not separated yet:** balancing vs training length (the pilot and the balanced run differ in both and in batch
  size).

## 10. Fine-tuning Qwen3.5-4B as a decision model (2026-11, in progress)

Notebook: [notebooks/training/train_qwen35_4b.ipynb](notebooks/training/train_qwen35_4b.ipynb). Outputs:
`runs/Qwen3.5-4B__review-criteria__record-title+abstract__label-abstract_else_final__train-neg1/` (not committed).
Same val records, requests and metrics as entry 9.

`unsloth/Qwen3.5-4B` (revision `3764fa35…`) with a **new decision head**, trained together with LoRA adapters
(r 16; backbone weights frozen). Balanced train requests as in entry 9 (27,004; 1 truncated at 8,192 tokens, 0 val
inputs truncated), effective batch 64 (16 × 4 accumulation), lr 2e-4 cosine (head 1e-4), 10 epochs of 422 steps,
validation at step 0 and after every epoch.

### Step 0 (untrained head)

| | Labels | MAP | R@10% | WSS@95% | TNR@95% |
|---|---|---|---|---|---|
| Random order (entry 9) | abs | 6.5 | 9.8 | 8.0 | 13.5 |
| Qwen3.5-4B, step 0 | abs | 7.1 | 18.9 | 19.4 | 25.0 |
| Random order (entry 9) | content | 4.7 | 10.2 | 12.0 | 17.4 |
| Qwen3.5-4B, step 0 | content | 5.1 | 20.9 | 22.3 | 27.7 |

- **Random level, as expected for a new head.** All scores lie between 0.486 and 0.501 (mean 0.494, SD 0.002);
  mean score of includes 0.4943 vs excludes 0.4937. The small lead over random order on WSS/TNR is what a random
  projection of the Qwen features happens to give; it carries no information, and it cannot show whether the
  yes/no columns are the right way round (check after epoch 1: MAP clearly below random means swapped).
- **Unlike Laya (entry 9), step 0 is no baseline here**: Laya's head was pretrained, Qwen's is not.
- **One validation pass takes about 32 minutes** (27,870 records, eval batch 32, about 46 GB GPU memory), about
  8.5× Laya's 225 s. This run used the PyTorch fallback for Qwen3.5's linear-attention layers
  (`flash-linear-attention` and `causal-conv1d` not installed); 10 more evaluations would take about 5.3 hours
  without training. The run is restarted with both packages installed; compare its evaluation time with these
  32 minutes.

## 11. Qwen3.5-0.8B: more training reviews, and an unstable start (2026-11-19)

Notebook: [notebooks/training/review_count_curve_qwen35_0.8b.ipynb](notebooks/training/review_count_curve_qwen35_0.8b.ipynb);
runs via [scripts/train_decision_model.py](scripts/train_decision_model.py), subsets via
[scripts/subset_train.py](scripts/subset_train.py). Outputs: `runs/Qwen3.5-0.8B__…__k<k>-s<seed>__steps1266__sched4220/`
(not committed). Same val records, requests and metrics as entries 9–10.

**Question:** does the 0.8B decision model get better with more training reviews? The train set has only 92
distinct criteria texts. **Only the number of train reviews changes**: 46 (a random subset, stratified by domain and
review size; all requests of a chosen review kept) vs all 92. Everything else as in the earlier 0.8B notebook run:
`unsloth/Qwen3.5-0.8B` (revision `23c69c53…`), new decision head + LoRA r 16, lr 2e-4 (head 1e-4), batch 64 × 1,
seed 3407, balanced train requests, cosine schedule over 4,220 steps without warmup, stopped at step 1,266 (= 3
epochs of the 92 reviews; 5.7 epochs of the 46). Val after steps 50, 100, 200, 422, 844, 1266; reported is the best
step by abstract-level MAP (slightly optimistic, same rule for every run). About 2 h per run on one GPU.

| | Labels | Best step (epoch) | MAP | R@10% | WSS@95% | TNR@95% |
|---|---|---|---|---|---|---|
| 0.8B, 46 reviews | abs | 422 (1.9) | 52.9 | 78.3 | 72.3 | 80.6 |
| 0.8B, 92 reviews | abs | 422 (1.0) | 54.4 | 77.0 | 75.3 | 84.0 |
| 0.8B, 92 reviews, earlier notebook run (failed start, see below) | abs | 1266 (3.0) | 21.4 | 49.2 | 55.7 | 63.1 |
| Qwen3.5-4B (entry 10) | abs | 844 (2.0) | 60.4 | 84.9 | 76.8 | 85.8 |
| Jev zero-shot (entry 9) | abs | – | 55.1 | 81.5 | 75.2 | 83.9 |
| 0.8B, 46 reviews | content | 422 (1.9) | 54.2 | 85.1 | 78.5 | 86.0 |
| 0.8B, 92 reviews | content | 422 (1.0) | 49.4 | 81.3 | 76.7 | 84.1 |
| 0.8B, 92 reviews, earlier notebook run | content | 1266 (3.0) | 15.9 | 53.0 | 61.0 | 67.6 |
| Qwen3.5-4B (entry 10) | content | 844 (2.0) | 59.5 | 91.1 | 82.1 | 89.8 |
| Jev zero-shot (entry 9) | content | – | 59.5 | 95.4 | 85.1 | 92.8 |

Abstract-level MAP at each evaluation (the score falls after the best step as the train set is memorised; train
loss reaches about 0.04 for 46 reviews and 0.12 for 92 by step 1,266):

| Step | 50 | 100 | 200 | 422 | 844 | 1266 |
|---|---|---|---|---|---|---|
| 46 reviews | 40.4 | 37.6 | 50.8 | **52.9** | 39.2 | 32.8 |
| 92 reviews | 32.2 | 41.9 | 42.0 | **54.4** | 49.7 | 46.0 |

Per-review differences at the best step, 95% bootstrap CI over val reviews, reviews better / worse:

| Comparison | Labels | ΔMAP | ΔWSS@95% |
|---|---|---|---|
| 92 − 46 reviews | abs | +1.5 [−1.9, +5.3], 10 / 6 | +3.0 [−1.3, +8.4], 9 / 7 |
| 92 − 46 reviews | content | −4.8 [−11.0, +1.0], 5 / 10 | −1.8 [−6.2, +1.7], 7 / 7 |
| 92 reviews − Jev | abs | −0.7 [−10.0, +8.9], 7 / 9 | +0.1 [−4.2, +5.4], 8 / 8 |
| 92 reviews − Jev | content | −10.1 [−21.7, +1.2], 6 / 9 | −8.4 [−16.9, −0.9], 4 / 11 |

- **Doubling the train reviews from 46 to 92 gives no clear gain.** +1.5 MAP at abstract level (below the 3 MAP set
  in advance as worth acting on; CI includes 0), −4.8 at content level. More reviews of this kind are not what limits
  the 0.8B model. With 92 reviews the score falls more slowly after the peak (46.0 vs 32.8 at step 1,266), so more
  data makes the result less dependent on when training stops, but the peak is no higher.
- **Memorisation is not the main limit.** Both runs peak after 1–2 epochs and get worse afterwards; picking the best
  step handles that. The peak itself is the limit: it ties zero-shot Jev at abstract level and is below it at content
  level. A possible reason (not tested): with only 92 different criteria texts the model learns what includes
  typically look like rather than how to apply the given criteria. Entry 12 rules this out for the 4B model; for
  0.8B it is untested.
- **The 0.8B model is about 6 MAP behind 4B, not 40.** The earlier result "0.8B peaks at 21.4" came from a failed
  start.
- **Caveats:** one subset and one training seed per number of reviews; the start can vary a lot between runs (below),
  so a 1.5 MAP difference is within what a second seed could change. The best step lies somewhere between 200 and
  844, but only 422 is evaluated there. The two label levels disagree in direction.

### Why the earlier 0.8B run reached only 21 MAP: it collapsed at the start

The earlier notebook run ([notebooks/training/train_qwen35_0.8b.ipynb](notebooks/training/train_qwen35_0.8b.ipynb),
`runs/Qwen3.5-0.8B__…__train-neg1/`) and the fresh 92-review run have the same settings, data, package versions and
seed; their losses are identical up to step 20. Then they part:

| Step | 10 | 20 | 30 | 40 | 50 | 100 | 200 |
|---|---|---|---|---|---|---|---|
| Earlier run: train loss (grad norm) | 0.695 (0.03) | 0.692 (0.13) | 0.676 (0.51) | 0.676 (0.41) | **0.697 (0.09)** | 0.694 (0.04) | 0.692 (0.08) |
| Fresh run: train loss (grad norm) | 0.695 (0.02) | 0.692 (0.22) | 0.665 (0.25) | 0.634 (1.52) | **0.540 (7.29)** | 0.444 (3.24) | 0.373 (1.56) |

- Both start to learn at steps 30–40; the earlier run then falls back to loss 0.69, which on balanced data is the loss
  of answering 50/50 for every record, with very small gradients. It stays there for about 300 steps (0.615 at step
  410) and reaches only MAP 21.4 by step 1,266, although its train loss then falls (0.36 at step 1,210).
- **Cause, as far as the logs show:** training is on a knife-edge at steps 30–50, and the small numerical
  differences between two GPU runs in bf16 (not bit-for-bit reproducible) decide whether the model keeps learning or
  collapses. Something left over in the long-lived notebook kernel cannot be fully ruled out. A likely trigger is the
  full learning rate (2e-4) from step 1 with no warmup on a new head; not tested.
- **Consequence:** one run of this setup can land anywhere between about 21 and 54 MAP. Reported 0.8B (and
  probably 4B) numbers need either a warmup that prevents the collapse or several training seeds. Check the train
  loss over the first 100 steps of every run: if it is back at 0.69, the run has collapsed.

## 12. Does the trained decision model read the criteria? Qwen3.5-4B with changed criteria (2026-11-19)

Notebook: [notebooks/training/criteria_test.ipynb](notebooks/training/criteria_test.ipynb); scores via
[scripts/criteria_test.py](scripts/criteria_test.py), in `runs/Qwen3.5-4B__…__train-neg1/criteria_test/step-844/`
(not committed). Same val records, requests and metrics as entries 9–11.

The Qwen3.5-4B decision model at its best step (844; entry 10) scores the val records again with **only
`state.review.criteria` changed**: *other review* = the criteria of another val review (random derangement, seed 0,
no review keeps its own; `swap.json`), *no criteria* = empty string. Record, question and labels unchanged; metrics
against each record's true labels for its own review. The reloaded checkpoint reproduces the scores saved during
training (258 records of the 3 smallest reviews: Spearman 0.9999, max |Δp| 0.006). About 30 min per variant on one GPU.

| | Labels | MAP | R@10% | WSS@95% | TNR@95% |
|---|---|---|---|---|---|
| Own criteria | abs | 60.4 | 84.9 | 76.8 | 85.8 |
| Other review's criteria | abs | 9.8 | 16.0 | 26.3 | 32.6 |
| No criteria | abs | 11.0 | 22.5 | 26.1 | 32.3 |
| Random order (entry 9) | abs | 6.5 | 9.8 | 8.0 | 13.5 |
| Own criteria | content | 59.5 | 91.1 | 82.1 | 89.8 |
| Other review's criteria | content | 6.3 | 23.0 | 35.8 | 41.6 |
| No criteria | content | 9.9 | 28.1 | 29.0 | 34.8 |
| Random order (entry 9) | content | 4.7 | 10.2 | 12.0 | 17.4 |

Per review, variant − own criteria (95% bootstrap CI, reviews better / worse): other review ΔMAP −50.6 [−60.8, −40.4]
abs (0 / 17), −53.2 [−65.4, −40.2] content (0 / 16); no criteria −49.4 [−58.9, −38.7] abs (0 / 17), −49.6
[−62.2, −36.2] content (1 / 15). Rank correlation (Spearman) of the scores with own vs changed criteria within a
review: mean 0.24 for both variants (range −0.30 to 0.69).

- **The 4B model reads the criteria.** Without fitting criteria, MAP falls from 60 to about 10, close to random order,
  on every review, and the order of the records within a review changes almost completely. About 50 of its 60 MAP
  depend on the criteria; what remains (WSS@95% about 26 vs 8 for random order) is a small generic signal that pushes
  some obvious non-studies to the end.
- **The scores behave sensibly.** Mean P(include) is 0.20 with own criteria, 0.01 with another review's criteria
  (almost everything excluded) and 0.68 with no criteria (nothing to exclude on).
- **The hypothesis of entry 11 does not hold for 4B:** the trained model does not just learn what includes look
  like. Fine-tuning on the 92 SYNERGY+ reviews can teach criteria-reading.
- **Not shown:** whether it applies the finer criteria within a topic (study design, comparator). The swapped criteria
  are from unrelated topics, so the test shows at least that the criteria tell the model what the review is about.
  A variant that changes one criterion and keeps the topic (e.g. remove the study-design sentence) would test that.
  Also not shown: whether the 0.8B model reads them (its runs in entry 11 saved no weights).

## 13. Qwen3.5-0.8B with warmup: three seeds, and where the noise comes from (2026-11-19)

Runs via [scripts/train_decision_model.py](scripts/train_decision_model.py) and `runs/seeds-warmup.sh`; outputs
`runs/Qwen3.5-0.8B__…__train-neg1__steps844__sched4220__warmup60__seed<1-3>/` (not committed, checkpoints saved).
Same val records, requests and metrics as entries 9–12.

Setup as the 92-review run of entry 11, with **a linear warmup over the first 60 steps** (against the collapse) and
**three training seeds**, stopped at step 844 (2 epochs; cosine schedule still over 4,220 steps). Val after steps 100,
200, 300, 422, 633, 844. From here on, runs are compared by the **mean over all eval steps and seeds**, not by the best
step, because the best step is chosen on val and is noisy.

| Run set (mean over 6 steps × 3 seeds) | MAP abs | MAP content | WSS@95 abs | WSS@95 content |
|---|---|---|---|---|
| 0.8B, 92 reviews, warmup 60 | 46.6 | 42.7 | 71.2 | 75.6 |

- **No collapse in any seed.** Train loss leaves 0.69 within the first 100 steps in all three.
- **Single checkpoints are very noisy.** The best step differs per seed (MAP abs 52.7 at 844, 53.6 at 422, 51.5 at
  300); the same seed moves by about 4 MAP between neighbouring eval steps, and seeds differ by about as much. Most of
  the noise comes from val reviews with 1–5 includes (per-review SD about 14 MAP); seed and step contribute about
  equally. A difference of under about 3 MAP between two single runs means nothing.
- **Averaging the scores of several checkpoints helps a lot** (computed from the saved val scores, no new training):

| Scores (P(include)) averaged over | MAP abs | MAP content | WSS@95 abs | WSS@95 content |
|---|---|---|---|---|
| nothing: single checkpoint (mean of 3 seeds × steps 422/633/844) | 48.2 | 44.2 | 70.9 | 75.8 |
| steps 422/633/844 of one seed (mean over seeds) | 51.0 | 47.0 | 71.9 | 76.3 |
| 3 seeds at step 844 | 55.1 | 52.8 | 72.2 | 77.1 |
| 3 seeds × 3 steps | 54.4 | 52.8 | 72.1 | 76.0 |
| Qwen3.5-4B, step 844 (entry 10) | 60.4 | 59.5 | 76.8 | 82.1 |

  Averaging closes about half of the MAP gap to 4B, so much of the 0.8B's weakness is instability. WSS@95% hardly
  moves: the hardest includes at the end of the ranking are not fixed by averaging.

## 14. Do labels from full-text screening add noise? Train only on reviews with title/abstract labels (2026-11-20)

Runs via `runs/labels-abstract.sh`; request set `…__train-neg1__labels-abstract` built with
`scripts/subset_train.py --label-source abstract`; outputs `runs/Qwen3.5-0.8B__…__labels-abstract__steps844__…__seed<1-3>/`.

50 of the 92 SYNERGY+ train reviews have only final (post-full-text) labels. For the title/abstract task these are
partly wrong: about 5% of their final excludes were includes at the abstract stage (median per review 6.4%, max
40.5%); about 1–2% of all train requests. **Only change: train on the 42 reviews with title/abstract labels**
(17,198 requests), same steps (844) and recipe as entry 13, 3 seeds.

| Run set (mean over 6 steps × 3 seeds) | MAP abs | MAP content | WSS@95 abs | WSS@95 content |
|---|---|---|---|---|
| 92 reviews, all labels (entry 13) | **46.6** | **42.7** | **71.2** | **75.6** |
| 42 reviews, title/abstract labels only | 45.7 | 40.0 | 69.2 | 72.2 |

- **Leaving out the final-label reviews hurts on every metric**, most for content-level WSS@95% (−3.4). The seeds
  vary much more (MAP abs per seed 40.3–50.6 vs 43.9–48.9), and the runs get worse after about 2 epochs (WSS@95 abs
  65–66 at step 844).
- **Keep all 92 reviews.** The variety of the 50 extra reviews is worth more than the noise of their labels.
- **Caveats:** same steps means more epochs on the smaller set (3.1 vs 2.0 at step 844), and the domain mix differs
  (42: Health 15, Social 19, Physical 6, Life 2; 92: Health 42, Social 23, Physical 16, Life 11).

## 15. Distillation: Qwen3.5-4B teaches Qwen3.5-0.8B (2026-11-20)

Runs via `runs/distill.sh`; teacher scores via [scripts/score_train.py](scripts/score_train.py) in
`runs/Qwen3.5-4B__…__train-neg1/train_scores/` ; students `runs/Qwen3.5-0.8B__…__warmup60__kd-4b-w0.5__seed<1-3>/`.

The 4B model of entry 10 (checkpoint 844, trained on all 92 train reviews) scores every train request once
(about 25 min). The 0.8B student then trains on the soft target **0.5 × label + 0.5 × 4B P(include)**
(`--soft-targets … --soft-weight 0.5`; with cross-entropy this is the same as 0.5 × label loss + 0.5 × teacher loss).
**Only the target changes**; everything else as in entry 13 (warmup 60, 844 steps, 3 seeds). Only train reviews are
scored, so val and test labels play no part.

The teacher scored its own training records, so its scores could have been close to the labels. They were not:
mean 4B P(include) 0.86 for includes, 0.16 for excludes, so the targets are genuinely soft (lowest possible train loss
0.21; the students reach about 0.30).

| Run set (mean over 6 steps × 3 seeds) | MAP abs | MAP content | WSS@95 abs | WSS@95 content |
|---|---|---|---|---|
| **0.8B distilled from 4B** | **49.2** | **45.2** | **72.0** | **76.6** |
| 0.8B, labels only (entry 13) | 46.6 | 42.7 | 71.2 | 75.6 |
| 0.8B, 42 reviews (entry 14) | 45.7 | 40.0 | 69.2 | 72.2 |
| Qwen3.5-4B teacher (step 844, one run) | 60.4 | 59.5 | 76.8 | 82.1 |

Seed means of MAP abs: distilled 51.4 / 49.3 / 47.1 vs labels only 48.9 / 43.9 / 47.0.

- **Distillation helps on every metric:** +2.6 MAP at both label levels, about +1 WSS@95%.
- **It is more stable.** The weakest distilled seed is above the mean of the labels-only runs; from step 300 on no
  distilled checkpoint falls below 46 MAP abs (labels only: down to 39).
- **Still far from 4B:** about 11–14 MAP and 5 WSS@95%.
- **Not yet separated:** how much comes from the 4B's knowledge and how much from softer targets alone (label
  smoothing also stabilises training). Control: uniform targets of the same average softness, no teacher.

### Why the scores move late in training

Distilled students, mean of the 3 seeds per eval step:

| Step (epoch) | MAP abs | MAP content | WSS@95 abs | WSS@95 content |
|---|---|---|---|---|
| 100 (0.24) | 45.0 | 41.6 | 71.3 | 76.8 |
| 200 (0.47) | 47.6 | 43.8 | 73.8 | 77.3 |
| 300 (0.71) | 51.0 | 46.5 | 73.0 | 76.8 |
| 422 (1.00) | 51.8 | 46.7 | 71.9 | 77.8 |
| 633 (1.50) | 48.4 | 45.5 | 71.0 | 74.7 |
| 844 (2.00) | 51.6 | 47.3 | 71.0 | 76.3 |

MAP rises until about 1 epoch and then stays level with a dip at step 633 (all three seeds); WSS@95% abs peaks at
step 200 and loses about 3 points. Three causes:

1. **From the second epoch the model fits the train records themselves.** Train loss drops right at the epoch
   boundary (about 0.35 before step 422, 0.30 after). That helps the top of the ranking little and costs the hardest
   includes, which WSS@95% depends on. Entry 14 shows the same more strongly (3 epochs → WSS@95 abs 65).
2. **The learning rate never decays.** The cosine schedule runs over 4,220 steps but training stops at 844, where the
   learning rate is still 1.83e-4 (92% of the peak). Every checkpoint is a snapshot of weights that are still moving a
   lot; hence the up-and-down from step to step instead of a settled end.
3. **Val noise:** 17 reviews, several with 1–5 includes; about 4 MAP per step (entry 13).

## Recipe for later decision-model runs (decided 2026-11-20)

These are fixed for all later runs unless a run tests one of them:

1. **Warmup** over the first 60 steps (entry 13; prevents the collapse of entry 11).
2. **All 92 train reviews**, including those with only final labels (entry 14).
3. **Distillation from Qwen3.5-4B**, target 0.5 × label + 0.5 × teacher score (entry 15).
4. **Learning rate decays to zero at the last step** (`--schedule-steps` = `--max-steps`), and **training stops after
   about 1–1.5 epochs** (422–633 steps of the 92 reviews). The last checkpoint is then the one to report, with no step
   chosen on val (entry 15). Not yet run: the first run with this schedule also checks that it does not cost MAP.
5. **Compare run sets by the mean over seeds (3) and eval steps**, never by one best checkpoint (entry 13). Save
   checkpoints.
6. Keep the val records, request set and metrics fixed so runs stay comparable.

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
8. ~~Stable start for the decision models~~: done, warmup 60 steps prevents the collapse; seeds vary by about 4 MAP
   (entry 13).
9. **Does the trained model use the criteria?** Answered for 4B: yes (entry 12). Still open: the 0.8B model (needs
   a retrained run with checkpoints saved), and whether finer criteria within a topic are applied (change one
   criterion, keep the topic).
10. Optional for entry 11: a second 46-review subset (k46-s2) to see how much the result depends on which reviews
    were picked; the 12- and 23-review points only if more reviews ever show a gain.
11. **Label-smoothing control for entry 15:** uniform soft targets (includes 0.93, excludes 0.08, the average of the
    distillation targets), no teacher, 3 seeds. Shows how much of the distillation gain is the 4B's knowledge.
12. **First run with the new recipe** (decaying learning rate, stop at about 1–1.5 epochs, distillation; see "Recipe
    for later decision-model runs").
13. **Better teachers:** teacher weight 1.0; or two 4B teachers trained on disjoint halves of the train reviews, each
    scoring the half it never saw (more informative targets; about 4.5 h extra).
14. **Weight averaging:** average the LoRA weights of several checkpoints of one run into one model, to get part of
    the score-averaging gain of entry 13 without running several models.
15. **Full text for the final-label reviews:** give the full text in the input where the label is the final decision,
    so label and input match (entry 14). Needs open-access full texts; first check their coverage.
