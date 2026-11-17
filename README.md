# decision-model-screener

Can non-generative decision models (Jev-like: typed questions in, probabilities out) screen
records for systematic reviews of **interventional studies**? Title/abstract first, full text maybe
later.

**Current focus:** which information the model needs to screen best. Model comparison, including
open-source models, comes later.

Status: planning. Nothing is implemented yet.

## Data

| What | Source | Use |
|---|---|---|
| Eligibility criteria | [LGAR](https://arxiv.org/abs/2505.24757) `info.json` files in [XITASO/lgar](https://github.com/XITASO/lgar/tree/main/implementation/data) | Inclusion/exclusion criteria, research questions, objectives per review |
| [CLEF TAR 2019](https://github.com/CLEF-TAR/tar/tree/master/2019-TAR/Task2) Task 2, Intervention test split | 20 Cochrane reviews, all with LGAR criteria | Main benchmark; has abstract-level and full-text-level labels |
| [SYNERGY+](https://github.com/asreview/synergy-dataset) | 114 reviews (119 datasets), about 169k records, CC0; eligibility criteria for every review; official 80/20 train/test split; mixed domains (medicine, software engineering, education, …); no Cochrane reviews, so no review overlap with CLEF TAR. Replaces the 26-review SYNERGY that LGAR used | Training (train split) and out-of-domain test (test split); labels are post-full-text (`label_included`), some datasets also have title/abstract labels |
| CLEF TAR 2019 Task 2, Intervention training split | 20 Cochrane reviews, abstract- and content-level labels; criteria from the Task 1 files. CD011571 and CD012164 are also test reviews | Validation (without those two) |
| [Guo et al.](https://data.mendeley.com/datasets/np79tmhkh5/1) | 11 reviews with LGAR criteria | Trying things out |

## Models

Called through OpenRouter (`POST https://openrouter.ai/api/alpha/decisions`) with
`OPENROUTER_API_KEY` from `.env`. Fixed model for now: `typesafe/jev-1.14`. API details are from
the [OpenRouter guide](https://openrouter.ai/docs/guides/community/jev.md) and
[TypeSafe docs](https://docs.typesafe.ai/introduction) and are untested here.

## Experiments

Zero-shot, one factor changed at a time against the LGAR input (review title + research questions
+ criteria, record title + abstract, one yes/no question).

- **Review information:** title only; + research questions; + objectives; inclusion criteria only;
  + exclusion criteria; everything; criteria restructured as PICOS; one PICOS element removed at a
  time.
- **Record information:** title only; abstract only; title + abstract; + publication type and MeSH
  terms; + journal and year; full text (later).
- **How it is asked:** one yes/no question; one graded score; one yes/no per criterion, aggregated;
  fixed PICOS questions; review information in `instructions` vs in `state`.

Metrics, as in LGAR, macro-averaged over reviews: MAP, recall@k%, WSS@95%, TNR@95%. Also cost per
run.

The zero-shot Jev experiments are notebooks in [notebooks/baseline_experiments/](notebooks/baseline_experiments/),
numbered like the entries in FINDINGS.md (`01_lgar_baseline.ipynb` is the LGAR baseline with BM25;
entry 8 has four notebooks, `08a`–`08d`). Their cached API responses are in `results/<run name>/`.

Results and conclusions so far: [FINDINGS.md](FINDINGS.md).
What this implies for training a new model later: [TRAINING_NOTES.md](TRAINING_NOTES.md).

Shared code (data, BM25, API calls with on-disk cache, metrics) is in [src/screening.py](src/screening.py);
each notebook only defines its `build_body`.

## Training data

Title/abstract only, no Cochrane mining yet. Everything is downloaded and built from scratch on the
machine that runs it; nothing in `data/` is committed.

```bash
python scripts/prepare_training_data.py   # downloads SYNERGY+ (1.5 GB), CLEF TAR files, PubMed abstracts
python scripts/build_kev_jsonl.py         # Kev files: selection criteria (always) + record title/abstract
```

| Split | Data |
|---|---|
| train | SYNERGY+ train split |
| val | CLEF TAR 2019 Intervention training reviews, without CD011571 and CD012164 (also test reviews) |
| test | CLEF TAR 2019 Intervention test reviews; SYNERGY+ test split |

Outputs are in `data/processed/training/` (`reviews.jsonl`, `records.jsonl`, `summary.json`, and
`kev/<config>/`). Details and label definitions are at the top of each script.

- **Abstracts:** SYNERGY+ ships abstracts only for open-access works. The rest are filled from PubMed
  (by PMID, then by DOI) and, for papers not in PubMed, the OpenAlex API. OpenAlex allows 1,000
  calls/day without a key: set `OPENALEX_API_KEY` in `.env` (free) or re-run the script until it no
  longer prints `INCOMPLETE`. An optional `NCBI_API_KEY` (free) makes the PubMed step about 3× faster.
  Remaining records without abstract are kept and scored on the title.
- **Labels:** title/abstract labels where the source has them (CLEF TAR; 54 of 119 SYNERGY+ datasets),
  else post-full-text labels. The label source per request is in the `.meta.jsonl` files.
- **Leakage guard (train side only; val and test stay as published):** train reviews with the same title
  as a val/test review, or sharing more than 20% of one's records, are dropped (currently
  `Kerschbaumer_2024a`, `Parodis_2023a`, `Webers_2022`). Every other train record that is the same paper
  (PMID, DOI or title) as a val/test record is removed (about 8k records).

## Training

[notebooks/training/train_decision_model.ipynb](notebooks/training/train_decision_model.ipynb) fine-tunes a
decision model with [Unsloth](https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth)
(default: Laya multilingual; Qwen3.5 as an option) on the Kev files above. It trains LoRA adapters only
(Laya's decision head stays frozen; a Qwen model must train its new head), runs on **GPU 1 only**, checks train/val for shared reviews and papers before training,
and evaluates on val every N steps (MAP, WSS@95%, TNR@95%, R@10% at both CLEF TAR label levels). Outputs go
to `runs/<name>/` (not committed). Set `PILOT = True` for a short test run first.

```bash
pip install -r requirements-train.txt
tmux new -s train
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
    notebooks/training/train_decision_model.ipynb --output-dir runs/notebooks
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your OpenRouter key
```
