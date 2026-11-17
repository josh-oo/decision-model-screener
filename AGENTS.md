# AGENTS.md

Research code for the study described in [README.md](README.md). Read that first.

## Scope

- Interventional reviews only.
- Current focus is which input information helps screening. Keep the model fixed
  (`typesafe/jev-1.14`); do not start model comparisons unless asked.
- Title/abstract before full text.

## Setup

- Plain `.venv` and `requirements.txt`. No other tooling.
- No tests needed for experiment code. Keep scripts simple.
- `OPENROUTER_API_KEY` is read from `.env`. Never commit, print or log it.

## Rules that keep results valid

- Zero-shot: labels of a review never influence how that review is screened.
- Never drop records. Missing abstract: score on the title. Failed call: lowest score, and report
  the count.
- Change one input factor at a time and use the same records for every configuration.
- State which labels a metric uses (CLEF TAR abstract-level or content-level, SYNERGY
  post-full-text).
- Pin the model version; never use a `latest` alias for a reported result.
- Cache every raw response on disk and compute metrics from the cache.
- Calls cost money: try one small review first and ask before a full-benchmark run.

## Data

- Do not commit downloaded datasets, abstracts or full texts (`data/raw/`, `data/processed/`).
- Do not edit the LGAR `info.json` files; put derived versions in separate files.
- Criteria quirks: one string with `- ` bullets; `exclusion_criteria` is `"/"` when absent; the Guo
  file has no `objectives` or `link`.
- The API details in the README are from documentation, not from a live call. Check against a real
  response before relying on them.
