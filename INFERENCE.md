# Inference throughput

How fast the trained Qwen3.5 decision models score records, and how long one pass over PubMed takes.
Measured 2026-11-20 on one NVIDIA RTX PRO 6000 Blackwell (96 GB), physical GPU 0 (with permission; training
stays on GPU 1, which was busy with other jobs at the time).

## Setup

- **Engine:** vLLM 0.34.0 serves these models natively as `ClefFlashForJointSchema` (Qwen3.5 backbone + joint
  schema head, auto-detected from `joint_head_config.json`); see the
  [vLLM structured decisions docs](https://docs.vllm.ai/en/latest/models/structured_decisions_models/). It runs
  backbone and head in the engine. No Docker service needed.
- **Separate venv:** `~/.venvs/vllm` (Python 3.13, `pip install vllm`). The training `.venv` (Python 3.14, pinned
  torch for Unsloth) is not touched.
- **Model export:** vLLM takes no LoRA here, so adapters are merged first:
  `.venv/bin/python scripts/export_merged.py --run <run> --adapter checkpoints/checkpoint-<step>` writes
  `runs/<run>/merged/<adapter>/`. A single-shard export also needs `model.safetensors.index.json` (the script
  writes it); without it vLLM loads `joint_head.safetensors` as backbone weights and fails.
- **Inputs:** token ids and question/option spans come from Unsloth's `encode_record`, exactly as in training,
  and go to `LLM.encode(..., pooling_task="token_classify")` with the spans in
  `PoolingParams.extra_kwargs["joint_schema"]` (`bench/bench_vllm.py`).
- **Checkpoints:** 0.8B `Qwen3.5-0.8B__…__train-neg1` checkpoint-4220; 4B `Qwen3.5-4B__…__train-neg1`
  checkpoint-1266. Weights do not change speed; these have saved val scores to compare against.
- **Sample:** 5,000 val requests (seed 0, `bench/make_samples.py`, written to `data/processed/bench/`), 1,060
  tokens on average. Two variants for the extrapolation: the 4,555 records with abstract (1,100 tokens) and all
  5,000 with the abstract removed (713 tokens).

## Results

Records per second, one GPU, bf16 unless noted. vLLM: `max_num_seqs=256`, `max_num_batched_tokens=8192`
(16,384 for FP8), prefix caching off.

| Config | rec/s, with abstract | rec/s, title only | Full PubMed, 1 GPU | 2 GPUs |
|---|---|---|---|---|
| Unsloth/HF path (as in training eval), 0.8B, batch 32 | 68 (mixed sample) | – | – | – |
| Unsloth/HF path, 4B, batch 32 | 15 (mixed sample) | – | – | – |
| **vLLM 0.8B** | 133 | 191 | **79 h (3.3 days)** | ~40 h |
| **vLLM 4B** | 26 | 39 | **406 h (17 days)** | ~8.5 days |
| vLLM 4B FP8 (not validated, see below) | 40 | 61 | 261 h (11 days) | ~5.5 days |

PubMed size from NCBI E-utilities on 2026-11-20: 41,467,472 records, 29,766,160 with abstract (11,701,312
title only). Hours = 29.77M / rate(with abstract) + 11.70M / rate(title only). This is **one pass for one
review**; time scales with the number of reviews. The 2-GPU column assumes two independent engines (data
parallel), not measured.

Raw numbers: `bench/sweep_08b.txt`, `bench/sweep2.txt`, `bench/final.txt`.

## Agreement with training-time scores

Same 5,000 records, compared with `runs/<run>/val_scores/step-<step>.jsonl` (`bench/agree.py`, `bench/metrics.py`;
MAP and WSS@95 on CLEF TAR abstract-level labels, macro over the 15 reviews with an include in the sample).

| Scores | Spearman vs training | mean abs. diff in p | MAP abs | WSS@95 abs |
|---|---|---|---|---|
| 4B, training-time | – | – | 66.09 | 82.41 |
| 4B, vLLM bf16 | 0.99993 | 0.0014 | 65.82 | 82.68 |
| 4B, vLLM FP8 | 0.99866 | 0.0078 | 66.21 | 82.86 |
| 0.8B, training-time | – | – | 25.68 | 42.36 |
| 0.8B, vLLM bf16 (2,000 records) | 0.988 | 0.0011 | – | – |
| 0.8B, Unsloth path re-run (2,000 records) | 0.993 | 0.0009 | – | – |
| 0.8B, vLLM FP8 | 0.864 | 0.0115 | 24.56 | 41.52 |

- vLLM bf16 reproduces the training-time scores; the 0.8B differences are the same size as re-running the
  Unsloth path itself (bf16 batch effects).
- **FP8 breaks the 0.8B model** (some p flip by almost 1.0). Do not use it.
- 4B FP8 keeps MAP/WSS on this sample, but that is 15 reviews; check on full val before using it for a reported
  result.

## What limits throughput

- **Batch settings barely matter.** `max_num_seqs` 256–2,048 and `max_num_batched_tokens` 8k–131k: 0.8B
  between 118 and 137 rec/s, bigger is slower; 4B between 25.0 and 26.6. Sorting by length: no gain.
- **The backbone is the limit.** 0.8B without the head (last-token pooling) runs 152 rec/s, ~160k tokens/s, a
  reasonable share of the GPU's bf16 peak. The head adds ~12% for 0.8B (8.7 s of 39 s, ~1.5 ms per record,
  one request at a time in a Python loop) and a few percent for 4B.
- **Prefix caching cannot help the current models.** The state is rendered with sorted keys, so `record` comes
  before `review`: only 41 tokens are shared between records of one review. The criteria are ~590 of the ~1,060
  tokens.

## Options not tried

- **Criteria before record** (retrain with review-first ordering): vLLM could compute the criteria once per
  review. Unique tokens per record would drop to ~510 (with abstract) and ~125 (title only); likely 2× or more,
  most for title-only records. Changes the model input, so it needs retraining and a new val check.
- **Batched head** inside vLLM: up to ~12% for 0.8B.
- **Second GPU:** ~2×.

## Reproduce

```bash
.venv/bin/python bench/make_samples.py
.venv/bin/python scripts/export_merged.py --run <run> --adapter checkpoints/checkpoint-<step>
~/.venvs/vllm/bin/python bench/bench_vllm.py --model runs/<run>/merged/<adapter> \
    --tok data/processed/bench/tok_abs.jsonl --max-batched 8192 --out vllm.jsonl
.venv/bin/python bench/bench_hf.py --run <run> --step <step> --batch 32 --out hf.jsonl     # baseline
```
