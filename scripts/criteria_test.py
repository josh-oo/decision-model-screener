"""Does a trained decision model read the review criteria? Score the val records with changed criteria.

  python scripts/criteria_test.py --run Qwen3.5-4B__<request set> --step 844

Loads runs/<run>/checkpoints/checkpoint-<step> (LoRA adapters + decision head) onto the pinned base model, set up
exactly as in training, and scores the val requests of the run's request set with only state.review.criteria changed:

  check          the original criteria, on the 3 smallest val reviews only: must reproduce the saved scores
                 (runs/<run>/val_scores/step-<step>.jsonl), otherwise the script stops
  other-review   each val review gets the criteria of another val review (random derangement, --seed)
  no-criteria    criteria = "" (empty string)

The record, question and labels are unchanged, and no labels are used to choose anything. Writes
runs/<run>/criteria_test/step-<step>/<variant>.jsonl (review_id, record_id, p = P(include)) and swap.json. Metrics are
computed in notebooks/training/criteria_test.ipynb from these files. Uses physical GPU 1 only.
"""
import argparse, json, os, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "training"
MODELS = {"unsloth/Qwen3.5-0.8B": "23c69c53358a07516b5827588b3fdb12ae78fd65",   # pinned Hub revisions
          "unsloth/Qwen3.5-4B": "3764fa359b9082ea5a1e4a5e3ac3aaf6e9671636"}

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True, help="directory name under runs/")
ap.add_argument("--step", type=int, required=True, help="checkpoint step")
ap.add_argument("--variants", default="other-review,no-criteria")
ap.add_argument("--eval-batch", type=int, default=32)
ap.add_argument("--seed", type=int, default=0, help="seed of the criteria swap")
args = ap.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"         # physical GPU 1; GPU 0 is never used
import unsloth  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from transformers import TrainingArguments  # noqa: E402
from unsloth import FastDecisionModel, DecisionTrainer  # noqa: E402
from unsloth.models.decision import _load_clef_checkpoint  # noqa: E402

assert torch.cuda.is_available() and torch.cuda.device_count() == 1, "expected exactly one visible GPU (physical GPU 1)"

RUN = ROOT / "runs" / args.run
CKPT = RUN / "checkpoints" / f"checkpoint-{args.step}"
OUT = RUN / "criteria_test" / f"step-{args.step}"
cfg = json.loads((RUN / "run_config.json").read_text())
assert cfg["revision"] == MODELS[cfg["model"]], "unpinned or unknown base model"
assert CKPT.is_dir(), f"no checkpoint {CKPT}"
OUT.mkdir(parents=True, exist_ok=True)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


REQ_DIR = DATA / "requests" / cfg["requests"]
val_rows, val_meta = read_jsonl(REQ_DIR / "val.jsonl"), read_jsonl(REQ_DIR / "val.meta.jsonl")
criteria = {}
for row, m in zip(val_rows, val_meta):
    criteria.setdefault(m["review_id"], row["state"]["review"]["criteria"])
    assert criteria[m["review_id"]] == row["state"]["review"]["criteria"], "criteria differ within a review"
reviews = sorted(criteria)

# Random derangement: every review gets the criteria of a different review
rng = random.Random(args.seed)
while True:
    donors = reviews[:]
    rng.shuffle(donors)
    if all(a != b for a, b in zip(reviews, donors)):
        break
swap = dict(zip(reviews, donors))
(OUT / "swap.json").write_text(json.dumps({"seed": args.seed, "criteria_of": swap}, indent=2))

sizes = pd.Series([m["review_id"] for m in val_meta]).value_counts()
check_reviews = set(sizes.nsmallest(3).index)


def variant_rows(variant):
    keep, rows = [], []
    for i, (row, m) in enumerate(zip(val_rows, val_meta)):
        if variant == "check" and m["review_id"] not in check_reviews:
            continue
        new = json.loads(json.dumps(row))
        if variant == "other-review":
            new["state"]["review"]["criteria"] = criteria[swap[m["review_id"]]]
        elif variant == "no-criteria":
            new["state"]["review"]["criteria"] = ""
        keep.append(i)
        rows.append({"state": new["state"],
                     "questions": {n: {k: v for k, v in q.items() if k != "label"} for n, q in new["questions"].items()},
                     "gold": {n: q["label"] for n, q in new["questions"].items()}})
    return keep, rows


# Model: as in training (same base, LoRA settings, seed), then the checkpoint's adapters and head
model, tokenizer = FastDecisionModel.from_pretrained(model_name=cfg["model"], revision=cfg["revision"],
                                                     max_seq_length=cfg["max_seq_length"], dtype=torch.bfloat16,
                                                     load_in_4bit=False, random_state=cfg["seed"])
model = FastDecisionModel.get_peft_model(model, r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=0.0,
                                         random_state=cfg["seed"])
_load_clef_checkpoint(model, CKPT)
model.eval()
YES = 0 if getattr(model, "is_clef", False) else 1
trainer = DecisionTrainer(model=model, tokenizer=tokenizer,
                          args=TrainingArguments(output_dir=str(OUT / "trainer"), per_device_eval_batch_size=args.eval_batch,
                                                 bf16=True, report_to="none"))


def score(variant):
    keep, rows = variant_rows(variant)
    items, report = FastDecisionModel.build_dataset(rows, tokenizer, model)
    print(variant, report, flush=True)
    assert report["skipped"] == 0, "records were skipped"
    assert [it["row"] for it in items] == list(range(len(rows))), "order changed"
    pred = trainer.predict(items).predictions
    logits = pred[0] if isinstance(pred, tuple) else pred
    p = torch.softmax(torch.tensor(np.asarray(logits)[:, :2], dtype=torch.float32), -1)[:, YES].numpy()
    df = pd.DataFrame({"review_id": [val_meta[i]["review_id"] for i in keep],
                       "record_id": [val_meta[i]["record_id"] for i in keep], "p": p})
    df.to_json(OUT / f"{variant}.jsonl", orient="records", lines=True)
    return df, report


# 1. Loading check against the scores saved during training
df, _ = score("check")
saved = pd.read_json(RUN / "val_scores" / f"step-{args.step:06d}.jsonl", lines=True)
both = df.merge(saved[["review_id", "record_id", "p"]], on=["review_id", "record_id"], suffixes=("", "_saved"))
assert len(both) == len(df)
diff = (both["p"] - both["p_saved"]).abs()
rho = both[["p", "p_saved"]].corr(method="spearman").iloc[0, 1]
print(f"check on {len(both)} records: max |Δp| {diff.max():.4f}, mean {diff.mean():.4f}, Spearman {rho:.4f}", flush=True)
assert rho > 0.99 and diff.max() < 0.05, "checkpoint does not reproduce the saved scores: stop"

# 2. The test variants
for variant in args.variants.split(","):
    _, report = score(variant)
print("done:", OUT)
