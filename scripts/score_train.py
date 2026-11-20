"""Score the train requests of a request set with a trained checkpoint (teacher scores for distillation).

  python scripts/score_train.py --run Qwen3.5-4B__<fold A request set>__... --step 422 --requests <fold B request set>

Loads runs/<run>/checkpoints/checkpoint-<step> onto the pinned base model, set up exactly as in training, and scores
every request in data/processed/training/requests/<requests>/train.jsonl. The scored reviews must not be among the
reviews the run was trained on (out-of-fold), unless --allow-seen is given: then the teacher may score its own
training records, whose scores are closer to the labels than scores on unseen records.
Writes runs/<run>/train_scores/<requests>__step-<step>.jsonl (review_id, record_id, p = P(include)), in the order of
train.jsonl. Only train data is scored; val and test are not touched. Uses physical GPU 1 only.
"""
import argparse, json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "training"
MODELS = {"unsloth/Qwen3.5-0.8B": "23c69c53358a07516b5827588b3fdb12ae78fd65",   # pinned Hub revisions
          "unsloth/Qwen3.5-4B": "3764fa359b9082ea5a1e4a5e3ac3aaf6e9671636"}

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True, help="directory name under runs/")
ap.add_argument("--step", type=int, required=True, help="checkpoint step")
ap.add_argument("--requests", required=True, help="request set whose train.jsonl is scored")
ap.add_argument("--eval-batch", type=int, default=32)
ap.add_argument("--allow-seen", action="store_true", help="allow scoring reviews the run was trained on")
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


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


RUN = ROOT / "runs" / args.run
CKPT = RUN / "checkpoints" / f"checkpoint-{args.step}"
OUT = RUN / "train_scores" / f"{args.requests}__step-{args.step}.jsonl"
cfg = json.loads((RUN / "run_config.json").read_text())
assert cfg["revision"] == MODELS[cfg["model"]], "unpinned or unknown base model"
assert CKPT.is_dir(), f"no checkpoint {CKPT}"
assert not OUT.exists(), f"{OUT} exists: move it away first"

rows = read_jsonl(DATA / "requests" / args.requests / "train.jsonl")
meta = read_jsonl(DATA / "requests" / args.requests / "train.meta.jsonl")
seen = {m["review_id"] for m in read_jsonl(DATA / "requests" / cfg["requests"] / "train.meta.jsonl")}
scored = {m["review_id"] for m in meta}
assert args.allow_seen or not seen & scored, f"out-of-fold violated: the run was trained on {sorted(seen & scored)}"
print(f"scoring {len(rows):,} requests of {len(scored)} reviews; teacher was trained on {len(seen & scored)} of them",
      flush=True)

model, tokenizer = FastDecisionModel.from_pretrained(model_name=cfg["model"], revision=cfg["revision"],
                                                     max_seq_length=cfg["max_seq_length"], dtype=torch.bfloat16,
                                                     load_in_4bit=False, random_state=cfg["seed"])
model = FastDecisionModel.get_peft_model(model, r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=0.0,
                                         random_state=cfg["seed"])
_load_clef_checkpoint(model, CKPT)
model.eval()
YES = 0 if getattr(model, "is_clef", False) else 1
trainer = DecisionTrainer(model=model, tokenizer=tokenizer,
                          args=TrainingArguments(output_dir=str(RUN / "train_scores" / "trainer"),
                                                 per_device_eval_batch_size=args.eval_batch, bf16=True, report_to="none"))

items, report = FastDecisionModel.build_dataset(
    [{"state": r["state"], "questions": {n: {k: v for k, v in q.items() if k != "label"} for n, q in r["questions"].items()},
      "gold": {n: q["label"] for n, q in r["questions"].items()}} for r in rows], tokenizer, model)
print(report, flush=True)
assert report["skipped"] == 0, "records were skipped"
assert [it["row"] for it in items] == list(range(len(rows))), "order changed"
pred = trainer.predict(items).predictions
logits = pred[0] if isinstance(pred, tuple) else pred
p = torch.softmax(torch.tensor(np.asarray(logits)[:, :2], dtype=torch.float32), -1)[:, YES].numpy()
OUT.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"review_id": [m["review_id"] for m in meta], "record_id": [m["record_id"] for m in meta],
              "p": p}).to_json(OUT, orient="records", lines=True)
y = np.array([r["questions"]["include"]["label"] for r in rows], dtype=float)
print(f"done: {OUT}  mean p | include {p[y == 1].mean():.3f}, exclude {p[y == 0].mean():.3f}", flush=True)
