"""Train a decision model (LoRA + decision head, Unsloth) on a request set for a fixed number of steps; script version
of notebooks/training/train_qwen35_0.8b.ipynb for runs that are repeated many times (learning curves).

  python scripts/train_decision_model.py --requests <request set name>
  python scripts/train_decision_model.py --requests <name> --model unsloth/Qwen3.5-4B --batch 16

Differences from the notebook: training stops at a step cap (--max-steps); the cosine schedule runs over
--schedule-steps (default: the cap; 4220 = the 10-epoch schedule of the notebook's 0.8B run, which stopped at 1266),
validation runs at fixed steps (--eval-steps) instead of every epoch, and no checkpoints are saved (the scores at
each evaluation are; --save-checkpoints also keeps adapters + head at each evaluation, in checkpoints/checkpoint-N).
--warmup-steps adds a linear learning-rate warmup (default 0, as in the notebook). Model, LoRA, learning rates,
effective batch 64, seed, metric code and checks are the same.

Writes runs/<model>__<requests>__steps<max>[__sched<schedule>][__warmup<n>][__seed<seed>]/ (seed only if not 3407): run_config.json, pip_freeze.txt, val_metrics.jsonl,
val_scores/step-NNNNNN.jsonl (P(include) per val record, both CLEF TAR label levels).
Uses physical GPU 1 only.
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "training"
MODELS = {"unsloth/Qwen3.5-0.8B": "23c69c53358a07516b5827588b3fdb12ae78fd65",   # pinned Hub revisions
          "unsloth/Qwen3.5-4B": "3764fa359b9082ea5a1e4a5e3ac3aaf6e9671636"}

ap = argparse.ArgumentParser()
ap.add_argument("--requests", required=True, help="directory name under data/processed/training/requests/")
ap.add_argument("--model", default="unsloth/Qwen3.5-0.8B", choices=sorted(MODELS))
ap.add_argument("--max-steps", type=int, default=1266, help="training steps (1266 = 3 epochs of the full train set)")
ap.add_argument("--schedule-steps", type=int, help="length of the cosine schedule (default: --max-steps)")
ap.add_argument("--eval-steps", default="50,100,200,422,844,1266", help="steps after which val is scored")
ap.add_argument("--batch", type=int, default=16, help="per device; gradient accumulation keeps the effective batch at 64")
ap.add_argument("--eval-batch", type=int, default=128)
ap.add_argument("--seed", type=int, default=3407)
ap.add_argument("--warmup-steps", type=int, default=0, help="linear learning-rate warmup")
ap.add_argument("--save-checkpoints", action="store_true", help="keep adapters + head at each evaluation step")
ap.add_argument("--soft-targets", help="distillation: comma-separated teacher score files (review_id, record_id, p) "
                                       "covering every train request, e.g. runs/<teacher>/train_scores/<...>.jsonl")
ap.add_argument("--soft-weight", type=float, default=0.5, help="target = (1 - w) * label + w * teacher p")
ap.add_argument("--soft-name", help="short name of the teacher, used in the run directory name")
args = ap.parse_args()
EVAL_STEPS = sorted({int(s) for s in args.eval_steps.split(",")})
SCHEDULE = args.schedule_steps or args.max_steps
assert EVAL_STEPS[-1] <= args.max_steps <= SCHEDULE and 64 % args.batch == 0
assert (args.soft_targets is None) == (args.soft_name is None), "--soft-targets and --soft-name go together"
assert 0 < args.soft_weight <= 1

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # number GPUs as nvidia-smi does
os.environ["CUDA_VISIBLE_DEVICES"] = "1"         # physical GPU 1; GPU 0 is never used
import unsloth  # noqa: E402,F401  (patches transformers, so it is imported first)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from transformers import TrainingArguments, TrainerCallback  # noqa: E402
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen35  # noqa: E402
from unsloth import FastDecisionModel, DecisionTrainer  # noqa: E402

sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from build_requests import paper_keys, norm_title  # noqa: E402
from screening import metrics  # noqa: E402

assert torch.cuda.is_available() and torch.cuda.device_count() == 1, "expected exactly one visible GPU (physical GPU 1)"

# Qwen3.5's linear-attention layers must use the fast kernels, not the slow PyTorch reference
for name in ("causal_conv1d_fn", "causal_conv1d_update", "torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
    fn = getattr(qwen35, name)
    impl = dict(zip(fn.__code__.co_freevars, (c.cell_contents for c in fn.__closure__ or ()))).get("implementation")
    assert impl is not None and not impl.__module__.startswith("transformers."), f"{name}: slow or unknown kernel"

REQ_DIR = DATA / "requests" / args.requests
RUN = ROOT / "runs" / (f"{args.model.split('/')[-1]}__{args.requests}__steps{args.max_steps}"
                       + (f"__sched{SCHEDULE}" if SCHEDULE != args.max_steps else "")
                       + (f"__warmup{args.warmup_steps}" if args.warmup_steps else "")
                       + (f"__kd-{args.soft_name}-w{args.soft_weight:g}" if args.soft_targets else "")
                       + (f"__seed{args.seed}" if args.seed != 3407 else ""))
assert not (RUN / "val_metrics.jsonl").exists(), f"{RUN} already has results: move them away first"
RUN.mkdir(parents=True, exist_ok=True)
print("run:", RUN)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


# Data and leakage check (same rule as the build and the notebooks)
build_config = json.loads((REQ_DIR / "config.json").read_text())
train_rows, train_meta = read_jsonl(REQ_DIR / "train.jsonl"), read_jsonl(REQ_DIR / "train.meta.jsonl")
val_rows, val_meta = read_jsonl(REQ_DIR / "val.jsonl"), read_jsonl(REQ_DIR / "val.meta.jsonl")
reviews = {r["review_id"]: r for r in read_jsonl(DATA / "reviews.jsonl")}
data_dirs = [DATA] + [ROOT / d for d in build_config.get("extra_train_data", [])]   # e.g. the Cochrane train set
for d in data_dirs[1:]:
    extra = {r["review_id"]: r for r in read_jsonl(d / "reviews.jsonl")}
    assert not set(extra) & set(reviews), f"review ids in {d} clash with earlier data"
    reviews.update(extra)
needed = {(m["review_id"], m["record_id"]) for m in train_meta + val_meta}
records = {}
for d in data_dirs:
    for r in map(json.loads, open(d / "records.jsonl")):
        if (r["review_id"], r["record_id"]) in needed:
            records[r["review_id"], r["record_id"]] = r
assert len(records) == len(needed), f"{len(needed) - len(records)} train/val records not found in {data_dirs}"
train_reviews, val_reviews = {m["review_id"] for m in train_meta}, {m["review_id"] for m in val_meta}
assert not train_reviews & val_reviews, "review in train and val"
assert not ({norm_title(reviews[r]["title"]) for r in train_reviews}
            & {norm_title(reviews[r]["title"]) for r in val_reviews}), "same review title in train and val"
train_keys = set().union(*(paper_keys(records[m["review_id"], m["record_id"]]) for m in train_meta))
assert not any(paper_keys(records[m["review_id"], m["record_id"]]) & train_keys for m in val_meta), "val paper in train"
val_labels = {lvl: [records[m["review_id"], m["record_id"]][f"label_{lvl}"] for m in val_meta]
              for lvl in ("abstract", "final")}
del records
print(f"no overlap: {len(train_reviews)} train reviews / {len(train_rows):,} requests, "
      f"{len(val_reviews)} val reviews / {len(val_rows):,} requests")


def to_unsloth(row, soft=None):   # request format (label on the question) -> Unsloth format (answers in "gold")
    gold = {n: q["label"] for n, q in row["questions"].items()}
    if soft is not None:   # distillation: soft target P(true) = (1 - w) * label + w * teacher p
        assert set(gold) == {"include"}
        gold = {"include": {"label": gold["include"],
                            "probabilities": {"true": (1 - args.soft_weight) * float(gold["include"])
                                              + args.soft_weight * soft}}}
    return {"state": row["state"],
            "questions": {n: {k: v for k, v in q.items() if k != "label"} for n, q in row["questions"].items()},
            "gold": gold}


# Teacher scores for distillation: one per train request, from teachers that never saw that request's review
soft = [None] * len(train_rows)
soft_info = None
if args.soft_targets:
    teacher = {}
    for f in args.soft_targets.split(","):
        for t in read_jsonl(f):
            key = (t["review_id"], t["record_id"])
            assert key not in teacher, f"teacher score for {key} given twice"
            teacher[key] = float(t["p"])
    soft = [teacher[m["review_id"], m["record_id"]] for m in train_meta]   # KeyError if a request has no score
    assert len(teacher) == len(train_rows), "teacher scores for records outside the train set"
    y = np.array([r["questions"]["include"]["label"] for r in train_rows], dtype=float)
    s_ = np.array(soft)
    soft_info = {"files": args.soft_targets.split(","), "weight": args.soft_weight,
                 "teacher_mean_p_include": round(float(s_[y == 1].mean()), 4),
                 "teacher_mean_p_exclude": round(float(s_[y == 0].mean()), 4)}
    print("distillation:", soft_info)


# Model: LoRA adapters on the frozen backbone + a new decision head
model, tokenizer = FastDecisionModel.from_pretrained(model_name=args.model, revision=MODELS[args.model],
                                                     max_seq_length=8192, dtype=torch.bfloat16, load_in_4bit=False,
                                                     random_state=args.seed)
model = FastDecisionModel.get_peft_model(model, r=16, lora_alpha=16, lora_dropout=0.0, random_state=args.seed)
for n, p in model.named_parameters():
    if not n.startswith("encoder."):
        p.requires_grad_(True)   # Qwen's decision head is new and must train
trainable = [n for n, p in model.named_parameters() if p.requires_grad]
assert not [n for n in trainable if n.startswith("encoder.") and "lora_" not in n], "backbone weights would train"
assert any("lora_" in n for n in trainable), "no LoRA adapters are trainable"
YES = 0 if getattr(model, "is_clef", False) else 1   # position of the "yes" option in the logits

train_items, train_report = FastDecisionModel.build_dataset([to_unsloth(r, t) for r, t in zip(train_rows, soft)], tokenizer, model)
val_items, val_report = FastDecisionModel.build_dataset([to_unsloth(r) for r in val_rows], tokenizer, model)
print("train:", train_report, "\nval:  ", val_report)
assert train_report["skipped"] == 0 and val_report["skipped"] == 0, "records were skipped"
assert [it["row"] for it in val_items] == list(range(len(val_rows))), "val order changed"
assert val_report["truncated"] == 0, "val inputs exceed the maximum length"


# Validation metrics: ranking per review by P(include), ties broken by record id
trainer = None


def compute_metrics(pred):
    logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    p = torch.softmax(torch.tensor(np.asarray(logits)[:, :2], dtype=torch.float32), -1)[:, YES].numpy()
    assert len(p) == len(val_meta)
    df = pd.DataFrame({"review_id": [m["review_id"] for m in val_meta], "record_id": [m["record_id"] for m in val_meta],
                       "p": p, **{f"label_{k}": v for k, v in val_labels.items()}})
    out = {}
    for level in ("abstract", "final"):
        per_review = []
        for _, g in df.groupby("review_id"):
            m = metrics(g.sort_values(["p", "record_id"], ascending=[False, True])[f"label_{level}"].astype(int))
            if m:
                per_review.append(m)
        mean = pd.DataFrame(per_review).mean() * 100
        for name, key in (("MAP", "map"), ("WSS@95%", "wss95"), ("TNR@95%", "tnr95"), ("R@10%", "r10")):
            out[f"{key}_{level}"] = round(float(mean[name]), 2)
        out[f"reviews_{level}"] = len(per_review)
    step = trainer.state.global_step
    (RUN / "val_scores").mkdir(exist_ok=True)
    df.to_json(RUN / "val_scores" / f"step-{step:06d}.jsonl", orient="records", lines=True)
    epoch = round(step * 64 / len(train_items), 2)
    with open(RUN / "val_metrics.jsonl", "a") as f:
        f.write(json.dumps({"step": step, "epoch": epoch, **out}) + "\n")
    print(f"step {step} (epoch {epoch}):", out, flush=True)
    return out


class EvalAtSteps(TrainerCallback):
    """Evaluate at EVAL_STEPS; stop at --max-steps (the schedule may be longer)."""
    def on_step_end(self, a, state, control, **kw):
        control.should_evaluate = state.global_step in EVAL_STEPS
        control.should_save = args.save_checkpoints and state.global_step in EVAL_STEPS
        if state.global_step >= args.max_steps:
            control.should_training_stop = True
        return control


targs = TrainingArguments(output_dir=str(RUN / "checkpoints"), per_device_train_batch_size=args.batch,
                          gradient_accumulation_steps=64 // args.batch, per_device_eval_batch_size=args.eval_batch,
                          max_steps=SCHEDULE, learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=args.warmup_steps, weight_decay=0.01,
                          bf16=True, logging_steps=10, eval_strategy="no", save_strategy="no", save_total_limit=None, seed=args.seed,
                          report_to="none")
trainer = DecisionTrainer(model=model, tokenizer=tokenizer, args=targs, train_dataset=train_items,
                          eval_dataset=val_items, compute_metrics=compute_metrics, head_learning_rate=1e-4,
                          callbacks=[EvalAtSteps()])

(RUN / "run_config.json").write_text(json.dumps({
    "model": args.model, "revision": MODELS[args.model], "requests": args.requests, "build_config": build_config,
    "max_seq_length": 8192, "lora_r": 16, "lora_alpha": 16, "lr": 2e-4, "head_lr": 1e-4, "schedule": f"cosine over {SCHEDULE} steps, warmup {args.warmup_steps} steps",
    "save_checkpoints": args.save_checkpoints, "distillation": soft_info,
    "max_steps": args.max_steps, "schedule_steps": SCHEDULE, "eval_steps": EVAL_STEPS, "batch": args.batch, "accum": 64 // args.batch,
    "train_requests": len(train_items), "epochs_at_max_steps": round(args.max_steps * 64 / len(train_items), 2),
    "seed": args.seed, "gpu": "physical GPU 1", "train_report": train_report, "val_report": val_report},
    indent=2, default=str))
(RUN / "pip_freeze.txt").write_text(subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                                   capture_output=True, text=True).stdout)

trainer.train()
log = [l for l in trainer.state.log_history if "loss" in l]
(RUN / "train_log.jsonl").write_text("".join(json.dumps(l) + "\n" for l in log))
print("done:", RUN)
