"""Baseline: Unsloth/HF inference path (as scripts/score_train.py) on the fixed sample, physical GPU 0.
.venv/bin/python bench/bench_hf.py --run <run> --step <n> --n 2000 --batch 32
Prints records/s and writes p per row for the vLLM agreement check."""
import argparse, json, os, sys, time
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("--run"); ap.add_argument("--step", type=int); ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--batch", type=int, default=32); ap.add_argument("--out")
args = ap.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"; os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import unsloth  # noqa
import numpy as np, torch
from transformers import TrainingArguments
from unsloth import FastDecisionModel, DecisionTrainer
from unsloth.models.decision import _load_clef_checkpoint
ROOT = Path("/home/joshua/decision-model-screener"); RUN = ROOT / "runs" / args.run
cfg = json.loads((RUN / "run_config.json").read_text())
rows = [json.loads(l) for l in open(Path(__file__).resolve().parent.parent / "data/processed/bench/sample5000.jsonl")][: args.n]
model, tok = FastDecisionModel.from_pretrained(model_name=cfg["model"], revision=cfg["revision"], max_seq_length=cfg["max_seq_length"],
                                               dtype=torch.bfloat16, load_in_4bit=False, random_state=cfg["seed"])
model = FastDecisionModel.get_peft_model(model, r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=0.0, random_state=cfg["seed"])
_load_clef_checkpoint(model, RUN / "checkpoints" / f"checkpoint-{args.step}"); model.eval()
YES = 0 if getattr(model, "is_clef", False) else 1
tr = DecisionTrainer(model=model, tokenizer=tok, args=TrainingArguments(output_dir="/tmp/hf_bench_trainer",
                     per_device_eval_batch_size=args.batch, bf16=True, report_to="none"))
items, rep = FastDecisionModel.build_dataset([{"state": r["state"], "questions": {n: {k: v for k, v in q.items() if k != "label"}
                                               for n, q in r["questions"].items()}, "gold": {n: q["label"] for n, q in r["questions"].items()}} for r in rows], tok, model)
assert rep["skipped"] == 0
tr.predict(items[:64])                        # warm-up
torch.cuda.synchronize(); t = time.perf_counter()
pred = tr.predict(items).predictions
pred = pred[0] if isinstance(pred, tuple) else pred
torch.cuda.synchronize(); dt = time.perf_counter() - t
z = torch.tensor(np.asarray(pred, dtype=np.float32))[:, :2]
p = torch.softmax(z, -1)[:, YES].tolist()
ntok = sum(len(it["input_ids"]) for it in items)
print(json.dumps({"engine": "unsloth-hf", "model": cfg["model"], "n": len(items), "batch": args.batch, "seconds": round(dt, 1),
                  "rec_per_s": round(len(items) / dt, 1), "tok_per_s": round(ntok / dt)}))
if args.out:
    with open(args.out, "w") as f:
        for r, pp in zip(rows, p): f.write(json.dumps({"row": r["row"], "p": pp}) + "\n")
