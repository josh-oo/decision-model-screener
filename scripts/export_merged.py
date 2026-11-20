"""Merge a trained decision model's LoRA adapters into its pinned base and save a full Clef-layout folder for vLLM.

  python scripts/export_merged.py --run <run name> --adapter best_adapter        # or --adapter checkpoints/checkpoint-844

Loads the base exactly as in training (pinned revision from run_config.json), loads the adapters and the joint head,
merges the adapters into the backbone and writes runs/<run>/merged/<adapter name>/ (backbone safetensors, config.json,
tokenizer, joint_head.safetensors, joint_head_config.json): the layout vLLM serves as a Clef joint-schema model.
Uses physical GPU 0 (throughput experiments, with permission; training stays on GPU 1).
"""
import argparse, json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = {"unsloth/Qwen3.5-0.8B": "23c69c53358a07516b5827588b3fdb12ae78fd65",   # pinned Hub revisions
          "unsloth/Qwen3.5-4B": "3764fa359b9082ea5a1e4a5e3ac3aaf6e9671636"}

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--adapter", required=True, help="folder under runs/<run>/ with adapter_model + joint_head")
ap.add_argument("--gpu", default="0")
args = ap.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
import unsloth  # noqa: E402,F401
import torch  # noqa: E402
from unsloth import FastDecisionModel  # noqa: E402
from unsloth.models.decision import _load_clef_checkpoint, _save_clef  # noqa: E402

RUN = ROOT / "runs" / args.run
SRC = RUN / args.adapter
OUT = RUN / "merged" / args.adapter.replace("/", "__")
cfg = json.loads((RUN / "run_config.json").read_text())
assert cfg["revision"] == MODELS[cfg["model"]], "unpinned or unknown base model"
assert not OUT.exists(), f"{OUT} exists"

model, tokenizer = FastDecisionModel.from_pretrained(model_name=cfg["model"], revision=cfg["revision"],
                                                     max_seq_length=cfg["max_seq_length"], dtype=torch.bfloat16,
                                                     load_in_4bit=False, random_state=cfg["seed"])
model = FastDecisionModel.get_peft_model(model, r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=0.0,
                                         random_state=cfg["seed"])
_load_clef_checkpoint(model, SRC)
model.eval()
model.encoder = model.encoder.merge_and_unload()          # no peft_config left: _save_clef writes merged weights
_save_clef(model, OUT, tokenizer)
if (OUT / "model.safetensors").is_file() and not (OUT / "model.safetensors.index.json").exists():
    # Without an index vLLM loads every *.safetensors as backbone weights, joint_head.safetensors included, and fails.
    from safetensors import safe_open
    with safe_open(str(OUT / "model.safetensors"), "pt") as f:
        keys = list(f.keys())
    (OUT / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": (OUT / "model.safetensors").stat().st_size},
         "weight_map": {k: "model.safetensors" for k in keys}}))
print("wrote", OUT, sorted(p.name for p in OUT.iterdir()))
