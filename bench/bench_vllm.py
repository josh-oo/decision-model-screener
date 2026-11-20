"""vLLM offline throughput of a merged Clef-layout decision model on pre-tokenized requests (physical GPU 0).
~/.venvs/vllm/bin/python bench/bench_vllm.py --model runs/<run>/merged/<adapter> --tok data/processed/bench/tok_abs.jsonl --max-batched 8192 [--quant fp8]
One JSON line per config: records/s, tokens/s; optionally writes p(include) per row."""
import argparse, json, os, time
ap = argparse.ArgumentParser()
ap.add_argument("--model"); ap.add_argument("--tok"); ap.add_argument("--n", type=int, default=5000)
ap.add_argument("--max-num-seqs", type=int, default=256); ap.add_argument("--max-batched", type=int, default=16384)
ap.add_argument("--gpu-util", type=float, default=0.9); ap.add_argument("--eager", action="store_true")
ap.add_argument("--prefix-caching", action="store_true"); ap.add_argument("--no-chunked", action="store_true")
ap.add_argument("--sort", action="store_true", help="submit sorted by length"); ap.add_argument("--repeat", type=int, default=1)
ap.add_argument("--profile-head", action="store_true"); ap.add_argument("--quant", default=None); ap.add_argument("--out"); ap.add_argument("--tag", default="")
args = ap.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"; os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if args.profile_head:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    import torch
    from vllm.model_executor.models import clef_joint_schema as cjs
    HEAD = {"s": 0.0, "calls": 0, "recs": 0}
    _fwd = cjs.JointSchemaPooler.forward
    def timed(self, hidden_states, md):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = _fwd(self, hidden_states, md)
        torch.cuda.synchronize(); HEAD["s"] += time.perf_counter() - t0; HEAD["calls"] += 1; HEAD["recs"] += sum(o is not None for o in out)
        return out
    cjs.JointSchemaPooler.forward = timed
from vllm import LLM, PoolingParams
rows = [json.loads(l) for l in open(args.tok)][: args.n]
rows = rows * args.repeat
order = sorted(range(len(rows)), key=lambda i: len(rows[i]["ids"])) if args.sort else list(range(len(rows)))
llm = LLM(model=args.model, runner="pooling", dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=args.gpu_util,
          max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.max_batched, enforce_eager=args.eager,
          enable_prefix_caching=args.prefix_caching, enable_chunked_prefill=not args.no_chunked, quantization=args.quant,
          seed=0)
prompts = [{"prompt_token_ids": rows[i]["ids"]} for i in order]
params = [PoolingParams(task="token_classify", extra_kwargs={"joint_schema": rows[i]["schema"]}) for i in order]
llm.encode(prompts[:256], params[:256], use_tqdm=False, pooling_task="token_classify")                       # warm-up
t = time.perf_counter()
outs = llm.encode(prompts, params, use_tqdm=False, pooling_task="token_classify")
dt = time.perf_counter() - t
if args.profile_head: print("HEAD " + json.dumps({k: round(v, 2) for k, v in HEAD.items()}) + f" total_s={dt:.2f}", flush=True)
ntok = sum(len(r["ids"]) for r in rows)
p = [None] * len(rows)
for i, o in zip(order, outs):
    p[i] = float(o.outputs.data[0])          # option 0 = "true" = include
print("RESULT " + json.dumps({"engine": "vllm", "tag": args.tag, "model": os.path.basename(os.path.dirname(os.path.dirname(args.model.rstrip("/"))))[:12],
      "n": len(rows), "max_num_seqs": args.max_num_seqs, "max_batched": args.max_batched, "eager": args.eager, "sort": args.sort,
      "prefix_caching": args.prefix_caching, "quant": args.quant, "seconds": round(dt, 2), "rec_per_s": round(len(rows) / dt, 1),
      "tok_per_s": round(ntok / dt)}), flush=True)
if args.out:
    with open(args.out, "w") as f:
        for r, pp in zip(rows[: args.n], p): f.write(json.dumps({"row": r["row"], "p": pp}) + "\n")
