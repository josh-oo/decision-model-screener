import json, os, sys, time
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"; os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from vllm import LLM
def main():
    model, tokf, task, mb = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    rows = [json.loads(l) for l in open(tokf)]
    llm = LLM(model=model, runner="pooling", dtype="bfloat16", max_model_len=8192, max_num_seqs=256, max_num_batched_tokens=mb,
              enable_prefix_caching=False, gpu_memory_utilization=0.9, **({"limit_mm_per_prompt": {"image": 0, "video": 0}} if len(sys.argv) > 5 else {}))
    P = [{"prompt_token_ids": r["ids"]} for r in rows]
    llm.encode(P[:256], use_tqdm=False, pooling_task=task)
    t = time.perf_counter(); llm.encode(P, use_tqdm=False, pooling_task=task); dt = time.perf_counter() - t
    print("RESULT", json.dumps({"task": task, "mb": mb, "mm_off": len(sys.argv) > 5, "rec_per_s": round(len(rows) / dt, 1), "tok_per_s": round(sum(len(r["ids"]) for r in rows) / dt)}), flush=True)
    
if __name__ == '__main__':
    main()