#!/usr/bin/env python3
"""LLM-as-Judge inference. v2: local paths, Falcon fallback, vLLM 0.19."""
import argparse, json, os, sys, time, math
import numpy as np

os.environ.setdefault("HF_HOME", "/autodl-fs/data/partitioned-bayes-rlhf/hf_cache")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

JUDGE_TEMPLATE = """You are comparing two responses to a prompt. Read the prompt and both responses, then answer which response is better overall (more helpful, more correct, clearer).

PROMPT:
{prompt}

RESPONSE A:
{response_a}

RESPONSE B:
{response_b}

Which response is better? Answer with a single letter: A or B.

Answer:"""

def build_prompts(pairs, tokenizer, order="AB"):
    out = []
    has_chat = hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None
    for p in pairs:
        a, b = (p["response_a"], p["response_b"]) if order == "AB" else (p["response_b"], p["response_a"])
        content = JUDGE_TEMPLATE.format(prompt=p["prompt"][:2000], response_a=a[:1500], response_b=b[:1500])
        if has_chat:
            try:
                chat = tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
                out.append(chat); continue
            except Exception:
                pass
        out.append(content)
    return out

def extract_ab_logprob(output, tok_a_ids, tok_b_ids):
    gen = output.outputs[0]
    if not gen.logprobs or len(gen.logprobs) == 0:
        return None, None
    first_tok = gen.logprobs[0]
    lp_a, lp_b = -float("inf"), -float("inf")
    for tid, info in first_tok.items():
        lp = info.logprob if hasattr(info, "logprob") else info
        if tid in tok_a_ids: lp_a = max(lp_a, lp)
        if tid in tok_b_ids: lp_b = max(lp_b, lp)
    if lp_a == -float("inf") and lp_b == -float("inf"):
        return None, None
    m = max(lp_a, lp_b)
    ea = math.exp(lp_a - m) if lp_a > -float("inf") else 0.0
    eb = math.exp(lp_b - m) if lp_b > -float("inf") else 0.0
    s = ea + eb
    return (ea / s, eb / s) if s > 0 else (None, None)

def get_ab_token_ids(tokenizer):
    a_ids, b_ids = set(), set()
    for v in ["A", " A", "a", " a"]:
        ids = tokenizer.encode(v, add_special_tokens=False)
        if len(ids) == 1: a_ids.add(ids[0])
    for v in ["B", " B", "b", " b"]:
        ids = tokenizer.encode(v, add_special_tokens=False)
        if len(ids) == 1: b_ids.add(ids[0])
    return a_ids, b_ids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pairs_file", required=True)
    ap.add_argument("--output_file", required=True)
    ap.add_argument("--n_pairs", type=int, default=1000)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--dtype", default="auto")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    with open(args.pairs_file) as f:
        pairs = json.load(f)
    if args.n_pairs > 0:
        pairs = pairs[:args.n_pairs]
    name = os.path.basename(args.model)
    print(f"[{name}] {len(pairs)} pairs", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    a_ids, b_ids = get_ab_token_ids(tokenizer)
    print(f"[{name}] A-ids={sorted(a_ids)}, B-ids={sorted(b_ids)}", flush=True)

    t0 = time.time()
    try:
        llm = LLM(model=args.model, dtype=args.dtype, max_model_len=args.max_model_len,
                   gpu_memory_utilization=args.gpu_memory_utilization, enforce_eager=True, trust_remote_code=True)
    except Exception as e:
        print(f"[{name}] load failed ({e}), retrying float16", flush=True)
        llm = LLM(model=args.model, dtype="float16", max_model_len=args.max_model_len,
                   gpu_memory_utilization=args.gpu_memory_utilization, enforce_eager=True, trust_remote_code=True)
    print(f"[{name}] loaded in {time.time()-t0:.1f}s", flush=True)

    sampling = SamplingParams(temperature=0.0, max_tokens=3, logprobs=20)
    results = {"model": name, "n_pairs": len(pairs), "pairs": []}

    for order in ["AB", "BA"]:
        prompts = build_prompts(pairs, tokenizer, order=order)
        t1 = time.time()
        outs = llm.generate(prompts, sampling)
        dur = time.time() - t1
        print(f"[{name}] {order}: {len(outs)} in {dur:.1f}s", flush=True)
        for i, out in enumerate(outs):
            p_a, p_b = extract_ab_logprob(out, a_ids, b_ids)
            if i >= len(results["pairs"]):
                results["pairs"].append({"id": pairs[i].get("id", i), "gold": pairs[i].get("gold_label")})
            if order == "AB":
                results["pairs"][i]["p_a_ab"] = p_a; results["pairs"][i]["p_b_ab"] = p_b
            else:
                results["pairs"][i]["p_a_ba"] = p_b; results["pairs"][i]["p_b_ba"] = p_a

    for r in results["pairs"]:
        vals = [v for v in [r.get("p_a_ab"), r.get("p_a_ba")] if v is not None]
        r["p_a_mean"] = float(np.mean(vals)) if vals else None

    valid = sum(1 for r in results["pairs"] if r["p_a_mean"] is not None)
    print(f"[{name}] Valid: {valid}/{len(results['pairs'])}", flush=True)

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{name}] Saved {args.output_file}", flush=True)

if __name__ == "__main__":
    main()
