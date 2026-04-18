#!/usr/bin/env python3
"""
Build unified preference-pair JSON for LLM-judge inference.

Supports:
    --source ultrafeedback    (HuggingFaceH4/ultrafeedback_binarized, train_prefs split)
    --source helpsteer2       (nvidia/HelpSteer2, construct pairs from same-prompt responses)

Output format (list of dicts):
    {
        "id": "uf_0001",
        "prompt": "...",
        "response_a": "...",
        "response_b": "...",
        "gold_label": 1,       # 1 = A is preferred, 0 = B is preferred (from dataset metadata)
        "source": "ultrafeedback"
    }
"""
import argparse, json, os, random
from collections import defaultdict
from itertools import combinations

os.environ.setdefault("HF_HOME", "/autodl-fs/data/partitioned-bayes-rlhf/hf_cache")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from datasets import load_dataset


def build_ultrafeedback(n):
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    print(f"[UF] loaded {len(ds)} samples")
    out = []
    rng = random.Random(42)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    for k, i in enumerate(idxs[:n]):
        s = ds[i]
        prompt = s["prompt"]
        chosen = s["chosen"]
        rejected = s["rejected"]
        # chosen/rejected are multi-turn; take last assistant message
        def last_assistant(msgs):
            for m in reversed(msgs):
                if m.get("role") == "assistant":
                    return m.get("content", "")
            return ""
        chosen_text = last_assistant(chosen) if isinstance(chosen, list) else str(chosen)
        rejected_text = last_assistant(rejected) if isinstance(rejected, list) else str(rejected)
        if not chosen_text or not rejected_text:
            continue
        # Randomize A/B assignment per pair; record gold
        if rng.random() < 0.5:
            resp_a, resp_b, gold = chosen_text, rejected_text, 1
        else:
            resp_a, resp_b, gold = rejected_text, chosen_text, 0
        out.append({
            "id": f"uf_{k:05d}",
            "prompt": prompt,
            "response_a": resp_a,
            "response_b": resp_b,
            "gold_label": gold,
            "source": "ultrafeedback",
        })
    return out


def build_helpsteer2(n):
    ds = load_dataset("nvidia/HelpSteer2", split="train")
    print(f"[HS2] loaded {len(ds)} samples")
    # Group by prompt
    groups = defaultdict(list)
    for i, s in enumerate(ds):
        groups[s["prompt"][:300]].append({
            "response": s["response"],
            "h": s["helpfulness"],
            "c": s["correctness"],
        })
    multi = {k: v for k, v in groups.items() if len(v) >= 2}
    print(f"[HS2] {len(multi)} prompts with >= 2 responses")

    out = []
    rng = random.Random(42)
    items = list(multi.items())
    rng.shuffle(items)
    k = 0
    for prompt, resps in items:
        for ra, rb in combinations(resps, 2):
            if ra["h"] == rb["h"]:
                continue
            # Gold: response with higher helpfulness wins
            if ra["h"] > rb["h"]:
                resp_a, resp_b, gold = ra["response"], rb["response"], 1
            else:
                resp_a, resp_b, gold = ra["response"], rb["response"], 0
            # Randomize A/B so there's no positional leakage
            if rng.random() < 0.5:
                resp_a, resp_b = resp_b, resp_a
                gold = 1 - gold
            out.append({
                "id": f"hs_{k:05d}",
                "prompt": prompt,
                "response_a": resp_a,
                "response_b": resp_b,
                "gold_label": gold,
                "source": "helpsteer2",
            })
            k += 1
            if k >= n:
                return out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["ultrafeedback", "helpsteer2"], required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if args.source == "ultrafeedback":
        pairs = build_ultrafeedback(args.n)
    else:
        pairs = build_helpsteer2(args.n)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(pairs, f, indent=1)
    print(f"Saved {len(pairs)} pairs to {args.output}")
    # Sanity: label balance
    labels = [p["gold_label"] for p in pairs]
    print(f"Gold label distribution: {sum(labels)}/{len(labels)} = {sum(labels)/len(labels):.3f}")


if __name__ == "__main__":
    main()
