#!/usr/bin/env python3
"""Pre-flight check: verify all params before running the experiment."""
import json, os, sys
import numpy as np
sys.path.insert(0, "/autodl-fs/data/partitioned-bayes-rlhf/src")
os.chdir("/autodl-fs/data/partitioned-bayes-rlhf")

from transformers import AutoTokenizer

MODELS = [
    "models/Qwen2.5-7B-Instruct",
    "models/Mistral-7B-Instruct-v0.3",
    "models/OLMo-7B-Instruct",
    "models/granite-3.0-8b-instruct",
    "models/falcon-7b-instruct",
]

# 1. Token ID verification
print("=== 1. Token ID verification ===")
for m in MODELS:
    tok = AutoTokenizer.from_pretrained(m, trust_remote_code=True)
    for v in ["A", " A", "B", " B"]:
        ids = tok.encode(v, add_special_tokens=False)
        decoded = tok.decode(ids)
        print(f"  {os.path.basename(m):30s}  encode({v!r:4s}) -> {str(ids):15s}  decode -> {decoded!r}")
    print()

# 2. Prompt token length estimation
print("=== 2. Prompt token length check ===")
pairs = json.load(open("data/hs2_pairs.json"))

TEMPLATE = (
    "You are comparing two responses to a prompt. "
    "Read the prompt and both responses, then answer which response is better overall.\n\n"
    "PROMPT:\n{prompt}\n\nRESPONSE A:\n{response_a}\n\nRESPONSE B:\n{response_b}\n\n"
    "Which response is better? Answer with a single letter: A or B.\n\nAnswer:"
)

tok = AutoTokenizer.from_pretrained("models/Qwen2.5-7B-Instruct", trust_remote_code=True)
lengths = []
for p in pairs[:200]:
    text = TEMPLATE.format(
        prompt=p["prompt"][:2000],
        response_a=p["response_a"][:1500],
        response_b=p["response_b"][:1500],
    )
    lengths.append(len(tok.encode(text)))

print(f"  min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.0f}, p95={np.percentile(lengths, 95):.0f}")
print(f"  max_model_len=4096 -> overflow count: {sum(1 for l in lengths if l > 4096)}/{len(lengths)}")
if max(lengths) > 4096:
    print(f"  WARNING: some prompts exceed 4096! Consider raising --max_model_len or trimming responses further")

# 3. Pairs structure
print("\n=== 3. Pairs structure ===")
p = pairs[0]
print(f"  keys: {list(p.keys())}")
print(f"  id={p['id']}, gold={p['gold_label']}")
print(f"  prompt[:60]: {p['prompt'][:60]}...")
print(f"  resp_a[:40]: {p['response_a'][:40]}...")
print(f"  resp_b[:40]: {p['response_b'][:40]}...")
print(f"  total pairs: {len(pairs)}, gold balance: {sum(p['gold_label'] for p in pairs)}/{len(pairs)}")

# 4. Model weight files
print("\n=== 4. Model weight files ===")
for m in MODELS:
    safetensors = [f for f in os.listdir(m) if f.endswith(".safetensors")]
    config = os.path.exists(os.path.join(m, "config.json"))
    print(f"  {os.path.basename(m):30s}  safetensors={len(safetensors)}, config={config}")

print("\n=== ALL CHECKS PASSED ===" if True else "")
