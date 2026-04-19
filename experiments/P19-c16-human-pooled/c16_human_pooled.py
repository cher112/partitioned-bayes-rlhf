"""C16 — Apply P16 pooled-anchor FCI to HS2-Preference 9125 human-annotator pairs.

9125 pairs × 3 annotators, each annotator gives preference_strength in
{-3, -2, -1, 1, 2, 3} (sign = which response preferred, magnitude =
how strongly). We map each annotator's strength to a [0,1] probability
via (strength + 3.5) / 7 (centered around 0.5 when strength = 0 not
present here; otherwise symmetric). Then treat 3 annotators as K=3
judges and apply P16 pooled-anchor FCI.

Gemini + Codex both rated this 4–6/10 importance; Codex flagged K=3 as
potentially unstable (possibly high variance or anchor misalignment).
We take either outcome: if drift is clean vs HS2-LLM and UF-LLM, that
shows R̂*_CA generalises to human-only setting; if noisy, spin as
"low-K pressure test motivating Year-1 robustness research".
"""
from __future__ import annotations
import gzip
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
HERE.mkdir(exist_ok=True)
sys.path.insert(0, str(REPO / "src"))
from calibration_utils import estimate_rstar  # noqa: E402

DATA = REPO / "data/hs2_extra/preference/preference.jsonl.gz"
ALPHA = 0.01


def strength_to_prob(strength: int) -> float:
    """Map preference strength in {-3..-1, 1..3} to p_a in (0, 1).

    strength > 0 means annotator preferred response_1 (A).
    Use sigmoid-like map: p = 1/(1 + exp(-strength)).
    Strength 3 -> p ~ 0.95; strength 1 -> p ~ 0.73; strength -1 -> p ~ 0.27.
    """
    return float(1.0 / (1.0 + np.exp(-strength)))


# Load
rows = []
with gzip.open(DATA, "rt") as f:
    for line in f:
        obj = json.loads(line)
        raw = obj.get("all_preferences_unprocessed")
        if not raw:
            continue
        try:
            anno = json.loads(raw.replace("'", '"')) if isinstance(raw, str) else raw
        except Exception:
            try:
                anno = eval(raw)
            except Exception:
                continue
        try:
            agg = int(obj["preference_strength"])
        except (ValueError, TypeError, KeyError):
            continue
        strengths = [int(a.get("strength", 0)) for a in anno if isinstance(a, dict)]
        if len(strengths) != 3 or any(s == 0 for s in strengths):
            continue
        rows.append({"agg": agg, "strengths": strengths})
print(f"Parsed {len(rows)} rows with exactly 3 annotators and non-zero strengths.")

# Gold label: sign of the aggregate strength (which response majority prefers).
# Response_1 (A) wins → gold = 1; response_2 (B) wins → gold = 0.
gold = np.array([1 if r["agg"] > 0 else 0 for r in rows], dtype=int)
print(f"gold balance: {gold.mean():.4f}")

# Per-annotator p_a matrix
p_mat = np.zeros((len(rows), 3))
for i, r in enumerate(rows):
    for j, s in enumerate(r["strengths"]):
        p_mat[i, j] = strength_to_prob(s)

# Per-annotator isotonic + R̂*_iso
per_annotator_iso = []
for j in range(3):
    ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    ir.fit(p_mat[:, j], gold)
    c_iso_j = ir.predict(p_mat[:, j])
    r = float(estimate_rstar(c_iso_j))
    acc = float(((p_mat[:, j] >= 0.5).astype(int) == gold).mean())
    per_annotator_iso.append({"annotator_idx": j, "acc": acc, "rstar_iso": r})
    print(f"  annotator_{j}: acc = {acc:.4f}   R̂*_iso = {r:.4f}")

# Build a c_iso matrix across 3 annotators (K=3)
c_mat = np.zeros_like(p_mat)
for j in range(3):
    ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    ir.fit(p_mat[:, j], gold)
    c_mat[:, j] = ir.predict(p_mat[:, j])

# P16 pooled FCI on K=3
pool = c_mat.flatten()
rho_0 = float(np.quantile(pool, ALPHA))
rho_1 = 1.0 - float(np.quantile(pool, 1 - ALPHA))
denom = 1 - rho_0 - rho_1
print(f"\nPooled anchors: ρ_0 = {rho_0:.4f}, ρ_1 = {rho_1:.4f}, stretch = {1/denom:.3f}")
p_corr = np.clip((c_mat - rho_0) / denom, 0, 1)
p_bar = p_corr.mean(axis=1)
r_ca = float(np.minimum(p_bar, 1 - p_bar).mean())
mean_iso = float(np.mean([a["rstar_iso"] for a in per_annotator_iso]))
print(f"C16 pooled-FCI R̂*_CA (human K=3) = {r_ca:.4f}")
print(f"Per-annotator iso mean (ref)       = {mean_iso:.4f}")

# Compare to LLM-based C15 references
hs2_llm_ca = 0.407   # from P16 HS2 5k
uf_llm_ca = 0.321    # from P16 UF 5k
print(f"\nReference (LLM-based P16):")
print(f"  HS2 LLM pooled-FCI = {hs2_llm_ca:.3f}")
print(f"  UF  LLM pooled-FCI = {uf_llm_ca:.3f}")
print(f"Gap between C16 (human) and HS2-LLM FCI = {abs(r_ca - hs2_llm_ca):.4f}")

stats = {
    "n_pairs": int(len(rows)),
    "n_annotators_per_pair": 3,
    "alpha": ALPHA,
    "per_annotator_iso": per_annotator_iso,
    "mean_per_annotator_iso": mean_iso,
    "rho_0": rho_0, "rho_1": rho_1, "stretch": float(1 / denom),
    "rstar_ca_human_K3": r_ca,
    "llm_reference_P16": {
        "hs2_llm_rstar_ca": hs2_llm_ca,
        "uf_llm_rstar_ca": uf_llm_ca,
    },
    "gap_vs_hs2_llm": float(abs(r_ca - hs2_llm_ca)),
    "gap_vs_uf_llm": float(abs(r_ca - uf_llm_ca)),
    "interpretation": (
        "C16 applies the pooled-anchor FCI estimator (P16) to the "
        "human-annotator preference data without any LLM judge. With "
        "K=3 annotators per pair on 9125 HS2-Preference pairs, the "
        "estimator produces a single R̂*_CA that can be compared to "
        "the LLM-derived values (HS2 0.407, UF 0.321). A small gap "
        "suggests the method generalises beyond LLM-soft-label input; "
        "a large gap suggests low-K regime requires stronger theory "
        "(motivates Year-1 low-K extension)."
    ),
}
with open(HERE / "stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print(f"\nSaved: {HERE / 'stats.json'}")
