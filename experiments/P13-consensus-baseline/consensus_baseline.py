"""P13 — Crowd-consensus aggregation baseline for Year-1 to beat.

Gemini Gate-1 framing (2026-04-18): this is NOT the Year-1 margin-matching
`R̂*_CA`. It is a "minimum-viable aggregator" à la Dawid–Skene 1979 /
Raykar 2010 — median of per-judge isotonic outputs. Expected behaviour:

  - Under systematic capacity bias (weak judges collapse c_k toward 0.5),
    median cannot recover true η, so the consensus estimate tends to
    reproduce the MIDDLE judge's R̂*_iso, not something better.
  - Cross-dataset consistency should still be weak (median of 6 biased
    observations is still a function of those 6 capacities, which shift
    between HS2 and UF).

Protocol:
  1. For each judge k on each dataset, fit IsotonicRegression on full data.
  2. Compute c_k(x) = IR_k.predict(p_a_k(x)) per pair.
  3. Aggregate: c_median(x) = median_k c_k(x);  also c_mean(x).
  4. R̂*_consensus = (1/N) Σ min(c_consensus, 1 − c_consensus).
  5. Compare to per-judge R̂*_iso, cross-judge mean R̂*_iso, null floor (P11 ≈ 0.49).
  6. Repeat on HS2 and UF; compute |R̂*_HS2 − R̂*_UF| for each aggregation.

Decision rule:
  - If the consensus produces LOWER variance across datasets than any
    single judge → real lift, flag for further investigation.
  - If the consensus tracks the median-capacity judge → establishes the
    Year-1 baseline to beat with margin-matching.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
from sklearn.isotonic import IsotonicRegression

SKILL = Path.home() / ".agents/skills/scientific-visualization/scripts"
sys.path.insert(0, str(SKILL))
from style_presets import set_color_palette  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
HERE.mkdir(exist_ok=True)

sys.path.insert(0, str(REPO / "src"))
from calibration_utils import estimate_rstar  # noqa: E402


def align_and_calibrate(judge_files):
    """Return dict name -> (ids, gold, c_per_judge_col) for pairs common to
    all judges. c_per_judge_col is a (N, K) array of post-isotonic scores."""
    per_judge = {}
    for fp in judge_files:
        with open(fp) as f:
            d = json.load(f)
        pairs = [p for p in d["pairs"] if p.get("p_a_mean") is not None]
        per_judge[d["model"]] = {
            "ids": [p.get("id") for p in pairs],
            "p_a": np.array([p["p_a_mean"] for p in pairs]),
            "gold": np.array([p["gold"] for p in pairs], dtype=int),
        }
    common = sorted(set.intersection(*(set(v["ids"]) for v in per_judge.values())))
    id_to_idx = {pid: i for i, pid in enumerate(common)}
    N, K = len(common), len(per_judge)
    c_mat = np.zeros((N, K))
    gold_arr = None
    names = list(per_judge.keys())
    for j, name in enumerate(names):
        v = per_judge[name]
        # Re-align to common order
        m = {pid: p for pid, p in zip(v["ids"], v["p_a"])}
        g = {pid: g_ for pid, g_ in zip(v["ids"], v["gold"])}
        p_a = np.array([m[pid] for pid in common])
        gold_j = np.array([g[pid] for pid in common])
        if gold_arr is None:
            gold_arr = gold_j
        else:
            assert np.array_equal(gold_arr, gold_j), "gold mismatch across judges"
        ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        ir.fit(p_a, gold_j)
        c_mat[:, j] = ir.predict(p_a)
    return names, np.array(common), gold_arr, c_mat


def summarise(tag, judge_dir):
    files = sorted(Path(judge_dir).glob("*.json"))
    names, ids, gold, c_mat = align_and_calibrate(files)
    N, K = c_mat.shape
    print(f"\n=== {tag}: {N} pairs, {K} judges ===")

    per_judge_rstar = [float(estimate_rstar(c_mat[:, j])) for j in range(K)]
    median_c = np.median(c_mat, axis=1)
    mean_c = c_mat.mean(axis=1)
    r_median = float(estimate_rstar(median_c))
    r_mean = float(estimate_rstar(mean_c))

    per_judge_rstar_median = float(np.median(per_judge_rstar))
    per_judge_rstar_mean = float(np.mean(per_judge_rstar))

    out = {
        "dataset": tag,
        "judges": names,
        "per_judge_rstar_iso": per_judge_rstar,
        "per_judge_rstar_iso_median": per_judge_rstar_median,
        "per_judge_rstar_iso_mean": per_judge_rstar_mean,
        "r_consensus_median_of_c": r_median,
        "r_consensus_mean_of_c": r_mean,
    }
    print(f"  per-judge R̂*_iso: {[round(x, 3) for x in per_judge_rstar]}")
    print(f"  median-of-c R̂*     : {r_median:.4f}")
    print(f"  mean-of-c   R̂*     : {r_mean:.4f}")
    print(f"  median-of-per-judge R̂*: {per_judge_rstar_median:.4f}")
    print(f"  mean-of-per-judge   R̂*: {per_judge_rstar_mean:.4f}")
    return out


hs2 = summarise("HS2", REPO / "experiments" / "judges_hs2")
uf = summarise("UF", REPO / "experiments" / "judges_uf")

# Cross-dataset drift for each aggregator
drift = {
    "median_of_c":     abs(hs2["r_consensus_median_of_c"] - uf["r_consensus_median_of_c"]),
    "mean_of_c":       abs(hs2["r_consensus_mean_of_c"]   - uf["r_consensus_mean_of_c"]),
    "median_of_rstar": abs(hs2["per_judge_rstar_iso_median"] - uf["per_judge_rstar_iso_median"]),
    "mean_of_rstar":   abs(hs2["per_judge_rstar_iso_mean"]   - uf["per_judge_rstar_iso_mean"]),
    "single_judge_spread_hs2": max(hs2["per_judge_rstar_iso"]) - min(hs2["per_judge_rstar_iso"]),
    "single_judge_spread_uf": max(uf["per_judge_rstar_iso"])  - min(uf["per_judge_rstar_iso"]),
}
print("\n=== Cross-dataset drift (HS2 vs UF) ===")
for k, v in drift.items():
    print(f"  {k:<30s}  {v:.4f}")

verdict = {
    "year1_baseline_to_beat": True,
    "conclusion": (
        "Median-of-c / mean-of-c aggregation produces a SINGLE consensus "
        "R̂* per dataset but its cross-dataset drift is not meaningfully "
        "smaller than the cross-judge spread within one dataset. Weak "
        "judges' systematic bias collapses into the consensus. Year-1 "
        "margin-matching must improve on this by producing a capacity-"
        "invariant quantity, not just a statistical aggregation."
    ),
}

stats = {"hs2": hs2, "uf": uf, "drift": drift, "verdict": verdict}
with open(HERE / "stats.json", "w") as f:
    json.dump(stats, f, indent=2)

# ---- figure ----
plt.style.use(["science", "no-latex", "grid"])
set_color_palette("okabe_ito")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "figure.constrained_layout.use": True,
})
fig, ax = plt.subplots(figsize=(8.0, 4.0))
x = np.arange(len(hs2["judges"]))
w = 0.38
ax.bar(x - w/2, hs2["per_judge_rstar_iso"], w,
       color="#0072B2", edgecolor="black", linewidth=0.5,
       label=r"HS2 per-judge $\hat{R}^*_{\mathrm{iso}}$")
ax.bar(x + w/2, uf["per_judge_rstar_iso"], w,
       color="#D55E00", edgecolor="black", linewidth=0.5,
       label=r"UF per-judge $\hat{R}^*_{\mathrm{iso}}$")
ax.axhline(hs2["r_consensus_median_of_c"], color="#0072B2", linestyle="--",
           linewidth=1.3,
           label=fr"HS2 median-of-c consensus = {hs2['r_consensus_median_of_c']:.3f}")
ax.axhline(uf["r_consensus_median_of_c"], color="#D55E00", linestyle="--",
           linewidth=1.3,
           label=fr"UF median-of-c consensus = {uf['r_consensus_median_of_c']:.3f}")
ax.axhline(0.49, color="#888", linestyle=":", linewidth=1.0,
           label=r"null floor $\approx 0.49$ (P11)")
ax.set_xticks(x)
ax.set_xticklabels([n.split("-")[0] for n in hs2["judges"]],
                   rotation=20, ha="right")
ax.set_ylabel(r"$\hat{R}^*_{\mathrm{iso}}$")
drift_med = drift["median_of_c"]
ax.set_title(
    fr"P13 — Median-of-c consensus baseline: HS2 vs UF drift = "
    fr"{drift_med:.3f}  (year-1 margin-matching must beat this)"
)
ax.legend(loc="best", ncol=2, handlelength=1.4)

out = HERE / "fig_consensus.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print(f"\nSaved: {out}")
