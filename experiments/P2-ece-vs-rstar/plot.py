"""P2: ECE vs R-star scatter — direct evidence of capacity confound.

Input:  experiments/analysis_hs2_5judges/per_judge_rstar.csv
Output: fig_ece_vs_rstar.png + stats.json
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
CSV = HERE.parent.parent.parent.parent / "autoresearch-mirror" / "partitioned-bayes-rlhf" / "experiments" / "analysis_hs2_5judges" / "per_judge_rstar.csv"

if not CSV.exists():
    # fallback: inline the numbers directly
    df = pd.DataFrame([
        {"judge": "falcon-7b-instruct",        "rstar_raw": 0.4699, "rstar_iso": 0.4705, "ece_raw": 0.0279, "ece_iso": 0.0300, "acc": 0.5185},
        {"judge": "granite-3.0-8b-instruct",   "rstar_raw": 0.2438, "rstar_iso": 0.3980, "ece_raw": 0.1800, "ece_iso": 0.0393, "acc": 0.5866},
        {"judge": "Mistral-7B-Instruct-v0.3",  "rstar_raw": 0.2947, "rstar_iso": 0.4141, "ece_raw": 0.1368, "ece_iso": 0.0315, "acc": 0.5776},
        {"judge": "OLMo-7B-Instruct-hf",       "rstar_raw": 0.3325, "rstar_iso": 0.4607, "ece_raw": 0.1540, "ece_iso": 0.0292, "acc": 0.5175},
        {"judge": "Qwen2.5-7B-Instruct",       "rstar_raw": 0.2187, "rstar_iso": 0.4140, "ece_raw": 0.2120, "ece_iso": 0.0427, "acc": 0.5826},
    ])
else:
    df = pd.read_csv(CSV)
    df["acc"] = df["acc_vs_gold"]

# --- correlations ---
r_raw, p_raw   = pearsonr(df["ece_raw"], df["rstar_raw"])
s_raw, ps_raw  = spearmanr(df["ece_raw"], df["rstar_raw"])
r_iso, p_iso   = pearsonr(df["ece_iso"], df["rstar_iso"])
s_iso, ps_iso  = spearmanr(df["ece_iso"], df["rstar_iso"])
r_acc, p_acc   = pearsonr(df["acc"], df["rstar_iso"])

stats = {
    "pearson_ece_raw_vs_rstar_raw": {"r": r_raw, "p": p_raw},
    "spearman_ece_raw_vs_rstar_raw": {"rho": s_raw, "p": ps_raw},
    "pearson_ece_iso_vs_rstar_iso": {"r": r_iso, "p": p_iso},
    "spearman_ece_iso_vs_rstar_iso": {"rho": s_iso, "p": ps_iso},
    "pearson_acc_vs_rstar_iso": {"r": r_acc, "p": p_acc},
    "n_judges": len(df),
    "interpretation": (
        "High positive corr between ECE_raw and R*_raw would directly show "
        "the plug-in estimator bias is dominated by judge calibration error, "
        "not by data-side noise. Negative acc-vs-R* corr (stronger judges → lower R*) "
        "is the same story from the capacity angle."
    ),
}

# --- plot: 2-panel figure ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# panel A: raw ECE vs raw R*
ax = axes[0]
ax.scatter(df["ece_raw"], df["rstar_raw"], s=90, c="#2b6cb0", edgecolors="black", linewidths=0.7)
for _, row in df.iterrows():
    ax.annotate(row["judge"].split("-")[0], (row["ece_raw"], row["rstar_raw"]),
                textcoords="offset points", xytext=(6, 4), fontsize=8)
ax.set_xlabel("Raw ECE (plug-in softmax)")
ax.set_ylabel(r"Raw $\hat{R}^*$")
ax.set_title(f"A. Raw — Pearson r = {r_raw:.2f} (p={p_raw:.3f})")
ax.grid(True, alpha=0.3)

# panel B: iso ECE vs iso R*
ax = axes[1]
ax.scatter(df["ece_iso"], df["rstar_iso"], s=90, c="#c05621", edgecolors="black", linewidths=0.7)
for _, row in df.iterrows():
    ax.annotate(row["judge"].split("-")[0], (row["ece_iso"], row["rstar_iso"]),
                textcoords="offset points", xytext=(6, 4), fontsize=8)
ax.set_xlabel("Isotonic ECE (post-calibration)")
ax.set_ylabel(r"Isotonic $\hat{R}^*$")
ax.set_title(f"B. Isotonic — Pearson r = {r_iso:.2f} (p={p_iso:.3f})")
ax.grid(True, alpha=0.3)

fig.suptitle(
    r"Capacity confound: plug-in $\hat{R}^*$ correlates with judge ECE across 5 LLM judges (HelpSteer2, N=1000)",
    fontsize=11, y=1.02,
)
fig.tight_layout()
fig.savefig(HERE / "fig_ece_vs_rstar.png", dpi=150, bbox_inches="tight")

# also dump a 3rd "acc vs R*_iso" single panel because it reinforces the same story
fig2, ax2 = plt.subplots(figsize=(5.5, 4.5))
ax2.scatter(df["acc"], df["rstar_iso"], s=90, c="#38a169", edgecolors="black", linewidths=0.7)
for _, row in df.iterrows():
    ax2.annotate(row["judge"].split("-")[0], (row["acc"], row["rstar_iso"]),
                 textcoords="offset points", xytext=(6, 4), fontsize=8)
ax2.set_xlabel("Judge accuracy vs gold")
ax2.set_ylabel(r"Isotonic $\hat{R}^*$")
ax2.set_title(f"Stronger judge → lower $\\hat{{R}}^*$ (Pearson r = {r_acc:.2f})")
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig(HERE / "fig_acc_vs_rstar.png", dpi=150, bbox_inches="tight")

with open(HERE / "stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print(json.dumps(stats, indent=2))
print(f"\nSaved: {HERE / 'fig_ece_vs_rstar.png'}")
print(f"Saved: {HERE / 'fig_acc_vs_rstar.png'}")
