"""P8 — Baseline comparison: 1-accuracy vs plug-in vs isotonic vs Margin-CA.

Question a reviewer will ask: "Why R̂*_iso? Can't we just use 1 - accuracy
as a cheap proxy for Bayes error?"

Answer we want to defend: 1 - accuracy is a property of the *classifier*
(judge), not of the data. We show this by computing four competing Bayes-
error proxies per judge on both HS2 and UF, and report:

  (a) cross-judge range of each proxy — smaller range = less capacity
      contamination, closer to a data-side quantity.
  (b) Spearman rank correlation with acc — anything near 1 is redundant
      with acc and cannot add information beyond it.

Outputs: per-dataset CSV + a bar-chart figure + stats.json.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
from scipy.stats import spearmanr

SKILL = Path.home() / ".agents/skills/scientific-visualization/scripts"
sys.path.insert(0, str(SKILL))
from style_presets import set_color_palette  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
HERE.mkdir(exist_ok=True)

DATASETS = {
    "HS2": REPO / "experiments" / "analysis_hs2_6judges" / "per_judge_rstar.csv",
    "UF":  REPO / "experiments" / "analysis_uf_6judges" / "per_judge_rstar.csv",
}

# ---------------- per-dataset table ----------------
summary = {}
frames = {}
for tag, path in DATASETS.items():
    df = pd.read_csv(path)
    df = df.sort_values("acc_vs_gold").reset_index(drop=True)
    df["1-acc"] = 1.0 - df["acc_vs_gold"]
    # Simple cross-judge ranges and Spearman vs accuracy
    metrics = {}
    for col, nice in [("1-acc", "1 - accuracy"),
                      ("rstar_raw", "plug-in R̂*"),
                      ("rstar_iso", "isotonic R̂*")]:
        rho, p = spearmanr(df["acc_vs_gold"], df[col])
        metrics[nice] = {
            "range": float(df[col].max() - df[col].min()),
            "spearman_r_with_acc": float(rho),
            "spearman_p": float(p),
        }
    summary[tag] = metrics
    frames[tag] = df

# Dump stats
with open(HERE / "stats.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))

# Dump combined per-judge CSV for easy inspection
for tag, df in frames.items():
    df_out = df[["judge", "acc_vs_gold", "1-acc", "rstar_raw", "rstar_iso"]].copy()
    df_out.to_csv(HERE / f"per_judge_{tag}.csv", index=False)

# ---------------- figure ----------------
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
C_ACC = "#999999"    # gray — the trivial baseline
C_RAW = "#D55E00"    # vermillion — plug-in
C_ISO = "#0072B2"    # blue — isotonic (our primary)
colors = [C_ACC, C_RAW, C_ISO]
labels = ["1 - accuracy", r"plug-in $\hat{R}^*$", r"isotonic $\hat{R}^*$"]

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, (tag, df) in zip(axes, frames.items()):
    judges = [j.split("-")[0] if "-" in j else j for j in df["judge"]]
    x = np.arange(len(judges))
    w = 0.27
    ax.bar(x - w, df["1-acc"], w, color=C_ACC, edgecolor="black",
           linewidth=0.5, label=labels[0])
    ax.bar(x, df["rstar_raw"], w, color=C_RAW, edgecolor="black",
           linewidth=0.5, label=labels[1])
    ax.bar(x + w, df["rstar_iso"], w, color=C_ISO, edgecolor="black",
           linewidth=0.5, label=labels[2])
    ax.set_xticks(x)
    ax.set_xticklabels(judges, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Estimator value")
    rng1 = summary[tag]["1 - accuracy"]["range"]
    rng2 = summary[tag]["plug-in R̂*"]["range"]
    rng3 = summary[tag]["isotonic R̂*"]["range"]
    ax.set_title(f"{tag} — cross-judge ranges: "
                 f"1-acc {rng1:.2f}, plug-in {rng2:.2f}, iso {rng3:.2f}")
    if tag == list(frames.keys())[0]:
        ax.legend(loc="best", handlelength=1.4)

fig.suptitle(
    r"P8 — Three Bayes-error proxies across 6 judges: none converges to a single "
    r"capacity-invariant value. Motivates $\hat{R}^*_{CA}$ (Year-1).",
    fontsize=10.5, y=1.03,
)
out = HERE / "fig_baselines.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
