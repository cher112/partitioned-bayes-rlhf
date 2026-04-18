"""P10 — Cross-dataset judge-ranking consistency (revised M4).

Gemini Gate-1 critique of the original M4 OOD-transfer design rejected
it as NO-GO:
  (a) HS2 gold = human majority, UF gold = GPT-4 score gap.
      Concept shift confounds data-distribution shift — if Spearman
      were low we could not distinguish "calibrator failed to
      transfer" from "gold labels disagree."
  (b) N=6 Spearman is statistically weak at typical effect sizes.

Revised claim (P10): we no longer claim calibrator zero-shot transfer.
Instead we test a weaker, cleaner question:

    "Are judge-level R̂*_iso estimates *rank-consistent* across two
     datasets with different gold definitions?"

If yes: judges' uncertainty signal tracks a judge-property not fully
reducible to one dataset's gold. That is good empirical hygiene
(cross-dataset stability) without overclaiming covariate-shift
robustness.

If no: dataset-specific gold dominates the measured R̂* — this is itself
reportable and motivates Year-1 work to design a gold-invariant estimator.

Outputs: Spearman + Pearson between HS2-refit R̂*_iso and UF-refit
R̂*_iso, plus same for accuracy. Also plots the 6 judges on a HS2 × UF
scatter.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
from scipy.stats import pearsonr, spearmanr

SKILL = Path.home() / ".agents/skills/scientific-visualization/scripts"
sys.path.insert(0, str(SKILL))
from style_presets import set_color_palette  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
HERE.mkdir(exist_ok=True)

sys.path.insert(0, str(REPO / "src"))
from bootstrap_ci import fisher_z_ci_correlation  # noqa: E402

hs2 = pd.read_csv(REPO / "experiments" / "analysis_hs2_6judges" / "per_judge_rstar.csv")
uf  = pd.read_csv(REPO / "experiments" / "analysis_uf_6judges" / "per_judge_rstar.csv")

# Align by judge name
merged = pd.merge(
    hs2[["judge", "acc_vs_gold", "rstar_iso"]].rename(
        columns={"acc_vs_gold": "acc_hs2", "rstar_iso": "rstar_hs2"}),
    uf[["judge",  "acc_vs_gold", "rstar_iso"]].rename(
        columns={"acc_vs_gold": "acc_uf",  "rstar_iso": "rstar_uf"}),
    on="judge",
)
print(merged.to_string(index=False))

results = {}
for name, x, y in [("rstar_iso", merged["rstar_hs2"], merged["rstar_uf"]),
                   ("accuracy",  merged["acc_hs2"],   merged["acc_uf"])]:
    rho, p_s = spearmanr(x, y)
    r, p_p  = pearsonr(x, y)
    flo, fhi = fisher_z_ci_correlation(float(r), len(x))
    results[name] = {
        "spearman": {"rho": float(rho), "p": float(p_s)},
        "pearson":  {"r": float(r), "p": float(p_p),
                     "fisher_z_ci_95": [flo, fhi]},
    }
    print(f"{name:10s}: Spearman rho = {rho:+.3f} (p={p_s:.3f})  "
          f"Pearson r = {r:+.3f} (p={p_p:.3f}) Fisher-z CI [{flo:+.3f},{fhi:+.3f}]")

results["n_judges"] = len(merged)
results["caveat"] = (
    "HS2 gold = human-majority helpfulness; UF gold = overall_score gap >= 2 "
    "(GPT-4 scores). This tests ranking CONSISTENCY under both covariate and "
    "concept shift — not a clean OOD-calibrator transfer. N=6 small-sample "
    "Spearman; CI reported via Fisher-z for Pearson."
)
with open(HERE / "stats.json", "w") as f:
    json.dump(results, f, indent=2)

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
C = "#0072B2"
fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))
for ax, (a, b, xcol, ycol, xlab, ylab, title_key) in zip(axes, [
    (axes[0], axes[0], "rstar_hs2", "rstar_uf",
     r"$\hat{R}^*_{\mathrm{iso}}$ on HS2",
     r"$\hat{R}^*_{\mathrm{iso}}$ on UF", "rstar_iso"),
    (axes[1], axes[1], "acc_hs2", "acc_uf",
     "accuracy on HS2", "accuracy on UF", "accuracy"),
]):
    ax.scatter(merged[xcol], merged[ycol], s=80, c=C,
               edgecolors="black", linewidths=0.6)
    for _, row in merged.iterrows():
        short = row["judge"].split("-")[0]
        ax.annotate(short, (row[xcol], row[ycol]),
                    textcoords="offset points", xytext=(5, 4), fontsize=8)
    # Identity line
    lo = min(merged[xcol].min(), merged[ycol].min()) - 0.02
    hi = max(merged[xcol].max(), merged[ycol].max()) + 0.02
    ax.plot([lo, hi], [lo, hi], linestyle=":", color="#888", linewidth=0.8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    rho = results[title_key]["spearman"]["rho"]
    r = results[title_key]["pearson"]["r"]
    flo, fhi = results[title_key]["pearson"]["fisher_z_ci_95"]
    ax.set_title(
        f"{title_key}: Spearman ρ = {rho:+.2f},  "
        f"Pearson r = {r:+.2f} [Fisher-z CI {flo:+.2f}, {fhi:+.2f}]")
fig.suptitle(
    r"P10 — Cross-dataset (HS2 ↔ UF) judge-ranking consistency  "
    r"(N=6, small-sample caveat applies)", fontsize=10.5, y=1.03)
out = HERE / "fig_cross_dataset_rank.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
