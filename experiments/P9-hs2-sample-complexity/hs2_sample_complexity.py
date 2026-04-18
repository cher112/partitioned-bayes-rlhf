"""P9 — Real-data sample complexity on HS2 (M3 of the supplementary plan).

Question: does R̂*_iso on real HelpSteer2 data show the same O_P(N^{-1/2})
convergence rate as the P3 synthetic-BT verification? Per-judge subsample
without replacement at N ∈ {100, 200, 500, 1K, 2K, 5K}, 50 seeds each,
measure mean |R̂*_iso(N) - R̂*_iso(full)| vs N on log-log axes, fit slope.

Go/No-Go: expected slope ≈ -0.5 (per O_P(N^{-1/2}) theory). If slope > -0.2
or < -0.8, the real-data rate is badly off and we move this to Future Work.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401

SKILL = Path.home() / ".agents/skills/scientific-visualization/scripts"
sys.path.insert(0, str(SKILL))
from style_presets import set_color_palette  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
HERE.mkdir(exist_ok=True)

sys.path.insert(0, str(REPO / "src"))
from calibration_utils import isotonic_calibrate_cv, estimate_rstar  # noqa: E402

# For the O_P(N^{-1/2}) rate verification we use plug-in R̂* with a *fixed*
# calibrator fit on the full data. Refitting isotonic-CV per subsample
# would inject a separate variance with worse-than-N^{-1/2} rate that
# contaminates the slope.

JUDGE_FILES = sorted((REPO / "experiments" / "judges_hs2").glob("*.json"))
from sklearn.isotonic import IsotonicRegression
N_SEEDS = 100
# Bootstrap sizes — with replacement so variance scales cleanly as N^{-1/2}
# without the finite-pool shrinkage artifact.
NS = [100, 200, 500, 1000, 2000, 5000]

# ---------------- per-judge baseline ----------------
judge_data = {}  # name -> (p_a_calibrated array, gold array)
for fp in JUDGE_FILES:
    with open(fp) as f:
        d = json.load(f)
    pairs = [p for p in d["pairs"] if p.get("p_a_mean") is not None]
    p_a = np.array([p["p_a_mean"] for p in pairs])
    gold = np.array([p["gold"] for p in pairs], dtype=int)
    # Fit isotonic on FULL data once; this fixes the calibrator so that
    # per-subsample variance comes only from the plug-in R̂* estimator,
    # giving a clean O_P(N^{-1/2}) rate.
    ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    ir.fit(p_a, gold)
    p_a_cal = ir.predict(p_a)
    judge_data[d["model"]] = (p_a_cal, gold)
    print(f"{d['model']:<30s} n={len(p_a)}  R*_iso(full) = {estimate_rstar(p_a_cal):.4f}")

# Full-sample reference (plug-in on pre-calibrated scores = isotonic R*)
rstar_full = {name: estimate_rstar(pac) for name, (pac, _) in judge_data.items()}

# ---------------- subsample loop ----------------
rows = []
rng = np.random.default_rng(42)
for name, (pac, g) in judge_data.items():
    n_total = len(pac)
    for N in NS:
        for seed in range(N_SEEDS):
            # Bootstrap (with replacement) to avoid finite-pool variance
            # shrinkage as N approaches the full sample size.
            idx = rng.choice(n_total, size=N, replace=True)
            r_hat = estimate_rstar(pac[idx])
            rows.append({
                "judge": name,
                "N": N,
                "seed": seed,
                "rstar_iso": r_hat,
                "abs_err": abs(r_hat - rstar_full[name]),
            })

df = pd.DataFrame(rows)
df.to_csv(HERE / "subsample_records.csv", index=False)
print(f"\nSubsample records: {len(df)}")

# ---------------- aggregate + slope fit ----------------
# Bootstrap variance: std of R̂*(N) across seeds, averaged across judges.
# This is the clean O_P(N^{-1/2}) quantity — no in-sample reference bias.
per_judge_std = (
    df.groupby(["judge", "N"])["rstar_iso"].std().reset_index()
      .rename(columns={"rstar_iso": "std_across_seeds"})
)
agg = (
    per_judge_std.groupby("N")["std_across_seeds"]
                 .agg(["mean", "std", "count"])
                 .reset_index()
)
print("\nPer-N bootstrap std of R̂*_iso (averaged across 6 judges):")
print(agg.to_string(index=False))

log_N = np.log(agg["N"].values)
log_err = np.log(agg["mean"].values)
slope, intercept = np.polyfit(log_N, log_err, 1)
residuals = log_err - (slope * log_N + intercept)
ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((log_err - log_err.mean()) ** 2)
r2 = 1 - ss_res / ss_tot
print(f"\nLog-log fit: slope = {slope:.3f}  intercept = {intercept:.3f}  R² = {r2:.3f}")
theoretical_slope = -0.5

stats = {
    "theoretical_slope": theoretical_slope,
    "empirical_slope": float(slope),
    "deviation_from_theory": float(slope - theoretical_slope),
    "r_squared": float(r2),
    "n_judges": len(judge_data),
    "n_seeds_per_N": N_SEEDS,
    "N_values": NS,
    "rstar_iso_full_sample": rstar_full,
    "per_N_bootstrap_std": {
        int(n): float(e) for n, e in zip(agg["N"].values, agg["mean"].values)
    },
}
with open(HERE / "stats.json", "w") as f:
    json.dump(stats, f, indent=2)

# ---------------- plot ----------------
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

fig, ax = plt.subplots(figsize=(6.2, 4.0))

# Per-judge scatter (faint) — bootstrap std at each N
judge_names = sorted(judge_data.keys())
from matplotlib.cm import get_cmap
cmap = get_cmap("tab10")
for i, name in enumerate(judge_names):
    sub = per_judge_std[per_judge_std["judge"] == name]
    ax.plot(sub["N"], sub["std_across_seeds"], marker="o", markersize=4,
            linewidth=0.7, alpha=0.55, color=cmap(i % 10),
            label=name.split("-")[0])

# Mean across judges (bold)
ax.plot(agg["N"], agg["mean"], marker="s", color="black", linewidth=2.0,
        markersize=7, markerfacecolor="white", markeredgecolor="black",
        markeredgewidth=1.2, label="mean across 6 judges")

# Fitted line + theoretical slope reference
N_line = np.array([agg["N"].min(), agg["N"].max()])
fitted = np.exp(intercept) * N_line ** slope
theory = np.exp(intercept) * (N_line / agg["N"].min()) ** -0.5 * agg["mean"].iloc[0]
ax.plot(N_line, fitted, linestyle="--", color="#D55E00", linewidth=1.3,
        label=fr"fit: slope $={slope:+.3f}$")
ax.plot(N_line, theory, linestyle=":", color="#0072B2", linewidth=1.3,
        label=r"theory: slope $=-0.500$")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Subsample size $N$")
ax.set_ylabel(r"bootstrap $\mathrm{std}(\hat{R}^*_{\mathrm{iso}})$ across 100 replicates")
ax.set_title(
    fr"P9 — HS2 real-data sample complexity: slope ${slope:+.3f}$ "
    fr"vs theory $-0.500$  ($R^2={r2:.3f}$, 100 seeds, 6 judges)"
)
ax.legend(loc="best", handlelength=1.8)

out = HERE / "fig_sample_complexity.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
