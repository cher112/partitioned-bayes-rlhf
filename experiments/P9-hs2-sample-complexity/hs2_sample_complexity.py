"""P9 — Real-data sample complexity on HS2 with re-fit isotonic (M3).

What we are NOT measuring (an earlier version did this and it was wrong):
    std of R̂*_iso on a *fixed* calibrator fit to full data, across
    bootstrap resamples. That is just verifying the central-limit
    theorem for a sample mean — the slope is trivially -0.5 by CLT
    and has nothing to do with the estimator's statistical property.

What this script ACTUALLY measures:
    For each subsample size N, fit the isotonic calibrator *from
    scratch* on that subsample, compute R̂*_iso on the same subsample.
    This captures the joint sample complexity of (calibrator fit +
    plug-in estimator) that Nguyen 2005 / Ushio 2026 bound.

    Sampling is *without replacement* (subsampling, not bootstrap) to
    avoid the ties that degrade isotonic regression on bootstrap
    resamples.

The rate is a genuine finding: if slope != -0.5 it means isotonic
contributes non-trivially to the effective sample complexity, which
itself motivates margin-matching R̂*_CA (Year-1).
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
# Subsample sizes (without replacement) — capped at 800 since full pool
# per judge is ~1000 and we want sampled subsets to be meaningfully
# independent from one another (coupon-collector argument: at N=900
# of 1000 the overlap between any two subsamples is 81% on average).
NS = [100, 150, 200, 300, 500, 700, 800]

# ---------------- per-judge data ----------------
judge_data = {}  # name -> (p_a_mean raw, gold)
for fp in JUDGE_FILES:
    with open(fp) as f:
        d = json.load(f)
    pairs = [p for p in d["pairs"] if p.get("p_a_mean") is not None]
    p_a = np.array([p["p_a_mean"] for p in pairs])
    gold = np.array([p["gold"] for p in pairs], dtype=int)
    judge_data[d["model"]] = (p_a, gold)
    # Reference = refit isotonic on full, plug-in on the same set
    ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    ir.fit(p_a, gold)
    r_full = estimate_rstar(ir.predict(p_a))
    print(f"{d['model']:<30s} n={len(p_a)}  R*_iso(full) = {r_full:.4f}")

# ---------------- subsample loop — REFIT isotonic each time ----------------
rows = []
rng = np.random.default_rng(42)
for name, (pa, g) in judge_data.items():
    n_total = len(pa)
    for N in NS:
        if N >= n_total:
            continue  # subsampling without replacement requires N < pool
        for seed in range(N_SEEDS):
            idx = rng.choice(n_total, size=N, replace=False)
            sub_pa, sub_g = pa[idx], g[idx]
            # Re-fit isotonic on this subsample from scratch
            ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
            ir.fit(sub_pa, sub_g)
            cal = ir.predict(sub_pa)
            r_hat = estimate_rstar(cal)
            rows.append({
                "judge": name,
                "N": N,
                "seed": seed,
                "rstar_iso": r_hat,
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
        label=r"CLT reference: slope $=-0.500$")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Subsample size $N$")
ax.set_ylabel(r"bootstrap $\mathrm{std}(\hat{R}^*_{\mathrm{iso}})$ across 100 replicates")
ax.set_title(
    fr"P9 — HS2 sample complexity with per-subsample isotonic refit: "
    fr"empirical slope ${slope:+.3f}$  ($R^2={r2:.3f}$, 100 seeds, 6 judges)"
)
ax.legend(loc="best", handlelength=1.8)

out = HERE / "fig_sample_complexity.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
