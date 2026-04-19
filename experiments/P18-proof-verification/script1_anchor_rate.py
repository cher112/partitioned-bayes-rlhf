"""P18 Script 1 — Lemma 1 verification: anchor rate O_P(N^{-1/2}).

Under A2 tightened to affine capacity truncation
  c_iso_k(i) = (1 - rho*_0 - rho*_1) * eta_i + rho*_0 + noise_k(i)
the pooled empirical quantile ρ̂_α = quantile(concat_k c_iso_k, α) is
a sample quantile of a smooth-density distribution at an interior
point. By classical sample-quantile CLT:

    sqrt(N) * (ρ̂_α - ρ*_α)  -->  N(0, α(1-α) / f(ρ*_α)^2)

so  |ρ̂_α - ρ*_α| = O_P(N^{-1/2}).

(The K > 1 dimension only tightens the variance constant via noise
averaging; it does NOT change the N^{-1/2} rate, because the N data
points are the independent units of empirical process convergence.)

Pilot runs (2026-04-19):
  noise=0:     slope = -0.582  (N=100..20000, 100 seeds)
  noise=0.01:  slope = -0.4996 (near-exact match to theory)

Go/No-Go for the theorem:
  PASS  if log-log slope in [-0.60, -0.40]  (i.e. within 0.1 of -1/2)
  NO-GO if slope > -0.3 or slope < -0.7   (indicates rate mismatch)
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

SKILL = Path.home() / ".agents/skills/scientific-visualization/scripts"
sys.path.insert(0, str(SKILL))
from style_presets import set_color_palette  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from synthetic_generator import gen_bt_data, analytic_rho  # noqa: E402


RHO_0 = 0.15
RHO_1 = 0.10
ALPHA = 0.01
K = 6
NS = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]
N_SEEDS = 100

# Analytic truth (noise-free affine)
rho_0_star, _ = analytic_rho(ALPHA, {"mode": "affine", "rho_0": RHO_0, "rho_1": RHO_1})
_, rho_1_analytic = analytic_rho(ALPHA, {"mode": "affine", "rho_0": RHO_0, "rho_1": RHO_1})
rho_1_star = rho_1_analytic    # = 1 - (1-alpha)-quantile of pool
print(f"Analytic rho*_0.01 = {rho_0_star:.4f}  1-quantile(0.99) = {rho_1_star:.4f}")

records = []
for noise_sigma in [0.0, 0.01, 0.03]:
    ns, ys = [], []
    for N in NS:
        errs = []
        for seed in range(N_SEEDS):
            _, c, _ = gen_bt_data(n=N, k=K, seed=seed, mode="affine",
                                  rho_0=RHO_0, rho_1=RHO_1,
                                  noise_sigma=noise_sigma)
            rho0_hat = float(np.quantile(c.flatten(), ALPHA))
            errs.append(abs(rho0_hat - rho_0_star))
        m = float(np.mean(errs))
        ns.append(N); ys.append(m)
        records.append({"noise_sigma": noise_sigma, "N": N,
                        "mean_abs_err": m, "n_seeds": N_SEEDS})
    slope, intercept = np.polyfit(np.log(ns), np.log(ys), 1)
    records.append({"noise_sigma": noise_sigma, "log_log_slope": float(slope),
                    "theoretical": -0.5, "pass_window": [-0.6, -0.4]})
    print(f"noise={noise_sigma:.3f}  log-log slope = {slope:+.4f}  (target -0.500)")

with open(HERE / "script1_stats.json", "w") as f:
    json.dump(records, f, indent=2)

# ---------- figure ----------
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
fig, ax = plt.subplots(figsize=(6.4, 4.2))
colors = ["#D55E00", "#0072B2", "#009E73"]
for i, noise_sigma in enumerate([0.0, 0.01, 0.03]):
    subset = [r for r in records if r.get("noise_sigma") == noise_sigma and "N" in r]
    ns = [r["N"] for r in subset]
    ys = [r["mean_abs_err"] for r in subset]
    slope_rec = next(r for r in records
                     if r.get("noise_sigma") == noise_sigma and "log_log_slope" in r)
    ax.plot(ns, ys, marker="o", color=colors[i], linewidth=1.5,
            markersize=5, markerfacecolor="white", markeredgewidth=1.0,
            label=fr"$\sigma = {noise_sigma:.2f}$  (slope {slope_rec['log_log_slope']:+.3f})")
# Reference -1/2 line
N_line = np.array([min(NS), max(NS)])
y0 = records[0]["mean_abs_err"]   # anchor at first N with noise=0
ax.plot(N_line, y0 * (N_line / NS[0]) ** -0.5, linestyle="--",
        color="#888", linewidth=1.0, label=r"theory $N^{-1/2}$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Sample size $N$ (K = 6 judges pooled)")
ax.set_ylabel(r"$|\hat{\rho}_{\alpha} - \rho^*_{\alpha}|$  (mean across 100 seeds)")
ax.set_title(
    r"P18 Script 1 — Lemma 1$'$: pooled-quantile anchor rate verification"
)
ax.legend(loc="best", handlelength=1.8)
fig.savefig(HERE / "fig_script1_anchor_rate.png", dpi=300,
            bbox_inches="tight", facecolor="white")
print(f"Saved: {HERE / 'fig_script1_anchor_rate.png'}")
