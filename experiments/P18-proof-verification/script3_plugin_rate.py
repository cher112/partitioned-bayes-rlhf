"""P18 Script 3 — Thm 3 verification: plug-in rate O_P(N^{-1/2}).

With TRUE anchors (ρ*_0, ρ*_1) given (so Lemma 1 error is eliminated),
the remaining error is the classical Hoeffding rate for a sample mean
of bounded [0, 1/2]-valued random variables:

    std_over_seeds( R̂*_CA(ρ*) )  =  O(N^{-1/2})

Go/No-Go:
  PASS if log-log slope ∈ [-0.6, -0.4]
  NO-GO if slope > -0.3 or slope < -0.7
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
from synthetic_generator import gen_bt_data, rstar_ca_pooled, analytic_rho  # noqa: E402


RHO_0 = 0.15
RHO_1 = 0.10
ALPHA = 0.01
K = 6
NS = [100, 200, 500, 1000, 2000, 5000, 10000]
N_SEEDS = 100

rho_0_star, rho_1_star = analytic_rho(
    ALPHA, {"mode": "affine", "rho_0": RHO_0, "rho_1": RHO_1})

ns_arr, std_arr = [], []
per_n = []
for N in NS:
    rs = []
    for seed in range(N_SEEDS):
        _, c, _ = gen_bt_data(n=N, k=K, seed=seed, mode="affine",
                              rho_0=RHO_0, rho_1=RHO_1, noise_sigma=0.02)
        r, _ = rstar_ca_pooled(c, rho_override=(rho_0_star, rho_1_star))
        rs.append(r)
    s = float(np.std(rs))
    ns_arr.append(N); std_arr.append(s)
    per_n.append({"N": N, "mean_rstar_ca": float(np.mean(rs)),
                  "std_rstar_ca": s, "n_seeds": N_SEEDS})
    print(f"  N = {N:>6}   std(R̂*_CA across seeds) = {s:.5f}   "
          f"N^(-1/2) = {N**(-0.5):.5f}")

slope, intercept = np.polyfit(np.log(ns_arr), np.log(std_arr), 1)
residuals = np.log(std_arr) - (slope * np.log(ns_arr) + intercept)
ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((np.log(std_arr) - np.mean(np.log(std_arr))) ** 2)
r2 = 1 - ss_res / ss_tot

pass_ = -0.6 <= slope <= -0.4
print(f"\nlog-log slope = {slope:+.4f}  (target -0.500, R² = {r2:.3f})  "
      f"PASS: {pass_}")

stats = {
    "rho_star": [float(rho_0_star), float(rho_1_star)],
    "per_N": per_n,
    "log_log_slope": float(slope),
    "r_squared": float(r2),
    "theoretical_slope": -0.5,
    "pass_window": [-0.6, -0.4],
    "pass": bool(pass_),
}
with open(HERE / "script3_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

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
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.plot(ns_arr, std_arr, marker="o", color="#0072B2", linewidth=1.6,
        markersize=6, markerfacecolor="white", markeredgewidth=1.1,
        label=r"empirical $\mathrm{std}(\hat{R}^*_{CA})$")
N_line = np.array([min(NS), max(NS)])
fit_line = np.exp(intercept) * N_line ** slope
ax.plot(N_line, fit_line, linestyle="--", color="#D55E00", linewidth=1.2,
        label=fr"fit: slope {slope:+.3f}  ($R^2 = {r2:.3f}$)")
theory_line = std_arr[0] * (N_line / NS[0]) ** -0.5
ax.plot(N_line, theory_line, linestyle=":", color="#888", linewidth=1.0,
        label=r"theory $N^{-1/2}$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Sample size $N$")
ax.set_ylabel(r"$\mathrm{std}(\hat{R}^*_{CA}(\rho^*))$ across 100 seeds")
ax.set_title(
    fr"P18 Script 3 — Thm 3: plug-in rate with true $\rho^*$  "
    f"[{'PASS' if pass_ else 'FAIL'}]"
)
ax.legend(loc="best", handlelength=1.8)
fig.savefig(HERE / "fig_script3_plugin_rate.png", dpi=300,
            bbox_inches="tight", facecolor="white")
print(f"Saved: {HERE / 'fig_script3_plugin_rate.png'}")
