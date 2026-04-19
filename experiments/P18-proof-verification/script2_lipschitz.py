"""P18 Script 2 — Lemma 2 verification: Lipschitz bridge.

Under margin condition (A5: ρ_0 + ρ_1 ≤ 1 − eps, eps > 0), the
estimator R̂*_CA is Lipschitz in the anchor pair (ρ_0, ρ_1):

    |R̂*_CA(ρ̂_0, ρ̂_1) - R̂*_CA(ρ*_0, ρ*_1)|  ≤  L · ||(ρ̂ - ρ*)||_∞

with L = 1 / (1 − ρ*_0 − ρ*_1).

Proof sketch (two lines): min(p, 1-p) is Lipschitz-1 in p.
p_bar = mean_k p_corr_k is Lipschitz-1 in each p_corr_k.
p_corr_k = (c_iso_k - ρ_0) / (1 - ρ_0 - ρ_1) has partial derivative in
(ρ_0, ρ_1) bounded by 1 / (1 - ρ_0 - ρ_1)^2 in the worst case; the
projection through mean_k and then min averaging gives the stated L.

Empirical test: fix true (ρ*_0, ρ*_1), perturb anchors by Δρ ∈
{±0.01, ±0.02, ±0.05, ±0.1}, measure |R̂*_CA(ρ* + Δρ) - R̂*_CA(ρ*)| / |Δρ|.

Go/No-Go:
  PASS if empirical Lipschitz ≤ 1.5 × theoretical L
  NO-GO if empirical > 2 × L  (indicates un-bounded sensitivity)
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
N = 5000
N_SEEDS = 50
DELTAS = [-0.10, -0.05, -0.02, -0.01, 0.01, 0.02, 0.05, 0.10]

rho_0_star, rho_1_star = analytic_rho(
    ALPHA, {"mode": "affine", "rho_0": RHO_0, "rho_1": RHO_1})
L_theory = 1.0 / (1.0 - rho_0_star - rho_1_star)

print(f"Analytic anchors (ρ*_0, ρ*_1) = ({rho_0_star:.4f}, {rho_1_star:.4f})")
print(f"Theoretical Lipschitz bound L = 1 / (1 - ρ*_0 - ρ*_1) = {L_theory:.4f}")

rows = []
for delta in DELTAS:
    sensitivities = []
    for seed in range(N_SEEDS):
        _, c, _ = gen_bt_data(n=N, k=K, seed=seed, mode="affine",
                              rho_0=RHO_0, rho_1=RHO_1, noise_sigma=0.02)
        # R̂*_CA at true anchors
        r_star_anchor, _ = rstar_ca_pooled(
            c, rho_override=(rho_0_star, rho_1_star))
        # R̂*_CA at perturbed anchors (perturb both sides symmetrically)
        r_perturbed, _ = rstar_ca_pooled(
            c, rho_override=(rho_0_star + delta, rho_1_star + delta))
        sensitivity = abs(r_perturbed - r_star_anchor) / abs(delta)
        sensitivities.append(sensitivity)
    sens_mean = float(np.mean(sensitivities))
    sens_std = float(np.std(sensitivities))
    rows.append({"delta_rho": delta,
                 "empirical_lipschitz_mean": sens_mean,
                 "empirical_lipschitz_std": sens_std})
    print(f"  Δρ = {delta:+.3f}   mean |ΔR̂*_CA|/|Δρ| = {sens_mean:.3f}"
          f" ± {sens_std:.3f}   (L_theory = {L_theory:.3f})")

# PASS criterion
max_emp = max(r["empirical_lipschitz_mean"] for r in rows)
ratio = max_emp / L_theory
pass_ = bool(ratio <= 1.5)
print(f"\nmax empirical Lipschitz / L_theory = {ratio:.3f}  "
      f"PASS <=1.5: {pass_}")

stats = {
    "L_theoretical": float(L_theory),
    "rho_star": [float(rho_0_star), float(rho_1_star)],
    "per_delta": rows,
    "max_empirical_lipschitz": max_emp,
    "ratio_to_theory": float(ratio),
    "pass_criterion": "max empirical L / L_theory <= 1.5",
    "pass": pass_,
}
with open(HERE / "script2_stats.json", "w") as f:
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
deltas = [r["delta_rho"] for r in rows]
means = [r["empirical_lipschitz_mean"] for r in rows]
stds = [r["empirical_lipschitz_std"] for r in rows]
ax.errorbar(deltas, means, yerr=stds, fmt="o", color="#0072B2",
            markersize=6, capsize=3, markerfacecolor="white",
            markeredgewidth=1.1,
            label=r"empirical $|\Delta \hat{R}^*_{CA}| / |\Delta\rho|$")
ax.axhline(L_theory, color="#D55E00", linestyle="--", linewidth=1.2,
           label=fr"theory $L = 1 / (1 - \rho^*_0 - \rho^*_1) = {L_theory:.2f}$")
ax.axhline(L_theory * 1.5, color="#888", linestyle=":", linewidth=0.9,
           label=r"PASS window $1.5 \times L$")
ax.set_xlabel(r"Perturbation $\Delta\rho$")
ax.set_ylabel(r"Empirical Lipschitz ratio")
ax.set_title(
    fr"P18 Script 2 — Lemma 2: Lipschitz bridge in $\rho$  "
    f"[{'PASS' if pass_ else 'FAIL'}]"
)
ax.legend(loc="best", handlelength=1.8)
fig.savefig(HERE / "fig_script2_lipschitz.png", dpi=300,
            bbox_inches="tight", facecolor="white")
print(f"Saved: {HERE / 'fig_script2_lipschitz.png'}")
