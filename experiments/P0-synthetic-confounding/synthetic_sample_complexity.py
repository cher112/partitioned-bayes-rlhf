"""P0 V5 — Synthetic sample-complexity on the 2-Gaussian toy (validates C9).

C9 reported slope −0.750 on real HS2 with per-subsample isotonic refit.
This script reproduces the same subsampling + refit protocol on the
synthetic toy (analytic `R* = Φ(−1) ≈ 0.1587`) to check whether −0.750
is an estimator property (replicates on synthetic) or a real-data
artifact (slope differs on synthetic).

Protocol:
  - Train the same MLP as P0 V3 on 2-Gaussian data.
  - Get softmax p_a on a 5000-sample test pool (with known gold).
  - Subsample (without replacement) at N ∈ {100, 150, 200, 300, 500, 700, 800,
    1000, 1500, 2000, 3000, 4000}.
  - Per subsample: refit IsotonicRegression from scratch, compute R̂*_iso.
  - 100 seeds per N; aggregate std(R̂*_iso) per N across seeds, fit slope.

Pass criterion: slope within ±0.15 of the HS2 −0.750 result → C9 is
reproducibly an estimator property.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression

SKILL = Path.home() / ".agents/skills/scientific-visualization/scripts"
sys.path.insert(0, str(SKILL))
from style_presets import set_color_palette  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

sys.path.insert(0, str(REPO / "src"))
from calibration_utils import estimate_rstar  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)

# ---- train MLP on 2-Gaussian data (same setup as P0 V3) ----
def gen_gauss(n, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    mu0 = np.array([-1.0, 0.0]); mu1 = np.array([+1.0, 0.0])
    x = np.where(y[:, None] == 0, mu0, mu1) + rng.normal(0, 1, (n, 2))
    return x.astype(np.float32), y.astype(np.int64)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x):
        return self.net(x)


Xtr, Ytr = gen_gauss(10000, seed=0)
Xte, Yte = gen_gauss(5000, seed=1)
m = MLP()
opt = torch.optim.Adam(m.parameters(), lr=3e-3)
xt = torch.tensor(Xtr); yt = torch.tensor(Ytr)
for _ in range(300):
    opt.zero_grad()
    l = nn.functional.cross_entropy(m(xt), yt)
    l.backward(); opt.step()

with torch.no_grad():
    logits = m(torch.tensor(Xte)).numpy()
p_a = np.exp(logits[:, 1]) / np.exp(logits).sum(1)
acc = float(((p_a >= 0.5).astype(int) == Yte).mean())
TRUE_R = 0.15865525393145707   # Φ(−1)
print(f"Synthetic setup: N_test={len(p_a)}  train accuracy check={acc:.4f}  "
      f"analytic R*={TRUE_R:.4f}")

# ---- subsampling + isotonic refit ----
N_SEEDS = 100
NS = [100, 150, 200, 300, 500, 700, 800, 1000, 1500, 2000, 3000, 4000]
rng = np.random.default_rng(42)
records = []
for N in NS:
    if N >= len(p_a):
        continue
    for seed in range(N_SEEDS):
        idx = rng.choice(len(p_a), size=N, replace=False)
        sub_pa, sub_g = p_a[idx], Yte[idx]
        ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        ir.fit(sub_pa, sub_g)
        cal = ir.predict(sub_pa)
        r_hat = estimate_rstar(cal)
        records.append({"N": N, "seed": seed, "rstar_iso": float(r_hat)})

import pandas as pd
df = pd.DataFrame(records)
agg = df.groupby("N")["rstar_iso"].agg(["std", "mean"]).reset_index()
print("\nBootstrap std per N (synthetic):")
print(agg.to_string(index=False))

log_N = np.log(agg["N"].values)
log_std = np.log(agg["std"].values)
slope, intercept = np.polyfit(log_N, log_std, 1)
residuals = log_std - (slope * log_N + intercept)
r2 = 1 - np.sum(residuals ** 2) / np.sum((log_std - log_std.mean()) ** 2)

hs2_slope = -0.750   # C9 real-data result
pass_ = bool(abs(slope - hs2_slope) < 0.15)
print(f"\nSynthetic slope = {slope:.3f}  (HS2 real-data slope = {hs2_slope:.3f})")
print(f"Deviation from HS2: {abs(slope - hs2_slope):.3f}   R² = {r2:.3f}")
print(f"PASS (|syn - hs2| < 0.15): {pass_}")

stats = {
    "n_seeds_per_N": N_SEEDS,
    "N_values": NS,
    "synthetic_slope": float(slope),
    "hs2_real_slope_c9": hs2_slope,
    "synthetic_vs_hs2_deviation": float(abs(slope - hs2_slope)),
    "synthetic_r2": float(r2),
    "true_r_star": TRUE_R,
    "pass_criterion": "|synthetic_slope - hs2_slope| < 0.15",
    "pass": pass_,
    "per_N_std": {int(n): float(s) for n, s in zip(agg["N"].values, agg["std"].values)},
    "per_N_mean": {int(n): float(m) for n, m in zip(agg["N"].values, agg["mean"].values)},
}
with open(HERE / "results_v5_sample_complexity.json", "w") as f:
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
    "figure.constrained_layout.use": True,
    "legend.fontsize": 8,
})
fig, ax = plt.subplots(figsize=(6.5, 4.0))
ax.plot(agg["N"], agg["std"], marker="s", color="black", linewidth=1.8,
        markersize=6, markerfacecolor="white", markeredgewidth=1.2,
        label=r"synthetic $\hat{R}^*_{\mathrm{iso}}$ std")
N_line = np.array([agg["N"].min(), agg["N"].max()])
fit_line = np.exp(intercept) * N_line ** slope
ax.plot(N_line, fit_line, linestyle="--", color="#D55E00", linewidth=1.3,
        label=fr"fit: slope ${slope:+.3f}$")
hs2_line = agg["std"].iloc[0] * (N_line / agg["N"].iloc[0]) ** hs2_slope
ax.plot(N_line, hs2_line, linestyle=":", color="#0072B2", linewidth=1.3,
        label=fr"HS2 real-data slope ${hs2_slope:+.3f}$ (C9)")
clt_line = agg["std"].iloc[0] * (N_line / agg["N"].iloc[0]) ** -0.5
ax.plot(N_line, clt_line, linestyle="-.", color="#888", linewidth=1.0,
        label=r"CLT reference $-0.500$")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Subsample size $N$")
ax.set_ylabel(r"Bootstrap $\mathrm{std}(\hat{R}^*_{\mathrm{iso}})$")
ax.set_title(
    fr"P0 V5 — Synthetic sample complexity: slope ${slope:+.3f}$ "
    fr"vs HS2 real-data $-0.750$  ({'PASS' if pass_ else 'FAIL'})"
)
ax.legend(loc="best", handlelength=1.8)

out = HERE / "fig_synthetic_sample_complexity.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print(f"\nSaved: {out}")
