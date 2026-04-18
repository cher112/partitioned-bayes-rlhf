"""P0 V3: Tight correlation with signed confidence bias.

V2 showed `Pearson r(ECE, R̂*) = +0.55` on the T-scaling experiment, which is
weak because ECE is a symmetric metric: both T=0.3 (overconfident, negative R̂*
bias) and T=4.0 (underconfident, positive R̂* bias) have high ECE.

The right predictor is a *signed* calibration quantity. In fact, by definition,
    R̂* = E_N[min(η̂, 1-η̂)] = 0.5 - E_N[|η̂ - 0.5|],
so R̂* is mathematically identical to `0.5 - sharpness(η̂)`.

V3 demonstrates:
    Pearson r(sharpness, R̂*) = -1.0  (tautology, crisp slide)
    Pearson r(avg_confidence, R̂* - true_R*) ≈ -1 (per-T signed bias)
with 10 temperature points and the same underlying Gaussian problem.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm as scipy_norm, pearsonr

HERE = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
np.random.seed(42)

N_TRAIN, N_TEST, SIGMA = 20000, 10000, 1.0
MU = 2.0
true_rstar = scipy_norm.cdf(-MU / 2 / SIGMA)
print(f"True R* = {true_rstar:.4f}")

def gen_gauss(n, seed):
    rng = np.random.RandomState(seed)
    y = rng.randint(0, 2, size=n)
    mu_pos, mu_neg = np.array([+MU/2, 0.0]), np.array([-MU/2, 0.0])
    x = np.where(y[:, None] == 1,
                 mu_pos + SIGMA * rng.randn(n, 2),
                 mu_neg + SIGMA * rng.randn(n, 2))
    return x.astype(np.float32), y.astype(np.int64)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 64), nn.ReLU(),
                                  nn.Linear(64, 64), nn.ReLU(),
                                  nn.Linear(64, 2))
    def forward(self, x): return self.net(x)

x_tr, y_tr = gen_gauss(N_TRAIN, 42)
model = MLP().to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
x, y = torch.from_numpy(x_tr).to(DEVICE), torch.from_numpy(y_tr).to(DEVICE)
for _ in range(40):
    perm = torch.randperm(N_TRAIN)
    for i in range(0, N_TRAIN, 512):
        idx = perm[i:i+512]
        loss = F.cross_entropy(model(x[idx]), y[idx])
        opt.zero_grad(); loss.backward(); opt.step()

@torch.no_grad()
def logits_on_test():
    x_te, y_te = gen_gauss(N_TEST, 43)
    lg = model(torch.from_numpy(x_te).to(DEVICE)).cpu().numpy()
    return lg, y_te

logits, y_test = logits_on_test()

def ece(probs, labels, n_bins=15):
    confs, preds = probs.max(axis=-1), probs.argmax(axis=-1)
    correct = (preds == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (confs >= lo) & (confs < hi)
        if m.sum() > 0:
            e += (m.sum()/len(confs)) * abs(correct[m].mean() - confs[m].mean())
    return e

temps = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
results = []
for T in temps:
    s = logits / T
    pr = np.exp(s - s.max(-1, keepdims=True)); pr /= pr.sum(-1, keepdims=True)
    eta = pr[:, 1]
    acc = float((pr.argmax(-1) == y_test).mean())
    avg_conf = float(pr.max(-1).mean())
    sharpness = float(np.abs(eta - 0.5).mean())       # E[|η̂ - 1/2|]
    rstar = 0.5 - sharpness
    signed_bias = avg_conf - acc                       # overconfident when > 0
    results.append({
        "T": T, "acc": acc, "avg_conf": avg_conf,
        "signed_bias": signed_bias, "ece": ece(pr, y_test),
        "sharpness": sharpness, "rstar_plugin": rstar,
        "gap_rstar": rstar - float(true_rstar),
    })

print(f"\n{'T':>6} {'acc':>6} {'conf':>6} {'signed':>8} {'ECE':>6} {'sharp':>7} {'R̂*':>7} {'gap':>8}")
for r in results:
    print(f"{r['T']:6.2f} {r['acc']:6.3f} {r['avg_conf']:6.3f} "
          f"{r['signed_bias']:+8.4f} {r['ece']:6.4f} {r['sharpness']:7.4f} "
          f"{r['rstar_plugin']:7.4f} {r['gap_rstar']:+8.4f}")

# correlations
sb = np.array([r["signed_bias"] for r in results])
sharp = np.array([r["sharpness"] for r in results])
rstars = np.array([r["rstar_plugin"] for r in results])
gap = np.array([r["gap_rstar"] for r in results])
eces = np.array([r["ece"] for r in results])

corrs = {
    "sharpness_vs_rstar": pearsonr(sharp, rstars),
    "signed_bias_vs_gap": pearsonr(sb, gap),
    "ece_vs_rstar": pearsonr(eces, rstars),
    "ece_vs_abs_gap": pearsonr(eces, np.abs(gap)),
}
print("\nCorrelations:")
for k, (r, p) in corrs.items():
    print(f"  {k}: r={r:+.4f} p={p:.2e}")

# ------ plot ------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

ax = axes[0]
ax.plot(sharp, rstars, "o-", ms=9, lw=2, color="#2b6cb0")
ax.plot([0, 0.5], [0.5, 0], "k:", alpha=0.4, label=r"$\hat{R}^{*} = 0.5 - \mathrm{sharpness}$")
ax.axhline(true_rstar, color="red", ls="--", label=f"True R* = {true_rstar:.3f}")
ax.set_xlabel(r"Sharpness $\mathbb{E}[|\hat{\eta} - 1/2|]$")
ax.set_ylabel(r"Plug-in $\hat{R}^*$")
r_s, _ = corrs["sharpness_vs_rstar"]
ax.set_title(f"A. $\\hat{{R}}^* \\equiv 0.5 - \\mathrm{{sharpness}}$  (r = {r_s:+.2f})")
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(sb, gap, "o-", ms=9, lw=2, color="#c05621")
for r in results:
    ax.annotate(f"T={r['T']:.1f}", (r["signed_bias"], r["gap_rstar"]),
                xytext=(5, 4), textcoords="offset points", fontsize=7)
ax.axhline(0, color="k", ls=":", alpha=0.5)
ax.axvline(0, color="k", ls=":", alpha=0.5)
ax.set_xlabel("Signed confidence bias (avg_conf − accuracy)")
ax.set_ylabel(r"$\hat{R}^* - R^{*}_{\mathrm{true}}$")
r_sb, _ = corrs["signed_bias_vs_gap"]
ax.set_title(f"B. Signed bias predicts R̂* gap  (r = {r_sb:+.2f})")
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.scatter(eces, rstars, s=100, edgecolors="black", color="#38a169")
for r in results:
    ax.annotate(f"T={r['T']:.1f}", (r["ece"], r["rstar_plugin"]),
                xytext=(5, 4), textcoords="offset points", fontsize=7)
ax.axhline(true_rstar, color="red", ls="--", label=f"True R* = {true_rstar:.3f}")
r_e, _ = corrs["ece_vs_rstar"]
ax.set_xlabel("ECE (unsigned)")
ax.set_ylabel(r"$\hat{R}^*$")
ax.set_title(f"C. ECE is symmetric — confounds sign  (r = {r_e:+.2f})")
ax.legend(); ax.grid(True, alpha=0.3)

fig.suptitle(
    "P0 V3: The clean diagnostic is signed calibration bias, not ECE. "
    r"$\hat{R}^*$ is mathematically identical to $0.5 - \mathrm{sharpness}(\hat{\eta})$.",
    fontsize=12, y=1.00,
)
fig.tight_layout()
fig.savefig(HERE / "fig_synthetic_rstar_v3.png", dpi=150, bbox_inches="tight")

with open(HERE / "results_v3.json", "w") as f:
    json.dump({
        "true_rstar": float(true_rstar),
        "mu_dist": MU,
        "temperatures": temps,
        "per_T": results,
        "pearson": {k: {"r": float(v[0]), "p": float(v[1])} for k, v in corrs.items()},
        "identity_note": "R̂* = E_N[min(η̂, 1-η̂)] = 0.5 - E_N[|η̂ - 0.5|] exactly",
    }, f, indent=2)

print(f"\nSaved {HERE / 'fig_synthetic_rstar_v3.png'}")
