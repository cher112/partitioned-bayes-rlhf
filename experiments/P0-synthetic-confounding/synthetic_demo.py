"""P0: Synthetic 2D confounding demo.

Goal
----
Construct a 2D classification problem with KNOWN ground-truth Bayes error R*.
Train 3 MLPs of increasing capacity. Show:
  1. all three agree on accuracy ordering (bigger = slightly better)
  2. but their plug-in R̂* diverges significantly and
  3. this R̂* divergence is explained by per-model ECE.

Core message
------------
If the confound exists even on fully controlled synthetic data where true R*
is FIXED, then observed R̂* variance in real RLHF data cannot be interpreted
as data-side noise without first correcting for calibration error.

Output
------
- fig_synthetic_rstar.png (3-panel figure)
- results.json (metrics)
"""
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# -----------------------------
# 1.  Data generation with KNOWN Bayes error
# -----------------------------
# Two Gaussians with shared covariance.
# Closed-form Bayes error for equal-prior, equal-cov Gaussians:
#   R* = Phi(-d/2)  where d = Mahalanobis distance between means
# d=2 gives R* = Phi(-1) ≈ 0.1587
from scipy.stats import norm as scipy_norm
N_TRAIN = 20000
N_TEST = 10000
DIM = 2
MU_DIST = 2.0
SIGMA = 1.0


def gen_gauss(n, seed):
    rng = np.random.RandomState(seed)
    y = rng.randint(0, 2, size=n)
    mu_pos = np.array([+MU_DIST / 2, 0.0])
    mu_neg = np.array([-MU_DIST / 2, 0.0])
    x = np.where(
        y[:, None] == 1,
        mu_pos + SIGMA * rng.randn(n, DIM),
        mu_neg + SIGMA * rng.randn(n, DIM),
    )
    return x.astype(np.float32), y.astype(np.int64)


x_train, y_train = gen_gauss(N_TRAIN, SEED)
x_test, y_test = gen_gauss(N_TEST, SEED + 1)

true_rstar = scipy_norm.cdf(-MU_DIST / 2 / SIGMA)
print(f"True R* (analytical) = {true_rstar:.4f}")


# -----------------------------
# 2.  MLP with configurable width
# -----------------------------
class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DIM, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, 2),
        )

    def forward(self, x):
        return self.net(x)


def train_model(width, epochs=30, bs=512, lr=1e-3):
    model = MLP(width).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x = torch.from_numpy(x_train).to(DEVICE)
    y = torch.from_numpy(y_train).to(DEVICE)
    for _ in range(epochs):
        perm = torch.randperm(N_TRAIN)
        for i in range(0, N_TRAIN, bs):
            idx = perm[i:i + bs]
            logits = model(x[idx])
            loss = F.cross_entropy(logits, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def eval_model(model):
    x = torch.from_numpy(x_test).to(DEVICE)
    y = torch.from_numpy(y_test).to(DEVICE)
    logits = model(x)
    probs = F.softmax(logits, dim=-1).cpu().numpy()
    y_np = y.cpu().numpy()
    preds = probs.argmax(axis=-1)
    acc = (preds == y_np).mean()
    # η̂ = P(Y=1 | x)
    eta = probs[:, 1]
    # plug-in R̂* = E[min(η̂, 1-η̂)]
    rstar_plugin = np.minimum(eta, 1 - eta).mean()
    # ECE (equal-width, 15 bins)
    confs = probs.max(axis=-1)
    correct = (preds == y_np).astype(np.float64)
    bins = np.linspace(0, 1, 16)
    ece = 0.0
    for b0, b1 in zip(bins[:-1], bins[1:]):
        mask = (confs >= b0) & (confs < b1)
        if mask.sum() > 0:
            ece += (mask.sum() / len(confs)) * abs(
                correct[mask].mean() - confs[mask].mean()
            )
    # Brier score
    y_onehot = np.eye(2)[y_np]
    brier = ((probs - y_onehot) ** 2).sum(axis=-1).mean()
    return {
        "acc": float(acc),
        "rstar_plugin": float(rstar_plugin),
        "ece": float(ece),
        "brier": float(brier),
        "avg_confidence": float(confs.mean()),
    }


# -----------------------------
# 3.  Train 3 MLPs of varying capacity
# -----------------------------
WIDTHS = [4, 16, 128]  # under → mid → over capacity
results = []
for w in WIDTHS:
    print(f"Training MLP width={w} ...")
    m = train_model(w, epochs=40)
    metrics = eval_model(m)
    metrics["width"] = w
    metrics["gap_rstar"] = metrics["rstar_plugin"] - true_rstar
    results.append(metrics)
    print(f"  width={w}: acc={metrics['acc']:.3f}, "
          f"R̂*={metrics['rstar_plugin']:.4f} (true={true_rstar:.4f}, gap={metrics['gap_rstar']:+.4f}), "
          f"ECE={metrics['ece']:.4f}")

# -----------------------------
# 4.  Plot
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
widths = np.array([r["width"] for r in results])
rstars = np.array([r["rstar_plugin"] for r in results])
eces = np.array([r["ece"] for r in results])
accs = np.array([r["acc"] for r in results])

# panel A: width vs R̂*
ax = axes[0]
ax.plot(widths, rstars, "o-", markersize=10, color="#2b6cb0")
ax.axhline(true_rstar, color="red", ls="--", label=f"True R* = {true_rstar:.3f}")
ax.set_xscale("log")
ax.set_xlabel("MLP width (capacity)")
ax.set_ylabel(r"Plug-in $\hat{R}^*$")
ax.set_title("A. Capacity changes $\\hat{R}^*$ while true R* is fixed")
ax.legend()
ax.grid(True, alpha=0.3)

# panel B: ECE vs R̂*
ax = axes[1]
for r in results:
    ax.scatter(r["ece"], r["rstar_plugin"], s=150, edgecolors="black", linewidths=0.7)
    ax.annotate(f"w={r['width']}", (r["ece"], r["rstar_plugin"]),
                xytext=(8, 4), textcoords="offset points", fontsize=9)
ax.axhline(true_rstar, color="red", ls="--", label=f"True R* = {true_rstar:.3f}")
from scipy.stats import pearsonr
r_corr, p_corr = pearsonr(eces, rstars)
ax.set_xlabel("ECE")
ax.set_ylabel(r"Plug-in $\hat{R}^*$")
ax.set_title(f"B. R̂* tracks ECE (Pearson r={r_corr:+.2f})")
ax.legend()
ax.grid(True, alpha=0.3)

# panel C: accuracy is almost flat
ax = axes[2]
ax.plot(widths, accs, "o-", markersize=10, color="#38a169")
ax.axhline(1 - true_rstar, color="red", ls="--",
           label=f"Bayes accuracy = {1 - true_rstar:.3f}")
ax.set_xscale("log")
ax.set_xlabel("MLP width (capacity)")
ax.set_ylabel("Test accuracy")
ax.set_title("C. Accuracy converges (capacity effect saturates)")
ax.legend()
ax.grid(True, alpha=0.3)

fig.suptitle(
    f"P0: Plug-in $\\hat{{R}}^*$ is confounded by classifier capacity "
    f"(synthetic 2D Gaussians, known R*={true_rstar:.3f}, N_test={N_TEST})",
    fontsize=11, y=1.02,
)
fig.tight_layout()
fig.savefig(HERE / "fig_synthetic_rstar.png", dpi=150, bbox_inches="tight")

with open(HERE / "results.json", "w") as f:
    json.dump({
        "true_rstar": float(true_rstar),
        "bayes_accuracy": float(1 - true_rstar),
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "mu_dist": MU_DIST,
        "sigma": SIGMA,
        "device": DEVICE,
        "per_model": results,
        "pearson_ece_rstar": {"r": float(r_corr), "p": float(p_corr)},
    }, f, indent=2)

print(f"\nSaved {HERE / 'fig_synthetic_rstar.png'}")
print(f"Saved {HERE / 'results.json'}")
