"""P0 V2: Temperature-scaling reveals the capacity confound isolated.

V1 used 3 MLPs of different widths on 2D Gaussians — confound too weak because
all widths converge to near-Bayes posterior. V2 fixes this by isolating the
confound mechanism: a single trained MLP whose output is then temperature-scaled
to T ∈ {0.3, 0.5, 1.0, 2.0, 4.0}.

All 5 'judges' come from the same underlying η, so the true Bayes error is
identical. Only ECE varies with T. The plug-in R̂* curve against T makes the
confound unambiguous for the Section-1 figure of the proposal.

The second panel reproduces the V1 setting for completeness but with overlapping
Gaussians (mu_dist=1.2 instead of 2.0) so capacity effects are visible on the
tails.
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

# ------------------------- shared utilities -------------------------
def ece(probs, labels, n_bins=15):
    confs = probs.max(axis=-1)
    preds = probs.argmax(axis=-1)
    correct = (preds == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (confs >= lo) & (confs < hi)
        if m.sum() > 0:
            e += (m.sum() / len(confs)) * abs(correct[m].mean() - confs[m].mean())
    return e

def rstar_plugin(eta):
    return float(np.minimum(eta, 1 - eta).mean())

# ------------------------- data -------------------------
N_TRAIN, N_TEST, SIGMA = 20000, 10000, 1.0
def gen_gauss(n, mu_dist, seed):
    rng = np.random.RandomState(seed)
    y = rng.randint(0, 2, size=n)
    mu_pos = np.array([+mu_dist / 2, 0.0])
    mu_neg = np.array([-mu_dist / 2, 0.0])
    x = np.where(y[:, None] == 1,
                 mu_pos + SIGMA * rng.randn(n, 2),
                 mu_neg + SIGMA * rng.randn(n, 2))
    return x.astype(np.float32), y.astype(np.int64)

class MLP(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, 2),
        )
    def forward(self, x): return self.net(x)

def train(mu_dist, width=64, epochs=40, bs=512, lr=1e-3):
    x_train, y_train = gen_gauss(N_TRAIN, mu_dist, 42)
    model = MLP(width).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x = torch.from_numpy(x_train).to(DEVICE)
    y = torch.from_numpy(y_train).to(DEVICE)
    for _ in range(epochs):
        perm = torch.randperm(N_TRAIN)
        for i in range(0, N_TRAIN, bs):
            idx = perm[i:i+bs]
            loss = F.cross_entropy(model(x[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return model

@torch.no_grad()
def extract_logits(model, mu_dist):
    x_test, y_test = gen_gauss(N_TEST, mu_dist, 43)
    logits = model(torch.from_numpy(x_test).to(DEVICE)).cpu().numpy()
    return logits, y_test

# ------------------------- Experiment 1: Temperature scaling -------------------------
print("=" * 60)
print("EXP 1: Single MLP, temperature scaling T ∈ {0.3,0.5,1,2,4}")
print("=" * 60)

MU_DIST = 2.0
true_rstar = scipy_norm.cdf(-MU_DIST / 2 / SIGMA)
print(f"True R* (analytical) = {true_rstar:.4f}")

model = train(mu_dist=MU_DIST)
logits, y_test = extract_logits(model, MU_DIST)

temps = [0.3, 0.5, 1.0, 2.0, 4.0]
results_temp = []
for T in temps:
    scaled = logits / T
    probs = np.exp(scaled - scaled.max(axis=-1, keepdims=True))
    probs = probs / probs.sum(axis=-1, keepdims=True)
    eta = probs[:, 1]
    r = {
        "T": T,
        "ece": ece(probs, y_test),
        "acc": float((probs.argmax(-1) == y_test).mean()),
        "rstar_plugin": rstar_plugin(eta),
        "avg_conf": float(probs.max(-1).mean()),
    }
    r["gap_rstar"] = r["rstar_plugin"] - true_rstar
    results_temp.append(r)
    print(f"  T={T:.2f}: acc={r['acc']:.3f}, R̂*={r['rstar_plugin']:.4f} "
          f"(true={true_rstar:.4f}, gap={r['gap_rstar']:+.4f}), ECE={r['ece']:.4f}")

eces_t = np.array([r["ece"] for r in results_temp])
rstars_t = np.array([r["rstar_plugin"] for r in results_temp])
corr_t, p_t = pearsonr(eces_t, rstars_t)
print(f"  Pearson r(ECE, R̂*) = {corr_t:+.4f}  p={p_t:.5f}")

# ------------------------- Experiment 2: Capacity under harder regime -------------------------
print("=" * 60)
print("EXP 2: Widths {4,16,64,256,1024}, harder regime (mu_dist=1.2)")
print("=" * 60)

MU2 = 1.2
true_rstar_2 = scipy_norm.cdf(-MU2 / 2 / SIGMA)
print(f"True R* (mu=1.2) = {true_rstar_2:.4f}")

widths = [4, 16, 64, 256, 1024]
results_cap = []
for w in widths:
    m = train(mu_dist=MU2, width=w, epochs=25)
    lg, yt = extract_logits(m, MU2)
    p = np.exp(lg - lg.max(-1, keepdims=True)); p = p / p.sum(-1, keepdims=True)
    r = {
        "width": w,
        "acc": float((p.argmax(-1) == yt).mean()),
        "rstar_plugin": rstar_plugin(p[:, 1]),
        "ece": ece(p, yt),
        "gap_rstar": rstar_plugin(p[:, 1]) - true_rstar_2,
    }
    results_cap.append(r)
    print(f"  w={w:>4}: acc={r['acc']:.3f}, R̂*={r['rstar_plugin']:.4f} "
          f"(true={true_rstar_2:.4f}, gap={r['gap_rstar']:+.4f}), ECE={r['ece']:.4f}")

eces_w = np.array([r["ece"] for r in results_cap])
rstars_w = np.array([r["rstar_plugin"] for r in results_cap])
corr_w, p_w = pearsonr(eces_w, rstars_w)
print(f"  Pearson r(ECE, R̂*) = {corr_w:+.4f}  p={p_w:.5f}")

# ------------------------- Plot: 2x2 figure -------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# Panel A: T vs R̂*
ax = axes[0, 0]
ax.plot(temps, [r["rstar_plugin"] for r in results_temp], "o-", ms=10, lw=2, color="#2b6cb0")
ax.axhline(true_rstar, color="red", ls="--", label=f"True R* = {true_rstar:.3f}")
ax.set_xlabel("Temperature T")
ax.set_ylabel(r"Plug-in $\hat{R}^*$")
ax.set_xscale("log")
ax.set_title("A. Same classifier, different T ⇒ $\\hat{R}^*$ drifts")
ax.legend(); ax.grid(True, alpha=0.3)

# Panel B: ECE vs R̂* (temperature)
ax = axes[0, 1]
for r in results_temp:
    ax.scatter(r["ece"], r["rstar_plugin"], s=140, edgecolors="black")
    ax.annotate(f"T={r['T']}", (r["ece"], r["rstar_plugin"]),
                xytext=(8, 4), textcoords="offset points", fontsize=9)
ax.axhline(true_rstar, color="red", ls="--", label=f"True R* = {true_rstar:.3f}")
ax.set_xlabel("ECE")
ax.set_ylabel(r"Plug-in $\hat{R}^*$")
ax.set_title(f"B. ECE ↔ $\\hat{{R}}^*$ (Pearson r={corr_t:+.2f})")
ax.legend(); ax.grid(True, alpha=0.3)

# Panel C: width vs R̂* (harder regime)
ax = axes[1, 0]
ax.plot([r["width"] for r in results_cap], [r["rstar_plugin"] for r in results_cap],
        "o-", ms=10, lw=2, color="#c05621")
ax.axhline(true_rstar_2, color="red", ls="--", label=f"True R* = {true_rstar_2:.3f}")
ax.set_xlabel("MLP width")
ax.set_ylabel(r"Plug-in $\hat{R}^*$")
ax.set_xscale("log")
ax.set_title("C. Capacity sweep (mu_dist=1.2, harder regime)")
ax.legend(); ax.grid(True, alpha=0.3)

# Panel D: ECE vs R̂* (capacity)
ax = axes[1, 1]
for r in results_cap:
    ax.scatter(r["ece"], r["rstar_plugin"], s=140, edgecolors="black")
    ax.annotate(f"w={r['width']}", (r["ece"], r["rstar_plugin"]),
                xytext=(8, 4), textcoords="offset points", fontsize=9)
ax.axhline(true_rstar_2, color="red", ls="--", label=f"True R* = {true_rstar_2:.3f}")
ax.set_xlabel("ECE")
ax.set_ylabel(r"Plug-in $\hat{R}^*$")
ax.set_title(f"D. ECE ↔ $\\hat{{R}}^*$ (Pearson r={corr_w:+.2f})")
ax.legend(); ax.grid(True, alpha=0.3)

fig.suptitle(
    "P0 V2: Plug-in $\\hat{R}^*$ is a calibration-driven quantity, not a data-noise proxy",
    fontsize=12, y=1.00,
)
fig.tight_layout()
fig.savefig(HERE / "fig_synthetic_rstar_v2.png", dpi=150, bbox_inches="tight")

with open(HERE / "results_v2.json", "w") as f:
    json.dump({
        "exp1_temperature_scaling": {
            "mu_dist": MU_DIST,
            "true_rstar": float(true_rstar),
            "results": results_temp,
            "pearson_ece_rstar": {"r": float(corr_t), "p": float(p_t)},
        },
        "exp2_capacity_harder": {
            "mu_dist": MU2,
            "true_rstar": float(true_rstar_2),
            "results": results_cap,
            "pearson_ece_rstar": {"r": float(corr_w), "p": float(p_w)},
        },
        "device": DEVICE,
    }, f, indent=2)

print(f"\nSaved {HERE / 'fig_synthetic_rstar_v2.png'}")
print(f"Saved {HERE / 'results_v2.json'}")
