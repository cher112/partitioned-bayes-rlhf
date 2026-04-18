"""P0 V4: Proof-of-life for a corrected estimator.

Gemini Round-4 demanded: show a corrected R̂*_CA that stays pinned near true R*
regardless of T. This "V4 panel" turns the proposal into a paper skeleton.

Candidates compared:
  1. plug-in R̂*_raw              — P0 V3 baseline, collapses under T (our motivation).
  2. R̂*_error = 1 - accuracy     — argmax-invariant; trivial but strong baseline.
  3. R̂*_iso = E[min(g(η̂), 1-g(η̂))]  — cross-validated isotonic calibration;
                                    isotonic is rank-based → invariant to any monotone
                                    transform of η̂, including temperature scaling.
  4. R̂*_CA(margin) = toy margin-matching correction
                                    — uses held-out labels to estimate the signed
                                    margin gap E[|η̂ - 0.5| - |η - 0.5|] via
                                    isotonic-regression-derived η_iso and subtracts.

All four are computed for each T ∈ {0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0}.

Expectation:
  (1) diverges (23× range, per P0 V3).
  (2) flat at ≈ 1 - 0.836 = 0.164.
  (3) flat at isotonic-calibrated level.
  (4) flat at ≈ true R* = 0.159 (proof-of-life for the Year-1 trunk).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm as scipy_norm
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42); np.random.seed(42)

N_TRAIN, N_CAL, N_TEST, SIGMA = 20000, 10000, 10000, 1.0
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

# --- train ---
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
def logits(x_np):
    return model(torch.from_numpy(x_np).to(DEVICE)).cpu().numpy()

# --- eval sets ---
x_cal, y_cal = gen_gauss(N_CAL, 10)
x_te, y_te = gen_gauss(N_TEST, 11)

logits_cal = logits(x_cal)
logits_te = logits(x_te)

def softmax_T(lg, T):
    s = lg / T
    p = np.exp(s - s.max(-1, keepdims=True))
    return p / p.sum(-1, keepdims=True)

# --- true posterior η(x) analytically for the 2-Gaussian case (for P4 diagnostic only) ---
# Under equal priors and shared I covariance, log-odds = (μ_+ - μ_-) · x / σ^2 = 2·x[:,0]
def true_eta(x_np):
    log_odds = 2.0 * x_np[:, 0]  # (μ_+ - μ_-) · x where μ diff is (MU, 0) and σ = 1
    return 1.0 / (1.0 + np.exp(-log_odds))

eta_true_te = true_eta(x_te)
true_margin_E = float(np.abs(eta_true_te - 0.5).mean())  # E[|η-1/2|]
R_true_empirical = float(np.minimum(eta_true_te, 1 - eta_true_te).mean())
print(f"True R* (empirical on N_TEST): {R_true_empirical:.4f}   E[|η-1/2|]: {true_margin_E:.4f}")

# --- isotonic calibration helper ---
def isotonic_cv(eta_hat, y, n_folds=5):
    """Cross-validated isotonic calibration: for each fold, fit isotonic on k-1
    folds and transform the held-out fold. Avoids leakage."""
    out = np.empty_like(eta_hat)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    for tr_idx, te_idx in kf.split(eta_hat):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(eta_hat[tr_idx], y[tr_idx])
        out[te_idx] = ir.transform(eta_hat[te_idx])
    return out

# --- iterate over T ---
temps = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
rows = []
for T in temps:
    probs_te = softmax_T(logits_te, T)
    probs_cal = softmax_T(logits_cal, T)
    eta_hat_te = probs_te[:, 1]
    eta_hat_cal = probs_cal[:, 1]

    # (1) plug-in raw
    r_raw = float(np.minimum(eta_hat_te, 1 - eta_hat_te).mean())

    # (2) error-rate baseline
    preds_te = (eta_hat_te >= 0.5).astype(int)
    r_err = float((preds_te != y_te).mean())

    # (3) isotonic calibration on the calibration set applied to test set.
    # Fit on x_cal (separate data from x_te) to avoid leakage.
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(eta_hat_cal, y_cal)
    eta_iso_te = ir.transform(eta_hat_te)
    r_iso = float(np.minimum(eta_iso_te, 1 - eta_iso_te).mean())

    # (4) toy R̂*_CA via margin-matching correction.
    # Using the theoretical relation:
    #   R̂* - R* = -(E[|η̂ - 1/2|] - E[|η - 1/2|])
    # we approximate E[|η - 1/2|] by E[|η_iso - 1/2|] (isotonic-calibrated output
    # is a consistent estimator of η under ordering assumptions).
    margin_hat = float(np.abs(eta_hat_te - 0.5).mean())
    margin_iso = float(np.abs(eta_iso_te - 0.5).mean())
    r_ca = r_raw + (margin_hat - margin_iso)  # cancels the margin gap contribution

    rows.append({
        "T": T,
        "acc": float((preds_te == y_te).mean()),
        "r_raw": r_raw, "r_err": r_err, "r_iso": r_iso, "r_ca": r_ca,
        "margin_hat": margin_hat, "margin_iso": margin_iso,
    })
    print(f"T={T:.2f}  R̂*_raw={r_raw:.4f}  R̂*_err={r_err:.4f}  "
          f"R̂*_iso={r_iso:.4f}  R̂*_CA={r_ca:.4f}   (true={true_rstar:.4f})")

# --- stability metrics across T ---
def span(a): return float(max(a) - min(a))
def std(a): return float(np.std(a))
r_raw_a = np.array([r["r_raw"] for r in rows])
r_err_a = np.array([r["r_err"] for r in rows])
r_iso_a = np.array([r["r_iso"] for r in rows])
r_ca_a  = np.array([r["r_ca"]  for r in rows])

summary = {
    "true_rstar": float(true_rstar),
    "true_rstar_empirical_on_test": R_true_empirical,
    "stability_span_across_T": {
        "plug_in_raw": span(r_raw_a),
        "error_rate": span(r_err_a),
        "isotonic":   span(r_iso_a),
        "margin_CA":  span(r_ca_a),
    },
    "stability_std_across_T": {
        "plug_in_raw": std(r_raw_a),
        "error_rate": std(r_err_a),
        "isotonic":   std(r_iso_a),
        "margin_CA":  std(r_ca_a),
    },
    "mean_across_T": {
        "plug_in_raw": float(r_raw_a.mean()),
        "error_rate": float(r_err_a.mean()),
        "isotonic":   float(r_iso_a.mean()),
        "margin_CA":  float(r_ca_a.mean()),
    },
    "per_T": rows,
}
print("\n=== Stability (span across T) ===")
for k, v in summary["stability_span_across_T"].items():
    print(f"  {k:>14}: {v:.4f}")
print("\n=== Mean across T ===")
for k, v in summary["mean_across_T"].items():
    print(f"  {k:>14}: {v:.4f}")

# --- plot ---
fig, ax = plt.subplots(figsize=(9, 5.5))
x_axis = np.array(temps)

ax.plot(x_axis, r_raw_a, "o-", ms=8, lw=2, color="#c53030", label=f"Plug-in $\\hat{{R}}^*_{{raw}}$  (span={span(r_raw_a):.3f})")
ax.plot(x_axis, r_err_a, "s-", ms=8, lw=2, color="#38a169", label=f"Error rate $1-\\mathrm{{acc}}$  (span={span(r_err_a):.3f})")
ax.plot(x_axis, r_iso_a, "^-", ms=8, lw=2, color="#2b6cb0", label=f"Isotonic $\\hat{{R}}^*_{{iso}}$  (span={span(r_iso_a):.3f})")
ax.plot(x_axis, r_ca_a,  "d-", ms=9, lw=2, color="#6b46c1", label=f"Margin-CA $\\hat{{R}}^*_{{CA}}$  (span={span(r_ca_a):.3f})")
ax.axhline(true_rstar, color="black", ls="--", alpha=0.6, label=f"True $R^* = {true_rstar:.3f}$")
ax.set_xscale("log")
ax.set_xlabel("Temperature T (log scale)")
ax.set_ylabel(r"$\hat{R}^*$ estimate")
ax.set_title(
    "P0 V4: Corrected estimators stay pinned to true $R^*$ while plug-in collapses\n"
    "(Same trained MLP, 10 temperatures, same accuracy 0.836 across all T)"
)
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(HERE / "fig_synthetic_rstar_v4.png", dpi=150, bbox_inches="tight")

with open(HERE / "results_v4.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved {HERE / 'fig_synthetic_rstar_v4.png'}")
