"""Add bootstrap 95% CI to the 6-judge cross-correlation analysis (P2, with Llama-3).

Operates on the already-produced `experiments/analysis_hs2_6judges/per_judge_rstar.csv`
(no additional LLM inference needed). Adds CIs for:
  - Pearson r(accuracy, R*_iso)
  - Pearson r(ECE_raw, R*_raw)
  - Pearson r(ECE_iso, R*_iso)
  - Cross-judge var(R*_iso)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from bootstrap_ci import bootstrap_ci_correlation, bootstrap_ci_statistic, fisher_z_ci_correlation  # noqa: E402

HERE = Path(__file__).resolve().parent
CSV = REPO / "experiments" / "analysis_hs2_6judges" / "per_judge_rstar.csv"
N_BOOTSTRAP = 10_000   # correlations with N=6 need more resamples for stable CI
SEED = 42

df = pd.read_csv(CSV)
print(f"Loaded {len(df)} judges:\n{df.to_string(index=False)}")

out = {
    "n_judges": len(df),
    "n_bootstrap": N_BOOTSTRAP,
    "seed": SEED,
    "correlations": {},
    "cross_judge_variance": {},
}

# Correlations
pairs = [
    ("acc_vs_rstar_iso", "acc_vs_gold", "rstar_iso"),
    ("ece_raw_vs_rstar_raw", "ece_raw", "rstar_raw"),
    ("ece_iso_vs_rstar_iso", "ece_iso", "rstar_iso"),
]
for name, xcol, ycol in pairs:
    r, p = pearsonr(df[xcol], df[ycol])
    # Bootstrap CI (with NaN filtering) — may have wide CI for N=5
    r_lo_boot, r_hi_boot = bootstrap_ci_correlation(df[xcol].values, df[ycol].values,
                                                    n_resamples=N_BOOTSTRAP, seed=SEED)
    # Fisher-z CI (closed-form, standard for small N)
    r_lo_fz, r_hi_fz = fisher_z_ci_correlation(float(r), len(df))
    out["correlations"][name] = {
        "pearson_r": float(r),
        "p_value": float(p),
        "bootstrap_ci_95": [r_lo_boot, r_hi_boot],
        "fisher_z_ci_95": [r_lo_fz, r_hi_fz],
    }
    print(f"  {name:>28}: r = {r:+.4f}  "
          f"bootstrap CI [{r_lo_boot:+.4f}, {r_hi_boot:+.4f}]  "
          f"Fisher-z CI [{r_lo_fz:+.4f}, {r_hi_fz:+.4f}]  (p={p:.4f})")

# Cross-judge variance of R*_iso
rstar_iso_arr = df["rstar_iso"].values
var_mean = float(np.var(rstar_iso_arr))
var_lo, var_hi = bootstrap_ci_statistic(rstar_iso_arr, stat_fn=lambda x: float(np.var(x)),
                                        n_resamples=N_BOOTSTRAP, seed=SEED + 10)
out["cross_judge_variance"] = {
    "R_star_iso_mean": float(rstar_iso_arr.mean()),
    "R_star_iso_std": float(rstar_iso_arr.std()),
    "R_star_iso_variance": var_mean,
    "variance_ci_lo": var_lo,
    "variance_ci_hi": var_hi,
}
print(f"\n  cross-judge var(R*_iso) = {var_mean:.6f}  [95% CI {var_lo:.6f}, {var_hi:.6f}]")

with open(HERE / "stats_with_ci_6judges.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {HERE / 'stats_with_ci.json'}")
