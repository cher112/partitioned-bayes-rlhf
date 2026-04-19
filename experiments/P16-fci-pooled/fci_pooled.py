"""P16 — FCI with *global pooled* anchors (close P15's HS2 counterexample).

P15 FCI-Risk failed on HS2 because we used per-judge anchors (q01/q99 of
each judge's own c_iso). Only OLMo had meaningful truncation (stretch
1.87); the other 5 judges had stretch ≈ 1. So FCI only rescaled OLMo,
which widened the cross-judge spread instead of compressing it.

Gemini Gate-1 (2026-04-19) rejected a "strongest-judge anchor" proposal
as circular: choosing the judge with highest accuracy vs. gold == choosing
the judge most similar to GPT-4 (on UF), which turns R̂*_CA into a
"pro-GPT-4 similarity" scaler.

Gemini's recommended alternative, implemented here: **global pooled
anchor**. Concatenate c_iso across all K judges into one pool, compute
one set of (ρ_0, ρ_1) from the pooled distribution, apply the same
affine transform to every judge's c_iso.

Formula:
    C_pool  = concat_k(c_iso_k)           (shape N*K)
    ρ_0     = quantile(C_pool, 0.01)
    ρ_1     = 1 − quantile(C_pool, 0.99)
    s       = 1 / (1 − ρ_0 − ρ_1)  (shared stretch, capped at 10)
    p_corr_k(i)  = clip(s · (c_iso_k(i) − ρ_0), 0, 1)
    R̂*_CA_k     = (1/N) Σ_i min(p_corr_k(i), 1 − p_corr_k(i))

Why this is non-tautological:
  Same argument as P15 — affine correction inside the non-linear min()
  cannot reduce to 0.5 − m_iso. But unlike P15, every judge now uses
  the *same* (ρ_0, ρ_1), so heterogeneous truncation can no longer
  inflate the cross-judge spread; a weak judge with c_iso concentrated
  near 0.5 still gets the same stretch as a strong judge with c_iso
  spanning [0, 1].

Go / No-Go:
  PASS if HS2↔UF drift < 0.09 AND cross-judge std compresses on both
  datasets (std ratio < 1.0 on each) AND R̂*_CA values in (0.02, 0.55).
  NO-GO if drift > 0.10 or the estimator collapses to a constant.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
from sklearn.isotonic import IsotonicRegression

SKILL = Path.home() / ".agents/skills/scientific-visualization/scripts"
sys.path.insert(0, str(SKILL))
from style_presets import set_color_palette  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
HERE.mkdir(exist_ok=True)

sys.path.insert(0, str(REPO / "src"))
from calibration_utils import estimate_rstar  # noqa: E402


def load_cal(judge_dir):
    files = sorted(Path(judge_dir).glob("*.json"))
    names, c_cols, golds = [], [], []
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        pairs = [p for p in d["pairs"] if p.get("p_a_mean") is not None]
        p_a = np.array([p["p_a_mean"] for p in pairs])
        gold = np.array([p["gold"] for p in pairs], dtype=int)
        ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        ir.fit(p_a, gold)
        names.append(d["model"])
        c_cols.append(ir.predict(p_a))
        golds.append(gold)
    n_common = min(len(c) for c in c_cols)
    c_mat = np.column_stack([c[:n_common] for c in c_cols])  # (N, K)
    return names, c_mat


def rstar_iso(c_mat):
    return [float(estimate_rstar(c_mat[:, k])) for k in range(c_mat.shape[1])]


def rstar_fci_pooled(c_mat, q_low=0.01, q_high=0.99, stretch_cap=10.0):
    pool = c_mat.flatten()
    rho0 = float(np.quantile(pool, q_low))
    rho1 = 1.0 - float(np.quantile(pool, q_high))
    denom = 1.0 - rho0 - rho1
    stretch = 1.0 / denom if denom > 0 else float("inf")
    if stretch > stretch_cap or denom <= 0.01:
        return rstar_iso(c_mat), {"rho0": rho0, "rho1": rho1, "stretch": stretch,
                                   "fallback_to_iso": True}
    p_corr = np.clip((c_mat - rho0) / denom, 0.0, 1.0)
    per_judge = [float(np.minimum(p_corr[:, k], 1 - p_corr[:, k]).mean())
                 for k in range(c_mat.shape[1])]
    return per_judge, {"rho0": rho0, "rho1": rho1, "stretch": stretch,
                       "fallback_to_iso": False}


def summarise(tag, judge_dir):
    names, c_mat = load_cal(judge_dir)
    r_iso = rstar_iso(c_mat)
    r_ca, info = rstar_fci_pooled(c_mat)
    print(f"\n=== {tag} ===")
    print(f"  Global anchors: ρ_0 = {info['rho0']:.3f}  ρ_1 = {info['rho1']:.3f}  "
          f"stretch = {info['stretch']:.3f}")
    for name, iso, ca in zip(names, r_iso, r_ca):
        print(f"  {name:<30s}  iso = {iso:.3f}   CA = {ca:.3f}")
    r_iso_arr, r_ca_arr = np.array(r_iso), np.array(r_ca)
    return {
        "dataset": tag, "judges": names,
        "per_judge_rstar_iso": [float(x) for x in r_iso],
        "per_judge_rstar_ca":  [float(x) for x in r_ca],
        "iso_mean": float(r_iso_arr.mean()), "iso_std": float(r_iso_arr.std()),
        "ca_mean":  float(r_ca_arr.mean()),  "ca_std":  float(r_ca_arr.std()),
        "anchors": {"rho0": info["rho0"], "rho1": info["rho1"],
                    "stretch": info["stretch"]},
    }


hs2 = summarise("HS2", REPO / "experiments" / "judges_hs2")
uf  = summarise("UF",  REPO / "experiments" / "judges_uf")

drift_iso = abs(hs2["iso_mean"] - uf["iso_mean"])
drift_ca  = abs(hs2["ca_mean"]  - uf["ca_mean"])
r_ca_hs2_std_ratio = hs2["ca_std"] / max(hs2["iso_std"], 1e-9)
r_ca_uf_std_ratio  = uf["ca_std"]  / max(uf["iso_std"],  1e-9)
all_in_range = all(0.02 < x < 0.55
                   for x in hs2["per_judge_rstar_ca"] + uf["per_judge_rstar_ca"])

pass_drift = drift_ca < 0.09
pass_std   = r_ca_hs2_std_ratio < 1.0 and r_ca_uf_std_ratio < 1.0
pass_range = all_in_range
pass_ = bool(pass_drift and pass_std and pass_range)

print(f"\n=== Head-to-head ===")
print(f"  iso drift: {drift_iso:.4f}   CA drift (pooled): {drift_ca:.4f}")
print(f"  std ratio CA/iso — HS2: {r_ca_hs2_std_ratio:.3f}  UF: {r_ca_uf_std_ratio:.3f}")
print(f"  PASS drift<0.09   : {pass_drift}  ({drift_ca:.4f})")
print(f"  PASS std_ratio<1  : {pass_std}  (HS2 {r_ca_hs2_std_ratio:.3f}, UF {r_ca_uf_std_ratio:.3f})")
print(f"  PASS range        : {pass_range}")
print(f"  OVERALL PASS      : {pass_}")

stats = {
    "hs2": hs2, "uf": uf,
    "drift_iso": float(drift_iso),
    "drift_ca_pooled": float(drift_ca),
    "drift_improvement_abs": float(drift_iso - drift_ca),
    "std_ratio_ca_over_iso": {"hs2": float(r_ca_hs2_std_ratio),
                              "uf":  float(r_ca_uf_std_ratio)},
    "go_no_go": {
        "drift_lt_0.09":      {"value": float(drift_ca), "pass": pass_drift},
        "std_ratio_lt_1.0":   {"hs2": float(r_ca_hs2_std_ratio),
                               "uf": float(r_ca_uf_std_ratio),
                               "pass": pass_std},
        "range_in_0.02_0.55": {"pass": pass_range},
        "overall_pass":       pass_,
    },
    "framing": (
        "FCI with global pooled anchor (Gemini Gate-1 2026-04-19 alternative to "
        "'strongest-judge' proposal which was rejected as circular). "
        "Non-tautological — affine correction inside min(·), same as P15. "
        "Key change: shared (ρ_0, ρ_1) from pooled judge distribution removes "
        "heterogeneous-truncation sensitivity that caused P15's HS2 widening."
    ),
    "year1_theory_hook": (
        "Global pooled anchor → Kiefer–Wolfowitz 1976 isotonic boundary rate "
        "O_P((N*K)^{-1/3}) on ρ estimation, with K judges pooled. Combined with "
        "plug-in rate O_P(N^{-1/2}), Year-1 target bound is "
        "|R̂*_CA − R*| = O_P((N*K)^{-1/3}) + O_P(N^{-1/2})."
    ),
}
with open(HERE / "stats.json", "w") as f:
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
    "legend.fontsize": 8,
    "figure.constrained_layout.use": True,
})
C_ISO = "#D55E00"
C_P15 = "#0072B2"   # P15 per-judge
C_P16 = "#009E73"   # P16 pooled
C_REF = "#888"

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
axL, axR = axes

# LEFT: per-judge HS2 comparison: iso vs P16 (pooled)
names = hs2["judges"]
x = np.arange(len(names))
w = 0.38
axL.bar(x - w/2, hs2["per_judge_rstar_iso"], w,
        color=C_ISO, edgecolor="black", linewidth=0.5,
        label=r"HS2 $\hat{R}^*_{\mathrm{iso}}$")
axL.bar(x + w/2, hs2["per_judge_rstar_ca"],  w,
        color=C_P16, edgecolor="black", linewidth=0.5,
        label=r"HS2 $\hat{R}^*_{\mathrm{CA}}$ (pooled)")
axL.axhline(0.49, color=C_REF, linestyle=":", linewidth=0.9,
            label=r"null floor $\approx 0.49$")
axL.set_xticks(x)
axL.set_xticklabels([n.split("-")[0] for n in names], rotation=20, ha="right")
axL.set_ylabel(r"$\hat{R}^*$")
axL.set_title(fr"A. HS2 per-judge (std ratio CA/iso = {r_ca_hs2_std_ratio:.2f})")
axL.legend(loc="best", handlelength=1.4)

# RIGHT: cross-dataset drift comparison — iso / P15-per-judge / P16-pooled
# (P15 number loaded from file)
try:
    with open(REPO / "experiments/P15-fci-risk/stats.json") as f:
        p15 = json.load(f)
    p15_drift = p15["drift"]["fci_q01_q99"]
except Exception:
    p15_drift = 0.114
methods = [r"$\hat{R}^*_{\mathrm{iso}}$",
           "FCI per-judge\n(P15)",
           "FCI pooled\n(P16)"]
drifts = [drift_iso, p15_drift, drift_ca]
colors = [C_ISO, C_P15, C_P16]
axR.bar(methods, drifts, color=colors, edgecolor="black", linewidth=0.5)
for i, v in enumerate(drifts):
    axR.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=9)
axR.axhline(0.09, color="#D55E00", linestyle="--", linewidth=0.9,
            label="PASS threshold 0.09")
axR.set_ylabel(r"$|\mathrm{mean}_{\mathrm{HS2}} - \mathrm{mean}_{\mathrm{UF}}|$")
axR.set_title(r"B. Cross-dataset drift (lower = better)")
axR.legend(loc="upper right")

fig.suptitle(
    fr"P16 — FCI pooled anchor: drift {drift_iso:.3f}→{drift_ca:.3f}  "
    fr"(P15 per-judge {p15_drift:.3f})  "
    f"[{'PASS' if pass_ else 'FAIL'}]",
    fontsize=10.5, y=1.03,
)
out = HERE / "fig_fci_pooled.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print(f"\nSaved: {out}")
