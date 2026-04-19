"""P15 — FCI-Risk: Forward-Corrected Isotonic Risk, non-tautological `R̂*_CA` candidate.

Motivation
----------
P14 found that the earlier margin-matching `R̂*_CA` was algebraically
identical to `R̂*_iso`:

    R̂*_raw  = E[min(p, 1-p)]    = 0.5 − m_hat
    R̂*_iso  = E[min(c_iso, 1-c_iso)] = 0.5 − m_iso
    R̂*_CA_old = R̂*_raw + (m_hat − m_iso)
              = (0.5 − m_hat) + m_hat − m_iso
              = 0.5 − m_iso
              = R̂*_iso     ∎ (tautology)

Gemini-3.1-pro proposed an alternative, inspired by Patrini et al. 2017
(forward-correction for noisy labels) and Liu & Tao 2015 (anchor-point
estimation of noise rates): push the affine correction **inside** the
non-linear `min()` so the algebra cannot collapse.

FCI-Risk formula
----------------
For each judge k on N pairs with isotonic-calibrated scores c_iso:

    ρ_0 = low-quantile(c_iso)      (lower capacity anchor)
    ρ_1 = 1 − high-quantile(c_iso) (upper capacity anchor)
    p_corr_i = clip( (c_iso_i − ρ_0) / (1 − ρ_0 − ρ_1),  0, 1 )
    R̂*_FCI  = (1/N) Σ_i  min(p_corr_i,  1 − p_corr_i)

Why non-tautological
--------------------
`min(a·c + b, 1 − (a·c + b))` is piecewise-linear with a kink at
`c = (0.5 − b) / a` that is NOT at c = 0.5. So the mapping
`c_iso → min(p_corr, 1−p_corr)` has a different integration region
than `c_iso → min(c_iso, 1−c_iso)`. Concretely, when the stretch
factor `s = 1 / (1 − ρ_0 − ρ_1) > 1`, a pair with c_iso slightly
above 0.5 − (1−ρ_0−ρ_1)/2  + ρ_0  gets pushed across the 0.5 axis
by the affine map, flipping which of {p, 1-p} is the min — and
this discrete re-assignment cannot be reduced to a linear shift of
m_iso. Algebraically,

    E[min(p_corr, 1-p_corr)]
      = 0.5 − E[|p_corr − 0.5|]
      = 0.5 − E[| s·(c_iso − 0.5) + s·(ρ_1 − ρ_0)/2 |]
      ≠ 0.5 − E[|c_iso − 0.5|]  (unless s = 1 and ρ_0 = ρ_1)
      ≠ R̂*_iso      (in the non-degenerate case).

For any judge where c_iso does not already span the full [0, 1] (i.e.
capacity-truncated), the result differs from R̂*_iso.

Robustness
----------
Using empirical min/max as ρ_0, ρ_1 is fragile — one extreme pair can
drive the stretch factor to absurdly large values. We implement **two
variants**: `FCI_minmax` (min/max anchors) and `FCI_q01_q99` (1st / 99th
quantile anchors). The latter is the recommended one.

If the stretch factor `s > 10` we treat the judge as degenerate and fall
back to `R̂*_iso` with a logged warning — at that point the empirical
anchors carry too little information.

Go / No-Go
----------
PASS iff ALL three:
  (a) HS2↔UF drift of the q01/q99 variant < 0.09
  (b) cross-judge std ratio std(FCI_q01_q99) / std(iso) < 0.85
      on at least one dataset
  (c) every per-judge R̂*_FCI in (0.02, 0.55) (no collapse to 0 or 0.5)

Year-1 positioning
------------------
This is a *heuristic precursor* to the Year-1 margin-matching theorem,
not the Year-1 deliverable. The Year-1 work must (i) derive the correct
M^{-1/3} rate for the plug-in anchor estimator (Kiefer–Wolfowitz-style
boundary rate of isotonic regression) and (ii) prove a finite-sample
bias bound on R̂*_CA under BT. FCI-Risk is what the theorem should bound
if it bounds anything.
"""

import json
import sys
import warnings
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


def fci_risk(c_iso, anchor_mode="q01_q99", stretch_cap=10.0):
    """Forward-Corrected Isotonic Risk.

    Returns (r_fci, info_dict).
    If the stretch factor exceeds `stretch_cap`, falls back to R̂*_iso.
    """
    c_iso = np.asarray(c_iso, dtype=float)
    if anchor_mode == "minmax":
        rho0 = float(c_iso.min())
        rho1 = 1.0 - float(c_iso.max())
    elif anchor_mode == "q01_q99":
        rho0 = float(np.quantile(c_iso, 0.01))
        rho1 = 1.0 - float(np.quantile(c_iso, 0.99))
    else:
        raise ValueError(anchor_mode)
    denom = 1.0 - rho0 - rho1
    stretch = 1.0 / denom if denom > 0 else float("inf")
    r_iso = float(estimate_rstar(c_iso))
    if denom <= 0.01 or stretch > stretch_cap:
        warnings.warn(
            f"FCI degenerate (stretch={stretch:.2f}, denom={denom:.3f}): "
            f"falling back to R̂*_iso = {r_iso:.4f}"
        )
        return r_iso, {
            "rho0": rho0, "rho1": rho1, "denom": denom,
            "stretch": stretch, "fallback_to_iso": True,
        }
    p_corr = np.clip((c_iso - rho0) / denom, 0.0, 1.0)
    r_fci = float(np.minimum(p_corr, 1.0 - p_corr).mean())
    return r_fci, {
        "rho0": rho0, "rho1": rho1, "denom": denom,
        "stretch": stretch, "fallback_to_iso": False,
    }


def per_judge(judge_dir):
    files = sorted(Path(judge_dir).glob("*.json"))
    rows = []
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        pairs = [p for p in d["pairs"] if p.get("p_a_mean") is not None]
        p_a = np.array([p["p_a_mean"] for p in pairs])
        gold = np.array([p["gold"] for p in pairs], dtype=int)
        ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        ir.fit(p_a, gold)
        c_iso = ir.predict(p_a)
        r_iso = float(estimate_rstar(c_iso))
        r_fci_mm, info_mm = fci_risk(c_iso, "minmax")
        r_fci_q,  info_q  = fci_risk(c_iso, "q01_q99")
        acc = float(((p_a >= 0.5).astype(int) == gold).mean())
        rows.append({
            "judge": d["model"],
            "n": int(len(p_a)),
            "acc_vs_gold": acc,
            "rstar_iso": r_iso,
            "rstar_fci_minmax": r_fci_mm,
            "rstar_fci_q01_q99": r_fci_q,
            "minmax_anchors": [info_mm["rho0"], info_mm["rho1"]],
            "minmax_stretch": info_mm["stretch"],
            "minmax_fallback": info_mm["fallback_to_iso"],
            "q01_q99_anchors": [info_q["rho0"], info_q["rho1"]],
            "q01_q99_stretch": info_q["stretch"],
            "q01_q99_fallback": info_q["fallback_to_iso"],
        })
    return rows


def summarise(tag, rows):
    r_iso = np.array([r["rstar_iso"] for r in rows])
    r_mm  = np.array([r["rstar_fci_minmax"] for r in rows])
    r_q   = np.array([r["rstar_fci_q01_q99"] for r in rows])
    s = {
        "dataset": tag,
        "rows": rows,
        "iso_mean": float(r_iso.mean()),
        "iso_std":  float(r_iso.std()),
        "fci_mm_mean": float(r_mm.mean()),
        "fci_mm_std":  float(r_mm.std()),
        "fci_q_mean": float(r_q.mean()),
        "fci_q_std":  float(r_q.std()),
    }
    print(f"\n=== {tag} ===")
    for r in rows:
        print(f"  {r['judge']:<30s}  acc={r['acc_vs_gold']:.3f}  "
              f"iso={r['rstar_iso']:.3f}  fci_mm={r['rstar_fci_minmax']:.3f}  "
              f"fci_q={r['rstar_fci_q01_q99']:.3f}  "
              f"s_mm={r['minmax_stretch']:.2f}  s_q={r['q01_q99_stretch']:.2f}")
    print(f"  mean  iso={s['iso_mean']:.3f}  fci_mm={s['fci_mm_mean']:.3f}  fci_q={s['fci_q_mean']:.3f}")
    print(f"  std   iso={s['iso_std']:.3f}  fci_mm={s['fci_mm_std']:.3f}  fci_q={s['fci_q_std']:.3f}")
    return s


hs2_rows = per_judge(REPO / "experiments" / "judges_hs2")
uf_rows  = per_judge(REPO / "experiments" / "judges_uf")
hs2 = summarise("HS2", hs2_rows)
uf  = summarise("UF",  uf_rows)

drift_iso = abs(hs2["iso_mean"] - uf["iso_mean"])
drift_mm  = abs(hs2["fci_mm_mean"] - uf["fci_mm_mean"])
drift_q   = abs(hs2["fci_q_mean"]  - uf["fci_q_mean"])

std_ratio_hs2 = hs2["fci_q_std"] / max(hs2["iso_std"], 1e-9)
std_ratio_uf  = uf["fci_q_std"]  / max(uf["iso_std"],  1e-9)
in_range = all(0.02 < r["rstar_fci_q01_q99"] < 0.55 for r in hs2_rows + uf_rows)

pass_a = drift_q < 0.09
pass_b = min(std_ratio_hs2, std_ratio_uf) < 0.85
pass_c = in_range
pass_ = bool(pass_a and pass_b and pass_c)

print(f"\n=== Head-to-head drift (lower = better) ===")
print(f"  iso          drift = {drift_iso:.4f}")
print(f"  fci_minmax   drift = {drift_mm:.4f}")
print(f"  fci_q01_q99  drift = {drift_q:.4f}")
print(f"  std ratio FCI_q/iso:  HS2 {std_ratio_hs2:.3f}   UF {std_ratio_uf:.3f}")
print(f"  PASS (a) drift < 0.09           : {pass_a}   ({drift_q:.4f})")
print(f"  PASS (b) std ratio < 0.85       : {pass_b}   (HS2 {std_ratio_hs2:.3f}, UF {std_ratio_uf:.3f})")
print(f"  PASS (c) all R̂*_FCI in (0.02, 0.55): {pass_c}")
print(f"  OVERALL PASS = {pass_}")

stats = {
    "hs2": hs2, "uf": uf,
    "drift": {
        "iso": float(drift_iso),
        "fci_minmax": float(drift_mm),
        "fci_q01_q99": float(drift_q),
        "p13_naive_consensus_reference": 0.146,
    },
    "std_ratio_fci_q_over_iso": {"hs2": float(std_ratio_hs2), "uf": float(std_ratio_uf)},
    "go_no_go": {
        "criterion_a_drift_q_lt_0.09":       {"value": float(drift_q),         "pass": pass_a},
        "criterion_b_std_ratio_lt_0.85":     {"hs2": float(std_ratio_hs2), "uf": float(std_ratio_uf), "pass": pass_b},
        "criterion_c_all_in_range_0.02_0.55": {"pass": pass_c},
        "overall_pass": pass_,
    },
    "framing": (
        "FCI-Risk is a *heuristic precursor* to the Year-1 margin-matching "
        "theorem. Unlike the P0 V4 formula which collapsed to R̂*_iso (P14), "
        "FCI-Risk pushes an affine correction inside the non-linear min() and "
        "therefore cannot be algebraically reduced to isotonic. Theoretical "
        "lineage: Patrini et al. 2017 forward-correction, Liu & Tao 2015 "
        "anchor points. Year-1 must derive the M^{-1/3} boundary rate and "
        "finite-sample bias bound that this estimator family satisfies "
        "under BT; this script only checks empirical feasibility."
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
C_ISO = "#D55E00"; C_MM = "#0072B2"; C_Q = "#009E73"; C_REF = "#888"

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
axL, axR = axes

# LEFT: HS2 per-judge grouped bars
names = [r["judge"] for r in hs2_rows]
short = [n.split("-")[0] for n in names]
x = np.arange(len(names))
w = 0.27
axL.bar(x - w, [r["rstar_iso"]          for r in hs2_rows], w,
        color=C_ISO, edgecolor="black", linewidth=0.5,
        label=r"$\hat{R}^*_{\mathrm{iso}}$")
axL.bar(x,     [r["rstar_fci_minmax"]   for r in hs2_rows], w,
        color=C_MM,  edgecolor="black", linewidth=0.5,
        label=r"$\hat{R}^*_{\mathrm{FCI,\ min/max}}$")
axL.bar(x + w, [r["rstar_fci_q01_q99"] for r in hs2_rows], w,
        color=C_Q,   edgecolor="black", linewidth=0.5,
        label=r"$\hat{R}^*_{\mathrm{FCI,\ q01/q99}}$")
axL.axhline(0.49, color=C_REF, linestyle=":", linewidth=0.9,
            label=r"null floor $\approx 0.49$ (P11)")
axL.set_xticks(x)
axL.set_xticklabels(short, rotation=20, ha="right")
axL.set_ylabel(r"$\hat{R}^*$")
axL.set_title(fr"A. HS2 per-judge (std ratio FCI_q/iso = {std_ratio_hs2:.2f})")
axL.legend(loc="best", ncol=2, handlelength=1.4)

# RIGHT: drift bars across methods
methods = [r"naive consensus" "\n" "(P13)", r"$\hat{R}^*_{\mathrm{iso}}$",
           r"FCI min/max", r"FCI q01/q99"]
drifts = [0.146, drift_iso, drift_mm, drift_q]
colors = [C_REF, C_ISO, C_MM, C_Q]
axR.bar(methods, drifts, color=colors, edgecolor="black", linewidth=0.5)
for i, v in enumerate(drifts):
    axR.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=9)
axR.axhline(0.09, color="#D55E00", linestyle="--", linewidth=0.9,
            label="PASS threshold 0.09")
axR.set_ylabel(r"$|\mathrm{mean}_{\mathrm{HS2}} - \mathrm{mean}_{\mathrm{UF}}|$")
axR.set_title("B. Cross-dataset drift")
axR.legend(loc="upper right")

fig.suptitle(
    fr"P15 — FCI-Risk: non-tautological $\hat{{R}}^*_{{\mathrm{{CA}}}}$ candidate  "
    fr"(drift q01/q99 = {drift_q:.3f}, std ratio HS2 {std_ratio_hs2:.2f})  "
    f"[{'PASS' if pass_ else 'FAIL'}]",
    fontsize=10.5, y=1.03,
)
out = HERE / "fig_fci.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print(f"\nSaved: {out}")
