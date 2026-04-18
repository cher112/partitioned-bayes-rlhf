"""P14 — Apply the P0 V4 `R̂*_CA` formula on real LLM-judge data.

!!!  CRITICAL NEGATIVE RESULT  !!!
This script discovered that the P0 V4 `R̂*_CA` formula

    R̂*_CA  =  R̂*_raw  +  (E[|η̂ − 1/2|]  −  E[|η_iso − 1/2|])

is algebraically identical to `R̂*_iso`:

    R̂*_raw   = E[min(p, 1-p)] = 0.5 − E[|p − 0.5|] = 0.5 − m_hat
    R̂*_iso   = E[min(c_iso, 1-c_iso)] = 0.5 − m_iso
    R̂*_CA    = R̂*_raw + (m_hat − m_iso) = 0.5 − m_hat + m_hat − m_iso
             = 0.5 − m_iso
             = R̂*_iso   ∎

On HS2 / UF, every per-judge R̂*_CA matches the per-judge R̂*_iso to
machine precision (std ratio 1.000, drift identical). The synthetic
P0 V4 "R̂*_CA pins R* within 0.001" result was not a new correction;
it was merely re-verifying that isotonic calibration works in the
monotone-coupled regime.

Implications for the proposal:
  - C4's claim — "R̂*_CA pins true R* in the monotone regime" — is TRUE
    but uninformative: it is R̂*_iso. The separate "R̂*_CA" label in
    earlier writeups was a bookkeeping error, not a new estimator.
  - Year-1 margin-matching MUST be an asymmetric / signed-bias-based
    correction that is not collapsible to isotonic. A concrete
    candidate: `R̂*_CA = R̂*_iso + f(signed_bias)` where f handles
    over- and under-confidence asymmetrically, so the formula does
    not simplify back to `0.5 − m_iso`.

This script is kept as a CAUTIONARY-NEGATIVE-RESULT: the heuristic
correction must be redesigned for Year-1, and the synthetic-only
"proof of concept" in P0 V4 was a tautology masquerading as a fix.
"""

On the toy, R̂*_CA pinned the truth within 0.001 across 10 temperatures
(23× plug-in swing, Fig. 3). This script evaluates the same formula
per-judge on real HS2 + UF data. No new estimator — just a head-to-head
evaluation of the already-proposed Year-1 candidate on real preference
data.

The experiment is a preliminary feasibility check, not the Year-1
deliverable. The Year-1 theorem is about finite-sample bias bounds
under BT; here we only ask whether the heuristic formula already
reduces the cross-judge spread in practice. If it does, the Year-1
theory has an empirical anchor; if it does not, Year-1 must add
something beyond the simple margin-gap substitution.

Gate-1 considerations (2026-04-18):
  - Not tautological: we never enforce R̂*_CA to match across judges.
    Each judge's formula uses only its own raw/iso outputs.
  - Not a "new idea": it is literally the R̂*_CA from P0 V4 on real data.

Go / No-Go:
  PASS if
    (a) cross-judge std(R̂*_CA) < 0.7 × cross-judge std(R̂*_iso), AND
    (b) HS2↔UF drift of mean(R̂*_CA) < HS2↔UF drift of mean(R̂*_iso), AND
    (c) R̂*_CA values stay in (0.05, 0.55) (no degenerate collapse).
  NO-GO if the formula produces values outside (0, 0.5) or std ratio > 1.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
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


def per_judge_rstars(judge_dir):
    files = sorted(Path(judge_dir).glob("*.json"))
    rows = []
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        pairs = [p for p in d["pairs"] if p.get("p_a_mean") is not None]
        p_a = np.array([p["p_a_mean"] for p in pairs])
        gold = np.array([p["gold"] for p in pairs], dtype=int)
        # Isotonic from scratch
        ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        ir.fit(p_a, gold)
        c_iso = ir.predict(p_a)
        # Plug-in estimators
        r_raw = float(np.minimum(p_a, 1 - p_a).mean())
        r_iso = float(np.minimum(c_iso, 1 - c_iso).mean())
        # Margin-gap identity for R̂*_CA
        margin_hat = float(np.abs(p_a - 0.5).mean())
        margin_iso = float(np.abs(c_iso - 0.5).mean())
        r_ca = r_raw + (margin_hat - margin_iso)
        rows.append({
            "judge": d["model"],
            "n": int(len(p_a)),
            "r_raw": r_raw,
            "r_iso": r_iso,
            "margin_hat": margin_hat,
            "margin_iso": margin_iso,
            "r_ca": r_ca,
        })
    return rows


def summarise(tag, rows):
    r_raw = np.array([r["r_raw"] for r in rows])
    r_iso = np.array([r["r_iso"] for r in rows])
    r_ca  = np.array([r["r_ca"]  for r in rows])
    print(f"\n=== {tag} ===")
    for r in rows:
        print(f"  {r['judge']:<30s}  "
              f"raw={r['r_raw']:.3f}  iso={r['r_iso']:.3f}  "
              f"m_hat={r['margin_hat']:.3f}  m_iso={r['margin_iso']:.3f}  "
              f"CA={r['r_ca']:.3f}")
    return {
        "dataset": tag,
        "rows": rows,
        "mean_r_iso": float(r_iso.mean()),
        "mean_r_ca":  float(r_ca.mean()),
        "std_r_iso":  float(r_iso.std()),
        "std_r_ca":   float(r_ca.std()),
        "span_r_iso": float(r_iso.max() - r_iso.min()),
        "span_r_ca":  float(r_ca.max()  - r_ca.min()),
    }


hs2_rows = per_judge_rstars(REPO / "experiments" / "judges_hs2")
uf_rows  = per_judge_rstars(REPO / "experiments" / "judges_uf")
hs2 = summarise("HS2", hs2_rows)
uf  = summarise("UF",  uf_rows)

# Decision
drift_iso = abs(hs2["mean_r_iso"] - uf["mean_r_iso"])
drift_ca  = abs(hs2["mean_r_ca"]  - uf["mean_r_ca"])
std_ratio_hs2 = hs2["std_r_ca"] / max(hs2["std_r_iso"], 1e-9)
std_ratio_uf  = uf["std_r_ca"]  / max(uf["std_r_iso"],  1e-9)
in_range = all(0.05 < r["r_ca"] < 0.55 for r in hs2_rows + uf_rows)

pass_spread = std_ratio_hs2 < 0.7 and std_ratio_uf < 0.7
pass_drift  = drift_ca < drift_iso
pass_range  = in_range
pass_ = bool(pass_spread and pass_drift and pass_range)

print(f"\n=== Head-to-head ===")
print(f"  HS2  mean: iso={hs2['mean_r_iso']:.3f}  CA={hs2['mean_r_ca']:.3f}")
print(f"  UF   mean: iso={uf['mean_r_iso']:.3f}   CA={uf['mean_r_ca']:.3f}")
print(f"  Cross-dataset drift: iso {drift_iso:.4f}  CA {drift_ca:.4f}")
print(f"  HS2 cross-judge std: iso {hs2['std_r_iso']:.4f}  CA {hs2['std_r_ca']:.4f}  ratio {std_ratio_hs2:.3f}")
print(f"  UF  cross-judge std: iso {uf['std_r_iso']:.4f}   CA {uf['std_r_ca']:.4f}   ratio {std_ratio_uf:.3f}")
print(f"  All R̂*_CA in (0.05, 0.55): {in_range}")
print(f"  PASS = {pass_} (spread {pass_spread}, drift {pass_drift}, range {pass_range})")

stats = {
    "hs2": hs2, "uf": uf,
    "drift_iso": float(drift_iso),
    "drift_ca": float(drift_ca),
    "drift_improvement": float(drift_iso - drift_ca),
    "std_ratio_ca_over_iso": {"hs2": float(std_ratio_hs2), "uf": float(std_ratio_uf)},
    "pass_criterion": ("cross-judge std(CA)/std(iso) < 0.7 on both datasets "
                       "AND cross-dataset drift reduced AND all R̂*_CA in (0.05, 0.55)"),
    "pass": pass_,
    "framing": (
        "Direct evaluation on real data of the Year-1 candidate R̂*_CA "
        "formula derived in P0 V4 (synthetic). NOT the Year-1 deliverable: "
        "Year-1 is the finite-sample bias bound theorem under BT. This "
        "script only checks whether the heuristic formula already works "
        "in practice. If so, Year-1 theory has an empirical anchor; if "
        "not, Year-1 must add more than the simple margin-gap substitution."
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
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
axL, axR = axes

# LEFT: per-judge R̂*_iso vs R̂*_CA (HS2)
names = [r["judge"] for r in hs2_rows]
short = [n.split("-")[0] for n in names]
x = np.arange(len(names))
w = 0.38
axL.bar(x - w/2, [r["r_iso"] for r in hs2_rows], w,
        color="#D55E00", edgecolor="black", linewidth=0.5,
        label=r"HS2 $\hat{R}^*_{\mathrm{iso}}$")
axL.bar(x + w/2, [r["r_ca"]  for r in hs2_rows], w,
        color="#0072B2", edgecolor="black", linewidth=0.5,
        label=r"HS2 $\hat{R}^*_{\mathrm{CA}}$ (margin-matched)")
axL.axhline(hs2["mean_r_ca"], color="#0072B2", linestyle="--",
            linewidth=0.9,
            label=fr"HS2 mean $\hat{{R}}^*_{{\mathrm{{CA}}}}$ = {hs2['mean_r_ca']:.3f}")
axL.set_xticks(x)
axL.set_xticklabels(short, rotation=20, ha="right")
axL.set_ylabel(r"Bayes-error estimate")
axL.set_title(fr"A. Per-judge iso vs CA (HS2):  "
              fr"std ratio {std_ratio_hs2:.2f}")
axL.legend(loc="best", handlelength=1.4)

# RIGHT: cross-dataset drift comparison
methods = ["mean $\\hat{R}^*_{\\mathrm{iso}}$", "mean $\\hat{R}^*_{\\mathrm{CA}}$"]
drifts = [drift_iso, drift_ca]
colors = ["#D55E00", "#0072B2"]
axR.bar(methods, drifts, color=colors, edgecolor="black", linewidth=0.5)
for i, v in enumerate(drifts):
    axR.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)
axR.set_ylabel(r"$|\mathrm{mean}_{\mathrm{HS2}} - \mathrm{mean}_{\mathrm{UF}}|$")
axR.set_title(r"B. Cross-dataset drift (lower = better)")

fig.suptitle(
    fr"P14 — Margin-matching $\hat{{R}}^*_{{\mathrm{{CA}}}}$ on real data: "
    fr"drift {drift_iso:.3f} → {drift_ca:.3f}  "
    fr"std(iso/CA) HS2 {std_ratio_hs2:.2f}  UF {std_ratio_uf:.2f}  "
    f"[{'PASS' if pass_ else 'FAIL'}]",
    fontsize=10.5, y=1.03,
)
out = HERE / "fig_rstar_ca_real.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print(f"\nSaved: {out}")
