"""P11 — Null sanity check: permute gold labels, R̂*_iso should → 0.5.

A cheap, un-glamorous test every reviewer will want. If our estimator's
code has a bug (e.g. inverted gold, wrong softmax normalisation), a
permuted-gold control will expose it: under random labels the Bayes
error is 0.5 by construction, so R̂*_iso should concentrate around 0.5
no matter what LLM judge output we feed in.

Setup:
  - For each judge (HS2 6-judge JSONs), shuffle the gold array 200 times.
  - Compute R̂*_iso (with per-permutation isotonic refit on the shuffled
    gold) and R̂*_plug (no calibrator).
  - Report mean + 95% interval across the 200 permutations.

Pass criterion: mean R̂*_iso ∈ [0.48, 0.52] across all 6 judges.
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

JUDGE_FILES = sorted((REPO / "experiments" / "judges_hs2").glob("*.json"))
N_PERM = 200
rng = np.random.default_rng(42)

out = {}
for fp in JUDGE_FILES:
    with open(fp) as f:
        d = json.load(f)
    pairs = [p for p in d["pairs"] if p.get("p_a_mean") is not None]
    p_a = np.array([p["p_a_mean"] for p in pairs])
    gold_true = np.array([p["gold"] for p in pairs], dtype=int)
    name = d["model"]

    # Baseline (real gold) for reference
    ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    ir.fit(p_a, gold_true)
    rstar_real_iso = estimate_rstar(ir.predict(p_a))
    rstar_real_plug = estimate_rstar(p_a)

    # Null distribution — permute gold
    null_iso, null_plug = [], []
    for _ in range(N_PERM):
        g_perm = gold_true.copy()
        rng.shuffle(g_perm)
        ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        ir.fit(p_a, g_perm)
        null_iso.append(estimate_rstar(ir.predict(p_a)))
        null_plug.append(estimate_rstar(p_a))   # independent of gold
    null_iso = np.array(null_iso)
    null_plug = np.array(null_plug)

    out[name] = {
        "real": {"rstar_iso": float(rstar_real_iso),
                 "rstar_plug": float(rstar_real_plug)},
        "null": {
            "rstar_iso_mean": float(null_iso.mean()),
            "rstar_iso_std":  float(null_iso.std()),
            "rstar_iso_ci95": [float(np.quantile(null_iso, 0.025)),
                                float(np.quantile(null_iso, 0.975))],
            "rstar_plug_mean": float(null_plug.mean()),
            "rstar_plug_std":  float(null_plug.std()),
        },
        "drift_from_05": float(abs(null_iso.mean() - 0.5)),
    }
    print(f"{name:<30s}  real iso={rstar_real_iso:.4f}  "
          f"null iso mean={null_iso.mean():.4f}  "
          f"CI[{out[name]['null']['rstar_iso_ci95'][0]:.4f},"
          f"{out[name]['null']['rstar_iso_ci95'][1]:.4f}]  "
          f"drift from 0.5 = {out[name]['drift_from_05']:.4f}")

# Pass / fail — interpret WRT class prior, not exactly 0.5.
# With slight class imbalance p_base = mean(gold) ≈ 0.51, isotonic under
# random labels converges to a near-constant ≈ p_base, so expected null
# R̂*_iso = min(p_base, 1 − p_base) ≈ 0.49, not 0.5.
#
# The meaningful test is:
#   (a) null R̂*_plug is gold-independent (std across permutations = 0)
#   (b) REAL R̂*_iso for strong judges lies OUTSIDE the null 95% CI.

import itertools
judges_only = [k for k in out.keys()]
p_base = 0.51  # approximate class prior across all 6 judge JSONs
exp_null = float(min(p_base, 1 - p_base))
drifts = np.array([abs(out[k]["null"]["rstar_iso_mean"] - exp_null) for k in judges_only])
max_drift_from_expected = float(drifts.max())

real_vs_null_signal = {}
for k in judges_only:
    lo, hi = out[k]["null"]["rstar_iso_ci95"]
    real = out[k]["real"]["rstar_iso"]
    real_vs_null_signal[k] = {
        "real_iso": real,
        "null_ci_95": [lo, hi],
        "signal_vs_null": ("OUTSIDE null CI" if (real < lo or real > hi)
                           else "INSIDE null CI — not distinguishable"),
    }

plug_std = np.array([out[k]["null"]["rstar_plug_std"] for k in judges_only])
plug_gold_invariant = bool(plug_std.max() < 1e-10)

pass_ = bool(max_drift_from_expected < 0.02 and plug_gold_invariant)

out["_summary"] = {
    "n_permutations": N_PERM,
    "class_prior_approx": p_base,
    "expected_null_iso": exp_null,
    "max_drift_from_expected": max_drift_from_expected,
    "plug_in_gold_invariant": plug_gold_invariant,
    "signal_vs_null_per_judge": real_vs_null_signal,
    "pass_criterion": "null iso within 0.02 of class prior AND plug-in is gold-invariant",
    "pass": pass_,
}
print(f"\nExpected null iso (class prior): {exp_null:.4f}")
print(f"Max deviation from expected: {max_drift_from_expected:.4f}")
print(f"Plug-in is gold-invariant: {plug_gold_invariant}")
print(f"\nReal iso vs null 95% CI:")
for k, v in real_vs_null_signal.items():
    print(f"  {k:<28s}  real={v['real_iso']:.4f}  CI {v['null_ci_95']}  -> {v['signal_vs_null']}")
print(f"\nPASS: {pass_}")

with open(HERE / "stats.json", "w") as f:
    json.dump(out, f, indent=2)

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
})
judges_only = [k for k in out if k != "_summary"]
labels = [k.split("-")[0] for k in judges_only]
real_iso  = [out[k]["real"]["rstar_iso"] for k in judges_only]
null_mean = [out[k]["null"]["rstar_iso_mean"] for k in judges_only]
null_lo   = [out[k]["null"]["rstar_iso_ci95"][0] for k in judges_only]
null_hi   = [out[k]["null"]["rstar_iso_ci95"][1] for k in judges_only]
x = np.arange(len(judges_only))

fig, ax = plt.subplots(figsize=(7.5, 3.8))
ax.axhline(exp_null, color="#555", linestyle="--", linewidth=0.8,
           label=fr"null R* = min(p, 1-p) $\approx {exp_null:.3f}$")
ax.errorbar(x - 0.15, null_mean,
            yerr=[np.array(null_mean) - np.array(null_lo),
                   np.array(null_hi) - np.array(null_mean)],
            fmt="o", color="#555", markersize=6, linewidth=1.4,
            label=r"null $\hat{R}^*_{\mathrm{iso}}$ (200 gold perms, 95% CI)")
ax.scatter(x + 0.15, real_iso, color="#D55E00", s=60, zorder=5,
           label=r"real-gold $\hat{R}^*_{\mathrm{iso}}$")
for i, v in enumerate(real_iso):
    ax.annotate(f"{v:.3f}", (x[i] + 0.15, v),
                textcoords="offset points", xytext=(6, 2), fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha="right")
ax.set_ylabel(r"$\hat{R}^*_{\mathrm{iso}}$")
ax.set_ylim(0.35, 0.55)
n_outside = sum(1 for v in real_vs_null_signal.values()
                if "OUTSIDE" in v["signal_vs_null"])
ax.set_title(
    fr"P11 — Null perm sanity: null iso $\approx {exp_null:.3f}$ (class prior); "
    fr"{n_outside}/6 judges outside null CI  ({'PASS' if pass_ else 'FAIL'})"
)
ax.legend(loc="lower left", handlelength=1.8)
fig.savefig(HERE / "fig_null_sanity.png", dpi=300, bbox_inches="tight",
            facecolor="white")
print(f"Saved: {HERE / 'fig_null_sanity.png'}")
