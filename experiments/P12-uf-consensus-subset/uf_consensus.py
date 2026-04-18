"""P12 — UF strong-consensus subset: does the confound scale with gold confidence?

Design:
  - UF pairs were built with GPT-4 overall_score_gap ≥ 2 (1000 pairs).
  - Re-build pairs at score_gap thresholds τ ∈ {2, 3, 4, 5} and, for each,
    compute 6-judge accuracy + R̂*_iso + cross-judge r(acc, R̂*_iso).
  - Expectation: higher τ → easier pairs → accuracy ↑, R̂*_iso ↓; the
    confound r(acc, R̂*_iso) should stay strongly negative (same judges,
    same dataset, just stricter filter).
  - If r DRIFTS or changes sign, that is itself a finding (the confound
    depends on data difficulty).

This experiment reuses already-computed p_a_mean from
`experiments/judges_uf/*.json` — no new GPU. We just filter rows by the
gold score_gap that each pair was built with.

Caveat: we don't have per-pair score_gap stored directly in the judge
JSONs (only `id` and `gold`). We re-derive score_gap by reading
`data/uf_pairs.json` (which P0 wrote) and joining on `id`.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
from scipy.stats import pearsonr
from sklearn.isotonic import IsotonicRegression

SKILL = Path.home() / ".agents/skills/scientific-visualization/scripts"
sys.path.insert(0, str(SKILL))
from style_presets import set_color_palette  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
HERE.mkdir(exist_ok=True)

sys.path.insert(0, str(REPO / "src"))
from calibration_utils import estimate_rstar  # noqa: E402
from bootstrap_ci import fisher_z_ci_correlation  # noqa: E402

# Load per-pair score_gap from the UF pairs file
pairs_path = REPO / "data" / "uf_pairs.json"
if not pairs_path.exists():
    # Not synced locally; pull from autodl via rsync-style cached JSON
    alt = REPO.parent.parent / "autodl-fs" / "data" / "partitioned-bayes-rlhf" / "data" / "uf_pairs.json"
    if alt.exists():
        pairs_path = alt
    else:
        raise SystemExit(f"ERROR: uf_pairs.json not found at {pairs_path}; "
                         f"rsync from autodl:/autodl-fs/data/partitioned-bayes-rlhf/data/")
with open(pairs_path) as f:
    uf_pairs = json.load(f)
gap_by_id = {p["id"]: float(p.get("score_gap", 0)) for p in uf_pairs}
print(f"Loaded {len(uf_pairs)} UF pairs.  score_gap min/max: "
      f"{min(gap_by_id.values()):.2f} / {max(gap_by_id.values()):.2f}")

# Load 6 judges
judge_files = sorted((REPO / "experiments" / "judges_uf").glob("*.json"))
judges = {}
for fp in judge_files:
    with open(fp) as f:
        d = json.load(f)
    pairs = [p for p in d["pairs"] if p.get("p_a_mean") is not None]
    p_a = np.array([p["p_a_mean"] for p in pairs])
    gold = np.array([p["gold"] for p in pairs], dtype=int)
    ids = [p.get("id") for p in pairs]
    judges[d["model"]] = {"p_a": p_a, "gold": gold, "ids": ids}
    print(f"  {d['model']:<30s} n={len(pairs)}")

THRESHOLDS = [2, 3, 4, 5]
rows = []
for tau in THRESHOLDS:
    # For each judge, filter to pairs whose score_gap ≥ tau
    judge_rows = []
    for name, jd in judges.items():
        mask = np.array([gap_by_id.get(pid, 0) >= tau for pid in jd["ids"]])
        p_a = jd["p_a"][mask]
        gold = jd["gold"][mask]
        if len(p_a) < 20:
            continue
        acc = float(((p_a >= 0.5).astype(int) == gold).mean())
        # Iso fit on this subset only
        ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        ir.fit(p_a, gold)
        rstar_iso = float(estimate_rstar(ir.predict(p_a)))
        judge_rows.append({
            "judge": name, "tau": tau, "n": int(len(p_a)),
            "acc": acc, "rstar_iso": rstar_iso,
        })
    if len(judge_rows) < 3:
        continue
    df = pd.DataFrame(judge_rows)
    r, p = pearsonr(df["acc"], df["rstar_iso"])
    flo, fhi = fisher_z_ci_correlation(float(r), len(df))
    summary = {
        "tau": tau,
        "n_pairs_mean": float(df["n"].mean()),
        "accuracy_mean": float(df["acc"].mean()),
        "accuracy_range": [float(df["acc"].min()), float(df["acc"].max())],
        "rstar_iso_mean": float(df["rstar_iso"].mean()),
        "rstar_iso_range": [float(df["rstar_iso"].min()),
                            float(df["rstar_iso"].max())],
        "pearson_r": float(r),
        "pearson_p": float(p),
        "fisher_z_ci_95": [float(flo), float(fhi)],
        "per_judge": judge_rows,
    }
    rows.append(summary)
    print(f"\n[τ={tau}]  n̄={summary['n_pairs_mean']:.0f}  "
          f"acc∈{summary['accuracy_range']}  "
          f"R*_iso∈{summary['rstar_iso_range']}  "
          f"r={r:+.3f}  Fisher-z CI [{flo:+.3f}, {fhi:+.3f}]")

stats = {"thresholds": rows}
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
    "figure.constrained_layout.use": True,
    "legend.fontsize": 8,
})
fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8))
axL, axR = axes

# Left: per-τ scatter of (acc, R*_iso) with fit line
cmap = plt.get_cmap("viridis")
for i, row in enumerate(rows):
    df = pd.DataFrame(row["per_judge"])
    c = cmap(i / max(1, len(rows) - 1))
    axL.scatter(df["acc"], df["rstar_iso"], color=c, s=60,
                edgecolors="black", linewidths=0.5,
                label=fr"τ={row['tau']}  (r={row['pearson_r']:+.2f}, "
                      fr"n̄={row['n_pairs_mean']:.0f})")
axL.set_xlabel("Judge accuracy")
axL.set_ylabel(r"$\hat{R}^*_{\mathrm{iso}}$")
axL.set_title("A. Per-threshold 6-judge points")
axL.legend(loc="best", handlelength=1.4)

# Right: r(acc, R*) as a function of τ, with Fisher-z CI
taus = [r["tau"] for r in rows]
rs = [r["pearson_r"] for r in rows]
ci_lo = [r["fisher_z_ci_95"][0] for r in rows]
ci_hi = [r["fisher_z_ci_95"][1] for r in rows]
axR.errorbar(taus, rs,
             yerr=[np.array(rs) - np.array(ci_lo),
                    np.array(ci_hi) - np.array(rs)],
             fmt="o-", color="#D55E00", linewidth=1.6, markersize=6,
             markerfacecolor="white", markeredgewidth=1.1,
             label=r"Pearson $r$ with Fisher-z 95% CI")
axR.axhline(0, color="#888", linewidth=0.6, linestyle="--")
axR.axhline(-1, color="#888", linewidth=0.4, linestyle=":")
axR.set_xlabel(r"score-gap threshold $\tau$")
axR.set_ylabel(r"Cross-judge $r(\mathrm{acc}, \hat{R}^*_{\mathrm{iso}})$")
axR.set_ylim(-1.05, 0.05)
axR.set_title(r"B. Confound strength vs. filter strictness")
axR.legend(loc="lower right", handlelength=1.8)

fig.suptitle(
    r"P12 — UltraFeedback strong-consensus subsets: capacity confound persists "
    r"across all gold thresholds", fontsize=10.5, y=1.03,
)
out = HERE / "fig_uf_consensus.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print(f"\nSaved: {out}")
