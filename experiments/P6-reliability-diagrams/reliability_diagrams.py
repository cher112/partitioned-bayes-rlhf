"""P6: Per-judge reliability diagrams (before and after isotonic calibration).

Shows how LLM judge softmax is systematically mis-calibrated. For each judge:
  - 5x2 grid of reliability plots (raw / isotonic after CV)
  - Each plot bins predicted confidence (max(p, 1-p)) into 15 equal-width bins
  - Y-axis = empirical accuracy per bin
  - Bar widths = bin mass

Well-calibrated classifier → points on y = x dashed line.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

HERE = Path(__file__).resolve().parent
JUDGE_DIR = REPO / "experiments" / "judges_hs2"

JUDGES = {
    "Qwen2.5-7B":      "qwen.json",
    "Mistral-7B":      "mistral.json",
    "Granite-3.0-8B":  "granite.json",
    "OLMo-7B-hf":      "olmo.json",
    "Falcon-7B":       "falcon.json",
}


def isotonic_cv(p, y, n_folds=5, seed=42):
    out = np.empty_like(p)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in kf.split(p):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(p[tr], y[tr])
        out[te] = ir.transform(p[te])
    return out


def reliability(p, y, n_bins=15):
    """Return (bin_centers, bin_acc, bin_conf, bin_count) for the diagonal diagram."""
    preds = (p >= 0.5).astype(int)
    conf = np.maximum(p, 1 - p)
    correct = (preds == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    accs, confs, counts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum():
            accs.append(correct[m].mean())
            confs.append(conf[m].mean())
            counts.append(int(m.sum()))
        else:
            accs.append(np.nan); confs.append(np.nan); counts.append(0)
    return centers, np.array(accs), np.array(confs), np.array(counts)


def compute_ece(p, y, n_bins=15):
    _, accs, confs, counts = reliability(p, y, n_bins)
    N = counts.sum()
    mask = counts > 0
    return float(np.sum(counts[mask] / N * np.abs(accs[mask] - confs[mask])))


# ---- Run ----
fig, axes = plt.subplots(2, 5, figsize=(19, 7.5), sharex=True, sharey=True)

for col, (short, fname) in enumerate(JUDGES.items()):
    d = json.load(open(JUDGE_DIR / fname))
    valid = [p for p in d["pairs"] if p.get("p_a_mean") is not None and p.get("gold") in (0, 1)]
    p_raw = np.array([x["p_a_mean"] for x in valid])
    y = np.array([x["gold"] for x in valid])

    p_iso = isotonic_cv(p_raw, y)

    for row, (p_use, label, color) in enumerate([
        (p_raw, "Raw softmax", "#c05621"),
        (p_iso, "Isotonic-CV (5-fold)", "#2b6cb0"),
    ]):
        ax = axes[row, col]
        centers, accs, confs, counts = reliability(p_use, y, n_bins=15)
        # Perfect diagonal
        ax.plot([0.5, 1], [0.5, 1], "k--", alpha=0.4, label="perfect" if col == 0 else None)
        # Bars: bin confidence vs accuracy
        mask = counts > 0
        widths = counts[mask] / counts.sum() * 0.5   # scale bar width to bin mass
        ax.bar(confs[mask], accs[mask], width=widths, alpha=0.6, color=color,
               edgecolor="black", linewidth=0.6,
               label=f"{label} (ECE={compute_ece(p_use, y):.3f})")
        ax.fill_between([0.5, 1], [0.5, 1], color="gray", alpha=0.05)

        ax.set_xlim(0.5, 1.0)
        ax.set_ylim(0.4, 1.0)
        if row == 0:
            ax.set_title(short, fontsize=10)
        if col == 0:
            ax.set_ylabel(f"{'raw' if row == 0 else 'isotonic'}\nEmpirical accuracy", fontsize=9)
        if row == 1:
            ax.set_xlabel("Predicted confidence max(p, 1-p)", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)

fig.suptitle(
    "P6 · Per-judge reliability diagrams (top: raw softmax, bottom: isotonic-calibrated)",
    fontsize=12, y=1.00,
)
fig.tight_layout()
fig.savefig(HERE / "fig_reliability_diagrams.png", dpi=150, bbox_inches="tight")

# Save ECE table
ece_rows = []
for short, fname in JUDGES.items():
    d = json.load(open(JUDGE_DIR / fname))
    valid = [p for p in d["pairs"] if p.get("p_a_mean") is not None and p.get("gold") in (0, 1)]
    p_raw = np.array([x["p_a_mean"] for x in valid])
    y = np.array([x["gold"] for x in valid])
    p_iso = isotonic_cv(p_raw, y)
    ece_rows.append({
        "judge": short,
        "n_pairs": int(len(valid)),
        "ece_raw": compute_ece(p_raw, y),
        "ece_iso": compute_ece(p_iso, y),
        "ece_reduction_pct": 100 * (1 - compute_ece(p_iso, y) / max(compute_ece(p_raw, y), 1e-6)),
    })

with open(HERE / "ece_per_judge.json", "w") as f:
    json.dump({
        "n_bins": 15,
        "cv_folds": 5,
        "seed": 42,
        "per_judge": ece_rows,
    }, f, indent=2)

print("ECE reduction via isotonic CV:")
for r in ece_rows:
    print(f"  {r['judge']:>16}: ECE raw = {r['ece_raw']:.4f}  "
          f"→ iso = {r['ece_iso']:.4f}  "
          f"({r['ece_reduction_pct']:+.1f}% reduction)")

print(f"\nSaved {HERE / 'fig_reliability_diagrams.png'}")
print(f"Saved {HERE / 'ece_per_judge.json'}")
