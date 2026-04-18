"""P7 — Llama-3-8B conditional-preference temperature scan on HS2.

Mathematical claim (handles the "are you really doing temperature scaling"
question upfront):

    Full-vocabulary temperature scaling scales every logit by 1/T:
        P(t | T) = softmax(l / T)[t] = exp(l_t/T) / Z(T)
    The *conditional* probability P(A | {A,B}) then simplifies:
        P(A|A∪B, T) = exp(l_A/T) / (exp(l_A/T) + exp(l_B/T))
                    = σ((l_A - l_B) / T)
    The partition function Z(T) cancels. So scanning T and reading off
    P(A|A∪B) via the binomial reduction IS full-vocab T-scaling on the
    conditional preference distribution — exact, not approximate.

Reuses 1000-pair Llama-3 HS2 JSON: each pair stores
p_a_ab = softmax([l_A, l_B])[0] from the vLLM top-20 logprobs. We
reconstruct Δ = logit(p_a_ab) = l_A - l_B and sweep T.

Per T we compute:
  - accuracy  (invariant under T — classifier's argmax is T-invariant)
  - plug-in R̂*  (T-sensitive, reflects calibration)
  - isotonic-calibrated R̂*  (T-insensitive under monotone calibration)
  - signed calibration bias  (E[p_a|y=1] - 0.5) + (E[1-p_a|y=0] - 0.5)

Finding (not replication): r(signed_bias, R̂*_plug_gap) = -0.908.
Synthetic P0 V3 has r = -1.00 by closed-form monotonicity; the ~9%
unexplained variance on real LLM output is itself a real-world finding,
meaning T-scaling alone cannot perfectly isolate calibration from
capacity on LLM preference judgments.
"""
import sys, json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
from scipy.stats import pearsonr
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

SKILL = Path.home() / ".agents/skills/scientific-visualization/scripts"
sys.path.insert(0, str(SKILL))
from style_presets import set_color_palette  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SRC_JSON = REPO / "experiments" / "judges_hs2" / "llama3.json"
HERE = Path(__file__).resolve().parent
HERE.mkdir(exist_ok=True)

# ---------- load + reconstruct logit gaps ----------
with open(SRC_JSON) as f:
    data = json.load(f)
pairs = [p for p in data["pairs"] if p.get("p_a_mean") is not None]
gold = np.array([p["gold"] for p in pairs], dtype=int)

def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

# Mean logit gap: average of AB and BA logit gaps (BA is stored as p_a_ba = P(A wins | BA order))
d_ab = logit(np.array([p["p_a_ab"] for p in pairs]))
d_ba = logit(np.array([p["p_a_ba"] for p in pairs]))
delta = 0.5 * (d_ab + d_ba)
print(f"n = {len(delta)}   mean|Δ| = {np.mean(np.abs(delta)):.3f}")

# ---------- temperature scan ----------
Ts = np.array([0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0])
records = []
for T in Ts:
    p_a = 1.0 / (1.0 + np.exp(-delta / T))
    acc = float((((p_a >= 0.5).astype(int) == gold)).mean())
    # plug-in R̂*
    rstar_plug = float(np.mean(np.minimum(p_a, 1 - p_a)))
    # isotonic with 5-fold CV (out-of-fold predictions)
    cal = np.zeros_like(p_a)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for tr, te in kf.split(p_a):
        ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        ir.fit(p_a[tr], gold[tr])
        cal[te] = ir.predict(p_a[te])
    rstar_iso = float(np.mean(np.minimum(cal, 1 - cal)))
    # signed calibration bias: how much p_a over- or under-shoots on each class
    sb = float((p_a[gold == 1].mean() - 0.5) - (p_a[gold == 0].mean() - 0.5))
    records.append({
        "T": float(T),
        "acc": acc,
        "rstar_plug": rstar_plug,
        "rstar_iso": rstar_iso,
        "signed_bias": sb,
    })

T_arr = np.array([r["T"] for r in records])
acc_arr = np.array([r["acc"] for r in records])
rplug = np.array([r["rstar_plug"] for r in records])
riso = np.array([r["rstar_iso"] for r in records])
sbias = np.array([r["signed_bias"] for r in records])

# canonical gap: how much plug-in deviates from the stable isotonic value
gap = rplug - np.mean(riso)
r_sb, p_sb = pearsonr(sbias, gap)

stats = {
    "n_pairs": int(len(delta)),
    "n_temperatures": int(len(Ts)),
    "temperatures": Ts.tolist(),
    "rstar_plug_span": float(rplug.max() - rplug.min()),
    "rstar_plug_range": [float(rplug.min()), float(rplug.max())],
    "rstar_iso_span": float(riso.max() - riso.min()),
    "rstar_iso_range": [float(riso.min()), float(riso.max())],
    "accuracy_span": float(acc_arr.max() - acc_arr.min()),
    "accuracy_mean": float(acc_arr.mean()),
    "pearson_signed_bias_vs_plug_gap": {
        "r": float(r_sb),
        "p_value": float(p_sb),
    },
    "per_T": records,
}
with open(HERE / "stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print(f"plug-in R* span = {stats['rstar_plug_span']:.4f} "
      f"(range [{stats['rstar_plug_range'][0]:.4f}, "
      f"{stats['rstar_plug_range'][1]:.4f}])")
print(f"isotonic R* span = {stats['rstar_iso_span']:.4f}")
print(f"accuracy span = {stats['accuracy_span']:.4f} (mean {stats['accuracy_mean']:.4f})")
print(f"r(signed_bias, plug-gap) = {r_sb:+.4f}, p = {p_sb:.2e}")

# ---------- plot ----------
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

C_PLUG = "#D55E00"
C_ISO = "#0072B2"
C_ACC = "#009E73"
C_REF = "#555555"

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6),
                         gridspec_kw={"width_ratios": [1.2, 1.0]})
axL, axR = axes

# LEFT: R̂* vs T, with accuracy as dashed reference
axL.plot(T_arr, rplug, marker="o", color=C_PLUG, linewidth=1.8,
         markersize=6, markeredgecolor="white", markeredgewidth=0.7,
         label=r"plug-in $\hat{R}^*$")
axL.plot(T_arr, riso, marker="s", color=C_ISO, linewidth=1.6,
         markersize=5, markeredgecolor="white", markeredgewidth=0.6,
         label=r"isotonic $\hat{R}^*$")
axL.axhline(acc_arr.mean(), color=C_ACC, linewidth=0.9, linestyle="--",
            label=fr"accuracy (flat, $={acc_arr.mean():.3f}$)")
axL.set_xscale("log")
axL.set_xlabel(r"Softmax temperature $T$")
axL.set_ylabel(r"Estimator value")
axL.set_title(fr"A. Llama-3-8B on HS2: plug-in $\hat{{R}}^*$ swings "
              fr"${rplug.min():.2f}\!\rightarrow\!{rplug.max():.2f}$")
axL.legend(loc="best", handlelength=1.8)

# RIGHT: signed-bias vs plug-gap (the P0 V3 r ≈ -1.00 story)
axR.scatter(sbias, gap, s=70, c=T_arr, cmap="viridis",
            edgecolors="black", linewidths=0.5)
for xi, yi, ti in zip(sbias, gap, T_arr):
    axR.annotate(f"T={ti}", (xi, yi), textcoords="offset points",
                 xytext=(5, 3), fontsize=7, color=C_REF)
axR.axhline(0, color=C_REF, linewidth=0.6, alpha=0.5)
axR.axvline(0, color=C_REF, linewidth=0.6, alpha=0.5)
axR.set_xlabel("Signed calibration bias")
axR.set_ylabel(r"$\hat{R}^*_{\mathrm{plug}}(T) - \overline{\hat{R}^*_{\mathrm{iso}}}$")
axR.set_title(fr"B. Pearson $r = {r_sb:+.3f}$  (p = {p_sb:.1e})")

fig.suptitle(
    r"P7 — Llama-3-8B on HS2 replicates the temperature-scaling capacity confound "
    fr"(plug-in span {stats['rstar_plug_span']:.3f}, acc span {stats['accuracy_span']:.3f})",
    fontsize=10.5, y=1.04,
)

out = HERE / "fig_tempscan.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
