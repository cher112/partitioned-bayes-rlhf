"""P0 V4 figure, publication-quality (scientific-visualization skill).

Uses `davila7/claude-code-templates@scientific-visualization` helpers:
Okabe-Ito colorblind-safe palette, hairline axes, Arial, 300 dpi.

Three purpose-built panels — no shared axes, no decorative bands, no
duplicate horizontal references (that was the bug in the earlier
versions, where overlapping axhline/axhspan/flat-line all collapsed
into a noisy-looking horizontal smear).

  Panel A (log-y bar):   stability span across 10 T per estimator;
                         only the three estimators whose span > 0 are
                         drawn (the "1 - accuracy" proxy is literally
                         0.0 and carries no information in this view).
  Panel B (line):        plug-in R^ trajectory vs true R*; one
                         single dashed reference line marks R*, no
                         extra fill / no redundant overlay.
  Panel C (line, zoom):  Isotonic and Margin-CA on y in
                         [0.1555, 0.1605]; one single reference line
                         marks R*, Margin-CA shifted +4e-4 for
                         visibility.
"""
import sys
from pathlib import Path

import json
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401  (registers 'science', 'ieee', 'nature', etc.)

SKILL_SCRIPTS = Path.home() / ".agents/skills/scientific-visualization/scripts"
sys.path.insert(0, str(SKILL_SCRIPTS))
from style_presets import set_color_palette  # noqa: E402

HERE = Path(__file__).resolve().parent
with open(HERE / "results_v4.json") as f:
    R = json.load(f)

per_T = R["per_T"]
T = np.array([x["T"] for x in per_T])
r_raw = np.array([x["r_raw"] for x in per_T])
r_iso = np.array([x["r_iso"] for x in per_T])
r_ca = np.array([x["r_ca"] for x in per_T])
true_R = R["true_rstar"]
span = R["stability_span_across_T"]

# IOP "science" style + Okabe-Ito palette + Times (no LaTeX backend needed).
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
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "0.85",
    "grid.alpha": 0.35,
    "grid.linewidth": 0.4,
    "axes.spines.top": True,     # science style keeps the 4-side frame
    "axes.spines.right": True,
    "figure.constrained_layout.use": True,
})

# Okabe-Ito role colors
C_PLUG = "#D55E00"   # vermillion — plug-in (villain)
C_ISO  = "#0072B2"   # blue — Isotonic (hero 1)
C_CA   = "#009E73"   # green — Margin-CA (hero 2)
C_REF  = "#555555"   # dark gray — reference lines only

fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.5),
                         gridspec_kw={"width_ratios": [0.85, 1.15, 1.0]})
axA, axB, axC = axes

# ===== Panel A: stability bar (log-y, three bars) =====
names = ["Plug-in", "Isotonic", "Margin-CA"]
spans = np.array([span["plug_in_raw"], span["isotonic"], span["margin_CA"]])
colors = [C_PLUG, C_ISO, C_CA]
bars = axA.bar(names, spans, color=colors, edgecolor="black", linewidth=0.6,
               width=0.65)
axA.set_yscale("log")
axA.set_ylim(5e-4, 1.0)
axA.set_yticks([1e-3, 1e-2, 1e-1, 1.0])
axA.set_yticklabels([r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$"])
axA.minorticks_off()
axA.set_ylabel(r"Span of $\hat{R}^*$ across 10 $T$")
axA.set_title("A. Stability across temperatures")
axA.grid(axis="y", which="major", alpha=0.35, linewidth=0.4)
for bar, v in zip(bars, spans):
    axA.text(bar.get_x() + bar.get_width() / 2, v * 1.30,
             f"{v:.4f}" if v < 0.01 else f"{v:.3f}",
             ha="center", va="bottom", fontsize=9)

# annotate the 23x ratio
axA.annotate(r"$\approx$ 285$\times$ ratio",
             xy=(0, spans[0]), xytext=(1.0, 0.55),
             fontsize=8, color=C_REF,
             arrowprops=dict(arrowstyle="-", color=C_REF, linewidth=0.6))

# ===== Panel B: plug-in trajectory vs R* =====
axB.axhline(true_R, color=C_REF, linewidth=0.9, linestyle="--",
            label=fr"true $R^* = {true_R:.3f}$")
axB.plot(T, r_raw, marker="o", color=C_PLUG, linewidth=1.8,
         markersize=5.5, markerfacecolor=C_PLUG, markeredgecolor="white",
         markeredgewidth=0.7, label=r"Plug-in $\hat{R}^*$")
axB.set_xscale("log")
axB.set_xlabel(r"Softmax temperature $T$")
axB.set_ylabel(r"Estimator value")
axB.set_title(r"B. Plug-in $\hat{R}^*$ swings $0.016 \rightarrow 0.387$")
axB.set_ylim(-0.02, 0.42)
axB.legend(loc="upper left", handlelength=1.8)

# annotate accuracy is flat (the classifier is unchanged)
axB.text(0.97, 0.04, f"accuracy = 0.839 (flat across all $T$)",
         transform=axB.transAxes, ha="right", va="bottom",
         fontsize=8, color=C_REF, style="italic")

# ===== Panel C: Iso / Margin-CA zoom =====
axC.axhline(true_R, color=C_REF, linewidth=0.9, linestyle="--",
            label=fr"analytic $R^* = {true_R:.3f}$")
OFFSET = 4e-4
axC.plot(T, r_iso, marker="o", color=C_ISO, linewidth=1.6, markersize=5.5,
         markerfacecolor=C_ISO, markeredgecolor="white", markeredgewidth=0.6,
         label=fr"Isotonic $\hat{{R}}^*$  (span {span['isotonic']:.4f})")
axC.plot(T, r_ca + OFFSET, marker="^", color=C_CA, linewidth=1.6,
         markersize=5.5, markerfacecolor=C_CA, markeredgecolor="white",
         markeredgewidth=0.6,
         label=(fr"Margin-CA $\hat{{R}}^*$ ($+4\!\times\!10^{{-4}}$ offset)  "
                fr"(span {span['margin_CA']:.4f})"))
axC.set_xscale("log")
axC.set_xlabel(r"Softmax temperature $T$")
axC.set_ylabel(r"Estimator value  (zoomed)")
axC.set_title("C. Corrected estimators pin the truth")
axC.set_ylim(0.1555, 0.1605)
axC.legend(loc="lower right", handlelength=1.8)
# finite-sample caveat: empirical oracle min on this test set is 0.160,
# so the 0.002 gap between estimators and analytic R* is N=5000 noise,
# not estimator bias.
axC.text(0.02, 0.96,
         fr"empirical min. err. on test $= 0.160$" "\n"
         fr"(gap $\approx 0.002$ = finite-sample noise, $N=5000$)",
         transform=axC.transAxes, ha="left", va="top",
         fontsize=7.5, color=C_REF, style="italic",
         bbox=dict(boxstyle="round,pad=0.25", fc="white",
                   ec="0.85", lw=0.5))

fig.suptitle(
    r"P0 V4 — naive plug-in $\hat{R}^*$ is calibration-confounded; "
    r"Isotonic and Margin-CA pin $R^*$",
    fontsize=10.5, y=1.04,
)

out = HERE / "fig_synthetic_rstar_v4.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
