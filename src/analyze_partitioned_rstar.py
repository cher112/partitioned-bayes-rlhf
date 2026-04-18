#!/usr/bin/env python3
"""Cross-partition R* analysis with permutation test + figure."""
import argparse, json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibration_utils import isotonic_calibrate_cv, estimate_rstar, compute_ece

def load_judge(fp):
    with open(fp) as f: data = json.load(f)
    return data["model"], [{"id": p["id"], "gold": p.get("gold"), "p_a": p["p_a_mean"]}
                           for p in data["pairs"] if p.get("p_a_mean") is not None]

def align_judges(judge_records):
    sets = [set(r["id"] for r in recs) for _, recs in judge_records]
    common = sorted(set.intersection(*sets))
    gold_map = {}
    for _, recs in judge_records:
        for r in recs:
            if r["id"] in common: gold_map.setdefault(r["id"], r["gold"])
    N, K = len(common), len(judge_records)
    P = np.zeros((N, K))
    for j, (_, recs) in enumerate(judge_records):
        m = {r["id"]: r["p_a"] for r in recs}
        for i, pid in enumerate(common): P[i, j] = m[pid]
    gold = np.array([gold_map[pid] for pid in common], dtype=int)
    return common, gold, P

def per_judge_rstar(P, gold, names):
    rows = []
    for j, name in enumerate(names):
        p = P[:, j]
        cal = isotonic_calibrate_cv(p, gold)
        rows.append({
            "judge": name, "n": int(len(p)),
            "acc_vs_gold": round(float((( p >= 0.5).astype(int) == gold).mean()), 4),
            "rstar_raw": round(estimate_rstar(p), 4),
            "rstar_iso": round(estimate_rstar(cal), 4),
            "ece_raw": round(compute_ece(p, gold.astype(float)), 4),
            "ece_iso": round(compute_ece(cal, gold.astype(float)), 4),
        })
    return pd.DataFrame(rows)

def cross_partition_stats(P, gold, names, n_boot=500, n_perm=300):
    K = len(names)
    rstar_per_k = np.array([estimate_rstar(isotonic_calibrate_cv(P[:, j], gold)) for j in range(K)])
    var_k, std_k, mean_k = float(np.var(rstar_per_k)), float(np.std(rstar_per_k)), float(np.mean(rstar_per_k))

    rng = np.random.default_rng(42); N = len(gold)
    var_boot = np.array([float(np.var([estimate_rstar(isotonic_calibrate_cv(P[rng.integers(0, N, N), j], gold[rng.integers(0, N, N)])) for j in range(K)])) for _ in range(n_boot)])
    lo, hi = np.quantile(var_boot, [0.025, 0.975])

    var_perm = np.zeros(n_perm)
    for b in range(n_perm):
        Ps = np.copy(P)
        for i in range(N): rng.shuffle(Ps[i])
        var_perm[b] = float(np.var([estimate_rstar(isotonic_calibrate_cv(Ps[:, j], gold)) for j in range(K)]))
    p_value = float(np.mean(var_perm >= var_k))

    return {
        "rstar_per_judge": {n: float(r) for n, r in zip(names, rstar_per_k)},
        "mean_rstar": mean_k, "std_rstar": std_k, "var_rstar": var_k,
        "var_bootstrap_ci": [float(lo), float(hi)],
        "permutation_null_mean_var": float(np.mean(var_perm)),
        "permutation_p_value": p_value,
    }

def plot_figure(df, stats, path, label=""):
    sns.set_theme(style="whitegrid", font="serif", font_scale=1.05)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    palette = sns.color_palette("muted")

    ax = axes[0]; x = np.arange(len(df)); w = 0.35
    ax.bar(x - w/2, df["rstar_raw"], w, label="Raw R*", color=palette[0], edgecolor="white")
    ax.bar(x + w/2, df["rstar_iso"], w, label="Isotonic R*", color=palette[3], edgecolor="white")
    for i, (raw, iso) in enumerate(zip(df["rstar_raw"], df["rstar_iso"])):
        ax.text(i - w/2, raw + 0.003, f"{raw:.3f}", ha="center", fontsize=7.5)
        ax.text(i + w/2, iso + 0.003, f"{iso:.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(df["judge"], rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("R*"); ax.set_title(f"(a) R*|A_k per judge {label}", fontweight="bold"); ax.legend(fontsize=8)

    ax = axes[1]; rvals = list(stats["rstar_per_judge"].values()); nms = list(stats["rstar_per_judge"].keys())
    ax.scatter(range(len(rvals)), rvals, s=120, color=palette[3], edgecolor="white", zorder=5)
    ax.axhline(stats["mean_rstar"], ls="--", color=palette[0], alpha=0.7, label=f"mean={stats['mean_rstar']:.4f}")
    ax.axhspan(stats["mean_rstar"]-stats["std_rstar"], stats["mean_rstar"]+stats["std_rstar"], alpha=0.15, color=palette[0], label=f"std={stats['std_rstar']:.4f}")
    for i, (n, r) in enumerate(zip(nms, rvals)):
        ax.annotate(f"{r:.4f}", (i, r), textcoords="offset points", xytext=(7, 0), fontsize=8)
    ax.set_xticks(range(len(nms))); ax.set_xticklabels(nms, rotation=15, ha="right", fontsize=9)
    pstr = f" (p={stats['permutation_p_value']:.3f})" if stats.get("permutation_p_value") is not None else ""
    ax.set_ylabel("R*|A_k (iso)"); ax.set_title(f"(b) Cross-partition consistency{pstr}", fontweight="bold"); ax.legend(fontsize=8)

    plt.tight_layout(); plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Figure saved: {path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge_files", nargs="+", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--dataset_label", default="")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    judge_records = [(name, recs) for fp in sorted(args.judge_files) for name, recs in [load_judge(fp)]]
    for name, recs in judge_records: print(f"{name}: {len(recs)} valid pairs")

    ids, gold, P = align_judges(judge_records)
    print(f"Aligned: {len(ids)} common pairs across {len(judge_records)} judges")

    names = [n for n, _ in judge_records]
    df = per_judge_rstar(P, gold, names)
    df.to_csv(os.path.join(args.output_dir, "per_judge_rstar.csv"), index=False)
    print(df)

    stats = cross_partition_stats(P, gold, names)
    with open(os.path.join(args.output_dir, "cross_partition_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))

    plot_figure(df, stats, os.path.join(args.output_dir, "fig_rstar_partition.png"), args.dataset_label)

if __name__ == "__main__":
    main()
