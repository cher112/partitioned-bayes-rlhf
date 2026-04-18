"""Add bootstrap 95% CI to every per-partition metric of P1.

Ingests the raw preference data (same as analyze_preference.py) and produces
partition_stats_with_ci.json containing all estimates with CIs.
"""
import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# add src/ to path so we can import bootstrap_ci
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from bootstrap_ci import bootstrap_ci_mean, fmt_mean_ci  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = REPO / "data" / "hs2_extra" / "preference" / "preference.jsonl.gz"
N_BOOTSTRAP = 1000
SEED = 42


def _safe_parse(raw):
    if isinstance(raw, str):
        items = []
        for m in re.finditer(r"\{'strength': (-?\d+),\s*'justification': '([^']{0,80})", raw):
            items.append({"strength": int(m.group(1)),
                          "justification_snippet": m.group(2)})
    else:
        items = [{"strength": int(d["strength"]),
                  "justification_snippet": str(d.get("justification", ""))[:80]}
                 for d in raw]
    out = []
    for it in items:
        ref = re.search(r"@Response\s*(\d)", it["justification_snippet"])
        preferred = int(ref.group(1)) if ref else None
        out.append({"strength": it["strength"], "preferred_response": preferred})
    return out


def load_pairs(path):
    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            obj = json.loads(line)
            annos = _safe_parse(obj["all_preferences_unprocessed"])
            if not annos:
                continue
            try:
                agg = int(obj["preference_strength"])
            except (ValueError, TypeError):
                continue
            rows.append({"agg_strength": agg, "annotators": annos})
    return rows


def partition_pair_metrics(sub):
    """For each pair in sub, emit:
       - pair_agree_rate   (1 if all 3 agree, else fraction-of-majority-class)
       - each annotator's disagree-with-majority flag (0/1), appended per-annotator
       - strength_variance over the 3 annotators."""
    pair_agree_rates = []
    annotator_flags = []   # flat: 0/1 per annotator per pair
    strength_vars = []

    for r in sub:
        signs = [a["preferred_response"] for a in r["annotators"] if a["preferred_response"] in (1, 2)]
        if len(signs) < 2:
            continue
        maj = Counter(signs).most_common(1)[0][0]
        pair_agree_rates.append(sum(1 for s in signs if s == maj) / len(signs))
        for s in signs:
            annotator_flags.append(0 if s == maj else 1)
        strength_vars.append(float(np.var([abs(a["strength"]) for a in r["annotators"]])))

    return pair_agree_rates, annotator_flags, strength_vars


def main():
    print(f"Loading {DATA}")
    if not DATA.exists():
        print(f"  WARNING: {DATA} not found. Run the HS2 download step first.")
        sys.exit(1)
    rows = load_pairs(DATA)
    print(f"Parsed {len(rows)} preference pairs")

    strengths = sorted({abs(r["agg_strength"]) for r in rows})
    print(f"|preference_strength| levels: {strengths}")

    partitions = []
    for k in strengths:
        sub = [r for r in rows if abs(r["agg_strength"]) == k]
        agree_rates, ann_flags, str_vars = partition_pair_metrics(sub)

        agree_mean = float(np.mean(agree_rates)) if agree_rates else float("nan")
        agree_lo, agree_hi = bootstrap_ci_mean(agree_rates, N_BOOTSTRAP, seed=SEED)

        err_mean = float(np.mean(ann_flags)) if ann_flags else float("nan")
        err_lo, err_hi = bootstrap_ci_mean(ann_flags, N_BOOTSTRAP, seed=SEED + 1)

        svar_mean = float(np.mean(str_vars)) if str_vars else float("nan")
        svar_lo, svar_hi = bootstrap_ci_mean(str_vars, N_BOOTSTRAP, seed=SEED + 2)

        partitions.append({
            "strength_bin": k,
            "n_pairs": len(sub),
            "agree_rate": {"mean": agree_mean, "ci_lo": agree_lo, "ci_hi": agree_hi},
            "per_annotator_error_rate": {"mean": err_mean, "ci_lo": err_lo, "ci_hi": err_hi,
                                         "n_annotations": len(ann_flags)},
            "strength_variance": {"mean": svar_mean, "ci_lo": svar_lo, "ci_hi": svar_hi},
        })
        print(f"  k={k}: n={len(sub):>5}  "
              f"err_rate = {fmt_mean_ci(err_mean, err_lo, err_hi, 4)}  "
              f"agree = {fmt_mean_ci(agree_mean, agree_lo, agree_hi, 4)}")

    out = {
        "n_pairs_total": len(rows),
        "n_bootstrap_resamples": N_BOOTSTRAP,
        "bootstrap_seed": SEED,
        "confidence_level": 0.95,
        "partitions": partitions,
    }
    with open(HERE / "partition_stats_with_ci.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {HERE / 'partition_stats_with_ci.json'}")


if __name__ == "__main__":
    main()
