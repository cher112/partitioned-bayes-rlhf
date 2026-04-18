"""Shared calibration and R* utilities (ported from time-series repo)."""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

def isotonic_calibrate_cv(probs, labels, n_splits=5, random_state=42):
    probs, labels = np.asarray(probs, float), np.asarray(labels, int)
    cal = np.zeros_like(probs)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for tr, va in skf.split(probs, labels):
        ir = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
        ir.fit(probs[tr], labels[tr].astype(float))
        cal[va] = ir.predict(probs[va])
    return cal

def temperature_scale_cv(probs, labels, n_splits=5, random_state=42):
    from sklearn.metrics import log_loss
    probs, labels = np.asarray(probs, float), np.asarray(labels, int)
    cal = np.zeros_like(probs)
    two = np.stack([1 - probs, probs], axis=1)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for tr, va in skf.split(probs, labels):
        logits_tr = np.log(np.clip(two[tr], 1e-10, 1.0))
        best_T, best = 1.0, float("inf")
        for T in np.arange(0.2, 5.01, 0.1):
            s = np.exp(logits_tr / T); s = s / s.sum(1, keepdims=True)
            try:
                ll = log_loss(labels[tr], s, labels=[0, 1])
            except: continue
            if ll < best: best, best_T = ll, T
        lv = np.log(np.clip(two[va], 1e-10, 1.0))
        sv = np.exp(lv / best_T); sv = sv / sv.sum(1, keepdims=True)
        cal[va] = sv[:, 1]
    return cal

def estimate_rstar(eta):
    return float(np.mean(np.minimum(np.asarray(eta, float), 1 - np.asarray(eta, float))))

def compute_ece(probs, labels, n_bins=15):
    probs, labels = np.asarray(probs, float), np.asarray(labels, float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (probs >= bins[i]) & (probs < bins[i + 1])
        if m.sum() > 0: ece += m.sum() * abs(probs[m].mean() - labels[m].mean())
    return float(ece / len(labels))

def bootstrap_ci(fn, data, n_boot=1000, alpha=0.05, random_state=42):
    rng = np.random.default_rng(random_state)
    n = len(data)
    stats = np.array([fn(np.asarray(data)[rng.integers(0, n, n)]) for _ in range(n_boot)])
    return float(np.quantile(stats, alpha/2)), float(np.quantile(stats, 1-alpha/2))
