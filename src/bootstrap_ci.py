"""Bootstrap 95% CI helpers for P0 / P1 / P2 metrics.

Per Gemini Round-5 death-line #3: every point estimate in the repo must carry
a bootstrap 95% CI.

Usage
-----
    from bootstrap_ci import bootstrap_ci_mean
    lo, hi = bootstrap_ci_mean(samples, n_resamples=1000, confidence=0.95, seed=42)
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def bootstrap_ci_mean(samples, n_resamples: int = 1000, confidence: float = 0.95,
                      seed: int = 42) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of a 1D array."""
    samples = np.asarray(samples, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(samples)
    means = np.empty(n_resamples, dtype=float)
    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[b] = samples[idx].mean()
    alpha = (1.0 - confidence) / 2.0
    lo = float(np.quantile(means, alpha))
    hi = float(np.quantile(means, 1 - alpha))
    return lo, hi


def bootstrap_ci_statistic(samples, stat_fn: Callable[[np.ndarray], float],
                           n_resamples: int = 1000, confidence: float = 0.95,
                           seed: int = 42) -> tuple[float, float]:
    """Percentile bootstrap CI for an arbitrary statistic."""
    samples = np.asarray(samples, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(samples)
    stats = np.empty(n_resamples, dtype=float)
    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        stats[b] = stat_fn(samples[idx])
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(stats, alpha)), float(np.quantile(stats, 1 - alpha))


def bootstrap_ci_correlation(x, y, n_resamples: int = 1000, confidence: float = 0.95,
                             seed: int = 42, method: str = "pearson") -> tuple[float, float]:
    """Percentile bootstrap CI for Pearson/Spearman correlation of two paired arrays.
    Skips resamples producing constant arrays (NaN correlation), which matters for small N.
    """
    from scipy.stats import pearsonr, spearmanr
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    assert len(x) == len(y)
    rng = np.random.default_rng(seed)
    n = len(x)
    rs: list[float] = []
    attempts = 0
    while len(rs) < n_resamples and attempts < n_resamples * 20:
        attempts += 1
        idx = rng.integers(0, n, size=n)
        xs, ys = x[idx], y[idx]
        if xs.std() < 1e-12 or ys.std() < 1e-12:
            continue  # constant resample, correlation undefined
        if method == "pearson":
            r, _ = pearsonr(xs, ys)
        else:
            r, _ = spearmanr(xs, ys)
        if np.isfinite(r):
            rs.append(float(r))
    alpha = (1.0 - confidence) / 2.0
    rs_arr = np.asarray(rs)
    return float(np.quantile(rs_arr, alpha)), float(np.quantile(rs_arr, 1 - alpha))


def fisher_z_ci_correlation(r: float, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Fisher z-transform based CI for Pearson correlation — the standard closed-form
    alternative when N is small and bootstrap is unstable."""
    from scipy.stats import norm
    if n < 4:
        return float("nan"), float("nan")
    z = 0.5 * np.log((1 + r) / (1 - r))
    se = 1.0 / np.sqrt(n - 3)
    alpha = (1.0 - confidence) / 2.0
    zcrit = norm.ppf(1 - alpha)
    z_lo, z_hi = z - zcrit * se, z + zcrit * se
    r_lo = (np.exp(2 * z_lo) - 1) / (np.exp(2 * z_lo) + 1)
    r_hi = (np.exp(2 * z_hi) - 1) / (np.exp(2 * z_hi) + 1)
    return float(r_lo), float(r_hi)


def fmt_mean_ci(mean: float, lo: float, hi: float, decimals: int = 4) -> str:
    """Format '0.4315 [0.4260, 0.4370]' for README tables."""
    return f"{mean:.{decimals}f} [{lo:.{decimals}f}, {hi:.{decimals}f}]"
