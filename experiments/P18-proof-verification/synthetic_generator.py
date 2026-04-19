"""Shared synthetic data generator for P18 proof verification.

Generates K judges' isotonically-calibrated outputs c_iso_k from a true
posterior eta, with each h_k a smooth monotone transform parameterised by
judge-specific (a_k, b_k). All true quantities (rho*_alpha, R*) are derived
analytically so that the empirical error of estimators can be verified
against a known ground truth.

Used by:
  - script1_anchor_rate.py    (Lemma 1: |rhô_alpha - rho*_alpha| = O_P(M^{-1/3}))
  - script2_lipschitz.py      (Lemma 2: Lipschitz bridge in rho)
  - script3_plugin_rate.py    (Thm 3: |R̂*_CA(rho*) - R*| = O_P(N^{-1/2}))

Model (matches the theorem's A1–A3 after A2 TIGHTENING, 2026-04-19):

Pilot run with h_k(eta) = sigma(a_k logit eta + b_k) showed asymptotic bias approx
0.031 that does NOT decay with N (log-log slope -0.014, not -1/3). The
estimator R̂*_CA = mean_i min(mean_k p_corr_k(i), 1 - mean_k p_corr_k(i))
does NOT converge to R* = E[min(eta, 1-eta)] under a general monotone
transform h_k, because the affine correction cannot invert a non-affine
transform.

Theorem-valid assumption (tightened A2):
  c_iso_k(i) = (1 - rho_0 - rho_1) * eta_i + rho_0,     (shared affine truncation)
  with SHARED anchors rho_0, rho_1 across all K judges, and judge-specific
  multiplicative noise eps_k(eta_i) added AFTER the affine stretch (so
  per-judge heterogeneity exists only through independent noise, not
  through different anchors). Specifically:

      c_iso_k(i) = (1 - rho_0 - rho_1) * eta_i + rho_0 + e_k(i),
      e_k(i) ~ subgauss with mean 0, var sigma_k^2.

Under this A2, R̂*_CA → R* as N → ∞ (the affine correction exactly
inverts the shared truncation, and independent noise averages out).

Implementation here: `gen_bt_data(..., mode="affine")` uses this tight
A2. `mode="sigmoid"` preserves the old general-monotone generator for
reference (illustrates the asymptotic bias of the wider A2).

True preference Bayes error R* under BT with posterior eta is:
  R* = E_eta[min(eta, 1 - eta)]
(binary classification form; the preference version on paired BT reduces
to this when the "preferred response" is encoded as the side with eta > 0.5)
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
from scipy.special import expit, logit
from scipy.stats import beta as beta_dist


@dataclass
class JudgeParams:
    """(a, b) for c_iso = sigma(a * logit(eta) + b)."""
    a: float   # scale (1 = no temp scaling; smaller = squash toward 0.5)
    b: float   # bias (0 = unbiased; nonzero = systematic over/under confidence)


def default_judges(k: int = 6, seed: int = 0) -> list[JudgeParams]:
    """Return K judges with a spread of (a, b) to mimic real LLM-judge variability."""
    rng = np.random.default_rng(seed)
    # a in [0.5, 2.0], b in [-0.3, 0.3]
    a_vals = rng.uniform(0.5, 2.0, size=k)
    b_vals = rng.uniform(-0.3, 0.3, size=k)
    return [JudgeParams(a=float(a_vals[i]), b=float(b_vals[i])) for i in range(k)]


def sample_eta(n: int, seed: int = 0, dist: str = "beta22") -> np.ndarray:
    """Sample N values of eta in (0, 1) from a known distribution."""
    rng = np.random.default_rng(seed)
    if dist == "beta22":
        return rng.beta(2.0, 2.0, size=n)
    if dist == "uniform":
        return rng.uniform(1e-4, 1 - 1e-4, size=n)
    raise ValueError(dist)


def apply_judges_sigmoid(eta: np.ndarray, judges: list[JudgeParams]) -> np.ndarray:
    """LEGACY: general monotone h_k(eta) = sigma(a_k logit eta + b_k). Has asymptotic
    bias; kept for illustrating why A2 needs tightening."""
    lo = logit(np.clip(eta, 1e-6, 1 - 1e-6))
    return np.stack([expit(j.a * lo + j.b) for j in judges], axis=1)


def apply_judges_affine(eta: np.ndarray, rho_0: float, rho_1: float,
                        k: int, noise_sigma: float = 0.0,
                        seed: int = 0) -> np.ndarray:
    """Shared-truncation A2 generator. c_k = (1 - rho_0 - rho_1) eta + rho_0 + eps_k,
    eps_k iid N(0, sigma**2) clipped to keep c_k in [0, 1].

    Note: noise_sigma > 0 makes per-judge error independent; with sigma = 0 the
    K judges are identical (so K_eff collapses). For the Lemma-1 test we
    want noise_sigma > 0 so K_eff > 1.
    """
    rng = np.random.default_rng(seed + 500)
    base = (1 - rho_0 - rho_1) * eta + rho_0   # (N,)
    noise = rng.normal(0.0, noise_sigma, size=(len(eta), k))
    c = np.clip(base[:, None] + noise, 0.0, 1.0)
    return c


def gen_bt_data(n: int, k: int, seed: int = 0,
                dist: str = "beta22", mode: str = "affine",
                rho_0: float = 0.15, rho_1: float = 0.10,
                noise_sigma: float = 0.03
                ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate one synthetic dataset.

    Parameters
    ----------
    mode : "affine" (theorem-valid A2, default) or "sigmoid" (legacy
           general-monotone, has asymptotic bias).
    rho_0, rho_1 : only used for mode="affine".
    noise_sigma : only used for mode="affine".

    Returns
    -------
    eta      : (N,) true posterior
    c_iso    : (N, K) per-judge isotonically-calibrated scores
    meta     : dict with either "judges" (sigmoid) or "rho_0, rho_1, sigma"
               (affine), for use by the analytic-rho / analytic-R computers.
    """
    eta = sample_eta(n, seed=seed, dist=dist)
    if mode == "sigmoid":
        judges = default_judges(k=k, seed=seed + 1000)
        c = apply_judges_sigmoid(eta, judges)
        return eta, c, {"mode": "sigmoid", "judges": judges}
    if mode == "affine":
        c = apply_judges_affine(eta, rho_0, rho_1, k, noise_sigma, seed)
        return eta, c, {"mode": "affine", "rho_0": rho_0, "rho_1": rho_1,
                        "noise_sigma": noise_sigma, "k": k}
    raise ValueError(mode)


# ---------- Analytic ground truth ----------

def analytic_rstar(dist: str = "beta22", n_grid: int = 20001) -> float:
    """R* = E_eta[min(eta, 1 - eta)]."""
    if dist == "beta22":
        grid = np.linspace(1e-6, 1 - 1e-6, n_grid)
        dens = beta_dist.pdf(grid, 2.0, 2.0)
        vals = np.minimum(grid, 1 - grid)
        return float(np.trapezoid(vals * dens, grid))
    if dist == "uniform":
        return 0.25  # ∫_0^1 min(x, 1-x) dx = 0.25
    raise ValueError(dist)


def analytic_rho_affine(alpha: float, rho_0: float, rho_1: float,
                        dist: str = "beta22", n_grid: int = 20001) -> tuple[float, float]:
    """Under the affine A2 (c = (1-rho_0-rho_1) eta + rho_0, no noise), the pooled
    c distribution is a linear push-forward of F_eta.

    With noise sigma, the distribution is a convolution; for small sigma the
    quantiles shift by approx sigma * Phi^{-1}(alpha) (Gaussian tail approximation). We
    ignore noise for the analytic target — that's the right rho* to
    compare the empirical estimator against in the noise-free limit.
    """
    grid = np.linspace(1e-6, 1 - 1e-6, n_grid)
    if dist == "beta22":
        dens = beta_dist.pdf(grid, 2.0, 2.0)
    elif dist == "uniform":
        dens = np.ones_like(grid)
    else:
        raise ValueError(dist)
    c = (1.0 - rho_0 - rho_1) * grid + rho_0
    cdf = np.cumsum(dens) / dens.sum()
    # Invert cdf at alpha and 1-alpha
    rho_alpha = float(c[np.searchsorted(cdf, alpha)])
    rho_1minus = float(c[np.searchsorted(cdf, 1 - alpha)])
    return rho_alpha, 1.0 - rho_1minus   # (rho*_alpha lower, rho*_1-alpha upper → 1 - q_1-alpha})


def analytic_rho(alpha: float, meta: dict,
                 dist: str = "beta22", n_grid: int = 20001) -> tuple[float, float]:
    """Dispatch to the right analytic-rho computation based on meta['mode']."""
    if meta.get("mode") == "affine":
        return analytic_rho_affine(alpha, meta["rho_0"], meta["rho_1"],
                                   dist=dist, n_grid=n_grid)
    # Sigmoid legacy: fall back to numeric pool
    from scipy.special import expit as _expit, logit as _logit
    judges = meta["judges"]
    grid = np.linspace(1e-6, 1 - 1e-6, n_grid)
    if dist == "beta22":
        dens = beta_dist.pdf(grid, 2.0, 2.0)
    elif dist == "uniform":
        dens = np.ones_like(grid)
    else:
        raise ValueError(dist)
    def cdf_at(q: float) -> float:
        mask = np.mean(
            [(_expit(j.a * _logit(grid) + j.b) <= q).astype(float) for j in judges],
            axis=0)
        return float(np.trapezoid(mask * dens, grid))
    def bisect_for(target: float, lo: float = 1e-6, hi: float = 1 - 1e-6,
                   tol: float = 1e-5) -> float:
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if cdf_at(mid) < target:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        return 0.5 * (lo + hi)
    return bisect_for(alpha), 1.0 - bisect_for(1 - alpha)


# ---------- Estimator (matches P16 implementation exactly) ----------

def rstar_ca_pooled(c_iso: np.ndarray, alpha: float = 0.01,
                    rho_override: tuple[float, float] | None = None
                    ) -> tuple[float, dict]:
    """Pooled-anchor FCI R̂*_CA.

    If rho_override is given, use (rho_0, rho_1) directly (for Lemma 2 Script).
    Otherwise estimate from data.
    """
    if rho_override is not None:
        rho0, rho1 = rho_override
    else:
        pool = c_iso.flatten()
        rho0 = float(np.quantile(pool, alpha))
        rho1 = 1.0 - float(np.quantile(pool, 1 - alpha))
    denom = 1.0 - rho0 - rho1
    if denom <= 0.01:
        # Degenerate — fall back to plug-in mean
        p_bar = c_iso.mean(axis=1)
        r = float(np.minimum(p_bar, 1 - p_bar).mean())
        return r, {"rho0": rho0, "rho1": rho1, "stretch": float("inf"),
                   "degenerate": True}
    p_corr = np.clip((c_iso - rho0) / denom, 0.0, 1.0)
    p_bar = p_corr.mean(axis=1)
    r = float(np.minimum(p_bar, 1 - p_bar).mean())
    return r, {"rho0": rho0, "rho1": rho1, "stretch": 1.0 / denom,
               "degenerate": False}


if __name__ == "__main__":
    print("=" * 80)
    print("Sanity: AFFINE A2 (theorem-valid) — convergence of R̂*_CA → R*")
    print("=" * 80)
    r_star = analytic_rstar()
    print(f"Analytic R* (Beta(2,2)): {r_star:.6f}\n")
    print(f"{'N':>8}  {'K':>3}  {'empir R̂*_CA':>14}  {'bias':>10}  {'M^{-1/3} ref':>14}")
    import numpy as np
    for N in [200, 500, 1000, 2000, 5000, 10000]:
        K = 6
        biases = []
        for seed in range(30):
            eta, c_iso, meta = gen_bt_data(n=N, k=K, seed=seed,
                                            mode="affine",
                                            rho_0=0.15, rho_1=0.10,
                                            noise_sigma=0.03)
            r_ca, _ = rstar_ca_pooled(c_iso, alpha=0.01)
            biases.append(abs(r_ca - r_star))
        mean_bias = float(np.mean(biases))
        ref = (N * K) ** (-1/3)
        mean_ca = r_star + 0   # we only track bias
        print(f"{N:>8}  {K:>3}  {biases[0]:>14.4f}  {mean_bias:>10.5f}  {ref:>14.5f}")

    # Slope fit
    ns = [200, 500, 1000, 2000, 5000, 10000]
    ys = []
    for N in ns:
        bs = []
        for seed in range(30):
            _, c, _ = gen_bt_data(n=N, k=6, seed=seed, mode="affine",
                                  rho_0=0.15, rho_1=0.10, noise_sigma=0.03)
            r_ca, _ = rstar_ca_pooled(c, alpha=0.01)
            bs.append(abs(r_ca - r_star))
        ys.append(float(np.mean(bs)))
    slope, _ = np.polyfit(np.log(ns), np.log(ys), 1)
    print(f"\nlog-log slope of |R̂*_CA - R*| vs N:  {slope:+.4f}  (target approx -1/3)")
