from itertools import combinations
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable = np.mean,
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    data = np.asarray(data, dtype=float)
    point = float(statistic(data))
    boot = np.array([
        statistic(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = 1.0 - ci
    lo = float(np.percentile(boot, 100.0 * alpha / 2.0))
    hi = float(np.percentile(boot, 100.0 * (1.0 - alpha / 2.0)))
    return point, lo, hi


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n1, n2 = len(x), len(y)
    s_pool = np.sqrt(
        ((n1 - 1) * x.std(ddof=1) ** 2 + (n2 - 1) * y.std(ddof=1) ** 2)
        / (n1 + n2 - 2)
    )
    if s_pool == 0.0:
        return 0.0
    return float((x.mean() - y.mean()) / s_pool)


def fdr_correction(
    pvalues: List[float],
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    pv = np.asarray(pvalues, dtype=float)
    m = len(pv)
    if m == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)

    order = np.argsort(pv)
    ranked = pv[order]
    threshold = (np.arange(1, m + 1) / m) * alpha

    below = ranked <= threshold
    if below.any():
        max_idx = int(np.max(np.where(below)[0]))
        reject_ordered = np.zeros(m, dtype=bool)
        reject_ordered[: max_idx + 1] = True
    else:
        reject_ordered = np.zeros(m, dtype=bool)

    reject = np.empty(m, dtype=bool)
    reject[order] = reject_ordered

    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    corrected = np.empty(m, dtype=float)
    corrected[order] = adj

    return reject, corrected


def wilcoxon_pairwise(
    groups: Dict[str, np.ndarray],
    alternative: str = "two-sided",
) -> Dict[Tuple[str, str], dict]:
    pairs = list(combinations(groups.keys(), 2))
    raw: Dict[Tuple[str, str], dict] = {}
    pvals: List[float] = []

    for a, b in pairs:
        stat, p = stats.mannwhitneyu(
            groups[a], groups[b], alternative=alternative
        )
        d = cohens_d(groups[a], groups[b])
        raw[(a, b)] = {
            "statistic": float(stat),
            "p_raw":     float(p),
            "cohens_d":  d,
        }
        pvals.append(float(p))

    if pvals:
        reject, corrected = fdr_correction(pvals)
        for idx, (a, b) in enumerate(pairs):
            raw[(a, b)]["p_fdr"] = float(corrected[idx])
            raw[(a, b)]["reject"] = bool(reject[idx])

    return raw


def spearman_bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    rho_obs = float(stats.spearmanr(x, y).statistic)
    idx_all = np.arange(len(x))
    boot = []
    for _ in range(n_bootstrap):
        idx = rng.choice(idx_all, size=len(x), replace=True)
        r = stats.spearmanr(x[idx], y[idx]).statistic
        if np.isfinite(r):
            boot.append(r)
    boot = np.array(boot) if boot else np.array([rho_obs])
    alpha = 1.0 - ci
    lo = float(np.percentile(boot, 100.0 * alpha / 2.0))
    hi = float(np.percentile(boot, 100.0 * (1.0 - alpha / 2.0)))
    return rho_obs, lo, hi


def powerlaw_fit(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    valid = (x > 0) & (y > 0)
    lx, ly = np.log(x[valid]), np.log(y[valid])

    alpha_obs, log_a_obs, r, *_ = stats.linregress(lx, ly)
    r2 = float(r ** 2)

    rng = np.random.default_rng(seed)
    idx_all = np.arange(len(lx))
    boot_alphas = []
    for _ in range(n_bootstrap):
        idx = rng.choice(idx_all, size=len(lx), replace=True)
        if lx[idx].std() == 0:
            continue
        slope, *_ = stats.linregress(lx[idx], ly[idx])
        boot_alphas.append(slope)
    boot_alphas = np.array(boot_alphas) if boot_alphas else np.array([alpha_obs])

    alpha_ci = 1.0 - ci
    lo = float(np.percentile(boot_alphas, 100.0 * alpha_ci / 2.0))
    hi = float(np.percentile(boot_alphas, 100.0 * (1.0 - alpha_ci / 2.0)))

    return {
        "alpha":         float(alpha_obs),
        "log_a":         float(log_a_obs),
        "alpha_ci_low":  lo,
        "alpha_ci_high": hi,
        "r_squared":     r2,
    }


def summary_stats(data: np.ndarray, n_bootstrap: int = 5_000) -> dict:
    data = np.asarray(data, dtype=float)
    point, lo, hi = bootstrap_ci(data, n_bootstrap=n_bootstrap)
    return {
        "mean":         point,
        "std":          float(data.std(ddof=1)),
        "median":       float(np.median(data)),
        "ci95_low":     lo,
        "ci95_high":    hi,
        "n":            len(data),
    }
