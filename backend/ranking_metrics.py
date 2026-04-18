"""
Ranking / top-K diagnostics for model scores vs realized forward returns.
Uses pandas/numpy only (no scipy).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ranking_edge_report(
    scores: np.ndarray,
    forward_returns: np.ndarray,
    *,
    top_frac: float = 0.10,
    cost_fraction: float = 0.0008,
    n_perm: int = 3000,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict:
    """
    - Spearman IC between score and forward return (+ permutation p-value on |IC|).
    - Top-K vs rest mean forward return, lift, and approximate net after cost on top-K.
    - Bootstrap 95% CI on lift (resample rows).
    """
    s = np.asarray(scores, dtype=float)
    r = np.asarray(forward_returns, dtype=float)
    m = np.isfinite(s) & np.isfinite(r)
    s, r = s[m], r[m]
    n = len(s)
    if n < 30:
        return {
            "n": int(n),
            "spearman_ic": None,
            "spearman_ic_pvalue_perm": None,
            "top_frac": float(top_frac),
            "top_k_mean_fwd": None,
            "rest_mean_fwd": None,
            "lift_mean_fwd": None,
            "top_k_mean_net_after_cost_est": None,
            "lift_bootstrap_ci95_low": None,
            "lift_bootstrap_ci95_high": None,
        }

    ic = float(pd.Series(s).corr(pd.Series(r), method="spearman"))
    if not np.isfinite(ic):
        ic = 0.0

    rng = np.random.default_rng(seed)
    null_abs_ic = []
    for _ in range(int(n_perm)):
        rp = rng.permutation(r)
        v = float(pd.Series(s).corr(pd.Series(rp), method="spearman"))
        null_abs_ic.append(abs(v) if np.isfinite(v) else 0.0)
    p_ic = float((np.array(null_abs_ic) >= abs(ic)).mean())

    k = max(1, int(n * float(top_frac)))
    order = np.argsort(-s)
    top = r[order[:k]]
    rest = r[order[k:]]
    top_mean = float(np.mean(top))
    rest_mean = float(np.mean(rest)) if len(rest) else float("nan")
    lift = float(top_mean - rest_mean) if len(rest) else float("nan")
    top_net = float(top_mean - cost_fraction)
    if not np.isfinite(top_net):
        top_net = float("nan")

    lifts_boot = []
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        s_b, r_b = s[idx], r[idx]
        o = np.argsort(-s_b)
        kb = max(1, int(n * top_frac))
        lifts_boot.append(float(np.mean(r_b[o[:kb]]) - np.mean(r_b[o[kb:]])))
    lifts_boot = np.array(lifts_boot)
    lo, hi = float(np.percentile(lifts_boot, 2.5)), float(np.percentile(lifts_boot, 97.5))

    def _sf(x):
        if x is None:
            return None
        xf = float(x)
        return xf if np.isfinite(xf) else None

    return {
        "n": int(n),
        "spearman_ic": _sf(ic),
        "spearman_ic_pvalue_perm": _sf(p_ic),
        "top_frac": float(top_frac),
        "top_k_mean_fwd": _sf(top_mean),
        "rest_mean_fwd": _sf(rest_mean),
        "lift_mean_fwd": _sf(lift),
        "top_k_mean_net_after_cost_est": _sf(top_net),
        "lift_bootstrap_ci95_low": _sf(lo),
        "lift_bootstrap_ci95_high": _sf(hi),
    }
