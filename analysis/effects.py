"""Effect-size helpers used across the research-question analyses."""

from __future__ import annotations

import numpy as np


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for two independent samples (pooled SD). NaN if undefined."""

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)
    if pooled <= 0:
        return 0.0 if a.mean() == b.mean() else float("nan")
    return float((a.mean() - b.mean()) / np.sqrt(pooled))


def eta_squared(groups: list[np.ndarray]) -> float:
    """One-way eta-squared = SS_between / SS_total. NaN if undefined."""

    arrs = [np.asarray(g, dtype=float) for g in groups if len(g) > 0]
    allv = np.concatenate(arrs) if arrs else np.array([])
    if allv.size < 2:
        return float("nan")
    grand = allv.mean()
    ss_total = float(((allv - grand) ** 2).sum())
    if ss_total == 0:
        return 0.0
    ss_between = float(sum(len(g) * (g.mean() - grand) ** 2 for g in arrs))
    return ss_between / ss_total


def odds_ratio_2x2(a: int, b: int, c: int, d: int) -> float:
    """Odds ratio for a 2x2 table with Haldane-Anscombe 0.5 correction."""

    a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    return float((a * d) / (b * c))
