"""Assumption checks: normality (Shapiro-Wilk), variance homogeneity (Levene),
and sphericity (Mauchly + Greenhouse-Geisser) for the repeated-measures parts.

Each returns a plain dict and degrades gracefully (too few points, zero variance,
missing groups) rather than raising.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from analysis.degeneracy import is_constant


def shapiro_by_group(df: pd.DataFrame, outcome: str, group: str = "condition") -> dict:
    """Shapiro-Wilk per group. Skips groups with n<3 or zero variance."""

    results: dict[str, dict] = {}
    for name, g in df.groupby(group):
        x = pd.to_numeric(g[outcome], errors="coerce").dropna().to_numpy()
        if len(x) < 3 or is_constant(x):
            results[str(name)] = {"W": None, "p": None, "n": int(len(x)), "note": "skipped"}
            continue
        w, p = stats.shapiro(x)
        results[str(name)] = {"W": float(w), "p": float(p), "n": int(len(x)), "normal": p > 0.05}
    return results


def levene_across_groups(df: pd.DataFrame, outcome: str, group: str = "condition") -> dict:
    """Levene's test of homogeneity of variance across groups."""

    samples = [
        pd.to_numeric(g[outcome], errors="coerce").dropna().to_numpy() for _, g in df.groupby(group)
    ]
    samples = [s for s in samples if len(s) > 1]
    if len(samples) < 2:
        return {"stat": None, "p": None, "note": "insufficient groups"}
    if is_constant(np.concatenate(samples)):
        # scipy returns (nan, nan) here, and `nan > 0.05` is False -- which the report
        # would print as the positive claim "equal_variance": false. Make no claim.
        return {
            "stat": None,
            "p": None,
            "equal_variance": None,
            "skipped": True,
            "skipped_reason": "zero_variance",
            "note": f"{outcome} is constant across all groups; Levene's test is undefined",
        }
    stat, p = stats.levene(*samples, center="median")
    return {"stat": float(stat), "p": float(p), "equal_variance": bool(p > 0.05)}


def sphericity(df: pd.DataFrame, outcome: str, subject: str, within: str) -> dict:
    """Mauchly's sphericity (via pingouin) for a repeated-measures factor."""

    import pingouin as pg

    try:
        spher, w, chi2, dof, pval = pg.sphericity(
            data=df, dv=outcome, subject=subject, within=within
        )
        return {
            "spherical": bool(spher),
            "W": float(w),
            "chi2": float(chi2),
            "dof": float(dof) if dof is not None else None,
            "p": float(pval),
        }
    except Exception as exc:  # pragma: no cover - degenerate repeated-measures data
        return {"spherical": None, "note": f"unavailable: {exc}"}
