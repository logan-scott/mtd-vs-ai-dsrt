"""Instrument reliability: ICC(2,1), agreement/kappa, and Cronbach's alpha.

Measurement reliability is quantified across the dedicated repeatability runs
(same checksum-verified configuration, attacker adaptation disabled):

- continuous outcomes: **ICC(2,1)** — two-way random-effects, absolute-agreement,
  single-measurement (pingouin ``ICC(A,1)``), Koo & Li (2016) thresholds (>= .75 good,
  >= .90 excellent);
- binary outcomes: **percent agreement + Cohen's kappa**.

IMPORTANT — reliability is NOT estimable from the offline pipeline. The mock
attacker is deterministic and the offline repeatability runs re-execute a single
configuration, so there is no measurement variation to partition, and there is no
second, independent re-measurement of the *same* realization to correlate. ICC(2,1)
and kappa require the **live** design: R independent re-executions of each
checksum-verified configuration, across multiple configurations, where timing/
scheduling jitter, telemetry sampling noise, clock-sync residual, nondeterministic
tools, and LLM stochasticity produce genuine variation. The offline pipeline's
identical re-parse is *computational determinism* (a software property verified by
the unit tests), NOT reliability evidence — the two are reported separately and not
conflated.

The ``icc2`` and ``cohen_kappa`` helpers below implement the correct estimators for
that live targets x raters layout; ``reliability_report`` explains why they are not
computed from offline data.

Cronbach's alpha is applied ONLY to the operational-overhead composite (its
indicators plausibly reflect a common resource-cost dimension); the security-efficacy
indicators are formative measures of distinct dimensions and are examined for
redundancy under RQ4, not via alpha.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from measurement.models import TrialRow

OVERHEAD_COMPOSITE = [
    "latency_overhead_ms",
    "cpu_overhead_pct",
    "memory_overhead_pct",
    "network_overhead_pct",
]

_KOO_LI = "Koo & Li (2016): ICC >= .75 good, >= .90 excellent"

# Reporting (not inference) threshold for the missingness prose. The over-representation
# factor is a ratio of two shares and is therefore already unit-free and scale-relative;
# 1.25 means a group supplies >= 25% more of the complete-case rows than of the analytic
# sample. It selects which SENTENCE is printed -- both branches print the exact factor and
# the exact counts, plus a threshold-free total-variation distance, so no claim depends on
# where the bar sits.
_DOMINANCE_FACTOR = 1.25

# Conventional alpha bar (Nunnally-style) reused as a DIAGNOSTIC screen, not a test
# statistic: there is no null distribution behind `between_condition_artifact`.
_ALPHA_BAR = 0.70


def item_missingness(
    df: pd.DataFrame,
    columns: list[str],
    group_col: str = "condition",
    missing_reason: str | None = None,
) -> dict:
    """Per-item n, complete-case n, and complete-case composition for a composite.

    ``cronbach_alpha`` (and RQ4's VIF/PCA/regularized selection) listwise-delete before
    the estimator is called, so the reported statistic describes only the complete-case
    rows. This makes the size and the composition of that subset explicit.
    """

    present = [c for c in columns if c in df.columns]
    out: dict = {
        "items_requested": list(columns),
        "items_present": present,
        "items_missing_from_dataset": [c for c in columns if c not in df.columns],
        "n_input_rows": int(len(df)),
    }
    if not present or not len(df):
        out["note"] = "no requested items present / empty frame"
        return out
    sub = df[present].apply(pd.to_numeric, errors="coerce")
    n_item = {c: int(sub[c].notna().sum()) for c in present}
    out["n_per_item"] = n_item
    if group_col and group_col in df.columns:
        out[f"n_per_item_by_{group_col}"] = {
            str(g): {c: int(pd.to_numeric(g_df[c], errors="coerce").notna().sum()) for c in present}
            for g, g_df in df.groupby(group_col, observed=True)
        }
    cc_idx = sub.dropna().index
    out["n_complete_case"] = int(len(cc_idx))
    out["complete_case_share_of_input"] = round(len(cc_idx) / len(df), 4)
    binding = min(n_item, key=lambda c: n_item[c])
    out["binding_item"] = binding if n_item[binding] < len(df) else None
    out["binding_item_missing_reason"] = missing_reason
    if group_col and group_col in df.columns and len(cc_idx):
        counts = df.loc[cc_idx, group_col].value_counts()
        full = df[group_col].value_counts()
        shares = {str(k): round(v / len(cc_idx), 4) for k, v in counts.items()}
        base = {str(k): round(v / len(df), 4) for k, v in full.items()}
        over = {k: (round(shares[k] / base[k], 2) if base.get(k) else None) for k in shares}
        tvd = 0.5 * sum(abs(shares.get(k, 0.0) - base[k]) for k in base)
        out["complete_case_composition"] = {
            "group_col": group_col,
            "counts": {str(k): int(v) for k, v in counts.items()},
            "share_of_complete_case": shares,
            "share_of_analytic_sample": base,
            "over_representation_factor": over,
            "total_variation_distance": round(float(tvd), 4),
        }
        if over:
            worst = max(over, key=lambda k: over[k] if over[k] is not None else 0.0)
            out["most_over_represented_group"] = worst
            out["most_over_represented_factor"] = over[worst]
        else:
            out["complete_case_composition"][
                "note"
            ] = f"'{group_col}' is missing on every complete-case row; composition not computable"
    out["missing_data_handling"] = (
        "listwise deletion: rows with ANY missing indicator are dropped before the estimator "
        "(analysis.reliability.cronbach_alpha calls .dropna() on the item matrix, so pingouin's "
        "own nan_policy never sees a missing value)."
    )
    out["missingness_note"] = _missingness_note(out)
    return out


def _missingness_note(m: dict) -> str:
    """Numerically parameterized prose for the missingness caveat."""

    n_cc, n_in = m.get("n_complete_case"), m.get("n_input_rows")
    if not n_in or n_cc is None:
        return "item-level n unavailable"
    if n_cc == n_in:
        return (
            f"No missing data: all {n_in} analyzed trials contribute "
            f"({len(m.get('items_present', []))} indicators complete)."
        )
    parts = [
        f"Listwise deletion: computed on {n_cc} of {n_in} analyzed trials ({n_cc / n_in:.1%})."
    ]
    binding = m.get("binding_item")
    if binding:
        reason = m.get("binding_item_missing_reason")
        parts.append(
            f"'{binding}' is the binding indicator (n = {m['n_per_item'][binding]})"
            + (f"; {reason}" if reason else "")
            + "."
        )
    comp = m.get("complete_case_composition")
    if comp and comp.get("counts"):
        counts = comp["counts"]
        share = comp["share_of_complete_case"]
        base = comp["share_of_analytic_sample"]
        parts.append(
            f"Complete-case composition by {comp['group_col']}: "
            + ", ".join(
                f"{k} {counts[k]} ({share[k]:.1%})"
                for k in sorted(counts, key=lambda x: -counts[x])
            )
            + " (total-variation distance from the analytic sample "
            + f"{comp['total_variation_distance']})."
        )
        worst = m.get("most_over_represented_group")
        factor = m.get("most_over_represented_factor")
        if worst and factor is not None and factor >= _DOMINANCE_FACTOR:
            parts.append(
                f"'{worst}' supplies {share[worst]:.1%} of the complete-case rows against "
                f"{base[worst]:.1%} of the analytic sample ({factor}x over-represented, bar "
                f"{_DOMINANCE_FACTOR}x), so this statistic describes a sample dominated by that "
                "group."
            )
        elif worst and factor is not None:
            parts.append(
                "Complete-case composition tracks the analytic sample closely (largest "
                f"over-representation {factor}x for '{worst}', below the {_DOMINANCE_FACTOR}x "
                "reporting bar)."
            )
    elif comp:
        parts.append(comp.get("note", "composition not computable."))
    return " ".join(parts)


def _standardize(sub: pd.DataFrame):
    """Z-score the non-constant columns; returns (z, kept_columns) or (None, kept)."""

    sd = sub.std(ddof=0)
    keep = sd[sd > 0].index.tolist()
    if len(keep) < 2 or len(sub) < 2:
        return None, keep
    return (sub[keep] - sub[keep].mean()) / sub[keep].std(ddof=0), keep


def _alpha_value(sub: pd.DataFrame):
    """Cronbach's alpha on a raw item frame, or None when it is not finite.

    pingouin returns -inf (with a divide-by-zero RuntimeWarning) when the item covariance
    matrix sums to zero, which json.dumps would emit as the non-standard ``-Infinity``
    token. Non-finite alphas are reported as null instead.
    """

    import pingouin as pg

    z, keep = _standardize(sub)
    if z is None:
        return None, keep
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha, _ = pg.cronbach_alpha(data=z)
    a = float(alpha)
    return (a if math.isfinite(a) else None), keep


def between_condition_artifact_check(
    df: pd.DataFrame,
    columns: list[str],
    pooled_alpha: float | None,
    group_col: str = "condition",
    contrast_col: str = "defense",
) -> dict:
    """Is a high pooled alpha internal consistency, or just the treatment contrast?

    When every indicator is a two-level step (baseline ~0, all treated arms ~equal and
    high), the inter-item correlations reproduce the between-condition contrast and alpha
    approaches 1 with no within-condition covariation behind it. Diagnostic: recompute
    alpha on GROUP-MEAN-CENTERED items (the between-condition step removed) and within
    each contrast group.
    """

    present = [c for c in columns if c in df.columns]
    if len(present) < 2 or pooled_alpha is None:
        return {"note": "artifact check needs >= 2 present items and a pooled alpha"}
    if group_col not in df.columns:
        return {"note": f"artifact check needs the '{group_col}' column"}
    sub = df[present].apply(pd.to_numeric, errors="coerce").dropna()
    if not len(sub):
        return {"note": "no complete cases", "n_complete_case": 0, "n_groups": 0}
    grp = df.loc[sub.index, group_col]
    n_groups = int(grp.nunique())
    if len(sub) < 3 or n_groups < 2:
        return {
            "note": "artifact check needs >= 3 complete cases across >= 2 groups",
            "n_complete_case": int(len(sub)),
            "n_groups": n_groups,
        }
    centered = sub - sub.groupby(grp.to_numpy()).transform("mean")
    a_cen, keep_cen = _alpha_value(centered)
    within: dict = {}
    means: dict = {}
    if contrast_col in df.columns:
        contrast = df.loc[sub.index, contrast_col]
        for g, g_idx in sub.groupby(contrast.to_numpy()).groups.items():
            a_g, keep_g = _alpha_value(sub.loc[g_idx])
            within[str(g)] = {
                "alpha": round(a_g, 4) if a_g is not None else None,
                "n": int(len(g_idx)),
                "n_non_constant_items": len(keep_g),
            }
        means = sub.groupby(contrast.to_numpy()).mean().round(4).to_dict("index")
    out = {
        "group_col": group_col,
        "contrast_col": contrast_col,
        "n_groups": n_groups,
        "n_complete_case": int(len(sub)),
        "pooled_alpha": round(float(pooled_alpha), 4),
        "alpha_group_mean_centered": round(a_cen, 4) if a_cen is not None else None,
        "n_items_after_centering": len(keep_cen),
        "alpha_by_contrast_group": within,
        "item_means_by_contrast_group": {str(k): v for k, v in means.items()},
        "rule": (
            f"artifact = pooled alpha >= {_ALPHA_BAR:.2f} AND the {group_col}-mean-centered alpha "
            f"is computable AND < {_ALPHA_BAR:.2f}, i.e. the inter-item correlations reproduce "
            "the between-condition contrast rather than within-condition covariation among the "
            "indicators. The verdict is NULL (not false) when the centered alpha is not "
            "computable, and false when the pooled alpha is itself below the bar. This is a "
            "screening diagnostic, not a test statistic: there is no null distribution behind it."
        ),
        "method_note": (
            "alpha_group_mean_centered is a DIAGNOSTIC, not the reported alpha: items are "
            f"centered on their {group_col} means (one mean per group, df not adjusted) so only "
            "within-condition covariance among indicators remains. Centering can change which "
            "items are non-constant, so n_items_after_centering is reported alongside."
        ),
    }
    detail = " and ".join(
        f"within {g} it is {v['alpha']}"
        for g, v in sorted(within.items())
        if v.get("alpha") is not None
    )
    if a_cen is None:
        out["between_condition_artifact"] = None
        out["caveat"] = (
            f"Artifact check NOT COMPUTABLE: after centering on {group_col} means fewer than two "
            f"indicators retained non-zero variance (n_items_after_centering = "
            f"{out['n_items_after_centering']}), or the centered item covariance summed to zero, "
            "so no within-condition alpha exists. Read the pooled alpha as UNDIAGNOSED -- it is "
            "not evidence either way."
        )
    elif float(pooled_alpha) < _ALPHA_BAR:
        out["between_condition_artifact"] = False
        out["caveat"] = (
            f"The pooled alpha ({out['pooled_alpha']}) is itself below {_ALPHA_BAR:.2f}, so no "
            f"inflation claim arises; the {group_col}-mean-centered alpha is "
            f"{out['alpha_group_mean_centered']}" + (", " + detail if detail else "") + "."
        )
    elif a_cen < _ALPHA_BAR:
        out["between_condition_artifact"] = True
        out["caveat"] = (
            "CAVEAT - this pooled alpha is a BETWEEN-CONDITION artifact, not internal "
            "consistency. Each indicator is effectively a two-level step (static baseline at ~0, "
            "every MTD arm at a common elevated level), so the inter-item correlations reproduce "
            f"the static-vs-MTD contrast. Recomputed on {group_col}-mean-centered items the alpha "
            f"is {out['alpha_group_mean_centered']}"
            + (", " + detail if detail else "")
            + ". Do not read the pooled alpha as evidence that these indicators cohere into a "
            "single operational-cost dimension; it restates the treatment effect."
        )
    else:
        out["between_condition_artifact"] = False
        out["caveat"] = (
            "The pooled alpha is inflated by the static-vs-MTD step, but the composite also "
            f"holds up within condition ({group_col}-mean-centered alpha = "
            f"{out['alpha_group_mean_centered']} >= {_ALPHA_BAR:.2f}"
            + (", " + detail if detail else "")
            + "), so the indicators covary beyond the treatment contrast."
        )
    return out


def cronbach_alpha(
    df: pd.DataFrame,
    columns: list[str],
    *,
    group_col: str = "condition",
    contrast_col: str = "defense",
    missing_reason: str | None = None,
) -> dict:
    """Cronbach's alpha over standardized items (zero-variance items dropped).

    Missing data are handled by LISTWISE DELETION before pingouin is called, so the
    reported alpha describes only the complete-case rows; ``item_missingness`` makes the
    size and composition of that subset explicit, and ``between_condition_artifact_check``
    tests whether a high alpha is internal consistency or just the treatment contrast.

    Items that are ENTIRELY missing are dropped BEFORE the listwise-deletion filter.
    Without that, one all-NaN column (e.g. a derived item whose source is absent) would
    drop every row and silently replace the composite with an "insufficient observations"
    note rather than degrading to the items that are present. ``dropped_all_nan_items``
    records what was dropped; ``item_missingness`` still accounts for every REQUESTED
    item, so when it is non-empty its ``n_complete_case`` (over all requested items) is
    deliberately smaller than the ``n`` the reported alpha was computed on.
    """

    present = [c for c in columns if c in df.columns]
    dropped_all_nan = [
        c for c in present if not pd.to_numeric(df[c], errors="coerce").notna().any()
    ]
    cols = [c for c in present if c not in dropped_all_nan]
    missing = item_missingness(df, columns, group_col=group_col, missing_reason=missing_reason)
    sub = df[cols].apply(pd.to_numeric, errors="coerce").dropna() if cols else df.loc[:, []]
    if sub.shape[0] < 2 or sub.shape[1] < 2:
        return {
            "note": "insufficient items/observations",
            "n_items": int(sub.shape[1]),
            "dropped_all_nan_items": dropped_all_nan,
            "item_missingness": missing,
        }
    alpha, keep = _alpha_value(sub)
    if len(keep) < 2:
        return {
            "note": "fewer than 2 non-constant items",
            "n_items": len(keep),
            "dropped_all_nan_items": dropped_all_nan,
            "item_missingness": missing,
        }
    if alpha is None:
        return {
            "note": "alpha undefined (zero total item covariance)",
            "n_items": len(keep),
            "n": int(sub.shape[0]),
            "items": keep,
            "dropped_all_nan_items": dropped_all_nan,
            "item_missingness": missing,
        }
    import pingouin as pg

    z = (sub[keep] - sub[keep].mean()) / sub[keep].std(ddof=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        _, ci = pg.cronbach_alpha(data=z)
    interp = (
        "high internal consistency (indicators cohere -> composite justified)"
        if alpha >= _ALPHA_BAR
        else "low internal consistency (do not combine into one composite)"
    )
    return {
        "alpha": float(alpha),
        "ci95": [
            (float(ci[0]) if math.isfinite(float(ci[0])) else None),
            (float(ci[1]) if math.isfinite(float(ci[1])) else None),
        ],
        "n_items": int(z.shape[1]),
        "n": int(z.shape[0]),
        "items": keep,
        "interpretation": interp,
        "dropped_all_nan_items": dropped_all_nan,
        "item_missingness": missing,
        "between_condition_artifact_check": between_condition_artifact_check(
            df, cols, float(alpha), group_col=group_col, contrast_col=contrast_col
        ),
    }


def _icc_interpretation(icc: float | None) -> str:
    if icc is None or icc != icc:
        return "undefined"
    if icc >= 0.90:
        return "excellent (>= .90; Koo & Li, 2016)"
    if icc >= 0.75:
        return "good (>= .75)"
    if icc >= 0.50:
        return "moderate"
    return "poor"


def icc2(long_df: pd.DataFrame, targets: str, raters: str, ratings: str) -> dict:
    """ICC(2,1) for a genuine targets x raters layout (LIVE repeatability).

    ``targets`` are the re-measured configurations, ``raters`` are the independent
    repeated executions (occasions), and ``ratings`` is the continuous outcome. This
    is the estimator to run on the live repeatability data; it is NOT called on the
    offline dataset (see ``reliability_report``).
    """

    import pingouin as pg

    d = long_df[[targets, raters, ratings]].dropna()
    if d[targets].nunique() < 2 or d[raters].nunique() < 2:
        return {"note": "ICC(2,1) needs >= 2 targets and >= 2 raters/occasions"}
    # within-target variation is required for a non-degenerate ICC
    within = d.groupby(targets, observed=True)[ratings].nunique()
    if (within <= 1).all():
        return {"note": "no within-target variation (identical re-measurements) -> ICC undefined"}
    res = pg.intraclass_corr(data=d, targets=targets, raters=raters, ratings=ratings)
    # ICC(2,1) = two-way random, ABSOLUTE agreement, single measurement.
    # pingouin >= 0.5.4 labels this "ICC(A,1)" (McGraw & Wong); older uses "ICC2".
    row = res[res["Type"].isin(["ICC(A,1)", "ICC2"])]
    if not len(row):
        return {"note": "ICC(2,1) / ICC(A,1) row not returned by pingouin"}
    icc = float(row["ICC"].iloc[0])
    ci_col = next((c for c in ("CI95", "CI95%") if c in res.columns), None)
    ci = row[ci_col].iloc[0] if ci_col else None
    return {
        "ICC2_1": icc,
        "ci95": [float(ci[0]), float(ci[1])] if ci is not None else None,
        "interpretation": _icc_interpretation(icc),
        "pingouin_type": str(row["Type"].iloc[0]),
        "n_targets": int(d[targets].nunique()),
        "n_raters": int(d[raters].nunique()),
        "thresholds": _KOO_LI,
    }


def cohen_kappa(rater_a: list[int], rater_b: list[int]) -> dict:
    """Percent agreement + Cohen's kappa between two independent re-measurements.

    ``rater_a`` and ``rater_b`` are paired binary vectors from two independent
    re-executions of the same configurations (NOT one series compared to itself).
    """

    from sklearn.metrics import cohen_kappa_score

    a = [int(x) for x in rater_a]
    b = [int(x) for x in rater_b]
    if len(a) != len(b) or len(a) < 2:
        return {"note": "need >= 2 paired observations of equal length"}
    pct = float(np.mean([x == y for x, y in zip(a, b, strict=True)]))
    if len(set(a) | set(b)) < 2:
        return {
            "percent_agreement": pct,
            "cohen_kappa": None,
            "note": "single class -> kappa undefined",
        }
    return {"percent_agreement": pct, "cohen_kappa": float(cohen_kappa_score(a, b)), "n": len(a)}


def reliability_report(rep_rows: list[TrialRow] | pd.DataFrame) -> dict:
    """Measurement-reliability report.

    Offline, reliability is NOT estimable (deterministic mock, single configuration,
    one value per run, no independent re-measurement). This returns the explanation
    plus the live-study method, rather than a fabricated ICC/kappa of 1.0.
    """

    return {
        "estimable_offline": False,
        "measurement_reliability": (
            "NOT estimable from the offline pipeline: the mock attacker is deterministic and the "
            "repeatability runs re-execute a single configuration, so there is no measurement "
            "variation to partition and no independent re-measurement of the same realization to "
            "correlate. ICC(2,1) and kappa are computed in the LIVE study (R independent "
            "re-executions of each checksum-verified configuration across multiple configurations)."
        ),
        "computational_determinism": (
            "Separately, the parsing/correlation/metric routines are deterministic (identical "
            "inputs -> identical outputs), verified by the unit/integration tests. This is a "
            "software property, NOT reliability evidence, and is not offered as such."
        ),
        "live_methods": {
            "continuous_outcomes": "ICC(2,1) via analysis.reliability.icc2(long_df, targets, raters, ratings)",
            "binary_outcomes": "percent agreement + Cohen's kappa via analysis.reliability.cohen_kappa(a, b)",
        },
        "thresholds": _KOO_LI,
        "n_repeatability_runs": len(rep_rows),
    }
