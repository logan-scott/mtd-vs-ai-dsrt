"""Survival analysis of service uptime (not time-to-compromise).

Using the service-uptime event log emitted by the harness:
- Kaplan-Meier estimate of the uptime (time-to-failure) survival function per
  condition.
- Multivariate log-rank test comparing uptime across conditions.
- Cox proportional-hazards regression of the failure hazard on technique and
  reconfiguration cadence, with a Schoenfeld-residual proportional-hazards check.

Degrades gracefully when there are no failures (e.g., a static baseline that never
goes down) or too few events to fit a model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def uptime_dataframe(uptime_events: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(uptime_events)


def _kaplan_meier(df: pd.DataFrame) -> dict:
    from lifelines import KaplanMeierFitter

    out: dict = {}
    for cond, g in df.groupby("condition"):
        kmf = KaplanMeierFitter().fit(g["time_to_failure_s"], g["failure_observed"])
        median = kmf.median_survival_time_
        # lifelines returns +inf (NOT NaN) when the KM curve never reaches S(t)=0.5 -- e.g. a
        # static baseline that never fails. ``pd.isna(inf)`` is False, so the old guard let a
        # bare ``Infinity`` token reach the report writer, which strict JSON parsers reject.
        # ``write_report`` now nulls it and logs the substitution, but "null" alone cannot tell
        # a reader "undefined" from "not measured". Report it as an explicitly UNDEFINED median.
        defined = bool(pd.notna(median) and np.isfinite(median))
        out[str(cond)] = {
            "n": int(len(g)),
            "failures": int(g["failure_observed"].sum()),
            "median_uptime_s": float(median) if defined else None,
            "median_uptime_defined": defined,
        }
    return out


def _logrank(df: pd.DataFrame) -> dict:
    from lifelines.statistics import multivariate_logrank_test

    if df["condition"].nunique() < 2 or df["failure_observed"].sum() == 0:
        return {"note": "insufficient events/conditions for log-rank"}
    lr = multivariate_logrank_test(df["time_to_failure_s"], df["condition"], df["failure_observed"])
    return {"test_statistic": float(lr.test_statistic), "p": float(lr.p_value)}


def _cox(df: pd.DataFrame) -> dict:
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test

    if df["failure_observed"].sum() < 5 or df["technique"].nunique() < 2:
        return {"note": "insufficient events for Cox regression"}
    dummies = pd.get_dummies(df["technique"], prefix="tech", drop_first=True).astype(int)
    design = pd.concat(
        [
            df[["time_to_failure_s", "failure_observed"]].reset_index(drop=True),
            dummies.reset_index(drop=True),
        ],
        axis=1,
    )
    # drop zero-variance covariates
    for col in list(dummies.columns):
        if design[col].std(ddof=0) == 0:
            design = design.drop(columns=col)
    try:
        cph = CoxPHFitter(penalizer=0.1).fit(
            design, duration_col="time_to_failure_s", event_col="failure_observed"
        )
        hr = {k: float(v) for k, v in cph.hazard_ratios_.items()}
        # Schoenfeld-residual proportional-hazards check (module function form).
        ph = proportional_hazard_test(cph, design, time_transform="rank")
        min_ph_p = float(ph.summary["p"].min()) if len(ph.summary) else None
        return {
            "hazard_ratios": hr,
            "min_schoenfeld_p": min_ph_p,
            "proportional_hazards_ok": None if min_ph_p is None else bool(min_ph_p > 0.05),
            "concordance": float(cph.concordance_index_),
        }
    except Exception as exc:  # pragma: no cover - degenerate survival data
        return {"note": f"Cox model unavailable: {exc}"}


def survival_uptime(uptime_events: list[dict]) -> dict:
    """Run KM + log-rank + Cox on the service-uptime event log."""

    df = uptime_dataframe(uptime_events)
    if df.empty or "time_to_failure_s" not in df.columns:
        return {"note": "no uptime events"}
    return {
        "kaplan_meier": _kaplan_meier(df),
        "logrank": _logrank(df),
        "cox": _cox(df),
    }
