"""Descriptive statistics by defense condition."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Outcomes summarized by default (present subset is used).
DEFAULT_METRICS = [
    "exploit_success",
    "exploitation_success",
    "time_to_compromise_s",
    "time_to_enumeration_s",
    "time_to_first_valid_exploit_s",
    "recon_accuracy_f1",
    "attack_path_churn",
    "attempts_per_success",
    "latency_overhead_ms",
    "cpu_overhead_pct",
    "memory_overhead_pct",
    "network_overhead_pct",
    "operational_event_volume",
    # Both orientations are summarized: the frozen column stores availability RETAINED
    # (higher = better) and the derived twin is the cost-oriented loss (higher = worse).
    "service_availability_impact",
    "service_availability_loss",
]


def describe_by_condition(df: pd.DataFrame, metrics: list[str] | None = None) -> pd.DataFrame:
    """Per-condition n, mean, SD, median, IQR for each metric (long format)."""

    metrics = [m for m in (metrics or DEFAULT_METRICS) if m in df.columns]
    records = []
    for cond, g in df.groupby("condition"):
        for m in metrics:
            s = pd.to_numeric(g[m], errors="coerce").dropna()
            q1, q3 = np.percentile(s, [25, 75]) if len(s) else (np.nan, np.nan)
            records.append(
                {
                    "condition": cond,
                    "metric": m,
                    "n": int(len(s)),
                    "mean": float(s.mean()) if len(s) else np.nan,
                    "sd": float(s.std(ddof=1)) if len(s) > 1 else np.nan,
                    "median": float(s.median()) if len(s) else np.nan,
                    "iqr": float(q3 - q1) if len(s) else np.nan,
                }
            )
    return pd.DataFrame.from_records(records)


def censoring_by_condition(df: pd.DataFrame, flag: str = "ttc_censored") -> pd.DataFrame:
    """Proportion of censored trials per condition for a censoring flag."""

    if flag not in df.columns:
        return pd.DataFrame(columns=["condition", "prop_censored", "n"])
    out = df.groupby("condition")[flag].agg(prop_censored="mean", n="count").reset_index()
    return out
