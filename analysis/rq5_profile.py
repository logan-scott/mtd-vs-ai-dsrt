"""RQ5 (H05/Ha5): the security-improvement-to-operational-cost profile, with uncertainty.

The profile of a technique is its security benefit divided by its operational cost:

- **benefit** = RMST(technique) - RMST(static_baseline), the restricted mean time to
  compromise over [0, horizon_s], in seconds. The H05 decision uses ``DECISION_HORIZON_S``
  (300 s, decided 2026-10-28 and approved by the chair, matching RQ1's time endpoint); the
  trial window is reported as a sensitivity (``pipeline._rq5_block``). Called without a
  horizon, the function uses the trial window;
- **cost** = the median wall-clock duration of one reconfiguration in the reconfiguration-cost
  benchmark (``docs/results/benchmark/reconfig_cost.csv``, 20 replicates per technique) -- the
  measured cost, not the per-trial overhead columns of the archived trial datasets, which were
  produced by the offline telemetry stand-in (``harness.telemetry``) and are not measurements.

``analysis.rq5`` reports the profile's two scales separately and contrasts them on the
per-trial cost columns; until 2026-10-27 nothing tested the benefit/cost profile itself, so a
decision on H05 had no test behind it. ``benchmark_profile`` supplies one: a bootstrap that
resamples whole attacker sequences within each condition (benefit; the 8 attempts of a
sequence stay together) and benchmark replicates within each technique (cost), recomputing
every profile in every replicate. The planned contrasts are the combined condition against
the single-technique mean (H05's comparison) and against each single technique (secondary,
Holm-adjusted). The static baseline's profile is 0 by construction (benefit is measured
relative to it), so the combined-vs-static comparison is definitional and is not tested.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analysis import clustered
from analysis.dataset import STATIC
from analysis.fdr import holm

COMBINED = "combined_mtd_deception"
SINGLE = ("endpoint_mutation", "network_shuffling", "port_rotation", "deceptive_presentation")
N_BOOT = 10000
SEED = clustered.BOOT_SEED_DEFAULT
DEFAULT_HORIZON_S = 1800.0
#: The H05 decision's benefit horizon (2026-10-28, approved by the chair): RQ1's 300 s.
DECISION_HORIZON_S = 300.0

_PROFILE_NOTE = (
    "profile = security benefit (s) / median reconfiguration cost (s): seconds of added "
    "restricted mean time-to-compromise per second of reconfiguration. 95% intervals are "
    "percentile bootstrap intervals; each replicate resamples whole sequences within every "
    "condition (static baseline included, since every benefit is measured against it) and "
    "benchmark replicates within every technique, then recomputes all profiles. p-values are "
    "two-sided percentile-bootstrap p for a zero difference, 2 x min(share <= 0, share >= 0) "
    "with the (count + 1)/(B + 1) correction, floored at 2/(B + 1): report a floored p as "
    "'p < .001'. The benefit scale is horizon-dependent (benefit is close to horizon x the "
    "share of trials never compromised), so every profile scales with horizon_s."
)
_HORIZON_NOTE = (
    "RMST as mean(min(T, horizon_s)) is exact here because every censored trial is censored at "
    "the trial window (administrative_censoring), so none is censored before horizon_s."
)
_H05_NOTE = (
    "H05's comparison is the combined condition against the single-technique conditions. The "
    "primary contrast uses the mean of the four single-technique profiles (the comparator the "
    "dissertation reports); the per-technique contrasts are secondary and Holm-adjusted. "
    "Ha5 predicts the combined profile is MORE favorable (a positive difference); the direction "
    "fields say which way each contrast points."
)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_benchmark(path: str | Path) -> pd.DataFrame:
    """The per-replicate benchmark rows that completed (returncode 0, not timed out)."""

    b = pd.read_csv(path)
    need = {"technique", "wall_clock_s"}
    if not need <= set(b.columns):
        raise ValueError(f"{path} is not a reconfiguration-cost CSV (needs {sorted(need)})")
    ok = pd.Series(True, index=b.index)
    if "returncode" in b.columns:
        ok &= pd.to_numeric(b["returncode"], errors="coerce").fillna(1) == 0
    if "timed_out" in b.columns:
        ok &= ~b["timed_out"].astype(str).str.lower().isin({"true", "1"})
    out = b.loc[ok, ["technique", "wall_clock_s"]].copy()
    out["wall_clock_s"] = pd.to_numeric(out["wall_clock_s"], errors="coerce")
    return out.dropna()


def _ci(draws: np.ndarray) -> list[float] | None:
    b = draws[np.isfinite(draws)]
    if not b.size:
        return None
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def _p(draws: np.ndarray) -> float | None:
    from analysis.separation import boot_p  # ties at zero count in both tails

    return boot_p(draws)


def cost_column_provenance(df: pd.DataFrame) -> dict:
    """Where the trial dataset's per-trial cost columns came from, if the run recorded it."""

    col = "operational_cost_source"
    if col not in df.columns:
        return {
            "recorded": False,
            "note": (
                "The dataset does not record operational_cost_source (runs collected before "
                "2026-09-12). Those runs filled the per-trial cpu/memory/network/latency/"
                "availability columns from the offline telemetry stand-in (harness.telemetry), "
                "so they are NOT measurements: do not cite the cost-side results of the rq5 "
                "block computed from them. The measured cost is the reconfiguration benchmark "
                "(rq5.benchmark_profile)."
            ),
        }
    vals = df[col].astype(str).value_counts().to_dict()
    measured = set(vals) <= {"measured"}
    return {
        "recorded": True,
        "values": {str(k): int(v) for k, v in vals.items()},
        "all_measured": bool(measured),
        "note": (
            "per-trial cost columns are measured"
            if measured
            else "some per-trial cost rows are synthetic or not measured; cite only the benchmark"
        ),
    }


def benchmark_profile(
    df: pd.DataFrame,
    bench: pd.DataFrame,
    *,
    horizon: float | None = None,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    bench_provenance: dict | None = None,
) -> dict:
    """Per-technique benefit, cost and profile with bootstrap intervals; planned contrasts."""

    from analysis.rq5 import _administrative_censoring, _config_scalar

    need = {"condition", "sequence_id", "time_to_compromise_s", "ttc_censored"}
    if not need <= set(df.columns):
        return {"note": f"trial dataset lacks {sorted(need - set(df.columns))}"}
    window, _ = _config_scalar(df, "trial_duration_limit_s", DEFAULT_HORIZON_S)
    if horizon is None:
        horizon = window
    horizon = float(horizon)
    # The censoring check is a property of the data: every censored trial censored at the trial
    # window. It makes mean(min(T, h)) the exact RMST for any h up to the window.
    censoring = _administrative_censoring(df, window)
    if not censoring["verified"]:
        return {
            "note": (
                "RMST as mean(min(T, horizon)) is exact only when no trial is censored before "
                "the horizon; this check requires every censored trial to be censored at the "
                "trial window, which is not the case here, so the profile was not computed."
            ),
            "administrative_censoring": censoring,
        }
    if not (0.0 < horizon <= window):
        return {
            "note": (
                f"horizon {horizon} s is outside (0, trial window {window} s]; RMST beyond the "
                "follow-up is not identified, so the profile was not computed."
            ),
            "horizon_s": horizon,
            "trial_window_s": float(window),
        }
    techniques = [COMBINED, *SINGLE]
    present = set(df["condition"].astype(str))
    missing_trials = [c for c in [STATIC, *techniques] if c not in present]
    bench_counts = bench.groupby("technique")["wall_clock_s"].size().to_dict()
    missing_bench = [t for t in techniques if bench_counts.get(t, 0) < 2]
    if missing_trials or missing_bench:
        return {
            "note": "conditions missing from the trial data or the benchmark",
            "missing_from_trials": missing_trials,
            "missing_from_benchmark": missing_bench,
        }

    rt = pd.to_numeric(df["time_to_compromise_s"], errors="coerce").clip(upper=horizon)
    in_profile = df["condition"].astype(str).isin([STATIC, *techniques])
    n_missing = int((rt.isna() & in_profile).sum())
    if n_missing:
        # Dropping them would bias RMST (a censored trial with no time is a horizon-length
        # trial), and the censoring check above cannot see a missing time.
        return {
            "note": (
                f"{n_missing} trial(s) in the profile conditions have no time_to_compromise_s; "
                "RMST needs a time for every trial, so the profile was not computed."
            ),
            "n_trials_missing_time": n_missing,
        }
    seq = (
        df.loc[in_profile]
        .assign(_rt=rt[in_profile])
        .groupby(["condition", "sequence_id"], observed=True)["_rt"]
        .agg(["sum", "count"])
    )
    rng = np.random.default_rng(seed)

    def rmst_draws(cond: str) -> tuple[float, np.ndarray, int]:
        s = seq.loc[cond]
        sums, counts = s["sum"].to_numpy(float), s["count"].to_numpy(float)
        k = len(sums)
        idx = rng.integers(0, k, size=(n_boot, k))
        return float(sums.sum() / counts.sum()), sums[idx].sum(1) / counts[idx].sum(1), k

    rmst_static, rmst_static_b, k_static = rmst_draws(STATIC)
    per: dict[str, dict] = {}
    profile_b: dict[str, np.ndarray] = {}
    for t in techniques:
        rmst, rmst_b, k = rmst_draws(t)
        vals = bench.loc[bench["technique"] == t, "wall_clock_s"].to_numpy(float)
        cost = float(np.median(vals))
        cost_b = np.median(vals[rng.integers(0, len(vals), size=(n_boot, len(vals)))], axis=1)
        benefit, benefit_b = rmst - rmst_static, rmst_b - rmst_static_b
        profile_b[t] = benefit_b / cost_b
        per[t] = {
            "rmst_s": rmst,
            "security_benefit_s": benefit,
            "security_benefit_ci95": _ci(benefit_b),
            "median_cost_s": cost,
            "median_cost_ci95": _ci(cost_b),
            "profile": benefit / cost,
            "profile_ci95": _ci(profile_b[t]),
            "n_sequences": int(k),
            "n_benchmark_replicates": int(len(vals)),
        }

    single_mean = float(np.mean([per[t]["profile"] for t in SINGLE]))
    single_mean_b = np.mean(np.vstack([profile_b[t] for t in SINGLE]), axis=0)
    comb, comb_b = per[COMBINED]["profile"], profile_b[COMBINED]

    def contrast(other: float, other_b: np.ndarray) -> dict:
        diff_b = comb_b - other_b
        ratio_ok = bool(np.all(other_b > 0))
        out = {
            "difference_combined_minus_other": comb - other,
            "difference_ci95": _ci(diff_b),
            "p": _p(diff_b),
            "direction": "combined MORE favorable" if comb > other else "combined LESS favorable",
            "ratio_combined_over_other": (comb / other) if other > 0 else None,
            "ratio_ci95": _ci(comb_b / other_b) if ratio_ok else None,
        }
        if not ratio_ok:
            out["ratio_note"] = "the comparator's bootstrap profile reaches 0 or below; no ratio CI"
        return out

    primary = {
        "comparator": "mean of the four single-technique profiles",
        "comparator_value": single_mean,
        "comparator_ci95": _ci(single_mean_b),
        **contrast(single_mean, single_mean_b),
    }
    secondary = {t: contrast(per[t]["profile"], profile_b[t]) for t in SINGLE}
    adj = holm({t: v["p"] for t, v in secondary.items() if v["p"] is not None})
    for t, v in secondary.items():
        v["p_holm"] = adj.get(t)

    costs = {
        t: bench.loc[bench["technique"] == t, "wall_clock_s"].to_numpy(float)
        for t in sorted(bench["technique"].unique())
    }
    mtd_costs = [costs[t] for t in techniques if t in costs]
    med = {t: float(np.median(costs[t])) for t in techniques}

    def kw(groups: list[np.ndarray]) -> dict:
        res = stats.kruskal(*groups)
        n, k = int(sum(len(g) for g in groups)), len(groups)
        h = float(res.statistic)
        return {
            "H": h,
            "df": k - 1,
            "p": float(res.pvalue),
            "n": n,
            # eta^2_H = (H - k + 1) / (n - k) (Tomczak & Tomczak, 2014)
            "eta_squared": (h - k + 1) / (n - k) if n > k else None,
        }

    return {
        "horizon_s": float(horizon),
        "trial_window_s": float(window),
        "horizon_note": _HORIZON_NOTE,
        "administrative_censoring": censoring,
        "rmst_static_s": rmst_static,
        "n_sequences_static": int(k_static),
        "by_technique": per,
        "h05_primary_contrast": primary,
        "per_technique_contrasts": secondary,
        "combined_vs_static": {
            "status": "definitional, not tested",
            "note": (
                "The static baseline's benefit is 0 by construction (benefit is measured "
                "relative to it), so any technique with a positive benefit has a more favorable "
                "profile than static."
            ),
        },
        "cost_tests": {
            "kruskal_wallis_five_mtd_techniques": kw(mtd_costs),
            "kruskal_wallis_with_static": (
                kw([costs[t] for t in (STATIC, *techniques)]) if STATIC in costs else None
            ),
            "max_over_min_median_cost_five_mtd": max(med.values()) / min(med.values()),
            "orders_of_magnitude_five_mtd": float(np.log10(max(med.values()) / min(med.values()))),
        },
        "n_boot": int(n_boot),
        "seed": int(seed),
        "benchmark": bench_provenance or {},
        "note": _PROFILE_NOTE,
        "h05_note": _H05_NOTE,
    }
