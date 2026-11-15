"""RQ5 (H05/Ha5): which MTD + deception configurations give the best
security-improvement-to-operational-cost profile?

The profile is reported on **two separate natural-unit scales** rather than collapsed
into one signed z-difference:

- **Security benefit** — ``security_benefit_s`` = RMST(condition) - RMST(static_baseline),
  the restricted mean time to compromise over the fixed observation window, in seconds.
- **Operational cost** — ``resource_overhead_index_pct``, the unweighted mean of
  cpu/memory/network overhead (all three share the unit "% increase over the static
  baseline", metrics_specification 3.2), in percentage points.

The previous single composite (``profile = z(time_to_compromise_s) - z(cpu_overhead_pct)``)
was removed because its implied exchange rate between seconds and percent is set by the
**arm allocation**, not by the data: for a variable that is constant within an arm with a
fraction p of rows in the minority arm, the minority z is exactly -sqrt((1-p)/p), so with
one static arm against five MTD arms the static cost z is -sqrt(5) = -2.236 by construction.
Recomputing the same composite on subsets of the same trials moves the static score from
+1.588 (as run) to +0.204 (static vs. combined only), and re-weighting the sample so static
holds half the rows moves it to +0.536 and reverses which arm ranks first.

**Inference.** Trials are clustered within ``sequence_id`` (8 attempts each). The primary
inference is therefore at the **cluster level**: per-sequence means, compared with a BCa
bootstrap CI on the difference (the estimand is a difference of restricted means), Welch's
t (unequal variances) and Mann-Whitney U (distribution-free), Bonferroni-adjusted over the
three planned contrasts. A trial-level homoscedastic random-intercept LMM is **not** used
for the headline: ``security_benefit_s`` has a 28-76% point mass at the observation-window
ceiling and the arm variances differ by a factor of ~2.2e4 (Levene p = 5.1e-19), so the
LMM's single pooled residual variance inflates the static EMMean SE to 81.4 s against a
cluster-level SE of 0.32 s while deflating the single-technique SE. The LMM is retained as
a clearly labelled secondary block, and only when it converges.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from analysis import clustered
from analysis.dataset import AVAILABILITY_LOSS, STATIC, availability_loss

SINGLE = {"network_shuffling", "port_rotation", "endpoint_mutation", "deceptive_presentation"}
COMBINED = "combined_mtd_deception"
_ORDER = ["static", "single", "combined"]

CLUSTER = "sequence_id"
RESOURCE_COST_ITEMS = ["cpu_overhead_pct", "memory_overhead_pct", "network_overhead_pct"]
COST_INDEX = "resource_overhead_index_pct"
SECURITY_BENEFIT = "security_benefit_s"

# Pre-registered trial ceiling (metrics_specification 4: ``trial_duration_limit_s``).
DEFAULT_HORIZON_S = 1800.0
# Pre-registered monitoring cadence (metrics_specification 3.2), used only to corroborate
# the observed time-to-compromise measurement grid -- never as a statistical tolerance.
DEFAULT_MONITORING_INTERVAL_S = 5.0
# Observed measurement grid of ``time_to_compromise_s``: every distinct uncensored value
# (35, 45, 55, 105, 115, 125, 135, 145, 155) is a multiple of 5 s, and so is the 1800 s
# ceiling. Reported as provenance; not used to decide dominance.
TTC_MEASUREMENT_GRID_S = 5.0
# Denominator floor for the seconds-per-percentage-point ratio. The static baseline's cost
# index is 0.0048 pp on the 2026-07-22 frozen run -- measurement noise around a quantity
# that is zero by construction -- so any denominator at or below 1.0 pp is treated as "no
# cost incurred" and the ratio is reported as null rather than as a spuriously large or
# zero number. This is deliberately NOT the same constant as the frontier tie tolerance.
MIN_COST_FOR_RATIO_PCT = 1.0
# Frontier tie band, in cluster-level standard errors of the between-condition difference.
# 1.96 = nominal 95%. The frontier is a descriptive ordering device, not a
# multiplicity-controlled test; the confirmatory inference is the three planned contrasts.
FRONTIER_Z = 1.96
FRONTIER_Z_SENSITIVITY = [0.0, 1.96, 2.935]
# Equivalence margin for the combined-vs-single operational-cost contrast, in percentage
# points of the resource-overhead index. 1.0 pp is 1.3% of the 77.15 pp that ANY MTD
# configuration already costs relative to static on the 2026-07-22 frozen run, and ~14x the
# cluster-level SE of a between-condition cost difference (0.074 pp) on that run.
# POST HOC, not pre-registered: the report also emits ``smallest_supporting_margin_pct``
# so a reader can apply their own margin.
COST_EQUIVALENCE_MARGIN_PCT = 1.0
BOOTSTRAP_RESAMPLES = 20000
BOOTSTRAP_SEED = 0


def _group(cond: str) -> str:
    """Pre-registered grouping: static / single-technique / combined."""

    if cond == STATIC:
        return "static"
    if cond == COMBINED:
        return "combined"
    return "single"


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _config_scalar(df: pd.DataFrame, col: str, default: float) -> tuple[float, str | None]:
    """A run-configuration constant, read only when it really is constant.

    Taking ``.max()`` over the column would let one stray cell redefine the observation
    window or the measurement grid for the whole analysis, so a non-constant column falls
    back to the pre-registered default and says so.
    """

    s = _num(df, col).dropna()
    if not len(s):
        return default, f"{col} absent or all-missing; using the pre-registered default {default}"
    uniq = sorted(float(v) for v in pd.unique(s))
    if len(uniq) == 1:
        return uniq[0], None
    return default, (
        f"{col} is not constant across trials ({uniq[:5]}); using the pre-registered "
        f"default {default}"
    )


def _mean_or_nan(s: pd.Series) -> float:
    return float(s.mean()) if s.notna().any() else float("nan")


def restricted_time_to_compromise(df: pd.DataFrame, horizon: float) -> pd.Series:
    """Per-trial ``min(time_to_compromise_s, horizon)``."""

    return _num(df, "time_to_compromise_s").clip(upper=horizon)


def resource_overhead_index(df: pd.DataFrame) -> pd.Series:
    """Per-trial mean of the unit-commensurable resource-overhead percentages."""

    cols = [c for c in RESOURCE_COST_ITEMS if c in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)


def _administrative_censoring(d: pd.DataFrame, horizon: float) -> dict:
    """Is every censored trial censored exactly at the observation window?

    ``restricted_time_to_compromise`` equals the RMST only under administrative censoring
    at the horizon. Nothing else in the pipeline checks that, so a future run with
    early-terminated trials would silently produce a downward-biased "RMST".
    """

    cen = _num(d, "ttc_censored") == 1
    ttc = _num(d, "time_to_compromise_s")
    vals = sorted(float(v) for v in pd.unique(ttc[cen].dropna().round(6)))
    verified = bool(cen.sum() == 0 or (len(vals) == 1 and vals[0] == float(horizon)))
    return {
        "verified": verified,
        "n_censored": int(cen.sum()),
        "censored_time_values": vals[:5],
        "rmst_method": (
            "sample_mean_exact (all censoring administrative at the observation window)"
            if verified
            else "sample_mean_BIASED_LOW (some trials censored before the window)"
        ),
        "note": (
            "mean(min(time_to_compromise_s, observation_window_s)) equals the restricted mean "
            "survival time exactly only when every censored trial is censored at the window. "
            "When verified is false the reported RMST is biased low and must be replaced by a "
            "Kaplan-Meier RMST before it is interpreted."
        ),
    }


def _cluster_table(d: pd.DataFrame, key: str, outcome: str) -> pd.DataFrame | None:
    """Per-level mean / sd / n / se of the per-``sequence_id`` means of ``outcome``."""

    if CLUSTER not in d.columns or outcome not in d.columns:
        return None
    per = (
        d.dropna(subset=[outcome])
        .groupby([CLUSTER, key], observed=True)[outcome]
        .mean()
        .reset_index()
    )
    if not len(per):
        return None
    tab = per.groupby(key, observed=True)[outcome].agg(["mean", "std", "count"])
    tab["se"] = tab["std"] / np.sqrt(tab["count"])
    return tab


def _bca_ci(a: np.ndarray, b: np.ndarray) -> tuple[list[float] | None, str]:
    """95% CI for ``mean(a) - mean(b)`` by bootstrap over cluster means."""

    if len(a) < 2 or len(b) < 2:
        return None, "unavailable (fewer than 2 clusters in an arm)"

    def stat(x: np.ndarray, y: np.ndarray, axis: int = -1) -> np.ndarray:
        return x.mean(axis=axis) - y.mean(axis=axis)

    for method in ("BCa", "percentile"):
        try:
            res = stats.bootstrap(
                (a, b),
                stat,
                n_resamples=BOOTSTRAP_RESAMPLES,
                method=method,
                rng=np.random.default_rng(BOOTSTRAP_SEED),
                vectorized=True,
            )
        except Exception:  # noqa: BLE001 - degenerate resamples fall through to percentile
            continue
        lo, hi = float(res.confidence_interval.low), float(res.confidence_interval.high)
        if np.isfinite(lo) and np.isfinite(hi):
            label = (
                f"{method} bootstrap over sequence means "
                f"({BOOTSTRAP_RESAMPLES} resamples, seed {BOOTSTRAP_SEED})"
            )
            return [lo, hi], label
    return None, "unavailable (degenerate bootstrap distribution)"


def _pair(a: np.ndarray, b: np.ndarray, g1: str, g2: str) -> dict:
    from analysis.effects import cohen_d

    ci, ci_method = _bca_ci(a, b)
    out: dict = {
        "g1": g1,
        "g2": g2,
        "n_clusters_g1": int(len(a)),
        "n_clusters_g2": int(len(b)),
        "diff": float(a.mean() - b.mean()) if len(a) and len(b) else float("nan"),
        "diff_orientation": f"{g1} minus {g2}",
        "ci95": ci,
        "ci_method": ci_method,
        "cohen_d": cohen_d(a, b),
    }
    if len(a) < 2 or len(b) < 2:
        out["note"] = "fewer than 2 sequences in an arm; no test computed"
        return out
    t, p = stats.ttest_ind(a, b, equal_var=False)
    out["welch_t"] = float(t)
    out["welch_p"] = float(p)
    try:
        u, pu = stats.mannwhitneyu(a, b, alternative="two-sided")
        out["mannwhitney_u"] = float(u)
        out["mannwhitney_p"] = float(pu)
    except Exception as exc:  # noqa: BLE001
        out["mannwhitney_note"] = f"unavailable: {exc}"
    return out


def _bonferroni(pairs: list[dict], src: str, dst: str) -> None:
    from statsmodels.stats.multitest import multipletests

    idx = [i for i, p in enumerate(pairs) if isinstance(p.get(src), float) and np.isfinite(p[src])]
    if not idx:
        return
    _, padj, _, _ = multipletests([pairs[i][src] for i in idx], method="bonferroni")
    for k, i in enumerate(idx):
        pairs[i][dst] = float(padj[k])


def _eta2_per_sequence(d: pd.DataFrame, outcome: str, levels: list[str]) -> float | None:
    from analysis.effects import eta_squared

    if CLUSTER not in d.columns:
        return None
    per = (
        d.dropna(subset=[outcome])
        .groupby([CLUSTER, "group"], observed=True)[outcome]
        .mean()
        .reset_index()
    )
    arrs = [per.loc[per["group"] == lv, outcome].to_numpy() for lv in levels]
    arrs = [a for a in arrs if len(a)]
    if len(arrs) < 2:
        return None
    v = eta_squared(arrs)
    return None if not np.isfinite(v) else float(v)


def _cluster_contrasts(d: pd.DataFrame, outcome: str, levels: list[str]) -> dict:
    """PRIMARY inference: planned contrasts on per-sequence means."""

    if CLUSTER not in d.columns:
        return {
            "outcome": outcome,
            "note": f"cluster-level contrasts unavailable: no {CLUSTER} column",
        }
    tab = _cluster_table(d, "group", outcome)
    if tab is None:
        return {"outcome": outcome, "note": "cluster-level contrasts unavailable: no observations"}
    per = (
        d.dropna(subset=[outcome])
        .groupby([CLUSTER, "group"], observed=True)[outcome]
        .mean()
        .reset_index()
    )

    def arr(level: str) -> np.ndarray:
        return per.loc[per["group"] == level, outcome].to_numpy(dtype=float)

    wanted = [("combined", "single"), ("combined", "static"), ("single", "static")]
    pairs = [_pair(arr(g1), arr(g2), g1, g2) for g1, g2 in wanted if g1 in levels and g2 in levels]
    _bonferroni(pairs, "welch_p", "welch_p_bonf")
    _bonferroni(pairs, "mannwhitney_p", "mannwhitney_p_bonf")
    for p in pairs:
        wp, mp = p.get("welch_p_bonf"), p.get("mannwhitney_p_bonf")
        p["significant_bonf"] = None if wp is None else bool(wp < 0.05)
        p["tests_agree"] = None if (wp is None or mp is None) else bool((wp < 0.05) == (mp < 0.05))
    named = {f"{p['g1']}_vs_{p['g2']}": p for p in pairs}
    return {
        "outcome": outcome,
        "unit_of_inference": "sequence (cluster) mean",
        "cluster_means": {
            str(lv): {
                "mean": float(r["mean"]),
                "sd": float(r["std"]) if np.isfinite(r["std"]) else None,
                "se": float(r["se"]) if np.isfinite(r["se"]) else None,
                "n_sequences": int(r["count"]),
            }
            for lv, r in tab.iterrows()
            if str(lv) in levels
        },
        "eta_squared_per_sequence": _eta2_per_sequence(d, outcome, levels),
        **named,
        "method_note": (
            "Primary inference for RQ5. Trials are clustered within sequence_id, so every "
            "quantity here is computed on per-sequence means (independent across sequences). "
            "The estimate and its interval come from a bootstrap CI on the difference of "
            "cluster means, because the estimand is a difference of restricted means; Welch's "
            "t (unequal variances) supplies the p-value and Mann-Whitney U a distribution-free "
            "corroboration. All three are computed for all three contrasts -- none is selected "
            "after seeing the others. Bonferroni is applied over the three planned contrasts, "
            "separately within each test family. Every contrast's 'diff' is g1 minus g2, as "
            "recorded in its diff_orientation field."
        ),
    }


def _between_sequence_share(d: pd.DataFrame, outcome: str) -> dict:
    """Share of WITHIN-CONDITION variance in ``outcome`` that lies BETWEEN sequences.

    A DATA-SIDE companion to the fit's ``random_intercept_var``: the random intercept is
    the model's estimate of essentially this quantity, so the two can be compared without
    trusting the fit. Computed on the frozen column, so it is invariant to the
    linear-algebra stack and to CSV float-parsing precision -- unlike ``converged``, which
    on a boundary fit is not (see ``secondary_reliability_note``).

    Reported WITH its null reference. Sequence means of m independent trials vary even at
    zero intraclass correlation, and their share of the total is then approximately 1/m,
    so a raw share is uninterpretable on its own: only ``share`` meaningfully above
    ``null_reference_at_zero_icc`` is evidence of genuine between-sequence clustering.
    """

    out: dict = {"share": None, "null_reference_at_zero_icc": None, "mean_cluster_size": None}
    if CLUSTER not in d.columns or "condition" not in d.columns:
        return out
    sub = pd.DataFrame({"y": _num(d, outcome), "c": d["condition"], "k": d[CLUSTER]}).dropna()
    n_clusters = int(sub["k"].nunique())
    if len(sub) < 3 or n_clusters < 2:
        return out
    m = len(sub) / n_clusters
    out["mean_cluster_size"] = float(m)
    out["null_reference_at_zero_icc"] = float(1.0 / m) if m > 0 else None
    # Remove the between-condition (treatment) mean first: the LMM's random intercept
    # partitions the residual variance, not the treatment effect.
    sub["y"] = sub["y"] - sub.groupby("c", observed=True)["y"].transform("mean")
    total = float(sub["y"].var(ddof=1))
    if not np.isfinite(total) or total <= 0:
        out["share"] = 0.0
        return out
    between = float(sub.groupby("k", observed=True)["y"].transform("mean").var(ddof=1))
    if np.isfinite(between):
        out["share"] = float(min(1.0, max(0.0, between / total)))
    return out


def _mixed_secondary(d: pd.DataFrame, outcome: str, levels: list[str]) -> dict:
    """SECONDARY, model-based contrasts. Estimates are emitted only when the LMM converges.

    The data-side cluster diagnostic is emitted on EVERY path, including the paths where
    no model is fitted at all. It is computed from the frozen column and never reads the
    fit object, so it is exactly the signal a reader needs when the fit fails or is
    withheld -- the case where a fit-derived ``random_intercept_var`` is unavailable or
    untrustworthy. Attaching it only on success would delete it precisely when it carries
    the most information, and would make the report's field set depend on a boundary-fit
    ``converged`` flag that is not stable across linear-algebra stacks (see
    ``secondary_reliability_note``).
    """

    # Computed once, before any return, so every branch reports the identical value.
    between_share = _between_sequence_share(d, outcome)
    share_key = "between_sequence_variance_share_within_condition"

    if CLUSTER not in d.columns:
        return {
            "outcome": outcome,
            "note": f"mixed model unavailable: no {CLUSTER} column",
            share_key: between_share,
        }
    fit = clustered.lmm_continuous(d, outcome, "C(group)")
    if "note" in fit:
        return {"outcome": outcome, "note": fit["note"], share_key: between_share}
    model = fit.pop("model", None)
    var_y = float(_num(d, outcome).var(ddof=1))
    re_var = fit.get("random_intercept_var")
    diag = {
        "converged": fit.get("converged"),
        "random_intercept_var": re_var,
        "outcome_variance": var_y if np.isfinite(var_y) else None,
        "random_intercept_var_ratio": (
            float(re_var / var_y)
            if (re_var is not None and np.isfinite(var_y) and var_y > 0)
            else None
        ),
        "marginal_r2_fixed": fit.get("marginal_r2_fixed"),
        share_key: between_share,
        "secondary_reliability_note": (
            "Read this block only alongside the diagnostics above it. "
            "between_sequence_variance_share_within_condition is computed from the FROZEN "
            "COLUMN, not from the fit: it is the share of WITHIN-CONDITION variance lying "
            "between sequences, the data-side analogue of what the random intercept models "
            "(note the denominators differ -- random_intercept_var_ratio is denominated on the "
            "TOTAL outcome variance, which still contains the treatment effect, so the two are "
            "not the same number). It is reported with the share that m independent trials per "
            "sequence would produce at zero intraclass correlation "
            "(null_reference_at_zero_icc = 1/m). A share at or "
            "below that reference is what INDEPENDENT trials look like -- it is not evidence of "
            "clustering, and a random intercept fitted to it has nothing to estimate, so the "
            "model degenerates towards OLS on pseudoreplicated trials and its EMMean SEs "
            "understate the clustered uncertainty. 'converged' is a FIT-HEALTH flag, not a "
            "data-side routing signal: on a fit sitting at the boundary of the parameter space "
            "it can flip between runs differing only in last-ULP CSV float parsing or BLAS "
            "threading -- observed on the 2026-07-22 resource_overhead_index_pct fit, which "
            "reports converged=false under pandas' default parser and converged=true under "
            "float_precision='round_trip', with an identical random_intercept_var_ratio of "
            "1.25e-05 either way. Nothing in the reported inference depends on which way it "
            "lands: the primary contrasts are the cluster-level ones, and the ONLY equivalence "
            "claim in RQ5 comes from cost_equivalence (a CI/TOST test), never from a large "
            "p-value here."
        ),
    }
    if model is None:
        return {"outcome": outcome, "note": "LMM returned no fitted model object", **diag}
    if fit.get("converged") is False:
        return {
            "outcome": outcome,
            **diag,
            "note": (
                "Secondary mixed model did NOT converge; its EMMeans, Wald test and pairwise "
                "p-values are withheld rather than reported. Convergence is a property of the "
                "fit, not of any p-value, so withholding here does not condition the reported "
                "inference on its own result. The primary cluster-level contrasts are unaffected."
            ),
        }
    try:
        emmeans, omnibus, pairs = clustered.emmeans_pairwise(model, "group", levels)
    except Exception as exc:  # noqa: BLE001
        return {"outcome": outcome, **diag, "note": f"EMMeans contrasts unavailable: {exc}"}

    def pick(g1: str, g2: str) -> dict:
        for p in pairs:
            if {p["g1"], p["g2"]} == {g1, g2}:
                out = dict(p)
                out["diff_orientation"] = f"{out['g1']} minus {out['g2']}"
                return out
        return {"note": "contrast unavailable"}

    return {
        "outcome": outcome,
        "label": "SECONDARY (model-based); the primary inference is the cluster-level contrast",
        **diag,
        "emmeans": emmeans,
        "omnibus_wald": omnibus,
        "combined_vs_single": pick("combined", "single"),
        "combined_vs_static": pick("combined", "static"),
        "single_vs_static": pick("single", "static"),
        "orientation_note": (
            "These model-based contrasts are named for the PAIR, not for the sign: EMMeans "
            "orders g1/g2 itself, so a key such as combined_vs_single can carry a diff of "
            "'single minus combined'. Read diff_orientation on each contrast before comparing "
            "it with the identically-named key under primary_contrasts_*_per_sequence, whose "
            "diff is always g1 minus g2 in the key's own order."
        ),
    }


def _variance_heterogeneity(d: pd.DataFrame, outcome: str) -> dict:
    from analysis import assumptions

    v = d.groupby("group", observed=True)[outcome].var(ddof=1).dropna()
    ratio = float(v.max() / v.min()) if len(v) >= 2 and v.min() > 0 else None
    per = (
        d.dropna(subset=[outcome])
        .groupby([CLUSTER, "group"], observed=True)[outcome]
        .mean()
        .reset_index()
        if CLUSTER in d.columns
        else None
    )
    return {
        "outcome": outcome,
        "levene_trial_level": assumptions.levene_across_groups(d, outcome, group="group"),
        "levene_cluster_level": (
            assumptions.levene_across_groups(per, outcome, group="group")
            if per is not None
            else {"note": f"no {CLUSTER} column"}
        ),
        "variance_ratio_max_over_min": ratio,
        "ceiling_share_by_condition": {
            str(k): _mean_or_nan(_num(g, "ttc_censored"))
            for k, g in d.groupby("condition", observed=True)
        },
        "note": (
            "The random-intercept LMM assumes ONE residual variance across arms. When the "
            "outcome has a large point mass at the observation-window ceiling in some arms and "
            "none in others, the arm variances diverge (on the 2026-07-22 frozen run by a "
            "factor of ~2.2e4) and the LMM's EMMean SEs are pooled, not arm-specific: on that "
            "run it reported 81.4 s for static against a cluster-level SE of 0.32 s, and 40.7 s "
            "for single-technique against a cluster-level SE of 48.6 s -- inflated in one "
            "direction and deflated in the other. Compare "
            "secondary_contrasts_security_mixed.emmeans[*].se against "
            "primary_contrasts_security_per_sequence.cluster_means[*].se for the current run. "
            "This is why the primary inference is cluster-level."
        ),
    }


def _cost_source_text(d: pd.DataFrame) -> str:
    if "operational_cost_source" not in d.columns:
        return ""
    vals = sorted(set(d["operational_cost_source"].dropna().astype(str)))
    return f" (operational_cost_source: {', '.join(vals)})" if vals else ""


def _cost_not_recorded(d: pd.DataFrame, what: str, columns: list[str]) -> dict:
    """The block for a per-trial cost analysis when the run recorded no per-trial cost."""

    names = ", ".join(columns[:-1]) + f" and {columns[-1]}" if len(columns) > 1 else columns[0]
    verb = "are" if len(columns) > 1 else "is"
    return {
        "computed": False,
        "reason": "per_trial_cost_not_recorded",
        "note": (
            f"{what} was not run. The dataset records no per-trial cost: {names} {verb} empty "
            f"in every trial{_cost_source_text(d)}. Operational cost is measured per technique "
            "by the reconfiguration benchmark: read rq5.benchmark_profile "
            "(by_technique.median_cost_s and cost_tests). No equivalence or cost claim is made "
            "from the per-trial columns."
        ),
    }


def _equivalence(d: pd.DataFrame, outcome: str, g1: str, g2: str, margin: float) -> dict:
    """CI-based equivalence for a contrast expected to be null."""

    if outcome in d.columns and not _num(d, outcome).notna().any():
        return {
            "outcome": outcome,
            **_cost_not_recorded(d, "The equivalence test (TOST)", RESOURCE_COST_ITEMS),
        }
    if CLUSTER not in d.columns:
        return {"note": f"equivalence test unavailable: no {CLUSTER} column"}
    per = (
        d.dropna(subset=[outcome])
        .groupby([CLUSTER, "group"], observed=True)[outcome]
        .mean()
        .reset_index()
    )
    a = per.loc[per["group"] == g1, outcome].to_numpy(dtype=float)
    b = per.loc[per["group"] == g2, outcome].to_numpy(dtype=float)
    if len(a) < 2 or len(b) < 2:
        return {"note": "equivalence test unavailable: fewer than 2 sequences in an arm"}
    ci, ci_method = _bca_ci(a, b)
    out: dict = {
        "contrast": f"{g1}_vs_{g2}",
        "outcome": outcome,
        "diff": float(a.mean() - b.mean()),
        "diff_orientation": f"{g1} minus {g2}",
        "ci95": ci,
        "ci_method": ci_method,
        "equivalence_margin_pct": float(margin),
        "margin_provenance": (
            "POST HOC, not pre-registered. On the 2026-07-22 frozen run 1.0 percentage point is "
            "1.3% of the ~77.15 pp resource overhead that ANY MTD configuration already costs "
            "relative to the static baseline, and ~14x the cluster-level SE of a "
            "between-condition cost difference (0.074 pp) on that run. "
            "smallest_supporting_margin_pct lets a reader substitute their own."
        ),
    }
    try:
        from statsmodels.stats.weightstats import ttost_ind

        p_tost, _, _ = ttost_ind(a, b, -margin, margin, usevar="unequal")
        out["tost_p"] = float(p_tost)
    except Exception as exc:  # noqa: BLE001
        out["tost_note"] = f"TOST unavailable: {exc}"
    if ci is not None:
        smallest = float(max(abs(ci[0]), abs(ci[1])))
        out["smallest_supporting_margin_pct"] = smallest
        out["equivalent"] = bool(smallest < margin)
    else:
        out["equivalent"] = None
    out["note"] = (
        "Equivalence is concluded from the CI lying strictly inside the margin (and the TOST "
        "p-value), NOT from a non-significant difference test. A non-significant Wald p is "
        "absence of evidence, not evidence of equivalence."
    )
    return out


def _ratio(benefit: float, cost: float) -> float | None:
    if not np.isfinite(benefit) or not np.isfinite(cost) or cost <= MIN_COST_FOR_RATIO_PCT:
        return None
    return float(benefit / cost)


def _rows(d: pd.DataFrame, key: str, baseline_rmst: float) -> list[dict]:
    out: list[dict] = []
    for lvl, g in d.groupby(key, observed=True):
        rmst = _mean_or_nan(g["_rmst_t"])
        cost = _mean_or_nan(g[COST_INDEX])
        benefit = rmst - baseline_rmst
        lat = _num(g, "latency_overhead_ms")
        out.append(
            {
                key: str(lvl),
                "n": int(len(g)),
                "n_sequences": int(g[CLUSTER].nunique()) if CLUSTER in g.columns else None,
                "compromise_rate": _mean_or_nan(_num(g, "exploitation_success")),
                "prop_censored": _mean_or_nan(_num(g, "ttc_censored")),
                "rmst_time_to_compromise_s": rmst,
                SECURITY_BENEFIT: float(benefit),
                COST_INDEX: cost,
                "operational_event_volume": _mean_or_nan(_num(g, "operational_event_volume")),
                AVAILABILITY_LOSS: _mean_or_nan(_num(g, AVAILABILITY_LOSS)),
                "latency_overhead_ms": float(lat.mean()) if lat.notna().any() else None,
                "latency_overhead_n": int(lat.notna().sum()),
                "security_per_resource_cost_s_per_pct": _ratio(benefit, cost),
            }
        )
    return out


def _nondominated(
    names: list[str], ben: pd.DataFrame, cost: pd.DataFrame, z: float
) -> tuple[list[str], dict[str, list[str]]]:
    def cmp(i: str, j: str, tab: pd.DataFrame) -> int:
        diff = float(tab.loc[i, "mean"] - tab.loc[j, "mean"])
        se_i, se_j = float(tab.loc[i, "se"]), float(tab.loc[j, "se"])
        se = float(np.sqrt(np.nan_to_num(se_i) ** 2 + np.nan_to_num(se_j) ** 2))
        if abs(diff) <= z * se:
            return 0
        return 1 if diff > 0 else -1

    nondom: list[str] = []
    dominated: dict[str, list[str]] = {}
    for i in names:
        doms = []
        for j in names:
            if i == j:
                continue
            better_b = cmp(j, i, ben)  # j's benefit vs i's (higher is better)
            better_c = -cmp(j, i, cost)  # j's cost vs i's (lower is better)
            if better_b >= 0 and better_c >= 0 and (better_b > 0 or better_c > 0):
                doms.append(j)
        if doms:
            dominated[i] = sorted(doms)
        else:
            nondom.append(i)
    return sorted(nondom), dominated


def _frontier(d: pd.DataFrame, key: str, baseline_rmst: float) -> dict:
    ben = _cluster_table(d, key, "_rmst_t")
    cost = _cluster_table(d, key, COST_INDEX)
    if ben is None or cost is None:
        return {
            "note": (
                "efficiency frontier unavailable: cluster-level means could not be computed for "
                f"one or both axes (needs a {CLUSTER} column, time_to_compromise_s and at least "
                "one of cpu/memory/network_overhead_pct)"
            )
        }
    names = sorted(set(ben.index.astype(str)) & set(cost.index.astype(str)))
    ben.index = ben.index.astype(str)
    cost.index = cost.index.astype(str)
    names = [
        n for n in names if np.isfinite(ben.loc[n, "mean"]) and np.isfinite(cost.loc[n, "mean"])
    ]
    if not names:
        return {"note": "efficiency frontier unavailable: no condition has both axes"}
    cost_incurring = [n for n in names if float(cost.loc[n, "mean"]) > MIN_COST_FOR_RATIO_PCT]

    nd, dom = _nondominated(names, ben, cost, FRONTIER_Z)
    nd_point, _ = _nondominated(names, ben, cost, 0.0)
    nd_excl, _ = _nondominated(cost_incurring, ben, cost, FRONTIER_Z)
    sens = {f"{z:g}": _nondominated(names, ben, cost, z)[0] for z in FRONTIER_Z_SENSITIVITY}
    return {
        "pareto_nondominated": nd,
        "pareto_nondominated_point_estimate": nd_point,
        "pareto_nondominated_excluding_reference": nd_excl,
        "dominated_by": dom,
        "uncertainty_changes_frontier": bool(sorted(nd) != sorted(nd_point)),
        "frontier_z": FRONTIER_Z,
        "frontier_z_sensitivity": sens,
        "stable_above_z1_96": bool(
            len({tuple(v) for k, v in sens.items() if float(k) >= FRONTIER_Z}) <= 1
        ),
        "baseline_rmst_time_to_compromise_s": baseline_rmst,
        "note": (
            "Pareto dominance in (restricted mean time to compromise: higher is better; "
            "resource_overhead_index_pct: lower is better). Two arms are treated as TIED on an "
            "axis when the difference of their per-sequence means is inside "
            "frontier_z x SE(difference), with the SE computed at the cluster (sequence) level "
            "-- a scale-relative band derived from the data, not a fixed magnitude. "
            "pareto_nondominated_point_estimate uses zero tolerance and is reported so the "
            "effect of that judgement is auditable. The static baseline is non-dominated on the "
            "cost axis BY CONSTRUCTION -- every overhead metric is defined as a change from it, "
            "so nothing can be cheaper -- which is a statement about the reference point, not a "
            "security ranking; pareto_nondominated_excluding_reference restricts the frontier to "
            "arms that actually incur cost and is the set Ha5 is evaluated on."
        ),
    }


def _cost_discrimination(d: pd.DataFrame) -> dict:
    """Do the cost indicators distinguish MTD configurations, or only MTD on/off?"""

    from analysis.effects import eta_squared

    items = [
        c
        for c in [
            *RESOURCE_COST_ITEMS,
            "latency_overhead_ms",
            "operational_event_volume",
            AVAILABILITY_LOSS,
            COST_INDEX,
        ]
        if c in d.columns
    ]
    out: dict = {}
    mtd = d[d["condition"] != STATIC]
    for c in items:
        allg = [_num(g, c).dropna().to_numpy() for _, g in d.groupby("condition", observed=True)]
        mtdg = [_num(g, c).dropna().to_numpy() for _, g in mtd.groupby("condition", observed=True)]
        e_all = eta_squared([a for a in allg if len(a)])
        e_mtd = eta_squared([a for a in mtdg if len(a)])
        out[c] = {
            "eta2_across_all_conditions": (
                None if not np.isfinite(e_all) else round(float(e_all), 4)
            ),
            "eta2_within_mtd_conditions": (
                None if not np.isfinite(e_mtd) else round(float(e_mtd), 4)
            ),
        }
    return {
        "by_indicator": out,
        "note": (
            "On the 2026-07-22 frozen run every operational-cost indicator is a near-perfect "
            "dichotomy between MTD-enabled and static (eta^2 ~ .999 across all conditions) but "
            "explains almost nothing among the MTD configurations themselves (eta^2 <= .02 "
            "within the MTD arms); read eta2_across_all_conditions against "
            "eta2_within_mtd_conditions above for the current run. Where that pattern holds the "
            "instrument's cost side measures WHETHER MTD is enabled, not WHICH MTD "
            "configuration is used. Two consequences are reported rather than hidden: (a) the "
            "combined-vs-single cost equivalence is expected, not a discovery; (b) "
            "security_per_resource_cost_s_per_pct is a monotone rescaling of security_benefit_s "
            "among the cost-incurring arms, because its denominator is effectively constant "
            "there, so it adds no ordering information beyond the benefit ranking. A cost "
            "measure that discriminates among MTD configurations is a stated instrument "
            "limitation, not a finding."
        ),
    }


def _single_composition(d: pd.DataFrame, baseline_rmst: float) -> dict:
    """The pooled 'single-technique' arm is a mixture; say so with numbers."""

    sub = d[d["group"] == "single"]
    if not len(sub):
        return {"note": "no single-technique conditions present"}
    tab = _cluster_table(sub, "condition", "_rmst_t")
    if tab is None:
        return {"note": "per-condition breakdown unavailable"}
    per_cond = {
        str(lv): {
            "security_benefit_s": float(r["mean"] - baseline_rmst),
            "se": float(r["se"]) if np.isfinite(r["se"]) else None,
            "n_sequences": int(r["count"]),
        }
        for lv, r in tab.iterrows()
    }
    vals = [v["security_benefit_s"] for v in per_cond.values()]
    return {
        "conditions": per_cond,
        "benefit_range_s": [float(min(vals)), float(max(vals))] if vals else None,
        "note": (
            "The pre-registered 'single-technique' level pools four conditions whose security "
            "benefit spans the full range of the study. Its pooled mean is a mixture and "
            "describes no individual technique; the combined-vs-single contrast is the "
            "pre-registered grouped comparison, and the per-condition frontier is the "
            "technique-level answer."
        ),
    }


def _interaction(d: pd.DataFrame) -> dict:
    from analysis.effects import eta_squared

    if "cpu_overhead_pct" not in d.columns or "time_to_compromise_s" not in d.columns:
        return {"note": "group x overhead interaction unavailable: required columns missing"}
    if not _num(d, "cpu_overhead_pct").notna().any():
        return _cost_not_recorded(d, "The group x overhead interaction", ["cpu_overhead_pct"])
    if CLUSTER not in d.columns:
        return {"note": f"group x overhead interaction unavailable: no {CLUSTER} column"}
    inter = clustered.lmm_continuous(d, "time_to_compromise_s", "C(group) * cpu_overhead_pct")
    if "note" in inter:
        out: dict = {"note": inter["note"]}
    else:
        inter.pop("model", None)
        sd_y = float(_num(d, "time_to_compromise_s").std(ddof=0) or 0.0)
        sd_x = float(_num(d, "cpu_overhead_pct").std(ddof=0) or 0.0)
        terms = {k: v for k, v in inter.get("terms", {}).items() if ":cpu_overhead_pct" in k}
        for v in terms.values():
            v["standardized_beta"] = (
                float(v["coef"] * sd_x / sd_y) if sd_y > 0 and sd_x > 0 else None
            )
        out = {
            "converged": inter.get("converged"),
            "random_intercept_var": inter.get("random_intercept_var"),
            "interaction_terms": terms,
            "standardized_beta_note": (
                "standardized_beta = coef * SD(cpu_overhead_pct) / SD(time_to_compromise_s), the "
                "standardized regression coefficient the analysis plan reports for this term."
            ),
        }
    groups = [
        _num(g, "cpu_overhead_pct").dropna().to_numpy()
        for _, g in d.groupby("group", observed=True)
    ]
    eta = eta_squared([g for g in groups if len(g)]) if len(groups) >= 2 else float("nan")
    out["group_overhead_collinearity_eta2"] = None if not np.isfinite(eta) else float(eta)
    out["label"] = "DIAGNOSTIC ONLY -- weakly identified; not interpreted as an elasticity"
    out["collinearity_note"] = (
        "Operational cost is a near-deterministic function of the arm (see "
        "group_overhead_collinearity_eta2; on the 2026-07-22 frozen run every MTD arm sits at "
        "~77% CPU overhead and static at ~0%, with only measurement noise within an arm). The "
        "group x overhead interaction is therefore weakly identified -- its slopes are estimated "
        "almost entirely from within-arm measurement noise -- and must NOT be read as an "
        "efficacy-per-cost elasticity. The efficiency question is answered by the frontier and "
        "the planned contrasts instead."
    )
    return out


def rq5(df: pd.DataFrame) -> dict:
    """Security-benefit / operational-cost profile, reported on separate scales."""

    if "condition" not in df.columns or not len(df):
        return {"note": "RQ5 unavailable: empty dataset", "reference_condition": STATIC}

    d = df.copy()
    d["group"] = d["condition"].map(_group)
    if AVAILABILITY_LOSS not in d.columns:
        d[AVAILABILITY_LOSS] = availability_loss(d)
    horizon, horizon_note = _config_scalar(d, "trial_duration_limit_s", DEFAULT_HORIZON_S)
    interval, interval_note = _config_scalar(
        d, "monitoring_interval_s", DEFAULT_MONITORING_INTERVAL_S
    )
    d["_rmst_t"] = restricted_time_to_compromise(d, horizon)
    d[COST_INDEX] = resource_overhead_index(d)

    base = d.loc[d["condition"] == STATIC, "_rmst_t"]
    if not base.notna().any():
        return {
            "note": (
                "RQ5 unavailable: no static_baseline trials with a time-to-compromise, so the "
                "security-improvement-over-baseline profile is undefined."
            ),
            "reference_condition": STATIC,
        }
    baseline_rmst = float(base.mean())
    d[SECURITY_BENEFIT] = d["_rmst_t"] - baseline_rmst

    levels = [g for g in _ORDER if g in set(d["group"])]
    config_notes = [n for n in (horizon_note, interval_note) if n]
    unexpected = sorted(set(d["condition"].astype(str)) - {STATIC, COMBINED} - SINGLE)
    if unexpected:
        config_notes.append(
            f"conditions not in the pre-registered design were grouped as single-technique: "
            f"{unexpected}"
        )

    return {
        "reference_condition": STATIC,
        "observation_window_s": horizon,
        "config_notes": config_notes,
        "unexpected_conditions": unexpected,
        "measurement_grid": {
            "ttc_grid_s": TTC_MEASUREMENT_GRID_S,
            "monitoring_interval_s_observed": interval,
            "agrees": bool(interval == TTC_MEASUREMENT_GRID_S),
            "note": (
                "The time-to-compromise measurement grid is a property of the timestamps, not "
                "of the telemetry sampling cadence; monitoring_interval_s is reported only to "
                "show that the two agree. Neither is used as a statistical tolerance."
            ),
        },
        "administrative_censoring": _administrative_censoring(d, horizon),
        "baseline_rmst_time_to_compromise_s": baseline_rmst,
        "by_condition": _rows(d, "condition", baseline_rmst),
        "by_group": _rows(d, "group", baseline_rmst),
        "primary_contrasts_security_per_sequence": _cluster_contrasts(d, SECURITY_BENEFIT, levels),
        "primary_contrasts_cost_per_sequence": _cluster_contrasts(d, COST_INDEX, levels),
        "cost_equivalence": _equivalence(
            d, COST_INDEX, "combined", "single", COST_EQUIVALENCE_MARGIN_PCT
        ),
        "security_variance_heterogeneity": _variance_heterogeneity(d, SECURITY_BENEFIT),
        "secondary_contrasts_security_mixed": _mixed_secondary(d, SECURITY_BENEFIT, levels),
        "secondary_contrasts_cost_mixed": _mixed_secondary(d, COST_INDEX, levels),
        "efficiency_frontier": _frontier(d, "condition", baseline_rmst),
        "efficiency_frontier_by_group": _frontier(d, "group", baseline_rmst),
        "cost_indicator_discrimination": _cost_discrimination(d),
        "single_group_composition": _single_composition(d, baseline_rmst),
        "regression_group_x_overhead_mixed": _interaction(d),
        "metric_definitions": {
            "rmst_time_to_compromise_s": (
                "Restricted mean time to compromise over the fixed observation window: "
                "mean(min(time_to_compromise_s, observation_window_s)). Exact only when every "
                "censored trial is censored at that window -- checked and reported under "
                "administrative_censoring."
            ),
            SECURITY_BENEFIT: (
                "RMST(condition) - RMST(static_baseline), in seconds. Zero for the reference "
                "condition by construction. Monotone in the survival function, so a condition "
                "whose survival curve dominates the baseline's is guaranteed a strictly higher "
                "score, and invariant to how many arms were allocated to each group."
            ),
            COST_INDEX: (
                "Unweighted mean of cpu/memory/network_overhead_pct. All three share one unit "
                "(% increase over the static baseline's utilisation, per metrics_specification "
                "3.2), so the mean is unit-coherent and requires no standardisation."
            ),
            AVAILABILITY_LOSS: (
                "1 - service_availability_impact: the fraction of legitimate requests NOT "
                "answered within the timeout. Higher is worse. The dataset column "
                "service_availability_impact stores availability RETAINED (higher is better) "
                "despite its name (metrics_specification 3.4); it is not renamed, and the "
                "analysis layer derives this cost-oriented twin for cost-side use."
            ),
            "security_per_resource_cost_s_per_pct": (
                "security_benefit_s / resource_overhead_index_pct: seconds of additional "
                "restricted survival per percentage point of mean resource overhead. null when "
                f"the denominator is <= {MIN_COST_FOR_RATIO_PCT} pp -- on the 2026-07-22 frozen "
                "run the static baseline's index is 0.0048 pp, measurement noise around a cost "
                "that is zero by construction, and 0/0 is undefined rather than infinitely "
                "favourable. See cost_indicator_discrimination: among the cost-incurring arms "
                "the denominator is effectively constant, so this ratio adds no ordering "
                "information."
            ),
        },
        "composite_note": (
            "Security benefit and operational cost are reported on SEPARATE natural-unit scales "
            "rather than collapsed into one signed z-difference. They are not commensurable: any "
            "single scalar silently fixes an exchange rate between seconds of survival and "
            "percent resource overhead. The previous composite (profile = z(time_to_compromise_s) "
            "- z(cpu_overhead_pct)) set that exchange rate from the pooled SDs, which are a "
            "function of how many arms were allocated to each group rather than of the data: "
            "with 1 static arm against 5 MTD arms the static cost z is exactly -sqrt(5) = "
            "-2.236, and on the 2026-07-22 frozen run recomputing it on static-vs-combined "
            "trials alone moves the static score from +1.588 to +0.204, while re-weighting the "
            "sample so static holds half the rows moves it to +0.536 and reverses which arm "
            "ranks first. Every quantity reported here is invariant to that allocation."
        ),
        "static_baseline_interpretation": (
            "The static baseline is genuinely the cheapest configuration: it is the zero-cost, "
            "zero-benefit ORIGIN of the trade-off plane by construction, because every overhead "
            "metric is defined as a change from it. It is simultaneously the LEAST secure arm "
            "(100% compromise, 0% censoring, lowest restricted mean time to compromise). It is "
            "therefore not ranked as a 'best profile': benefit per unit cost is undefined at "
            "zero cost. Ha5 is evaluated among the arms that actually incur operational cost."
        ),
        "inference_note": (
            "Primary = primary_contrasts_*_per_sequence (cluster level). Secondary = "
            "secondary_contrasts_*_mixed (trial-level random-intercept LMM), reported only when "
            "it converges and always alongside security_variance_heterogeneity, which documents "
            "why its pooled SEs are not arm-specific. Equivalence claims come only from "
            "cost_equivalence."
        ),
    }
