"""Clustered / mixed-effects estimators shared by the RQ analyses.

Trials are clustered within ``sequence_id`` (8 attempts each, retained context), so
between-condition comparisons must model that non-independence rather than assume
independent trials. This module provides the primary clustered estimators the
revised analysis plan specifies, each degrading gracefully on degenerate input:

- ``binary_clustered`` — logit **GLMM** (random intercept for the cluster) with a
  **GEE** (exchangeable) fallback; returns odds ratios + 95% CIs per term.
- ``cox_cluster`` — Cox PH with **cluster-robust** (sequence) variance, a practical
  stand-in for a shared-frailty Cox (lifelines has no gamma frailty); hazard ratios.
- ``lmm_continuous`` / ``lmm_log_event_time`` — linear mixed models (random
  intercept for the cluster).
- ``glm_cluster_or`` — one odds ratio from a logit GLM with sequence-clustered robust SEs
  (the RQ3 attempt-effect summaries and the within-block drift check).
- ``per_sequence_ttest`` — Welch's t / Mann-Whitney U on per-sequence summaries
  (independent across sequences) for the labeled **sensitivity** analyses.
- ``rmst_difference`` / ``hazard_profile`` / ``rmst_by_condition`` — the **PH-robust**
  survival layer used when the Schoenfeld screen rejects proportional hazards. All
  inference here is a **sequence cluster bootstrap** stratified on the design's
  condition factor; nothing depends on a model-based covariance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from analysis.degeneracy import tolerance as _scale_tolerance
from analysis.degeneracy import zero_variance_skip
from analysis.effects import cohen_d

CLUSTER = "sequence_id"
# Design stratum resampled by the sequence cluster bootstrap: the study allocates a fixed
# number of sequences to each condition, so a replicate must reproduce that allocation.
BOOT_STRATA = "condition"
BOOT_DEFAULT = 2000
BOOT_SEED_DEFAULT = 20260722
MAX_HAZARD_INTERVALS = 8
PH_TRANSFORMS = ("rank", "km", "identity", "log")
_GROUPCOL = "condition"


def _skip(d: pd.DataFrame, outcome: str, label: str | None = None) -> dict | None:
    """Zero-variance screen keyed on the outcome column, before any model is fit.

    Returns the skip payload (``note``/``skipped``/``skipped_reason``/
    ``variance_profile``) when the outcome carries no resolvable variance, else
    ``None``. See ``analysis.degeneracy`` for the tolerance and its defence.
    """

    groupcol = _GROUPCOL if _GROUPCOL in d.columns else None
    return zero_variance_skip(d, outcome, groupcol, label=label)


# --------------------------------------------------------------------------- #
# Binary: logit GLMM (random intercept) with GEE fallback
# --------------------------------------------------------------------------- #


def _glmm_terms(d: pd.DataFrame, outcome: str, rhs: str, group: str) -> dict | None:
    """Logit GLMM via variational Bayes; None if unavailable/degenerate.

    ``fit_vb`` seeds its variational parameters from the global ``np.random`` state, which
    otherwise makes the reported ORs/p-values non-reproducible run-to-run. We fix the seed
    (and restore the global state afterwards, so nothing else is perturbed) so the analysis
    is deterministic — identical trial data yields a byte-identical ``analysis_report.json``.
    This changes neither the data, the model, nor the estimand; it only pins the VB
    approximation's random initialization to a reproducible value.
    """

    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    model = BinomialBayesMixedGLM.from_formula(
        f"{outcome} ~ {rhs}", {"cluster": f"0 + C({group})"}, d
    )
    _rng_state = np.random.get_state()
    np.random.seed(0)
    try:
        res = model.fit_vb()
    finally:
        np.random.set_state(_rng_state)
    names = list(res.model.exog_names)
    terms: dict = {}
    for i, nm in enumerate(names):
        if nm == "Intercept":
            continue
        mean, sd = float(res.fe_mean[i]), float(res.fe_sd[i])
        if not (np.isfinite(mean) and np.isfinite(sd)) or sd <= 0:
            return None
        z = mean / sd
        terms[nm] = {
            "odds_ratio": float(np.exp(mean)),
            "ci95": [float(np.exp(mean - 1.96 * sd)), float(np.exp(mean + 1.96 * sd))],
            "p": float(2.0 * (1.0 - stats.norm.cdf(abs(z)))),
        }
    return terms or None


def _gee_terms(d: pd.DataFrame, outcome: str, rhs: str, group: str) -> dict:
    """Logit GEE (exchangeable working correlation clustered on `group`)."""

    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    m = smf.gee(
        f"{outcome} ~ {rhs}",
        groups=group,
        data=d,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable(),
    ).fit()
    ci = m.conf_int()
    terms: dict = {}
    for nm in m.params.index:
        if nm == "Intercept":
            continue
        terms[nm] = {
            "odds_ratio": float(np.exp(m.params[nm])),
            "ci95": [float(np.exp(ci.loc[nm, 0])), float(np.exp(ci.loc[nm, 1]))],
            "p": float(m.pvalues[nm]),
        }
    return terms


#: Every slope fitted with sequence-clustered standard errors must rest on at least this many
#: informative sequences -- sequences in which the outcome departs from the condition's most
#: common value (decided 2026-10-29, before any model was fitted to re-run data). A condition's own
#: slope needs that many of its own; a pooled slope (one common slope with condition fixed
#: effects) needs that many across the conditions it pools. With fewer, the clustered standard
#: error is biased far downward; with one informative sequence the condition's cluster
#: covariance has rank 1 and its slope looks exact, so a test that compares the conditions'
#: own slopes rejects on noise. ``scripts/h03_inclusion_calibration.py`` checks the threshold
#: by running H03's code on simulated data with no attempt effect.
MIN_INFORMATIVE_SEQUENCES = 20


def informative_sequences(d: pd.DataFrame, outcome: str, group: str = CLUSTER) -> dict:
    """How many trials and sequences depart from the outcome's most common value.

    For a binary outcome that is the minority outcome; for a restricted time it is every trial
    not at the most common time (for a condition that is rarely compromised, the compromised
    trials). ``enough`` applies ``MIN_INFORMATIVE_SEQUENCES``.
    """

    y = pd.to_numeric(d[outcome], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(y)
    yv = np.round(y[ok], 9)
    if not len(yv):
        return {
            "n_trials_departing": 0,
            "n_sequences_departing": 0,
            "min_sequences_required": MIN_INFORMATIVE_SEQUENCES,
            "enough": False,
        }
    # The most common value (ties: the smallest); positional, so duplicate index labels in
    # ``d`` cannot pull in other rows.
    mode = float(pd.Series(yv).mode().iloc[0])
    dep = yv != mode
    n_seq = int(pd.Series(d[group].to_numpy()[ok][dep]).nunique())
    return {
        "most_common_value": mode,
        "n_trials_departing": int(dep.sum()),
        "n_sequences_departing": n_seq,
        "min_sequences_required": MIN_INFORMATIVE_SEQUENCES,
        "enough": bool(n_seq >= MIN_INFORMATIVE_SEQUENCES),
    }


def too_few_informative_note(info: dict) -> str:
    """Why a condition gets no clustered slope of its own."""

    return (
        f"too few informative sequences: the outcome departs from its most common value in "
        f"{info.get('n_sequences_departing', 0)} sequences "
        f"({info.get('n_trials_departing', 0)} trials); a condition's own clustered slope needs "
        f"{MIN_INFORMATIVE_SEQUENCES}, or its standard error collapses"
    )


def too_few_pooled_note(n_sequences: int) -> str:
    """Why a pooled slope (one common slope over several conditions) is not fitted."""

    return (
        f"too few informative sequences across the pooled conditions: the outcome departs from "
        f"each condition's most common value in {n_sequences} sequences in all; a pooled "
        f"clustered slope needs {MIN_INFORMATIVE_SEQUENCES}"
    )


def glm_cluster_or(
    d: pd.DataFrame, outcome: str, rhs: str, term: str, group: str = CLUSTER
) -> dict:
    """Odds ratio for one ``term`` of a logit GLM with cluster-robust (sandwich) SEs.

    The independence working model with ``group``-clustered standard errors. Its point
    estimates equal those of a GEE with an independence working correlation; its standard
    errors carry statsmodels' small-sample correction G/(G-1) x (N-1)/(N-K), so p-values
    differ slightly from a GEE's. It is the estimator of the dissertation's RQ3 post hoc
    sensitivity, and the RQ3 attempt-effect summaries and the within-block drift check use it
    so that those numbers come from one model family. Rows with a missing value in the model
    are dropped by the caller.

    A term that is not identified -- perfect or quasi-complete separation, or a non-finite
    standard error -- returns a note instead of an odds ratio, never a number like 1e203.
    """

    import warnings

    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m = smf.glm(f"{outcome} ~ {rhs}", d, family=sm.families.Binomial()).fit(
                cov_type="cluster", cov_kwds={"groups": pd.factorize(d[group])[0]}
            )
        b, se = float(m.params[term]), float(m.bse[term])
        ci, p = m.conf_int().loc[term], float(m.pvalues[term])
    except Exception as exc:  # noqa: BLE001 - degenerate subset
        return {"note": f"cluster-robust logit unavailable: {exc}"}
    separated = any("separation" in type(w.message).__name__.lower() for w in caught) or any(
        "separation" in str(w.message).lower() for w in caught
    )
    if separated or not (np.isfinite(b) and np.isfinite(se) and np.isfinite(p)) or abs(b) > 20:
        return {
            "note": (
                f"{term} is not identified in this subset (perfect or quasi-complete "
                "separation, or a non-finite standard error); no odds ratio is reported"
            ),
            "n": int(m.nobs),
            "n_clusters": int(d[group].nunique()),
        }
    return {
        "odds_ratio": float(np.exp(b)),
        "ci95": [float(np.exp(ci.iloc[0])), float(np.exp(ci.iloc[1]))],
        "p": p,
        "n": int(m.nobs),
        "n_clusters": int(d[group].nunique()),
    }


def binary_clustered(d: pd.DataFrame, outcome: str, rhs: str, group: str = CLUSTER) -> dict:
    """Logit GLMM (primary) with GEE fallback; odds ratios + 95% CIs per term."""

    d = d.dropna(subset=[outcome]).copy()
    # `nunique` is also fooled by float noise (recon_accuracy_f1 has a 5.55e-17 gap
    # between two nominally identical values in the confirmatory dataset), so a
    # range-with-tolerance screen is strictly better than a distinct-value count.
    skip = _skip(d, outcome)
    if skip is not None:
        return skip
    try:
        terms = _glmm_terms(d, outcome, rhs, group)
        if terms is not None:
            return {
                "method": "logit_glmm",
                "terms": terms,
                "inference_note": (
                    "GLMM via variational Bayes (BinomialBayesMixedGLM.fit_vb); mean-field VB can "
                    "underestimate fixed-effect posterior SD, so Wald CIs/p may be mildly "
                    "anti-conservative. The GEE fallback (cluster-robust) corroborates borderline "
                    "effects."
                ),
            }
    except Exception:  # noqa: BLE001 - GLMM is optional; fall back to GEE
        pass
    try:
        return {"method": "gee_fallback", "terms": _gee_terms(d, outcome, rhs, group)}
    except Exception as exc:  # noqa: BLE001
        return {"note": f"clustered binary model unavailable: {exc}"}


# --------------------------------------------------------------------------- #
# Censored time: cluster-robust Cox + LMM on log event time
# --------------------------------------------------------------------------- #


def cox_cluster(
    d: pd.DataFrame, duration: str, censor_flag: str, rhs: str, cluster: str = CLUSTER
) -> dict:
    """Cox PH with cluster-robust (sequence) variance; hazard ratios per term.

    ``event = 1 - censor_flag`` (a verified event vs. right-censored at the window).
    Cluster-robust SEs account for within-sequence correlation and stand in for a
    shared-frailty Cox (not natively available in lifelines).
    """

    from lifelines import CoxPHFitter

    rhs_vars = [v.strip() for v in rhs.split("+")]
    cols = [duration, censor_flag, cluster, *rhs_vars]
    d = d.dropna(subset=[c for c in cols if c in d.columns]).copy()
    d = d[d[duration] > 0]
    d["_event"] = (1 - d[censor_flag]).astype(int)
    if d["_event"].sum() < 3 or len(d) < 5:
        return {"note": "insufficient events for Cox"}
    design = d[[duration, "_event", cluster, *rhs_vars]]

    def _fit(penalizer: float):
        cph = CoxPHFitter(penalizer=penalizer)
        cph.fit(design, duration_col=duration, event_col="_event", cluster_col=cluster, formula=rhs)
        return cph

    # Report the unpenalized max-partial-likelihood estimate; only fall back to a
    # tiny ridge if the unpenalized fit fails (e.g., quasi-separation), and disclose.
    penalizer_used = 0.0
    try:
        cph = _fit(0.0)
    except Exception:  # noqa: BLE001
        try:
            cph = _fit(0.01)
            penalizer_used = 0.01
        except Exception as exc:  # noqa: BLE001
            return {"note": f"Cox model unavailable: {exc}"}

    ci = cph.confidence_intervals_
    hrs = {
        term: {
            "hazard_ratio": float(np.exp(cph.params_[term])),
            "ci95": [float(np.exp(ci.loc[term].iloc[0])), float(np.exp(ci.loc[term].iloc[1]))],
            "p": float(cph.summary.loc[term, "p"]),
        }
        for term in cph.params_.index
    }
    # Schoenfeld-residual proportional-hazards check (spec: all time-to-event models).
    ph = ph_diagnostics(cph, design)
    per_term = ph.get("per_term_min_p_any_transform", {})
    for term in hrs:
        tp = per_term.get(term)
        hrs[term]["ph_min_p"] = tp
        hrs[term]["ph_valid"] = None if tp is None else bool(tp > 0.05)
        hrs[term]["ph_validity_label"] = (
            _PH_LABEL_UNKNOWN if tp is None else (_PH_LABEL_OK if tp > 0.05 else _PH_LABEL_VIOLATED)
        )
    return {
        "method": "cox_cluster_robust",
        "hazard_ratios": hrs,
        "n_events": int(d["_event"].sum()),
        "penalizer": penalizer_used,
        "proportional_hazards": ph,
        "estimand_note": (
            "MARGINAL (population-average) HR with cluster-robust (sequence) variance; a practical "
            "stand-in for a subject-specific shared-frailty Cox (no gamma frailty in lifelines). "
            "Interpret as the marginal, not conditional, hazard ratio."
        ),
    }


def lmm_log_event_time(
    d: pd.DataFrame, duration: str, censor_flag: str, rhs: str, group: str = CLUSTER
) -> dict:
    """LMM on the log event time of the *uncensored* trials (random intercept)."""

    import statsmodels.formula.api as smf

    unc = d[d[censor_flag] == 0].dropna(subset=[duration]).copy()
    if unc[group].nunique() < 2 or len(unc) < 5:
        return {"note": "insufficient uncensored trials"}
    unc["_log_t"] = np.log(unc[duration].clip(lower=1e-6))
    skip = _skip(unc, "_log_t", label=f"log({duration})")
    if skip is not None:
        return skip
    try:
        m = smf.mixedlm(f"_log_t ~ {rhs}", unc, groups=unc[group]).fit(
            method="lbfgs", reml=True, disp=False
        )
        return {
            "terms": {
                nm: {"coef": float(m.params[nm]), "p": float(m.pvalues[nm])}
                for nm in m.params.index
                if nm not in ("Intercept", "Group Var")
            },
            "converged": bool(m.converged),
            "n_uncensored": int(len(unc)),
        }
    except Exception as exc:  # noqa: BLE001
        return {"note": f"log-time LMM unavailable: {exc}"}


# --------------------------------------------------------------------------- #
# PH-robust survival: Kaplan-Meier primitives, sequence cluster bootstrap, RMST
# --------------------------------------------------------------------------- #


def _km_steps(t: np.ndarray, e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier step function: ``(distinct times, S(t))``. Pure numpy, no fitter."""

    keep = np.isfinite(t) & np.isfinite(e) & (t >= 0)
    t, e = t[keep], e[keep]
    if t.size == 0:
        return np.empty(0), np.empty(0)
    order = np.argsort(t, kind="mergesort")
    t, e = t[order], e[order]
    uniq, first = np.unique(t, return_index=True)
    at_risk = t.size - first
    events = np.add.reduceat(e, first)
    with np.errstate(divide="ignore", invalid="ignore"):
        surv = np.cumprod(np.where(at_risk > 0, 1.0 - events / at_risk, 1.0))
    return uniq, surv


def km_rmst(durations, events, horizon: float) -> float:
    """Restricted mean survival time: exact integral of the KM curve on ``[0, horizon]``."""

    uniq, surv = _km_steps(np.asarray(durations, dtype=float), np.asarray(events, dtype=float))
    if uniq.size == 0 or not np.isfinite(horizon) or horizon <= 0:
        return float("nan")
    grid = np.concatenate(([0.0], uniq))
    sv = np.concatenate(([1.0], surv))
    inside = grid <= horizon
    grid, sv = grid[inside], sv[inside]
    grid = np.concatenate((grid, [float(horizon)]))
    return float(np.sum(sv * np.diff(grid)))


def km_median(durations, events) -> float | None:
    """KM median, or ``None`` when S(t) never reaches 0.5 (an UNDEFINED median, not infinity).

    Convention: ``inf{t : S(t) <= 0.5}`` (Klein & Moeschberger). Where the KM curve sits at
    EXACTLY 0.5 over a plateau this returns the START of the plateau, whereas lifelines'
    ``median_survival_time_`` returns the first time S(t) drops strictly BELOW 0.5. On the
    frozen confirmatory dataset the two agree for every condition and both time outcomes,
    so no reported median depends on the choice.
    """

    uniq, surv = _km_steps(np.asarray(durations, dtype=float), np.asarray(events, dtype=float))
    if uniq.size == 0:
        return None
    hit = np.nonzero(surv <= 0.5)[0]
    return None if hit.size == 0 else float(uniq[hit[0]])


def with_exposure(d: pd.DataFrame, exposure: str = "is_mtd") -> pd.DataFrame:
    """Ensure the 0/1 exposure column exists (derive ``is_mtd`` from ``defense``)."""

    if exposure in d.columns:
        return d
    if exposure == "is_mtd" and "defense" in d.columns:
        out = d.copy()
        out["is_mtd"] = (out["defense"] == "mtd").astype(int)
        return out
    return d


def survival_horizon(d: pd.DataFrame, duration: str) -> float:
    """Fixed RMST horizon: the trial duration limit when constant, else max observed time."""

    if "trial_duration_limit_s" in d.columns:
        vals = pd.to_numeric(d["trial_duration_limit_s"], errors="coerce").dropna().unique()
        if vals.size == 1 and np.isfinite(vals[0]) and vals[0] > 0:
            return float(vals[0])
    mx = pd.to_numeric(d[duration], errors="coerce").max()
    return float(mx) if np.isfinite(mx) and mx > 0 else float("nan")


def _boot_plan(d: pd.DataFrame, cluster: str, strata_key: str | None):
    """Row positions per cluster + the cluster lists that are resampled independently."""

    codes = d[cluster].to_numpy()
    order = np.argsort(codes, kind="mergesort")
    uniq, first = np.unique(codes[order], return_index=True)
    seq_rows = dict(zip(uniq, np.split(order, first[1:]), strict=True))
    if strata_key is None:
        return seq_rows, [np.asarray(uniq, dtype=object)]
    smap = d.groupby(cluster, sort=True)[strata_key].first()
    groups = [
        np.asarray(smap.index[smap == key].to_numpy(), dtype=object)
        for key in sorted(smap.unique(), key=str)
    ]
    return seq_rows, [g for g in groups if g.size]


def _boot_indices(seq_rows: dict, groups: list, rng) -> np.ndarray:
    """One cluster-bootstrap replicate: row positions of the resampled clusters."""

    picks: list = []
    for members in groups:
        picks.extend(rng.choice(members, size=members.size, replace=True))
    return np.concatenate([seq_rows[s] for s in picks])


def _event_arrays(d: pd.DataFrame, duration: str, censor_flag: str, exposure: str):
    t = pd.to_numeric(d[duration], errors="coerce").to_numpy(dtype=float)
    e = 1.0 - pd.to_numeric(d[censor_flag], errors="coerce").to_numpy(dtype=float)
    x = pd.to_numeric(d[exposure], errors="coerce").to_numpy(dtype=float)
    return t, e, x


def _rmst_diff_stat(t: np.ndarray, e: np.ndarray, x: np.ndarray, h: float) -> float:
    m0, m1 = x == 0, x == 1
    if not m0.any() or not m1.any():
        return float("nan")
    return km_rmst(t[m1], e[m1], h) - km_rmst(t[m0], e[m0], h)


_RMST_SIGN = (
    "rmst_difference_s = RMST(exposed) - RMST(reference) = RMST(mtd) - RMST(static), in SECONDS. "
    "POSITIVE means the MTD arm stays uncompromised LONGER on average over [0, horizon_s] "
    "(defender-favourable); negative would mean MTD is compromised sooner. Both arms are bounded "
    "above by horizon_s, so the difference cannot exceed horizon_s."
)
_RMST_HORIZON_NOTE = (
    "ADMISSIBILITY OF THE HORIZON. horizon_s is the pre-registered trial duration limit and "
    "exceeds max_followup_reference_s, which is normally inadmissible for RMST. It is admissible "
    "here only because both Kaplan-Meier curves are fully determined on [0, horizon_s] with no "
    "censoring-induced uncertainty: censoring_times_before_horizon = 0 (every censoring time is "
    "exactly the administrative horizon, no dropout) and reference_survival_at_max_followup = 0.0 "
    "(the reference arm is fully observed). Equivalently RMST reduces to mean(min(T, horizon_s)) "
    "in both arms. If censoring_times_before_horizon were non-zero, the horizon would have to be "
    "cut back to min over arms of the largest observed follow-up time."
)
_RMST_INFERENCE = (
    "CI and p are a PERCENTILE SEQUENCE CLUSTER BOOTSTRAP: whole attacker sequences (the "
    "independent unit -- the 8 attempts inside one sequence share retained attacker context) are "
    "resampled with replacement WITHIN the design stratum named by bootstrap_strata, so the "
    "study's fixed allocation of sequences per condition is reproduced in every replicate and the "
    "within-sequence dependence is carried intact. Analytic (Greenwood / pseudo-value) RMST "
    "variance is deliberately NOT used: it assumes independent trials and would understate the "
    "SE. p_percentile is floored at 1/(B+1) and must be written up as 'p < .001', never quoted to "
    "significant figures. p_normal_approx is an unfloored Wald p from the same cluster-bootstrap "
    "SE; it exists so that Benjamini-Hochberg has a p that is not floored, and it is a NORMAL "
    "APPROXIMATION far into a tail that B replicates cannot resolve -- it too must be written up "
    "as 'p < .001', never quoted. ci95_condition_as_random resamples "
    "sequences within EXPOSURE ARM instead, i.e. it also treats the five MTD techniques as a "
    "random sample from a technique population; it is the wider, conservative reading and is "
    "reported for transparency, not used as the primary interval."
)
_RMST_POOLED = (
    "POOLED ESTIMAND. rmst_exposed_s is a design-weighted average over the MTD conditions in the "
    "study's fixed allocation, NOT a technique-neutral 'MTD effect'. The conditions are "
    "heterogeneous; read km_by_condition for the per-condition RMSTs before quoting this number."
)


def rmst_difference(
    d: pd.DataFrame,
    duration: str,
    censor_flag: str,
    *,
    exposure: str = "is_mtd",
    horizon: float | None = None,
    cluster: str = CLUSTER,
    strata: str | None = BOOT_STRATA,
    n_boot: int = BOOT_DEFAULT,
    seed: int = BOOT_SEED_DEFAULT,
) -> dict:
    """RMST(exposed) - RMST(reference) at a fixed horizon; valid without proportional hazards."""

    d = with_exposure(d, exposure)
    need = [c for c in (duration, censor_flag, exposure) if c in d.columns]
    if len(need) < 3:
        return {"note": "RMST needs duration, censor flag and exposure columns"}
    d = d.dropna(subset=need).copy()
    if d[exposure].nunique() < 2:
        return {"note": "RMST needs two exposure groups"}
    h = float(horizon) if horizon is not None else survival_horizon(d, duration)
    if not np.isfinite(h) or h <= 0:
        return {"note": "no finite RMST horizon"}
    t, e, x = _event_arrays(d, duration, censor_flag, exposure)
    m0, m1 = x == 0, x == 1
    if m0.sum() < 2 or m1.sum() < 2:
        return {"note": "insufficient trials per exposure group for RMST"}
    r0, r1 = km_rmst(t[m0], e[m0], h), km_rmst(t[m1], e[m1], h)
    _, s0 = _km_steps(t[m0], e[m0])
    out: dict = {
        "method": "kaplan_meier_rmst",
        "horizon_s": h,
        "rmst_reference_s": r0,
        "rmst_exposed_s": r1,
        "rmst_difference_s": float(r1 - r0),
        "sign_convention": _RMST_SIGN,
        "n_reference": int(m0.sum()),
        "n_exposed": int(m1.sum()),
        "n_events_reference": int(e[m0].sum()),
        "n_events_exposed": int(e[m1].sum()),
        "max_followup_reference_s": float(t[m0].max()),
        "max_followup_exposed_s": float(t[m1].max()),
        "reference_survival_at_max_followup": float(s0[-1]) if s0.size else None,
        "censoring_times_before_horizon": int(((e == 0) & (t < h)).sum()),
        "horizon_note": _RMST_HORIZON_NOTE,
        "note_pooled_estimand": _RMST_POOLED,
    }
    strata_key = strata if (strata and strata in d.columns) else None
    if cluster in d.columns and d[cluster].nunique() >= 4:
        boot = _boot_rmst(d, t, e, x, h, cluster, strata_key, n_boot, seed)
        if boot.size >= 20:
            lo, hi = np.percentile(boot, [2.5, 97.5])
            se = float(boot.std(ddof=1))
            p_pct = 2.0 * min(float(np.mean(boot <= 0.0)), float(np.mean(boot >= 0.0)))
            floor = 1.0 / (boot.size + 1)
            z = (float(r1 - r0) / se) if se > 0 else float("nan")
            out.update(
                {
                    "ci95": [float(lo), float(hi)],
                    "se_bootstrap": se,
                    "p_percentile": float(min(1.0, max(p_pct, floor))),
                    "p_percentile_is_floored": bool(p_pct < floor),
                    "p_normal_approx": (
                        float(2.0 * stats.norm.sf(abs(z))) if np.isfinite(z) else None
                    ),
                    "n_boot": int(boot.size),
                    "bootstrap_seed": int(seed),
                    "bootstrap_cluster": cluster,
                    "bootstrap_strata": strata_key,
                    "n_clusters": int(d[cluster].nunique()),
                }
            )
        if strata_key is not None:
            alt = _boot_rmst(d, t, e, x, h, cluster, exposure, n_boot, seed)
            if alt.size >= 20:
                lo2, hi2 = np.percentile(alt, [2.5, 97.5])
                out["ci95_condition_as_random"] = [float(lo2), float(hi2)]
                out["se_bootstrap_condition_as_random"] = float(alt.std(ddof=1))
    out.setdefault("ci95", None)
    out["inference_note"] = (
        _RMST_INFERENCE
        if out.get("ci95") is not None
        else "too few sequences for a cluster bootstrap; point estimate only"
    )
    return out


def _boot_rmst(d, t, e, x, h, cluster, strata_key, n_boot, seed) -> np.ndarray:
    seq_rows, groups = _boot_plan(d, cluster, strata_key)
    rng = np.random.default_rng(seed)
    boot = np.empty(int(n_boot))
    for b in range(int(n_boot)):
        idx = _boot_indices(seq_rows, groups, rng)
        boot[b] = _rmst_diff_stat(t[idx], e[idx], x[idx], h)
    return boot[np.isfinite(boot)]


def rmst_by_condition(
    d: pd.DataFrame,
    duration: str,
    censor_flag: str,
    *,
    horizon: float | None = None,
    condition_col: str = "condition",
    reference: str | None = None,
) -> dict:
    """Per-condition KM median, censored proportion, RMST and RMST difference vs. the baseline."""

    if reference is None:
        from analysis.dataset import STATIC

        reference = STATIC
    if any(c not in d.columns for c in (duration, censor_flag, condition_col)):
        return {"note": "missing duration/censor/condition column"}
    d = d.dropna(subset=[duration, censor_flag]).copy()
    if d.empty:
        return {"note": "no trials with a recorded event time"}
    h = float(horizon) if horizon is not None else survival_horizon(d, duration)
    if not np.isfinite(h) or h <= 0:
        return {"note": "no finite RMST horizon"}
    d["_event"] = 1 - pd.to_numeric(d[censor_flag], errors="coerce")
    ref = d[d[condition_col] == reference]
    r_ref = km_rmst(ref[duration], ref["_event"], h) if len(ref) else float("nan")
    note = (
        "km_median_s is null when the arm is majority-censored so the KM curve never reaches "
        "S(t)=0.5 -- the median is UNDEFINED, not infinite; km_median_defined makes that "
        "explicit. rmst_s is the PH-free alternative summary and is always finite because it is "
        "bounded by horizon_s."
    )
    if not len(ref):
        note += (
            " WARNING: the reference condition is absent from this dataset, so "
            "rmst_diff_vs_reference_s is null for every condition."
        )
    out: dict = {
        "horizon_s": h,
        "reference_condition": reference,
        "reference_present": bool(len(ref)),
        "note": note,
    }
    for c, g in d.groupby(condition_col, observed=True):
        med = km_median(g[duration], g["_event"])
        r = km_rmst(g[duration], g["_event"], h)
        out[str(c)] = {
            "n": int(len(g)),
            "n_events": int(g["_event"].sum()),
            "censored_proportion": float(pd.to_numeric(g[censor_flag], errors="coerce").mean()),
            "km_median_s": med,
            "km_median_defined": med is not None,
            "rmst_s": None if not np.isfinite(r) else float(r),
            "rmst_diff_vs_reference_s": (
                None if not (np.isfinite(r) and np.isfinite(r_ref)) else float(r - r_ref)
            ),
        }
    return out


_RMST_BY_CONDITION_BOOT_NOTE = (
    "Per-condition restricted mean time to compromise on [0, horizon_s] and its difference from "
    "the reference condition (positive = the condition stays uncompromised LONGER than the "
    "reference). 95% intervals and p are a PERCENTILE SEQUENCE CLUSTER BOOTSTRAP: whole "
    "sequences are resampled with replacement within each condition, so each condition keeps "
    "its number of sequences and the 8 attempts of a sequence stay together. p is two-sided, "
    "2 x min(share of replicates <= 0, share >= 0) with the (count + 1)/(B + 1) correction, so "
    "it is floored at 2/(B + 1): write a floored p as 'p < .001'. p_holm adjusts the "
    "differences from the reference for the number of conditions compared. method_used is "
    "'mean_min_t_h' when no trial is censored before the horizon (RMST then equals "
    "mean(min(T, horizon)) exactly) and 'kaplan_meier' otherwise. A condition whose every "
    "trial is an event at the same time, or that never varies across replicates, has a "
    "degenerate interval; read its n_events."
)


def _rmst_condition_draws(
    d: pd.DataFrame,
    duration: str,
    censor_flag: str,
    *,
    horizon: float,
    condition_col: str,
    cluster: str,
    n_boot: int,
    seed: int,
) -> dict:
    """Each condition's RMST at ``horizon`` and its sequence cluster bootstrap replicates.

    One pass over every condition in ``d``, in sorted order, from one generator seeded with
    ``seed``: the same data and seed always give the same replicates, so the per-condition
    summary (:func:`rmst_by_condition_bootstrap`) and the pairwise comparisons
    (:func:`rmst_pairwise_bootstrap`) read identical draws. Each replicate resamples whole
    sequences within every condition. Returns ``{"note": ...}`` when the inputs cannot support
    the computation.
    """

    need = (duration, censor_flag, condition_col, cluster)
    if any(c not in d.columns for c in need):
        return {"note": "missing duration/censor/condition/sequence column"}
    if not np.isfinite(horizon) or horizon <= 0:
        return {"note": "no finite RMST horizon"}
    d = d.dropna(subset=[duration, censor_flag, condition_col, cluster]).copy()
    if d.empty:
        return {"note": "no trials with a recorded event time"}
    t_all = pd.to_numeric(d[duration], errors="coerce").to_numpy(dtype=float)
    e_all = 1.0 - pd.to_numeric(d[censor_flag], errors="coerce").to_numpy(dtype=float)
    early_censoring = int(((e_all == 0) & (t_all < horizon)).sum())
    fast = early_censoring == 0
    conds = sorted(d[condition_col].astype(str).unique().tolist())
    cond_arr = d[condition_col].astype(str).to_numpy()
    seq_arr = d[cluster].astype(str).to_numpy()
    rng = np.random.default_rng(seed)

    point: dict[str, float] = {}
    draws: dict[str, np.ndarray] = {}
    counts: dict[str, dict] = {}
    for c in conds:
        m = cond_arr == c
        t, e, s = t_all[m], e_all[m], seq_arr[m]
        counts[c] = {
            "n": int(m.sum()),
            "n_sequences": int(len(pd.unique(s))),
            "n_events": int(e.sum()),
        }
        if fast:
            clipped = np.minimum(t, horizon)
            frame = pd.DataFrame({"s": s, "v": clipped})
            agg = frame.groupby("s", sort=True)["v"].agg(["sum", "count"])
            sums = agg["sum"].to_numpy(dtype=float)
            cnts = agg["count"].to_numpy(dtype=float)
            point[c] = float(sums.sum() / cnts.sum())
            k = len(sums)
            idx = rng.integers(0, k, size=(int(n_boot), k))
            draws[c] = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
        else:
            point[c] = km_rmst(t, e, horizon)
            uniq = pd.unique(s)
            rows = {u: np.nonzero(s == u)[0] for u in uniq}
            out_b = np.empty(int(n_boot))
            for b in range(int(n_boot)):
                pick = rng.choice(uniq, size=uniq.size, replace=True)
                ii = np.concatenate([rows[u] for u in pick])
                out_b[b] = km_rmst(t[ii], e[ii], horizon)
            draws[c] = out_b
    return {
        "conds": conds,
        "point": point,
        "draws": draws,
        "counts": counts,
        "early_censoring": early_censoring,
        "fast": fast,
    }


def _interval95(x: np.ndarray) -> list[float] | None:
    x = x[np.isfinite(x)]
    return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))] if x.size else None


def _boot_p_two_sided(x: np.ndarray) -> float | None:
    """2 x min(share <= 0, share >= 0) with the (count + 1)/(B + 1) correction."""

    x = x[np.isfinite(x)]
    if not x.size:
        return None
    tol = 1e-9 * max(1.0, float(np.max(np.abs(x))))
    lo = (np.sum(x <= tol) + 1.0) / (x.size + 1.0)
    hi = (np.sum(x >= -tol) + 1.0) / (x.size + 1.0)
    return float(min(1.0, 2.0 * min(lo, hi)))


def rmst_by_condition_bootstrap(
    d: pd.DataFrame,
    duration: str,
    censor_flag: str,
    *,
    horizon: float,
    reference: str | None = None,
    condition_col: str = "condition",
    cluster: str = CLUSTER,
    n_boot: int = BOOT_DEFAULT,
    seed: int = BOOT_SEED_DEFAULT,
) -> dict:
    """Per-condition RMST at ``horizon`` and its difference from ``reference``, with intervals.

    The same sequence cluster bootstrap as :func:`rmst_difference`, run once over all
    conditions: each replicate resamples whole sequences within every condition and recomputes
    every condition's RMST, so all differences share one set of replicates.
    """

    from analysis.fdr import holm

    if reference is None:
        from analysis.dataset import STATIC

        reference = STATIC
    dr = _rmst_condition_draws(
        d,
        duration,
        censor_flag,
        horizon=horizon,
        condition_col=condition_col,
        cluster=cluster,
        n_boot=n_boot,
        seed=seed,
    )
    if "note" in dr:
        return {"note": dr["note"]}
    conds, point, draws, counts = dr["conds"], dr["point"], dr["draws"], dr["counts"]

    by_condition: dict[str, dict] = {}
    for c in conds:
        entry = {**counts[c], "rmst_s": point[c], "rmst_ci95": _interval95(draws[c])}
        if reference in point and c != reference:
            diff = draws[c] - draws[reference]
            entry["difference_vs_reference_s"] = float(point[c] - point[reference])
            entry["difference_ci95"] = _interval95(diff)
            entry["difference_p"] = _boot_p_two_sided(diff)
        by_condition[c] = entry
    adj = holm(
        {c: v["difference_p"] for c, v in by_condition.items() if v.get("difference_p") is not None}
    )
    for c, p in adj.items():
        by_condition[c]["difference_p_holm"] = p
    return {
        "horizon_s": float(horizon),
        "reference_condition": reference,
        "reference_present": reference in point,
        "method_used": "mean_min_t_h" if dr["fast"] else "kaplan_meier",
        "censoring_times_before_horizon": dr["early_censoring"],
        "by_condition": by_condition,
        "n_boot": int(n_boot),
        "bootstrap_seed": int(seed),
        "bootstrap_cluster": cluster,
        "note": _RMST_BY_CONDITION_BOOT_NOTE,
    }


_RMST_PAIRWISE_NOTE = (
    "Every pairwise difference in restricted mean time to compromise over [0, horizon_s] among "
    "the listed conditions (first - second, both in sorted order; POSITIVE = the first "
    "condition stays uncompromised LONGER), in seconds. The replicates are those of "
    "rmst_by_condition_bootstrap on the same data and seed: whole sequences resampled within "
    "every condition of the dataset passed in, so the per-condition RMSTs and intervals match "
    "that block's. ci95 is the per-pair percentile interval; ci_simultaneous is the "
    "Bonferroni-adjusted percentile interval at 1 - alpha / m (m = the number of pairs), so all "
    "m intervals hold together at 1 - alpha; p is the two-sided bootstrap p (+1 correction, "
    "floored at 2/(B + 1): write a floored p as 'p < .001') and p_holm its Holm adjustment over "
    "the m pairs. The simultaneous interval sits far in the bootstrap tails (about alpha / (2m) "
    "of B replicates on each side), so its ends are less stable than the 95% interval's."
)


def rmst_pairwise_bootstrap(
    d: pd.DataFrame,
    duration: str,
    censor_flag: str,
    *,
    horizon: float,
    levels: tuple[str, ...] | list[str],
    condition_col: str = "condition",
    cluster: str = CLUSTER,
    n_boot: int = BOOT_DEFAULT,
    seed: int = BOOT_SEED_DEFAULT,
    alpha: float = 0.05,
) -> dict:
    """All pairwise RMST differences at ``horizon`` among ``levels``, with Holm-adjusted p.

    Pass the whole dataset (every condition), not only ``levels``: the replicates are then
    exactly those of :func:`rmst_by_condition_bootstrap`. Levels absent from ``d`` are listed
    in ``levels_absent`` and left out of the family.
    """

    import itertools

    from analysis.fdr import holm

    if condition_col not in d.columns:
        return {"note": f"{condition_col} not in the dataset"}
    lab = d[condition_col].astype(str)
    present = sorted(lev for lev in set(levels) if (lab == lev).any())
    absent = sorted(set(levels) - set(present))
    if len(present) < 2:
        return {"levels_absent": absent, "note": "needs at least two of the listed levels"}
    dr = _rmst_condition_draws(
        d,
        duration,
        censor_flag,
        horizon=horizon,
        condition_col=condition_col,
        cluster=cluster,
        n_boot=n_boot,
        seed=seed,
    )
    if "note" in dr:
        return {"levels_absent": absent, "note": dr["note"]}
    point, draws, counts = dr["point"], dr["draws"], dr["counts"]
    present = [lev for lev in present if lev in point]
    pairs = list(itertools.combinations(present, 2))
    m = len(pairs)
    q = alpha / m if m else float("nan")
    out_pairs: dict = {}
    for a, b in pairs:
        diff = draws[a] - draws[b]
        fin = diff[np.isfinite(diff)]
        sim = (
            [float(np.quantile(fin, q / 2)), float(np.quantile(fin, 1 - q / 2))]
            if fin.size
            else None
        )
        out_pairs[f"{a} - {b}"] = {
            "difference_s": float(point[a] - point[b]),
            "ci95": _interval95(diff),
            "ci_simultaneous": sim,
            "p": _boot_p_two_sided(diff),
        }
    for k, adj in holm({k: v["p"] for k, v in out_pairs.items()}).items():
        out_pairs[k]["p_holm"] = adj
    bounds = [
        abs(x) for v in out_pairs.values() if v["ci_simultaneous"] for x in v["ci_simultaneous"]
    ]
    return {
        "outcome": duration,
        "horizon_s": float(horizon),
        "levels": present,
        "levels_absent": absent,
        "rmst_by_level": {
            lev: {**counts[lev], "rmst_s": point[lev], "rmst_ci95": _interval95(draws[lev])}
            for lev in present
        },
        "n_pairs": m,
        "alpha": float(alpha),
        "simultaneous_level": float(1 - alpha),
        "per_pair_alpha_bonferroni": float(q),
        "pairs": out_pairs,
        "largest_abs_simultaneous_bound_s": float(max(bounds)) if bounds else None,
        "method_used": "mean_min_t_h" if dr["fast"] else "kaplan_meier",
        "censoring_times_before_horizon": dr["early_censoring"],
        "n_boot": int(n_boot),
        "bootstrap_seed": int(seed),
        "bootstrap_cluster": cluster,
        "note": _RMST_PAIRWISE_NOTE,
    }


# --------------------------------------------------------------------------- #
# PH-robust survival: non-parametric interval hazard ratios
# --------------------------------------------------------------------------- #


def _hazard_intervals(t, e, x, max_intervals: int):
    """Comparable horizon + the groups of distinct event times that form the intervals."""

    m0, m1 = x == 0, x == 1
    t_comp = float(min(t[m0].max(), t[m1].max()))
    ev = np.unique(t[(e == 1) & (t <= t_comp)])
    if ev.size == 0:
        return t_comp, []
    if ev.size <= max_intervals:
        return t_comp, [np.array([v]) for v in ev]
    return t_comp, [b for b in np.array_split(ev, max_intervals) if b.size]


def _cumhaz_increments(t, e, x, groups) -> np.ndarray:
    """Aggregated Nelson-Aalen increments (rows = intervals, cols = reference/exposed)."""

    out = np.full((len(groups), 2), np.nan)
    for i, g in enumerate(groups):
        for a in (0, 1):
            m = x == a
            tm, em = t[m], e[m]
            total, ok = 0.0, True
            for tt in g:
                n_at = float((tm >= tt).sum())
                if n_at <= 0:
                    ok = False
                    break
                total += float(((tm == tt) & (em == 1)).sum()) / n_at
            if ok:
                out[i, a] = total
    return out


_HAZARD_NOTE = (
    "NON-PARAMETRIC interval hazard ratios: within each interval the arm-specific aggregated "
    "Nelson-Aalen increment (sum of d/n over the distinct event times in the interval) is formed "
    "and the exposed increment is divided by the reference increment. No Cox model, no partial "
    "likelihood, no tie approximation and NO proportional-hazards assumption is involved -- the "
    "point of the table is to show how the hazard ratio MOVES. Intervals are the distinct "
    "observed event times below comparable_horizon_s, i.e. the decomposition is SATURATED and has "
    "no tuning parameter; max_intervals only caps the table on a future dataset with a dense "
    "event-time grid, by binning consecutive distinct event times into equal-count groups -- read "
    "the boolean `saturated` before interpreting the interval boundaries. "
    "comparable_horizon_s is the last instant at which BOTH arms still have trials at risk; "
    "beyond it the exposure is constant within every risk set and no hazard ratio is identifiable "
    "at all, which is exactly why RMST -- not any hazard-ratio summary -- is the headline "
    "estimate. Read n_at_risk_reference before quoting a late interval: it can be very small. "
    "CIs and the time_trend p come from the SAME sequence cluster bootstrap as the RMST, so they "
    "respect the within-sequence dependence; they are NOT model-based."
)


def hazard_profile(
    d: pd.DataFrame,
    duration: str,
    censor_flag: str,
    *,
    exposure: str = "is_mtd",
    cluster: str = CLUSTER,
    strata: str | None = BOOT_STRATA,
    n_boot: int = BOOT_DEFAULT,
    seed: int = BOOT_SEED_DEFAULT,
    max_intervals: int = MAX_HAZARD_INTERVALS,
) -> dict:
    """How the hazard ratio moves over follow-up, without assuming proportional hazards."""

    d = with_exposure(d, exposure)
    need = [c for c in (duration, censor_flag, exposure) if c in d.columns]
    if len(need) < 3:
        return {"note": "hazard profile needs duration, censor flag and exposure columns"}
    d = d.dropna(subset=need).copy()
    if d[exposure].nunique() < 2:
        return {"note": "hazard profile needs two exposure groups"}
    t, e, x = _event_arrays(d, duration, censor_flag, exposure)
    if e.sum() < 3:
        return {"note": "too few events for a hazard profile"}
    t_comp, groups = _hazard_intervals(t, e, x, max_intervals)
    if not groups:
        return {"note": "no events below the comparable horizon", "comparable_horizon_s": t_comp}
    obs = _cumhaz_increments(t, e, x, groups)
    strata_key = strata if (strata and strata in d.columns) else None
    boot = None
    if cluster in d.columns and d[cluster].nunique() >= 4:
        seq_rows, bgroups = _boot_plan(d, cluster, strata_key)
        rng = np.random.default_rng(seed)
        acc = np.empty((int(n_boot), len(groups)))
        for b in range(int(n_boot)):
            idx = _boot_indices(seq_rows, bgroups, rng)
            inc = _cumhaz_increments(t[idx], e[idx], x[idx], groups)
            with np.errstate(divide="ignore", invalid="ignore"):
                acc[b] = np.log(inc[:, 1] / inc[:, 0])
        boot = acc
    intervals = [_hazard_interval_record(i, g, obs, boot, t, e, x) for i, g in enumerate(groups)]
    n_distinct = int(np.unique(t[(e == 1) & (t <= t_comp)]).size)
    saturated = len(groups) == n_distinct
    return {
        "method": "nonparametric_interval_hazard_ratio",
        "comparable_horizon_s": t_comp,
        "n_distinct_event_times": n_distinct,
        "max_intervals": int(max_intervals),
        "saturated": bool(saturated),
        "window_rule": (
            "one interval per distinct observed event time below comparable_horizon_s "
            "(saturated); no tuning parameter"
            if saturated
            else (
                f"{len(groups)} equal-count bins of consecutive distinct event times: the "
                f"{n_distinct} distinct event times below comparable_horizon_s exceed the "
                f"max_intervals table cap ({int(max_intervals)}), so the decomposition is NOT "
                "saturated and max_intervals acts as a tuning parameter here -- read "
                "saturated:false before interpreting the interval boundaries"
            )
        ),
        "intervals": intervals,
        "time_trend": _hazard_trend(intervals, obs, boot),
        "bootstrap_cluster": cluster,
        "bootstrap_strata": strata_key,
        "bootstrap_seed": int(seed),
        "estimand_note": _HAZARD_NOTE,
    }


def _hazard_interval_record(i, g, obs, boot, t, e, x) -> dict:
    h0, h1 = obs[i, 0], obs[i, 1]
    in_g = np.isin(t, g)
    rec = {
        "interval": int(i),
        "event_times_s": [float(v) for v in g],
        "start_s": float(g[0]),
        "stop_s": float(g[-1]),
        "n_at_risk_reference": int((t[x == 0] >= g[0]).sum()),
        "n_at_risk_exposed": int((t[x == 1] >= g[0]).sum()),
        "n_events_reference": int((in_g & (e == 1) & (x == 0)).sum()),
        "n_events_exposed": int((in_g & (e == 1) & (x == 1)).sum()),
        "cumhaz_increment_reference": None if not np.isfinite(h0) else float(h0),
        "cumhaz_increment_exposed": None if not np.isfinite(h1) else float(h1),
        "ci95": None,
    }
    if not (np.isfinite(h0) and np.isfinite(h1) and h0 > 0 and h1 > 0):
        rec["identified"] = False
        rec["hazard_ratio"] = None
        rec["unidentified_reason"] = (
            "no_trials_at_risk_in_one_arm"
            if not (np.isfinite(h0) and np.isfinite(h1))
            else "no_events_in_one_arm"
        )
        return rec
    rec["identified"] = True
    rec["hazard_ratio"] = float(h1 / h0)
    if boot is not None:
        col = boot[:, i]
        col = col[np.isfinite(col)]
        if col.size >= 20:
            lo, hi = np.percentile(col, [2.5, 97.5])
            rec["ci95"] = [float(np.exp(lo)), float(np.exp(hi))]
            rec["se_log_hr_bootstrap"] = float(col.std(ddof=1))
            rec["n_boot"] = int(col.size)
    return rec


def _hazard_trend(intervals: list, obs: np.ndarray, boot) -> dict:
    ident = [r["interval"] for r in intervals if r["identified"]]
    if len(ident) < 2 or boot is None:
        return {"note": "fewer than two identified intervals; no time-trend contrast"}
    i0, i1 = ident[0], ident[-1]
    est = float(np.log(obs[i1, 1] / obs[i1, 0]) - np.log(obs[i0, 1] / obs[i0, 0]))
    diff = boot[:, i1] - boot[:, i0]
    diff = diff[np.isfinite(diff)]
    if diff.size < 20:
        return {"note": "too few finite bootstrap replicates for a time-trend contrast"}
    lo, hi = np.percentile(diff, [2.5, 97.5])
    se = float(diff.std(ddof=1))
    return {
        "contrast": f"log_hazard_ratio(interval {i1}) - log_hazard_ratio(interval {i0})",
        "estimate_log_hr_change": est,
        "hazard_ratio_fold_change": float(np.exp(est)),
        "ci95_log_hr_change": [float(lo), float(hi)],
        "se_bootstrap": se,
        "p_normal_approx": float(2.0 * stats.norm.sf(abs(est / se))) if se > 0 else None,
        "n_boot": int(diff.size),
        "note": (
            "H0: the hazard ratio is the same in the first and last identified interval (i.e. "
            "proportional hazards). Rejection corroborates the Schoenfeld verdict using a "
            "sequence cluster bootstrap and no time transform. This is a DESCRIPTIVE "
            "decomposition of a PH violation already established by the Schoenfeld test, not a "
            "confirmatory test in its own right."
        ),
    }


# --------------------------------------------------------------------------- #
# PH diagnostics and the PH-robust block
# --------------------------------------------------------------------------- #

_PH_LABEL_VIOLATED = (
    "NOT PH-VALID. The proportional-hazards assumption is rejected for this term (Schoenfeld "
    "residuals), so this hazard ratio is a follow-up-time-averaged summary of a hazard ratio that "
    "CHANGES over the attack window, not a constant multiplicative effect. Retained only for "
    "continuity with the pre-registered analysis plan; it MUST NOT be reported as the headline "
    "survival estimate -- use ph_robust.rmst_difference (and ph_robust.hazard_profile for how the "
    "hazard ratio moves) instead."
)
_PH_LABEL_OK = (
    "PH-valid. The proportional-hazards assumption is not rejected for this term; the hazard "
    "ratio is interpretable as a constant multiplicative effect on the hazard over follow-up."
)
_PH_LABEL_UNKNOWN = (
    "PH status UNKNOWN -- the Schoenfeld test could not be computed for this term; treat the "
    "hazard ratio as provisional and prefer ph_robust.rmst_difference."
)
_PH_TIE_NOTE = (
    "min_schoenfeld_p uses lifelines' default time_transform='rank', which is implemented as "
    "np.cumsum(event) over the fitted row order and is therefore SENSITIVE TO ROW ORDER when "
    "event times are heavily tied (they are here). by_time_transform reports the same test under "
    "four transforms so the verdict does not rest on that artefact. assumption_ok_all_transforms "
    "is the tie-robust verdict that triggers the PH-robust path: it is TRUE only when the test "
    "fails to reject under EVERY transform for EVERY term (it is a minimum over n_tests "
    "correlated tests, so it is a deliberately LIBERAL SCREEN, not a test at alpha = .05 -- its "
    "false-trigger rate under a true PH model is well above .05). Over-triggering is the intended "
    "failure direction: triggering only adds a PH-free summary block and never changes an "
    "estimate. Read per_term_min_p_any_transform to see WHICH term violates PH."
)


def ph_diagnostics(cph, design) -> dict:
    """Schoenfeld PH check under several time transforms; graceful on failure."""

    from lifelines.statistics import proportional_hazard_test

    out: dict = {"by_time_transform": {}}
    per_term: dict = {}
    for tf in PH_TRANSFORMS:
        try:
            summary = proportional_hazard_test(cph, design, time_transform=tf).summary
        except Exception:  # noqa: BLE001 - a transform can fail on degenerate data
            continue
        if not len(summary):
            continue
        out["by_time_transform"][tf] = {
            "per_term_p": {str(k): float(v) for k, v in summary["p"].items()},
            "min_p": float(summary["p"].min()),
        }
        for k, v in summary["p"].items():
            per_term[str(k)] = min(per_term.get(str(k), float("inf")), float(v))
    rank = out["by_time_transform"].get("rank")
    out["min_schoenfeld_p"] = rank["min_p"] if rank else None
    out["assumption_ok"] = None if rank is None else bool(rank["min_p"] > 0.05)
    mins = [v["min_p"] for v in out["by_time_transform"].values()]
    any_min = float(np.min(mins)) if mins else None
    out["per_term_min_p_any_transform"] = per_term
    out["min_schoenfeld_p_any_transform"] = any_min
    out["n_tests"] = int(len(out["by_time_transform"]) * len(per_term))
    ok = None if any_min is None else bool(any_min > 0.05)
    out["assumption_ok_all_transforms"] = ok
    out["assumption_ok_any_transform"] = ok
    out["assumption_ok_any_transform_deprecated_alias_of"] = "assumption_ok_all_transforms"
    out["tie_note"] = _PH_TIE_NOTE
    out["model_validity_label"] = (
        _PH_LABEL_UNKNOWN if ok is None else (_PH_LABEL_OK if ok else _PH_LABEL_VIOLATED)
    )
    return out


def stratified_cox(
    d: pd.DataFrame,
    duration: str,
    censor_flag: str,
    *,
    exposure: str = "is_mtd",
    strata: tuple[str, ...] = ("attempt_index",),
    cluster: str = CLUSTER,
) -> dict:
    """Cox with a separate baseline hazard per stratum; robustness check on the marginal HR."""

    from lifelines import CoxPHFitter

    d = with_exposure(d, exposure)
    strata = tuple(s for s in strata if s in d.columns)
    if not strata or exposure not in d.columns:
        return {"note": "no usable stratification variable / exposure"}
    cols = [c for c in (duration, censor_flag, cluster, exposure, *strata) if c in d.columns]
    d = d.dropna(subset=cols).copy()
    d = d[pd.to_numeric(d[duration], errors="coerce") > 0]
    if d.empty:
        return {"note": "no trials with a positive event time"}
    d["_event"] = (1 - pd.to_numeric(d[censor_flag], errors="coerce")).astype(int)
    if d[exposure].nunique() < 2 or d["_event"].sum() < 3:
        return {"note": "insufficient exposure variation or events for a stratified Cox"}
    design = d[[duration, "_event", cluster, exposure, *strata]].copy()
    for s in strata:
        design[s] = pd.to_numeric(design[s], errors="coerce").astype("Int64").astype(int)
    try:
        cph = CoxPHFitter()
        cph.fit(
            design,
            duration_col=duration,
            event_col="_event",
            cluster_col=cluster,
            strata=list(strata),
            formula=exposure,
        )
    except Exception as exc:  # noqa: BLE001
        return {"note": f"stratified Cox unavailable: {exc}"}
    ci = cph.confidence_intervals_
    return {
        "method": "stratified_cox_cluster_robust",
        "strata": list(strata),
        "hazard_ratio": float(np.exp(cph.params_[exposure])),
        "ci95": [float(np.exp(ci.loc[exposure].iloc[0])), float(np.exp(ci.loc[exposure].iloc[1]))],
        "p": float(cph.summary.loc[exposure, "p"]),
        "n_events": int(design["_event"].sum()),
        "estimand_note": (
            "Cox allowing a SEPARATE baseline hazard per attempt_index (position in the retained-"
            "context sequence), with lifelines' cluster-robust (sequence) variance. This is a "
            "ROBUSTNESS CHECK on the marginal hazard ratio, NOT a repair of proportional hazards: "
            "the Schoenfeld violation is carried by the exposure term itself, so no "
            "stratification on a covariate can remove it, and stratifying by condition/technique "
            "would remove it only by absorbing the very contrast the research question asks "
            "about. No Schoenfeld p is reported for this fit: lifelines' PH test is unreliable on "
            "a stratified fit with heavily tied event times (it returns rank p = 0.565 while the "
            "same fit gives 6.9e-07 under the 'km' transform)."
        ),
    }


def ph_robust_survival(
    d: pd.DataFrame,
    duration: str,
    censor_flag: str,
    *,
    exposure: str = "is_mtd",
    cluster: str = CLUSTER,
    strata: str | None = BOOT_STRATA,
    horizon: float | None = None,
    n_boot: int = BOOT_DEFAULT,
    seed: int = BOOT_SEED_DEFAULT,
) -> dict:
    """PH-free survival summaries used when the Schoenfeld screen rejects PH."""

    d = with_exposure(d, exposure)
    return {
        "headline_estimator": "rmst_difference",
        "why": (
            "The proportional-hazards assumption is rejected for this outcome, so the "
            "cluster-robust hazard ratio is not a defensible headline. RMST at a fixed horizon "
            "requires no PH assumption, is measured in seconds, and is bounded by the horizon."
        ),
        "rmst_difference": rmst_difference(
            d,
            duration,
            censor_flag,
            exposure=exposure,
            horizon=horizon,
            cluster=cluster,
            strata=strata,
            n_boot=n_boot,
            seed=seed,
        ),
        "hazard_profile": hazard_profile(
            d,
            duration,
            censor_flag,
            exposure=exposure,
            cluster=cluster,
            strata=strata,
            n_boot=n_boot,
            seed=seed,
        ),
        "stratified_cox": stratified_cox(
            d, duration, censor_flag, exposure=exposure, cluster=cluster
        ),
        "removed_estimator_note": (
            "A counting-process Cox with a step-function time-varying coefficient (entry_col + "
            "cluster_col) was evaluated and DELIBERATELY NOT SHIPPED. lifelines 0.30.3 computes "
            "its cluster-robust standard errors from score residuals that ignore left truncation "
            "(CoxPHFitter._compute_score_within_strata never receives the entry times, and "
            "RegressionFitter.compute_residuals refuses entry fits outright with "
            "NotImplementedError), and CoxPHFitter.variance_matrix_ is the naive model-based "
            "covariance, never the cluster sandwich. Measured against a genuine sequence cluster "
            "bootstrap of the same fit, lifelines' 'robust' SEs were wrong by factors of 0.66x to "
            "1.8x and its between-window correlations were ~0 where the true clustered "
            "correlations were 0.2-0.7. hazard_profile above answers the same question "
            "non-parametrically with honest cluster-bootstrap inference."
        ),
    }


# --------------------------------------------------------------------------- #
# Continuous: linear mixed model (random intercept)
# --------------------------------------------------------------------------- #


def lmm_continuous(d: pd.DataFrame, outcome: str, rhs: str, group: str = CLUSTER) -> dict:
    """Linear mixed model with a random intercept for ``group``."""

    import statsmodels.formula.api as smf

    d = d.dropna(subset=[outcome]).copy()
    # The zero-variance screen must run BEFORE the grouping check: a constant
    # outcome is not estimable regardless of how well-structured the clustering is,
    # and the skip payload is the more specific (and more useful) explanation.
    skip = _skip(d, outcome)
    if skip is not None:
        return skip
    if d[group].nunique() < 2:
        return {"note": "insufficient grouping structure"}
    try:
        m = smf.mixedlm(f"{outcome} ~ {rhs}", d, groups=d[group]).fit(
            method="lbfgs", reml=True, disp=False
        )
        re_var = (
            float(m.cov_re.iloc[0, 0])
            if getattr(m, "cov_re", None) is not None and m.cov_re.size
            else None
        )
        # Residual scale: reported so the degeneracy diagnostics can form an ICC.
        # Hoisted out of the marginal-R^2 block (and separately guarded) so a raise
        # here cannot turn a successful fit into a reported LMM failure.
        try:
            resid_var = float(m.scale)
        except Exception:  # noqa: BLE001
            resid_var = None
        # Marginal R^2 (fixed effects) via variance of fitted fixed-effect predictions.
        try:
            fe_pred = m.predict(d)
            var_f = float(np.nanvar(fe_pred, ddof=1))
            total = var_f + (re_var or 0.0) + (resid_var or 0.0)
            marginal_r2 = var_f / total if total > 0 else None
        except Exception:  # noqa: BLE001
            marginal_r2 = None
        return {
            "terms": {
                nm: {"coef": float(m.params[nm]), "p": float(m.pvalues[nm])}
                for nm in m.params.index
                if nm not in ("Intercept", "Group Var")
            },
            "converged": bool(m.converged),
            "random_intercept_var": re_var,
            "residual_var": resid_var,
            "marginal_r2_fixed": marginal_r2,
            "model": m,  # retained for EMMeans/contrasts by the caller
        }
    except Exception as exc:  # noqa: BLE001
        return {"note": f"LMM unavailable: {exc}"}


# --------------------------------------------------------------------------- #
# EMMeans + pairwise contrasts from a fitted mixed model
# --------------------------------------------------------------------------- #


def contrast_tolerance(emmeans: dict) -> float:
    """Tolerance below which a contrast between EMMeans is numerical round-off.

    Uses the same scale-relative rule as ``analysis.degeneracy`` (there is exactly
    one such rule in the codebase), applied to the magnitude of the estimated
    marginal means themselves: ``max(1e-12, 1e-9 * max(|emmean|, 1))``. Non-finite
    EMMeans are ignored rather than propagated, so a NaN level cannot silently
    disable the screen for every other level.

    The screen is scale-relative for a reason. Round-off in a contrast is
    proportional to the magnitude of the quantities being differenced, so a fixed
    absolute epsilon would be far too tight on a 1800-second outcome and far too
    loose on an F1 ratio. At 1e-9 relative it sits ~6 orders of magnitude above
    double-precision noise (~2.2e-16 relative) and ~7 orders of magnitude below the
    smallest genuine between-arm contrast in this study (8.66e-03 for
    ``recon_accuracy_f1``), so it cannot manufacture a false zero for a real effect.
    """

    vals = [abs(float(e.get("emmean", 0.0) or 0.0)) for e in emmeans.values()]
    return _scale_tolerance([v for v in vals if np.isfinite(v)])


def emmeans_pairwise(m, factor: str, levels: list[str]) -> tuple[dict, dict, list]:
    """EMMeans per level of ``C(factor)`` from a fitted MixedLM, an omnibus Wald
    test, and Bonferroni-adjusted pairwise contrasts (Tukey intent).

    ``m`` is a statsmodels ``MixedLMResults`` for a model ``outcome ~ C(factor)``
    (optionally with other terms). EMMeans hold the random effect at 0 and use only
    the fixed-effect covariance.

    **Multiplicity.** The Bonferroni denominator is ``m_family = C(k, 2)``, the
    number of pairwise contrasts the *design* generates from the k observed factor
    levels. It is fixed before any estimate is inspected and never shrinks in
    response to what the estimates turn out to be. Two kinds of contrast are
    reported but not tested, and **both stay in the denominator**:

    - a contrast whose magnitude is at or below ``contrast_tolerance`` is round-off,
      not an effect: it is reported as exactly zero with ``p_bonf``/``significant``
      null and ``clamped: true``;
    - a contrast whose standard error is not finite and positive has no defined Wald
      statistic: it is reported with ``untestable: true`` and null p.

    Dropping either from the denominator would make every surviving contrast up to
    m/(m-1) times more significant on the strength of a data-dependent decision --
    the exact anti-conservative pattern this function exists to avoid. Reporting
    them with ``p_bonf = 1.0`` would be worse still: that asserts "tested and found
    nothing" for a comparison that was never testable.
    """

    fe = [n for n in m.params.index if n != "Group Var"]
    params = m.params[fe].to_numpy()
    cov = m.cov_params().loc[fe, fe].to_numpy()

    def emm_vec(level: str) -> np.ndarray:
        v = pd.Series(0.0, index=fe)
        if "Intercept" in v.index:
            v["Intercept"] = 1.0
        term = f"C({factor})[T.{level}]"
        if term in v.index:
            v[term] = 1.0
        return v.to_numpy()

    emmeans = {
        lv: {
            "emmean": float(emm_vec(lv) @ params),
            "se": float(np.sqrt(emm_vec(lv) @ cov @ emm_vec(lv))),
        }
        for lv in levels
    }

    full = list(m.params.index)
    fac_terms = [n for n in full if n.startswith(f"C({factor})[T.")]
    omnibus: dict = {"note": "no factor terms"}
    if fac_terms:
        R = np.zeros((len(fac_terms), len(full)))
        for k, t in enumerate(fac_terms):
            R[k, full.index(t)] = 1.0
        try:
            w = m.wald_test(R, scalar=True)
            omnibus = {"wald_stat": float(np.ravel(w.statistic)[0]), "p": float(w.pvalue)}
        except Exception as exc:  # noqa: BLE001
            omnibus = {"note": f"wald test unavailable: {exc}"}

    tol = contrast_tolerance(emmeans)
    n_levels = len(levels)
    m_family = n_levels * (n_levels - 1) // 2  # design-determined, not data-determined
    alpha = 0.05
    pairs: list[dict] = []
    for i in range(n_levels):
        for j in range(i + 1, n_levels):
            v = emm_vec(levels[i]) - emm_vec(levels[j])
            est = float(v @ params)
            se = float(np.sqrt(v @ cov @ v))
            pr: dict = {
                "g1": levels[i],
                "g2": levels[j],
                "diff": est,
                "se": se if np.isfinite(se) else None,
            }
            if abs(est) <= tol:
                pr.update(
                    diff=0.0,
                    diff_raw=est,
                    p_bonf=None,
                    significant=None,
                    clamped=True,
                    clamp_tolerance=float(tol),
                    note=(
                        "contrast magnitude at or below the numerical tolerance; reported as "
                        "zero and not tested. It remains in the Bonferroni denominator."
                    ),
                )
                pairs.append(pr)
                continue
            if not (np.isfinite(se) and se > 0):
                # A non-finite or zero SE makes the Wald statistic undefined. Substituting
                # z = 0 (as `z = est / se if se > 0 else 0.0` silently did) yields p = 1.0,
                # asserting "tested, not significant" for a contrast that was never
                # testable. Report "not tested" instead.
                pr.update(
                    p_raw=None,
                    p_bonf=None,
                    significant=None,
                    untestable=True,
                    note=(
                        "the contrast standard error is not finite and positive, so no test "
                        "statistic is defined. It remains in the Bonferroni denominator."
                    ),
                )
                pairs.append(pr)
                continue
            p_raw = float(2.0 * (1.0 - stats.norm.cdf(abs(est / se))))
            pr["p_raw"] = p_raw
            pr["p_bonf"] = float(min(1.0, m_family * p_raw))
            pr["significant"] = bool(p_raw <= alpha / m_family)
            pairs.append(pr)

    n_clamped = sum(1 for pr in pairs if pr.get("clamped"))
    n_untestable = sum(1 for pr in pairs if pr.get("untestable"))
    omnibus["bonferroni_family_size"] = m_family
    omnibus["n_contrasts_clamped"] = n_clamped
    omnibus["n_contrasts_untestable"] = n_untestable
    if pairs and n_clamped == len(pairs):
        omnibus = {
            "note": (
                f"every pairwise EMMeans contrast is at or below the numerical tolerance "
                f"({tol:.3g}); the omnibus Wald test is not identifiable and no p-value "
                "is reported"
            ),
            "wald_stat": omnibus.get("wald_stat"),
            "p": None,
            "p_unclamped": omnibus.get("p"),
            "clamped": True,
            "bonferroni_family_size": m_family,
            "n_contrasts_clamped": n_clamped,
            "n_contrasts_untestable": n_untestable,
        }
    return emmeans, omnibus, pairs


# --------------------------------------------------------------------------- #
# Sensitivity: per-sequence summaries (independent across sequences)
# --------------------------------------------------------------------------- #


def per_sequence_ttest(d: pd.DataFrame, outcome: str, groupcol: str = "defense") -> dict:
    """Welch's t / Mann-Whitney U on per-sequence means (labeled sensitivity)."""

    skip = _skip(d.dropna(subset=[outcome]), outcome)
    if skip is not None:
        skip["label"] = "sensitivity"
        return skip
    per = (
        d.dropna(subset=[outcome])
        .groupby([CLUSTER, groupcol], observed=True)[outcome]
        .mean()
        .reset_index()
    )
    a = per.loc[per[groupcol] == "static", outcome].to_numpy()
    b = per.loc[per[groupcol] == "mtd", outcome].to_numpy()
    if len(a) < 2 or len(b) < 2:
        return {"note": "insufficient sequences", "label": "sensitivity"}
    res = stats.ttest_ind(a, b, equal_var=False)
    t, p = res.statistic, res.pvalue
    u, pu = stats.mannwhitneyu(a, b, alternative="two-sided")
    # Welch-Satterthwaite df, reported so the statistic can be written up as t(df). With one
    # arm constant (zero variance) it reduces to the other arm's n - 1.
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    den = (va**2 / (len(a) - 1)) + (vb**2 / (len(b) - 1))
    welch_df = float((va + vb) ** 2 / den) if den > 0 else None
    return {
        "label": "sensitivity (per-sequence summaries; independent across sequences)",
        "n_sequences_static": int(len(a)),
        "n_sequences_mtd": int(len(b)),
        "mean_static": float(a.mean()),
        "mean_mtd": float(b.mean()),
        "welch_t": {"t": float(t), "df": welch_df, "p": float(p), "cohen_d": cohen_d(a, b)},
        "mann_whitney_u": {"U": float(u), "p": float(pu)},
    }
