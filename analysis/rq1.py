"""RQ1 (H01/Ha1): to what extent does external MTD reduce AI-enabled exploit
success vs. the static baseline?

Trials are clustered within attacker sequences, so the primary comparisons model
that non-independence rather than assume independent trials:

- Binary primary outcomes ``exploit_success`` and co-primary
  ``exploitation_success``: logit **GLMM** (condition fixed, random intercept for
  sequence, attempt index covariate) with a **GEE** fallback; odds ratios + 95% CIs.
- Censored ``time_to_compromise_s`` / ``time_to_first_valid_exploit_s``:
  **cluster-robust Cox** (sequence) as primary + a linear mixed model on log event
  time of the uncensored trials; hazard ratios.
- Welch's t / Mann-Whitney U on **per-sequence summaries** are reported as labeled
  **sensitivity** analyses.

RE-RUN ANALYSIS UPDATE (2026-10-28; ``docs/rerun_runbook.md`` §6).

- **Time summary first: RMST at 300 s.** Each time outcome's block now opens with
  ``rmst_300s``: the restricted mean time to compromise over the first five minutes, MTD vs
  static (the existing sequence cluster bootstrap, ``clustered.rmst_difference``) and per
  condition against the static baseline (``clustered.rmst_by_condition_bootstrap``). The
  1,800-s values (the trial window) follow as ``rmst_trial_window_sensitivity``. The horizon is the
  decision of 2026-10-27 (runbook §1, C290), not a pre-registered value.
- **The 300-s RMST decides the time endpoint (2026-10-28, approved by the chair).**
  ``rmst_300s.mtd_vs_static`` is the decision estimate for H01's time endpoint
  (``time_to_compromise_s``) and supplies the RQ1 secondary family's p for
  ``time_to_first_valid_exploit_s`` (``pipeline._rmst_300s_or_previous``), whatever the
  proportional-hazards screen says. The trial-window RMST, the cluster-robust Cox hazard ratio
  and the ``ph_robust`` block are sensitivity analyses. ``rmst_300s.decision_rule`` records the
  rule in the report.
- **H01 is led by risk differences.** ``h01_lead`` gives, for each binary outcome, the
  compromise proportions and their difference (pooled MTD - static, and each condition -
  static, Holm across the five conditions) with sequence cluster bootstrap intervals, plus the
  Firth odds ratio. It is computed whether or not the static arm is exactly constant: an odds
  ratio against an arm at or near 100% is unstable either way.
"""

from __future__ import annotations

import pandas as pd

from analysis import clustered, separation
from analysis.dataset import STATIC

#: The time-summary horizon the re-run reports first (runbook §1, decided 2026-10-27; C290).
#: Since 2026-10-28 its RMST difference is also the time endpoint's decision estimate.
TIME_SUMMARY_HORIZON_S = 300.0
#: The date the 300-s RMST became the decision estimate (approved by the chair).
TIME_ENDPOINT_RULE_DATE = "2026-10-28"

BINARY_OUTCOMES = ["exploit_success", "exploitation_success"]
# time_to_compromise_s is the obj_b-anchored PRIMARY survival outcome (carries the MTD
# effect); time_to_first_valid_exploit_s is a secondary time-to-event. The enumeration
# diagnostic (time_to_enumeration_s) is a continuous secondary analyzed in RQ2, not here.
TIME_OUTCOMES = [
    "time_to_compromise_s",
    "time_to_first_valid_exploit_s",
]
CENSOR_FLAG = {
    "time_to_compromise_s": "ttc_censored",
    "time_to_first_valid_exploit_s": "tfve_censored",
}
_RHS = "is_mtd + attempt_index"


def _censored_proportion_by_condition(d: pd.DataFrame, flag: str) -> dict:
    if flag not in d.columns:
        return {}
    return {
        str(c): float(pd.to_numeric(g[flag], errors="coerce").mean())
        for c, g in d.groupby("condition", observed=True)
    }


def _ph_robust_block(d: pd.DataFrame, outcome: str, flag: str, cox: dict, horizon: float) -> dict:
    """PH-free survival summaries, computed only when the Schoenfeld screen rejects PH."""

    ph = cox.get("proportional_hazards", {}) if isinstance(cox, dict) else {}
    rank_ok = ph.get("assumption_ok")
    all_ok = ph.get("assumption_ok_all_transforms")
    triggered = (rank_ok is False) or (all_ok is False)
    out: dict = {
        "triggered": bool(triggered),
        "trigger_rule": (
            "Triggered when the Schoenfeld proportional-hazards test REJECTS (p <= .05) for ANY "
            "model term under ANY of the time transforms "
            f"{list(clustered.PH_TRANSFORMS)}. The rank transform alone is tie-sensitive, so the "
            "all-transform rule governs. This is a deliberately LIBERAL SCREEN, not a test at "
            "alpha = .05: it is a minimum over (transforms x terms) correlated tests, so its "
            "false-trigger rate under a true PH model is well above .05. Triggering only adds a "
            "PH-free summary block and never changes an estimate, so over-triggering is the "
            "intended failure direction. The decision is made from the assumption check alone, "
            "never from the effect estimate's own p-value."
        ),
        "ph_min_p_rank": ph.get("min_schoenfeld_p"),
        "ph_min_p_any_transform": ph.get("min_schoenfeld_p_any_transform"),
        "ph_per_term_min_p_any_transform": ph.get("per_term_min_p_any_transform"),
    }
    if not triggered:
        out["note"] = (
            "Proportional hazards not rejected, so the cluster-robust hazard ratio in cox_frailty "
            "is a valid summary. It is a sensitivity analysis: the time endpoint is decided by "
            "rmst_300s.mtd_vs_static (see rmst_300s.decision_rule)."
        )
        return out
    out.update(clustered.ph_robust_survival(d, outcome, flag, horizon=horizon))
    out["decision_note"] = (
        "This block describes how the effect changes over follow-up when proportional hazards "
        "fail. Its rmst_difference uses the trial window and is a sensitivity analysis; the time "
        "endpoint is decided by rmst_300s.mtd_vs_static (see rmst_300s.decision_rule)."
    )
    return out


_SEPARATION_WARNING = (
    "NOT DATA-IDENTIFIED: the static baseline (or the pooled MTD arm) has this outcome in every "
    "trial or in none, so this odds ratio has no maximum-likelihood value and the VB estimate "
    "is set by the estimator's prior (see separation_robust.vb_prior_sensitivity). Report "
    "separation_robust.risk_contrast and the per-sequence sensitivity analysis instead."
)


def _add_separation(d: pd.DataFrame, outcome: str, r: dict) -> None:
    """Flag a separated static-vs-MTD contrast and attach the estimates that are identified."""

    prof = separation.separation_profile(d, outcome, "condition")
    r["separation"] = prof
    if not prof.get("separated"):
        return
    levels = prof.get("levels") or {}
    static_constant = bool((levels.get(STATIC) or {}).get("constant"))
    mtd = pd.to_numeric(d.loc[d["is_mtd"] == 1, outcome], errors="coerce").dropna()
    mtd_constant = bool(len(mtd) and mtd.nunique() == 1)
    if not (static_constant or mtd_constant):
        return  # a single MTD level is constant, but the pooled static-vs-MTD contrast is not
    r["separation_robust"] = separation.separation_robust_block(
        d, outcome, _RHS, "condition", kind="static_vs_mtd", separated_terms=["is_mtd"]
    )
    if isinstance(r.get("mtd_vs_static"), dict):
        r["mtd_vs_static"] = {**r["mtd_vs_static"], "separation_warning": _SEPARATION_WARNING}


_H01_LEAD_NOTE = (
    "H01's binary endpoints are led by compromise proportions and their differences, which "
    "are identified when an arm is at 0% or 100%, where an odds ratio is not. "
    "risk_contrast is pooled MTD - static (a design-weighted average over the five MTD "
    "conditions); by_condition_vs_static gives each condition - static, with p_holm adjusting "
    "the five differences. Both use the sequence cluster bootstrap stratified by condition. "
    "firth is the Firth-penalized odds ratio for is_mtd (attempt_index adjusted); when the "
    "static arm is constant its contrast is flagged involves_separated_level and is finite only "
    "because of the penalty (see firth.note). The VB odds ratio in mtd_vs_static is NOT the "
    "lead estimate."
)


def _h01_lead(d: pd.DataFrame, outcome: str, r: dict) -> dict:
    """Risk-difference lead for a binary H01 endpoint, reusing the separation block if present."""

    sep = r.get("separation_robust") or {}
    static_vals = pd.to_numeric(d.loc[d["condition"] == STATIC, outcome], errors="coerce")
    static_constant = bool(len(static_vals.dropna()) and static_vals.dropna().nunique() == 1)
    risk = sep.get("risk_contrast") or separation.risk_contrast_static_vs_mtd(d, outcome)
    firth = sep.get("firth") or separation.firth_block(
        d, outcome, _RHS, separated_terms=["is_mtd"] if static_constant else []
    )
    by_cond = separation.technique_rates(d, outcome, "condition", reference=STATIC)
    return {
        "estimand": "difference in compromise proportion, MTD - static (per trial)",
        "risk_contrast": risk,
        "by_condition_vs_static": by_cond,
        "firth": firth,
        "static_outcome_constant": static_constant,
        "note": _H01_LEAD_NOTE,
    }


_RMST_300_HORIZON_NOTE = (
    "HORIZON. 300 s is the time-summary horizon decided on 2026-10-27 (runbook §1; C290), not a "
    "pre-registered value; since 2026-10-28 this RMST difference is the time endpoint's decision "
    "estimate (see decision_rule). The RMST difference scales with the horizon: over the 1,800-s trial "
    "window (rmst_trial_window_sensitivity) it mostly measures how many trials were never "
    "compromised (RMST is close to horizon x the uncompromised share). A horizon beyond the "
    "reference arm's last follow-up is admissible here only because no trial is censored before "
    "it (censoring_times_before_horizon = 0): every censored trial is censored at the end of the "
    "trial window, so both Kaplan-Meier curves are fully determined on [0, 300] and RMST equals "
    "mean(min(T, 300))."
)


_TIME_ENDPOINT_RULE = (
    "DECISION RULE (decided 2026-10-28 and approved by the chair; not pre-registered). The "
    "difference in restricted mean time to compromise over [0, 300 s], MTD - static "
    "(mtd_vs_static), decides the time endpoint, whatever the proportional-hazards screen says. "
    "For time_to_compromise_s (H01's primary time endpoint) it is evaluated at alpha = .05 "
    "without correction; its p is mtd_vs_static.p_normal_approx, the unfloored Wald p from the "
    "sequence cluster bootstrap, written as 'p < .001' when below that. For "
    "time_to_first_valid_exploit_s the same p enters the RQ1 secondary Benjamini-Hochberg family "
    "(fdr_secondary.rq1_p_source). A positive difference means the MTD arm stays uncompromised "
    "longer. by_condition gives each condition against the static baseline (Holm across the "
    "five). Sensitivity analyses: rmst_trial_window_sensitivity (the 1,800-s window), cox_frailty "
    "(the cluster-robust hazard ratio) and ph_robust. The rule was set while the main study was "
    "running, after the interim checks of exclusions and of per-condition compromise and "
    "staleness proportions and before any model was fitted to its data; the 300-s horizon itself "
    "was fixed on 2026-10-27, before the study started."
)


def _rmst_block(
    d: pd.DataFrame, outcome: str, flag: str, horizon: float, note: str, *, decision: bool = False
) -> dict:
    pooled = clustered.rmst_difference(d, outcome, flag, horizon=horizon)
    if isinstance(pooled, dict) and "horizon_note" in pooled:
        pooled = {**pooled, "horizon_note": note}
    out = {
        "horizon_s": float(horizon),
        "mtd_vs_static": pooled,
        "by_condition": clustered.rmst_by_condition_bootstrap(
            d, outcome, flag, horizon=horizon, reference=STATIC
        ),
    }
    if decision:
        out["role"] = "DECISION ESTIMATE for this time outcome (mtd_vs_static)"
        out["decision_rule"] = _TIME_ENDPOINT_RULE
    return out


def rq1(df: pd.DataFrame) -> dict:
    """Run the clustered RQ1 analyses on the (screened) trial dataset."""

    d = df.copy()
    d["is_mtd"] = (d["defense"] == "mtd").astype(int)
    results: dict = {}

    for outcome in BINARY_OUTCOMES:
        if outcome not in d.columns:
            continue
        r = clustered.binary_clustered(d, outcome, _RHS)
        if r.get("terms", {}).get("is_mtd"):
            r["mtd_vs_static"] = r["terms"]["is_mtd"]  # headline odds ratio
        r["sensitivity"] = clustered.per_sequence_ttest(d, outcome)
        if not r.get("skipped"):
            _add_separation(d, outcome, r)
            r = {"h01_lead": _h01_lead(d, outcome, r), **r}
        results[outcome] = r

    for outcome in TIME_OUTCOMES:
        if outcome not in d.columns:
            continue
        flag = CENSOR_FLAG[outcome]
        cox = clustered.cox_cluster(d, outcome, flag, _RHS)
        if cox.get("hazard_ratios", {}).get("is_mtd"):
            # The hazard ratio for is_mtd: a sensitivity analysis since 2026-10-28 (rmst_300s decides).
            cox["mtd_vs_static"] = cox["hazard_ratios"]["is_mtd"]
        horizon = clustered.survival_horizon(d, outcome)
        block = {
            # The re-run's time summary, reported FIRST (C290) and, since 2026-10-28, the time
            # endpoint's decision estimate; the trial window follows as a sensitivity.
            "rmst_300s": _rmst_block(
                d, outcome, flag, TIME_SUMMARY_HORIZON_S, _RMST_300_HORIZON_NOTE, decision=True
            ),
            "rmst_trial_window_sensitivity": {
                **_rmst_block(d, outcome, flag, horizon, clustered._RMST_HORIZON_NOTE),
                "label": "SENSITIVITY: the trial window (1,800 s in the study) as horizon",
            },
            "cox_frailty": cox,
            "lmm_log_event_time": clustered.lmm_log_event_time(d, outcome, flag, _RHS),
            "sensitivity": clustered.per_sequence_ttest(d, outcome),
            "censored_proportion_by_condition": _censored_proportion_by_condition(d, flag),
            "km_by_condition": clustered.rmst_by_condition(
                d, outcome, flag, horizon=horizon, reference=STATIC
            ),
        }
        block["ph_robust"] = _ph_robust_block(d, outcome, flag, cox, horizon)
        results[outcome] = block
    return results
