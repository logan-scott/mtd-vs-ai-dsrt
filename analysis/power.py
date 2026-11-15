"""A-priori power analysis (per primary outcome) + clustering design effect.

Computes the a-priori requirement for each primary outcome/test and reports the
most conservative:
- continuous (t / mixed): d = 0.50 -> 64 per condition;
- binary (two-proportion, and the chi-square test of the SAME 2x2 contrast): OR = 2.0
  (p .50 vs .667; Cohen's h = .340, w = .169) -> 136-138 per condition;
- censored (event-based survival / log-rank): required number of events.

Because trials are clustered within sequences of m = 8 attempts, the effective
sample size is reduced by the design effect DE = 1 + (m - 1)rho, where rho is the
within-sequence intraclass correlation estimated from the repeated-attempt data.
The requirement is therefore expressed in **sequences per condition**.

Two design-effect blocks are reported and must not be confused. ``a_priori_design_effect``
is the PILOT-derived justification the study was powered on and N was locked to;
``design_effect_adjustment`` recomputes rho from the CONFIRMATORY data and is POST HOC.
``power_adequacy`` compares them and refuses to claim over-powering when the primary
outcome is constant within conditions (where rho is 0 by construction, not by weak
clustering).

CORRECTION (2026-10-27). Until this date the binary requirement used the ONE-sample
arcsine formula ``((z_a + z_b) / h)^2``, which is half the per-group n of a comparison of
two independent proportions, ``2 ((z_a + z_b) / h)^2`` (Cohen, 1988, Table 6.4.1: n = 63
per group at h = .50). It reported 68 per condition instead of 136, and the chi-square line
used w = .30, which is not the effect size of the stated contrast (w = .169 for .50 vs .667
with equal groups). The study's N (33 sequences per condition) was locked to that
arithmetic. ``a_priori()["as_planned_uncorrected"]`` and
``a_priori_design_effect()["as_planned_uncorrected"]`` reproduce the archived plan exactly
so the record stays auditable; every other figure here is the corrected one.

The planning rho was miscomputed too. The pooled within-condition ICC used k - 1
between-sequence degrees of freedom after removing the condition means, where k - c is
correct (c conditions), and it counted conditions whose outcome is constant, which carry no
information about rho. On the four-sequence pilot (one condition, two sequences, with any
variance) the correct estimate is .429, not the .314 the study was planned on, and its
95% interval is [.014, .999]: the pilot does not constrain rho. The corrected a-priori chain
uses the corrected rho; ``formula_correction_only`` holds the planning rho fixed so the
effect of each correction can be traced.

RE-RUN ANALYSIS UPDATE (2026-10-28; ``docs/rerun_runbook.md`` §1, §6). The re-run's N is 80
sequences per condition, set from the corrected chain at the July pilot's rho (C243:
138 x DE 4.0 = 552 trials -> 69 sequences -> 80 with the 15% buffer). ``run_design_basis``
puts the run's own N (read from its ``_run_manifest.json`` by ``cli.py analyze``) in place of
the September run's 33 as the locked N. ``rerun_planning_check`` is the re-run's planning
check: the implied power of the planned primary contrast and of each of the 10 Bonferroni
pairwise comparisons among the five MTD techniques at N = 80, under the July pilot's rho and
under the October clock-validation pilot's rho (real-clock arm; ``OCTOBER_PILOT_RHO``) and its
interval ends. It uses the planning effect size, never an observed one, so it is not observed
power.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.power import FTestAnovaPower, GofChisquarePower, TTestIndPower

from analysis.effects import cohen_d

ATTEMPTS_PER_SEQUENCE = 8
_Z_ALPHA = 1.959963985  # z_{.975}
_Z_BETA = 0.841621234  # z_{.80}

# ---------------------------------------------------------------------------
# A-PRIORI design basis. rho is the ONLY frozen number: the design effect and the
# sequence chain are re-derived below with the same arithmetic the pilot used, so
# this constant cannot silently drift from the archived pilot report.
# Source of truth (all three agree):
#   docs/results/pilot/analysis_report.json
#       -> power.design_effect_adjustment            (rho .3143, DE 3.2, 28 -> 33 sequences)
#       -> power.within_sequence_icc_by_outcome
#          .exploitation_success.within_condition     (0.3142857142857143 == 11/35)
#   docs/pilot_calibration.md                         (rho 0.314 / DE 3.2 -> 28-33 sequences)
#   IRB power narrative                               (68 -> 218 -> 28 -> 33 -> 264)
# N was locked to 33 sequences/condition:
#   docs/results/full_20260722/_run_manifest.json -> config.sequences_per_condition
# The 68 in that chain is the UNCORRECTED (one-sample) binary requirement, and the .3143 is
# the miscomputed planning rho; see the module docstring. The corrected rho and its interval
# below are frozen from docs/results/pilot/trial_dataset.csv and pinned by a test
# against that archive.
# ---------------------------------------------------------------------------
PILOT_DESIGN_BASIS = {
    "source": "Method C live pilot, archived 2026-07-21",
    "source_report": (
        "docs/results/pilot/analysis_report.json -> power.design_effect_adjustment"
    ),
    "source_doc": "docs/pilot_calibration.md (rho = 0.314 / DE = 3.2 -> 28-33 sequences/condition)",
    "outcome_for_rho": "exploitation_success",
    "rho_basis": "within-condition ICC(1) on the binary co-primary (no censoring bias)",
    # The planning value, exactly as computed at the time: k - 1 between-sequence df, with the
    # constant static condition counted. MISCOMPUTED; kept to reproduce the archived chain.
    "planning_rho_as_computed": 11.0 / 35.0,  # 0.3142857...; published as 0.3143
    # The same pilot data computed correctly: nested df (k - c) over the conditions whose
    # outcome varies. Only port_rotation varies (sequences at 8/8 and 4/8), so k = 2, c = 1
    # and F = 7.0 on 1 and 14 df.
    "within_sequence_icc_rho": 3.0 / 7.0,  # 0.4285714...
    "within_sequence_icc_rho_ci95": [0.0137, 0.9988],
    "pilot_sequences": 4,
    "pilot_sequences_with_outcome_variance": 1,
    "pilot_sequences_used_for_rho": 2,
    "attempts_per_sequence_m": ATTEMPTS_PER_SEQUENCE,
    "exclusion_inflation": 0.15,
    "locked_sequences_per_condition": 33,
    "locked_source": (
        "docs/results/full_20260722/_run_manifest.json -> config.sequences_per_condition"
    ),
}

#: The re-run's design (docs/rerun_runbook.md §1, decided 2026-10-27; C243).
RERUN_DESIGN_BASIS = {
    "sequences_per_condition": 80,
    "attempts_per_sequence_m": ATTEMPTS_PER_SEQUENCE,
    "source": "docs/rerun_runbook.md §1 (decided 2026-10-27; C243)",
    "derivation": (
        "the corrected a-priori chain at the July pilot's rho (.429): 138 trials per condition "
        "(chi-square, w = .169) x design effect 4.0 = 552 trials -> 69 sequences -> 80 with the "
        "15% exclusion buffer"
    ),
}

#: The October clock-validation pilot's rho, the re-run's planning check (runbook §1). Its
#: real-clock (wall) arm is the arm that ran the re-run's clock; 24 sequences in the four
#: moving conditions, none of them constant. Pinned by a test against the archive.
OCTOBER_PILOT_RHO = {
    "source": "2026-10-26 clock-validation pilot, real-clock (wall) arm",
    "source_data": "docs/results/clock_validation/arm_wall/trial_dataset.csv",
    "outcome_for_rho": "exploitation_success",
    "rho": 0.12414965986394556,
    "rho_ci95": [0.0233, 0.3079],
    "sequences": 24,
    "conditions": [
        "combined_mtd_deception",
        "endpoint_mutation",
        "network_shuffling",
        "port_rotation",
    ],
    "df_between": 20,
    "df_within": 168,
    "method": (
        "within-condition ICC(1), nested df (k - c), constant conditions left out, F-based 95% "
        "interval (power.icc_with_interval)"
    ),
    "modeled_arm_rho": 0.0301,
    "modeled_arm_rho_ci95": [0.0, 0.1724],
}

# The planning contrast for the binary co-primary: compromise .50 under MTD vs .667 (OR = 2.0).
PLANNING_P1 = 0.5
PLANNING_ODDS_RATIO = 2.0

_POST_HOC_LABEL = (
    "POST HOC. rho here is recomputed from the CONFIRMATORY data, so this block describes the "
    "clustering that was realized, not the basis on which N was chosen. The a-priori "
    "justification (pilot rho, design effect, locked N) is power.a_priori_design_effect; the "
    "comparison of the two is power.power_adequacy. Do not cite this block as the sample-size "
    "justification -- doing so would be retrospective (observed-power) reasoning."
)


def _p2_from_odds_ratio(p1: float, odds_ratio: float) -> float:
    return (odds_ratio * p1) / (1 - p1 + odds_ratio * p1)


def cohens_h(p1: float, odds_ratio: float) -> float:
    """Cohen's h between ``p1`` and the proportion an odds ratio of ``odds_ratio`` implies."""

    p2 = _p2_from_odds_ratio(p1, odds_ratio)
    return abs(2 * math.asin(math.sqrt(p2)) - 2 * math.asin(math.sqrt(p1)))


def _two_proportion_n(p1: float, odds_ratio: float, z_alpha: float = _Z_ALPHA) -> int | None:
    """Per-group n for comparing two INDEPENDENT proportions via Cohen's arcsine h.

    n_per_group = 2 ((z_a + z_b) / h)^2. The arcsine transform stabilises each proportion's
    variance at 1/(4n); the DIFFERENCE of two independent arcsine estimates has variance
    2 x 1/(4n), which is where the factor of 2 comes from (Cohen, 1988, section 6.2; his
    Table 6.4.1 gives n = 63 per group at h = .50, alpha = .05 two-sided, power = .80, and
    2 x (2.8016 / .50)^2 = 62.8). Cross-checked in the tests against
    ``statsmodels.stats.power.NormalIndPower`` with ``proportion_effectsize``.

    ``z_alpha`` defaults to z_{.975}; pass a larger quantile for a multiplicity-adjusted
    per-contrast alpha.
    """

    h = cohens_h(p1, odds_ratio)
    if h <= 0:
        return None
    return math.ceil(2.0 * ((z_alpha + _Z_BETA) / h) ** 2)


def _two_proportion_n_as_planned(
    p1: float, odds_ratio: float, z_alpha: float = _Z_ALPHA
) -> int | None:
    """HISTORICAL, WRONG for planning: the one-sample formula the archived plan used.

    ``((z_a + z_b) / h)^2`` is the n for ONE proportion against a known value; for two
    independent groups it is half the correct per-group n. Retained only so that
    ``as_planned_uncorrected`` reproduces the archived pilot report and the IRB chain
    (68 -> 218 -> 28 -> 33) exactly. Never use it to size a study.
    """

    h = cohens_h(p1, odds_ratio)
    if h <= 0:
        return None
    return math.ceil(((z_alpha + _Z_BETA) / h) ** 2)


def two_by_two_w(p1: float, odds_ratio: float) -> float:
    """Cohen's w of the 2x2 (group x outcome) table for the planning contrast, equal groups.

    w^2 = sum (P1 - P0)^2 / P0 over the four cells reduces, for two equal groups, to
    w = |p1 - p2| / (2 sqrt(pbar (1 - pbar))) with pbar = (p1 + p2) / 2. For .50 vs .667
    this is .169, the effect size the chi-square test of THIS contrast has (the archived plan
    used w = .30, which corresponds to roughly .50 vs .79 instead).
    """

    p2 = _p2_from_odds_ratio(p1, odds_ratio)
    pbar = (p1 + p2) / 2.0
    return abs(p1 - p2) / (2.0 * math.sqrt(pbar * (1.0 - pbar)))


def _required_events(hazard_ratio: float, alloc: float = 0.5) -> int:
    """Schoenfeld required number of events for a log-rank/Cox comparison."""

    za, zb = 1.959963985, 0.841621234  # z_{.975}, z_{.80}
    events = (za + zb) ** 2 / (alloc * (1 - alloc) * math.log(hazard_ratio) ** 2)
    return math.ceil(events)


_AS_PLANNED_CHI_SQUARE_W = 0.3

_A_PRIORI_CORRECTION_NOTE = (
    "CORRECTED 2026-10-27. Binary requirements now use the two-independent-sample formula "
    "n per group = 2((z_a + z_b)/h)^2 (Cohen, 1988), and the chi-square line uses the effect "
    "size of the planning contrast itself (w = .169 for .50 vs .667 with equal groups) instead "
    "of w = .30. as_planned_uncorrected reproduces the figures the study was actually sized on "
    "(binary 68 per condition, chi-square N = 88 at w = .30); they are retained for the record "
    "and must not be cited as requirements. Continuous (t), ANOVA and survival lines were "
    "already correct and are unchanged."
)


def _chi_square_total(w: float, alpha: float = 0.05) -> int:
    """Total N for a 1-df chi-square test at Cohen's ``w`` (power = .80)."""

    return math.ceil(
        GofChisquarePower().solve_power(effect_size=w, alpha=alpha, power=0.80, n_bins=2)
    )


def a_priori() -> dict:
    """Per-primary-outcome a-priori requirements and the binding per-condition n."""

    n_t = math.ceil(
        TTestIndPower().solve_power(
            effect_size=0.5, alpha=0.05, power=0.80, alternative="two-sided"
        )
    )
    w = two_by_two_w(PLANNING_P1, PLANNING_ODDS_RATIO)
    chi_total = _chi_square_total(w)
    prop_per_group = _two_proportion_n(PLANNING_P1, PLANNING_ODDS_RATIO)
    anova_total = math.ceil(
        FTestAnovaPower().solve_power(effect_size=0.25, alpha=0.05, power=0.80, k_groups=5)
    )
    anova_per_group = math.ceil(anova_total / 5)

    per_cond_candidates = {
        "continuous_t": n_t,
        "binary_two_proportion": prop_per_group,
        "binary_chi_square": math.ceil(chi_total / 2),
        "anova_five_group": anova_per_group,
    }
    binding = max(v for v in per_cond_candidates.values() if v is not None)

    # The archived plan, reproduced with its own (incorrect) arithmetic -- for the record.
    planned_prop = _two_proportion_n_as_planned(PLANNING_P1, PLANNING_ODDS_RATIO)
    planned_chi_total = _chi_square_total(_AS_PLANNED_CHI_SQUARE_W)
    planned_candidates = {
        "continuous_t": n_t,
        "binary_two_proportion": planned_prop,
        "binary_chi_square": math.ceil(planned_chi_total / 2),
        "anova_five_group": anova_per_group,
    }
    p2 = _p2_from_odds_ratio(PLANNING_P1, PLANNING_ODDS_RATIO)
    return {
        "continuous_ttest": {
            "effect_size_d": 0.5,
            "n_per_condition": n_t,
            "n_total_two_group": 2 * n_t,
        },
        "binary_two_proportion": {
            "odds_ratio": PLANNING_ODDS_RATIO,
            "p1": PLANNING_P1,
            "p2": round(p2, 4),
            "cohens_h": round(cohens_h(PLANNING_P1, PLANNING_ODDS_RATIO), 4),
            "formula": "n per group = 2((z_a + z_b)/h)^2 (two independent proportions)",
            "n_per_condition": prop_per_group,
        },
        "binary_chi_square": {
            "effect_size_w": round(w, 4),
            "w_basis": (
                "2x2 (group x compromised) test of independence for the same planning contrast "
                "(.50 vs .667, equal groups); 1 df"
            ),
            "n_total": chi_total,
            "n_per_condition": math.ceil(chi_total / 2),
        },
        "anova_five_group": {
            "effect_size_f": 0.25,
            "n_total": anova_total,
            "n_per_condition": anova_per_group,
            # G*Power reports the total in whole groups (k x ceil(n/k)), not the continuous N.
            "n_total_whole_groups": 5 * anova_per_group,
        },
        "censored_survival": {
            "hazard_ratio": 2.0,
            "required_events": _required_events(2.0),
            "note": "trial requirement = required_events / pilot compromise rate",
        },
        "per_condition_by_test": per_cond_candidates,
        "binding_per_condition": binding,
        "as_planned_uncorrected": {
            "binary_two_proportion": {
                "n_per_condition": planned_prop,
                "formula": "((z_a + z_b)/h)^2 -- the ONE-sample formula; half the correct n",
            },
            "binary_chi_square": {
                "effect_size_w": _AS_PLANNED_CHI_SQUARE_W,
                "n_total": planned_chi_total,
                "note": "w = .30 is not the effect size of .50 vs .667 (that is w = .169)",
            },
            "per_condition_by_test": planned_candidates,
            "binding_per_condition": max(v for v in planned_candidates.values() if v is not None),
            "status": "SUPERSEDED -- reproduces the archived plan; do not cite as a requirement",
        },
        "correction_note": _A_PRIORI_CORRECTION_NOTE,
    }


def design_effect(rho: float, m: int = ATTEMPTS_PER_SEQUENCE) -> float:
    return 1.0 + (m - 1) * rho


_ICC_METHOD = (
    "one-way random-effects ICC(1) from the ANOVA mean squares (sequences as groups); the "
    "point estimate is truncated at 0. 95% interval: the F-based interval for ICC(1) "
    "(Shrout & Fleiss, 1979; McGraw & Wong, 1996), F = MSB/MSW on (df_between, df_within). It "
    "is exact for equal group sizes under the normal one-way random-effects model; with the "
    "coefficient n0 for unequal sizes, and for a binary outcome, it is an approximation. It is "
    "truncated to [0, 1]. Within-condition estimates center on condition means, use k - c "
    "between-sequence df, and leave out conditions whose outcome is constant (they carry no "
    "information about rho but would add degrees of freedom and narrow the interval)."
)


def icc_with_interval(
    df: pd.DataFrame,
    outcome: str,
    group: str = "sequence_id",
    condition: str | None = None,
    *,
    nested_df: bool = True,
    exclude_constant_conditions: bool | None = None,
    alpha: float = 0.05,
) -> dict | None:
    """ICC(1) of ``outcome`` within ``group``, its F statistic and an F-based interval.

    When ``condition`` is given, the outcome is centered **within each condition** first, so
    the ICC measures within-*condition* between-sequence clustering (the quantity a design
    effect needs) rather than the between-condition treatment effect. Removing the c condition
    means uses up c - 1 of the k - 1 between-sequence degrees of freedom, so the between-
    sequence mean square has **k - c** df (``nested_df=True``, the correct nested-design
    value). Conditions whose outcome is constant are left out first
    (``exclude_constant_conditions``, default on with ``nested_df``): they add nothing to either
    sum of squares, so they carry no information about rho, but their degrees of freedom would
    narrow the interval. ``nested_df=False`` (with constant conditions kept) is what the
    pipeline did until 2026-10-27, kept only so the archived pilot value (.3143) can be
    reproduced.

    Returns ``None`` when there are too few groups to estimate anything.
    """

    from scipy.stats import f as fdist

    cols = [group, outcome] + ([condition] if condition else [])
    if any(c not in df.columns for c in cols):
        return None
    d = df[cols].copy()
    d[outcome] = pd.to_numeric(d[outcome], errors="coerce")
    d = d.dropna(subset=[group, outcome])
    n_cond = 1
    excluded: list[str] = []
    if exclude_constant_conditions is None:
        exclude_constant_conditions = nested_df
    if condition is not None:
        d = d.dropna(subset=[condition])
        if exclude_constant_conditions:
            rng = d.groupby(condition, observed=True)[outcome].agg(lambda v: v.max() - v.min())
            scale = d.groupby(condition, observed=True)[outcome].agg(lambda v: v.abs().max())
            constant = rng <= 1e-9 * scale.clip(lower=1.0)
            excluded = sorted(str(c) for c in constant[constant].index)
            d = d[~d[condition].astype(str).isin(excluded)]
        # Remove each condition's mean so between-condition variance does not enter.
        d[outcome] = d[outcome] - d.groupby(condition, observed=True)[outcome].transform("mean")
        n_cond = int(d[condition].nunique())
    grouped = d.groupby(group, observed=True)[outcome]
    sizes = grouped.size().to_numpy(dtype=float)
    k = int(len(sizes))
    n_total = float(sizes.sum())
    df_b = (k - n_cond) if (condition is not None and nested_df) else (k - 1)
    df_w = n_total - k
    if k < 2 or df_b < 1 or df_w < 1:
        return None
    means = grouped.mean().to_numpy(dtype=float)
    grand = float(d[outcome].mean())
    ss_b = float((sizes * (means - grand) ** 2).sum())
    ss_w = float(((d[outcome] - grouped.transform("mean")) ** 2).sum())
    msb = ss_b / df_b
    msw = ss_w / df_w
    # n0, the "average" group size for unequal groups. In the nested design the squared sizes
    # are taken relative to each condition's total, which reduces to m when every sequence
    # has m attempts.
    if condition is not None and nested_df:
        per_cond = d.groupby([condition, group], observed=True).size()
        sq = per_cond.pow(2).groupby(level=0, observed=True).sum()
        tot = per_cond.groupby(level=0, observed=True).sum()
        n0 = (n_total - float((sq / tot).sum())) / df_b
    else:
        n0 = (n_total - float((sizes**2).sum()) / n_total) / df_b
    denom = msb + (n0 - 1.0) * msw
    raw = float((msb - msw) / denom) if denom > 0 else 0.0
    out = {
        "rho": max(0.0, raw),
        "rho_untruncated": raw,
        "ci95": None,
        "F": None,
        "df_between": int(df_b),
        "df_within": int(df_w),
        "n0": float(n0),
        "k_groups": k,
        "c_conditions": n_cond,
        "conditions_excluded_constant_outcome": excluded,
        "df_convention": "nested (k - c)" if (condition is not None and nested_df) else "k - 1",
        "method": _ICC_METHOD,
    }
    if msw <= 0:
        out["note"] = (
            "no within-sequence variance (every sequence constant): F and its interval are "
            "undefined"
        )
        return out
    F = msb / msw
    out["F"] = float(F)
    q = 1.0 - alpha / 2.0
    f_lo = F / float(fdist.ppf(q, df_b, df_w))
    f_hi = F * float(fdist.ppf(q, df_w, df_b))
    lo = (f_lo - 1.0) / (f_lo + n0 - 1.0)
    hi = (f_hi - 1.0) / (f_hi + n0 - 1.0)
    out["ci95"] = [float(min(1.0, max(0.0, lo))), float(min(1.0, max(0.0, hi)))]
    return out


def within_sequence_icc(
    df: pd.DataFrame,
    outcome: str,
    group: str = "sequence_id",
    condition: str | None = None,
) -> float | None:
    """One-way (ICC1) within-sequence intraclass correlation of ``outcome`` (point value).

    See :func:`icc_with_interval` for the estimator, the nested-design degrees of freedom
    and the interval.
    """

    res = icc_with_interval(df, outcome, group=group, condition=condition)
    return None if res is None else float(res["rho"])


def within_sequence_icc_by_outcome(df: pd.DataFrame, outcomes: list[str]) -> dict:
    """Within-sequence ICC(1) per outcome — pooled and within-condition.

    Surfaces ρ for the graded primary/co-primary outcomes directly. The pooled value
    conflates the treatment effect (sequences are nested in conditions); the
    ``within_condition`` value centers by condition first and is the design-effect-
    relevant clustering. Reported on the BINARY ``exploitation_success`` (no censoring)
    and on the times, so a reader can see that ρ became estimable + non-zero on the
    outcomes that carry the MTD effect. Time-based ρ on a heavily-censored outcome is
    computed on the recorded values (censored == the administrative limit) and is
    descriptive only — the design effect below uses the binary co-primary.
    """

    from analysis.degeneracy import variance_profile

    out: dict = {}
    for oc in outcomes:
        if oc not in df.columns:
            continue
        prof = variance_profile(df, oc, "condition")
        if prof["status"] in ("constant", "insufficient_data", "missing"):
            # ss_between and ss_within are both 0 here, so the ICC formula takes its
            # `denom <= 0 -> 0.0` branch. That 0.0 is the degenerate branch, not an
            # estimate, and must not be read as "no clustering".
            out[oc] = {
                "pooled": None,
                "within_condition": None,
                "skipped": True,
                "skipped_reason": (
                    "zero_variance" if prof["status"] == "constant" else prof["status"]
                ),
                "note": (
                    f"{oc} has no resolvable variance, so the intraclass correlation is "
                    "undefined; no rho is reported"
                ),
                "variance_profile": prof,
            }
            continue
        wc = icc_with_interval(df, oc, condition="condition")
        out[oc] = {
            "pooled": within_sequence_icc(df, oc),
            "within_condition": None if wc is None else wc["rho"],
            "within_condition_ci95": None if wc is None else wc["ci95"],
            "within_condition_detail": wc,
        }
    return out


def design_effect_adjustment(
    df: pd.DataFrame,
    outcome: str = "exploitation_success",
    m: int = ATTEMPTS_PER_SEQUENCE,
    exclusion_inflation: float = 0.15,
) -> dict:
    """Design-effect-adjusted requirement in sequences per condition.

    ``outcome`` defaults to the BINARY co-primary ``exploitation_success`` and ρ is
    computed **within condition** (below), so the design effect reflects genuine
    within-sequence clustering of the primary outcome, free of both the between-
    condition treatment effect and the censored-time magnitude bias that computing
    ρ on the censored ``time_to_compromise_s`` (censored == the 1800 s limit) would introduce.
    """

    binding = a_priori()["binding_per_condition"]
    # Within-condition ρ on the binary co-primary: the design-effect-relevant clustering,
    # free of the between-condition effect and of any censored-time magnitude bias.
    detail = icc_with_interval(df, outcome, condition="condition")
    rho = None if detail is None else detail["rho"]
    by_cond = rho_by_condition(df, outcome)
    n_degen = sum(
        1 for v in by_cond.values() if isinstance(v, dict) and v.get("outcome_degenerate")
    )
    if rho is None:
        return {
            "status": "post_hoc_realized",
            "is_the_sample_size_justification": False,
            "note": "cannot estimate within-sequence ICC",
            "binding_per_condition_unadjusted": binding,
            "rho_by_condition": by_cond,
            "n_conditions_with_degenerate_outcome": n_degen,
            "post_hoc_note": _POST_HOC_LABEL,
        }
    return {
        "status": "post_hoc_realized",
        "is_the_sample_size_justification": False,
        "rho_source": "confirmatory data (recomputed)",
        "outcome_for_rho": outcome,
        "rho_basis": "within-condition ICC(1) on the binary co-primary (no censoring bias)",
        **_sequences_from_rho(rho, binding, m, exclusion_inflation),
        "rho_ci95": detail["ci95"],
        "rho_untruncated": round(float(detail["rho_untruncated"]), 4),
        "rho_df": [detail["df_between"], detail["df_within"]],
        "rho_df_convention": detail["df_convention"],
        "rho_method": _ICC_METHOD,
        "rho_by_condition": by_cond,
        "n_conditions_with_degenerate_outcome": n_degen,
        "post_hoc_note": _POST_HOC_LABEL,
    }


def _sequences_from_rho(rho: float, binding: int, m: int, exclusion_inflation: float) -> dict:
    """rho -> design effect -> adjusted trials -> sequences per condition.

    Shared by the a-priori and post-hoc blocks so the two arithmetics cannot drift apart.
    """

    de = design_effect(rho, m)
    adjusted = math.ceil(binding * de)
    sequences = math.ceil(adjusted / m)
    sequences_infl = math.ceil(sequences * (1 + exclusion_inflation))
    return {
        "within_sequence_icc_rho": round(float(rho), 4),
        "attempts_per_sequence_m": int(m),
        "design_effect": round(float(de), 3),
        "binding_per_condition_unadjusted": int(binding),
        "adjusted_trials_per_condition": int(adjusted),
        "sequences_per_condition": int(sequences),
        "sequences_per_condition_with_15pct_exclusion": int(sequences_infl),
        "trials_per_condition_with_inflation": int(sequences_infl * m),
    }


def rho_by_condition(
    df: pd.DataFrame,
    outcome: str = "exploitation_success",
    group: str = "sequence_id",
) -> dict:
    """Per-condition within-sequence rho with an explicit outcome-degeneracy flag.

    A pooled within-condition rho of ~0 has two very different causes: genuinely weak
    clustering, or an outcome that is CONSTANT inside a condition (every trial the same),
    where the intraclass correlation is zero by construction and carries no information
    about clustering. ``outcome_degenerate`` is keyed on ``nunique() < 2``, which is
    invariant to the library versions that move the model-fit diagnostics around.
    """

    if "condition" not in df.columns or outcome not in df.columns or not len(df):
        return {"note": "cannot estimate per-condition rho"}
    out: dict = {}
    for cond, g in df.groupby("condition", observed=True):
        y = pd.to_numeric(g[outcome], errors="coerce").dropna()
        degenerate = bool(y.nunique() < 2)
        detail = None if degenerate else icc_with_interval(g, outcome, group=group)
        out[str(cond)] = {
            "n": int(len(y)),
            "n_sequences": int(g[group].nunique()) if group in g.columns else None,
            "outcome_mean": round(float(y.mean()), 4) if len(y) else None,
            "n_distinct_values": int(y.nunique()),
            "outcome_degenerate": degenerate,
            "rho": None if detail is None else float(detail["rho"]),
            "rho_ci95": None if detail is None else detail["ci95"],
        }
    return out


def a_priori_design_effect(basis: dict | None = None) -> dict:
    """The A-PRIORI (pilot-derived) design effect and the corrected sample-size requirement.

    This -- not ``design_effect_adjustment`` -- is the sample-size justification. Only rho is
    frozen (from the archived pilot); the design effect and the sequence chain are re-derived,
    so any drift against the archive shows up in ``locked_matches_plan`` and is caught by the
    test suite. Three chains are reported:

    - the headline: the corrected binary requirement at the CORRECTED pilot rho (nested df,
      constant conditions left out), with the range of requirements over its 95% interval;
    - ``formula_correction_only``: the corrected requirement at the miscomputed planning rho
      (.3143), isolating the effect of the formula correction;
    - ``as_planned_uncorrected``: the archived plan (68 -> 218 -> 28 -> 33), reproduced exactly.
    """

    b = dict(PILOT_DESIGN_BASIS if basis is None else basis)
    ap = a_priori()
    binding = ap["binding_per_condition"]
    m = b["attempts_per_sequence_m"]
    infl = b["exclusion_inflation"]
    planning_rho = b["planning_rho_as_computed"]
    chain = _sequences_from_rho(b["within_sequence_icc_rho"], binding, m, infl)
    formula_only = _sequences_from_rho(planning_rho, binding, m, infl)
    planned = _sequences_from_rho(
        planning_rho, ap["as_planned_uncorrected"]["binding_per_condition"], m, infl
    )
    ci = b.get("within_sequence_icc_rho_ci95")
    locked = b.get("locked_sequences_per_condition")
    out = {
        "status": "a_priori",
        "is_the_sample_size_justification": True,
        "rho_source": "pilot",
        "source": b["source"],
        "source_report": b["source_report"],
        "source_doc": b["source_doc"],
        "outcome_for_rho": b["outcome_for_rho"],
        "rho_basis": (
            "within-condition ICC(1) on the binary co-primary (no censoring bias), nested df "
            "(k - c) over the conditions whose outcome varies"
        ),
        "exclusion_inflation": infl,
        **chain,
        "rho_ci95": ci,
        "pilot_sequences": b.get("pilot_sequences"),
        "pilot_sequences_with_outcome_variance": b.get("pilot_sequences_with_outcome_variance"),
        "pilot_sequences_used_for_rho": b.get("pilot_sequences_used_for_rho"),
        "locked_sequences_per_condition": locked,
        "locked_source": b.get("locked_source"),
        "locked_matches_plan": locked == chain["sequences_per_condition_with_15pct_exclusion"],
        "formula_correction_only": {
            **formula_only,
            "note": (
                "The corrected binary requirement at the planning rho as it was computed "
                "(.3143: k - 1 df, constant condition counted). Isolates the formula "
                "correction; the rho correction is not applied."
            ),
        },
        "as_planned_uncorrected": {
            **planned,
            "locked_matches_as_planned": (
                locked == planned["sequences_per_condition_with_15pct_exclusion"]
            ),
            "status": (
                "SUPERSEDED -- the chain N was actually locked to, reproduced with the one-sample "
                "binary formula and the miscomputed planning rho; do not cite as a requirement"
            ),
        },
    }
    if ci:
        lo = _sequences_from_rho(ci[0], binding, m, infl)
        hi = _sequences_from_rho(ci[1], binding, m, infl)
        out["requirement_over_rho_ci95"] = {
            "sequences_per_condition_with_15pct_exclusion": [
                lo["sequences_per_condition_with_15pct_exclusion"],
                hi["sequences_per_condition_with_15pct_exclusion"],
            ],
            "design_effect": [lo["design_effect"], hi["design_effect"]],
            "note": (
                "The corrected requirement at the two ends of the pilot rho's 95% interval: the "
                "pilot did not constrain rho, so it did not constrain the requirement either."
            ),
        }
    rng = out.get("requirement_over_rho_ci95", {}).get(
        "sequences_per_condition_with_15pct_exclusion"
    )
    out["note"] = (
        "A-PRIORI, CORRECTED. Pilot rho = "
        f"{chain['within_sequence_icc_rho']} (within-condition ICC(1) on {b['outcome_for_rho']}, "
        f"nested df; 95% CI {ci}; from {out['pilot_sequences_used_for_rho']} of the "
        f"{out['pilot_sequences']} pilot sequences, those in the conditions whose outcome "
        f"varies; {out['pilot_sequences_with_outcome_variance']} sequence varies at all), "
        "design effect = "
        f"{chain['design_effect']}. With the corrected binary requirement ({binding} per "
        f"condition) this requires {chain['sequences_per_condition']} sequences per condition, "
        f"inflated 15% for exclusions -> {chain['sequences_per_condition_with_15pct_exclusion']} "
        f"sequences ({chain['trials_per_condition_with_inflation']} trials) per condition"
        + (f"; over the rho interval, {rng[0]} to {rng[1]} sequences" if rng else "")
        + ". Holding the planning rho as computed (.3143) the formula correction alone gives "
        f"{formula_only['sequences_per_condition_with_15pct_exclusion']} sequences. "
        + _locked_sentence(locked, chain, planned)
        + " power.design_effect_adjustment recomputes rho from the confirmatory data and is POST "
        "HOC; it must never be cited as the sample-size justification."
    )
    return out


def _locked_sentence(locked: int | None, chain: dict, planned: dict) -> str:
    """How the run's locked N relates to the corrected and the archived chains."""

    corrected = chain["sequences_per_condition_with_15pct_exclusion"]
    if locked is not None and locked == corrected:
        return (
            f"N was set to {locked} sequences per condition, the corrected a-priori requirement "
            "itself (the re-run's N; docs/rerun_runbook.md §1, C243), so the locked N MEETS the "
            "corrected requirement under the study's own planning effect size."
        )
    if locked is not None and locked == planned["sequences_per_condition_with_15pct_exclusion"]:
        return (
            f"N was locked to {locked} sequences from the uncorrected arithmetic "
            f"(as_planned_uncorrected: {planned['binding_per_condition_unadjusted']} -> "
            f"{planned['adjusted_trials_per_condition']} -> {planned['sequences_per_condition']} "
            f"-> {planned['sequences_per_condition_with_15pct_exclusion']}), so the locked N is "
            + ("BELOW" if locked < corrected else "NOT BELOW")
            + " the corrected a-priori requirement under the study's own planning effect size."
        )
    return (
        f"The run's N is {locked} sequences per condition, which is "
        + ("BELOW" if locked is not None and locked < corrected else "NOT BELOW")
        + f" the corrected a-priori requirement ({corrected}) under the study's own planning "
        "effect size."
    )


def run_design_basis(design_basis: dict | None = None) -> dict:
    """``PILOT_DESIGN_BASIS`` with the run's own N as the locked N.

    ``design_basis`` is what ``cli.py analyze`` reads from the run's ``_run_manifest.json``
    (``sequences_per_condition``, ``source``). Without it the September run's locked N (33)
    stays, so archived reports reproduce unchanged.
    """

    b = dict(PILOT_DESIGN_BASIS)
    n = (design_basis or {}).get("sequences_per_condition")
    if n is not None:
        b["locked_sequences_per_condition"] = int(n)
        b["locked_source"] = (
            f"{(design_basis or {}).get('source') or 'the run manifest'} -> "
            "config.sequences_per_condition"
        )
    return b


def implied_power(n_eff_per_condition: float, alpha: float, w: float | None = None) -> float:
    """Power of the 1-df chi-square test of the planning contrast, two groups of ``n_eff``."""

    w = two_by_two_w(PLANNING_P1, PLANNING_ODDS_RATIO) if w is None else float(w)
    return float(
        GofChisquarePower().power(
            effect_size=w, nobs=2.0 * float(n_eff_per_condition), alpha=alpha, n_bins=2
        )
    )


_PLANNING_CHECK_NOTE = (
    "Implied power at the run's design N, from the planning effect size (OR = 2.0, .50 vs .667; "
    "w = .169, 1-df chi-square, two-sided) -- never an observed effect, so this is not observed "
    "power. effective_n_per_condition = sequences x m / DE with DE = 1 + (m - 1) rho; each test "
    "compares two groups of that size, as the a-priori analysis frames the contrast, which is "
    "conservative for the pooled MTD-vs-static contrast (its MTD arm pools five conditions). "
    "power_primary_contrast is at alpha = .05 (the planned primary contrast, uncorrected); "
    "power_each_bonferroni_pairwise is each of the 10 pairwise comparisons among the five MTD "
    "techniques at alpha = .05/10. The July pilot's rho is the one N was set on; the October "
    "pilot's real-clock rho, estimated far more precisely (24 sequences), is the planning check, "
    "shown at its point value and at both ends of its 95% interval. sequences_required_* is "
    "the corrected requirement at that rho, with the 15% exclusion buffer."
)


def rerun_planning_check(design_basis: dict | None = None, *, n_mtd_arms: int = 5) -> dict:
    """The re-run's planning check: implied power at its N under the July and October rhos."""

    b = {**RERUN_DESIGN_BASIS, **{k: v for k, v in (design_basis or {}).items() if v is not None}}
    seqs = int(b["sequences_per_condition"])
    m = int(
        b.get("attempts_per_sequence_m") or b.get("attempts_per_sequence") or ATTEMPTS_PER_SEQUENCE
    )
    infl = PILOT_DESIGN_BASIS["exclusion_inflation"]
    w = two_by_two_w(PLANNING_P1, PLANNING_ODDS_RATIO)
    pair = bonferroni_pairwise_requirement(n_mtd_arms)
    n_pairs = int(pair["n_pairwise_contrasts"])
    alpha_pair = 0.05 / n_pairs
    req_primary = a_priori()["binding_per_condition"]
    req_pair = int(pair["binding_per_condition"])
    oct_ci = OCTOBER_PILOT_RHO["rho_ci95"]

    def scenario(rho: float, role: str) -> dict:
        de = design_effect(rho, m)
        n_eff = seqs * m / de
        return {
            "role": role,
            "rho": round(float(rho), 4),
            "design_effect": round(float(de), 3),
            "effective_n_per_condition": round(float(n_eff), 1),
            "power_primary_contrast": round(implied_power(n_eff, 0.05, w), 3),
            "power_each_bonferroni_pairwise": round(implied_power(n_eff, alpha_pair, w), 3),
            "effective_n_over_primary_requirement": round(n_eff / req_primary, 2),
            "effective_n_over_pairwise_requirement": round(n_eff / req_pair, 2),
            "sequences_required_primary": _sequences_from_rho(rho, req_primary, m, infl)[
                "sequences_per_condition_with_15pct_exclusion"
            ],
            "sequences_required_pairwise": _sequences_from_rho(rho, req_pair, m, infl)[
                "sequences_per_condition_with_15pct_exclusion"
            ],
        }

    return {
        "status": "a_priori_planning_check",
        "sequences_per_condition": seqs,
        "sequences_source": b.get("source"),
        "attempts_per_sequence_m": m,
        "effect_size_w": round(w, 4),
        "requirement_per_condition_primary": int(req_primary),
        "requirement_per_condition_pairwise": req_pair,
        "n_pairwise_comparisons": n_pairs,
        "alpha_primary": 0.05,
        "alpha_per_pairwise_comparison": round(alpha_pair, 6),
        "scenarios": {
            "july_pilot": scenario(
                PILOT_DESIGN_BASIS["within_sequence_icc_rho"],
                "the corrected July pilot rho, on which N was set",
            ),
            "october_pilot": scenario(
                OCTOBER_PILOT_RHO["rho"], "PLANNING CHECK: the October pilot's real-clock rho"
            ),
            "october_pilot_ci_low": scenario(oct_ci[0], "lower end of its 95% interval"),
            "october_pilot_ci_high": scenario(oct_ci[1], "upper end of its 95% interval"),
        },
        "october_pilot_rho": dict(OCTOBER_PILOT_RHO),
        "note": _PLANNING_CHECK_NOTE,
    }


def as_executed(
    df: pd.DataFrame,
    m: int = ATTEMPTS_PER_SEQUENCE,
    n_excluded: int | None = None,
) -> dict:
    """What actually ran, from the analyzed data (not a plan, not a recomputation)."""

    if "condition" not in df.columns or not len(df):
        return {"status": "as_executed", "note": "no analyzed trials"}
    seq = (
        df.groupby("condition", observed=True)["sequence_id"].nunique().to_dict()
        if "sequence_id" in df.columns
        else {}
    )
    trials = df["condition"].value_counts().to_dict()
    out = {
        "status": "as_executed",
        "n_conditions": int(df["condition"].nunique()),
        "sequences_per_condition": {str(k): int(v) for k, v in sorted(seq.items())},
        "trials_per_condition_analyzed": {str(k): int(v) for k, v in sorted(trials.items())},
        "n_trials_analyzed": int(len(df)),
        "min_sequences_per_condition": int(min(seq.values())) if seq else None,
        "balanced": (len(set(seq.values())) == 1) if seq else None,
        "attempts_per_sequence_assumed_for_design_effect": int(m),
    }
    if "sequence_id" in df.columns:
        per_seq = df.groupby("sequence_id", observed=True).size()
        out["attempts_per_sequence_realized_median"] = int(per_seq.median())
        out["attempts_per_sequence_realized_range"] = [int(per_seq.min()), int(per_seq.max())]
        out["attempts_per_sequence_matches_design_effect_assumption"] = bool(
            int(per_seq.min()) == int(m) and int(per_seq.max()) == int(m)
        )
    if n_excluded is not None:
        total = int(len(df)) + int(n_excluded)
        out["n_excluded"] = int(n_excluded)
        out["exclusion_rate"] = round(int(n_excluded) / total, 4) if total else None
        out["planned_exclusion_inflation"] = PILOT_DESIGN_BASIS["exclusion_inflation"]
    return out


def bonferroni_pairwise_requirement(
    n_arms: int,
    effect_size_d: float = 0.5,
    odds_ratio: float = 2.0,
    p1: float = 0.5,
    alpha: float = 0.05,
) -> dict:
    """Per-arm n required for all pairwise contrasts among ``n_arms`` under Bonferroni.

    Uses the SAME planning effect sizes as ``a_priori`` (d = 0.50, OR = 2.0, power = .80);
    only the per-contrast alpha changes. This is an a-priori-style calculation at a
    corrected alpha -- it is NOT observed power and uses no observed effect size.
    """

    if n_arms < 2:
        return {"note": "pairwise requirement needs >= 2 arms"}
    n_pairs = n_arms * (n_arms - 1) // 2
    alpha_c = alpha / n_pairs
    z_alpha = float(norm.ppf(1 - alpha_c / 2))
    n_t = math.ceil(
        TTestIndPower().solve_power(
            effect_size=effect_size_d, alpha=alpha_c, power=0.80, alternative="two-sided"
        )
    )
    n_p = _two_proportion_n(p1, odds_ratio, z_alpha=z_alpha)
    n_chi = math.ceil(_chi_square_total(two_by_two_w(p1, odds_ratio), alpha=alpha_c) / 2)
    candidates = {"continuous_t": n_t, "binary_two_proportion": n_p, "binary_chi_square": n_chi}
    return {
        "n_arms": int(n_arms),
        "n_pairwise_contrasts": int(n_pairs),
        "alpha_per_contrast": round(alpha_c, 6),
        "per_condition_by_test": candidates,
        "binding_per_condition": int(max(v for v in candidates.values() if v is not None)),
        "basis": (
            "same planning effect sizes as power.a_priori (d = 0.50 / OR = 2.0, power = .80) at "
            f"a Bonferroni per-contrast alpha of {alpha}/{n_pairs}. A-priori-style calculation "
            "at a corrected alpha; not observed power."
        ),
    }


def _adequacy_sentence(
    out: dict, de_a: float, trials: int, req: int, req_c, corrected: dict
) -> str:
    """Factual adequacy prose derived from the ratios.

    Never asserts a verdict the numbers do not support: each clause reads its own ratio.
    """

    eff = out["effective_n_per_condition_at_a_priori_de"]
    r_u = out["effective_n_over_uncorrected_requirement_at_a_priori_de"]
    parts = [
        f"Holding the A-PRIORI design effect ({de_a}) fixed, the {trials} analyzed trials per "
        f"condition give {eff} effective trials per condition."
    ]
    verdict_u = "MEETS" if r_u is not None and r_u >= 1.0 else "FALLS SHORT OF"
    parts.append(
        f"That {verdict_u} the {req} required by the planned primary MTD-vs-static contrast "
        "(binary co-primary, OR = 2.0, .50 vs .667; alpha = .05 two-sided, UNCORRECTED for "
        f"multiplicity; power = .80) ({r_u}x)."
    )
    n_pairs = corrected.get("n_pairwise_contrasts") or 0
    r_c = out["effective_n_over_corrected_requirement_at_a_priori_de"]
    if req_c and n_pairs >= 2:
        verdict_c = "MEETS" if r_c is not None and r_c >= 1.0 else "FALLS SHORT OF"
        parts.append(
            f"It {verdict_c} the {req_c} that all {n_pairs} pairwise contrasts among the "
            f"{corrected.get('n_arms')} MTD arms would require under Bonferroni at the same "
            f"effect size ({r_c}x)."
        )
    return " ".join(parts)


def power_adequacy(a_priori_de: dict, realized_de: dict, executed: dict) -> dict:
    """Post-hoc adequacy, with the realized-rho claim gated on outcome degeneracy.

    The headline statement is deliberately the one that does NOT use the realized rho:
    holding the a-priori design effect fixed, does the locked N clear the requirement?
    The realized design effect is reported, but when the primary outcome is constant
    within one or more conditions its ~1.0 value reflects degeneracy rather than weak
    clustering, and no over-powering claim is made.
    """

    if not isinstance(realized_de, dict) or "design_effect" not in realized_de:
        return {"note": "realized design effect unavailable"}
    tpc = executed.get("trials_per_condition_analyzed") or {}
    de_r = float(realized_de["design_effect"])
    if not tpc or de_r <= 0:
        return {"note": "no analyzed trials per condition"}
    de_a = float(a_priori_de["design_effect"])
    req = int(a_priori_de["binding_per_condition_unadjusted"])
    realized_trials = int(min(tpc.values()))
    eff_r = realized_trials / de_r
    eff_a = realized_trials / de_a if de_a > 0 else None

    by_cond = realized_de.get("rho_by_condition") or {}
    degenerate = sorted(
        c for c, v in by_cond.items() if isinstance(v, dict) and v.get("outcome_degenerate")
    )
    n_cond = len([v for v in by_cond.values() if isinstance(v, dict)])
    nonzero = sorted(
        c
        for c, v in by_cond.items()
        if isinstance(v, dict) and isinstance(v.get("rho"), (int, float)) and v["rho"] > 0
    )
    interpretable = bool(by_cond) and not degenerate

    n_arms = int(executed.get("n_conditions") or 0) - 1
    # >= 3 arms means >= 3 pairwise contrasts, where the Bonferroni correction is real.
    # With 2 arms the single contrast IS the uncorrected benchmark already reported as
    # required_trials_per_condition_a_priori_uncorrected, so a block labelled "corrected"
    # would carry an uncorrected number.
    corrected = bonferroni_pairwise_requirement(n_arms) if n_arms >= 3 else {}
    req_c = corrected.get("binding_per_condition")

    planned = a_priori_de.get("as_planned_uncorrected") or {}
    req_planned = planned.get("binding_per_condition_unadjusted")
    de_planned = planned.get("design_effect")
    eff_planned = realized_trials / de_planned if de_planned else None
    formula_only = a_priori_de.get("formula_correction_only") or {}
    de_formula = formula_only.get("design_effect")
    eff_formula = realized_trials / de_formula if de_formula else None
    out = {
        "required_trials_per_condition_a_priori_uncorrected": req,
        "requirement_basis": (
            "binding two-group requirement, UNCORRECTED for multiplicity: the largest of the "
            "per-test requirements in power.a_priori (binary two-proportion and chi-square at "
            "OR = 2.0, t at d = 0.50; alpha = .05; power = .80). It is the benchmark for the "
            "planned primary MTD-vs-static contrast, which is evaluated at alpha = .05 without "
            "correction, and a LOWER BOUND for the RQ2 technique-level pairwise contrasts, which "
            "are Bonferroni/FDR-adjusted. See multiplicity_adjusted_requirement."
        ),
        "multiplicity_adjusted_requirement": corrected,
        "required_trials_per_condition_as_planned_uncorrected": req_planned,
        "as_planned_design_effect": de_planned,
        "effective_n_per_condition_as_planned": (round(eff_planned, 1) if eff_planned else None),
        "effective_n_over_as_planned_requirement_at_a_priori_de": (
            round(eff_planned / req_planned, 2) if eff_planned and req_planned else None
        ),
        "as_planned_note": (
            "SUPERSEDED. The requirement and design effect the study was sized on (one-sample "
            "binary formula; miscomputed planning rho); the ratio is kept only so the previously "
            "reported adequacy figure can be traced. Do not cite it."
        ),
        "formula_correction_only_design_effect": de_formula,
        "effective_n_over_requirement_formula_correction_only": (
            round(eff_formula / req, 2) if eff_formula and req else None
        ),
        "realized_trials_per_condition_min": realized_trials,
        "a_priori_design_effect": de_a,
        "realized_design_effect": round(de_r, 3),
        "design_effect_ratio_a_priori_over_realized": round(de_a / de_r, 3),
        "effective_n_per_condition_at_a_priori_de": round(eff_a, 1) if eff_a else None,
        "effective_n_over_uncorrected_requirement_at_a_priori_de": (
            round(eff_a / req, 2) if eff_a and req else None
        ),
        "effective_n_over_corrected_requirement_at_a_priori_de": (
            round(eff_a / req_c, 2) if eff_a and req_c else None
        ),
        "effective_n_per_condition_at_realized_de": round(eff_r, 1),
        "effective_n_over_uncorrected_requirement_at_realized_de": (
            round(eff_r / req, 2) if req else None
        ),
        "conditions_with_degenerate_primary_outcome": degenerate,
        "n_conditions_with_degenerate_primary_outcome": len(degenerate),
        "n_conditions_evaluated_for_rho": n_cond,
        "conditions_with_nonzero_rho": nonzero,
        "realized_rho_interpretable": interpretable,
        "conservatively_overpowered": bool(interpretable and de_r < de_a and eff_r >= req),
    }
    headline = _adequacy_sentence(out, de_a, realized_trials, req, req_c, corrected)
    if degenerate:
        out["note"] = (
            "NO OVER-POWERING CLAIM IS MADE. The primary outcome is CONSTANT within "
            f"{len(degenerate)} of {n_cond} conditions ({', '.join(degenerate)}): the "
            "within-sequence ICC is undefined there (zero by construction, not weak clustering), "
            "so the realized design effect "
            f"({out['realized_design_effect']}) describes only the conditions whose outcome "
            "varies"
            + (
                f" (non-zero rho only in {', '.join(nonzero)}). "
                if nonzero
                else " (none has a non-zero rho). "
            )
            + "effective_n_per_condition_at_realized_de "
            f"({out['effective_n_per_condition_at_realized_de']}) is therefore an UPPER BOUND on "
            "precision (a design effect cannot be below 1), not an adequacy certificate. "
            + headline
            + " These are post-hoc adequacy statements computed from "
            "the PLANNING effect size, NOT retrospective (observed-power) justifications of the "
            "sample size."
        )
    else:
        out["note"] = (
            f"The a-priori design effect ({de_a}) was "
            f"{out['design_effect_ratio_a_priori_over_realized']}x the design effect realized in "
            f"the confirmatory data ({out['realized_design_effect']}), and the realized rho is "
            "interpretable (no condition has a constant primary outcome), so planning on the "
            "pilot rho was conservative. " + headline + " These are post-hoc adequacy statements "
            "computed from the PLANNING effect size, NOT retrospective (observed-power) "
            "justifications of the sample size."
        )
    return out


def pilot_variance(
    df: pd.DataFrame,
    outcome: str = "time_to_first_valid_exploit_s",
    censor_flag: str = "tfve_censored",
) -> dict:
    """Pilot-based variance + observed effect (static vs. MTD) for one CONTINUOUS outcome.

    Defaults to ``time_to_first_valid_exploit_s`` (censor flag ``tfve_censored``). When a
    censor flag column is present, **right-censored rows are dropped** so the SD / Cohen's d
    are computed on genuinely observed event times, never on the administrative-limit
    constant (which would fabricate an SD out of the censoring floor). The survival
    co-primary's condition effect is reported as the Cox hazard ratio in RQ1, not here; this
    observed-time effect describes only *when* the event occurred among the trials where it
    did. Pass ``censor_flag=None`` (or a name not in the frame) to use all rows.
    """

    if "defense" not in df.columns or outcome not in df.columns:
        return {"note": "missing columns"}
    d = df
    dropped = 0
    if censor_flag and censor_flag in df.columns:
        observed = pd.to_numeric(df[censor_flag], errors="coerce").fillna(0) == 0
        dropped = int((~observed).sum())
        d = df.loc[observed]
    a = pd.to_numeric(d.loc[d["defense"] == "static", outcome], errors="coerce").dropna()
    b = pd.to_numeric(d.loc[d["defense"] == "mtd", outcome], errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return {
            "note": "insufficient observed (uncensored) pilot data",
            "n_censored_dropped": dropped,
        }
    dd = cohen_d(a.to_numpy(), b.to_numpy())
    out = {
        "outcome": outcome,
        "observed_only": bool(censor_flag and censor_flag in df.columns),
        "n_censored_dropped": dropped,
        "sd_static": float(a.std(ddof=1)),
        "sd_mtd": float(b.std(ddof=1)),
        "observed_cohen_d": dd,
    }
    if dd and not math.isnan(dd) and abs(dd) > 1e-6:
        n1 = TTestIndPower().solve_power(
            effect_size=abs(dd), alpha=0.05, power=0.80, alternative="two-sided"
        )
        out["n_per_condition_for_observed_d"] = math.ceil(n1) if np.isfinite(n1) else None
    return out
