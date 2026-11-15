"""RQ2 (H02/Ha2): which external MTD techniques are most effective?

Across the five MTD technique categories (static baseline excluded), modeling the
clustering of trials within sequences:

- Continuous outcomes (time_to_enumeration, recon_accuracy, attack_path_churn):
  **linear mixed model** (technique fixed + random intercept for sequence) with
  estimated marginal means and pairwise contrasts (Bonferroni-adjusted; Tukey
  intent), and marginal R2 for technique. One-way ANOVA + Games-Howell / Kruskal-
  Wallis on per-sequence aggregates are the labeled **sensitivity** analyses.
- Binary outcomes (exploit_success, exploitation_success): logit **GLMM / GEE**
  across techniques (odds ratios vs. reference) with pairwise technique contrasts
  under FDR control.

RE-RUN ANALYSIS UPDATE (2026-10-28; ``docs/rerun_runbook.md`` §6): each binary outcome also
carries ``pairwise_moving_techniques``, the six pairwise risk differences among the four
conditions that move the genuine services (per-pair 95% and Bonferroni-simultaneous
intervals, Holm p), so a small difference between them can be stated as small.

THE H02 DECISION RULE (decided 2026-10-29; runbook §1). H02 is evaluated separately for each
outcome at alpha = .05, with no global decision across outcomes (the same rule as H01).
``h02_tests`` holds the tests for the primary outcomes, among all five technique categories:

- exploitation success: the 10 pairwise risk differences (sequence cluster bootstrap, Holm
  across the 10 pairs; ``separation.pairwise_rate_differences``);
- time-to-compromise: the 10 pairwise differences in restricted mean time to compromise over
  [0, 300 s] (the same bootstrap, Holm across the 10; ``clustered.rmst_pairwise_bootstrap``);
- exploit success: tested like exploitation success if it varies, otherwise reported as
  constant (the re-run's strict surface check makes it 1 in every trial).

Reconnaissance accuracy and attack-path churn are decided by their technique effect after
Benjamini-Hochberg adjustment across the RQ2 secondary outcomes (``fdr_secondary.rq2``, as
before). ``h02_decision`` (added by ``pipeline.run_all`` once the FDR family exists) applies the
rule outcome by outcome.
"""

from __future__ import annotations

import math

import pandas as pd
from scipy import stats

from analysis import clustered
from analysis.dataset import STATIC
from analysis.degeneracy import lmm_degeneracy
from analysis.effects import eta_squared

# time_to_compromise_s is now a censored survival outcome (RQ1 Cox), so the continuous
# technique-level LMM uses the enumeration diagnostic instead of a censored time.
CONTINUOUS = ["time_to_enumeration_s", "recon_accuracy_f1", "attack_path_churn"]
BINARY = ["exploit_success", "exploitation_success"]

#: The five MTD technique categories H02 compares (the static baseline is not one).
H02_TECHNIQUES = (
    "combined_mtd_deception",
    "deceptive_presentation",
    "endpoint_mutation",
    "network_shuffling",
    "port_rotation",
)
#: The date the H02 decision rule was set (decision A1 of 2026-10-29; not pre-registered).
H02_RULE_DATE = "2026-10-29"
H02_ALPHA = 0.05
#: The restricted-mean horizon for time-to-compromise (RQ1's time endpoint, 2026-10-28).
H02_HORIZON_S = 300.0
H02_N_PAIRS = 10


def _finite(x) -> bool:
    """True for a real, finite int/float (bools and None excluded)."""

    if isinstance(x, bool) or x is None:
        return False
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _continuous(mtd: pd.DataFrame, outcome: str) -> dict:
    """Fit, diagnose, and designate the one citable test for an RQ2 continuous outcome.

    The decision tree, in order (see ``analysis.degeneracy`` for both screens):

    1. No variance anywhere -> ``clustered.lmm_continuous`` returns a skip payload;
       no model is fit, no test is performed, and no per-sequence sensitivity
       analysis is computed either (on a globally constant outcome the ANOVA and
       Kruskal-Wallis are both NaN, so computing them would add noise, not evidence).
       ``reported_result.source`` is ``"none"``.
    2. Variance exists but the LMM is numerically degenerate -> the fit is retained
       verbatim for audit, ``lmm_reliable`` is false, and the pre-specified
       per-sequence test is PROMOTED to ``reported_result``.
    3. Otherwise -> the LMM's Wald omnibus is ``reported_result``.
    """

    res = clustered.lmm_continuous(mtd, outcome, "C(technique)")
    model = res.pop("model", None)  # absent when the LMM was skipped or raised
    if res.get("skipped"):
        # Fix-2 precedence: a globally constant outcome is SKIPPED, never promoted.
        # The router below is bypassed, so the citable-result contract is honoured here
        # explicitly -- every RQ2 continuous outcome carries a `reported_result`.
        res["lmm_reliable"] = False
        res["reported_result"] = {
            "source": "none",
            "method": None,
            "statistic": None,
            "p": None,
            "cite_this": False,
            "degeneracy_reason": res.get("skipped_reason"),
            "skipped_reason": res.get("skipped_reason"),
            "note": _SKIP_NOTE,
        }
        return res
    if model is not None:
        techniques = sorted(mtd["technique"].dropna().unique().tolist())
        try:
            emmeans, omnibus, pairs = clustered.emmeans_pairwise(model, "technique", techniques)
            res["emmeans"] = emmeans
            res["omnibus_wald"] = omnibus
            res["pairwise_contrasts"] = pairs
            res["pairwise_significant"] = [p for p in pairs if p.get("significant")]
        except Exception as exc:  # noqa: BLE001
            res["emmeans_note"] = f"EMMeans/contrasts unavailable: {exc}"
    # Unconditional: the per-sequence analysis is what a degenerate LMM is promoted to,
    # so it must exist even on the paths where the LMM raised.
    res["sensitivity"] = _sensitivity_continuous(mtd, outcome)
    values = mtd[outcome] if outcome in mtd.columns else []
    groups = mtd["technique"] if "technique" in mtd.columns else None
    res["lmm_diagnostics"] = lmm_degeneracy(res, values, groups=groups)
    res["lmm_reliable"] = not res["lmm_diagnostics"]["degenerate"]
    res["reported_result"] = _reported_result(res)
    return res


_LMM_EFFECT_NOTE = (
    "marginal R2 of the fixed effects at the trial level; not comparable with the "
    "per-sequence eta-squared reported when a result is promoted."
)
_SENS_EFFECT_NOTE = (
    "eta-squared computed on per-sequence means, not on trial-level observations; it is "
    "not comparable with the LMM branch's marginal_r2_fixed, which is trial-level."
)
_SKIP_NOTE = (
    "The outcome has no variance anywhere in the analysis subset, so there is no "
    "between-technique contrast to estimate and NO test is reported. Any p-value "
    "elsewhere in this block is a numerical artefact of a rank-deficient design matrix "
    "and must not be cited. The outcome is recorded under fdr_secondary.rq2.skipped and "
    "remains in the Benjamini-Hochberg denominator, so the correction applied to its "
    "siblings does not loosen because this hypothesis proved untestable."
)
_NOT_ESTIMABLE_NOTE = (
    "Neither the clustered linear mixed model nor the per-sequence sensitivity analysis "
    "is estimable for this outcome, so NO between-technique test is reported. Any "
    "p-value elsewhere in this block is a numerical artefact and must not be cited; the "
    "outcome is recorded under fdr_secondary.rq2.skipped and remains in the "
    "Benjamini-Hochberg denominator."
)
_PROMOTED_NOTE = (
    "PROMOTED sensitivity result. The clustered linear mixed model for this outcome is "
    "numerically degenerate (see lmm_diagnostics): the outcome is exactly constant "
    "within at least one technique, which leaves the sequence random intercept "
    "unidentified in that stratum, puts the REML solution on the boundary sigma_b^2 = 0, "
    "and makes the fixed-effect covariance -- hence the EMMean standard errors, the "
    "pairwise contrasts and the Wald omnibus -- untrustworthy. Those blocks are retained "
    "verbatim for audit but MUST NOT be cited. The reported test collapses each sequence "
    "to its mean (one independent observation per cluster, balanced across techniques) "
    "and uses the rank-based Kruskal-Wallis omnibus, which -- unlike the pooled-variance "
    "F -- stays well defined when some techniques have zero within-group variance. The "
    "parametric ANOVA is reported alongside as corroboration only."
)


def _reported_result(res: dict) -> dict:
    """Designate the one citable between-technique test for this outcome.

    The routing predicate reads ``lmm_diagnostics["reasons"]`` and the *finiteness*
    of the promoted test's statistic. It never reads the magnitude of any p-value,
    so no estimator here is selected on the basis of the result it produces.
    """

    diag = res.get("lmm_diagnostics") or {}
    sens = res.get("sensitivity") or {}
    reasons = list(diag.get("reasons") or [])
    reason = "; ".join(reasons)

    if not diag.get("degenerate"):  # ---- healthy: the LMM is the result
        omni = res.get("omnibus_wald") or {}
        return {
            "source": "lmm",
            "method": "lmm_omnibus_wald",
            "test": (
                "Wald omnibus for C(technique) from the random-intercept linear mixed "
                "model (trial-level, sequence random intercept)"
            ),
            "unit_of_analysis": "trial (sequence modeled as a random intercept)",
            "statistic": omni.get("wald_stat") if _finite(omni.get("wald_stat")) else None,
            "p": float(omni["p"]) if _finite(omni.get("p")) else None,
            "effect_size": {"marginal_r2_fixed": res.get("marginal_r2_fixed")},
            "effect_size_note": _LMM_EFFECT_NOTE,
            "pairwise": res.get("pairwise_contrasts"),
            "pairwise_significant_only": res.get("pairwise_significant"),
            "cite_this": True,
        }

    # ---- a globally constant outcome is SKIPPED, never promoted. Promoting it would
    # emit a NaN: the per-sequence ANOVA and Kruskal-Wallis are both NaN when every
    # group is the same constant.
    if "outcome_constant" in reasons:
        return {
            "source": "none",
            "method": None,
            "statistic": None,
            "p": None,
            "cite_this": False,
            "degeneracy_reason": reason,
            "skipped_reason": "zero_variance",
            "note": _SKIP_NOTE,
        }

    kw = sens.get("kruskal_wallis") or {}
    p, stat = kw.get("p"), kw.get("H")
    if not _finite(p):  # ---- nothing is estimable at the cluster level either
        return {
            "source": "none",
            "method": None,
            "statistic": None,
            "p": None,
            "cite_this": False,
            "degeneracy_reason": reason,
            "skipped_reason": "not_estimable",
            "note": _NOT_ESTIMABLE_NOTE,
        }

    anova = sens.get("anova") or {}
    return {  # ---- promoted per-sequence (independent-cluster) test
        "source": "sensitivity",
        "method": "kruskal_wallis_per_sequence",
        "test": "Kruskal-Wallis H across techniques on per-sequence means",
        "unit_of_analysis": "sequence (per-sequence mean; one observation per cluster)",
        "n_sequences_by_technique": sens.get("n_sequences_by_technique", {}),
        "k_groups": sens.get("k_groups"),
        "statistic": float(stat) if _finite(stat) else None,
        "p": float(p),
        "effect_size": {"eta_squared_per_sequence_means": sens.get("eta_squared")},
        "effect_size_note": _SENS_EFFECT_NOTE,
        "corroboration": {"anova_F": anova.get("F"), "anova_p": anova.get("p")},
        "pairwise": sens.get("games_howell"),
        "pairwise_significant_only": sens.get("games_howell_significant"),
        "cite_this": True,
        "degeneracy_reason": reason,
        "note": _PROMOTED_NOTE,
    }


def reported_p(outcome_result: dict) -> tuple[float | None, str]:
    """(p, provenance-label) of the citable test for one RQ2 continuous outcome."""

    rr = (outcome_result or {}).get("reported_result") or {}
    source = str(rr.get("source") or "unavailable")
    method = rr.get("method")
    label = f"{source}:{method}" if method else source
    p = rr.get("p")
    return (float(p) if _finite(p) else None), label


def _sensitivity_continuous(mtd: pd.DataFrame, outcome: str) -> dict:
    """ANOVA + Games-Howell / Kruskal-Wallis on per-sequence aggregates (labeled).

    This is the pre-specified RQ2 sensitivity analysis, and it is also the test a
    degenerate LMM is promoted to. Each sequence contributes one value (its mean over
    its attempts), so the observations are independent by design -- the sequence is
    the unit of randomisation.
    """

    out: dict = {"label": "sensitivity (per-sequence aggregates)"}
    if outcome not in mtd.columns or "sequence_id" not in mtd.columns:
        out["note"] = "outcome or cluster column absent"
        return out
    per = (
        mtd.dropna(subset=[outcome])
        .groupby(["sequence_id", "technique"], observed=True)[outcome]
        .mean()
        .reset_index()
    )
    # Techniques contributing a single sequence cannot enter the omnibus; record exactly
    # which ones DID so reported_result cannot over-state the arms its p covers.
    kept = [(t, g) for t, g in per.groupby("technique", observed=True) if len(g) > 1]
    groups = [g[outcome].to_numpy() for _, g in kept]
    out["n_sequences_by_technique"] = {str(t): int(len(g)) for t, g in kept}
    out["k_groups"] = len(kept)
    if len(groups) < 2:
        out["note"] = "insufficient sequences"
        return out
    out["eta_squared"] = eta_squared(groups)
    f, p = stats.f_oneway(*groups)
    out["anova"] = {"F": float(f), "p": float(p)}
    h, ph = stats.kruskal(*groups)
    out["kruskal_wallis"] = {"H": float(h), "p": float(ph)}
    try:
        import pingouin as pg

        gh = pg.pairwise_gameshowell(data=per, dv=outcome, between="technique")
        out["games_howell"] = [
            {
                "g1": str(r["A"]),
                "g2": str(r["B"]),
                "diff": float(r["diff"]) if _finite(r["diff"]) else None,
                # p is None (not NaN) when a pair is not estimable -- both arms constant
                # gives Welch t = inf / df = NaN. Keeps the report valid strict JSON.
                "p": float(r["pval"]) if _finite(r["pval"]) else None,
            }
            for _, r in gh.iterrows()
        ]
        out["games_howell_significant"] = [
            c for c in out["games_howell"] if c["p"] is not None and c["p"] < 0.05
        ]
    except Exception as exc:  # noqa: BLE001
        out["games_howell_note"] = f"unavailable: {exc}"
    return out


_SEPARATION_WARNING = (
    "NOT DATA-IDENTIFIED: this contrast involves a technique with the outcome in every trial "
    "or in none, so the odds ratio has no maximum-likelihood value and the VB estimate is set "
    "by the estimator's prior (see separation_robust.vb_prior_sensitivity). Its p-value in "
    "fdr_technique_terms inherits the same problem. Report separation_robust.rates instead."
)


def _binary(mtd: pd.DataFrame, outcome: str) -> dict:
    """Logit GLMM/GEE across techniques (OR vs. reference), FDR over technique terms."""

    from analysis.fdr import benjamini_hochberg
    from analysis.separation import (
        MOVING_TECHNIQUES,
        pairwise_rate_differences,
        separated_terms_for_factor,
        separation_profile,
        separation_robust_block,
    )

    rhs = "C(technique) + attempt_index"
    r = clustered.binary_clustered(mtd, outcome, rhs)
    terms = r.get("terms", {})
    tech_terms = {k: v for k, v in terms.items() if k.startswith("C(technique)")}
    if tech_terms:
        r["fdr_technique_terms"] = benjamini_hochberg({k: v["p"] for k, v in tech_terms.items()})
    if r.get("skipped") or "technique" not in mtd.columns:
        return r
    # Re-run update (runbook §6): the four conditions that move the genuine services,
    # compared pair by pair on the probability scale, so a small difference can be read as
    # small (Bonferroni-simultaneous intervals, Holm p). Descriptive; not an added test of H02.
    r["pairwise_moving_techniques"] = pairwise_rate_differences(
        mtd, outcome, "condition", MOVING_TECHNIQUES
    )
    prof = separation_profile(mtd, outcome, "technique")
    r["separation"] = prof
    sep_terms = separated_terms_for_factor(prof, "technique") if prof.get("separated") else []
    if sep_terms:
        r["separation_robust"] = separation_robust_block(
            mtd, outcome, rhs, "technique", kind="techniques", separated_terms=sep_terms
        )
        for t in sep_terms:
            if isinstance(terms.get(t), dict):
                terms[t] = {**terms[t], "separation_warning": _SEPARATION_WARNING}
    return r


_H02_TESTS_NOTE = (
    "The H02 decision tests (decided 2026-10-29): all 10 pairs among the five technique "
    "categories, per outcome, Holm across the 10 pairs, alpha = .05. Risk differences for the "
    "binary outcomes; for time-to-compromise, differences in restricted mean time to compromise "
    "over [0, 300 s] (positive = the first technique stays uncompromised longer). The six "
    "moving-technique pairs also appear in <outcome>.pairwise_moving_techniques, the descriptive "
    "block of 2026-10-28 (simultaneous over six pairs, no margin, no equivalence claim); that "
    "block resamples only the four moving conditions, so its intervals can differ from these in "
    "the last digits. p_holm adjusts over the pairs present; p_holm_planned_10 over the 10 "
    "planned pairs (an untested pair counting as p = 1), which is the decision's family. They "
    "agree when all five techniques are present. h02_decision applies the rule."
)


def _constant_outcome(vals: pd.Series, outcome: str) -> dict:
    return {
        "evaluated": False,
        "reason": "constant",
        "value": float(vals.iloc[0]) if len(vals) else None,
        "n_trials": int(len(vals)),
        "note": (
            f"{outcome} takes one value in every trial of the five technique categories, so no "
            "difference among them can be tested. Under the re-run's strict surface check every "
            "included scan enumerates a genuine service, which makes exploit success 1 by "
            "design; it is tested like exploitation success whenever it varies."
        ),
    }


def h02_tests(df: pd.DataFrame, *, n_boot: int | None = None, seed: int | None = None) -> dict:
    """The H02 decision tests: the 10 pairs among the five techniques, per primary outcome."""

    n_boot = clustered.BOOT_DEFAULT if n_boot is None else int(n_boot)
    seed = clustered.BOOT_SEED_DEFAULT if seed is None else int(seed)
    from analysis.separation import pairwise_rate_differences

    mtd = df[df["condition"] != STATIC]
    out: dict = {"note": _H02_TESTS_NOTE, "techniques": list(H02_TECHNIQUES)}
    for outcome in BINARY:
        if outcome not in mtd.columns:
            continue
        vals = pd.to_numeric(mtd[outcome], errors="coerce").dropna()
        if vals.nunique() < 2:
            out[outcome] = _constant_outcome(vals, outcome)
            continue
        out[outcome] = {
            "evaluated": True,
            "estimand": "difference in compromise proportion, first - second technique",
            **pairwise_rate_differences(
                mtd, outcome, "condition", H02_TECHNIQUES, n_boot=n_boot, seed=seed
            ),
        }
    if {"time_to_compromise_s", "ttc_censored"} <= set(df.columns):
        # The whole dataset goes in, static included, so the per-technique RMSTs and their
        # intervals are exactly those of rq1.time_to_compromise_s.rmst_300s.by_condition.
        out["time_to_compromise_s"] = {
            "evaluated": True,
            "estimand": (
                "difference in restricted mean time to compromise over [0, 300 s], first - "
                "second technique, seconds"
            ),
            **clustered.rmst_pairwise_bootstrap(
                df,
                "time_to_compromise_s",
                "ttc_censored",
                horizon=H02_HORIZON_S,
                levels=H02_TECHNIQUES,
                n_boot=n_boot,
                seed=seed,
            ),
        }
    # The decision's Holm family is the 10 planned pairs whatever is present: give each pair
    # that adjustment too, so a block with an absent technique cannot be cited at the
    # smaller family's p_holm.
    for outcome in (*BINARY, "time_to_compromise_s"):
        blk = out.get(outcome) or {}
        pairs = blk.get("pairs")
        if not isinstance(pairs, dict) or not pairs:
            continue
        adj, untested = _planned_holm(pairs)
        for k, v in pairs.items():
            v["p_holm_planned_10"] = adj.get(k)
        blk["pairs_untested_of_planned_10"] = untested
    return out


_H02_RULE = (
    "DECISION RULE (decided 2026-10-29; not pre-registered). H02 is evaluated separately for each "
    "outcome at alpha = .05; no global decision across outcomes is made. For exploitation success "
    "and for time-to-compromise (restricted mean at 300 s), H02 is rejected if any of the 10 "
    "pairwise differences among the technique categories has a Holm-adjusted p < .05. For "
    "reconnaissance accuracy and attack-path churn, it is rejected if the technique effect is "
    "significant after Benjamini-Hochberg adjustment across the RQ2 secondary outcomes. Exploit "
    "success is tested in the same way if it varies and is otherwise reported as constant. The "
    "rule was set while the main study was running, after its interim checks and before any "
    "model was fitted to its data."
)
_H02_NOT_REJECTED = "not rejected: no difference among the techniques detected"


def _planned_pairs() -> list[str]:
    import itertools

    return [f"{a} - {b}" for a, b in itertools.combinations(sorted(H02_TECHNIQUES), 2)]


def _planned_holm(pairs: dict) -> tuple[dict[str, float], list[str]]:
    """Holm over the 10 planned pairs, a pair that could not be tested counting as p = 1."""

    from analysis.fdr import holm

    pvals = {k: (pairs.get(k) or {}).get("p") for k in _planned_pairs()}
    untested = sorted(k for k, v in pvals.items() if not _finite(v))
    return holm({k: (float(v) if _finite(v) else 1.0) for k, v in pvals.items()}), untested


def _pairwise_decision(block: dict, basis: str) -> dict:
    """Holm over the 10 planned pairs; a pair that could not be tested counts as p = 1."""

    if not block:
        return {"result": "not evaluated", "reason": "outcome not in the dataset"}
    if not block.get("evaluated"):
        return {
            "result": "not evaluated",
            "reason": block.get("reason", "not tested"),
            "value": block.get("value"),
        }
    pairs = block.get("pairs") or {}
    planned = _planned_pairs()
    adj, untested = _planned_holm(pairs)
    if len(untested) == len(planned):
        return {"result": "not tested", "why": block.get("note", "no pair produced a p")}
    rejected = sorted(k for k, v in adj.items() if v < H02_ALPHA)
    out = {
        "basis": basis,
        "test": (
            "pairwise differences with their sequence cluster bootstrap p, Holm across the 10 "
            "planned pairs, alpha = .05"
        ),
        "n_pairs_planned": len(planned),
        "n_pairs_tested": len(planned) - len(untested),
        "pairs_rejected": rejected,
        "min_p_holm": float(min(adj.values())),
        "result": "rejected" if rejected else _H02_NOT_REJECTED,
    }
    if untested:
        out["untested_pairs"] = untested
        out["untested_note"] = "an untested pair counts as p = 1, so Holm still spans 10 pairs"
    return out


def h02_decision(rq2r: dict, fdr_rq2: dict | None) -> dict:
    """H02 under the rule of 2026-10-29, outcome by outcome (``_H02_RULE``)."""

    tests = (rq2r or {}).get("h02_tests") or {}
    outcomes: dict = {}
    for outcome in ("exploit_success", "exploitation_success", "time_to_compromise_s"):
        outcomes[outcome] = _pairwise_decision(tests.get(outcome), f"rq2.h02_tests.{outcome}")
    fam = fdr_rq2 or {}
    for outcome in ("recon_accuracy_f1", "attack_path_churn"):
        t = (fam.get("tests") or {}).get(outcome)
        sk = (fam.get("skipped") or {}).get(outcome)
        if t:
            outcomes[outcome] = {
                "basis": "fdr_secondary.rq2 (Benjamini-Hochberg across the RQ2 secondary outcomes)",
                "p": t.get("p"),
                "p_adj": t.get("p_adj"),
                "p_source": t.get("p_source"),
                "result": "rejected" if t.get("reject") else _H02_NOT_REJECTED,
            }
        elif sk:
            outcomes[outcome] = {
                "result": "not evaluated",
                "reason": sk.get("reason"),
                "detail": sk.get("detail"),
            }
        else:
            outcomes[outcome] = {"result": "not evaluated", "reason": "outcome not in the dataset"}
    return {
        "rule_date": H02_RULE_DATE,
        "alpha": H02_ALPHA,
        "rule": _H02_RULE,
        "outcomes": outcomes,
        "global_decision": None,
        "note": (
            "No global decision across outcomes is made: report H02 outcome by outcome. A "
            "rejection names the pairs that differ (pairs_rejected); Ha2's 'more effective' "
            "reading comes from the direction of those differences."
        ),
    }


def rq2(df: pd.DataFrame) -> dict:
    """Run the clustered RQ2 analyses across the MTD technique categories."""

    mtd = df[df["condition"] != STATIC].copy()
    results: dict = {}
    for outcome in CONTINUOUS:
        if outcome in mtd.columns:
            results[outcome] = _continuous(mtd, outcome)
    for outcome in BINARY:
        if outcome in mtd.columns:
            results[outcome] = _binary(mtd, outcome)
    if "condition" in df.columns and "sequence_id" in df.columns:
        results["h02_tests"] = h02_tests(df)
    return results
