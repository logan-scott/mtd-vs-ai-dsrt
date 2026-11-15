"""RQ3 (H03/Ha3): how does attacker adaptation across repeated attempts change
defensive outcomes over time?

Mixed-effects models with attempt number (within-subjects) x condition (between)
and a random intercept for trial sequence:
- continuous outcomes (time_to_compromise, recon_accuracy): linear mixed-effects (REML).
- both binary outcomes (exploit_success and co-primary exploitation_success): logit
  **GLMM** fitted by **variational Bayes** (``BinomialBayesMixedGLM.fit_vb``; not maximum
  likelihood) with a GEE fallback.

READING THE ATTEMPT TERM (2026-10-27). The interaction model uses treatment coding, so its
``attempt_index`` coefficient is the attempt slope in the REFERENCE condition (the first
condition in sorted order, ``combined_mtd_deception`` in this study), NOT an average "main
effect"; every other condition's slope is that coefficient times its interaction term.
``attempt_effects`` reports what the reference-coded term was being read as: the average
attempt effect across the conditions where the outcome varies, each condition's own slope,
and the POST HOC sensitivity with the first attempt excluded (average and per condition).
These are logit GLMs with sequence-clustered robust standard errors
(``clustered.glm_cluster_or``) -- the model of the dissertation's post hoc sensitivity, now
archived with every report -- so the intervals respect the eight-attempt clustering.

THE H03 DECISION RULE (decided 2026-10-29; ``docs/rerun_runbook.md`` §1). ``h03_tests`` gives
H03's two clauses one test each, for each outcome: exploitation success (standing in for exploit
success, which is tested the same way only if it varies) and time-to-compromise restricted to
300 s (min(T, 300), exact because every censored trial is censored at the trial window). The
attempt clause is decided by the average attempt effect (one common slope, condition fixed
effects); the interaction clause by one joint Wald test of the attempt x condition terms. Both
are fitted over the conditions where the outcome varies, with standard errors clustered on
sequence (a logit GLM for the binary outcome, a linear model for the restricted time). Every
clustered slope rests on at least 20 informative sequences, sequences in which the outcome
departs from the condition's most common value (``clustered.MIN_INFORMATIVE_SEQUENCES``): the
interaction test leaves out a condition with fewer than 20 of its own, whose own slope would
have a collapsed clustered standard error and make the joint test reject far too often, and the
attempt clause needs 20 across the conditions it pools (set 2026-10-29 after a code review and a
simulation of the design, before any model was fitted to re-run data). Each clause is decided at
alpha = .05 and stated clause by clause, as for H04. Excluding each sequence's first attempt is
a PLANNED sensitivity for the re-run (it was post hoc for the archived runs).
``h03_tests.decision`` applies the rule. ``attempt_effects`` and the drift checks apply the
same 20-sequence rule to their own slopes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _interaction_terms(pvalues: pd.Series) -> dict:
    # attempt x condition interaction terms (works for centered or raw attempt).
    terms = {k: float(v) for k, v in pvalues.items() if ":" in k and "C(condition)" in k}
    return {
        "terms": terms,
        "any_significant": any(v < 0.05 for v in terms.values()) if terms else None,
    }


def _lmm(df: pd.DataFrame, outcome: str) -> dict:
    import warnings

    import statsmodels.formula.api as smf
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    from analysis.degeneracy import zero_variance_skip

    d = df.dropna(subset=[outcome]).copy()
    # rq3._lmm builds its own mixedlm (attempt x condition) instead of routing through
    # clustered.lmm_continuous, so it needs the zero-variance screen independently.
    skip = zero_variance_skip(d, outcome, "condition" if "condition" in d.columns else None)
    if skip is not None:
        return skip
    if d["sequence_id"].nunique() < 2 or d["condition"].nunique() < 2:
        return {"note": "insufficient grouping structure"}
    # Center attempt number: decouples the main effect from the interaction and
    # improves numerical conditioning of the mixed model (fewer boundary warnings).
    d["attempt_c"] = d["attempt_index"] - d["attempt_index"].mean()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = smf.mixedlm(
                f"{outcome} ~ attempt_c * C(condition)", d, groups=d["sequence_id"]
            ).fit(method="lbfgs", reml=True, disp=False)
        boundary = any(issubclass(w.category, ConvergenceWarning) for w in caught)
        re_var = (
            float(model.cov_re.iloc[0, 0])
            if getattr(model, "cov_re", None) is not None and model.cov_re.size
            else float("nan")
        )
        out = {
            "converged": bool(model.converged) and not boundary,
            "attempt_index_coef": float(model.params.get("attempt_c", float("nan"))),
            "attempt_index_p": float(model.pvalues.get("attempt_c", float("nan"))),
            "interaction": _interaction_terms(model.pvalues),
            "random_intercept_var": re_var,
            "aic": float(model.aic) if model.aic == model.aic else None,
        }
        if not out["converged"]:
            # Say so plainly: an optimizer that did not report convergence (or stopped on the
            # boundary with a warning) has not certified its fixed effects either.
            out["note"] = (
                "DID NOT CONVERGE CLEANLY (optimizer warning and/or random-intercept variance at "
                "the boundary). The fixed effects are reported for completeness but are not "
                "certified by the fit; disclose the non-convergence wherever they are cited, "
                "and corroborate any claim with a model that converges."
            )
        elif re_var == re_var and re_var < 1e-6:
            # Converged, with near-zero between-sequence variance (boundary). The random
            # intercept is retained for the repeated-measures structure.
            out["note"] = (
                "converged with random-intercept variance near zero (boundary); random effect "
                "retained for the repeated-measures design"
            )
        return out
    except Exception as exc:  # pragma: no cover - degenerate model
        return {"note": f"LMM unavailable: {exc}"}


def _attempt_or(d: pd.DataFrame, outcome: str, rhs: str) -> dict:
    """Odds ratio per attempt: logit GLM, robust SEs clustered on sequence."""

    from analysis import clustered

    out = clustered.glm_cluster_or(d, outcome, rhs, "attempt_index")
    if "odds_ratio" in out:
        out["n_sequences"] = out.pop("n_clusters")
    return out


_ATTEMPT_EFFECTS_NOTE = (
    "Logit GLMs with robust standard errors clustered on sequence_id (independence working "
    "model; statsmodels' small-sample correction). Conditions whose outcome is constant are "
    "excluded: an attempt slope is not identified there (that is also why the interaction "
    "model's terms for them are numerical artefacts). average_attempt_effect = outcome ~ "
    "attempt_index + C(condition) over the varying conditions (one common slope); "
    "per_condition = outcome ~ attempt_index within each. Every clustered slope rests on at "
    "least 20 informative sequences (clustered.MIN_INFORMATIVE_SEQUENCES; sequences in which "
    "the outcome departs from the condition's most common value): a condition with fewer gets "
    "no slope of its own (own_slope_withheld) but stays in the average, which needs 20 across "
    "the conditions it pools. The *_excluding_first_attempt "
    "blocks drop attempt_index 0 (the only attempt that begins from the original configuration "
    "and, under prompt format 2, the only one without memory of earlier attempts). They were "
    "POST HOC for the archived runs; for the re-run, excluding the "
    "first attempt is a planned sensitivity (decided 2026-10-29; see h03_tests)."
)


def attempt_effects(df: pd.DataFrame, outcome: str) -> dict:
    """Average and per-condition attempt slopes, with the first-attempt-excluded sensitivity."""

    need = {outcome, "attempt_index", "condition", "sequence_id"}
    if not need <= set(df.columns):
        return {"note": f"missing columns {sorted(need - set(df.columns))}"}
    d = df.dropna(subset=list(need)).copy()
    d[outcome] = pd.to_numeric(d[outcome], errors="coerce")
    d = d.dropna(subset=[outcome])
    d["condition"] = d["condition"].astype(str)

    from analysis import clustered

    def split(frame: pd.DataFrame) -> tuple[list[str], dict, int]:
        """The varying conditions, those whose own slope is withheld, and the pooled count."""

        varying, withheld, pooled = [], {}, 0
        for c, g in frame.groupby("condition"):
            if g[outcome].nunique() <= 1:
                continue
            info = clustered.informative_sequences(g, outcome)
            varying.append(str(c))
            pooled += info["n_sequences_departing"]
            if not info["enough"]:
                withheld[str(c)] = clustered.too_few_informative_note(info)
        return sorted(varying), withheld, int(pooled)

    varying, withheld, pooled = split(d)
    constant = sorted(c for c in d["condition"].unique() if c not in varying)
    out: dict = {
        "conditions_included": varying,
        "conditions_excluded_constant_outcome": constant,
        "note": _ATTEMPT_EFFECTS_NOTE,
    }
    if withheld:
        out["own_slope_withheld"] = withheld
    if not varying:
        out["note"] = "no condition has a varying outcome; no attempt slope is identified"
        return out

    def average(frame: pd.DataFrame, conds: list[str], n_informative: int) -> dict:
        if n_informative < clustered.MIN_INFORMATIVE_SEQUENCES:
            return {"note": clustered.too_few_pooled_note(n_informative)}
        rhs = "attempt_index + C(condition)" if len(conds) > 1 else "attempt_index"
        return _attempt_or(frame[frame["condition"].isin(conds)], outcome, rhs)

    def per_condition(frame: pd.DataFrame, conds: list[str], held: dict) -> dict:
        return {
            c: (
                {"note": held[c]}
                if c in held
                else _attempt_or(frame[frame["condition"] == c], outcome, "attempt_index")
            )
            for c in conds
        }

    sub = d[d["condition"].isin(varying)]
    out["average_attempt_effect"] = average(sub, varying, pooled)
    out["per_condition"] = per_condition(sub, varying, withheld)
    later = sub[sub["attempt_index"] >= 1]
    later_varying, later_withheld, later_pooled = split(later)
    constant_later = sorted(set(varying) - set(later_varying))
    if constant_later:
        out["conditions_constant_excluding_first_attempt"] = constant_later
    if later_varying:
        out["average_excluding_first_attempt"] = {
            **average(later, later_varying, later_pooled),
            "label": (
                "sensitivity with the first attempt excluded (attempt_index >= 1): POST HOC for "
                "the archived runs; the re-run's planned version (2026-10-29) is "
                "h03_tests.outcomes.<outcome>.sensitivity_excluding_first_attempt"
            ),
        }
        out["per_condition_excluding_first_attempt"] = per_condition(
            later, later_varying, later_withheld
        )
    return out


def _binary_adaptation(df: pd.DataFrame, outcome: str) -> dict:
    """Logit GLMM/GEE for a binary outcome with attempt x condition (adaptation)."""

    from analysis import clustered

    r = clustered.binary_clustered(df, outcome, "attempt_index * C(condition)")
    terms = r.get("terms")
    if not terms:
        return r
    interaction = {k: v for k, v in terms.items() if ":" in k and "C(condition)" in k}
    # The treatment-coding reference is the first level of the rows the model used (patsy
    # drops rows with a missing value; a categorical keeps its own level order).
    used = df.dropna(subset=[c for c in (outcome, "attempt_index", "condition") if c in df])
    cond = used["condition"]
    if isinstance(cond.dtype, pd.CategoricalDtype):
        present = set(cond.astype(str))
        levels = [str(c) for c in cond.cat.categories if str(c) in present]
    else:
        levels = sorted(str(c) for c in cond.unique())
    reference = levels[0] if levels else None
    ref_y = pd.to_numeric(used.loc[cond.astype(str) == reference, outcome], errors="coerce")
    attempt = terms.get("attempt_index")
    if isinstance(attempt, dict) and reference is not None:
        interpretation = (
            f"attempt slope in the REFERENCE condition ({reference}) under treatment coding "
            "-- NOT an average main effect. See attempt_effects for the average and "
            "per-condition slopes."
        )
        if ref_y.nunique() < 2:
            interpretation += (
                " The reference condition's outcome is constant, so this slope is not "
                "identified (separation): do not report it."
            )
        attempt = {
            **attempt,
            "reference_condition": reference,
            "reference_outcome_constant": bool(ref_y.nunique() < 2),
            "interpretation": interpretation,
        }
    r["attempt_effect"] = attempt
    r["interaction"] = {
        "terms": {k: v["p"] for k, v in interaction.items()},
        "any_significant": (
            any(v["p"] < 0.05 for v in interaction.values()) if interaction else None
        ),
    }
    r["attempt_effects"] = attempt_effects(df, outcome)
    return r


# --------------------------------------------------------------------------- #
# H03's decision tests (rule of 2026-10-29)
# --------------------------------------------------------------------------- #

#: The date the H03 decision rule was set (decision B1 of 2026-10-29; not pre-registered).
H03_RULE_DATE = "2026-10-29"
H03_ALPHA = 0.05
#: Time-to-compromise is restricted to the time-summary horizon (RQ1's, 2026-10-28).
H03_HORIZON_S = 300.0
RESTRICTED_TTC = "restricted_ttc_300s"
#: Ha3's predicted direction of the attempt effect, per outcome.
HA3_DIRECTION = {
    "exploit_success": "increase",
    "exploitation_success": "increase",
    "time_to_compromise_s": "decrease",
}
ATTEMPT_CLAUSE = (
    "attempt sequence will have no statistically significant effect on defensive outcomes"
)
INTERACTION_CLAUSE = (
    "there will be no statistically significant interaction between attempt sequence and "
    "defense condition"
)
_H03_NOT_REJECTED = "not rejected: no effect detected (not evidence of no effect)"
_H03_PLANNED_SENSITIVITY = (
    "PLANNED SENSITIVITY (decided 2026-10-29): the same two tests with each sequence's first "
    "attempt excluded (attempt_index >= 1). In the conditions that move services the first "
    "attempt is the only one that begins from the original configuration, and under prompt "
    "format 2 it is the only one without memory of earlier attempts; on the archived runs the "
    "attempt association sat in the first transition."
)
_H03_RULE = (
    "DECISION RULE (decided 2026-10-29; not pre-registered). H03 is evaluated separately for "
    "exploitation success and for time-to-compromise restricted to 300 s, each at alpha = .05 "
    "and clause by clause. The attempt clause is rejected if the average attempt effect is "
    "significant, and the interaction clause if the joint attempt x condition test is "
    "significant. Both models use the conditions in which the outcome varies (for the binary "
    "outcome, those whose own attempt slope is identified), with standard errors clustered on "
    "sequence. Every clustered slope rests on at least 20 informative sequences, sequences in "
    "which the outcome departs from the condition's most common value "
    "(clustered.MIN_INFORMATIVE_SEQUENCES; with fewer, the clustered standard error collapses): "
    "the interaction test, which compares the conditions' own slopes, leaves out a condition "
    "with fewer than 20 of its own, and the attempt clause needs 20 across the conditions it "
    "pools. Excluding each sequence's first attempt is a planned sensitivity analysis. "
    "Exploit success is tested in the same way if it varies and is otherwise reported as "
    "constant. Ha3 is directional: a significant attempt effect in the other direction rejects "
    "H03 without supporting Ha3. The rule was set while the main study was running, after its "
    "interim checks and before any model was fitted to its data; the 20-sequence threshold was "
    "added the same day, after a code review and a simulation of the design, before any re-run "
    "data were analyzed (scripts/h03_inclusion_calibration.py)."
)


def _cluster_glm(d: pd.DataFrame, formula: str, binary: bool):
    """GLM with sequence-clustered (sandwich) SEs, as ``clustered.glm_cluster_or`` fits it."""

    import warnings

    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    from analysis import clustered

    family = sm.families.Binomial() if binary else sm.families.Gaussian()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = smf.glm(formula, d, family=family).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(d[clustered.CLUSTER])[0]}
        )
    separated = any(
        "separation" in type(w.message).__name__.lower() or "separation" in str(w.message).lower()
        for w in caught
    )
    return m, separated


def _attempt_slope(d: pd.DataFrame, outcome: str, rhs: str) -> dict:
    """Seconds per attempt: linear model with sequence-clustered SEs (restricted time)."""

    try:
        m, _ = _cluster_glm(d, f"{outcome} ~ {rhs}", binary=False)
        b, se, p = (
            float(m.params["attempt_index"]),
            float(m.bse["attempt_index"]),
            float(m.pvalues["attempt_index"]),
        )
        ci = m.conf_int().loc["attempt_index"]
    except Exception as exc:  # noqa: BLE001 - degenerate subset
        return {"note": f"cluster-robust linear model unavailable: {exc}"}
    if not (np.isfinite(b) and np.isfinite(se) and np.isfinite(p)):
        return {"note": "attempt slope not identified (non-finite estimate or standard error)"}
    return {
        "slope_s_per_attempt": b,
        "ci95": [float(ci.iloc[0]), float(ci.iloc[1])],
        "p": p,
        "n": int(m.nobs),
        "n_sequences": int(d["sequence_id"].nunique()),
    }


def _joint_interaction(d: pd.DataFrame, outcome: str, binary: bool) -> dict:
    """One Wald chi-square test that every attempt x condition term is zero (clustered)."""

    conds = sorted(d["condition"].astype(str).unique().tolist())
    if len(conds) < 2:
        return {"note": "only one condition has a varying outcome: no interaction to test"}
    try:
        m, separated = _cluster_glm(d, f"{outcome} ~ attempt_index * C(condition)", binary)
    except Exception as exc:  # noqa: BLE001 - degenerate subset
        return {"note": f"interaction model unavailable: {exc}"}
    names = [n for n in m.params.index if n.startswith("attempt_index:C(condition)")]
    b = m.params[names].to_numpy(dtype=float)
    cov = m.cov_params().loc[names, names].to_numpy(dtype=float)
    bad = (
        separated
        or not names
        or not np.all(np.isfinite(b))
        or not np.all(np.isfinite(cov))
        or (binary and np.any(np.abs(b) > 20))
    )
    if bad or np.linalg.matrix_rank(cov) < len(names):
        return {
            "note": (
                "the attempt x condition terms are not identified in this subset (separation, a "
                "non-finite estimate or a singular covariance); no joint test is reported"
            ),
            "conditions": conds,
        }
    stat = float(b @ np.linalg.solve(cov, b))
    df_ = len(names)
    return {
        "test": "Wald chi-square, all attempt x condition terms = 0 (sequence-clustered covariance)",
        "chi2": stat,
        "df": df_,
        "p": float(stats.chi2.sf(stat, df_)),
        "reference_condition": conds[0],
        "terms": {n: {"estimate": float(m.params[n]), "p": float(m.pvalues[n])} for n in names},
        "n": int(m.nobs),
        "n_sequences": int(d["sequence_id"].nunique()),
        "conditions": conds,
    }


def _clause_tests(d: pd.DataFrame, outcome: str, binary: bool) -> dict:
    """The attempt clause and the interaction clause over the conditions where ``outcome`` varies.

    Every clustered slope rests on at least ``clustered.MIN_INFORMATIVE_SEQUENCES`` informative
    sequences (sequences in which the outcome departs from the condition's most common value).
    The attempt clause pools every condition whose outcome varies (for the binary outcome, whose
    own attempt slope is identified) and needs that many across them. The interaction clause
    compares the conditions' own slopes, so it takes only the conditions with that many of their
    own; the others are listed in ``conditions_left_out_of_interaction_test``.
    """

    from analysis import clustered

    tol = 1e-9
    per_condition: dict = {}
    included: list[str] = []
    excluded: dict = {}
    left_out: dict = {}
    support: dict = {}
    for c, g in d.groupby("condition", sort=True):
        c = str(c)
        y = pd.to_numeric(g[outcome], errors="coerce").dropna()
        if not len(y) or float(y.max() - y.min()) <= tol:
            excluded[c] = "outcome constant"
            continue
        info = clustered.informative_sequences(g, outcome)
        support[c] = info
        r = (
            _attempt_or(g, outcome, "attempt_index")
            if binary
            else _attempt_slope(g, outcome, "attempt_index")
        )
        if binary and "odds_ratio" not in r:
            per_condition[c] = r
            excluded[c] = "attempt slope not identified (separation)"
            continue
        included.append(c)
        if info["enough"]:
            per_condition[c] = r
        else:
            left_out[c] = clustered.too_few_informative_note(info)
            per_condition[c] = {"note": left_out[c]}
    in_interaction = [c for c in included if c not in left_out]
    pooled = int(sum(support[c]["n_sequences_departing"] for c in included))
    out: dict = {
        "conditions_included": included,
        "conditions_excluded": excluded,
        "conditions_in_interaction_test": in_interaction,
        "conditions_left_out_of_interaction_test": left_out,
        "informative_sequences": support,
        "informative_sequences_pooled": pooled,
        "per_condition": per_condition,
    }
    if not included:
        out["note"] = "no condition has a varying outcome with an identified attempt slope"
        return out
    if pooled >= clustered.MIN_INFORMATIVE_SEQUENCES:
        sub = d[d["condition"].astype(str).isin(included)]
        rhs = "attempt_index + C(condition)" if len(included) > 1 else "attempt_index"
        out["attempt_clause"] = (
            _attempt_or(sub, outcome, rhs) if binary else _attempt_slope(sub, outcome, rhs)
        )
    else:
        out["attempt_clause"] = {"note": clustered.too_few_pooled_note(pooled)}
    if len(in_interaction) >= 2:
        sub = d[d["condition"].astype(str).isin(in_interaction)]
        out["interaction_clause"] = _joint_interaction(sub, outcome, binary)
    else:
        out["interaction_clause"] = {
            "note": (
                "fewer than two conditions whose outcome departs from its most common value in "
                f"at least {clustered.MIN_INFORMATIVE_SEQUENCES} sequences: no interaction to "
                "test"
            ),
            "conditions": in_interaction,
        }
    return out


def restricted_ttc(df: pd.DataFrame, horizon: float = H03_HORIZON_S) -> tuple[pd.DataFrame, dict]:
    """The trials with min(time_to_compromise_s, horizon) as ``RESTRICTED_TTC``.

    A trial censored before the horizon has no determined restricted time; it is dropped and
    counted (none is expected: every censored trial is censored at the trial window).
    """

    need = {"time_to_compromise_s", "ttc_censored", "attempt_index", "condition", "sequence_id"}
    if not need <= set(df.columns):
        return df.iloc[0:0], {"note": f"missing columns {sorted(need - set(df.columns))}"}
    d = df.dropna(subset=list(need)).copy()
    t = pd.to_numeric(d["time_to_compromise_s"], errors="coerce")
    c = pd.to_numeric(d["ttc_censored"], errors="coerce")
    early = (c == 1) & (t < horizon)
    d[RESTRICTED_TTC] = np.minimum(t, horizon)
    d = d[~early & t.notna() & c.notna()]
    return d, {"horizon_s": float(horizon), "censored_before_horizon_dropped": int(early.sum())}


def _clause_result(block: dict, key: str, p_key: str = "p") -> dict:
    r = block.get(key) or {}
    p = r.get(p_key)
    if p is None or not np.isfinite(p):
        return {"result": "not tested", "why": r.get("note") or block.get("note", "no p-value")}
    return {"p": float(p), "result": "rejected" if p < H03_ALPHA else _H03_NOT_REJECTED}


def _outcome_decision(outcome: str, blk: dict) -> dict:
    if not blk.get("evaluated"):
        return {"result": "not evaluated", "reason": blk.get("reason", "not tested")}
    att = blk.get("attempt_clause") or {}
    first = {
        "statement": ATTEMPT_CLAUSE,
        "conditions": blk.get("conditions_included", []),
        **_clause_result(blk, "attempt_clause"),
    }
    if "odds_ratio" in att:
        est, up = att["odds_ratio"], att["odds_ratio"] > 1
        first.update({"odds_ratio_per_attempt": est, "ci95": att.get("ci95")})
    elif "slope_s_per_attempt" in att:
        est, up = att["slope_s_per_attempt"], att["slope_s_per_attempt"] > 0
        first.update({"slope_s_per_attempt": est, "ci95": att.get("ci95")})
    else:
        up = None
    if up is not None:
        direction = "increase" if up else "decrease"
        first["direction"] = direction
        first["ha3_direction"] = HA3_DIRECTION.get(outcome)
        first["matches_ha3_direction"] = direction == HA3_DIRECTION.get(outcome)
    inter = blk.get("interaction_clause") or {}
    second = {
        "statement": INTERACTION_CLAUSE,
        "conditions": blk.get("conditions_in_interaction_test", []),
        **_clause_result(blk, "interaction_clause"),
    }
    if "chi2" in inter:
        second.update({"chi2": inter["chi2"], "df": inter["df"]})
    results = (first["result"], second["result"])
    if "rejected" in results:
        overall = "rejected"
    elif results == ("not tested", "not tested"):
        overall = "not tested"
    elif "not tested" in results:
        overall = "incomplete: one clause was not tested and the other was not rejected"
    else:
        overall = _H03_NOT_REJECTED
    sens = blk.get("sensitivity_excluding_first_attempt") or {}
    sens_first = _clause_result(sens, "attempt_clause")["result"]
    sens_second = _clause_result(sens, "interaction_clause")["result"]
    out = {
        "conditions_excluded": blk.get("conditions_excluded", {}),
        "attempt_clause": first,
        "interaction_clause": second,
        "h03_for_this_outcome": overall,
        "sensitivity_excluding_first_attempt": {
            "attempt_clause": sens_first,
            "interaction_clause": sens_second,
            "agrees_with_decision": (sens_first, sens_second) == results,
        },
    }
    if blk.get("conditions_left_out_of_interaction_test"):
        out["conditions_left_out_of_interaction_test"] = blk[
            "conditions_left_out_of_interaction_test"
        ]
    return out


def h03_tests(df: pd.DataFrame) -> dict:
    """H03's two clauses, one test each, per outcome (rule of 2026-10-29), with the decision."""

    need = {"attempt_index", "condition", "sequence_id"}
    if not need <= set(df.columns):
        return {"note": f"missing columns {sorted(need - set(df.columns))}"}
    outcomes: dict = {}
    for outcome in ("exploit_success", "exploitation_success"):
        if outcome not in df.columns:
            continue
        d = df.dropna(subset=[outcome, *need]).copy()
        d[outcome] = pd.to_numeric(d[outcome], errors="coerce")
        d = d.dropna(subset=[outcome])
        if d[outcome].nunique() < 2:
            outcomes[outcome] = {
                "evaluated": False,
                "reason": "constant",
                "value": float(d[outcome].iloc[0]) if len(d) else None,
                "n_trials": int(len(d)),
            }
            continue
        later = d[d["attempt_index"] >= 1]
        outcomes[outcome] = {
            "evaluated": True,
            "model": "logit GLM, sequence-clustered SEs; estimate = odds ratio per attempt",
            **_clause_tests(d, outcome, binary=True),
            "sensitivity_excluding_first_attempt": {
                "label": _H03_PLANNED_SENSITIVITY,
                **_clause_tests(later, outcome, binary=True),
            },
        }
    rt, info = restricted_ttc(df)
    if "note" in info:
        outcomes["time_to_compromise_s"] = {"evaluated": False, "reason": info["note"]}
    else:
        later = rt[rt["attempt_index"] >= 1]
        outcomes["time_to_compromise_s"] = {
            "evaluated": True,
            "model": (
                "linear model of min(time_to_compromise_s, 300) on attempt, sequence-clustered "
                "SEs; estimate = seconds of restricted time per attempt (negative = compromised "
                "sooner as attempts go on)"
            ),
            **info,
            **_clause_tests(rt, RESTRICTED_TTC, binary=False),
            "sensitivity_excluding_first_attempt": {
                "label": _H03_PLANNED_SENSITIVITY,
                **_clause_tests(later, RESTRICTED_TTC, binary=False),
            },
        }
    decision = {
        "rule_date": H03_RULE_DATE,
        "alpha": H03_ALPHA,
        "rule": _H03_RULE,
        "outcomes": {k: _outcome_decision(k, v) for k, v in outcomes.items()},
        "global_decision": None,
        "note": (
            "No global decision across outcomes is made, and each outcome's line restates its "
            "two clause decisions: cite the clauses. Under the joint null the chance that at "
            "least one of an outcome's two clauses is rejected can approach .10."
        ),
    }
    return {"outcomes": outcomes, "decision": decision}


def rq3(df: pd.DataFrame) -> dict:
    """Run the RQ3 mixed-effects adaptation analyses (attempt x condition)."""

    results: dict = {}
    # Continuous LMM outcomes: the enumeration diagnostic (mostly observed), not the now
    # obj_b-anchored time_to_compromise_s (a censored survival outcome — its adaptation is
    # captured by the binary exploitation_success below and by the RQ1 Cox).
    for outcome in ("time_to_enumeration_s", "recon_accuracy_f1"):
        if outcome in df.columns:
            results[outcome] = _lmm(df, outcome)
    for outcome in ("exploit_success", "exploitation_success"):
        if outcome in df.columns:
            results[outcome] = _binary_adaptation(df, outcome)
    results["h03_tests"] = h03_tests(df)
    return results
