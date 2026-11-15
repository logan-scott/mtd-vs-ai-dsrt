"""Variance screening and numerical-health diagnostics for the analysis layer.

An outcome that does not vary carries no information about the defense condition,
but ``statsmodels`` will still fit a model to it. On a column that is identically
25.0 the fixed-effect coefficients come back as double-precision round-off
(3.9e-14), the standard errors come back as round-off as well (1.3e-15), and the
ratio of round-off to round-off is a large z -- so the report carries ``p = 0.0``
and ``significant = true`` for a "difference" of 4e-14. The *same* degeneracy has
been observed with the opposite sign under a different ``statsmodels`` build:
standard errors of ~2.7e5 and an omnibus Wald ``p = 1.0`` on the identical frozen
CSV. Both are one pathology -- a collapsed random intercept on a
within-group-constant outcome -- manifesting in opposite directions.

That observation dictates the design of this module: **every routing signal keys
on the data** (the outcome column's own variance structure), and no routing
signal keys on a fitted symptom such as "the standard error is implausibly
large", "the random-intercept variance collapsed", or "p is suspiciously close to
1". Data-side signals are invariant to the linear-algebra stack because the data
is frozen; fitted symptoms are not. In particular **no signal here, routing or
advisory, reads the magnitude of a model's own p-value** -- doing so would be
p-value-conditional estimator selection, and it would be one-directional (a large
p could promote a different test while a small p never could).

Two screens live here, and they answer different questions.

1. **Pooled variance screen** (``variance_profile`` / ``zero_variance_skip`` /
   ``is_constant``): *does this outcome vary at all?* An outcome with no
   resolvable variance anywhere is not estimable: no model is fit, no test is
   performed, and the skip is recorded. This is the screen the RQ estimators call
   before handing a column to ``statsmodels``.

2. **Fit-conditioning screen** (``lmm_degeneracy``): *given that the outcome does
   vary, is the fitted random-intercept model numerically trustworthy?* A
   random-intercept LMM fitted to an outcome that is exactly constant *within* one
   or more levels of the fixed factor is not identified: inside that stratum both
   the between-cluster and the within-cluster variance are zero, the REML profile
   is maximised on the boundary ``sigma_b^2 = 0``, and the fixed-effect covariance
   is obtained by inverting a numerically singular Hessian. Everything downstream
   of ``cov_params()`` -- the EMMean standard errors, the pairwise contrasts and
   the omnibus Wald test -- is then not an estimate.

The two screens compose in a fixed order and are mutually exclusive by
construction, because a pooled-constant outcome returns from ``variance_profile``
before the within-group pass is reached:

===  ======================================================  ==========================  ==========================================
#    predicate                                               ``variance_profile`` status  action
===  ======================================================  ==========================  ==========================================
0    column absent                                           ``missing``                 skip the model and the test
1    fewer than two finite values                            ``insufficient_data``       skip the model and the test
2    ``ptp(pooled) <= tol``                                   ``constant``                skip the model and the test
3    pooled varies, some group has ``ptp(group) <= tol``      ``some_groups_constant``    estimable; hand to ``lmm_degeneracy``
4    otherwise                                               ``ok``                      normal path
===  ======================================================  ==========================  ==========================================

Row 3 is *not* on its own a licence to discard a result. ``exploitation_success``
carries that status -- it is exactly 1.0 in three of the six conditions -- and it
holds the study's headline obj_b ranking. It is analysed with a logit GLMM, not an
LMM, and nothing in this module touches it.

Tolerances
----------
The pooled screen calls an outcome constant when
``ptp(x) <= max(ATOL, RTOL * max(|x|, 1))`` with ``RTOL = 1e-9`` and
``ATOL = 1e-12``. Double-precision round-off at these magnitudes is ~2.2e-16
relative, so RTOL sits ~6 orders of magnitude above numerical noise; the smallest
genuine between-arm difference in the declared outcome set is 8.7e-3
(``recon_accuracy_f1``; 5.3e-2 for ``attack_path_churn``), ~7 orders of magnitude
above the tolerance. The screen therefore cannot manufacture a false zero and
cannot miss a genuine constant.

**Restriction:** ``max(|x|)`` is a faithful magnitude only for ratio-scaled
outcomes with a meaningful zero -- which every declared outcome is (elapsed
seconds, percentages, F1 ratios, counts, z-scores). Do not apply this screen to an
offset-scaled column such as an absolute epoch timestamp, where a real 0.5 s
spread on a 1e9 base would fall inside the tolerance.

The fit-conditioning screen's thresholds are defended individually on
``LMM_DEGENERACY_THRESHOLDS``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Screen 1: pooled variance ("is this outcome estimable at all?")
# --------------------------------------------------------------------------- #

# Relative tolerance on the outcome's own scale; see the module docstring for the
# derivation of both bounds. ATOL is the floor that protects outcomes whose scale
# is below 1 (proportions, F1 ratios).
ZERO_VARIANCE_RTOL = 1e-9
ZERO_VARIANCE_ATOL = 1e-12


def _values(series) -> np.ndarray:
    """Finite numeric values of ``series`` as a float array (non-numeric -> dropped)."""

    v = pd.to_numeric(pd.Series(series), errors="coerce").to_numpy(dtype=float)
    return v[np.isfinite(v)]


def tolerance(values) -> float:
    """Scale-relative tolerance below which a range counts as zero.

    ``values`` supplies the scale. Non-finite entries are ignored rather than
    propagated, so a single NaN cannot silently disable the screen.
    """

    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)] if v.size else v
    scale = float(np.max(np.abs(v))) if v.size else 1.0
    if not np.isfinite(scale):
        scale = 1.0
    return max(ZERO_VARIANCE_ATOL, ZERO_VARIANCE_RTOL * max(scale, 1.0))


def is_constant(series) -> bool:
    """True when ``series`` has no resolvable variance (fewer than 2 values counts)."""

    v = _values(series)
    if v.size < 2:
        return True
    return bool(float(np.ptp(v)) <= tolerance(v))


def variance_profile(
    d: pd.DataFrame,
    outcome: str,
    groupcol: str | None = "condition",
    *,
    label: str | None = None,
) -> dict:
    """Describe an outcome's variance. ``status`` drives the whole decision tree.

    status is one of:
      ``missing``             -- the column is not in the frame;
      ``insufficient_data``   -- fewer than 2 finite values;
      ``constant``            -- no variance anywhere (skip the model and the test);
      ``some_groups_constant``-- constant within >=1 group, varies between groups
                                 (estimable; a fit-conditioning question, handled
                                 by ``lmm_degeneracy``, not by this screen);
      ``ok``                  -- varies within and between.
    """

    name = label or outcome
    if outcome not in d.columns:
        return {"outcome": name, "status": "missing"}
    v = _values(d[outcome])
    tol = tolerance(v)
    prof: dict = {
        "outcome": name,
        "n": int(v.size),
        "tolerance": float(tol),
        "pooled_mean": float(np.mean(v)) if v.size else None,
        "pooled_range": float(np.ptp(v)) if v.size else None,
        "pooled_sd": float(np.std(v, ddof=1)) if v.size > 1 else None,
    }
    if v.size < 2:
        prof["status"] = "insufficient_data"
        return prof
    if float(np.ptp(v)) <= tol:
        prof["status"] = "constant"
        return prof
    if not groupcol or groupcol not in d.columns:
        prof["status"] = "ok"
        return prof

    within: dict[str, float] = {}
    means: list[float] = []
    for key, g in d.groupby(groupcol, observed=True):
        gv = _values(g[outcome])
        if gv.size == 0:
            continue
        within[str(key)] = float(np.ptp(gv))
        means.append(float(np.mean(gv)))
    const_groups = sorted(k for k, r in within.items() if r <= tol)
    prof.update(
        group=groupcol,
        n_groups=len(within),
        within_group_range=within,
        constant_groups=const_groups,
        between_group_mean_range=(float(np.ptp(np.asarray(means))) if means else None),
        status="some_groups_constant" if const_groups else "ok",
    )
    return prof


def zero_variance_skip(
    d: pd.DataFrame,
    outcome: str,
    groupcol: str | None = "condition",
    *,
    label: str | None = None,
) -> dict | None:
    """Skip payload for a non-estimable outcome, or ``None`` when it is estimable.

    The ``note`` key follows the codebase idiom (degrade gracefully by returning a
    note rather than raising). ``skipped`` / ``skipped_reason`` carry the same fact
    machine-readably, so the FDR layer can tell "not tested" from "tested and null"
    without parsing prose.
    """

    prof = variance_profile(d, outcome, groupcol, label=label)
    name = prof["outcome"]
    status = prof["status"]
    if status == "missing":
        return {
            "note": f"{name} is not present in the analyzed dataset; no model was fit",
            "skipped": True,
            "skipped_reason": "outcome_missing",
            "variance_profile": prof,
        }
    if status == "insufficient_data":
        return {
            "note": f"{name} has fewer than two non-missing values; no model was fit",
            "skipped": True,
            "skipped_reason": "insufficient_data",
            "variance_profile": prof,
        }
    if status == "constant":
        return {
            "note": (
                f"{name} is constant across all analyzed trials "
                f"(range {prof['pooled_range']:.3g} <= tolerance {prof['tolerance']:.3g}); "
                "no model was fit and no test was performed"
            ),
            "skipped": True,
            "skipped_reason": "zero_variance",
            "variance_profile": prof,
        }
    return None


# --------------------------------------------------------------------------- #
# Screen 2: fit conditioning ("is this fitted LMM numerically trustworthy?")
# --------------------------------------------------------------------------- #

LMM_DEGENERACY_THRESHOLDS: dict[str, float] = {
    # ROUTES. A level whose SD is below this fraction of the outcome's own SD is
    # treated as exactly constant. Scale-free (a ratio of two SDs in the same
    # units). On the confirmatory data the smallest NON-zero level SD ratio is
    # 0.0146/0.3998 = 0.0365 (attack_path_churn / endpoint_mutation) -- seven
    # orders of magnitude of headroom -- while the firing levels sit at exactly 0.
    "zero_level_sd_over_sd": 1e-9,
    # ROUTES. For y_ij = mu_l + b_i + e_ij, the EMMean of a level supported by
    # k >= 1 clusters of size m has Var = (sigma_b^2 + sigma_e^2/m)/k, which is at
    # most sigma_b^2 + sigma_e^2 <= Var(y). So SE(emmean) <= SD(y) is a hard
    # algebraic ceiling for an identified marginal mean; 10x is slack for
    # unbalanced / covariate-adjusted designs. This is an INDEPENDENT corroborating
    # signal that additionally catches the archived (huge-SE) manifestation; it is
    # not sufficient alone, because the local (tiny-SE) manifestation is invisible
    # to it.
    "max_emmean_se_over_sd": 10.0,
    # ADVISORY ONLY, NEVER ROUTES. Random-intercept variance as a fraction of
    # Var(y). Verified non-discriminating: memory_overhead_pct (healthy) sits at
    # 1.09e-08, BELOW attack_path_churn (degenerate) at 4.09e-08, so a collapsed
    # random intercept cannot separate the two. It is emitted so a reader can see
    # the collapse without it driving a decision.
    "re_var_boundary_over_var": 1e-6,
}

_REASON_TEXT = {
    "outcome_constant": (
        "the outcome has no variance anywhere in the analysis subset, so there is no "
        "between-group contrast to estimate"
    ),
    "zero_within_level_variance": (
        "at least one level of the fixed factor has exactly zero variance in this outcome, "
        "which leaves the cluster random intercept unidentified in that stratum and makes "
        "the fixed-effect covariance (EMMean SEs, contrasts, Wald omnibus) untrustworthy"
    ),
    "lmm_failed": "the mixed model or its EMMeans/contrast step raised",
    "lmm_not_converged": "the mixed-model optimizer reported non-convergence",
    "nonfinite_emmeans": (
        "at least one estimated marginal mean or its standard error is non-finite"
    ),
    "omnibus_unavailable": "the Wald omnibus for the factor could not be formed",
    "emmean_se_implausible": (
        "an EMMean standard error exceeds the outcome's own standard deviation by more "
        "than the allowed factor, which is impossible for an identified marginal mean"
    ),
}

_ADVISORY_TEXT = {
    "random_intercept_at_boundary": (
        "the estimated random-intercept variance is at the optimizer's zero boundary "
        "relative to the outcome's variance; diagnostic only -- a genuinely zero ICC is "
        "benign (the model reduces to OLS) and several healthy outcomes in this study "
        "sit here, so this signal never routes on its own"
    ),
}

_NOTE_DEGENERATE = (
    "degenerate=true means the clustered linear mixed model is not numerically "
    "trustworthy for this outcome; the fit is retained verbatim for audit but the "
    "result to cite is <rq>.<outcome>.reported_result. No signal in this dict -- "
    "routing or advisory -- reads the magnitude of the model's own p-value; the "
    "routing decision is a function of the data and of the fit's numerical health only."
)
_NOTE_HEALTHY = (
    "no degeneracy signal fired; the clustered LMM is the reported result. No signal "
    "in this dict -- routing or advisory -- reads the magnitude of the model's own "
    "p-value; the routing decision is a function of the data and of the fit's "
    "numerical health only."
)


def _finite(x) -> bool:
    """True for a real, finite int/float (bools and None excluded)."""

    if isinstance(x, bool) or x is None:
        return False
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _level_stats(values, groups, tol_rel: float) -> tuple[dict, list[str], dict]:
    """Per-level n / SD / mean from the raw data, plus the zero-variance level list."""

    try:
        frame = pd.DataFrame(
            {
                "_y": pd.to_numeric(pd.Series(list(values)), errors="coerce"),
                "_g": pd.Series(list(groups)).astype("object"),
            }
        ).dropna()
    except Exception:  # noqa: BLE001 - diagnostics must never raise
        return {}, [], {}
    if frame.empty:
        return {}, [], {}
    sd_all = float(frame["_y"].std(ddof=1)) if len(frame) > 1 else 0.0
    if not math.isfinite(sd_all):
        sd_all = 0.0
    cutoff = tol_rel * sd_all
    per_level: dict = {}
    zero_levels: list[str] = []
    means: dict = {}
    for lvl, g in frame.groupby("_g", observed=True):
        sd = float(g["_y"].std(ddof=1)) if len(g) > 1 else 0.0
        if not math.isfinite(sd):
            sd = 0.0
        key = str(lvl)
        per_level[key] = {"n": int(len(g)), "sd": sd, "mean": float(g["_y"].mean())}
        means[key] = float(g["_y"].mean())
        if len(g) >= 2 and sd_all > 0.0 and sd <= cutoff:
            zero_levels.append(key)
    return per_level, sorted(zero_levels), means


def lmm_degeneracy(res: dict, values, groups=None, thresholds: dict | None = None) -> dict:
    """Score a ``lmm_continuous``/``emmeans_pairwise`` result dict for degeneracy.

    ``res`` is the partially built per-outcome dict (may carry ``note``,
    ``emmeans_note``, ``converged``, ``random_intercept_var``, ``residual_var``,
    ``emmeans``, ``omnibus_wald``). ``values`` is the raw outcome column and
    ``groups`` the aligned fixed-factor column for the same analysis subset.

    The routing signals are ``outcome_constant`` and ``zero_within_level_variance``
    (both functions of the frozen data, hence invariant to the linear-algebra
    stack), plus the fit-health signals ``lmm_failed`` / ``lmm_not_converged`` /
    ``nonfinite_emmeans`` / ``omnibus_unavailable`` / ``emmean_se_implausible``.
    Only the *finiteness* of the omnibus p is ever read, never its magnitude.

    Returns a JSON-serialisable diagnostics dict; never raises.
    """

    th = {**LMM_DEGENERACY_THRESHOLDS, **(thresholds or {})}
    res = res or {}
    try:
        s = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    except Exception:  # noqa: BLE001 - diagnostics must never raise
        s = pd.Series(dtype=float)
    n = int(len(s))
    sd = float(s.std(ddof=1)) if n > 1 else 0.0
    if not math.isfinite(sd):
        sd = 0.0

    reasons: list[str] = []
    advisory: list[str] = []
    diag: dict = {"n": n, "outcome_sd": sd, "thresholds": th}

    # --- data-side signals (version-stable; these are the routing signals) --- #
    if n < 2 or sd == 0.0 or int(s.nunique()) < 2:
        reasons.append("outcome_constant")

    level_stats, zero_levels, obs_means = ({}, [], {})
    if groups is not None:
        level_stats, zero_levels, obs_means = _level_stats(
            values, groups, float(th["zero_level_sd_over_sd"])
        )
    diag["level_stats"] = level_stats
    diag["zero_variance_levels"] = zero_levels
    # Computed from the DATA, not from the fit: the corroborating evidence must not
    # come from the model under indictment. The fit's own spread is reported
    # separately and suffixed `_fit`.
    diag["observed_group_means"] = obs_means
    if len(obs_means) > 1:
        spread = float(max(obs_means.values()) - min(obs_means.values()))
        diag["observed_spread"] = spread
        diag["observed_spread_over_sd"] = (spread / sd) if sd > 0 else None
    else:
        diag["observed_spread"] = None
        diag["observed_spread_over_sd"] = None
    if zero_levels:
        reasons.append("zero_within_level_variance")

    # --- fit-health signals -------------------------------------------------- #
    err = res.get("note") or res.get("emmeans_note")
    if err:
        diag["error"] = str(err)
        reasons.append("lmm_failed")

    conv = res.get("converged")
    diag["converged"] = conv
    if conv is False:
        reasons.append("lmm_not_converged")

    re_var, resid = res.get("random_intercept_var"), res.get("residual_var")
    icc = None
    # ICC is the random intercept's share of the outcome's variance; it is undefined
    # (not zero) when the outcome has no variance, so do not report a ratio there.
    if sd > 0 and _finite(re_var) and _finite(resid):
        total = float(re_var) + float(resid)
        icc = float(re_var) / total if total > 0 else None
    diag["random_intercept_var"] = re_var if _finite(re_var) else None
    diag["residual_var"] = resid if _finite(resid) else None
    diag["icc_hat"] = icc
    re_ratio = (float(re_var) / (sd * sd)) if (_finite(re_var) and sd > 0) else None
    diag["re_var_over_outcome_var"] = re_ratio
    if re_ratio is not None and re_ratio < th["re_var_boundary_over_var"]:
        advisory.append("random_intercept_at_boundary")

    emm = res.get("emmeans") or {}
    ses = [v.get("se") for v in emm.values() if isinstance(v, dict)]
    mus = [v.get("emmean") for v in emm.values() if isinstance(v, dict)]
    fin_ses = [float(x) for x in ses if _finite(x)]
    fin_mus = [float(x) for x in mus if _finite(x)]
    if emm and (len(fin_ses) < len(ses) or len(fin_mus) < len(mus)):
        reasons.append("nonfinite_emmeans")

    max_se = max(fin_ses) if fin_ses else None
    fit_spread = (max(fin_mus) - min(fin_mus)) if len(fin_mus) > 1 else None
    diag["max_emmean_se"] = max_se
    diag["emmean_spread_fit"] = fit_spread
    diag["emmean_spread_over_sd_fit"] = (
        (fit_spread / sd) if (fit_spread is not None and sd > 0) else None
    )

    ratio = (max_se / sd) if (max_se is not None and sd > 0) else None
    diag["max_emmean_se_over_sd"] = ratio
    if ratio is not None and ratio > th["max_emmean_se_over_sd"]:
        reasons.append("emmean_se_implausible")

    omni_p = (res.get("omnibus_wald") or {}).get("p")
    diag["omnibus_p_finite"] = bool(_finite(omni_p))
    # Only the FINITENESS of the omnibus p is read, never its magnitude. When the LMM
    # itself raised, "lmm_failed" already covers it and this would be redundant.
    if not res.get("note") and not _finite(omni_p):
        reasons.append("omnibus_unavailable")

    diag["reasons"] = reasons
    diag["reason_text"] = [_REASON_TEXT[r] for r in reasons]
    diag["advisory_reasons"] = advisory
    diag["advisory_reason_text"] = [_ADVISORY_TEXT[a] for a in advisory]
    diag["degenerate"] = bool(reasons)
    diag["note"] = _NOTE_DEGENERATE if reasons else _NOTE_HEALTHY
    return diag
