"""Separation-robust estimates for a binary outcome that is constant within a group.

When one level of a comparison factor has the outcome in EVERY trial (or in none) -- the
static baseline and deceptive presentation were compromised in all 264 of their trials --
the logit odds ratio against that level has no maximum-likelihood value: the likelihood
keeps rising as the coefficient runs to infinity ("separation"; Albert & Anderson, 1984).
The variational-Bayes GLMM the pipeline uses still returns a finite odds ratio, but it is
set by the estimator's default normal prior on the fixed effects (SD 2), not by the data:
refitting with only that prior changed moves it by orders of magnitude (see
``vb_prior_sensitivity``). This module adds estimates that ARE identified by the data:

- ``risk_contrast_static_vs_mtd`` / ``technique_rates``: compromise proportions, their
  difference and ratio, with a sequence cluster bootstrap (well defined at 0% or 100%);
- ``firth_block``: Firth-penalized logistic regression (Firth, 1993; Heinze & Schemper,
  2002), the standard finite-sample remedy for separation, with a sequence cluster
  bootstrap for the interval because trials are clustered within sequences;
- ``vb_prior_sensitivity``: the VB odds ratios refitted at several prior SDs, so a reader
  can see which terms the prior determines.

Nothing here changes the existing estimates; it adds the blocks a separated comparison
needs. Detection (``separation_profile``) reads only whether a level's outcome is constant,
never an effect size or a p-value.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from analysis import clustered

N_BOOT = clustered.BOOT_DEFAULT
SEED = clustered.BOOT_SEED_DEFAULT
PRIOR_SDS = (1.0, 2.0, 5.0, 10.0)
DEFAULT_PRIOR_SD = 2.0  # statsmodels BinomialBayesMixedGLM.from_formula(fe_p=2)
# A term whose VB odds ratio changes more than this factor across PRIOR_SDS is reported as
# prior-dominated: the data are not pinning it down.
PRIOR_FOLD_FLAG = 2.0

_FIRTH_NOTE = (
    "Firth-penalized logistic regression (Jeffreys-prior penalty, 0.5 log|X'WX|; Firth 1993, "
    "Heinze & Schemper 2002): finite and approximately unbiased under separation, where the "
    "maximum-likelihood estimate does not exist. Point estimate from the full data; 95% "
    "interval and p from a PERCENTILE SEQUENCE CLUSTER BOOTSTRAP (whole sequences resampled "
    "within condition, so the 8 correlated attempts of a sequence stay together). p is "
    "two-sided, 2 x min(share of replicates <= 0, share >= 0) with the (count + 1)/(B + 1) "
    "correction, so it is floored at 2/(B + 1): write a floored p as 'p < .001'. CAVEAT for "
    "terms flagged involves_separated_level: when a level is compromised in every trial (or in "
    "none), the Firth odds ratio for a contrast with that level is finite only because of the "
    "penalty (for a saturated 2x2 table it equals adding half an observation to every cell), "
    "and its bootstrap interval is narrow because that level never varies across replicates "
    "-- it does not express uncertainty about that level's true rate. Report such a term, if "
    "at all, as a bias-reduced point value with this caveat, and lead with the risk "
    "difference. Terms not flagged are ordinary, data-identified odds ratios."
)
_RISK_NOTE = (
    "Compromise proportions are trial-level means (each sequence contributes its 8 attempts); "
    "95% intervals and p are a PERCENTILE SEQUENCE CLUSTER BOOTSTRAP stratified by condition. "
    "Unlike an odds ratio, a difference or ratio of proportions is well defined when one arm "
    "is at 0% or 100%. The pooled MTD proportion is a design-weighted average over the MTD "
    "conditions in the study's fixed allocation, not a technique-neutral effect. An arm that "
    "is constant (0% or 100% in every trial) does not vary across bootstrap replicates, so its "
    "own interval is degenerate and the interval of a difference with it reflects only the "
    "other arm's sampling variability; constant_arms gives the exact (Clopper-Pearson) 95% "
    "interval for such an arm's rate at the trial level and, more conservatively, at the "
    "sequence level (every sequence at the same value)."
)
_VB_NOTE = (
    "The pipeline's GLMM (variational Bayes, BinomialBayesMixedGLM.fit_vb) refitted with ONLY "
    "the fixed-effect prior SD changed (statsmodels default 2). A term whose odds ratio moves "
    f"more than {PRIOR_FOLD_FLAG:g}-fold across these priors is determined by the prior, not "
    "the data: do not report it as an estimate. Terms that do not move are data-determined."
)


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #


def separation_profile(d: pd.DataFrame, outcome: str, factor: str) -> dict:
    """Which levels of ``factor`` have ``outcome`` constant (all 0 or all 1)?

    Separation in the sense that matters here: a level whose outcome never varies while the
    outcome varies overall. Reads only the outcome's constancy per level.
    """

    if outcome not in d.columns or factor not in d.columns:
        return {"separated": False, "note": f"{outcome} or {factor} absent"}
    y = pd.to_numeric(d[outcome], errors="coerce")
    dd = d.assign(_y=y).dropna(subset=["_y", factor])
    if dd["_y"].nunique() < 2:
        return {"separated": False, "note": f"{outcome} is constant overall; nothing to compare"}
    levels: dict = {}
    separated: list[str] = []
    for level, g in dd.groupby(factor, observed=True):
        vals = g["_y"]
        constant = bool(vals.nunique() == 1)
        levels[str(level)] = {
            "n": int(len(vals)),
            "proportion": float(vals.mean()),
            "constant": constant,
        }
        if constant:
            separated.append(str(level))
    return {
        "separated": bool(separated),
        "factor": factor,
        "separated_levels": sorted(separated),
        "levels": levels,
        "note": (
            "A level listed in separated_levels has the outcome in every trial or in none, so a "
            "logit odds ratio against it has no maximum-likelihood value."
            if separated
            else "No level is constant; the logit odds ratios are identified by the data."
        ),
    }


# --------------------------------------------------------------------------- #
# Firth logistic regression
# --------------------------------------------------------------------------- #


def _penalized_loglik(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
    eta = X @ beta
    ll = float(np.sum(y * eta - np.logaddexp(0.0, eta)))
    p = 1.0 / (1.0 + np.exp(-eta))
    w = p * (1.0 - p)
    sign, logdet = np.linalg.slogdet(X.T @ (X * w[:, None]))
    if sign <= 0:
        return -math.inf
    return ll + 0.5 * float(logdet)


def firth_logit(
    X: np.ndarray, y: np.ndarray, *, max_iter: int = 200, tol: float = 1e-9, max_step: float = 5.0
) -> dict:
    """Firth-penalized logistic regression by modified-score Newton steps with halving.

    Maximizes l(b) + 0.5 log|X'W(b)X|. Each step solves (X'WX) delta = U*(b), where the
    modified score is U*(b) = X'(y - p + h (1/2 - p)) and h is the diagonal of the hat matrix
    W^1/2 X (X'WX)^-1 X' W^1/2. Steps are capped at ``max_step`` per coordinate and halved
    until the penalized log-likelihood does not decrease (as R's logistf does).
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    beta = np.zeros(X.shape[1])
    current = _penalized_loglik(X, y, beta)
    converged = False
    it = 0
    for it in range(1, max_iter + 1):  # noqa: B007 - the final count is reported
        eta = X @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        w = p * (1.0 - p)
        info = X.T @ (X * w[:, None])
        try:
            inv = np.linalg.inv(info)
        except np.linalg.LinAlgError:
            break
        h = w * np.einsum("ij,jk,ik->i", X, inv, X)
        score = X.T @ (y - p + h * (0.5 - p))
        step = inv @ score
        biggest = float(np.max(np.abs(step))) if step.size else 0.0
        if biggest > max_step:
            step = step * (max_step / biggest)
        t = 1.0
        new = beta + step
        new_ll = _penalized_loglik(X, y, new)
        while new_ll < current - 1e-12 and t > 1e-8:
            t /= 2.0
            new = beta + t * step
            new_ll = _penalized_loglik(X, y, new)
        beta, current = new, new_ll
        if float(np.max(np.abs(t * step))) < tol:
            converged = True
            break
    return {"beta": beta, "converged": converged, "iterations": it, "penalized_loglik": current}


def _design(d: pd.DataFrame, outcome: str, rhs: str):
    import patsy

    y, X = patsy.dmatrices(f"{outcome} ~ {rhs}", d, return_type="dataframe", NA_action="drop")
    return y.iloc[:, 0].to_numpy(dtype=float), X


def boot_p(draws: np.ndarray) -> float | None:
    """Two-sided percentile-bootstrap p for H0: parameter = 0, with the +1 correction.

    A replicate whose statistic is zero up to rounding (e.g. a log odds ratio of -1.7e-16
    when both arms are compromised in every trial) is a tie and counts in BOTH tails;
    counting it in one tail would make an uninformative replicate look like evidence.
    """

    b = np.asarray(draws, dtype=float)
    b = b[np.isfinite(b)]
    if not b.size:
        return None
    tol = 1e-9 * max(1.0, float(np.max(np.abs(b))))
    lo = (np.sum(b <= tol) + 1.0) / (b.size + 1.0)
    hi = (np.sum(b >= -tol) + 1.0) / (b.size + 1.0)
    return float(min(1.0, 2.0 * min(lo, hi)))


_boot_p = boot_p


def _boot_rows(d: pd.DataFrame, n_boot: int, seed: int, strata: str | None):
    """Yield row-position arrays for a sequence cluster bootstrap (stratified by ``strata``)."""

    seq_rows, groups = clustered._boot_plan(d, clustered.CLUSTER, strata)
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield clustered._boot_indices(seq_rows, groups, rng)


def firth_block(
    d: pd.DataFrame,
    outcome: str,
    rhs: str,
    *,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    strata: str | None = "condition",
    separated_terms: tuple[str, ...] | list[str] = (),
) -> dict:
    """Firth odds ratios for every term of ``outcome ~ rhs`` with cluster-bootstrap CIs.

    ``separated_terms`` names the terms whose contrast involves a level with a constant
    outcome; they are flagged ``involves_separated_level`` (see the caveat in the note).
    """

    d = d.dropna(subset=[outcome]).reset_index(drop=True)
    y, Xdf = _design(d, outcome, rhs)
    if len(y) != len(d):
        return {"note": "rows with missing covariates; Firth block not computed"}
    X = Xdf.to_numpy(dtype=float)
    names = list(Xdf.columns)
    fit = firth_logit(X, y)
    beta = fit["beta"]
    draws = np.full((n_boot, len(names)), np.nan)
    n_conv = 0
    strata_key = strata if (strata and strata in d.columns) else None
    for b, rows in enumerate(_boot_rows(d, n_boot, seed, strata_key)):
        r = firth_logit(X[rows], y[rows])
        if r["converged"]:
            draws[b] = r["beta"]
            n_conv += 1
    terms: dict = {}
    for j, nm in enumerate(names):
        if nm == "Intercept":
            continue
        col = draws[:, j]
        col = col[np.isfinite(col)]
        ci = (
            [float(np.exp(np.quantile(col, 0.025))), float(np.exp(np.quantile(col, 0.975)))]
            if col.size
            else None
        )
        terms[nm] = {
            "odds_ratio": float(np.exp(beta[j])),
            "log_odds_ratio": float(beta[j]),
            "ci95": ci,
            "p": _boot_p(col),
            "bootstrap_se_log": float(np.std(col, ddof=1)) if col.size > 1 else None,
            "involves_separated_level": nm in set(separated_terms),
        }
    return {
        "method": "firth_logit_cluster_bootstrap",
        "formula": f"{outcome} ~ {rhs}",
        "terms": terms,
        "converged": bool(fit["converged"]),
        "iterations": int(fit["iterations"]),
        "n": int(len(y)),
        "n_boot": int(n_boot),
        "n_boot_converged": int(n_conv),
        "seed": int(seed),
        "bootstrap_strata": strata_key,
        "note": _FIRTH_NOTE,
    }


# --------------------------------------------------------------------------- #
# Proportions, differences and ratios (identified at 0% / 100%)
# --------------------------------------------------------------------------- #


def _pct_ci(draws: np.ndarray) -> list[float] | None:
    b = draws[np.isfinite(draws)]
    if not b.size:
        return None
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def _exact_ci(k: int, n: int) -> list[float]:
    """Two-sided 95% Clopper-Pearson interval for k successes in n."""

    from scipy.stats import beta

    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return [lo, hi]


def _constant_arm(y: np.ndarray, seq: np.ndarray) -> dict | None:
    """Exact bounds for an arm whose outcome is the same in every trial; None otherwise."""

    if not len(y) or np.nanmin(y) != np.nanmax(y):
        return None
    rate = float(y[0])
    n, k_seq = int(len(y)), int(len(pd.unique(seq)))
    hit = int(round(rate))
    return {
        "rate": rate,
        "n_trials": n,
        "n_sequences": k_seq,
        "exact_ci95_trial_level": _exact_ci(hit * n, n),
        "exact_ci95_sequence_level": _exact_ci(hit * k_seq, k_seq),
    }


def risk_contrast_static_vs_mtd(
    d: pd.DataFrame,
    outcome: str,
    *,
    exposure: str = "is_mtd",
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict:
    """Compromise proportion static vs pooled MTD: difference (MTD - static) and ratio."""

    d = clustered.with_exposure(d.dropna(subset=[outcome]), exposure).reset_index(drop=True)
    y = pd.to_numeric(d[outcome], errors="coerce").to_numpy(dtype=float)
    x = pd.to_numeric(d[exposure], errors="coerce").to_numpy(dtype=float)
    if not (x == 0).any() or not (x == 1).any():
        return {"note": "needs both a static and an MTD arm"}

    def stats_for(rows: np.ndarray) -> tuple[float, float, float, float]:
        yy, xx = y[rows], x[rows]
        p0 = float(yy[xx == 0].mean())
        p1 = float(yy[xx == 1].mean())
        return p0, p1, p1 - p0, (p1 / p0) if p0 > 0 else float("nan")

    allrows = np.arange(len(d))
    p0, p1, rd, rr = stats_for(allrows)
    strata = "condition" if "condition" in d.columns else None
    boot = np.array([stats_for(r) for r in _boot_rows(d, n_boot, seed, strata)])
    seq = d["sequence_id"].astype(str).to_numpy()
    constant = {
        arm: c
        for arm, mask in (("static", x == 0), ("mtd", x == 1))
        if (c := _constant_arm(y[mask], seq[mask])) is not None
    }
    return {
        "proportion_static": p0,
        "proportion_static_ci95": _pct_ci(boot[:, 0]),
        "proportion_mtd": p1,
        "proportion_mtd_ci95": _pct_ci(boot[:, 1]),
        "risk_difference_mtd_minus_static": rd,
        "risk_difference_ci95": _pct_ci(boot[:, 2]),
        "risk_difference_p": _boot_p(boot[:, 2]),
        "risk_ratio_mtd_over_static": rr,
        "risk_ratio_ci95": _pct_ci(boot[:, 3]),
        "orientation": (
            "negative difference / ratio below 1 = MTD lowers the probability of compromise"
        ),
        "constant_arms": constant,
        "n_boot": int(n_boot),
        "seed": int(seed),
        "note": _RISK_NOTE,
    }


def technique_rates(
    d: pd.DataFrame,
    outcome: str,
    factor: str,
    *,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    reference: str | None = None,
) -> dict:
    """Per-level compromise proportions and their differences from the reference level.

    By default the reference is the first level in sorted order -- the same level patsy uses
    as the treatment-coding reference in the odds-ratio models -- so each difference lines up
    with the matching ``C(factor)[T.level]`` odds ratio. ``reference`` names another level
    (RQ1 compares every condition with the static baseline). Each difference carries both a
    Benjamini-Hochberg (``fdr_differences``) and a Holm (``p_holm``) adjustment.
    """

    from analysis.fdr import benjamini_hochberg, holm

    d = d.dropna(subset=[outcome, factor]).reset_index(drop=True)
    y = pd.to_numeric(d[outcome], errors="coerce").to_numpy(dtype=float)
    lv = d[factor].astype(str).to_numpy()
    levels = sorted(pd.unique(lv).tolist())
    if len(levels) < 2:
        return {"note": f"needs at least two levels of {factor}"}
    if reference is not None:
        if reference not in levels:
            return {"note": f"reference level {reference!r} absent from {factor}"}
        levels = [reference, *[lev for lev in levels if lev != reference]]
    ref = levels[0]

    def props(rows: np.ndarray) -> np.ndarray:
        yy, ll = y[rows], lv[rows]
        return np.array([yy[ll == lev].mean() if (ll == lev).any() else np.nan for lev in levels])

    point = props(np.arange(len(d)))
    strata = "condition" if "condition" in d.columns else factor
    boot = np.array([props(r) for r in _boot_rows(d, n_boot, seed, strata)])
    rates = {
        lev: {"proportion": float(point[i]), "ci95": _pct_ci(boot[:, i])}
        for i, lev in enumerate(levels)
    }
    diffs: dict = {}
    for i, lev in enumerate(levels[1:], start=1):
        draws = boot[:, i] - boot[:, 0]
        diffs[f"{lev} - {ref}"] = {
            "difference": float(point[i] - point[0]),
            "ci95": _pct_ci(draws),
            "p": _boot_p(draws),
        }
    fdr = benjamini_hochberg(
        {k: v["p"] for k, v in diffs.items() if v["p"] is not None and np.isfinite(v["p"])}
    )
    for k, adj in holm({k: v["p"] for k, v in diffs.items()}).items():
        diffs[k]["p_holm"] = adj
    seq = d["sequence_id"].astype(str).to_numpy()
    constant = {
        lev: c for lev in levels if (c := _constant_arm(y[lv == lev], seq[lv == lev])) is not None
    }
    return {
        "reference_level": ref,
        "proportions": rates,
        "differences_vs_reference": diffs,
        "fdr_differences": fdr,
        "constant_arms": constant,
        "n_boot": int(n_boot),
        "seed": int(seed),
        "note": _RISK_NOTE,
    }


#: The four conditions whose MTD moves the genuine services (the deceptive presentation
#: rotates decoys only, so its genuine services never go stale).
MOVING_TECHNIQUES = (
    "combined_mtd_deception",
    "endpoint_mutation",
    "network_shuffling",
    "port_rotation",
)

_PAIRWISE_NOTE = (
    "Every pairwise difference in compromise proportion among the listed conditions "
    "(first - second, both in sorted order), from the same sequence cluster bootstrap as the "
    "other proportions (whole sequences resampled within condition). ci95 is the per-pair "
    "percentile interval; ci_simultaneous is the Bonferroni-adjusted percentile interval at "
    "1 - alpha / m (m = the number of pairs), so all m intervals hold together at 1 - alpha; "
    "p is the two-sided bootstrap p (+1 correction, floored at 2/(B + 1)) and p_holm its Holm "
    "adjustment over the m pairs. largest_abs_simultaneous_bound is the largest absolute end "
    "of any simultaneous interval: every pairwise difference the data allow, at the family "
    "level, lies within plus or minus that value. Whether that range counts as a 'small' "
    "difference needs a margin that Chapter 3 does not state; none is applied here. The "
    "simultaneous interval sits far in the bootstrap tails (about alpha / (2m) of B replicates "
    "on each side), so its ends are less stable than the 95% interval's."
)


def pairwise_rate_differences(
    d: pd.DataFrame,
    outcome: str,
    factor: str,
    levels: tuple[str, ...] | list[str],
    *,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    alpha: float = 0.05,
) -> dict:
    """All pairwise differences in proportion among ``levels`` of ``factor``.

    Risk differences with per-pair 95% and Bonferroni-simultaneous intervals, bootstrap p and
    Holm-adjusted p (sequence cluster bootstrap stratified by condition). Levels that are
    absent from ``d`` are listed in ``levels_absent`` and left out of the family.
    """

    import itertools

    from analysis.fdr import holm

    if outcome not in d.columns or factor not in d.columns:
        return {"note": f"{outcome} or {factor} not in the dataset"}
    lab = d[factor].astype(str)
    present = sorted(lev for lev in set(levels) if (lab == lev).any())
    absent = sorted(set(levels) - set(present))
    sub = d.loc[lab.isin(present)].dropna(subset=[outcome]).reset_index(drop=True)
    if len(present) < 2:
        return {"levels_absent": absent, "note": "needs at least two of the listed levels"}
    y = pd.to_numeric(sub[outcome], errors="coerce").to_numpy(dtype=float)
    lv = sub[factor].astype(str).to_numpy()

    def props(rows: np.ndarray) -> np.ndarray:
        yy, ll = y[rows], lv[rows]
        return np.array([yy[ll == lev].mean() if (ll == lev).any() else np.nan for lev in present])

    point = props(np.arange(len(sub)))
    strata = "condition" if "condition" in sub.columns else factor
    boot = np.array([props(r) for r in _boot_rows(sub, n_boot, seed, strata)])
    pairs = list(itertools.combinations(range(len(present)), 2))
    m = len(pairs)
    q = alpha / m
    out_pairs: dict = {}
    for i, j in pairs:
        draws = boot[:, i] - boot[:, j]
        fin = draws[np.isfinite(draws)]
        sim = (
            [float(np.quantile(fin, q / 2)), float(np.quantile(fin, 1 - q / 2))]
            if fin.size
            else None
        )
        out_pairs[f"{present[i]} - {present[j]}"] = {
            "difference": float(point[i] - point[j]),
            "ci95": _pct_ci(draws),
            "ci_simultaneous": sim,
            "p": _boot_p(draws),
        }
    for k, adj in holm({k: v["p"] for k, v in out_pairs.items()}).items():
        out_pairs[k]["p_holm"] = adj
    bounds = [
        abs(b) for v in out_pairs.values() if v["ci_simultaneous"] for b in v["ci_simultaneous"]
    ]
    seq = sub["sequence_id"].astype(str).to_numpy()
    return {
        "outcome": outcome,
        "levels": present,
        "levels_absent": absent,
        "proportions": {
            lev: {"proportion": float(point[i]), "ci95": _pct_ci(boot[:, i])}
            for i, lev in enumerate(present)
        },
        "n_pairs": m,
        "alpha": float(alpha),
        "simultaneous_level": float(1 - alpha),
        "per_pair_alpha_bonferroni": float(q),
        "pairs": out_pairs,
        "largest_abs_simultaneous_bound": float(max(bounds)) if bounds else None,
        "constant_arms": {
            lev: c
            for lev in present
            if (c := _constant_arm(y[lv == lev], seq[lv == lev])) is not None
        },
        "n_boot": int(n_boot),
        "seed": int(seed),
        "note": _PAIRWISE_NOTE,
    }


# --------------------------------------------------------------------------- #
# How much does the VB GLMM depend on its prior?
# --------------------------------------------------------------------------- #


def vb_prior_sensitivity(
    d: pd.DataFrame,
    outcome: str,
    rhs: str,
    *,
    group: str = clustered.CLUSTER,
    prior_sds=PRIOR_SDS,
) -> dict:
    """Refit the pipeline's VB GLMM at several fixed-effect prior SDs; odds ratio per term."""

    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    d = d.dropna(subset=[outcome])
    by_sd: dict[str, dict] = {}
    for sd in prior_sds:
        model = BinomialBayesMixedGLM.from_formula(
            f"{outcome} ~ {rhs}", {"cluster": f"0 + C({group})"}, d, fe_p=float(sd)
        )
        state = np.random.get_state()
        np.random.seed(0)  # same reproducible initialization as clustered._glmm_terms
        try:
            res = model.fit_vb()
        finally:
            np.random.set_state(state)
        names = list(res.model.exog_names)
        by_sd[f"{sd:g}"] = {
            nm: float(np.exp(res.fe_mean[i])) for i, nm in enumerate(names) if nm != "Intercept"
        }
    terms = sorted({t for v in by_sd.values() for t in v})
    fold: dict[str, float] = {}
    for t in terms:
        vals = [v[t] for v in by_sd.values() if t in v and v[t] > 0]
        fold[t] = float(max(vals) / min(vals)) if vals else float("nan")
    dominated = sorted(t for t, f in fold.items() if np.isfinite(f) and f > PRIOR_FOLD_FLAG)
    return {
        "prior_sd_values": [float(s) for s in prior_sds],
        "default_prior_sd": DEFAULT_PRIOR_SD,
        "odds_ratio_by_prior_sd": by_sd,
        "max_over_min_fold_change": fold,
        "prior_dominated_terms": dominated,
        "note": _VB_NOTE,
    }


def separation_robust_block(
    d: pd.DataFrame,
    outcome: str,
    rhs: str,
    factor: str,
    *,
    kind: str,
    separated_terms: tuple[str, ...] | list[str] = (),
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict:
    """Everything a separated binary comparison needs, in one block.

    ``kind`` is ``"static_vs_mtd"`` (RQ1) or ``"techniques"`` (RQ2).
    """

    out: dict = {
        "firth": firth_block(
            d, outcome, rhs, n_boot=n_boot, seed=seed, separated_terms=separated_terms
        ),
        "vb_prior_sensitivity": vb_prior_sensitivity(d, outcome, rhs),
    }
    if kind == "static_vs_mtd":
        out["risk_contrast"] = risk_contrast_static_vs_mtd(d, outcome, n_boot=n_boot, seed=seed)
    else:
        out["rates"] = technique_rates(d, outcome, factor, n_boot=n_boot, seed=seed)
    out["headline_guidance"] = (
        "Lead with the compromise proportions and their difference (risk_contrast / rates) and "
        "the per-sequence comparison. Terms flagged involves_separated_level have no "
        "data-identified odds ratio: do not report their VB odds ratio (it moves with the "
        "prior; see vb_prior_sensitivity.prior_dominated_terms), and give their Firth value, if "
        "at all, only with the caveat in firth.note. Unflagged terms are data-identified odds "
        "ratios, but the two fits estimate slightly different things: the VB GLMM's sequence "
        "intercepts absorb always-compromised sequences, while the Firth model (no random "
        "intercept) averages over them, so a slope such as attempt_index can differ between the "
        "two when a constant arm is pooled into an exposure level. For the attempt effect itself "
        "cite rq3.<outcome>.attempt_effects."
    )
    return out


def separated_terms_for_factor(profile: dict, factor: str) -> list[str]:
    """Treatment-coded term names (``C(factor)[T.level]``) whose contrast hits a separated level.

    The reference is the first level in sorted order (patsy's default). If the reference
    itself is separated, every term contrasts against it.
    """

    levels = sorted((profile.get("levels") or {}).keys())
    if not levels:
        return []
    ref = levels[0]
    sep = set(profile.get("separated_levels") or [])
    return [f"C({factor})[T.{lev}]" for lev in levels[1:] if (lev in sep or ref in sep)]
