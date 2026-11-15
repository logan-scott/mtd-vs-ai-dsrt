"""RQ4 (H04/Ha4): which metrics most effectively and non-redundantly capture
defensive efficacy and operational trade-offs?

A defined, multi-step parsimony procedure (not informal inference):
1. Pearson (Spearman fallback) correlation matrix + **variance inflation factors**
   screen for collinear metric clusters.
2. **PCA** of the standardized metrics identifies the underlying dimensions and the
   metrics that load on each.
3. **Regularized selection** (LASSO / elastic net) predicting the primary efficacy
   outcomes chooses a parsimonious, non-redundant subset.

Cronbach's alpha is applied ONLY to the operational-overhead composite (formative
efficacy indicators are examined for redundancy here, not via alpha).

RE-RUN ANALYSIS UPDATE (2026-10-28; ``docs/rerun_runbook.md`` §6). H04 is decided with the
correlation tests (runbook §1). ``h04_correlation_tests`` reports them in their own block:
the correlations among the trial-level security-efficacy measures and between those measures
and the per-trial overhead measures that were actually measured, each with a sequence cluster
bootstrap interval and p and Holm adjustment over the pairs that can carry evidence; pairs
that are related by construction or by a shared definition, and measures that the design
fixes per condition, are flagged and kept out of that family. Operational cost is measured
per technique (the reconfiguration benchmark), so security-vs-cost tests run across the five
MTD techniques and carry the five-technique caveat. The parsimony procedure (VIF / PCA /
regularized selection) stays where it was, as a separate block.

The decision rule (2026-10-28, approved by the chair; ``_H04_DECISION_NOTE``) is computed in
``h04_correlation_tests(...)["decision"]``: the first clause from the pooled Pearson
correlations with Holm over the trial-level family (Spearman alongside, within-condition as
the sensitivity); the second clause from the technique-level Pearson tests of the compromise
proportion and the 300-s RMST benefit against the median benchmarked cost, Holm across the two
(exact Spearman alongside, the trial-window benefit as a sensitivity).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from analysis.reliability import OVERHEAD_COMPOSITE, cronbach_alpha, item_missingness

DISRUPTION = [
    "time_to_compromise_s",
    "time_to_first_valid_exploit_s",
    "recon_accuracy_f1",
    "attack_path_churn",
]
OVERHEAD = [
    "latency_overhead_ms",
    "cpu_overhead_pct",
    "memory_overhead_pct",
    "network_overhead_pct",
    "operational_event_volume",
    "service_availability_impact",
]
SECURITY = ["exploit_success", "exploitation_success"]
METRIC_SET = DISRUPTION + OVERHEAD  # standardized metric matrix for VIF / PCA
# The trial-level SECURITY measures without the overhead columns: the matrix behind the
# dissertation's RQ4 trial-level VIF/PCA (n = 1,386 complete cases in the September run).
SECURITY_MEASURE_SET = ["exploitation_success", *DISRUPTION]

_LATENCY_MISSING_REASON = (
    "it is recorded only when the service was reachable to probe, so it is missing on most MTD "
    "trials; the missingness is caused by the treatment and is therefore NOT missing at random"
)

# Declared direction of every metric in the correlation / PCA matrix. VIF, |loadings| and
# regularized selection are sign-invariant, so no column is re-signed or renamed here (the
# dataset column names are the measurement contract); the orientation is emitted so a reader
# can sign the correlations and loadings correctly. ``service_availability_impact`` is the
# one that reads backwards: it stores availability RETAINED (metrics_specification 3.4).
METRIC_ORIENTATION = {
    "exploit_success": "higher_is_worse_for_the_defender",
    "exploitation_success": "higher_is_worse_for_the_defender",
    "time_to_compromise_s": "higher_is_better_for_the_defender",
    "time_to_first_valid_exploit_s": "higher_is_better_for_the_defender",
    "recon_accuracy_f1": "higher_is_worse_for_the_defender",
    "attack_path_churn": "higher_is_better_for_the_defender",
    "latency_overhead_ms": "higher_is_worse_for_the_defender",
    "cpu_overhead_pct": "higher_is_worse_for_the_defender",
    "memory_overhead_pct": "higher_is_worse_for_the_defender",
    "network_overhead_pct": "higher_is_worse_for_the_defender",
    "operational_event_volume": "higher_is_worse_for_the_defender",
    "service_availability_impact": "higher_is_better_for_the_defender",
}


def _numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    # keep columns with >=2 non-null values and non-zero variance
    keep = [c for c in cols if sub[c].notna().sum() >= 2 and sub[c].std(ddof=0) > 0]
    return sub[keep]


def _corr(df: pd.DataFrame, cols: list[str], method: str) -> dict:
    sub = _numeric(df, cols)
    return sub.corr(method=method).round(3).to_dict()


def _vif(df: pd.DataFrame, cols: list[str]) -> dict:
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    sub = _numeric(df, cols).dropna()
    if sub.shape[0] <= sub.shape[1] + 1 or sub.shape[1] < 2:
        return {"note": "insufficient data for VIF"}
    z = (sub - sub.mean()) / sub.std(ddof=0)
    X = np.column_stack([np.ones(len(z)), z.to_numpy()])
    vifs: dict[str, float | None] = {}
    perfect: list[str] = []
    for i, c in enumerate(z.columns, start=1):
        try:
            v = float(variance_inflation_factor(X, i))
        except Exception:  # noqa: BLE001
            vifs[c] = None
            continue
        # A non-finite VIF means R^2 == 1: the indicator is an exact linear combination of
        # the others. json.dumps would write that as the non-standard `Infinity` token, so
        # it is reported as null plus an explicit perfect_collinearity list.
        if math.isfinite(v):
            vifs[c] = v
        else:
            vifs[c] = None
            perfect.append(c)
    high = [c for c, v in vifs.items() if v is not None and v > 5] + perfect
    return {
        "vif": vifs,
        "high_collinearity_gt5": high,
        "perfect_collinearity": perfect,
        "vif_note": (
            "vif = null means the factor was not computable; the indicator is listed in "
            "perfect_collinearity when the null is an exact linear dependency (R^2 = 1, VIF "
            "infinite) rather than a solver failure."
        ),
    }


def _pca(df: pd.DataFrame, cols: list[str]) -> dict:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    sub = _numeric(df, cols).dropna()
    if sub.shape[0] < 3 or sub.shape[1] < 2:
        return {"note": "insufficient data for PCA"}
    Z = StandardScaler().fit_transform(sub.to_numpy())
    pca = PCA().fit(Z)
    evr = pca.explained_variance_ratio_
    n90 = int(np.searchsorted(np.cumsum(evr), 0.90) + 1)
    cols = list(sub.columns)
    # loadings: components x metrics
    loadings = {
        f"PC{i + 1}": {c: round(float(pca.components_[i][j]), 3) for j, c in enumerate(cols)}
        for i in range(min(len(evr), 3))
    }
    # dominant (highest-|loading|) metric on each retained component = one metric per
    # distinct dimension (the "loads on separate components" set for the parsimony rule).
    distinct: list[str] = []
    for i in range(min(n90, len(evr))):
        dom = cols[int(np.argmax(np.abs(pca.components_[i])))]
        if dom not in distinct:
            distinct.append(dom)
    return {
        "explained_variance_ratio": [round(float(x), 3) for x in evr],
        "n_components_90pct": n90,
        "loadings_top3": loadings,
        "distinct_component_metrics": distinct,
    }


def _elastic_net_select(df: pd.DataFrame, target: str, predictors: list[str]) -> dict:
    from sklearn.linear_model import ElasticNetCV
    from sklearn.preprocessing import StandardScaler

    preds = [p for p in predictors if p != target]
    d = _numeric(df, [target, *preds]).dropna()
    preds = [p for p in preds if p in d.columns]
    # target dropped by _numeric == zero variance; check membership before .std().
    if target not in d.columns or len(d) < 10 or not preds:
        return {
            "note": "insufficient data / no variance in target for regularized selection",
            "target": target,
        }
    X = StandardScaler().fit_transform(d[preds].to_numpy())
    y = (d[target] - d[target].mean()) / d[target].std(ddof=0)
    try:
        cv = max(2, min(5, len(d) // 2))
        # random_state pins the CV/coordinate-descent RNG so the selection is reproducible.
        model = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0], cv=cv, max_iter=5000, random_state=0)
        model.fit(X, y.to_numpy())
        coefs = {p: float(c) for p, c in zip(preds, model.coef_, strict=True)}
        selected = [p for p, c in coefs.items() if abs(c) > 1e-6]
        return {
            "target": target,
            "method": "elastic_net_cv",
            "alpha": float(model.alpha_),
            "l1_ratio": float(model.l1_ratio_),
            "coefficients": {k: round(v, 4) for k, v in coefs.items()},
            "selected_subset": selected,
        }
    except Exception as exc:  # noqa: BLE001
        return {"note": f"regularized selection unavailable: {exc}", "target": target}


def _l1_logistic_select(df: pd.DataFrame, target: str, predictors: list[str]) -> dict:
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler

    preds = [p for p in predictors if p != target]
    d = _numeric(df, preds).copy()
    d[target] = pd.to_numeric(df.loc[d.index, target], errors="coerce")
    d = d.dropna()
    preds = [p for p in preds if p in d.columns]
    if len(d) < 10 or d[target].nunique() < 2 or not preds:
        return {"note": "insufficient data / no variance for L1 logistic", "target": target}
    X = StandardScaler().fit_transform(d[preds].to_numpy())
    y = d[target].astype(int).to_numpy()
    min_class = int(pd.Series(y).value_counts().min())
    try:
        # Cross-validate the penalty C (mirrors the ElasticNetCV treatment); fall back
        # to a fixed penalty if the class support is too small to cross-validate.
        if min_class >= 3:
            cv = max(2, min(5, min_class))
            model = LogisticRegressionCV(
                penalty="l1",
                solver="liblinear",
                cv=cv,
                Cs=10,
                max_iter=3000,
                scoring="neg_log_loss",
                random_state=0,  # liblinear shuffles coordinates; pin it for reproducibility
            ).fit(X, y)
            method = "l1_logistic_cv"
        else:
            model = LogisticRegression(
                penalty="l1", solver="liblinear", C=1.0, max_iter=3000, random_state=0
            ).fit(X, y)
            method = "l1_logistic_fixedC (insufficient class support to CV)"
        coefs = {p: float(c) for p, c in zip(preds, model.coef_[0], strict=True)}
        selected = [p for p, c in coefs.items() if abs(c) > 1e-6]
        return {
            "target": target,
            "method": method,
            "coefficients": {k: round(v, 4) for k, v in coefs.items()},
            "selected_subset": selected,
        }
    except Exception as exc:  # noqa: BLE001
        return {"note": f"L1 logistic unavailable: {exc}", "target": target}


def security_measure_parsimony(df: pd.DataFrame) -> dict:
    """VIF and PCA of the five trial-level security measures, without the overhead columns.

    The dissertation's RQ4 trial-level parsimony figures were computed on this matrix, which
    no report archived until 2026-10-27 (the ``vif``/``pca`` blocks of ``rq4`` use
    ``METRIC_SET``, whose overhead columns were produced by the offline telemetry stand-in in
    the archived runs). Same estimators as those blocks (``_vif``: standardized columns plus an
    intercept; ``_pca``: standardized PCA), listwise deletion over the five measures.
    ``attack_path_churn`` compares an attempt with the previous one, so it is undefined at
    ``attempt_index`` 0; ``missing_at_attempt_index_0`` says whether that is the only
    missingness.
    """

    cols = [c for c in SECURITY_MEASURE_SET if c in df.columns]
    sub = _numeric(df, cols)
    complete = sub.dropna()
    missing = {c: int(sub[c].isna().sum()) for c in sub.columns}
    out: dict = {
        "metrics": list(sub.columns),
        "metrics_dropped_by_variance_screen": [c for c in cols if c not in sub.columns],
        "n_trials": int(len(df)),
        "n_complete_cases": int(len(complete)),
        "missing_by_metric": missing,
    }
    if "attempt_index" in df.columns:
        rows_missing = sub.isna().any(axis=1)
        at0 = pd.to_numeric(df["attempt_index"], errors="coerce") == 0
        out["missing_at_attempt_index_0"] = int((rows_missing & at0).sum())
        out["missing_elsewhere"] = int((rows_missing & ~at0).sum())
    out["vif"] = _vif(df, cols)
    out["pca"] = _pca(df, cols)
    out["note"] = (
        "Trial-level security measures only (exploitation_success + the disruption metrics); "
        "VIF and PCA here are descriptive parsimony diagnostics, not significance tests. "
        "exploitation_success and time_to_compromise_s are near-collinear by construction "
        "(a censored trial is both a non-compromise and a time at the horizon)."
    )
    return out


# --------------------------------------------------------------------------- #
# H04: the correlation tests (re-run analysis update, 2026-10-28)
# --------------------------------------------------------------------------- #

H04_STATEMENT = (
    "There will be no statistically significant associations among the security efficacy and "
    "operational overhead measures, and operational overhead measures will not statistically "
    "significantly predict defensive efficacy."
)
#: Trial-level security-efficacy measures (the primary binary outcomes and the disruption set).
H04_SECURITY = [*SECURITY, *DISRUPTION]
#: The per-trial cost columns governed by ``operational_cost_source`` (metrics spec §3): they
#: are null in a not-measured live run and synthetic in every archived run, so they enter the
#: tests only when every row records them as measured.
GATED_COST_COLUMNS = (
    "latency_overhead_ms",
    "cpu_overhead_pct",
    "memory_overhead_pct",
    "network_overhead_pct",
    "service_availability_impact",
)
#: Measures the design fixes per condition: an association with one of them restates the
#: between-condition contrast rather than a covariation of two measured quantities.
DESIGN_DETERMINED = {
    "operational_event_volume": (
        "counts the modeled reconfiguration events in the trial window, which the condition's "
        "cadence fixes under Method C (0 for static_baseline by construction; metrics spec "
        "§3.3a): a property of the schedule, not a measured cost"
    ),
}
#: Pairs related by construction or by a shared definition (metrics spec §1-§2). Their
#: correlation is reported but is not evidence of an empirical association.
STRUCTURAL_PAIRS: dict[frozenset, tuple[str, str]] = {
    frozenset({"exploitation_success", "time_to_compromise_s"}): (
        "by_construction",
        "both record obj_b: a trial without obj_b is censored at the trial window, so the two "
        "correlate close to -1 by construction",
    ),
    frozenset({"exploit_success", "exploitation_success"}): (
        "shared_definition",
        "exploitation_success is obj_b, one of the objectives exploit_success counts "
        "(obj_a OR obj_b OR obj_c)",
    ),
    frozenset({"exploit_success", "time_to_compromise_s"}): (
        "shared_definition",
        "an observed time to compromise is an obj_b event, which sets exploit_success to 1",
    ),
    frozenset({"exploitation_success", "time_to_first_valid_exploit_s"}): (
        "shared_definition",
        "an obj_b compromise is itself a valid exploit, so exploitation_success = 1 implies an "
        "observed TFVE no later than the compromise",
    ),
    frozenset({"time_to_compromise_s", "time_to_first_valid_exploit_s"}): (
        "shared_definition",
        "same origin (the first reconnaissance command) and nested events: TFVE <= TTC "
        "whenever both are observed, and the first valid exploit is often the compromise",
    ),
    frozenset({"recon_accuracy_f1", "attack_path_churn"}): (
        "shared_definition",
        "both are computed from the attempt's recon snapshots (the same service-port-endpoint-"
        "address tuple sets): in the re-run each falls only when a later scan adds identities "
        "the services took after a reconfiguration (metrics spec §2.3-§2.4)",
    ),
}

_H04_TRIAL_NOTE = (
    "Trial-level correlations: pearson (point-biserial for a binary measure) and spearman over "
    "the trials where both measures are recorded (attack_path_churn is undefined at "
    "attempt_index 0). within_condition_pearson is the Pearson correlation of the "
    "condition-centered values: the association left after the condition contrast is removed "
    "(it is undefined for a measure that is constant within every condition). Every 95% "
    "interval and p is a PERCENTILE SEQUENCE CLUSTER BOOTSTRAP (whole sequences resampled "
    "within condition; two-sided p with the +1 correction, floored at 2/(B + 1)), because the "
    "8 attempts of a sequence are not independent. p_holm adjusts each statistic over "
    "holm_family: the pairs that are neither structural nor involve a design-determined "
    "measure. Structural and design-determined pairs are reported with their reason and are "
    "not evidence of an empirical association."
)
_FIVE_TECHNIQUE_CAVEAT = (
    "FIVE-TECHNIQUE CAVEAT. Operational cost is measured per technique by the reconfiguration "
    "benchmark (metrics spec §3), so a security-vs-cost association can be tested only across "
    "the five MTD techniques: each correlation rests on 5 points. Pearson's r then has 3 df "
    "and needs |r| > .878 for p < .05 (two-sided); the smallest exact two-sided p Spearman's "
    "rho can reach is 2/120 = .017. Only a near-perfect association can be detected, so a "
    "non-significant result is not evidence of no association, and a significant one can turn "
    "on a single technique. The benchmark's replicates make each technique's cost precise; they "
    "do not add observations of the association. Each technique's security summary is a point "
    "estimate whose own sampling error the 5-point test ignores. The security summaries are "
    "not independent of each other (the RMST benefits restate the compromise proportion: "
    "benefit is close to horizon x the share of trials never compromised), so their tests are "
    "near-duplicates rather than separate evidence."
)
#: The date the H04 decision rule was set (approved by the chair; not pre-registered).
H04_RULE_DATE = "2026-10-28"
H04_ALPHA = 0.05


def second_clause_summaries(horizon_s: float = 300.0) -> tuple[str, str]:
    """The technique-level security summaries that decide H04's second clause (Holm across them).

    The compromise proportion and the RMST benefit at the time-summary horizon. The
    trial-window RMST benefit is reported beside them as a sensitivity, outside the family.
    """

    return ("exploitation_proportion", f"rmst_benefit_{int(horizon_s)}s")


#: Which way each technique-level security summary points.
SUMMARY_ORIENTATION = {
    "exploitation_proportion": "share of trials compromised: LOWER = more effective defense",
    "rmst_benefit": "RMST(technique) - RMST(static_baseline), s: HIGHER = more effective defense",
}

_H04_DECISION_NOTE = (
    "DECISION RULE (decided 2026-10-28 and approved by the chair; not pre-registered). H04 has "
    "two clauses, each decided at alpha = .05. FIRST CLAUSE (no associations among the security "
    "efficacy and operational overhead measures): rejected when any pair in "
    "trial_level.holm_family has a pooled Pearson p_holm < .05. The family leaves out pairs "
    "related by construction or by a shared definition, and every pair with a design-determined "
    "measure (operational_event_volume, which the condition's cadence fixes); in a run whose "
    "per-trial cost columns are not measured it therefore holds security-security pairs only. "
    "Spearman is reported alongside, and the within-condition Pearson correlation (what is left "
    "after the condition contrast is removed) is the sensitivity analysis. SECOND CLAUSE "
    "(operational overhead does not predict defensive efficacy): rejected when either "
    "technique-level Pearson test in technique_level.holm_family -- each technique's compromise "
    "proportion, and its 300-s RMST benefit over the static baseline, against its median "
    "benchmarked reconfiguration cost, n = 5 techniques -- has p_holm < .05 (Holm across the "
    "two). With one predictor the Pearson test is the slope test of efficacy regressed on cost "
    "(the same t and p). The exact Spearman test is reported alongside; the trial-window RMST "
    "benefit is a sensitivity outside the family. A clause that is not rejected is reported as "
    "'no association detected', not as evidence of independence (see the five-technique "
    "caveat). The overall line restates the two clauses: H04 asserts both, so it is rejected "
    "when either is; it is not a separate test, and under the joint null the chance that at "
    "least one clause is rejected can approach .10, so the write-up states the decision clause "
    "by clause."
)


def _pair_key(a: str, b: str) -> str:
    return f"{a} | {b}"


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 3:
        return None
    sx, sy = float(np.std(x)), float(np.std(y))
    if sx == 0.0 or sy == 0.0:
        return None
    return float(np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy))


def _center_by(v: np.ndarray, g: np.ndarray, k: int) -> np.ndarray:
    n = np.bincount(g, minlength=k)
    s = np.bincount(g, weights=v, minlength=k)
    m = np.divide(s, n, out=np.zeros(k), where=n > 0)
    return v - m[g]


def _pair_stats(x: np.ndarray, y: np.ndarray, g: np.ndarray, k: int) -> tuple:
    from scipy.stats import rankdata

    ok = np.isfinite(x) & np.isfinite(y)
    x, y, g = x[ok], y[ok], g[ok]
    if x.size < 3:
        return None, None, None, int(x.size)
    r = _pearson(x, y)
    rho = _pearson(rankdata(x), rankdata(y)) if r is not None else None
    rw = _pearson(_center_by(x, g, k), _center_by(y, g, k))
    if rw is not None and abs(rw) < 1e-12:
        rw = 0.0
    return r, rho, rw, int(x.size)


def _boot_summary(draws: list) -> tuple[list[float] | None, float | None, int]:
    from analysis.separation import boot_p

    b = np.array([np.nan if v is None else v for v in draws], dtype=float)
    fin = b[np.isfinite(b)]
    if not fin.size:
        return None, None, 0
    ci = [float(np.quantile(fin, 0.025)), float(np.quantile(fin, 0.975))]
    return ci, boot_p(fin), int(fin.size)


def _overhead_measures(df: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    """The per-trial overhead measures that may enter the tests, and why the others do not."""

    src = (
        df["operational_cost_source"].astype(str)
        if "operational_cost_source" in df.columns
        else None
    )
    used: list[str] = []
    excluded: dict[str, str] = {}
    for c in OVERHEAD:
        if c not in df.columns:
            excluded[c] = "not in the dataset"
        elif c in GATED_COST_COLUMNS and src is None:
            excluded[c] = (
                "the dataset does not record operational_cost_source; the archived runs' "
                "per-trial cost columns are synthetic (metrics spec §3), not measurements"
            )
        elif c in GATED_COST_COLUMNS and not (src == "measured").all():
            excluded[c] = (
                "not a measurement in this run (operational_cost_source: "
                f"{', '.join(sorted(set(src)))})"
            )
        else:
            used.append(c)
    return used, excluded


def _trial_level(df: pd.DataFrame, n_boot: int, seed: int) -> dict:
    import itertools

    from analysis.fdr import holm
    from analysis.separation import _boot_rows

    overhead, excluded = _overhead_measures(df)
    candidates = [c for c in H04_SECURITY if c in df.columns] + overhead
    screened = _numeric(df, candidates)
    for c in candidates:
        if c not in screened.columns:
            excluded[c] = "no variance or fewer than 2 recorded values (variance screen)"
    sec = [c for c in H04_SECURITY if c in screened.columns]
    ovh = [c for c in overhead if c in screened.columns]
    pairs = [("among_security", a, b) for a, b in itertools.combinations(sec, 2)]
    pairs += [("security_vs_overhead", a, b) for a in sec for b in ovh]
    d = df.reset_index(drop=True)
    X = {c: pd.to_numeric(d[c], errors="coerce").to_numpy(dtype=float) for c in sec + ovh}
    cond = d["condition"].astype(str).to_numpy()
    levels = sorted(set(cond))
    g = np.searchsorted(levels, cond)
    k = len(levels)

    eta2: dict[str, float | None] = {}
    for c in sec + ovh:
        v = X[c]
        ok = np.isfinite(v)
        tot = float(np.var(v[ok])) if ok.sum() > 1 else 0.0
        eta2[c] = None if tot == 0.0 else float(1.0 - np.var(_center_by(v[ok], g[ok], k)) / tot)

    point = {(a, b): _pair_stats(X[a], X[b], g, k) for _, a, b in pairs}
    boot: dict = {(a, b): ([], [], []) for _, a, b in pairs}
    if pairs and "sequence_id" in d.columns:
        for rows in _boot_rows(d, n_boot, seed, "condition"):
            for _, a, b in pairs:
                r, rho, rw, _n = _pair_stats(X[a][rows], X[b][rows], g[rows], k)
                boot[(a, b)][0].append(r)
                boot[(a, b)][1].append(rho)
                boot[(a, b)][2].append(rw)

    out_pairs: dict = {}
    for family, a, b in pairs:
        r, rho, rw, n = point[(a, b)]
        rec: dict = {"family": family, "n": n}
        for name, val, draws in (
            ("pearson", r, boot[(a, b)][0]),
            ("spearman", rho, boot[(a, b)][1]),
            ("within_condition_pearson", rw, boot[(a, b)][2]),
        ):
            ci, p, nb = _boot_summary(draws)
            rec[name] = {"estimate": val, "ci95": ci, "p": p, "n_boot_finite": nb}
        struct = STRUCTURAL_PAIRS.get(frozenset({a, b}))
        rec["structural"] = struct[0] if struct else None
        rec["structural_reason"] = struct[1] if struct else None
        dd = [m for m in (a, b) if m in DESIGN_DETERMINED]
        rec["design_determined_measures"] = dd
        rec["in_holm_family"] = struct is None and not dd
        out_pairs[_pair_key(a, b)] = rec

    family = [k_ for k_, v in out_pairs.items() if v["in_holm_family"]]
    for stat in ("pearson", "spearman", "within_condition_pearson"):
        adj = holm({k_: out_pairs[k_][stat]["p"] for k_ in family})
        for k_ in family:
            out_pairs[k_][stat]["p_holm"] = adj.get(k_)

    def count(stat: str, key: str, fam: str | None = None) -> int:
        return sum(
            1
            for k_ in family
            if (fam is None or out_pairs[k_]["family"] == fam)
            and (out_pairs[k_][stat].get(key) is not None)
            and out_pairs[k_][stat][key] < 0.05
        )

    decision_inputs = {
        "n_holm_family": len(family),
        "n_holm_family_security_vs_overhead": sum(
            1 for k_ in family if out_pairs[k_]["family"] == "security_vs_overhead"
        ),
    }
    for stat in ("pearson", "spearman", "within_condition_pearson"):
        for key in ("p", "p_holm"):
            decision_inputs[f"n_{stat}_{key}_lt_05"] = count(stat, key)
            decision_inputs[f"n_{stat}_{key}_lt_05_security_vs_overhead"] = count(
                stat, key, "security_vs_overhead"
            )
    return {
        "security_measures": sec,
        "overhead_measures": ovh,
        "measures_excluded": excluded,
        "design_determined": {m: DESIGN_DETERMINED[m] for m in ovh if m in DESIGN_DETERMINED},
        "eta_squared_condition": eta2,
        "pairs": out_pairs,
        "holm_family": family,
        "decision_inputs": decision_inputs,
        "metric_orientation": {m: METRIC_ORIENTATION.get(m) for m in sec + ovh},
        "n_trials": int(len(d)),
        "n_boot": int(n_boot),
        "bootstrap_seed": int(seed),
        "note": _H04_TRIAL_NOTE,
    }


def _exact_spearman_p(x: np.ndarray, y: np.ndarray) -> float | None:
    """Two-sided exact permutation p for Spearman's rho (small n only)."""

    import itertools

    from scipy.stats import rankdata

    n = x.size
    if n < 3 or n > 8:
        return None
    rx, ry = rankdata(x), rankdata(y)
    obs = _pearson(rx, ry)
    if obs is None:
        return None
    hits = total = 0
    for perm in itertools.permutations(range(n)):
        r = _pearson(rx, ry[list(perm)])
        total += 1
        if r is not None and abs(r) >= abs(obs) - 1e-12:
            hits += 1
    return float(hits / total)


def _technique_level(df: pd.DataFrame, bench: pd.DataFrame | None, horizon_s: float) -> dict:
    from scipy import stats as st

    from analysis import clustered
    from analysis.dataset import STATIC
    from analysis.fdr import holm

    if bench is None or not len(bench):
        return {
            "computed": False,
            "note": (
                "No reconfiguration-cost benchmark was supplied (cli.py analyze "
                "--reconfig-bench), so the security-vs-cost tests across techniques were not "
                "run. " + _FIVE_TECHNIQUE_CAVEAT
            ),
        }
    cost = (
        bench.assign(wall_clock_s=pd.to_numeric(bench["wall_clock_s"], errors="coerce"))
        .dropna(subset=["wall_clock_s"])
        .groupby("technique")["wall_clock_s"]
        .median()
    )
    conds = sorted(
        c for c in df["condition"].astype(str).unique() if c != STATIC and c in cost.index
    )
    if STATIC not in set(df["condition"].astype(str)) or len(conds) < 3:
        return {
            "computed": False,
            "techniques": conds,
            "note": "needs the static baseline and at least 3 benchmarked MTD techniques",
        }
    window = clustered.survival_horizon(df, "time_to_compromise_s")

    def rmst(cond: str, h: float) -> float:
        g = df[df["condition"].astype(str) == cond]
        t = pd.to_numeric(g["time_to_compromise_s"], errors="coerce").to_numpy(dtype=float)
        e = 1 - pd.to_numeric(g["ttc_censored"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(t) & np.isfinite(e)
        return clustered.km_rmst(t[ok], e[ok], h)

    security: dict[str, dict[str, float]] = {c: {} for c in conds}
    if "exploitation_success" in df.columns:
        for c in conds:
            y = pd.to_numeric(
                df.loc[df["condition"].astype(str) == c, "exploitation_success"], errors="coerce"
            )
            security[c]["exploitation_proportion"] = float(y.mean())
    if {"time_to_compromise_s", "ttc_censored"} <= set(df.columns):
        for label, h in (
            (f"rmst_benefit_{int(horizon_s)}s", float(horizon_s)),
            ("rmst_benefit_trial_window", float(window)),
        ):
            base = rmst(STATIC, h)
            for c in conds:
                security[c][label] = float(rmst(c, h) - base)
    summaries = sorted({k for v in security.values() for k in v})
    in_family = set(second_clause_summaries(horizon_s))
    xcost = np.array([float(cost[c]) for c in conds])
    tests: dict = {}
    for s_name in summaries:
        ys = np.array([security[c].get(s_name, np.nan) for c in conds], dtype=float)
        ok = np.isfinite(ys)
        rec: dict = {"n_techniques": int(ok.sum())}
        if ok.sum() >= 3 and np.std(ys[ok]) > 0 and np.std(xcost[ok]) > 0:
            pr = st.pearsonr(ys[ok], xcost[ok])
            rec["pearson"] = {"estimate": float(pr[0]), "p": float(pr[1])}
            rec["spearman"] = {
                "estimate": _pearson(st.rankdata(ys[ok]), st.rankdata(xcost[ok])),
                "p_exact": _exact_spearman_p(ys[ok], xcost[ok]),
            }
        else:
            rec["note"] = "fewer than 3 techniques or no variance"
        rec["orientation"] = SUMMARY_ORIENTATION[
            "exploitation_proportion" if s_name == "exploitation_proportion" else "rmst_benefit"
        ]
        rec["in_holm_family"] = s_name in in_family
        if not rec["in_holm_family"]:
            rec["label"] = (
                "SENSITIVITY: the trial window as the benefit horizon; outside the Holm family"
            )
        tests[_pair_key(s_name, "reconfiguration_cost_median_s")] = rec
    family = [k_ for k_, v in tests.items() if v["in_holm_family"]]
    for stat, key in (("pearson", "p"), ("spearman", "p_exact")):
        adj = holm({k_: (tests[k_].get(stat) or {}).get(key) for k_ in family})
        for k_, a in adj.items():
            tests[k_][stat]["p_holm"] = a
    return {
        "computed": True,
        "techniques": conds,
        "n_techniques": len(conds),
        "security_summaries": security,
        "cost_median_wall_clock_s": {c: float(cost[c]) for c in conds},
        "tests": tests,
        "holm_family": family,
        "horizons_s": {"time_summary": float(horizon_s), "trial_window": float(window)},
        "note": (
            "Per-technique security summaries (point estimates from the trial data; benefits "
            "are RMST(technique) - RMST(static_baseline), s) against the median wall-clock "
            "duration of one benchmarked reconfiguration. p_holm adjusts each statistic over "
            "holm_family (the compromise proportion and the RMST benefit at the time-summary "
            "horizon); the trial-window benefit is a sensitivity outside it. "
            + _FIVE_TECHNIQUE_CAVEAT
        ),
    }


FIRST_CLAUSE = (
    "no statistically significant associations among the security efficacy and operational "
    "overhead measures"
)
SECOND_CLAUSE = (
    "operational overhead measures will not statistically significantly predict defensive "
    "efficacy"
)
_NOT_REJECTED = "not rejected: no association detected (not evidence of independence)"


def _significant(tests: dict, keys: list[str], stat: str) -> tuple[list[str], list[str]]:
    """(keys whose ``stat`` has a Holm p, keys among them with p_holm < alpha)."""

    tested = [k for k in keys if ((tests.get(k) or {}).get(stat) or {}).get("p_holm") is not None]
    return tested, [k for k in tested if tests[k][stat]["p_holm"] < H04_ALPHA]


def _clause_result(n_family: int, tested: list[str], sig: list[str], why_untested: str) -> dict:
    if not n_family or not tested:
        return {"result": "not tested", "why": why_untested}
    out: dict = {"result": "rejected" if sig else _NOT_REJECTED}
    if len(tested) < n_family:
        out["untested_members"] = n_family - len(tested)
    return out


def _h04_decision(trial: dict, tech: dict) -> dict:
    """H04's decision under the rule of 2026-10-28 (``_H04_DECISION_NOTE``)."""

    # First clause: pooled Pearson, Holm over the trial-level family.
    fam1 = list(trial.get("holm_family") or [])
    pairs = trial.get("pairs") or {}
    tested1, sig1 = _significant(pairs, fam1, "pearson")
    first = {
        "statement": FIRST_CLAUSE,
        "test": (
            "pooled Pearson correlation with its sequence cluster bootstrap p, Holm over "
            "trial_level.holm_family, alpha = .05"
        ),
        "n_family": len(fam1),
        "n_tested": len(tested1),
        "significant_pairs": sig1,
        "n_significant_by_family": {
            fam: sum(1 for k in sig1 if pairs[k].get("family") == fam)
            for fam in ("among_security", "security_vs_overhead")
        },
        **_clause_result(
            len(fam1),
            tested1,
            sig1,
            "no pair in trial_level.holm_family produced a cluster-bootstrap p",
        ),
    }
    alongside = {}
    for stat, label in (
        ("spearman", "spearman_alongside"),
        ("within_condition_pearson", "sensitivity_within_condition"),
    ):
        t, s = _significant(pairs, fam1, stat)
        res = _clause_result(len(fam1), t, s, f"no {stat} p in the family")["result"]
        alongside[label] = {
            "significant_pairs": s,
            "n_tested": len(t),
            "result_if_used": res,
            "agrees_with_decision": res == first["result"],
        }
    alongside["sensitivity_within_condition"]["note"] = (
        "Pooled correlations include the contrast between conditions (static against the moving "
        "techniques), so a pooled association can restate the MTD effect itself; the "
        "within-condition correlation is what is left after the condition means are removed."
    )
    first.update(alongside)

    # Second clause: technique-level Pearson (n = 5), Holm across the two summaries.
    if not tech.get("computed"):
        second = {
            "statement": SECOND_CLAUSE,
            "result": "not tested",
            "why": tech.get("note", "the technique-level tests were not run"),
        }
    else:
        tests = tech.get("tests") or {}
        fam2 = list(tech.get("holm_family") or [])
        tested2, sig2 = _significant(tests, fam2, "pearson")
        second = {
            "statement": SECOND_CLAUSE,
            "test": (
                "technique-level Pearson correlation of each security summary with the median "
                "benchmarked reconfiguration cost, n = 5 techniques, Holm across "
                "technique_level.holm_family, alpha = .05"
            ),
            "n_family": len(fam2),
            "n_tested": len(tested2),
            "significant_tests": sig2,
            **_clause_result(
                len(fam2), tested2, sig2, "no technique-level test in the family produced a p"
            ),
        }
        t_s, s_s = _significant(tests, fam2, "spearman")
        res_s = _clause_result(len(fam2), t_s, s_s, "no exact Spearman p")["result"]
        second["spearman_exact_alongside"] = {
            "significant_tests": s_s,
            "n_tested": len(t_s),
            "result_if_used": res_s,
            "agrees_with_decision": res_s == second["result"],
        }
        sens = {k: v for k, v in tests.items() if not v.get("in_holm_family")}
        second["sensitivity_trial_window"] = {
            k: {
                "pearson_p": (v.get("pearson") or {}).get("p"),
                "significant_unadjusted": (
                    None
                    if (v.get("pearson") or {}).get("p") is None
                    else bool(v["pearson"]["p"] < H04_ALPHA)
                ),
            }
            for k, v in sens.items()
        }
        second["caveat"] = _FIVE_TECHNIQUE_CAVEAT

    results = (first["result"], second["result"])
    if "rejected" in results:
        overall = "rejected"
    elif results == ("not tested", "not tested"):
        overall = "not tested"
    elif "not tested" in results:
        overall = "incomplete: one clause was not tested and the other was not rejected"
    else:
        overall = _NOT_REJECTED
    return {
        "rule_date": H04_RULE_DATE,
        "approved_by": "chair",
        "alpha": H04_ALPHA,
        "first_clause": first,
        "second_clause": second,
        "h04": overall,
        "h04_note": (
            "H04 asserts both clauses, so it is rejected when either clause is. This line "
            "restates the two clause decisions and is not a separate test; cite the clauses."
        ),
    }


def h04_correlation_tests(
    df: pd.DataFrame,
    bench: pd.DataFrame | None = None,
    *,
    n_boot: int | None = None,
    seed: int | None = None,
    horizon_s: float = 300.0,
) -> dict:
    """The H04 correlation tests: trial level (cluster bootstrap) and technique level (n = 5)."""

    from analysis import clustered

    n_boot = clustered.BOOT_DEFAULT if n_boot is None else int(n_boot)
    seed = clustered.BOOT_SEED_DEFAULT if seed is None else int(seed)
    trial = _trial_level(df, n_boot, seed)
    tech = _technique_level(df, bench, horizon_s)
    return {
        "hypothesis": H04_STATEMENT,
        "first_clause": FIRST_CLAUSE,
        "second_clause": SECOND_CLAUSE,
        "trial_level": trial,
        "technique_level": tech,
        "decision": _h04_decision(trial, tech),
        "decision_note": _H04_DECISION_NOTE,
        "parsimony_note": (
            "The parsimony procedure (vif, pca, security_measures_vif_pca, "
            "regularized_selection) is a separate block of rq4 and does not enter this decision."
        ),
    }


def rq4(df: pd.DataFrame, bench: pd.DataFrame | None = None) -> dict:
    """Run the RQ4 correlation + parsimony (VIF/PCA/regularized) analyses.

    ``bench`` is the reconfiguration-cost benchmark frame (``rq5_profile.load_benchmark``);
    it feeds only the technique-level H04 tests.
    """

    pca = _pca(df, METRIC_SET)
    screened = _numeric(df, METRIC_SET)
    reg_ttc = _elastic_net_select(df, "time_to_compromise_s", METRIC_SET)
    reg_exploitation = _l1_logistic_select(df, "exploitation_success", METRIC_SET)

    # Spec definition: the parsimonious subset = metrics that LOAD ON SEPARATE
    # PCA COMPONENTS *and* survive regularized selection.
    regularized_union = sorted(
        set(reg_ttc.get("selected_subset", [])) | set(reg_exploitation.get("selected_subset", []))
    )
    distinct = set(pca.get("distinct_component_metrics", []))
    if distinct:
        parsimonious = sorted(set(regularized_union) & distinct)
    else:
        parsimonious = regularized_union  # PCA unavailable: fall back to regularized only

    return {
        # H04's decision block (re-run update): the correlation tests, structural pairs flagged.
        "h04_correlation_tests": h04_correlation_tests(df, bench),
        "pearson": _corr(df, SECURITY + METRIC_SET, "pearson"),
        "spearman": _corr(df, SECURITY + METRIC_SET, "spearman"),
        "vif": _vif(df, METRIC_SET),
        # VIF / PCA / regularized selection all listwise-delete over METRIC_SET; this is the
        # size and composition of the rows they are actually fit on.
        "metric_set_missingness": {
            **item_missingness(df, METRIC_SET, group_col="condition"),
            "items_after_variance_screen": list(screened.columns),
            "items_dropped_by_variance_screen": [
                c for c in METRIC_SET if c in df.columns and c not in screened.columns
            ],
            "n_complete_case_used_by_vif_pca": int(len(screened.dropna())),
            "screen_note": (
                "rq4._numeric drops indicators with < 2 non-null values or zero variance BEFORE "
                "listwise deletion, so n_complete_case (over all present items) can exceed "
                "n_complete_case_used_by_vif_pca (over the screened items). Both are reported."
            ),
        },
        "pca": pca,
        # The five-measure security matrix without the overhead columns (the dissertation's
        # trial-level VIF/PCA); archived with every report from 2026-10-27.
        "security_measures_vif_pca": security_measure_parsimony(df),
        "regularized_selection": {
            "time_to_compromise_s": reg_ttc,
            "exploitation_success": reg_exploitation,
        },
        "regularized_union": regularized_union,
        "distinct_component_metrics": sorted(distinct),
        "parsimonious_subset": parsimonious,
        "parsimonious_note": (
            "Parsimonious subset = regularized-selection survivors that also load on distinct PCA "
            "components (per ch3 RQ4). See regularized_union / distinct_component_metrics for inputs."
        ),
        "cronbach_overhead_composite": cronbach_alpha(
            df, OVERHEAD_COMPOSITE, missing_reason=_LATENCY_MISSING_REASON
        ),
        "cronbach_note": (
            "Alpha applies ONLY to the operational-overhead composite (latency, CPU, memory, "
            "network -- metrics_specification 7 A2); security-efficacy indicators are formative "
            "and assessed via VIF/PCA/regularized selection above. Read the alpha together with "
            "cronbach_overhead_composite.item_missingness (which rows it was computed on) and "
            ".between_condition_artifact_check (whether it reflects internal consistency or the "
            "static-vs-MTD contrast)."
        ),
        "metric_orientation": METRIC_ORIENTATION,
        "metric_orientation_note": (
            "Direction of each metric in the pearson/spearman matrices and the PCA loadings. "
            "service_availability_impact stores availability RETAINED (higher = more available = "
            "better), not degradation, despite its name; its correlations and loadings must be "
            "read with the opposite sign from the other overhead indicators. The column is not "
            "renamed and not re-signed here -- VIF, |loadings| and regularized selection are "
            "sign-invariant -- and the analysis layer derives service_availability_loss "
            "(= 1 - retained) for the cost-oriented uses in RQ5."
        ),
    }
