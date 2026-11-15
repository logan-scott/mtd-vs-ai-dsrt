"""Within-block drift: does the outcome trend across a condition's sequences in run order?

The design runs each condition as ONE contiguous block (block order randomized once from the
master seed), so condition and execution period are confounded: no analysis of these data can
test a between-condition order effect. What CAN be checked is drift WITHIN each block -- the
testbed, the model API or the attacker's behaviour changing over the hours a block took -- by
asking whether the outcome trends across the block's sequences in the order they ran.

``within_block_drift`` fits, per condition whose outcome varies, a logit of the binary
co-primary on the sequence's position in the block (robust SEs clustered on sequence; the
estimator of ``clustered.glm_cluster_or``), plus a pooled model with one common slope and
condition fixed effects; and, as a distribution-free corroboration, Spearman's rho between
run position and each sequence's mean restricted time-to-compromise. Conditions whose outcome is constant have
nothing to drift and are listed as such. Every clustered slope rests on at least 20 informative
sequences (``clustered.MIN_INFORMATIVE_SEQUENCES``, 2026-10-29): a condition whose outcome
departs from its most common value in fewer gets no slope of its own (``own_slope_withheld``),
because its clustered standard error would collapse, but stays in the pooled slope, which needs
20 across the conditions it pools.

RE-RUN ANALYSIS UPDATE (2026-10-28; ``docs/rerun_runbook.md`` §6). The re-run interleaves the
conditions (``design.run_order: interleaved``, C209): round r runs sequence r of every
condition, in an order shuffled per round, so the ``-sN`` suffix is the round and condition is
no longer aliased with execution period. ``interleaved_drift`` is the check's interleaved form:
the outcome against round across all conditions with condition fixed effects and
sequence-clustered SEs (the logit over the conditions whose outcome varies, and the restricted
time to compromise over every condition), plus each condition's slope with Holm.
``detect_run_order`` reads the design from the trials' real start times (``wall_start_utc``):
conditions whose execution spans overlap ran interleaved, disjoint spans ran in blocks.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import stats

from analysis import clustered
from analysis.fdr import holm

_SEQ_RE = re.compile(r"-s(\d+)(?:-.*)?$")
PER = 10.0  # slopes are reported per 10 sequences of run order

_NOTE = (
    "Run position = the sequence's index within its condition block (sequence_id suffix -sN), "
    "which is the order the block executed in. Slopes are odds ratios per 10 sequences of run "
    "order from logit models with robust SEs clustered on sequence. A drift "
    "check can only detect change WITHIN a block; because each condition ran as one "
    "contiguous block, condition and execution period remain confounded BETWEEN conditions, "
    "which no analysis of these data can separate (that requires interleaved blocks). p_holm "
    "adjusts each family (the per-condition slopes; the per-condition Spearman tests) "
    "separately for the number of conditions tested. Execution order is the harness loop "
    "order (experiment_runner: for seq_i in range(sequences)); excluded trials are screened "
    "out before this check, as in the primary analysis."
)


RUN_ORDER_BLOCKED = "blocked"
RUN_ORDER_INTERLEAVED = "interleaved"

_NOTE_INTERLEAVED_DATA = (
    "THIS RUN WAS INTERLEAVED (see run_order): the blocked-design caveat above does not apply. "
    "Run position (-sN) is the round, which every condition shares, so these per-condition "
    "slopes and this pooled slope are the per-condition and pooled parts of "
    "design_checks.interleaved_drift; cite that block."
)


def detect_run_order(df: pd.DataFrame) -> dict:
    """Blocked or interleaved, from the trials' real start times (``wall_start_utc``).

    Each condition's execution span is [earliest, latest] start of its trials. Spans that
    overlap mean the conditions ran interleaved; pairwise disjoint spans mean blocks. Returns
    ``run_order = None`` when the data carry no real start times (modeled-clock runs without
    timing record ``NOT_RECORDED``).
    """

    if "wall_start_utc" not in df.columns or "condition" not in df.columns:
        return {"run_order": None, "source": "not determinable: no wall_start_utc column"}
    t = pd.to_datetime(df["wall_start_utc"], utc=True, errors="coerce", format="ISO8601")
    ok = t.notna()
    cond = df.loc[ok, "condition"].astype(str)
    if ok.sum() < 2 or cond.nunique() < 2:
        return {
            "run_order": None,
            "source": "not determinable: fewer than two conditions with a recorded real start",
        }
    spans = t[ok].groupby(cond).agg(["min", "max"]).sort_values("min")
    starts, ends = spans["min"].to_numpy(), spans["max"].to_numpy()
    overlap = bool(any(starts[i + 1] < ends[: i + 1].max() for i in range(len(starts) - 1)))
    return {
        "run_order": RUN_ORDER_INTERLEAVED if overlap else RUN_ORDER_BLOCKED,
        "source": "wall_start_utc (condition execution spans "
        + ("overlap" if overlap else "are disjoint")
        + ")",
        "condition_spans_utc": {
            str(c): [
                r["min"].isoformat().replace("+00:00", "Z"),
                r["max"].isoformat().replace("+00:00", "Z"),
            ]
            for c, r in spans.iterrows()
        },
        "n_rows_with_real_start": int(ok.sum()),
    }


def run_position(sequence_id: pd.Series) -> pd.Series:
    """The integer N of a ``<condition>-sN`` (optionally ``-sN-<arm>``) sequence id."""

    return sequence_id.astype(str).str.extract(_SEQ_RE, expand=False).astype(float)


def _slope(d: pd.DataFrame, outcome: str, rhs: str) -> dict:
    out = clustered.glm_cluster_or(d, outcome, rhs, "pos10")
    if "odds_ratio" in out:
        out = {"odds_ratio_per_10_sequences": out.pop("odds_ratio"), **out}
        out["n_sequences"] = out.pop("n_clusters")
    return out


def within_block_drift(
    df: pd.DataFrame,
    outcome: str = "exploitation_success",
    horizon: float | None = None,
    *,
    run_order: str | None = None,
) -> dict:
    """Per-condition and pooled trend of ``outcome`` across run position within each block.

    ``run_order`` (``blocked`` / ``interleaved``; None = not known) only changes the note: on
    interleaved data the blocked-design caveat does not apply.
    """

    need = {outcome, "condition", "sequence_id"}
    if not need <= set(df.columns):
        return {"note": f"missing columns {sorted(need - set(df.columns))}"}
    d = df.dropna(subset=list(need)).copy()
    d[outcome] = pd.to_numeric(d[outcome], errors="coerce")
    d = d.dropna(subset=[outcome])
    d["pos"] = run_position(d["sequence_id"])
    if d["pos"].isna().any():
        return {"note": "sequence ids do not carry a -sN run position; drift not computed"}
    d["pos10"] = d["pos"] / PER
    if horizon is None:
        horizon = 1800.0
        if "trial_duration_limit_s" in d.columns:
            lim = pd.to_numeric(d["trial_duration_limit_s"], errors="coerce").dropna().unique()
            if len(lim) == 1:
                horizon = float(lim[0])
    rt = None
    if "time_to_compromise_s" in d.columns:
        rt = pd.to_numeric(d["time_to_compromise_s"], errors="coerce").clip(upper=horizon)

    per: dict[str, dict] = {}
    varying: list[str] = []
    withheld: dict[str, str] = {}
    pooled_n = 0
    for cond, g in d.groupby("condition", observed=True):
        cond = str(cond)
        entry: dict = {"n_sequences": int(g["sequence_id"].nunique())}
        # Descriptive: the outcome rate in the first, middle and last third of the block.
        thirds = np.array_split(np.sort(g["pos"].unique()), 3)
        entry[f"{outcome}_rate_by_run_third"] = [
            float(g.loc[g["pos"].isin(t), outcome].mean()) if len(t) else None for t in thirds
        ]
        if g[outcome].nunique() > 1:
            varying.append(cond)
            info = clustered.informative_sequences(g, outcome)
            pooled_n += info["n_sequences_departing"]
            if info["enough"]:
                entry[outcome] = _slope(g, outcome, "pos10")
            else:
                withheld[cond] = clustered.too_few_informative_note(info)
                entry[outcome] = {"note": withheld[cond], **info}
        else:
            entry[outcome] = {"note": "outcome constant in this condition; nothing to drift"}
        if rt is not None:
            seqm = (
                g.assign(_rt=rt.loc[g.index])
                .groupby("sequence_id", observed=True)
                .agg(pos=("pos", "first"), rt=("_rt", "mean"))
            )
            if seqm["rt"].nunique() > 1:
                rho, p = stats.spearmanr(seqm["pos"], seqm["rt"])
                entry["restricted_ttc_spearman"] = {"rho": float(rho), "p": float(p)}
            else:
                entry["restricted_ttc_spearman"] = {"note": "constant across sequences"}
        per[cond] = entry

    for key, field in ((outcome, "p"), ("restricted_ttc_spearman", "p")):
        fam = {
            c: v[key][field]
            for c, v in per.items()
            if isinstance(v.get(key), dict) and v[key].get(field) is not None
        }
        for c, adj in holm(fam).items():
            per[c][key]["p_holm"] = adj

    pooled: dict = {"note": "no condition with a varying outcome"}
    if varying and pooled_n < clustered.MIN_INFORMATIVE_SEQUENCES:
        pooled = {"note": clustered.too_few_pooled_note(pooled_n), "conditions": varying}
    elif varying:
        sub = d[d["condition"].astype(str).isin(varying)]
        rhs = "pos10 + C(condition)" if len(varying) > 1 else "pos10"
        pooled = {**_slope(sub, outcome, rhs), "conditions": varying}
    out = {
        "outcome": outcome,
        "pooled_common_slope": pooled,
        "by_condition": per,
        "note": _NOTE,
    }
    if withheld:
        out["own_slope_withheld"] = withheld
    if run_order == RUN_ORDER_INTERLEAVED:
        out["run_order"] = run_order
        out["note"] = _NOTE + " " + _NOTE_INTERLEAVED_DATA
    return out


_INTERLEAVED_NOTE = (
    "Run position = the round (the sequence_id suffix -sN). Under design.run_order = "
    "interleaved (C209) round r runs sequence r of every condition, in an order shuffled per "
    "round from the master seed, so the round indexes execution period for every condition at "
    "once and condition is not aliased with period. Slopes are per 10 rounds. "
    "pooled_common_slope: logit of the outcome on round with condition fixed effects and robust "
    "SEs clustered on sequence, over the conditions whose outcome varies (a condition with the "
    "outcome in every trial or in none carries no information about a logit slope, and its "
    "fixed effect is not identified; such conditions are listed in conditions_constant). "
    "Every clustered slope rests on at least 20 informative sequences "
    "(clustered.MIN_INFORMATIVE_SEQUENCES; sequences in which the outcome departs from the "
    "condition's most common value): a condition with fewer gets no slope of its own "
    "(own_slope_withheld), because its clustered standard error would collapse, but stays in "
    "the pooled slope, which needs 20 across the conditions it pools. "
    "pooled_restricted_ttc: least squares of the restricted time to compromise, min(TTC, "
    "horizon) in seconds, on round with condition fixed effects and sequence-clustered SEs, over "
    "every condition (time varies even in an arm compromised in every trial); the slope is in "
    "seconds per 10 rounds. by_condition: each condition's own logit slope and Spearman's rho "
    "between round and the sequence's mean restricted TTC, each family Holm-adjusted across "
    "conditions. Within a round the conditions ran one after another in shuffled order, so the "
    "round locates a sequence in the run to within one round's duration. Excluded sequences "
    "are screened out before this check, as in the primary analysis."
)


def _ols_slope(d: pd.DataFrame, y: str, rhs: str, term: str, group: str) -> dict:
    """Least-squares slope for ``term`` with cluster-robust SEs (statsmodels' G/(G-1) form)."""

    import statsmodels.formula.api as smf

    try:
        m = smf.ols(f"{y} ~ {rhs}", d).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(d[group])[0]}
        )
        b, se, p = float(m.params[term]), float(m.bse[term]), float(m.pvalues[term])
        ci = m.conf_int().loc[term]
    except Exception as exc:  # noqa: BLE001 - degenerate subset
        return {"note": f"cluster-robust least squares unavailable: {exc}"}
    if not (np.isfinite(b) and np.isfinite(se) and np.isfinite(p)):
        return {"note": f"{term} has a non-finite estimate or standard error"}
    return {
        "slope_s_per_10_rounds": b,
        "ci95": [float(ci.iloc[0]), float(ci.iloc[1])],
        "p": p,
        "n": int(m.nobs),
        "n_sequences": int(d[group].nunique()),
    }


def _per_rounds(obj):
    """Relabel the blocked check's per-10-sequences slopes as per 10 rounds (same number)."""

    if isinstance(obj, dict):
        return {
            ("odds_ratio_per_10_rounds" if k == "odds_ratio_per_10_sequences" else k): _per_rounds(
                v
            )
            for k, v in obj.items()
        }
    return obj


def interleaved_drift(
    df: pd.DataFrame,
    outcome: str = "exploitation_success",
    horizon: float | None = None,
) -> dict:
    """The drift check's interleaved form: outcome against round across all conditions."""

    base = within_block_drift(df, outcome, horizon, run_order=RUN_ORDER_INTERLEAVED)
    if "by_condition" not in base:
        return base
    d = df.dropna(subset=["condition", "sequence_id"]).copy()
    d["pos10"] = run_position(d["sequence_id"]) / PER
    if horizon is None:
        horizon = 1800.0
        if "trial_duration_limit_s" in d.columns:
            lim = pd.to_numeric(d["trial_duration_limit_s"], errors="coerce").dropna().unique()
            if len(lim) == 1:
                horizon = float(lim[0])
    ttc: dict = {"note": "time_to_compromise_s not in the dataset"}
    if "time_to_compromise_s" in d.columns:
        d["_rttc"] = pd.to_numeric(d["time_to_compromise_s"], errors="coerce").clip(upper=horizon)
        sub = d.dropna(subset=["_rttc", "pos10"])
        rhs = "pos10 + C(condition)" if sub["condition"].nunique() > 1 else "pos10"
        ttc = {
            **_ols_slope(sub, "_rttc", rhs, "pos10", "sequence_id"),
            "horizon_s": float(horizon),
            "conditions": sorted(sub["condition"].astype(str).unique()),
        }
    per = _per_rounds(base["by_condition"])
    withheld = base.get("own_slope_withheld", {})
    constant = sorted(
        c
        for c, v in per.items()
        if isinstance(v.get(outcome), dict) and "note" in v[outcome] and c not in withheld
    )
    rounds = run_position(d["sequence_id"])
    out = {
        "outcome": outcome,
        "run_position": "round (sequence_id suffix -sN)",
        "n_rounds": int(rounds.nunique()),
        "pooled_common_slope": {
            **_per_rounds(base["pooled_common_slope"]),
            "conditions_constant": constant,
        },
        "pooled_restricted_ttc": ttc,
        "by_condition": per,
        "note": _INTERLEAVED_NOTE,
    }
    if withheld:
        out["own_slope_withheld"] = withheld
    return out
