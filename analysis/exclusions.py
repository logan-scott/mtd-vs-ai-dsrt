"""A-priori trial-exclusion accounting for ``analysis_report.json``.

Reconciles the two independent exclusion counters the instrument maintains -- the
ROW-level a-priori gate (``harness/trial_runner.py``) and the SEQUENCE-level fault
abort (``harness/experiment_runner.py``) -- and publishes the evidence that the
excluded rows are superseded regenerations rather than lost data.

Read side by side with no definitions, ``n_excluded: 63`` in the report and
``excluded=0`` on every line of the run log look like a contradiction. They are not:
they count different things over different denominators. Everything this module emits
exists so a reader never has to take that on trust.

Everything here consumes the UNSCREENED frame: ``analysis.dataset.prepare`` drops
excluded rows, so the accounting must be computed before (or beside) it.
"""

from __future__ import annotations

import itertools
import re

import pandas as pd

from analysis.dataset import to_frame

UNSPECIFIED_REASON = "unspecified"

# An attempt "slot": the unit the design guarantees exactly one included row for.
_SLOT = ["sequence_id", "attempt_index"]

# The three pre-registered primary outcomes (see analysis/pipeline.py docstring).
SENSITIVITY_OUTCOMES: tuple[str, ...] = (
    "exploitation_success",
    "exploit_success",
    "time_to_compromise_s",
)

# Outcomes that are right-censored at the trial horizon: their raw arithmetic mean is a
# censored-mean summary, NOT the estimand the reported model targets (a Cox fit).
_CENSOR_COLUMN = {"time_to_compromise_s": "ttc_censored"}

# Reporting convention (NOT a hypothesis test). A rank inversion between two conditions
# is treated as substantively empty when the SCREENED analysis separated them by less
# than this many pooled within-condition SDs. Cohen's smallest conventional benchmark for
# a standardized mean difference is d = 0.20 ("small"); 0.10 SD is half of that, i.e. the
# screened analysis was making no distinguishable claim about the pair to begin with. The
# raw deltas and SDs are published alongside so a reader can apply any other threshold.
NOISE_FLOOR_SD = 0.1

# Where each row-level exclusion_reason value comes from, quoted into the report so a
# reader never has to guess whether an "infrastructure_fault" row was a real testbed
# failure or the harness fault injector exercising the exclude-and-regenerate path.
EXCLUSION_REASON_PROVENANCE: dict[str, str] = {
    "checksum_mismatch": (
        "harness/trial_runner.py _inclusion(): the pre-trial baseline-equivalence checksum "
        "did not equal the checksum of the specified vulnerability profile, so the attempt "
        "did not run against the intended configuration. TERMINAL -- the attempt is NOT "
        "re-run; environment state and attacker context advance and the attempt slot ends "
        "with no included row."
    ),
    "attacker_init_or_logging_incomplete": (
        "harness/trial_runner.py _inclusion(): the attacker emitted no reconnaissance "
        "command, i.e. it failed to initialize or its timestamped log is incomplete. "
        "TERMINAL -- the attempt is NOT re-run; environment state and attacker context "
        "advance and the attempt slot ends with no included row."
    ),
    "infrastructure_fault": (
        "harness/trial_runner.py: a ROW-level 'infrastructure_fault' is emitted by exactly "
        "one statement in the codebase -- the harness fault injector (cli.py --fault-rate, "
        "default 0.03), whose draw comes from a dedicated RNG stream seeded from the "
        "sequence seed and is made independently of the attempt's observations. It flags a "
        "completed attempt as excluded and REGENERATES it in the same trial window without "
        "advancing state or attacker context. A GENUINE provisioning or attacker-API fault "
        "never produces this row-level value: it aborts the whole sequence and is recorded "
        "at sequence level (harness/experiment_runner.py _fault_reason -> "
        "ExperimentDataset.excluded_sequences -> run_meta.json n_excluded_sequences)."
    ),
}


# --------------------------------------------------------------------------------------
# masks and small helpers
# --------------------------------------------------------------------------------------


def excluded_mask(df: pd.DataFrame) -> pd.Series:
    """Exact complement of ``analysis.dataset.screen()``, which keeps ``excluded == 0``.

    Built from the SAME predicate ``screen()`` uses rather than a re-derivation, so
    ``n_included`` can never disagree with the analyzed sample for ANY dtype the column
    arrives in (int, float with NaN, bool, or an object column of strings). Anything the
    screen does not keep -- NaN, unexpected codes, string encodings -- is excluded. This
    is the single source of truth for the row-level count: ``analysis.pipeline._n_excluded``
    delegates to it so the top-level ``n_excluded`` and ``exclusions.n_excluded`` cannot
    drift apart.
    """

    keep = df["excluded"] == 0
    if keep.dtype != bool:  # nullable/masked dtypes yield pd.NA, which screen() rejects
        keep = keep.fillna(False).astype(bool)
    return ~keep


def _reason_series(df: pd.DataFrame) -> pd.Series:
    if "exclusion_reason" not in df.columns:
        return pd.Series([UNSPECIFIED_REASON] * len(df), index=df.index, dtype=object)
    # astype(object) FIRST: an all-empty CSV column loads as float64 and .where(..., None)
    # would keep NaN rather than None.
    s = df["exclusion_reason"].astype(object).where(df["exclusion_reason"].notna(), None)
    return s.map(lambda v: UNSPECIFIED_REASON if v in (None, "") else str(v))


def _pooled_sd(groups: list[pd.Series]) -> float | None:
    """sqrt(mean of the within-condition sample variances) -- the outcome's own noise scale."""

    variances = [float(g.var(ddof=1)) for g in groups if len(g) > 1]
    variances = [v for v in variances if pd.notna(v)]
    if not variances:
        return None
    return float((sum(variances) / len(variances)) ** 0.5)


def _arm_means(frame: pd.DataFrame, outcome: str) -> pd.Series:
    s = pd.to_numeric(frame[outcome], errors="coerce")
    return s.groupby(frame["condition"]).mean()


# --------------------------------------------------------------------------------------
# retry-structure evidence
# --------------------------------------------------------------------------------------


def _regeneration_evidence(df: pd.DataFrame, excl: pd.Series) -> dict:
    """Is every excluded row a superseded retry, or did an attempt slot actually lose data?"""

    if not set(_SLOT).issubset(df.columns):
        return {"note": "sequence_id/attempt_index absent; retry structure not checkable"}
    inc, exc = df.loc[~excl], df.loc[excl]
    inc_slots = set(map(tuple, inc[_SLOT].to_numpy().tolist()))
    all_slots = set(map(tuple, df[_SLOT].to_numpy().tolist()))
    exc_slots = list(map(tuple, exc[_SLOT].to_numpy().tolist()))
    superseded = sum(1 for s in exc_slots if s in inc_slots)
    excess = int(inc.duplicated(subset=_SLOT).sum()) if len(inc) else 0
    dup_slots = (
        int(inc.loc[inc.duplicated(subset=_SLOT, keep=False), _SLOT].drop_duplicates().shape[0])
        if len(inc)
        else 0
    )
    per_seq = inc.groupby("sequence_id").size() if len(inc) else pd.Series(dtype="int64")
    per_slot = exc.groupby(_SLOT).size() if len(exc) else pd.Series(dtype="int64")
    out = {
        "n_attempt_slots": int(len(all_slots)),
        "n_attempt_slots_with_an_included_row": int(len(inc_slots)),
        "n_attempt_slots_with_no_included_row": int(len(all_slots - inc_slots)),
        "n_excluded_rows_superseded_by_a_valid_retry": int(superseded),
        "n_excluded_rows_with_no_valid_retry": int(len(exc_slots) - superseded),
        "n_duplicated_included_slots": dup_slots,
        "n_excess_included_rows_in_duplicated_slots": excess,
        "max_exclusions_within_one_attempt_slot": int(per_slot.max()) if len(per_slot) else 0,
        "included_rows_per_sequence": {
            "min": int(per_seq.min()) if len(per_seq) else 0,
            "max": int(per_seq.max()) if len(per_seq) else 0,
            "n_sequences": int(len(per_seq)),
            "balanced": bool(len(per_seq) and per_seq.nunique() == 1),
        },
    }
    if not exc_slots:
        out["verdict"] = "no_exclusions"
        out["verdict_note"] = "no rows were excluded; the written dataset is the analyzed panel."
        return out
    clean = (
        out["n_excluded_rows_with_no_valid_retry"] == 0
        and out["n_attempt_slots_with_no_included_row"] == 0
        and excess == 0
    )
    out["verdict"] = "retry_overflow" if clean else "possible_data_loss"
    out["verdict_note"] = (
        "retry_overflow: every excluded row shares its (sequence_id, attempt_index) attempt "
        "slot with exactly one included row, so no attempt slot lost data and the analyzed "
        "panel is balanced; the excluded rows are superseded regenerations kept for audit."
        if clean
        else "possible_data_loss: an excluded row has NO included row in its attempt slot, or "
        "an attempt slot has more than one included row. These exclusions are NOT pure retry "
        "overflow -- the analyzed panel is unbalanced and is not simply the regenerated panel. "
        "This is the expected signature of a TERMINAL gate reason (checksum_mismatch or "
        "attacker_init_or_logging_incomplete), which the harness does not regenerate. "
        "Investigate before reporting."
    )
    return out


# --------------------------------------------------------------------------------------
# differential attrition
# --------------------------------------------------------------------------------------


def _exclusion_homogeneity(df: pd.DataFrame, excl: pd.Series, by: str) -> dict:
    """Differential-attrition check: does the exclusion rate depend on ``by``?

    A pre-registered gate is only innocuous if it fired independently of the condition
    (and of position within a sequence). This is the check that says so with a number
    instead of an assurance -- guarded, because the chi-square approximation is unreliable
    on the sparse tables a low exclusion rate produces.
    """

    if by not in df.columns or df.empty:
        return {"note": f"{by} absent; homogeneity not estimable"}
    tab = pd.crosstab(df[by], excl.astype(int))
    hits = tab[1] if 1 in tab.columns else pd.Series(0, index=tab.index)
    out: dict = {
        "exclusion_rate_by_group": {
            str(k): round(float(v), 6) for k, v in (hits / tab.sum(axis=1)).items()
        },
        "interpretation": (
            "a non-significant result is absence of evidence, not evidence of no differential "
            "attrition; read it together with exclusion_rate_by_group and min_expected_count."
        ),
    }
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        out["note"] = "homogeneity test not estimable (one group, or no exclusions at all)"
        return out
    try:
        from scipy import stats

        res = stats.chi2_contingency(tab.to_numpy())
        min_exp = float(res.expected_freq.min())
        valid = bool(min_exp >= 5.0)
        out.update(
            {
                "test": "chi2_homogeneity",
                "chi2": float(res.statistic),
                "dof": int(res.dof),
                "p": float(res.pvalue),
                "min_expected_count": round(min_exp, 3),
                "approximation_valid": valid,
                "differential_at_alpha_05": (bool(res.pvalue < 0.05) if valid else None),
            }
        )
        if not valid:
            out["approximation_note"] = (
                "smallest expected cell < 5: the chi-square approximation is unreliable here, "
                "so no significance decision is reported."
            )
    except Exception as exc:  # pragma: no cover - defensive
        out["note"] = f"homogeneity test unavailable: {exc}"
    return out


# --------------------------------------------------------------------------------------
# retention sensitivity
# --------------------------------------------------------------------------------------


def _compare_arm(
    screened: pd.Series,
    other: pd.Series,
    var_by_condition: dict,
    pooled_sd: float | None,
    degenerate: bool,
) -> dict:
    """Compare one counterfactual arm against the screened arm, on the outcome's own scale.

    The headline is ``max_abs_delta_in_pooled_sd`` rather than a rank-order boolean: rank
    order is a discontinuous function of the means, so two conditions that are effectively
    tied flip on noise and a boolean then records the noise as if it were a finding. Every
    inversion is instead named and sized against the separation the SCREENED analysis
    actually claimed for that pair.
    """

    from scipy.stats import rankdata, spearmanr

    conds = list(screened.index)
    r_screened, r_other = rankdata(screened.to_numpy()), rankdata(other.to_numpy())
    inversions = []
    for i, j in itertools.combinations(range(len(conds)), 2):
        d_screened = float(screened.iloc[i] - screened.iloc[j])
        d_other = float(other.iloc[i] - other.iloc[j])
        if d_screened == 0.0 and d_other == 0.0:
            continue
        flipped = (d_screened > 0) != (d_other > 0) or (d_screened == 0) != (d_other == 0)
        if not flipped:
            continue
        v_i, v_j = var_by_condition.get(conds[i]), var_by_condition.get(conds[j])
        pair_sd = None
        if v_i is not None and v_j is not None and pd.notna(v_i) and pd.notna(v_j):
            pair_sd = float(((float(v_i) + float(v_j)) / 2.0) ** 0.5)
        inversions.append(
            {
                "pair": [conds[i], conds[j]],
                "screened_delta": round(d_screened, 6),
                "counterfactual_delta": round(d_other, 6),
                "pair_pooled_sd": round(pair_sd, 6) if pair_sd else pair_sd,
                "screened_separation_in_pair_sd": (
                    round(abs(d_screened) / pair_sd, 6) if pair_sd else None
                ),
            }
        )
    max_delta = float((other - screened).abs().max())
    beyond_noise = None
    if not degenerate:
        beyond_noise = bool(
            all(
                iv["screened_separation_in_pair_sd"] is not None
                and iv["screened_separation_in_pair_sd"] < NOISE_FLOOR_SD
                for iv in inversions
            )
        )
    return {
        "mean_by_condition": {k: round(float(v), 6) for k, v in other.items()},
        "max_abs_delta": round(max_delta, 6),
        "max_abs_delta_in_pooled_sd": (round(max_delta / pooled_sd, 6) if pooled_sd else None),
        "spearman_rho_vs_screened": (
            None
            if degenerate
            else round(float(spearmanr(screened.to_numpy(), other.to_numpy()).statistic), 6)
        ),
        "rank_inversions": inversions,
        "n_rank_inversions": len(inversions),
        "rank_order_preserved": None if degenerate else bool((r_screened == r_other).all()),
        "rank_order_preserved_beyond_noise": beyond_noise,
    }


def _retention_sensitivity(df: pd.DataFrame, excl: pd.Series, outcomes) -> dict:
    """Counterfactual: what would each condition mean be had the exclusion rule not fired?"""

    out: dict = {
        "note": (
            "'screened' is the analyzed sample (a-priori inclusion rule applied: one row per "
            "attempt slot). Two counterfactual arms are reported. 'retained_all_emitted_rows' "
            "analyzes every emitted row including superseded regenerations -- note that this "
            "DOUBLE-WEIGHTS every retried slot and its sequence cluster, so it estimates a "
            "per-emitted-attempt mean rather than the per-attempt-slot mean. "
            "'first_emitted_row_per_slot' takes the first row emitted in each attempt slot, "
            "i.e. the balanced panel that would exist had the rule never fired and nothing "
            "been regenerated; it is the like-for-like counterfactual. Deltas are also "
            "expressed in pooled within-condition SDs so they are read on the outcome's own "
            "noise scale. Every rank inversion is named with the separation the screened "
            "analysis actually claimed for that pair; "
            "rank_order_preserved_beyond_noise is true when no inversion involves a pair the "
            f"screened analysis separated by >= {NOISE_FLOOR_SD} pooled SD. This is a raw-mean "
            "screening check, not a re-run of the LMM/GEE/GLMM/Cox models on the "
            "counterfactual samples."
        )
    }
    if "condition" not in df.columns or df.empty:
        out["note"] = "condition absent or dataset empty; retention sensitivity not estimable"
        return out
    try:
        import scipy.stats  # noqa: F401
    except Exception:  # pragma: no cover - defensive
        out["note"] = "scipy unavailable; retention sensitivity not estimable"
        return out

    inc = df.loc[~excl]
    has_slots = set(_SLOT).issubset(df.columns)
    first = df.groupby(_SLOT, sort=False).head(1) if has_slots else None
    for oc in outcomes:
        if oc not in df.columns:
            continue
        screened = _arm_means(inc, oc)
        union = _arm_means(df, oc)
        idx = sorted(set(screened.index) | set(union.index))
        screened, union = screened.reindex(idx), union.reindex(idx)
        keep = screened.notna() & union.notna()
        if int(keep.sum()) < 2:
            out[oc] = {"note": "fewer than two estimable conditions"}
            continue
        screened, union = screened[keep], union[keep]
        conds = list(screened.index)
        groups = [
            pd.to_numeric(inc.loc[inc["condition"] == c, oc], errors="coerce").dropna()
            for c in conds
        ]
        var_by = {
            c: (float(g.var(ddof=1)) if len(g) > 1 else None)
            for c, g in zip(conds, groups, strict=True)
        }
        pooled = _pooled_sd(groups)
        degenerate = bool(screened.nunique() <= 1)
        rec: dict = {
            "screened_mean_by_condition": {k: round(float(v), 6) for k, v in screened.items()},
            "pooled_within_condition_sd": round(pooled, 6) if pooled else pooled,
            "degenerate": degenerate,
            "retained_all_emitted_rows": _compare_arm(screened, union, var_by, pooled, degenerate),
        }
        if degenerate:
            rec["degenerate_note"] = (
                "every condition has the same screened mean on this outcome, so there is no "
                "rank order to preserve and no pooled dispersion to scale by; the rank "
                "booleans are reported as null rather than as a vacuous true."
            )
        if first is not None:
            fe = _arm_means(first, oc).reindex(screened.index)
            if fe.notna().all():
                rec["first_emitted_row_per_slot"] = _compare_arm(
                    screened, fe, var_by, pooled, degenerate
                )
        censor_col = _CENSOR_COLUMN.get(oc)
        if censor_col and censor_col in inc.columns:
            frac = float(pd.to_numeric(inc[censor_col], errors="coerce").fillna(0).mean())
            rec["censored_fraction_screened"] = round(frac, 6)
            rec["censoring_caveat"] = (
                f"{round(frac * 100, 1)}% of screened trials are right-censored at the trial "
                "horizon on this outcome, so these arithmetic means are censored-mean "
                "summaries, not the estimand of the reported cluster-robust Cox model. They "
                "are adequate for detecting whether the exclusion rule moved the data, and "
                "are not offered as effect estimates."
            )
        out[oc] = rec
    return out


# --------------------------------------------------------------------------------------
# sequence-level counter + run_meta corroboration
# --------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------
# Sequence exclusions by condition and cause (re-run analysis update, 2026-10-28)
# --------------------------------------------------------------------------------------

#: The faults that exclude a sequence in the re-run (docs/metrics_specification.md §8a), in
#: that table's order: (cause, reason, raised by, how the fault's own message begins).
EXCLUSION_CAUSES: tuple[tuple[str, str, str, str], ...] = (
    (
        "reconfiguration",
        "infrastructure_fault",
        "AnsibleProvisioner.apply",
        "ansible reconfigure for condition ... failed",
    ),
    (
        "readiness_gate",
        "infrastructure_fault",
        "readiness gate (surface_check: strict)",
        "readiness gate: ...",
    ),
    (
        "decoy_manipulation_check",
        "infrastructure_fault",
        "decoy manipulation check",
        "decoy manipulation check FAILED ...",
    ),
    ("scan", "infrastructure_fault", "SubprocessTransport.scan", "live scan failed twice: ..."),
    (
        "delivery",
        "infrastructure_fault",
        "SubprocessTransport.deliver",
        "the <service> primitive timed out after ... / returned no verdict ...; "
        "connect probe of <service> failed twice ...",
    ),
    ("api", "api_overload", "the attacker client", "(any message)"),
)
UNCLASSIFIED_CAUSE = "unclassified"

_PRIMITIVE_RE = re.compile(r"^the (\S+) primitive (?:timed out after|returned no verdict)")
_PROBE_RE = re.compile(r"^connect probe of (\S+) failed twice")
_CONDITION_RE = re.compile(r"^(.*)-s\d+(?:-.*)?$")


def classify_exclusion(reason: str | None, detail: str | None) -> tuple[str, str | None]:
    """(cause, service) for one lost sequence, from its reason and the fault's own message.

    ``service`` names the primitive or decoy for a delivery fault, else None. A message that
    matches no row of metrics spec §8a is ``unclassified`` (its reason and message are still
    reported).
    """

    text = " ".join(str(detail or "").split())
    if str(reason or "") == "api_overload":
        return "api", None
    if text.startswith("ansible reconfigure for condition"):
        return "reconfiguration", None
    if text.startswith("readiness gate:"):
        return "readiness_gate", None
    if text.startswith("decoy manipulation check FAILED"):
        return "decoy_manipulation_check", None
    if text.startswith("live scan failed twice"):
        return "scan", None
    for rx in (_PRIMITIVE_RE, _PROBE_RE):
        m = rx.match(text)
        if m:
            return "delivery", m.group(1)
    return UNCLASSIFIED_CAUSE, None


def _condition_of(sequence_id: str) -> str:
    m = _CONDITION_RE.match(str(sequence_id))
    return m.group(1) if m else UNSPECIFIED_REASON


def sequence_exclusion_details(records) -> dict:
    """Each condition's excluded sequences and their causes (metrics spec §8a).

    ``records`` is ``run_meta.json`` -> ``excluded_sequence_details`` (or
    ``ExperimentDataset.excluded_sequences``): one ``{sequence_id, reason, detail}`` per lost
    sequence. The condition is the sequence id's prefix.
    """

    recs = [r for r in (records or []) if isinstance(r, dict)]
    by_condition: dict[str, dict] = {}
    by_cause: dict[str, int] = {}
    delivery: dict[str, int] = {}
    for r in recs:
        sid = str(r.get("sequence_id") or "")
        cause, service = classify_exclusion(r.get("reason"), r.get("detail"))
        cond = _condition_of(sid)
        entry = by_condition.setdefault(
            cond, {"n_excluded_sequences": 0, "by_cause": {}, "sequences": []}
        )
        entry["n_excluded_sequences"] += 1
        entry["by_cause"][cause] = entry["by_cause"].get(cause, 0) + 1
        entry["sequences"].append(
            {
                "sequence_id": sid,
                "reason": r.get("reason"),
                "cause": cause,
                **({"service": service} if service else {}),
                "detail": r.get("detail"),
            }
        )
        by_cause[cause] = by_cause.get(cause, 0) + 1
        if service:
            delivery[service] = delivery.get(service, 0) + 1
    return {
        "n_excluded_sequences": len(recs),
        "n_with_detail": int(sum(1 for r in recs if r.get("detail"))),
        "by_condition": dict(sorted(by_condition.items())),
        "by_cause": dict(sorted(by_cause.items())),
        "delivery_faults_by_service": dict(sorted(delivery.items())),
        "cause_map": [
            {"cause": c, "reason": rs, "raised_by": rb, "message_begins": mb}
            for c, rs, rb, mb in EXCLUSION_CAUSES
        ],
        "note": (
            "Every lost sequence with its reason, its cause (the row of metrics spec §8a its "
            "message matches) and the fault's own message. A condition absent from by_condition "
            "lost no sequence. A delivery fault can occur only when the attacker delivers to a "
            "target, so delivery_faults_by_service shows whether such faults cluster on one "
            "primitive; if they cluster on one primitive or condition, the included sequences "
            "are no longer a random subset and that must be reported (§8a)."
        ),
    }


def _sequence_level(obj) -> dict:
    """Summarize the SEQUENCE-level exclusion log (record list, or a run_meta.json dict)."""

    if obj is None:
        return {
            "n_excluded_sequences": None,
            "note": (
                "sequence-level exclusion log not supplied to the analysis; see run_meta.json "
                "(n_excluded_sequences / n_lost_trials / excluded_sequence_reasons)."
            ),
        }
    if isinstance(obj, dict):  # a run_meta.json payload
        n = obj.get("n_excluded_sequences")
        if n is None:
            return {
                "n_excluded_sequences": None,
                "source": "run_meta.json",
                "note": (
                    "run_meta.json was supplied but carries no n_excluded_sequences key; the "
                    "sequence-level count is unavailable for this run and is NOT assumed to be 0."
                ),
            }
        lost = obj.get("n_lost_trials")
        out = {
            "n_excluded_sequences": int(n),
            "n_lost_trials": (int(lost) if lost is not None else None),
            "reasons": list(obj.get("excluded_sequence_reasons") or []),
            "source": "run_meta.json",
        }
        details = obj.get("excluded_sequence_details")
        if details is None:
            out["details"] = {
                "note": (
                    "run_meta.json carries no excluded_sequence_details (runs collected before "
                    "2026-10-27): the causes of the lost sequences are not recorded."
                )
            }
        else:
            out["details"] = sequence_exclusion_details(details)
            out["details"]["agrees_with_n_excluded_sequences"] = bool(
                out["details"]["n_excluded_sequences"] == int(n)
            )
        return out
    recs = list(obj)  # ExperimentDataset.excluded_sequences
    by_reason: dict[str, int] = {}
    for r in recs:
        k = str(r.get("reason") or UNSPECIFIED_REASON)
        by_reason[k] = by_reason.get(k, 0) + 1
    return {
        "n_excluded_sequences": len(recs),
        "n_lost_trials": int(sum(int(r.get("n_lost_attempts", 0) or 0) for r in recs)),
        "by_reason": by_reason,
        "source": "ExperimentDataset.excluded_sequences",
        "details": sequence_exclusion_details(recs),
    }


def _run_meta_agreement(obj, n_rows_written: int, n_excluded: int, by_reason: dict) -> dict | None:
    """Independent artifact-level corroboration of the row counters from run_meta.json."""

    if not isinstance(obj, dict):
        return None
    meta_rows = obj.get("n_trials")
    meta_excl = obj.get("n_excluded")
    meta_reasons = obj.get("exclusion_reasons")
    return {
        "run_meta_n_trials": (int(meta_rows) if meta_rows is not None else None),
        "run_meta_n_excluded": (int(meta_excl) if meta_excl is not None else None),
        "run_meta_exclusion_reasons": (list(meta_reasons) if meta_reasons is not None else None),
        "agrees_on_n_rows_written": (
            None if meta_rows is None else bool(int(meta_rows) == n_rows_written)
        ),
        "agrees_on_n_excluded": (None if meta_excl is None else bool(int(meta_excl) == n_excluded)),
        "agrees_on_reasons": (
            None if meta_reasons is None else bool(set(meta_reasons) == set(by_reason))
        ),
        "fault_rate_recorded": ("fault_rate" in obj),
        "run_meta_fault_rate": obj.get("fault_rate"),
        "note": (
            "run_meta.json is written by the harness at collection time from the TrialRow "
            "objects, independently of this analysis pass over the CSV; agreement is "
            "corroboration that the analyzed CSV is the run that was collected. "
            "fault_rate_recorded=false means the run predates recording the injector rate, so "
            "the injected origin of any 'infrastructure_fault' rows rests on the code path "
            "(harness/trial_runner.py is the only row-level source of that string) plus the "
            "observed rate, not on the sidecar."
        ),
    }


# --------------------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------------------


def exclusion_breakdown(
    rows_or_df,
    *,
    sequence_exclusions=None,
    outcomes: tuple[str, ...] = SENSITIVITY_OUTCOMES,
) -> dict:
    """Full a-priori-exclusion accounting from the UNSCREENED trial rows.

    Must be given the pre-screen data (``analysis.dataset.prepare`` drops excluded rows).
    Accepts ``list[TrialRow]`` or a raw DataFrame. ``sequence_exclusions`` is the
    SEQUENCE-level log -- ``ExperimentDataset.excluded_sequences`` or a parsed
    ``run_meta.json`` dict -- so the report reconciles the row counter against the run
    log's sequence counter instead of leaving them looking contradictory.
    """

    df = pd.DataFrame() if rows_or_df is None else to_frame(rows_or_df)
    # reset_index returns a NEW frame (no caller mutation) and makes every later
    # .loc[group.index] alignment safe on caller-supplied slices with duplicate labels.
    df = df.reset_index(drop=True)
    seq = _sequence_level(sequence_exclusions)

    if df.empty or "excluded" not in df.columns:
        n = int(len(df))
        return {
            "n_rows_written": n,
            "n_included": n,
            "n_excluded": 0,
            "exclusion_rate": 0.0,
            "by_reason": {},
            "sequence_level": seq,
            "reconciliation_note": (
                "no 'excluded' column, or the dataset is empty; row-level exclusion accounting "
                "is unavailable for this dataset."
            ),
        }

    excl = excluded_mask(df)
    by_reason = {str(k): int(v) for k, v in _reason_series(df)[excl].value_counts().items()}
    n_written, n_excl = int(len(df)), int(excl.sum())
    n_incl = n_written - n_excl

    by_condition: dict = {}
    if "condition" in df.columns:
        for cond, grp in df.groupby("condition", sort=True):
            m = excl.loc[grp.index]
            by_condition[str(cond)] = {
                "n_rows_written": int(len(grp)),
                "n_included": int((~m).sum()),
                "n_excluded": int(m.sum()),
                "exclusion_rate": round(float(m.mean()), 6),
            }
    by_attempt: dict = {}
    if "attempt_index" in df.columns and n_excl:
        counts = df.loc[excl, "attempt_index"].value_counts().sort_index()
        by_attempt = {str(k): int(v) for k, v in counts.items()}

    n_seq = seq.get("n_excluded_sequences")
    out = {
        "n_rows_written": n_written,
        "n_included": n_incl,
        "n_excluded": n_excl,
        "exclusion_rate": round(n_excl / n_written, 6),
        "by_reason": by_reason,
        "reason_provenance": {
            r: EXCLUSION_REASON_PROVENANCE.get(r, "reason not recognized by the analysis layer")
            for r in by_reason
        },
        "by_condition": by_condition,
        "by_attempt_index": by_attempt,
        "regeneration": _regeneration_evidence(df, excl),
        "differential_attrition_by_condition": _exclusion_homogeneity(df, excl, "condition"),
        "differential_attrition_by_attempt_index": _exclusion_homogeneity(
            df, excl, "attempt_index"
        ),
        "retention_sensitivity": _retention_sensitivity(df, excl, outcomes),
        "sequence_level": seq,
    }
    agreement = _run_meta_agreement(sequence_exclusions, n_written, n_excl, by_reason)
    if agreement is not None:
        out["run_meta_agreement"] = agreement
    out["reconciliation_note"] = (
        f"n_rows_written ({n_written}) counts every trial row the harness emitted, including "
        f"regenerated retries; n_included ({n_incl}) is the analyzed sample and n_excluded "
        f"({n_excl}) is what the a-priori inclusion rule removed, so n_included + n_excluded "
        f"== n_rows_written. The run log's per-sequence 'excluded=N' counter is a DIFFERENT "
        f"quantity: it counts whole SEQUENCES abandoned after an unrecoverable API or "
        f"provisioning fault (harness/experiment_runner.py; run_meta.json "
        f"'n_excluded_sequences'"
        + (f" = {n_seq}" if n_seq is not None else "")
        + "), not rows. An injector-flagged row-level exclusion never abandons a sequence: the "
        "attempt is regenerated immediately in the same trial window. A run can therefore "
        "correctly report 0 excluded sequences and a non-zero excluded-row count at the same "
        "time; 'regeneration' carries the row-level evidence for whether the excluded rows were "
        "superseded retries or real losses."
    )
    return out
