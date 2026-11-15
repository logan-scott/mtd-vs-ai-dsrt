"""End-to-end analysis pipeline driver.

Consolidates the dataset, runs the descriptive/assumption/RQ1-RQ5/reliability/
survival/power analyses, and applies Benjamini-Hochberg FDR within each RQ to the
secondary outcomes (the three primary outcomes -- exploit success, exploitation
success, and time-to-compromise -- are evaluated at alpha = .05 without correction).
Primary between-condition comparisons model the within-sequence clustering (GLMM /
GEE / cluster-robust Cox). Every step degrades gracefully on degenerate input.

DATA-PROVENANCE CAVEAT (2026-09-02) — this module analyzes whatever frame it is handed
and cannot detect a substrate fault, so the caveat lives here where the report is built:
every ``deceptive_presentation`` and ``combined_mtd_deception`` figure in the runs frozen
under ``docs/results/`` (including ``full_haiku_20260831``) **predates the decoy-publication
fix** and measures an arm whose manipulation never reached the wire. A rotated decoy was
never re-published on the host, so from ``attempt_index == 1`` onward there was no decoy
reachable at any port; the 33 ``deceptive_presentation`` trials with
``recon_accuracy_f1 != 1.000`` are exactly the 33 ``attempt_index == 0`` trials. Those two
arms must be RE-RUN before any deception result is reported; the other four arms are
unaffected (no technique but deception mutates a decoy, and their rendered decoy mappings
are byte-identical to the base compose file's). See
``docs/metrics_specification.md`` §4c, ``docs/endpoint_mutation_live.md`` and
``docs/null_condition_audit.md``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import (
    action_trace,
    adaptation,
    assumptions,
    descriptives,
    drift,
    power,
    race,
    rq1,
    rq2,
    rq3,
    rq4,
    rq5,
    rq5_profile,
    survival,
    targeting,
)
from analysis.dataset import prepare, to_frame
from analysis.degeneracy import variance_profile
from analysis.exclusions import excluded_mask, exclusion_breakdown
from analysis.fdr import bh_family
from analysis.reliability import reliability_report
from measurement.models import TrialRow


def _dig(d: dict, *keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


# The pre-registered secondary-outcome families, one per RQ, with the report path the
# outcome's CITABLE p-value lives at. Family membership -- and therefore the BH
# denominator -- is fixed here, before any data is seen; it is never edited in response
# to which outcomes turn out to be estimable.
#
# A value is either a path tuple, or a callable ``resolver(rq_results) -> path tuple``
# for a family whose citable estimate is chosen by a pre-specified assumption check
# rather than being fixed outright. A resolver may read model diagnostics; it must
# never read the magnitude of a p-value, which would make estimator selection
# conditional on the result it produces.
SecondaryPath = tuple[str, ...]

_P_SOURCE_NOTE = (
    "Since 2026-10-28 (approved by the chair) the RQ1 family's p for "
    "time_to_first_valid_exploit_s is the 300-s RMST difference's: "
    "rmst_300s.mtd_vs_static.p_normal_approx, an unfloored Wald p built from the sequence "
    "cluster-bootstrap SE, whatever the proportional-hazards screen says. The percentile "
    "bootstrap p in the same block is floored at 1/(B+1) and is deliberately NOT used here, "
    "because a floored p can sit above a small Benjamini-Hochberg threshold and spuriously fail "
    "to reject. Only when that p does not exist (no cluster bootstrap could be run) does the "
    "earlier rule apply: the trial-window PH-free RMST difference "
    "(ph_robust.rmst_difference.p_normal_approx) when the Schoenfeld screen rejects "
    "proportional hazards, else the Cox hazard ratio's p. Every switch is keyed on availability "
    "or on the assumption check, never on a p-value's magnitude. rq1_p_source records the full "
    "resolved path including the leaf, which refines the family member's own p_source (that "
    "label stops at the estimate block and so cannot distinguish the floored p from the "
    "unfloored one)."
)


def _ph_robust_or_cox(outcome: str) -> Callable[[dict], SecondaryPath]:
    """Resolver: the PH-free RMST p when PH is rejected, else the pre-registered Cox p.

    The pre-registered estimate for an RQ1 time-to-event outcome is the cluster-robust
    Cox hazard ratio. That hazard ratio only exists as a constant quantity if the
    proportional-hazards assumption holds; when the Schoenfeld screen rejects it, the HR
    is a follow-up-time-averaged summary of an effect that moves, and it cannot honestly
    carry the family's multiplicity-controlled inference while the same report stamps it
    ``ph_valid: false``. The citable p then comes from the PH-free RMST difference.

    **The switch reads the assumption check only.** ``ph_robust.triggered`` is set from
    the Schoenfeld screen (see ``rq1._ph_robust_block``), which is a pre-specified
    diagnostic; neither p-value's magnitude is inspected, so this is not p-value-conditional
    estimator selection. The secondary guard -- that ``p_normal_approx`` actually exists --
    tests availability, not size: a bootstrap that failed to produce an SE yields no p at
    all, and falling back to the pre-registered Cox path is strictly better than reporting
    the hypothesis as untested. Family membership, and therefore the BH denominator, is
    unaffected either way: it is fixed by hypothesis, not by which model supplies the p.
    """

    def resolve(res: dict) -> SecondaryPath:
        ph = _dig(res, outcome, "ph_robust") or {}
        rmst = ph.get("rmst_difference")
        if (
            ph.get("triggered")
            and isinstance(rmst, dict)
            and rmst.get("p_normal_approx") is not None
        ):
            return (outcome, "ph_robust", "rmst_difference", "p_normal_approx")
        return (outcome, "cox_frailty", "mtd_vs_static", "p")

    return resolve


def _rmst_300s_or_previous(outcome: str) -> Callable[[dict], SecondaryPath]:
    """Resolver: the 300-s RMST p (the decision estimate since 2026-10-28), else the earlier rule.

    ``rq1.<outcome>.rmst_300s.mtd_vs_static`` decides the time endpoint (runbook §1; the rule
    is recorded in the report as ``rmst_300s.decision_rule``), so its unfloored Wald p is the
    family's p whether or not proportional hazards hold. The guard tests availability only:
    when the cluster bootstrap could not run there is no ``p_normal_approx``, and the earlier
    pre-specified rule (``_ph_robust_or_cox``) applies rather than leaving the hypothesis
    untested. No p-value's magnitude is read.
    """

    previous = _ph_robust_or_cox(outcome)

    def resolve(res: dict) -> SecondaryPath:
        r300 = _dig(res, outcome, "rmst_300s", "mtd_vs_static")
        if isinstance(r300, dict) and r300.get("p_normal_approx") is not None:
            return (outcome, "rmst_300s", "mtd_vs_static", "p_normal_approx")
        return previous(res)

    return resolve


SECONDARY_FAMILIES: dict[str, dict[str, SecondaryPath | Callable[[dict], SecondaryPath]]] = {
    # TFVE is secondary. Since 2026-10-28 its p is the 300-s RMST difference's (the time
    # endpoint's decision estimate); the earlier Cox / PH-free rule is the fallback when that
    # p does not exist.
    "rq1": {
        "time_to_first_valid_exploit_s": _rmst_300s_or_previous("time_to_first_valid_exploit_s")
    },
    # RQ2 continuous secondary outcomes. The family consumes the p-value of each
    # outcome's REPORTED result (rq2.<outcome>.reported_result) -- the clustered LMM's
    # Wald omnibus when that model is numerically sound, the promoted per-sequence
    # independent-cluster test when it is not, and nothing at all when the outcome has
    # no variance to test. One p per hypothesis, and it is the same p the write-up
    # cites, so the reported `reject` flags describe the reported tests.
    "rq2": {
        "time_to_enumeration_s": ("time_to_enumeration_s", "reported_result", "p"),
        "recon_accuracy_f1": ("recon_accuracy_f1", "reported_result", "p"),
        "attack_path_churn": ("attack_path_churn", "reported_result", "p"),
    },
}


def _resolve_path(spec, res: dict) -> SecondaryPath:
    """A family member's report path, resolving a callable spec against ``res``."""

    return tuple(spec(res)) if callable(spec) else tuple(spec)


def _p_or_skip(res: dict, path: SecondaryPath) -> tuple[float | None, dict | None]:
    """The p-value at ``path``, or the most specific reason there is not one.

    Walks the path from the deepest node outward so a failure recorded at *any*
    intermediate level (e.g. ``rq1[outcome]["cox_frailty"]["note"]``) is named
    accurately instead of collapsing to a generic "no p-value here".
    """

    p = _dig(res, *path)
    if p is not None and p == p:
        return float(p), None
    for depth in range(len(path) - 1, 0, -1):
        node = _dig(res, *path[:depth])
        if not isinstance(node, dict):
            continue
        if node.get("skipped_reason"):
            return None, {"reason": node["skipped_reason"], "detail": node.get("note")}
        if node.get("clamped"):
            return None, {
                "reason": "numerically_degenerate_contrasts",
                "detail": node.get("note"),
            }
        if "note" in node:
            return None, {"reason": "model_unavailable", "detail": node["note"]}
    return None, {"reason": "p_unavailable", "detail": f"no p-value at {'.'.join(path)}"}


def _p_source(res: dict, path: SecondaryPath) -> str:
    """Provenance label for the estimate at ``path``.

    Prefers the estimate block's own self-description (``reported_result`` carries
    ``source``/``method``), and otherwise falls back to the dotted report path, so a
    later change of path is self-documenting without a parallel label table.
    """

    node = _dig(res, *path[:-1])
    if isinstance(node, dict) and node.get("source"):
        method = node.get("method")
        return f"{node['source']}:{method}" if method else str(node["source"])
    return ".".join(path[1:-1]) or ".".join(path)


def _fdr_secondary(rq1r: dict, rq2r: dict) -> dict:
    """BH-FDR over secondary-outcome p-values within each RQ.

    Primary outcomes (exploit_success, exploitation_success, time_to_compromise) are
    evaluated at alpha .05 without correction; the secondary outcomes in
    ``SECONDARY_FAMILIES`` are FDR-controlled.

    Each family's BH denominator is its **pre-registered** size (``n_planned``), not
    the number of outcomes that turned out to be estimable, so the correction cannot
    loosen in response to the data. Outcomes with no p-value are listed under
    ``skipped`` with a machine-readable reason rather than emitted with a null
    rejection (which would read as "tested and inconclusive" for a hypothesis that was
    never tested). Every member also carries ``p_source`` so a reader can see which
    estimate the correction was applied to; ``rq1_p_source`` additionally records the
    RQ1 family's fully resolved report path (including the leaf key), because that
    family's estimate is the 300-s RMST p with a pre-specified fallback -- see
    ``_rmst_300s_or_previous``, ``_ph_robust_or_cox`` and ``p_source_note``.
    """

    sources = {"rq1": rq1r, "rq2": rq2r}
    out: dict = {}
    for fam, specs in SECONDARY_FAMILIES.items():
        res = sources[fam]
        pvals: dict[str, float | None] = {}
        skipped: dict[str, dict] = {}
        labels: dict[str, str] = {}
        resolved: dict[str, str] = {}
        for outcome, spec in specs.items():
            path = _resolve_path(spec, res)
            p, why = _p_or_skip(res, path)
            pvals[outcome] = p
            labels[outcome] = _p_source(res, path)
            resolved[outcome] = ".".join(path[1:])
            if why is not None:
                skipped[outcome] = why
        fam_out = bh_family(pvals, skipped=skipped, family_size=len(specs))
        for name, member in fam_out["tests"].items():
            member["p_source"] = labels.get(name, "unavailable")
        for name, member in fam_out["skipped"].items():
            member["p_source"] = labels.get(name, "unavailable")
        out[fam] = fam_out
        if fam == "rq1":
            out["rq1_p_source"] = resolved
    out["family_note"] = (
        "Benjamini-Hochberg is applied within each RQ with the denominator fixed at the "
        "pre-registered family size (n_planned = bh_denominator). Outcomes under "
        "'skipped' produced no p-value -- they were not tested, cannot be rejected, and "
        "are retained in the denominator, which is conservative and keeps the family "
        "independent of the observed data. 'p_source' names the estimate each p came "
        "from: for RQ2 that is rq2.<outcome>.reported_result, which is 'lmm:...' when "
        "the clustered mixed model is numerically sound and "
        "'sensitivity:kruskal_wallis_per_sequence' when it was degenerate and the "
        "pre-specified per-sequence independent-cluster test was promoted."
    )
    out["p_source_note"] = _P_SOURCE_NOTE
    return out


# Outcomes whose invariance, if observed, is a property of the instrument rather than
# of this run. Surfaced only when the screen actually finds the outcome constant, so
# the note is a confirmed mechanism and never an unchecked assertion.
STRUCTURAL_INVARIANTS: dict[str, str] = {
    "time_to_enumeration_s": (
        "Structurally invariant, not sampling degeneracy: the simulated attacker stamps "
        "the recon command at cursor + _RECON_CMD_OFFSET_S (5 s) and the recon snapshot "
        "that verifies obj_a at cursor + _RECON_DURATION_S (30 s) (attacker/planner.py), "
        "so whenever obj_a is verified at the first recon snapshot this metric is "
        "identically 25 s and cannot respond to the defense condition."
    ),
}

SCREENED_OUTCOMES = [
    "exploit_success",
    "exploitation_success",
    "time_to_compromise_s",
    "time_to_first_valid_exploit_s",
    "time_to_enumeration_s",
    "recon_accuracy_f1",
    "attack_path_churn",
    "cpu_overhead_pct",
    "latency_overhead_ms",
    "memory_overhead_pct",
    "network_overhead_pct",
    "operational_event_volume",
    "service_availability_impact",
]


def _outcome_variance_screen(df) -> dict:
    """One place a reader can check which outcomes were estimable, and why not.

    This is the authoritative variance screen: it uses a scale-relative tolerance and
    covers every condition, whereas ``primary_outcome_variance_check`` is a narrower
    exact-zero check on the MTD arm of the two graded primaries.
    """

    profiles = {oc: variance_profile(df, oc, "condition") for oc in SCREENED_OUTCOMES}
    constant = sorted(k for k, v in profiles.items() if v.get("status") == "constant")
    partial = sorted(k for k, v in profiles.items() if v.get("status") == "some_groups_constant")
    return {
        "profiles": profiles,
        "constant_outcomes": constant,
        "partially_constant_outcomes": partial,
        "structural_invariance": {
            k: STRUCTURAL_INVARIANTS[k] for k in constant if k in STRUCTURAL_INVARIANTS
        },
        "note": (
            "constant_outcomes have no resolvable variance anywhere: no model was fit, no "
            "test was performed, and they are recorded under fdr_secondary.<rq>.skipped "
            "while remaining in the BH denominator. partially_constant_outcomes vary "
            "between conditions and remain fully estimable, but a random-intercept LMM "
            "fitted to one is poorly conditioned -- see rq2.<outcome>.lmm_diagnostics and "
            "rq2.<outcome>.reported_result for which estimate is citable. This screen is "
            "the authoritative one; primary_outcome_variance_check is a narrower "
            "exact-zero check on the MTD arm of the graded primaries."
        ),
    }


def _variance_check(df) -> dict:
    """Flag degenerate (zero-variance) primary outcomes in the MTD arm.

    The graded/varianced binary is contingent on the reconfiguration cadence/phase
    landing in the attacker's action window; a mis-calibrated cadence (e.g. far above
    the exploit offset with the phase disabled) can silently return a zero-variance
    ``exploitation_success`` column that the GLMM/GEE and design-effect steps then
    consume as if informative. This surfaces that explicitly instead of failing silently.
    """

    import pandas as pd

    out: dict = {}
    mtd = df[df["defense"] == "mtd"] if "defense" in df.columns else df
    for oc in ("exploitation_success", "time_to_compromise_s"):
        if oc not in df.columns:
            continue
        s = pd.to_numeric(mtd[oc], errors="coerce").dropna()
        sd = float(s.std(ddof=1)) if len(s) > 1 else 0.0
        out[oc] = {
            "mtd_arm_sd": sd,
            "degenerate": bool(len(s) < 2 or sd == 0.0),
            "n": int(len(s)),
        }
    out["note"] = (
        "degenerate=true means the MTD arm has no variance on that primary — check "
        "mtd.rotation_phase (should be 'randomized') and that cadence_s lands a tick in the "
        "recon->exploit window; the downstream models are uninformative until it is non-degenerate."
    )
    return out


def _as_frame(trial_rows: list[TrialRow] | pd.DataFrame) -> pd.DataFrame:
    """Normalize the pipeline input to an UNSCREENED trial-level DataFrame.

    ``prepare`` already accepts either form; this exists so the exclusion count can be
    taken from the frame *before* screening regardless of which form the caller passed.
    Delegates to ``analysis.dataset.to_frame`` so there is exactly one normalizer.
    """

    return to_frame(trial_rows)


def _n_excluded(frame: pd.DataFrame) -> int:
    """Count a-priori-excluded rows in an UNSCREENED frame.

    Delegates to ``analysis.exclusions.excluded_mask``, the exact complement of
    ``analysis.dataset.screen()``, so that for EVERY encoding of ``excluded`` (int, bool,
    float-with-NaN, or a string column from a third-party writer) this count is precisely
    the number of rows the screen removes -- i.e. ``n_trials_included + n_excluded ==
    n_rows_written`` holds by construction, and the top-level counter can never disagree
    with ``exclusions.n_excluded``. A heuristic that merely recognizes "truthy" codes
    cannot promise that: on ``excluded = ["True", "False", "true"]`` the screen keeps
    nothing (three rows removed) while a truthiness test reports two.

    Returns 0 when the column is absent: an already-screened frame reports no exclusions
    by definition, which is the honest answer rather than a crash. ``run_all`` reads the
    published figure out of ``exclusions`` instead; this is the standalone counter for
    callers (tests, tooling) that want the number without the whole block.

    NOTE ON DENOMINATORS: this is the ANALYSIS-level count of trial rows screened out of
    the modeled sample (63 in the confirmatory run: superseded infrastructure-fault rows
    that were retried successfully). It is deliberately NOT the RUN-level
    ``run_meta.json:n_excluded_sequences`` (0), which counts sequences that permanently
    lost an attempt. Both are correct over different denominators; ``results["exclusions"]``
    reconciles them in full.
    """

    if "excluded" not in frame.columns:
        return 0
    return int(excluded_mask(frame).sum())


_H05_DECISION_NOTE = (
    "H05 is decided by benchmark_profile: the security benefit is the RMST difference from the "
    "static baseline over [0, 300 s] (decided 2026-10-28 and approved by the chair, the same "
    "horizon as RQ1's time endpoint), divided by the technique's median benchmarked "
    "reconfiguration cost; the planned contrast is h05_primary_contrast. "
    "benchmark_profile_trial_window_sensitivity repeats it over the 1,800-s trial window. The "
    "per-trial blocks above it (by_condition, the per-sequence contrasts, the frontier) use the "
    "trial window and the per-trial cost columns; cite their cost side only when "
    "cost_column_provenance says those columns are measured."
)


def _rq5_block(df: pd.DataFrame, bench: pd.DataFrame | None, bench_provenance: dict | None) -> dict:
    """``rq5.rq5`` plus the cost-column provenance and the benchmark profile test.

    ``benchmark_profile`` is the H05 decision at ``rq5_profile.DECISION_HORIZON_S`` (300 s);
    ``benchmark_profile_trial_window_sensitivity`` is the same test over the trial window.
    """

    out = rq5.rq5(df)
    out["cost_column_provenance"] = rq5_profile.cost_column_provenance(df)
    if bench is None:
        not_run = (
            "No reconfiguration-cost benchmark was supplied (cli.py analyze "
            "--reconfig-bench PATH), so the H05 benefit/cost profile test was not run."
        )
        out["benchmark_profile"] = {"note": not_run}
        out["benchmark_profile_trial_window_sensitivity"] = {"note": not_run}
    else:
        out["benchmark_profile"] = {
            "role": (
                "DECISION ESTIMATE for H05 (h05_primary_contrast), benefit over "
                f"[0, {rq5_profile.DECISION_HORIZON_S:g} s]"
            ),
            **rq5_profile.benchmark_profile(
                df, bench, horizon=rq5_profile.DECISION_HORIZON_S, bench_provenance=bench_provenance
            ),
        }
        out["benchmark_profile_trial_window_sensitivity"] = {
            "label": "SENSITIVITY: the trial window (1,800 s in the study) as the benefit horizon",
            **rq5_profile.benchmark_profile(df, bench, bench_provenance=bench_provenance),
        }
    out["h05_decision_note"] = _H05_DECISION_NOTE
    return out


def _design_checks(df: pd.DataFrame, design_basis: dict | None) -> dict:
    """Drift across run order, in the form the run's design calls for.

    The run order is the one the run's manifest declares (``design_basis["run_order"]``) when
    it is supplied, else the one the trials' real start times show (``drift.detect_run_order``).
    """

    detected = drift.detect_run_order(df)
    declared = (design_basis or {}).get("run_order")
    run_order = declared or detected.get("run_order")
    out: dict = {
        "run_order": {
            "value": run_order,
            "declared": declared,
            "declared_source": (design_basis or {}).get("source") if declared else None,
            "detected": detected,
            "declared_matches_detected": (
                None
                if not (declared and detected.get("run_order"))
                else bool(declared == detected["run_order"])
            ),
        },
        "within_block_drift": drift.within_block_drift(df, run_order=run_order),
    }
    if run_order == drift.RUN_ORDER_INTERLEAVED:
        out["interleaved_drift"] = drift.interleaved_drift(df)
    else:
        out["interleaved_drift"] = {
            "note": (
                f"not computed: the run order is {run_order or 'not known'}, not interleaved; "
                "within_block_drift is the drift check for a blocked run"
            )
        }
    return out


def run_all(
    trial_rows: list[TrialRow] | pd.DataFrame,
    uptime_events: list[dict] | None = None,
    repeatability_rows: list[TrialRow] | pd.DataFrame | None = None,
    *,
    provenance: dict | None = None,
    sequence_exclusions=None,
    reconfig_benchmark: pd.DataFrame | None = None,
    reconfig_benchmark_provenance: dict | None = None,
    design_basis: dict | None = None,
) -> dict:
    """Run the full analysis pipeline; returns a nested results dict.

    ``trial_rows`` may be run-sheet rows (a live/offline run) or an already-loaded
    UNSCREENED DataFrame (``cli.py analyze`` re-analyzing an archived dataset). Pass the
    unscreened frame: ``n_excluded`` and the whole ``exclusions`` block are computed
    before screening, so a caller that screens first silently reports zero exclusions.

    ``sequence_exclusions`` is the SEQUENCE-level exclusion log -- either
    ``ExperimentDataset.excluded_sequences`` (an in-process run) or a parsed
    ``run_meta.json`` dict (``cli.py analyze``). It is what lets ``exclusions`` reconcile
    the run log's per-sequence ``excluded=N`` counter against the row-level count instead
    of leaving the two looking contradictory; when it is absent the block says so rather
    than assuming zero.

    The positional prefix ``(trial_rows, uptime_events, repeatability_rows)`` is a fixed
    contract -- ``cli.cmd_run``, ``cli.cmd_pilot`` and the determinism tests call it with
    three positional arguments -- so every future parameter must be keyword-only with a
    default, as ``provenance`` and ``sequence_exclusions`` are.

    ``reconfig_benchmark`` is the reconfiguration-cost benchmark (the frame
    ``analysis.rq5_profile.load_benchmark`` returns; ``cli.py analyze --reconfig-bench``).
    When it is supplied, ``rq5.benchmark_profile`` holds the H05 benefit/cost profile with
    bootstrap intervals and its planned contrasts; when it is not, that block says the test
    was not run. ``reconfig_benchmark_provenance`` (path, sha256) is recorded inside it.

    ``design_basis`` is the run's design as its ``_run_manifest.json`` records it
    (``sequences_per_condition``, ``attempts_per_sequence``, ``run_order``, ``source``; read
    by ``cli.py analyze``). It sets the locked N of ``power.a_priori_design_effect`` and the N
    of ``power.rerun_planning_check``, and the run order the drift check takes. Without it the
    September run's locked N (33) stays, the planning check uses the re-run's 80, and the run
    order is read from the trials' real start times.
    """

    raw = _as_frame(trial_rows)
    df = prepare(raw)  # screens excluded, adds derived columns
    exclusions = exclusion_breakdown(raw, sequence_exclusions=sequence_exclusions)
    rq1r = rq1.rq1(df)
    rq2r = rq2.rq2(df)
    fdr_secondary = _fdr_secondary(rq1r, rq2r)
    # H02's decision (rule of 2026-10-29) reads the RQ2 secondary family for reconnaissance
    # accuracy and attack-path churn, so it is added once that family exists.
    rq2r["h02_decision"] = rq2.h02_decision(rq2r, fdr_secondary.get("rq2"))
    # What the exploits were aimed at, by service name (decision A2, 2026-10-31): descriptive,
    # added after the RQ2 family exists so it can never enter it.
    rq2r["decoy_targeting"] = targeting.decoy_targeting(df)
    # The race between the attacker's scan-decide-deliver loop and the ticks (pre-specified
    # 2026-10-31, before the second re-run's data): descriptive, outside every FDR family.
    rq2r["race"] = race.race(df)
    # exclusion_breakdown is the single source of truth for this count; _n_excluded is
    # the same mask, so n_trials_included + n_excluded == exclusions.n_rows_written.
    n_excluded = int(exclusions.get("n_excluded", 0))
    ap_de = power.a_priori_design_effect(power.run_design_basis(design_basis))
    executed = power.as_executed(df, n_excluded=n_excluded)
    realized_de = power.design_effect_adjustment(df)

    results: dict = {
        "n_trials_included": int(len(df)),
        "n_excluded": n_excluded,
        "exclusions": exclusions,
        "descriptives": descriptives.describe_by_condition(df).to_dict("records"),
        # ttc_censored (compromise/obj_b) censors heavily by design; tten_censored
        # (enumeration diagnostic) rarely — report each.
        "censoring": {
            "ttc_censored": descriptives.censoring_by_condition(df, "ttc_censored").to_dict(
                "records"
            ),
            "tten_censored": descriptives.censoring_by_condition(df, "tten_censored").to_dict(
                "records"
            ),
        },
        # DESCRIPTIVE ONLY -- a sibling of `descriptives`, not an outcome. It is not
        # modelled, produces no p-value, and is not a member of any BH family: see
        # `_fdr_secondary`, whose inputs are only `rq1r`/`rq2r` and whose family
        # membership is the module-level SECONDARY_FAMILIES literal. It is computed
        # from `df` -- the same screened frame `descriptives` receives -- so
        # planner_action_trace["n_trials"] == results["n_trials_included"].
        "planner_action_trace": action_trace.summarize_action_trace(df),
        "primary_outcome_variance_check": _variance_check(df),
        "outcome_variance_screen": _outcome_variance_screen(df),
        "assumptions": {
            "time_to_compromise_s": {
                "shapiro": assumptions.shapiro_by_group(df, "time_to_compromise_s"),
                "levene": assumptions.levene_across_groups(df, "time_to_compromise_s"),
            },
            "time_to_enumeration_s": {
                "shapiro": assumptions.shapiro_by_group(df, "time_to_enumeration_s"),
                "levene": assumptions.levene_across_groups(df, "time_to_enumeration_s"),
            },
            "repeated_measures_note": (
                "The repeated-measures structure is modeled by mixed-effects / GEE with a "
                "random intercept (or exchangeable working correlation) for sequence, which "
                "supersedes RM-ANOVA sphericity; Mauchly / Greenhouse-Geisser are therefore "
                "not required. Time-to-event PH is checked via Schoenfeld residuals in the Cox fits."
            ),
        },
        "rq1": rq1r,
        "rq2": rq2r,
        # The re-run's within-attempt adaptation indicators (descriptive) sit beside the
        # attempt-index models.
        "rq3": {**rq3.rq3(df), "adaptation_indicators": adaptation.adaptation_indicators(df)},
        "rq4": rq4.rq4(df, reconfig_benchmark),
        "rq5": _rq5_block(df, reconfig_benchmark, reconfig_benchmark_provenance),
        "fdr_secondary": fdr_secondary,
        "power": {
            "a_priori": power.a_priori(),
            # THE sample-size justification (pilot-derived, a priori).
            "a_priori_design_effect": ap_de,
            "as_executed": executed,
            # POST HOC recomputation of rho from the confirmatory data -- never the
            # sample-size justification; see power_adequacy for the comparison.
            "design_effect_adjustment": realized_de,
            "power_adequacy": power.power_adequacy(ap_de, realized_de, executed),
            # The re-run's planning check: implied power at its N (80) under the July pilot's
            # rho and the October pilot's real-clock rho (C243).
            "rerun_planning_check": power.rerun_planning_check(design_basis),
            "pilot_variance": power.pilot_variance(df),
            "within_sequence_icc_by_outcome": power.within_sequence_icc_by_outcome(
                df,
                [
                    "exploitation_success",
                    "time_to_compromise_s",
                    "exploit_success",
                    "time_to_enumeration_s",
                ],
            ),
        },
        # Design checks that are not hypothesis tests: drift across run order. The re-run
        # interleaves the conditions (C209), so its check is outcome against round across all
        # conditions (interleaved_drift); within_block_drift is kept for blocked runs.
        "design_checks": _design_checks(df, design_basis),
    }

    if uptime_events is not None and len(uptime_events):
        results["survival_uptime"] = survival.survival_uptime(uptime_events)
    if repeatability_rows is not None and len(repeatability_rows):
        results["reliability"] = reliability_report(repeatability_rows)
    if design_basis is not None:
        results["design_basis"] = dict(design_basis)
    if provenance is not None:
        results["provenance"] = provenance
    return results


def _non_finite_label(value: float) -> str:
    if math.isnan(value):
        return "NaN"
    return "Infinity" if value > 0 else "-Infinity"


def _sanitize(obj, path: str, findings: list[dict]):
    """Recursively coerce ``obj`` to strictly-valid JSON, logging every substitution.

    JSON has no literal for NaN or +/-Infinity. Each non-finite float becomes ``null`` and
    its dotted path plus original value is appended to ``findings``, so the substitution is
    auditable rather than silent. ``numpy`` scalars are unwrapped to their Python values
    (``np.float64`` already subclasses ``float`` and is handled by the first branch).
    """

    if isinstance(obj, float):
        if math.isfinite(obj):
            return float(obj)
        findings.append({"path": path, "original": _non_finite_label(obj)})
        return None
    if isinstance(obj, dict):
        return {
            k: _sanitize(v, f"{path}.{k}" if path else str(k), findings) for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v, f"{path}[{i}]", findings) for i, v in enumerate(obj)]
    if isinstance(obj, np.generic):
        return _sanitize(obj.item(), path, findings)
    return obj


def write_report(results: dict, path: str | Path) -> Path:
    """Write the results dict as STRICTLY VALID JSON. Returns the path.

    Replaces a blunt ``text.replace("NaN", "null")`` on the serialized document, which
    (a) left bare ``Infinity`` tokens in place and (b) would corrupt any string VALUE
    containing the substring "NaN". The archived confirmatory report carries three bare
    ``Infinity`` tokens (two rq4 VIFs at perfect collinearity, and the static-baseline
    Kaplan-Meier median uptime, which is infinite because that arm never fails). That is
    not valid JSON: ``json.loads(..., parse_constant=raise)`` and JavaScript's
    ``JSON.parse`` both reject the file, and ``jq`` 1.7 is worse than either -- it accepts
    the token and silently rewrites it to ``1.7976931348623157e+308``, so a reader gets a
    plausible finite number where the model produced infinity.

    Non-finite floats are written as ``null`` and every substitution is listed under the
    top-level ``json_sanitization`` key, so no information is lost: a reader can see that
    ``survival_uptime.kaplan_meier.static_baseline.median_uptime_s`` was ``Infinity`` and
    not merely unavailable. ``allow_nan=False`` makes the writer fail loudly if a
    non-finite value ever reaches the serializer by a path the sanitizer does not walk.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    findings: list[dict] = []
    safe = _sanitize(results, "", findings)
    if isinstance(safe, dict):
        safe["json_sanitization"] = {
            "non_finite_replaced_with_null": findings,
            "note": (
                "JSON has no literal for NaN/+-Infinity. Each path above held a non-finite "
                "float in the computed results and is written as null; 'original' records "
                "the value that was replaced."
            ),
        }
    path.write_text(
        json.dumps(safe, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8"
    )
    return path
