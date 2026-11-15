"""Before -> after record of the 2026-10-27 analysis corrections for one archived run.

``cli.py analysis-corrections`` re-runs the current pipeline on an archived trial dataset and
hands this module two reports: the ARCHIVED report of record (for ``full_haiku_20260902``,
``analysis_report_lockstack.json``) and the CORRECTED report the current code produced from
the same dataset. ``build`` returns one record with a before -> after table for every number
the corrections change or add; ``render_markdown`` turns it into a readable summary. Nothing
here computes a statistic: every "after" value is read from the corrected report, every
"before" value from the archived report or, for figures that no report ever archived, from
the dissertation draft (``TEXT_FIGURES``, used only for the dataset they were computed on).

The six corrections and the housekeeping items:

1. power analysis (``analysis.power``): the two-proportion formula and the chi-square effect
   size, and everything downstream of the binding requirement;
2. ICC intervals and the nested-design degrees of freedom (``power.icc_with_interval``);
3. separation-robust RQ1/RQ2 estimates (``analysis.separation``);
4. the RQ3 average attempt effect (``rq3.attempt_effects``);
5. the RQ5 benefit/cost profile bootstrap and planned contrasts (``analysis.rq5_profile``);
6. the within-block drift check (``analysis.drift``);
7-9. housekeeping: the RQ4 five-measure VIF/PCA (n = 1,386), the RQ3 post hoc sensitivity
   and the Welch df of the RQ1 per-sequence test, now archived with the report.

``consistency_check`` compares every number present in BOTH reports and lists the ones that
differ outside the intended changes. Run on the platform that produced the archived report it
should list none; on another platform it lists the numerically platform-sensitive mixed-model
blocks (see ``cli._library_versions``). Values that a later analysis update changes on
purpose (``LATER_UPDATE_PREFIXES``, 2026-10-28: the 300-s RMST decides the time endpoint, so
the RQ1 secondary family's p for time to first valid exploit comes from it; the locked N's
source names the analyzed run's own manifest) are listed apart, with the reason. Since the same
date the H05 decision block (``rq5.benchmark_profile``) uses a 300-s benefit horizon, so
``_rq5`` compares the draft's trial-window figures with
``rq5.benchmark_profile_trial_window_sensitivity``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

SEPTEMBER_RUN_DATASET_SHA256 = "bde3838913baf1811d8550e8f1409dd872a8ee943480f9bfafec40727713ea97"
TEXT_SOURCE = "dissertation draft (writing-center review copy of 2026-10-07)"

# Figures the dissertation draft reports that no archived report holds, keyed by the dataset
# they were computed from. Values are as printed in the draft.
TEXT_FIGURES: dict[str, dict] = {
    SEPTEMBER_RUN_DATASET_SHA256: {
        "rq1_welch_df": (164, "Ch 3 RQ1, 'For the second co-primary outcome...': t(164)"),
        "rq3_post_hoc": {
            "where": "Ch 3 RQ3, 'Because attempt index 0 is the only attempt...'",
            "all_attempts": {"odds_ratio": 0.912, "ci95": [0.862, 0.965], "p": 0.001, "n": 1056},
            "excluding_first": {
                "odds_ratio": 0.950,
                "ci95": [0.882, 1.024],
                "p": 0.178,
                "n": 924,
            },
            "per_condition_excluding_first": {
                "combined_mtd_deception": {"odds_ratio": 0.924, "p": 0.380},
                "endpoint_mutation": {"odds_ratio": 0.968, "p": 0.677},
                "network_shuffling": {"odds_ratio": 0.929, "p": 0.306},
                "port_rotation": {"odds_ratio": 0.984, "p": 0.809},
            },
        },
        "rq4_vif_pca": {
            "where": "Ch 3 RQ4, 'Among the trial-level measures...'",
            "n": 1386,
            "vif": {
                "exploitation_success": 9348.25,
                "time_to_compromise_s": 9372.06,
                "time_to_first_valid_exploit_s": 1.04,
                "recon_accuracy_f1": 1.26,
                "attack_path_churn": 1.70,
            },
            "pca_pct": [52.1, 20.4, 18.4],
            "pca_cumulative_pct": 90.9,
        },
        "rq5": {
            "where": (
                "Ch 3 RQ5, 'Operational cost differed...', 'The security-improvement-to-"
                "operational-cost profile...' and Table 6; Ch 3 Testing and Evaluation Results "
                "('The reconfiguration-cost benchmark executed...')"
            ),
            "profile": {
                "endpoint_mutation": 1378.3,
                "network_shuffling": 934.8,
                "combined_mtd_deception": 41.7,
                "port_rotation": 17.2,
                "deceptive_presentation": 0.3,
            },
            "single_technique_mean": 582.7,
            "ratio_text": "'approximately 14 times' the combined value (about 0.071)",
            "benefit_s": {
                "combined_mtd_deception": 1236.95,
                "endpoint_mutation": 1248.77,
                "network_shuffling": 805.78,
                "port_rotation": 483.66,
                "deceptive_presentation": 2.61,
            },
            "median_cost_s": {
                "combined_mtd_deception": 29.674,
                "endpoint_mutation": 0.906,
                "network_shuffling": 0.862,
                "port_rotation": 28.046,
                "deceptive_presentation": 9.750,
            },
            "kw_mtd": {"H": 88.15, "df": 4},
            "kw_all": {"H": 110.91, "df": 5, "eta_squared": 0.929},
        },
    }
}

COMBINED = "combined_mtd_deception"
SINGLE = ("endpoint_mutation", "network_shuffling", "port_rotation", "deceptive_presentation")
LABEL = {
    "combined_mtd_deception": "combined MTD-plus-deception",
    "endpoint_mutation": "endpoint mutation",
    "network_shuffling": "network shuffling",
    "port_rotation": "port rotation",
    "deceptive_presentation": "deceptive presentation",
    "static_baseline": "static baseline",
}
_TECH_TERM = "C(technique)[T.{}]"

# Values of the archived report that the corrections are meant to change, by path prefix.
# Anything else that differs between the two reports is listed by consistency_check, so the
# list names exact fields rather than whole blocks.
_AP = ("power", "a_priori")
_DE = ("power", "a_priori_design_effect")
_DEA = ("power", "design_effect_adjustment")
_PA = ("power", "power_adequacy")
INTENDED_CHANGE_PREFIXES: tuple[tuple[str, ...], ...] = (
    _AP + ("binary_two_proportion", "n_per_condition"),
    _AP + ("binary_chi_square", "effect_size_w"),
    _AP + ("binary_chi_square", "n_total"),
    _AP + ("per_condition_by_test", "binary_two_proportion"),
    _AP + ("per_condition_by_test", "binary_chi_square"),
    _AP + ("binding_per_condition",),
    *(
        _DE + (k,)
        for k in (
            "rho_basis",
            "within_sequence_icc_rho",
            "design_effect",
            "binding_per_condition_unadjusted",
            "adjusted_trials_per_condition",
            "sequences_per_condition",
            "sequences_per_condition_with_15pct_exclusion",
            "trials_per_condition_with_inflation",
            "locked_matches_plan",
            "note",
        )
    ),
    *(
        _DEA + (k,)
        for k in (
            "binding_per_condition_unadjusted",
            "adjusted_trials_per_condition",
            "sequences_per_condition",
            "sequences_per_condition_with_15pct_exclusion",
            "trials_per_condition_with_inflation",
        )
    ),
    *(
        _PA + (k,)
        for k in (
            "required_trials_per_condition_a_priori_uncorrected",
            "requirement_basis",
            "multiplicity_adjusted_requirement",
            "a_priori_design_effect",
            "design_effect_ratio_a_priori_over_realized",
            "effective_n_per_condition_at_a_priori_de",
            "effective_n_over_uncorrected_requirement_at_a_priori_de",
            "effective_n_over_corrected_requirement_at_a_priori_de",
            "effective_n_over_uncorrected_requirement_at_realized_de",
            "note",
        )
    ),
    ("provenance",),
    ("json_sanitization",),
    ("rq3", "recon_accuracy_f1", "note"),
    ("rq3", "time_to_enumeration_s", "note"),
)

# Values of the archived report that a LATER analysis update changes on purpose, by path prefix,
# with the reason. They are not part of the 2026-10-27 corrections, so consistency_check lists
# them apart from both those corrections and the unexpected changes.
_TIME_ENDPOINT_DECISION = (
    "2026-10-28, approved by the chair: the 300-s RMST difference decides the time endpoint, so "
    "the RQ1 secondary family takes time_to_first_valid_exploit_s's p from "
    "rmst_300s.mtd_vs_static.p_normal_approx"
)
_RUN_MANIFEST_N = (
    "2026-10-28 re-run analysis update: cli.py analyze reads the locked N from the analyzed "
    "run's own _run_manifest.json, so the source names that file (the N is unchanged)"
)
LATER_UPDATE_PREFIXES: dict[tuple[str, ...], str] = {
    ("fdr_secondary", "rq1", "tests", "time_to_first_valid_exploit_s"): _TIME_ENDPOINT_DECISION,
    ("fdr_secondary", "rq1", "skipped", "time_to_first_valid_exploit_s"): _TIME_ENDPOINT_DECISION,
    ("fdr_secondary", "rq1_p_source", "time_to_first_valid_exploit_s"): _TIME_ENDPOINT_DECISION,
    ("fdr_secondary", "p_source_note"): _TIME_ENDPOINT_DECISION,
    ("power", "a_priori_design_effect", "locked_source"): _RUN_MANIFEST_N,
}

NOT_REPORTED = "not reported"
_MISSING = object()


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def dig(d: Any, path: Iterable[str], default: Any = None) -> Any:
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _path(path: Iterable[str]) -> str:
    return ".".join(path)


def fnum(x: Any, nd: int = 3) -> str:
    """A number for a table cell (thousands separators; tiny/huge values in e-notation)."""

    if x is None:
        return "--"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return f"{x:,}"
    if isinstance(x, float):
        if not math.isfinite(x):
            return str(x)
        ax = abs(x)
        if ax and (ax < 10 ** (-nd) or ax >= 1e7):
            return f"{x:.2e}"
        return f"{x:,.{nd}f}"
    return str(x)


def fsig(x: Any, sig: int = 3) -> str:
    """A value spanning orders of magnitude (prior sweeps): ``sig`` significant figures."""

    if x is None:
        return "--"
    ax = abs(float(x))
    if ax and (ax >= 1e7 or ax < 1e-3):
        return f"{x:.{sig - 1}e}"
    if ax >= 100:
        return f"{x:,.0f}"
    return f"{x:.{sig}g}"


def fp(p: float | None, floor: float | None = None) -> str:
    """APA-style p (no leading zero). A bootstrap p at its floor reads as a bound."""

    if p is None:
        return "p --"
    if floor is not None and p <= floor * (1 + 1e-9):
        return "p < .001" if floor < 0.001 else "p <= " + f"{floor:.4f}".lstrip("0")
    if p < 0.001:
        return "p < .001"
    return "p = " + f"{p:.3f}".lstrip("0")


def fci(ci: list | None, nd: int = 3) -> str:
    if not ci or len(ci) != 2:
        return ""
    return f"[{fnum(ci[0], nd)}, {fnum(ci[1], nd)}]"


def fest(est: float | None, ci: list | None = None, p: float | None = None, nd: int = 3) -> str:
    parts = [fnum(est, nd)]
    if ci:
        parts.append(fci(ci, nd))
    out = " ".join(parts)
    if p is not None:
        out += f", {fp(p)}"
    return out


def f_or(block: dict | None, nd: int = 3) -> str:
    if not isinstance(block, dict) or "odds_ratio" not in block:
        return NOT_REPORTED
    return fest(block.get("odds_ratio"), block.get("ci95"), block.get("p"), nd)


def row(
    label: str,
    before: Any,
    after: Any,
    *,
    before_src: str,
    after_src: str,
    kind: str = "changed",
    note: str | None = None,
    before_value: Any = _MISSING,
    after_value: Any = _MISSING,
) -> dict:
    """One table row: display strings plus the raw values they were rendered from."""

    out = {
        "quantity": label,
        "before": before,
        "after": after,
        "before_source": before_src,
        "after_source": after_src,
        "kind": kind,
    }
    if before_value is not _MISSING:
        out["before_value"] = before_value
    if after_value is not _MISSING:
        out["after_value"] = after_value
    if note:
        out["note"] = note
    return out


def num_row(
    label: str,
    A: dict,
    C: dict,
    path: tuple[str, ...],
    *,
    after_path: tuple[str, ...] | None = None,
    nd: int = 3,
    note: str | None = None,
) -> dict:
    """A row whose before and after are plain numbers at (usually) the same path."""

    ap = after_path or path
    b, a = dig(A, path), dig(C, ap)
    kind = "unchanged" if _same(b, a) else ("added" if b is None else "changed")
    return row(
        label,
        fnum(b, nd) if b is not None else NOT_REPORTED,
        fnum(a, nd),
        before_src=f"archived: {_path(path)}" if b is not None else "--",
        after_src=f"corrected: {_path(ap)}",
        kind=kind,
        note=note,
        before_value=b,
        after_value=a,
    )


def _kind_archived(text_value: float | None, value: float | None, nd: int) -> str:
    """Kind for a draft figure now produced by the pipeline: does it match at nd decimals?"""

    if value is None:
        return "not computed"
    if text_value is None:
        return "added"
    return "archived" if round(value, nd) == round(text_value, nd) else "archived (differs)"


def _kind_1dp(text_value: float | None, value: float | None) -> str:
    """Kind for a figure the draft printed to one decimal and the report now carries a CI."""

    if text_value is None:
        return "added"
    if value is None:
        return "not computed"
    return "interval added" if _same(round(value, 1), text_value) else "changed"


def _same(a: Any, b: Any, sig: int = 6) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not (math.isfinite(a) and math.isfinite(b)):
            return a == b
        return float(f"{a:.{sig}g}") == float(f"{b:.{sig}g}")
    return a == b


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #


def _power(A: dict, C: dict) -> dict:
    ap, de, ad, dea = _AP, _DE, _PA, _DEA
    fo = de + ("formula_correction_only",)
    rows = [
        num_row(
            "Two-proportion requirement, trials per condition (OR = 2.0, .50 vs .667)",
            A,
            C,
            ap + ("binary_two_proportion", "n_per_condition"),
            nd=0,
        ),
        num_row("Chi-square effect size w", A, C, ap + ("binary_chi_square", "effect_size_w")),
        num_row("Chi-square requirement, total N", A, C, ap + ("binary_chi_square", "n_total")),
        num_row(
            "Chi-square requirement, per condition",
            A,
            C,
            ap + ("per_condition_by_test", "binary_chi_square"),
        ),
        num_row(
            "Binding (largest) unadjusted requirement, trials per condition",
            A,
            C,
            ap + ("binding_per_condition",),
        ),
        num_row(
            "Pilot rho used for the design effect (see section 2)",
            A,
            C,
            de + ("within_sequence_icc_rho",),
            nd=4,
        ),
        num_row("Design effect, 1 + 7 rho", A, C, de + ("design_effect",), nd=1),
        num_row(
            "Requirement x design effect, trials per condition",
            A,
            C,
            de + ("adjusted_trials_per_condition",),
        ),
        num_row("-> sequences per condition", A, C, de + ("sequences_per_condition",)),
        num_row(
            "-> sequences per condition with the 15% allowance",
            A,
            C,
            de + ("sequences_per_condition_with_15pct_exclusion",),
        ),
        num_row(
            "-> trials per condition with the allowance",
            A,
            C,
            de + ("trials_per_condition_with_inflation",),
        ),
        row(
            "Sequences per condition with allowance over the pilot rho's 95% interval",
            NOT_REPORTED,
            " to ".join(
                fnum(x)
                for x in dig(
                    C,
                    de
                    + (
                        "requirement_over_rho_ci95",
                        "sequences_per_condition_with_15pct_exclusion",
                    ),
                    [],
                )
            )
            or NOT_REPORTED,
            before_src="--",
            after_src=f"corrected: {_path(de + ('requirement_over_rho_ci95',))}",
            kind="added",
        ),
        num_row(
            "Formula correction alone (planning rho .3143 kept): sequences with allowance",
            A,
            C,
            de + ("sequences_per_condition_with_15pct_exclusion",),
            after_path=fo + ("sequences_per_condition_with_15pct_exclusion",),
            note=(
                "Isolates the formula correction: the corrected requirement at the miscomputed "
                "planning rho. The headline chain also corrects rho."
            ),
        ),
        num_row(
            "Locked N (33 sequences) matches the a priori plan",
            A,
            C,
            de + ("locked_matches_plan",),
        ),
        num_row(
            "Bonferroni requirement (10 pairwise contrasts, alpha = .005), trials per condition",
            A,
            C,
            ad + ("multiplicity_adjusted_requirement", "binding_per_condition"),
        ),
        num_row(
            "Effective trials per condition at the a priori design effect",
            A,
            C,
            ad + ("effective_n_per_condition_at_a_priori_de",),
            nd=1,
        ),
        num_row(
            "Effective trials / requirement",
            A,
            C,
            ad + ("effective_n_over_uncorrected_requirement_at_a_priori_de",),
            nd=2,
        ),
        num_row(
            "Effective trials / Bonferroni requirement",
            A,
            C,
            ad + ("effective_n_over_corrected_requirement_at_a_priori_de",),
            nd=2,
        ),
        num_row(
            "Formula correction alone: effective trials (DE 3.2) / requirement",
            A,
            C,
            ad + ("effective_n_over_uncorrected_requirement_at_a_priori_de",),
            after_path=ad + ("effective_n_over_requirement_formula_correction_only",),
            nd=2,
        ),
        num_row(
            "Effective trials at the realized DE / requirement (post hoc)",
            A,
            C,
            ad + ("effective_n_over_uncorrected_requirement_at_realized_de",),
            nd=2,
        ),
        num_row(
            "A priori DE / realized DE",
            A,
            C,
            ad + ("design_effect_ratio_a_priori_over_realized",),
            nd=1,
        ),
    ]
    for key, label in (
        ("adjusted_trials_per_condition", "requirement x realized DE, trials per condition"),
        ("sequences_per_condition", "sequences per condition"),
        ("sequences_per_condition_with_15pct_exclusion", "sequences per condition with allowance"),
        ("trials_per_condition_with_inflation", "trials per condition with allowance"),
    ):
        rows.append(num_row(f"Post hoc chain at the realized rho: {label}", A, C, dea + (key,)))
    unchanged = [
        num_row(
            "t-test requirement (d = 0.50), per condition",
            A,
            C,
            ap + ("continuous_ttest", "n_per_condition"),
        ),
        num_row(
            "ANOVA requirement (f = 0.25), total N", A, C, ap + ("anova_five_group", "n_total")
        ),
        num_row(
            "Survival requirement (HR = 2.0), events",
            A,
            C,
            ap + ("censored_survival", "required_events"),
        ),
        num_row("Realized design effect", A, C, ad + ("realized_design_effect",), nd=1),
    ]
    unchanged = [u for u in unchanged if u["kind"] == "unchanged"]
    req = dig(C, ap + ("binding_per_condition",))
    seqs = dig(C, de + ("sequences_per_condition_with_15pct_exclusion",))
    seqs_fo = dig(C, fo + ("sequences_per_condition_with_15pct_exclusion",))
    rng = dig(C, de + ("requirement_over_rho_ci95", "sequences_per_condition_with_15pct_exclusion"))
    trials = dig(C, ad + ("realized_trials_per_condition_min",))
    eff = dig(C, ad + ("effective_n_per_condition_at_a_priori_de",))
    ratio = dig(C, ad + ("effective_n_over_uncorrected_requirement_at_a_priori_de",))
    old_ratio = dig(A, ad + ("effective_n_over_uncorrected_requirement_at_a_priori_de",))
    locked = dig(C, de + ("locked_sequences_per_condition",))
    return {
        "id": "power",
        "title": "1. Power analysis (analysis/power.py)",
        "triage": ["C243", "C274"],
        "dissertation_locations": [
            "Ch 3 Population and Sampling Procedures: 'For the continuous primary outcome...' "
            "and 'Because trials are clustered...'",
            "Ch 3 Testing and Evaluation Results > Pilot Testing: 'The pilot's principal design "
            "product...'",
            "Ch 3 Supplementary Findings: 'Fifth...'; Table 1; Appendix E",
        ],
        "what_changed": (
            "Two errors in the a-priori arithmetic are corrected. The two-proportion requirement "
            "used the one-sample formula ((z_a + z_b)/h)^2, half the per-group n of "
            "2((z_a + z_b)/h)^2, and the chi-square line used w = .30, which is not the effect "
            f"size of .50 vs .667 (w = .169): the binding requirement is {fnum(req)} trials per "
            "condition. The pilot rho was also miscomputed (section 2). With both corrected the "
            f"requirement is {fnum(seqs)} sequences per condition with the 15% allowance "
            + (f"({fnum(rng[0])} to {fnum(rng[1])} over the pilot rho's interval) " if rng else "")
            + f"against the {fnum(locked)} that ran; the formula correction alone gives "
            f"{fnum(seqs_fo)}. The {fnum(trials)} analyzed trials per condition give "
            f"{fnum(eff, 1)} effective trials at the corrected design effect, {fnum(ratio, 2)} of "
            f"the requirement (reported before as {fnum(old_ratio, 2)})."
        ),
        "rows": rows,
        "unchanged": unchanged,
    }


def _icc(A: dict, C: dict) -> dict:
    de, w, dea = _DE, ("power", "within_sequence_icc_by_outcome"), _DEA
    pilot, pilot_ci = dig(C, de + ("within_sequence_icc_rho",)), dig(C, de + ("rho_ci95",))
    rows = [
        row(
            "Pilot within-condition ICC(1), exploitation success",
            fnum(dig(A, de + ("within_sequence_icc_rho",)), 3)
            + " (k - 1 df; constant static condition counted)",
            fest(pilot, pilot_ci) + " (nested df k - c; conditions with variance only)",
            before_src=f"archived: {_path(de + ('within_sequence_icc_rho',))}",
            after_src=f"corrected: {_path(de + ('within_sequence_icc_rho',))}, rho_ci95",
            kind="changed",
            before_value=dig(A, de + ("within_sequence_icc_rho",)),
            after_value={"rho": pilot, "ci95": pilot_ci},
        ),
        row(
            "Pilot sequences used for rho / in all / varying within themselves",
            NOT_REPORTED,
            f"{dig(C, de + ('pilot_sequences_used_for_rho',))} / "
            f"{dig(C, de + ('pilot_sequences',))} / "
            f"{dig(C, de + ('pilot_sequences_with_outcome_variance',))}",
            before_src="--",
            after_src=f"corrected: {_path(de + ('pilot_sequences_used_for_rho',))}",
            kind="added",
        ),
    ]
    for outcome, label in (
        ("exploitation_success", "exploitation success"),
        ("time_to_compromise_s", "time to compromise"),
    ):
        det = dig(C, w + (outcome, "within_condition_detail")) or {}
        excl = det.get("conditions_excluded_constant_outcome") or []
        rows.append(
            row(
                f"Confirmatory within-condition ICC(1), {label}",
                fnum(dig(A, w + (outcome, "within_condition")), 3),
                fest(det.get("rho"), det.get("ci95"))
                + (
                    f" (untruncated {fnum(det.get('rho_untruncated'), 3)}; "
                    f"F({det.get('df_between')}, {det.get('df_within')}) = "
                    f"{fnum(det.get('F'), 2)}"
                    + (f"; constant conditions left out: {', '.join(excl)}" if excl else "")
                    + ")"
                    if det
                    else ""
                ),
                before_src=f"archived: {_path(w + (outcome, 'within_condition'))}",
                after_src=f"corrected: {_path(w + (outcome, 'within_condition_detail'))}",
                kind="interval added",
                after_value=det or None,
            )
        )
    for cond in (
        "combined_mtd_deception",
        "endpoint_mutation",
        "network_shuffling",
        "port_rotation",
    ):
        rb = dig(C, dea + ("rho_by_condition", cond)) or {}
        rows.append(
            row(
                f"Confirmatory ICC(1) within {LABEL[cond]}",
                fnum(dig(A, dea + ("rho_by_condition", cond, "rho")), 3),
                fest(rb.get("rho"), rb.get("rho_ci95")),
                before_src=f"archived: {_path(dea + ('rho_by_condition', cond, 'rho'))}",
                after_src=f"corrected: {_path(dea + ('rho_by_condition', cond))}",
                kind="interval added",
            )
        )
    conf = dig(C, w + ("exploitation_success", "within_condition_detail")) or {}
    return {
        "id": "icc",
        "title": "2. ICC estimates, intervals and degrees of freedom (power.icc_with_interval)",
        "triage": ["C274"],
        "dissertation_locations": [
            "Ch 3 Testing and Evaluation Results > Pilot Testing: 'The pilot's principal design "
            "product...'",
            "Ch 3 Population and Sampling Procedures: 'Because trials are clustered...'",
            "Ch 3 Supplementary Findings: 'Fifth...'",
        ],
        "what_changed": (
            "Within-condition ICC(1) now uses the nested-design df (k - c, not k - 1), leaves out "
            "conditions whose outcome is constant (they carry no information about rho but "
            "narrowed the interval), and carries an F-based 95% interval. The pilot value the "
            f"study was planned on, {fnum(dig(A, de + ('within_sequence_icc_rho',)), 3)}, becomes "
            f"{fest(pilot, pilot_ci)}: it rests on the "
            f"{dig(C, de + ('pilot_sequences_used_for_rho',))} sequences of the only pilot "
            "condition whose outcome varied, and its interval spans almost the whole range. The "
            f"confirmatory within-condition ICC stays {fnum(conf.get('rho'), 3)}, now with an "
            f"upper bound of {fnum((conf.get('ci95') or [None, None])[1], 3)}."
        ),
        "rows": rows,
        "unchanged": [
            u
            for u in [
                num_row(
                    "Pooled ICC(1), exploitation success (all conditions)",
                    A,
                    C,
                    w + ("exploitation_success", "pooled"),
                )
            ]
            if u["kind"] == "unchanged"
        ],
    }


def _separation(A: dict, C: dict) -> dict:
    r1 = ("rq1", "exploitation_success")
    sr1 = r1 + ("separation_robust",)
    rc = dig(C, sr1 + ("risk_contrast",)) or {}
    vb1 = dig(C, sr1 + ("vb_prior_sensitivity", "odds_ratio_by_prior_sd")) or {}
    f1 = dig(C, sr1 + ("firth", "terms")) or {}
    const = (rc.get("constant_arms") or {}).get("static") or {}

    def prior_sweep(vb: dict, term: str) -> str:
        if not vb:
            return NOT_REPORTED
        vals = [fsig(vb[k].get(term)) for k in sorted(vb, key=float)]
        sds = " / ".join(sorted(vb, key=float))
        return f"{' / '.join(vals)} at prior SD {sds}"

    rows = [
        row(
            "RQ1: MTD vs static, odds ratio (VB GLMM)",
            f_or(dig(A, r1 + ("mtd_vs_static",)), 4),
            "not data-identified: " + prior_sweep(vb1, "is_mtd"),
            before_src=f"archived: {_path(r1 + ('mtd_vs_static',))}",
            after_src=f"corrected: {_path(sr1 + ('vb_prior_sensitivity',))}",
            kind="withdrawn",
            note=(
                "The static arm is compromised in 264 of 264 trials, so the odds ratio has no "
                "maximum-likelihood value; the VB estimate is set by the prior."
            ),
        ),
        row(
            "RQ1: MTD - static compromise proportion (risk difference)",
            NOT_REPORTED,
            fest(
                rc.get("risk_difference_mtd_minus_static"),
                rc.get("risk_difference_ci95"),
                rc.get("risk_difference_p"),
            ),
            before_src="--",
            after_src=f"corrected: {_path(sr1 + ('risk_contrast',))}",
            kind="added (headline)",
        ),
        row(
            "RQ1: MTD / static (risk ratio)",
            NOT_REPORTED,
            fest(rc.get("risk_ratio_mtd_over_static"), rc.get("risk_ratio_ci95")),
            before_src="--",
            after_src=f"corrected: {_path(sr1 + ('risk_contrast',))}",
            kind="added",
        ),
        row(
            "RQ1: compromise proportion, static baseline",
            NOT_REPORTED,
            fnum(rc.get("proportion_static"), 3)
            + (
                f" (exact 95% CI {fci(const.get('exact_ci95_trial_level'))} over trials, "
                f"{fci(const.get('exact_ci95_sequence_level'))} over sequences)"
                if const
                else ""
            ),
            before_src="--",
            after_src=f"corrected: {_path(sr1 + ('risk_contrast', 'constant_arms'))}",
            kind="added",
        ),
        row(
            "RQ1: compromise proportion, pooled MTD",
            NOT_REPORTED,
            fest(rc.get("proportion_mtd"), rc.get("proportion_mtd_ci95")),
            before_src="--",
            after_src=f"corrected: {_path(sr1 + ('risk_contrast',))}",
            kind="added",
        ),
        row(
            "RQ1: Firth-penalized odds ratio, MTD vs static",
            NOT_REPORTED,
            fnum((f1.get("is_mtd") or {}).get("odds_ratio"), 4)
            + " (penalty-driven; not a data-identified estimate)",
            before_src="--",
            after_src=f"corrected: {_path(sr1 + ('firth', 'terms', 'is_mtd'))}",
            kind="added (caveated)",
        ),
        row(
            "RQ1: attempt-index odds ratio in the RQ1 model",
            f_or(dig(A, r1 + ("terms", "attempt_index"))),
            f_or(dig(C, r1 + ("terms", "attempt_index")))
            + f" (VB; {prior_sweep(vb1, 'attempt_index')}); Firth "
            + f_or(f1.get("attempt_index")),
            before_src=f"archived: {_path(r1 + ('terms', 'attempt_index'))}",
            after_src=(
                f"corrected: {_path(r1 + ('terms', 'attempt_index'))} (VB); "
                f"{_path(sr1 + ('firth', 'terms', 'attempt_index'))} (Firth)"
            ),
            kind="unchanged (annotated)",
            note=(
                "Data-identified and prior-stable. The Firth model has no sequence intercept, so "
                "the always-compromised deceptive arm (pooled into MTD) flattens its slope; for "
                "the attempt effect itself cite RQ3's average attempt effect (section 4)."
            ),
        ),
    ]

    r2 = ("rq2", "exploitation_success")
    sr2 = r2 + ("separation_robust",)
    vb2 = dig(C, sr2 + ("vb_prior_sensitivity", "odds_ratio_by_prior_sd")) or {}
    f2 = dig(C, sr2 + ("firth", "terms")) or {}
    rates = dig(C, sr2 + ("rates",)) or {}
    floor2 = 2.0 / (float(rates.get("n_boot") or 2000) + 1.0)
    diffs = rates.get("differences_vs_reference") or {}
    fdr = rates.get("fdr_differences") or {}
    ref = rates.get("reference_level", COMBINED)
    for tech in (
        "deceptive_presentation",
        "port_rotation",
        "network_shuffling",
        "endpoint_mutation",
    ):
        term = _TECH_TERM.format(tech)
        sep = bool((f2.get(term) or {}).get("involves_separated_level"))
        before = f_or(dig(A, r2 + ("terms", term)), 2)
        if sep:
            after = "not data-identified: " + prior_sweep(vb2, term)
            kind = "withdrawn"
        else:
            vals = [vb2[k].get(term) for k in vb2] if vb2 else []
            after = f"Firth {f_or(f2.get(term), 2)}" + (
                f" (VB across prior SD 1-10: {fnum(min(vals), 2)}-{fnum(max(vals), 2)})"
                if vals
                else ""
            )
            kind = "changed"
        rows.append(
            row(
                f"RQ2: {LABEL[tech]} vs {LABEL[ref]}, odds ratio",
                before,
                after,
                before_src=f"archived: {_path(r2 + ('terms', term))}",
                after_src=f"corrected: {_path(sr2 + ('firth', 'terms', term))}",
                kind=kind,
            )
        )
        key = f"{tech} - {ref}"
        dd = diffs.get(key) or {}
        rows.append(
            row(
                f"RQ2: {LABEL[tech]} - {LABEL[ref]}, compromise proportion",
                NOT_REPORTED,
                fest(dd.get("difference"), dd.get("ci95"))
                + f", {fp(dd.get('p'), floor=floor2)}"
                + (
                    (
                        " (BH p_adj "
                        + (
                            f"<= {fnum((fdr.get(key) or {}).get('p_adj'), 4)}"
                            if dd.get("p") is not None and dd["p"] <= floor2 * (1 + 1e-9)
                            else fp((fdr.get(key) or {}).get("p_adj"))[2:]
                        )
                        + ")"
                    )
                    if key in fdr
                    else ""
                ),
                before_src="--",
                after_src=f"corrected: {_path(sr2 + ('rates', 'differences_vs_reference'))}",
                kind="added (headline)" if sep else "added",
            )
        )
    if sep_firth := (f2.get(_TECH_TERM.format("deceptive_presentation")) or {}):
        rows.append(
            row(
                "RQ2: Firth-penalized odds ratio, deceptive presentation vs combined",
                NOT_REPORTED,
                fnum(sep_firth.get("odds_ratio"), 0)
                + " (penalty-driven; not a data-identified estimate)",
                before_src="--",
                after_src=f"corrected: {_path(sr2 + ('firth', 'terms', _TECH_TERM.format('deceptive_presentation')))}",
                kind="added (caveated)",
            )
        )
    rows.append(
        row(
            "RQ2: attempt-index odds ratio in the RQ2 model",
            f_or(dig(A, r2 + ("terms", "attempt_index"))),
            f"VB {f_or(dig(C, r2 + ('terms', 'attempt_index')))}; Firth {f_or(f2.get('attempt_index'))}",
            before_src=f"archived: {_path(r2 + ('terms', 'attempt_index'))}",
            after_src=f"corrected: {_path(sr2 + ('firth', 'terms', 'attempt_index'))}",
            kind="unchanged (annotated)",
        )
    )
    fold_rq1 = dig(C, sr1 + ("vb_prior_sensitivity", "max_over_min_fold_change", "is_mtd"))
    return {
        "id": "separation",
        "title": "3. Separation-robust RQ1/RQ2 estimates (analysis/separation.py)",
        "triage": ["Part C-1", "C291"],
        "dissertation_locations": [
            "Ch 3 RQ1: 'For the second co-primary outcome, exploitation success...'",
            "Ch 3 RQ2: 'For exploitation success, the logistic generalized linear mixed "
            "model returned...'",
            "Ch 3 Summary: 'The confirmatory evaluation executed the locked design...'",
        ],
        "what_changed": (
            "The static baseline and deceptive presentation are compromised in every trial, so "
            "any odds ratio against them has no maximum-likelihood value; the variational-Bayes "
            "GLMM returns whatever its prior implies (the RQ1 OR moves by a factor of "
            f"{fsig(fold_rq1)} across prior SDs of 1 to 10). Those odds ratios are withdrawn. "
            "The headline "
            "becomes the compromise proportions and their differences, with sequence-cluster "
            "bootstrap intervals; the data-identified odds ratios are re-estimated by Firth-"
            "penalized logistic regression with the same bootstrap. The OR-type estimates for "
            "the non-separated techniques barely move."
        ),
        "rows": rows,
        "unchanged": [],
    }


def _rq3(A: dict, C: dict) -> dict:
    r3 = ("rq3", "exploitation_success")
    ae = dig(C, r3 + ("attempt_effects",)) or {}
    rows = [
        row(
            "Attempt-index OR reported as the 'main effect' (interaction model)",
            f_or(dig(A, r3 + ("attempt_effect",))),
            f_or(dig(C, r3 + ("attempt_effect",)))
            + f" = the attempt slope in the reference condition ({LABEL.get(str(dig(C, r3 + ('attempt_effect', 'reference_condition'))), '?')}) only",
            before_src=f"archived: {_path(r3 + ('attempt_effect',))}",
            after_src=f"corrected: {_path(r3 + ('attempt_effect', 'interpretation'))}",
            kind="relabeled",
        ),
        row(
            "Average attempt effect (conditions with outcome variance; common slope)",
            NOT_REPORTED,
            f_or(ae.get("average_attempt_effect"))
            + f", n = {fnum((ae.get('average_attempt_effect') or {}).get('n'))}",
            before_src="--",
            after_src=f"corrected: {_path(r3 + ('attempt_effects', 'average_attempt_effect'))}",
            kind="added (headline)",
        ),
    ]
    for cond, blk in (ae.get("per_condition") or {}).items():
        rows.append(
            row(
                f"Attempt slope within {LABEL.get(cond, cond)} (all 8 attempts)",
                NOT_REPORTED,
                f_or(blk),
                before_src="--",
                after_src=f"corrected: {_path(r3 + ('attempt_effects', 'per_condition', cond))}",
                kind="added",
            )
        )
    rec = ("rq3", "recon_accuracy_f1")
    b_note, a_note = dig(A, rec + ("note",)), dig(C, rec + ("note",))
    if b_note is not None or a_note is not None:
        rows.append(
            row(
                "Reconnaissance-accuracy LMM: convergence statement",
                (b_note or NOT_REPORTED)[:110] + ("..." if b_note and len(b_note) > 110 else ""),
                (a_note or NOT_REPORTED)[:110] + ("..." if a_note and len(a_note) > 110 else ""),
                before_src=f"archived: {_path(rec + ('note',))} (converged = {dig(A, rec + ('converged',))})",
                after_src=f"corrected: {_path(rec + ('note',))} (converged = {dig(C, rec + ('converged',))})",
                kind="unchanged" if b_note == a_note else "changed",
            )
        )
    ref3 = dig(C, r3 + ("attempt_effect", "reference_condition"))
    away = [
        c
        for c, blk in (ae.get("per_condition") or {}).items()
        if isinstance(blk, dict)
        and blk.get("ci95")
        and not (blk["ci95"][0] <= 1.0 <= blk["ci95"][1])
    ]
    return {
        "id": "rq3_attempt",
        "title": "4. RQ3 average attempt effect (rq3.attempt_effects)",
        "triage": ["C295", "C30"],
        "dissertation_locations": [
            "Ch 3 RQ3: 'In the model including the interaction, attempt index had a significant "
            "main effect...'",
            "Ch 3 RQ3 decision paragraph; Supplementary Findings; Ch 3 Summary",
        ],
        "what_changed": (
            "The interaction model uses treatment coding, so its attempt_index term (OR = "
            f"{fnum(dig(C, r3 + ('attempt_effect', 'odds_ratio')))}) is the slope in the "
            f"reference condition, {LABEL.get(str(ref3), ref3)}, not a main effect. The report "
            "now carries the average attempt effect over the conditions whose outcome varies and "
            "each condition's own slope (logit GLMs, robust SEs clustered on sequence). "
            + (
                "Conditions whose own slope excludes 1: "
                + ", ".join(LABEL.get(c, c) for c in away)
                + "."
                if away
                else "No condition's own slope excludes 1."
            )
        ),
        "rows": rows,
        "unchanged": [],
    }


def _rq5(A: dict, C: dict, T: dict | None) -> dict:
    # The draft's figures are over the 1,800-s trial window. Since 2026-10-28 the H05 decision
    # block (rq5.benchmark_profile) uses a 300-s benefit horizon, so this table reads the
    # trial-window block; a corrected report made before then has only benchmark_profile.
    bp = ("rq5", "benchmark_profile_trial_window_sensitivity")
    if dig(C, bp) is None:
        bp = ("rq5", "benchmark_profile")
    horizon_note = (
        " Since 2026-10-28 the H05 decision uses a 300-s benefit horizon "
        "(rq5.benchmark_profile); the draft's figures are over the trial window, so the rows "
        "here are read from rq5.benchmark_profile_trial_window_sensitivity."
        if bp[-1] == "benchmark_profile_trial_window_sensitivity"
        else ""
    )
    prof = dig(C, bp) or {}
    t5 = (T or {}).get("rq5") or {}
    src_t = f"text: {t5.get('where', TEXT_SOURCE)}" if t5 else "--"
    by = prof.get("by_technique") or {}
    rows: list[dict] = []
    if not by:
        rows.append(
            row(
                "H05 benefit/cost profile test",
                "no test",
                prof.get("note", "not run"),
                before_src="--",
                after_src=f"corrected: {_path(bp)}",
                kind="not run",
            )
        )
        return {
            "id": "rq5_profile",
            "title": "5. RQ5 benefit/cost profile bootstrap (analysis/rq5_profile.py)",
            "triage": ["C310", "C36"],
            "dissertation_locations": [],
            "what_changed": "The benchmark was not supplied, so the profile test did not run.",
            "rows": rows,
            "unchanged": [],
        }
    for tech in (
        "endpoint_mutation",
        "network_shuffling",
        COMBINED,
        "port_rotation",
        "deceptive_presentation",
    ):
        v = by.get(tech) or {}
        tv = (t5.get("profile") or {}).get(tech)
        note = None
        if (
            tech == "endpoint_mutation"
            and tv is not None
            and not _same(round(v.get("profile", 0), 1), tv)
        ):
            note = (
                f"The draft divided by the rounded median cost "
                f"{fnum((t5.get('median_cost_s') or {}).get(tech), 3)} s; the exact median is "
                f"{fnum(v.get('median_cost_s'), 4)} s."
            )
        rows.append(
            row(
                f"Profile, {LABEL[tech]} (s of RMST gained per s of reconfiguration)",
                fnum(tv, 1) if tv is not None else NOT_REPORTED,
                fest(v.get("profile"), v.get("profile_ci95"), nd=1),
                before_src=src_t if tv is not None else "--",
                after_src=f"corrected: {_path(bp + ('by_technique', tech))}",
                kind=_kind_1dp(tv, v.get("profile")),
                note=note,
            )
        )
    pc = prof.get("h05_primary_contrast") or {}
    tm = t5.get("single_technique_mean")
    rows += [
        row(
            "Mean of the four single-technique profiles",
            fnum(tm, 1) if tm is not None else NOT_REPORTED,
            fest(pc.get("comparator_value"), pc.get("comparator_ci95"), nd=1),
            before_src=src_t if tm is not None else "--",
            after_src=f"corrected: {_path(bp + ('h05_primary_contrast',))}",
            kind=_kind_1dp(tm, pc.get("comparator_value")),
            note=(
                "Includes endpoint mutation's exact profile (see that row)."
                if tm is not None
                and pc.get("comparator_value") is not None
                and not _same(round(pc["comparator_value"], 1), tm)
                else None
            ),
        ),
        row(
            "H05 planned contrast: combined - single-technique mean",
            "no test",
            fest(pc.get("difference_combined_minus_other"), pc.get("difference_ci95"), nd=1)
            + f", {fp(pc.get('p'), floor=2 / (prof.get('n_boot', 10000) + 1))}; "
            + str(pc.get("direction", "")),
            before_src="--",
            after_src=f"corrected: {_path(bp + ('h05_primary_contrast',))}",
            kind="added (headline)",
        ),
        row(
            "Combined / single-technique mean (ratio of profiles)",
            t5.get("ratio_text", NOT_REPORTED) if t5 else NOT_REPORTED,
            fest(pc.get("ratio_combined_over_other"), pc.get("ratio_ci95")),
            before_src=src_t,
            after_src=f"corrected: {_path(bp + ('h05_primary_contrast',))}",
            kind="interval added",
        ),
    ]
    floor = 2 / (prof.get("n_boot", 10000) + 1)
    for tech, v in (prof.get("per_technique_contrasts") or {}).items():
        rows.append(
            row(
                f"Combined - {LABEL[tech]} (secondary, Holm)",
                "no test",
                fest(v.get("difference_combined_minus_other"), v.get("difference_ci95"), nd=1)
                + f", Holm {fp(v.get('p_holm'), floor=4 * floor)}; {v.get('direction')}",
                before_src="--",
                after_src=f"corrected: {_path(bp + ('per_technique_contrasts', tech))}",
                kind="added",
            )
        )
    for tech in (
        COMBINED,
        "endpoint_mutation",
        "network_shuffling",
        "port_rotation",
        "deceptive_presentation",
    ):
        v = by.get(tech) or {}
        tb = (t5.get("benefit_s") or {}).get(tech)
        tc = (t5.get("median_cost_s") or {}).get(tech)
        rows.append(
            row(
                f"Security benefit (s), {LABEL[tech]}",
                fnum(tb, 2) if tb is not None else NOT_REPORTED,
                fest(v.get("security_benefit_s"), v.get("security_benefit_ci95"), nd=2),
                before_src=src_t if tb is not None else "--",
                after_src=f"corrected: {_path(bp + ('by_technique', tech, 'security_benefit_s'))}",
                kind="interval added",
            )
        )
        rows.append(
            row(
                f"Median reconfiguration cost (s), {LABEL[tech]}",
                fnum(tc, 3) if tc is not None else NOT_REPORTED,
                fest(v.get("median_cost_s"), v.get("median_cost_ci95"), nd=4),
                before_src=src_t if tc is not None else "--",
                after_src=f"corrected: {_path(bp + ('by_technique', tech, 'median_cost_s'))}",
                kind="interval added",
            )
        )
    ct = prof.get("cost_tests") or {}
    kw4, kw5 = (
        ct.get("kruskal_wallis_five_mtd_techniques") or {},
        ct.get("kruskal_wallis_with_static") or {},
    )
    tk4, tk5 = t5.get("kw_mtd") or {}, t5.get("kw_all") or {}
    rows += [
        row(
            "Kruskal-Wallis on cost, five MTD techniques",
            f"H({tk4.get('df')}) = {fnum(tk4.get('H'), 2)}" if tk4 else NOT_REPORTED,
            f"H({kw4.get('df')}) = {fnum(kw4.get('H'), 2)}, {fp(kw4.get('p'))}, eta^2 = {fnum(kw4.get('eta_squared'), 3)}",
            before_src=src_t if tk4 else "--",
            after_src=f"corrected: {_path(bp + ('cost_tests', 'kruskal_wallis_five_mtd_techniques'))}",
            kind=(
                (
                    "unchanged"
                    if tk4 and _same(round(kw4.get("H", 0), 2), tk4.get("H"))
                    else "changed"
                )
                + " (eta^2 added)"
                if tk4
                else "added"
            ),
        ),
        row(
            "Kruskal-Wallis on cost, with the static baseline",
            (
                f"H({tk5.get('df')}) = {fnum(tk5.get('H'), 2)}, eta^2 = {fnum(tk5.get('eta_squared'), 3)}"
                if tk5
                else NOT_REPORTED
            ),
            f"H({kw5.get('df')}) = {fnum(kw5.get('H'), 2)}, {fp(kw5.get('p'))}, eta^2 = {fnum(kw5.get('eta_squared'), 3)}",
            before_src=src_t if tk5 else "--",
            after_src=f"corrected: {_path(bp + ('cost_tests', 'kruskal_wallis_with_static'))}",
            kind=(
                ("unchanged" if _same(round(kw5.get("H", 0), 2), tk5.get("H")) else "changed")
                if tk5
                else "added"
            ),
        ),
    ]
    ccp = dig(C, ("rq5", "cost_column_provenance")) or {}
    rows.append(
        row(
            "Per-trial cost columns in the trial dataset",
            "used by the archived rq5 cost blocks",
            (
                "synthetic (offline telemetry stand-in), not measurements"
                if ccp.get("recorded") is False or ccp.get("all_measured") is False
                else "measured"
            ),
            before_src="archived: rq5.primary_contrasts_cost_per_sequence (and the other cost blocks)",
            after_src="corrected: rq5.cost_column_provenance",
            kind="flagged",
        )
    )
    changed_rows = [r["quantity"] for r in rows if r["kind"] == "changed"]
    return {
        "id": "rq5_profile",
        "title": "5. RQ5 benefit/cost profile bootstrap (analysis/rq5_profile.py)",
        "triage": ["C310", "C36"],
        "dissertation_locations": [
            "Ch 1 H05/Ha5",
            "Ch 3 RQ5: 'Operational cost differed...', 'The security-improvement-to-"
            "operational-cost profile...', Table 6, and the RQ5 decision paragraph",
            "Ch 3 Summary",
        ],
        "what_changed": (
            "Nothing archived tested the benefit/cost profile, so the H05 decision had no test "
            "behind it. Each profile now has a bootstrap interval (sequences resampled within "
            "condition for the RMST benefit, benchmark replicates within technique for the "
            "median cost), and the planned contrasts are tested: combined vs the "
            "single-technique mean (primary) and vs each technique (secondary, Holm). "
            + (
                "Point values that differ from the draft at its printed precision: "
                + ", ".join(changed_rows)
                + "."
                if changed_rows
                else "The point values match the draft at its printed precision."
            )
            + horizon_note
        ),
        "benchmark": prof.get("benchmark"),
        "rows": rows,
        "unchanged": [],
    }


def _drift(A: dict, C: dict) -> dict:
    dp = ("design_checks", "within_block_drift")
    dr = dig(C, dp) or {}
    outcome = dr.get("outcome", "exploitation_success")
    pooled = dr.get("pooled_common_slope") or {}
    rows = [
        row(
            "Test of order effects between conditions",
            "stated in the text; no test archived",
            "not possible: one contiguous block per condition aliases condition with execution "
            "period; within-block drift check added",
            before_src="text: Ch 3 Comparative Structure ('...tested for order effects...')",
            after_src=f"corrected: {_path(dp + ('note',))}",
            kind="changed",
        ),
        row(
            "Within-block drift, pooled common slope (OR per 10 sequences of run order)",
            NOT_REPORTED,
            fest(pooled.get("odds_ratio_per_10_sequences"), pooled.get("ci95"), pooled.get("p")),
            before_src="--",
            after_src=f"corrected: {_path(dp + ('pooled_common_slope',))}",
            kind="added",
        ),
    ]
    for cond, v in (dr.get("by_condition") or {}).items():
        e = v.get(outcome) or {}
        thirds = v.get(f"{outcome}_rate_by_run_third") or []
        if "odds_ratio_per_10_sequences" in e:
            after = (
                fest(e.get("odds_ratio_per_10_sequences"), e.get("ci95"), e.get("p"))
                + f", Holm {fp(e.get('p_holm'))}"
            )
        else:
            after = "outcome constant; nothing to drift"
        if thirds:
            after += "; rate by third of the block " + " / ".join(fnum(x, 3) for x in thirds)
        rows.append(
            row(
                f"Within-block drift, {LABEL.get(cond, cond)}",
                NOT_REPORTED,
                after,
                before_src="--",
                after_src=f"corrected: {_path(dp + ('by_condition', cond))}",
                kind="added",
            )
        )
        sp = v.get("restricted_ttc_spearman") or {}
        if "rho" in sp and sp.get("p") is not None and sp["p"] < 0.05:
            rows.append(
                row(
                    f"Restricted TTC vs run order, {LABEL.get(cond, cond)} (Spearman)",
                    NOT_REPORTED,
                    f"rho = {fnum(sp.get('rho'), 2)}, {fp(sp.get('p'))}, Holm {fp(sp.get('p_holm'))}",
                    before_src="--",
                    after_src=f"corrected: {_path(dp + ('by_condition', cond, 'restricted_ttc_spearman'))}",
                    kind="added",
                )
            )
    return {
        "id": "drift",
        "title": "6. Within-block drift check (analysis/drift.py)",
        "triage": ["C209"],
        "dissertation_locations": [
            "Ch 3 Research Project Design > Comparative Structure ('...tested for order "
            "effects in the analysis')",
            "Ch 3 limitations (aliasing of condition and execution period)",
        ],
        "what_changed": (
            "Each condition ran as one contiguous block, so a between-condition order test is "
            "not possible. The new check asks whether the outcome trended across a block's "
            "sequences in the order they ran (logit with robust SEs clustered on sequence; "
            "Holm across conditions)."
        ),
        "rows": rows,
        "unchanged": [],
    }


def _rq4_housekeeping(A: dict, C: dict, T: dict | None) -> dict:
    sp = ("rq4", "security_measures_vif_pca")
    s = dig(C, sp) or {}
    t4 = (T or {}).get("rq4_vif_pca") or {}
    src_t = f"text: {t4.get('where')}" if t4 else "--"
    vif = dig(s, ("vif", "vif")) or {}
    evr = dig(s, ("pca", "explained_variance_ratio")) or []
    rows = [
        row(
            "Complete cases (attack-path churn is undefined at attempt index 0)",
            fnum(t4.get("n")) if t4 else NOT_REPORTED,
            f"{fnum(s.get('n_complete_cases'))} of {fnum(s.get('n_trials'))} "
            + (
                f"({fnum(s.get('missing_at_attempt_index_0'))} missing, all at attempt index 0)"
                if s.get("missing_elsewhere") == 0
                else f"({fnum(s.get('missing_at_attempt_index_0'))} missing at attempt index 0, "
                f"{fnum(s.get('missing_elsewhere'))} elsewhere)"
            ),
            before_src=src_t,
            after_src=f"corrected: {_path(sp + ('n_complete_cases',))}",
            kind=_kind_archived(t4.get("n"), s.get("n_complete_cases"), 0),
        )
    ]
    for m in (
        "exploitation_success",
        "time_to_compromise_s",
        "time_to_first_valid_exploit_s",
        "recon_accuracy_f1",
        "attack_path_churn",
    ):
        tv = (t4.get("vif") or {}).get(m)
        rows.append(
            row(
                f"VIF, {m}",
                fnum(tv, 2) if tv is not None else NOT_REPORTED,
                fnum(vif.get(m), 2),
                before_src=src_t if tv is not None else "--",
                after_src=f"corrected: {_path(sp + ('vif', 'vif', m))}",
                kind=_kind_archived(tv, vif.get(m), 2),
            )
        )
    tp = t4.get("pca_pct") or []
    for i in range(min(3, len(evr))):
        rows.append(
            row(
                f"PCA component {i + 1}, % of variance",
                fnum(tp[i], 1) if i < len(tp) else NOT_REPORTED,
                fnum(100 * evr[i], 1),
                before_src=src_t if i < len(tp) else "--",
                after_src=f"corrected: {_path(sp + ('pca', 'explained_variance_ratio'))}",
                kind=_kind_archived(tp[i] if i < len(tp) else None, 100 * evr[i], 1),
            )
        )
    if len(evr) >= 3:
        rows.append(
            row(
                "PCA, cumulative % for three components",
                fnum(t4.get("pca_cumulative_pct"), 1) if t4 else NOT_REPORTED,
                fnum(100 * sum(evr[:3]), 1),
                before_src=src_t,
                after_src=f"corrected: {_path(sp + ('pca',))}",
                kind=_kind_archived(t4.get("pca_cumulative_pct"), 100 * sum(evr[:3]), 1),
            )
        )
    n_old = dig(A, ("rq4", "metric_set_missingness", "n_complete_case_used_by_vif_pca"))
    return {
        "id": "rq4_vif_pca",
        "title": "7. Housekeeping: RQ4 five-measure VIF/PCA (n = 1,386) archived",
        "triage": ["C31", "C302"],
        "dissertation_locations": ["Ch 3 RQ4: 'Among the trial-level measures...'"],
        "what_changed": (
            "The draft's trial-level VIF/PCA was computed outside the pipeline on the five "
            "security measures without the overhead columns; the archived rq4 vif/pca blocks "
            "used ten indicators including the synthetic overhead columns"
            + (f" ({fnum(n_old)} complete cases)" if n_old is not None else "")
            + ". The five-measure computation is now part of every report "
            "(rq4.security_measures_vif_pca)"
            + (
                " and reproduces the draft's figures."
                if all(r["kind"] == "archived" for r in rows)
                else "; rows marked 'archived (differs)' do not match the draft."
            )
        ),
        "rows": rows,
        "unchanged": [],
    }


def _rq3_post_hoc(A: dict, C: dict, T: dict | None) -> dict:
    ap = ("rq3", "exploitation_success", "attempt_effects")
    ae = dig(C, ap) or {}
    t3 = (T or {}).get("rq3_post_hoc") or {}
    src_t = f"text: {t3.get('where')}" if t3 else "--"

    def tfmt(t: dict | None) -> str:
        if not t:
            return NOT_REPORTED
        s = fest(t.get("odds_ratio"), t.get("ci95"), t.get("p"))
        return s + (f", n = {fnum(t['n'])}" if "n" in t else "")

    def cfmt(c: dict | None) -> str:
        if not c or "odds_ratio" not in c:
            return NOT_REPORTED
        return f_or(c) + f", n = {fnum(c.get('n'))}"

    rows = [
        row(
            "All attempts, conditions with variance (common slope)",
            tfmt(t3.get("all_attempts")),
            cfmt(ae.get("average_attempt_effect")),
            before_src=src_t,
            after_src=f"corrected: {_path(ap + ('average_attempt_effect',))}",
            kind=_kind_archived(
                (t3.get("all_attempts") or {}).get("odds_ratio"),
                (ae.get("average_attempt_effect") or {}).get("odds_ratio"),
                3,
            ),
        ),
        row(
            "Excluding attempt index 0 (post hoc)",
            tfmt(t3.get("excluding_first")),
            cfmt(ae.get("average_excluding_first_attempt")),
            before_src=src_t,
            after_src=f"corrected: {_path(ap + ('average_excluding_first_attempt',))}",
            kind=_kind_archived(
                (t3.get("excluding_first") or {}).get("odds_ratio"),
                (ae.get("average_excluding_first_attempt") or {}).get("odds_ratio"),
                3,
            ),
        ),
    ]
    tpc = t3.get("per_condition_excluding_first") or {}
    for cond, blk in (ae.get("per_condition_excluding_first_attempt") or {}).items():
        tv = tpc.get(cond)
        note = None
        if tv and "odds_ratio" in blk and fnum(blk["odds_ratio"]) != fnum(tv["odds_ratio"]):
            note = (
                f"To three decimals the odds ratio is {fnum(blk['odds_ratio'])} "
                f"({blk['odds_ratio']:.4f}); the draft prints {fnum(tv['odds_ratio'])}."
            )
        rows.append(
            row(
                f"Attempts 1-7 within {LABEL.get(cond, cond)} (post hoc)",
                f"{fnum(tv['odds_ratio'])}, {fp(tv['p'])}" if tv else NOT_REPORTED,
                f_or(blk),
                before_src=src_t if tv else "--",
                after_src=f"corrected: {_path(ap + ('per_condition_excluding_first_attempt', cond))}",
                kind=_kind_archived((tv or {}).get("odds_ratio"), blk.get("odds_ratio"), 3),
                note=note,
            )
        )
    return {
        "id": "rq3_post_hoc",
        "title": "8. Housekeeping: RQ3 post hoc first-attempt sensitivity archived",
        "triage": ["C296"],
        "dissertation_locations": ["Ch 3 RQ3: 'Because attempt index 0 is the only attempt...'"],
        "what_changed": (
            "The draft's post hoc sensitivity (logit with condition fixed effects and robust "
            "SEs clustered on sequence) was computed outside the pipeline. The same model now "
            "runs in every report (rq3.<outcome>.attempt_effects)"
            + (
                " and reproduces the draft."
                if all(r["kind"] == "archived" for r in rows)
                else "; rows marked 'archived (differs)' do not match the draft at its printed "
                "precision (see the notes)."
            )
        ),
        "rows": rows,
        "unchanged": [],
    }


def _welch_housekeeping(A: dict, C: dict, T: dict | None) -> dict:
    wp = ("rq1", "exploitation_success", "sensitivity", "welch_t", "df")
    tw = (T or {}).get("rq1_welch_df")
    return {
        "id": "welch_df",
        "title": "9. Housekeeping: Welch degrees of freedom archived",
        "triage": [],
        "dissertation_locations": ["Ch 3 RQ1: 'For the second co-primary outcome...' (t(164))"],
        "what_changed": (
            "The per-sequence Welch t-tests now record their Welch-Satterthwaite df, which the "
            "draft prints but no report held."
        ),
        "rows": [
            row(
                "RQ1 per-sequence Welch t, exploitation success: df",
                fnum(tw[0]) if tw else NOT_REPORTED,
                fnum(dig(C, wp), 2),
                before_src=f"text: {tw[1]}" if tw else "--",
                after_src=f"corrected: {_path(wp)}",
                kind=_kind_archived(tw[0] if tw else None, dig(C, wp), 0),
            )
        ],
        "unchanged": [],
    }


# --------------------------------------------------------------------------- #
# Consistency check: every other number that differs between the two reports
# --------------------------------------------------------------------------- #


def _leaves(o: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _leaves(v, path + (str(k),))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _leaves(v, path + (f"[{i}]",))
    else:
        yield path, o


def consistency_check(archived: dict, corrected: dict, sig: int = 6) -> dict:
    """Leaves present in both reports that differ (to ``sig`` significant figures)."""

    a, c = dict(_leaves(archived)), dict(_leaves(corrected))
    archived_nodes = {p[:i] for p in a for i in range(1, len(p) + 1)}
    changed, removed = [], []
    for p, va in a.items():
        if p not in c:
            removed.append(p)
            continue
        if not _same(va, c[p], sig):
            changed.append(p)
    added = [p for p in c if p not in a]

    def intended(p: tuple[str, ...]) -> bool:
        return any(p[: len(q)] == q for q in INTENDED_CHANGE_PREFIXES)

    def later(p: tuple[str, ...]) -> str | None:
        return next((why for q, why in LATER_UPDATE_PREFIXES.items() if p[: len(q)] == q), None)

    unexpected = [p for p in changed if not intended(p) and later(p) is None]
    expected = [p for p in changed if intended(p)]
    by_later = [p for p in changed if not intended(p) and later(p) is not None]
    removed_unexpected = [p for p in removed if not intended(p) and later(p) is None]

    def new_subtrees(paths: list[tuple[str, ...]]) -> list[str]:
        """The shortest prefix of each added leaf that the archived report does not have."""

        roots = set()
        for p in paths:
            i = next((i for i in range(1, len(p) + 1) if p[:i] not in archived_nodes), len(p))
            roots.add(_path(p[:i]))
        return sorted(roots)

    return {
        "n_numbers_compared": len([p for p in a if p in c]),
        "n_changed_intended": len(expected),
        "intended_changes": [
            {"path": _path(p), "archived": a[p], "corrected": c[p]} for p in expected
        ],
        "n_changed_unexpected": len(unexpected),
        "unexpected_changes": [
            {"path": _path(p), "archived": a[p], "corrected": c[p]} for p in unexpected[:200]
        ],
        "n_changed_by_later_updates": len(by_later),
        "later_update_changes": [
            {"path": _path(p), "archived": a[p], "corrected": c[p], "reason": later(p)}
            for p in by_later
        ],
        "n_removed": len(removed),
        "removed_unexpected": [_path(p) for p in removed_unexpected[:200]],
        "added_blocks": new_subtrees(added),
        "intended_change_prefixes": [_path(q) for q in INTENDED_CHANGE_PREFIXES],
        "later_update_prefixes": {_path(q): why for q, why in LATER_UPDATE_PREFIXES.items()},
        "significant_figures": sig,
        "note": (
            "Compares every number (and string) present in BOTH reports. Additions are "
            "expected (new blocks) and are listed by the first path level the archived report "
            "lacks. A change "
            "outside intended_change_prefixes means the corrected report differs from the "
            "archived one somewhere the corrections were not meant to reach; on a different "
            "platform than the archived report's, the mixed-model blocks are the usual cause "
            "(see provenance.platform and library_versions). Changes under "
            "later_update_prefixes come from an analysis update made after the 2026-10-27 "
            "corrections and are listed with its reason in later_update_changes."
        ),
    }


# --------------------------------------------------------------------------- #
# Build + render
# --------------------------------------------------------------------------- #


def build(
    archived: dict,
    corrected: dict,
    *,
    dataset_sha256: str | None = None,
    inputs: dict | None = None,
    provenance: dict | None = None,
) -> dict:
    """The corrections record: sections of before -> after rows plus the consistency check."""

    T = TEXT_FIGURES.get(dataset_sha256 or "")
    sections = [
        _power(archived, corrected),
        _icc(archived, corrected),
        _separation(archived, corrected),
        _rq3(archived, corrected),
        _rq5(archived, corrected, T),
        _drift(archived, corrected),
        _rq4_housekeeping(archived, corrected, T),
        _rq3_post_hoc(archived, corrected, T),
        _welch_housekeeping(archived, corrected, T),
    ]
    return {
        "title": "Analysis corrections (2026-10-27): before -> after",
        "status": (
            "SUPPLEMENT to the archived report of record, which is unchanged. 'Before' is the "
            "archived report (or, where no report held the figure, the dissertation draft); "
            "'after' is the corrected report generated alongside this file by the current "
            "pipeline from the same trial dataset."
        ),
        "inputs": inputs or {},
        "provenance": provenance or {},
        "text_figures_source": TEXT_SOURCE if T else None,
        "sections": sections,
        "consistency_check": consistency_check(archived, corrected),
    }


def _md_cell(s: Any) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def render_markdown(rec: dict) -> str:
    out: list[str] = [f"# {rec['title']}", "", rec["status"], ""]
    inputs = rec.get("inputs") or {}
    if inputs:
        out += ["## Inputs", "", "| Input | Path | SHA-256 |", "|---|---|---|"]
        for name, v in inputs.items():
            if isinstance(v, dict):
                out.append(f"| {name} | `{v.get('path')}` | `{v.get('sha256')}` |")
        out.append("")
    prov = rec.get("provenance") or {}
    if prov:
        out += ["## Provenance", ""]
        for k in (
            "generated_at",
            "platform",
            "git_commit",
            "git_dirty",
            "stack_matches_lock",
            "lock_mismatches",
        ):
            if k in prov:
                out.append(f"- **{k}:** `{prov[k]}`")
        lv = prov.get("library_versions") or {}
        if lv:
            out.append("- **library_versions:** " + ", ".join(f"{k} {v}" for k, v in lv.items()))
        out.append("")
    out += [
        "## How to read the tables",
        "",
        "Each row gives the value as archived (or as printed in the draft when no report held "
        "it) and the corrected value, with where each comes from. *Kind* says what happened: "
        "**changed**; **withdrawn** (an estimate the data cannot identify, with what replaces "
        "it); **relabeled** (same number, corrected meaning); **interval added**; **added** "
        "(new quantity); **archived** (a figure the draft already printed, now produced and "
        "stored by the pipeline; **archived (differs)** when it does not match the draft at the "
        "draft's printed precision). A p-value at a bootstrap's floor is written as a bound.",
        "",
    ]
    for sec in rec["sections"]:
        out += [f"## {sec['title']}", ""]
        if sec.get("triage"):
            out.append(f"Triage: {', '.join(sec['triage'])}. ")
        if sec.get("dissertation_locations"):
            out.append("Where it appears: " + "; ".join(sec["dissertation_locations"]) + ".")
        out += ["", sec["what_changed"], ""]
        out += ["| Quantity | Before | After | Kind |", "|---|---|---|---|"]
        for r in sec["rows"]:
            out.append(
                f"| {_md_cell(r['quantity'])} | {_md_cell(r['before'])} | "
                f"{_md_cell(r['after'])} | {_md_cell(r['kind'])} |"
            )
        out.append("")
        notes = [r for r in sec["rows"] if r.get("note")]
        for r in notes:
            out.append(f"- *{r['quantity']}:* {r['note']}")
        if notes:
            out.append("")
        if sec.get("unchanged"):
            out.append(
                "Unchanged: "
                + "; ".join(f"{u['quantity']} {u['after']}" for u in sec["unchanged"])
                + "."
            )
            out.append("")
        out.append("<details><summary>Sources for each row</summary>")
        out.append("")
        for r in sec["rows"]:
            out.append(
                f"- {r['quantity']}: before `{r['before_source']}`; after `{r['after_source']}`"
            )
        out += ["", "</details>", ""]
    cc = rec["consistency_check"]
    out += [
        "## Consistency check",
        "",
        f"{fnum(cc['n_numbers_compared'])} values present in both reports were compared to "
        f"{cc['significant_figures']} significant figures. Changed where intended: "
        f"{fnum(cc['n_changed_intended'])}. Changed by a later analysis update: "
        f"{fnum(cc.get('n_changed_by_later_updates', 0))}. Changed anywhere else: "
        f"**{fnum(cc['n_changed_unexpected'])}**. Removed: {fnum(cc['n_removed'])}.",
        "",
    ]
    if cc.get("later_update_changes"):
        out += ["Changed by a later analysis update:", ""]
        out += ["| Path | Archived | Corrected | Reason |", "|---|---|---|---|"]
        for u in cc["later_update_changes"]:
            out.append(
                f"| `{_md_cell(u['path'])}` | {_md_cell(str(u['archived'])[:60])} | "
                f"{_md_cell(str(u['corrected'])[:60])} | {_md_cell(u['reason'])} |"
            )
        out.append("")
    if cc.get("intended_changes"):
        out += [
            "<details><summary>Every value that changed where intended "
            f"({fnum(cc['n_changed_intended'])})</summary>",
            "",
            "| Path | Archived | Corrected |",
            "|---|---|---|",
        ]
        for u in cc["intended_changes"]:
            out.append(
                f"| `{_md_cell(u['path'])}` | {_md_cell(str(u['archived'])[:80])} | "
                f"{_md_cell(str(u['corrected'])[:80])} |"
            )
        out += ["", "</details>", ""]
    if cc["unexpected_changes"]:
        out += ["Changed outside the intended corrections:", ""]
        out += ["| Path | Archived | Corrected |", "|---|---|---|"]
        for u in cc["unexpected_changes"][:60]:
            out.append(
                f"| `{_md_cell(u['path'])}` | {_md_cell(str(u['archived'])[:60])} | "
                f"{_md_cell(str(u['corrected'])[:60])} |"
            )
        out.append("")
    if cc["removed_unexpected"]:
        out.append(
            "Removed outside the intended changes: "
            + ", ".join(f"`{p}`" for p in cc["removed_unexpected"][:30])
        )
        out.append("")
    out.append("New blocks: " + ", ".join(f"`{p}`" for p in cc["added_blocks"]))
    out += ["", cc["note"], ""]
    return "\n".join(out)
