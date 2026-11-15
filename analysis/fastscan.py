"""The fast-scan check set against the main study (decision D1 of 2026-10-29).

The fast-scan check runs the re-run's instrument with the reconnaissance scan aimed at the
targets host only (``live.scan_scope: target_host``, about 1 s) instead of the whole subnet
(about 27 s), in five conditions (deceptive presentation is left out: it moves only the decoys)
with 20 sequences each (``harness/experiment_config.full_fastscan.yaml``). Its job is to show
how much the scan's length drives the main study's result.

It is a PLANNED SENSITIVITY, reported as estimates (decided 2026-10-29, before the check runs):

- for each condition both runs share, the compromise proportion (``exploitation_success``) and
  the restricted mean time to compromise over [0, 300 s] under each scan, and the fast - subnet
  difference with a 95% interval;
- the moving conditions' benefit over the static baseline under each scan (each condition's
  own, and their equal-weight mean), and how much it changes.

No hypothesis is decided by it and no p-value is reported: 20 sequences per condition give
usable intervals but little power, and a non-significant test would be easy to misread as
evidence that scan length does not matter.

Inference is a sequence cluster bootstrap run separately in each dataset (whole sequences
resampled within each condition; the two runs are independent samples, with their own
per-sequence seeds), so a difference between the runs takes its interval from two independent
sets of replicates. The runs happen one after the other on the same testbed, so a difference
between them also carries any drift in the testbed between the runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import clustered
from analysis.dataset import STATIC
from analysis.separation import MOVING_TECHNIQUES

#: The date the comparison was specified (decision D1; before the fast-scan check ran).
FASTSCAN_RULE_DATE = "2026-10-29"
HORIZON_S = 300.0
_NEED = (
    "condition",
    "sequence_id",
    "exploitation_success",
    "time_to_compromise_s",
    "ttc_censored",
)

_ROLE = (
    "PLANNED SENSITIVITY (decided 2026-10-29, before the fast-scan check ran): estimates with "
    "95% intervals only. It decides no hypothesis and reports no p-value."
)
_ORIENTATION = {
    "compromise_proportion": (
        "share of trials compromised (exploitation_success). A positive fast - subnet difference "
        "means more compromises under the fast scan. A benefit over static is condition - "
        "static: NEGATIVE = the moving condition lowers compromise"
    ),
    "rmst_300s_s": (
        "restricted mean time to compromise over [0, 300 s], seconds. A positive fast - subnet "
        "difference means the condition stays uncompromised longer under the fast scan. A "
        "benefit over static is condition - static: POSITIVE = the moving condition delays "
        "compromise"
    ),
}
_NOTES = {
    "independent_samples": (
        "The two runs are independent samples (their sequences have their own seeds), so each "
        "run is bootstrapped separately: whole sequences resampled within each condition. An "
        "interval for a difference between the runs combines the two runs' replicates."
    ),
    "drift": (
        "The runs happen back to back on the same testbed, so a difference between them also "
        "carries any drift between the runs; the main study's interleaved drift check "
        "(design_checks.interleaved_drift) shows how stable the testbed was within it."
    ),
    "no_tests": (
        "No p-value is reported: with 20 sequences per condition a test has little power, and a "
        "non-significant result would read as evidence that scan length does not matter."
    ),
    "restricted_time": (
        "RMST over [0, 300 s] is mean(min(T, 300)): exact here because every censored trial is "
        "censored at the trial window. A trial censored before 300 s would have no determined "
        "restricted time; it is left out of the RMST (not of the compromise proportion) and "
        "counted in runs.<run>.censored_before_horizon_left_out_of_rmst (none is expected)."
    ),
}


def _prepared(d: pd.DataFrame, horizon: float) -> tuple[pd.DataFrame, int]:
    """Trials with an outcome; ``_rt`` is null where the restricted time is not determined.

    The compromise proportion uses every trial with an outcome. The RMST leaves out a trial
    censored before the horizon (its restricted time is not determined), which is counted.
    """

    d = d.dropna(subset=["condition", "sequence_id", "exploitation_success"]).copy()
    y = pd.to_numeric(d["exploitation_success"], errors="coerce")
    t = pd.to_numeric(d["time_to_compromise_s"], errors="coerce")
    c = pd.to_numeric(d["ttc_censored"], errors="coerce")
    early = (c == 1) & (t < horizon)
    d["_y"] = y
    d["_rt"] = np.minimum(t, horizon).where(~early & t.notna() & c.notna())
    d = d[y.notna()]
    d["condition"] = d["condition"].astype(str)
    d["sequence_id"] = d["sequence_id"].astype(str)
    return d, int(early.sum())


def _condition_draws(d: pd.DataFrame, conds: list[str], n_boot: int, rng) -> dict:
    """Per condition: point estimates and joint replicates of the proportion and the RMST.

    Both resample the same sequences in each replicate; the proportion's denominator is the
    trials with an outcome, the RMST's the trials whose restricted time is determined.
    """

    out: dict = {}
    for c in conds:
        g = d[d["condition"] == c].assign(_rt_n=lambda f: f["_rt"].notna().astype(float))
        agg = g.groupby("sequence_id", sort=True).agg(
            y=("_y", "sum"), n=("_y", "size"), rt=("_rt", "sum"), rt_n=("_rt_n", "sum")
        )
        ys = agg["y"].to_numpy(dtype=float)
        ns = agg["n"].to_numpy(dtype=float)
        rts = agg["rt"].to_numpy(dtype=float)
        rns = agg["rt_n"].to_numpy(dtype=float)
        k = len(ns)
        idx = rng.integers(0, k, size=(int(n_boot), k))
        with np.errstate(invalid="ignore", divide="ignore"):
            rmst_draws = rts[idx].sum(axis=1) / rns[idx].sum(axis=1)
        out[c] = {
            "n_trials": int(ns.sum()),
            "n_sequences": int(k),
            "prop": float(ys.sum() / ns.sum()),
            "rmst": float(rts.sum() / rns.sum()) if rns.sum() > 0 else float("nan"),
            "prop_draws": ys[idx].sum(axis=1) / ns[idx].sum(axis=1),
            "rmst_draws": rmst_draws,
        }
    return out


def _ci(x: np.ndarray) -> list[float] | None:
    x = x[np.isfinite(x)]
    return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))] if x.size else None


def _pair(sub_point, sub_draws, fast_point, fast_draws) -> dict:
    return {
        "subnet": sub_point,
        "subnet_ci95": _ci(sub_draws),
        "fast": fast_point,
        "fast_ci95": _ci(fast_draws),
        "fast_minus_subnet": fast_point - sub_point,
        "fast_minus_subnet_ci95": _ci(fast_draws - sub_draws),
    }


def compare(
    main: pd.DataFrame,
    fast: pd.DataFrame,
    *,
    horizon: float = HORIZON_S,
    n_boot: int = clustered.BOOT_DEFAULT,
    seed: int = clustered.BOOT_SEED_DEFAULT,
) -> dict:
    """The fast-scan check against the main study: estimates and intervals, no tests.

    ``main`` and ``fast`` are screened trial frames (``analysis.dataset.prepare``).
    """

    for name, frame in (("main", main), ("fast", fast)):
        missing = [c for c in _NEED if c not in frame.columns]
        if missing:
            return {"role": _ROLE, "note": f"the {name} dataset lacks columns {missing}"}
    dm, early_m = _prepared(main, horizon)
    dfa, early_f = _prepared(fast, horizon)
    cm, cf = set(dm["condition"]), set(dfa["condition"])
    shared = sorted(cm & cf)
    if not shared:
        return {"role": _ROLE, "note": "the two datasets share no condition"}
    rng_main, rng_fast = (
        np.random.default_rng(s) for s in np.random.SeedSequence(int(seed)).spawn(2)
    )
    dr_m = _condition_draws(dm, shared, n_boot, rng_main)
    dr_f = _condition_draws(dfa, shared, n_boot, rng_fast)

    by_condition: dict = {}
    for c in shared:
        m, f = dr_m[c], dr_f[c]
        by_condition[c] = {
            "n_sequences": {"subnet": m["n_sequences"], "fast": f["n_sequences"]},
            "n_trials": {"subnet": m["n_trials"], "fast": f["n_trials"]},
            "compromise_proportion": _pair(m["prop"], m["prop_draws"], f["prop"], f["prop_draws"]),
            "rmst_300s_s": _pair(m["rmst"], m["rmst_draws"], f["rmst"], f["rmst_draws"]),
        }

    moving = [c for c in MOVING_TECHNIQUES if c in shared]
    benefit: dict
    if STATIC not in shared or not moving:
        benefit = {"note": "needs the static baseline and at least one moving condition in both"}
    else:
        benefit = {
            "conditions": moving,
            "weighting": "equal weight per moving condition",
            "by_condition": {},
        }
        for key, pk, dk in (
            ("compromise_proportion", "prop", "prop_draws"),
            ("rmst_300s_s", "rmst", "rmst_draws"),
        ):
            pm = float(np.mean([dr_m[c][pk] for c in moving])) - dr_m[STATIC][pk]
            pf = float(np.mean([dr_f[c][pk] for c in moving])) - dr_f[STATIC][pk]
            bm = np.mean([dr_m[c][dk] for c in moving], axis=0) - dr_m[STATIC][dk]
            bf = np.mean([dr_f[c][dk] for c in moving], axis=0) - dr_f[STATIC][dk]
            benefit[key] = {
                "subnet": pm,
                "subnet_ci95": _ci(bm),
                "fast": pf,
                "fast_ci95": _ci(bf),
                "change_fast_minus_subnet": pf - pm,
                "change_ci95": _ci(bf - bm),
            }
            for c in moving:
                one_m = dr_m[c][pk] - dr_m[STATIC][pk]
                one_f = dr_f[c][pk] - dr_f[STATIC][pk]
                draws_m = dr_m[c][dk] - dr_m[STATIC][dk]
                draws_f = dr_f[c][dk] - dr_f[STATIC][dk]
                benefit["by_condition"].setdefault(c, {})[key] = {
                    "subnet": one_m,
                    "subnet_ci95": _ci(draws_m),
                    "fast": one_f,
                    "fast_ci95": _ci(draws_f),
                    "change_fast_minus_subnet": one_f - one_m,
                    "change_ci95": _ci(draws_f - draws_m),
                }

    return {
        "role": _ROLE,
        "rule_date": FASTSCAN_RULE_DATE,
        "horizon_s": float(horizon),
        "conditions_shared": shared,
        "conditions_main_only": sorted(cm - cf),
        "conditions_fast_only": sorted(cf - cm),
        "runs": {
            "main": {
                "scan": "subnet sweep",
                "n_trials": int(len(dm)),
                "n_sequences": int(dm["sequence_id"].nunique()),
                "censored_before_horizon_left_out_of_rmst": early_m,
            },
            "fast": {
                "scan": "targets host only",
                "n_trials": int(len(dfa)),
                "n_sequences": int(dfa["sequence_id"].nunique()),
                "censored_before_horizon_left_out_of_rmst": early_f,
            },
        },
        "by_condition": by_condition,
        "moving_benefit_over_static": benefit,
        "orientation": _ORIENTATION,
        "n_boot": int(n_boot),
        "seed": int(seed),
        "notes": _NOTES,
    }


def summary_lines(report: dict) -> list[str]:
    """A short plain-text table of the comparison for the terminal."""

    if "by_condition" not in report:
        return [f"fast-scan comparison not computed: {report.get('note')}"]

    def fmt(v, nd):
        return "  n/a" if v is None else f"{v:.{nd}f}"

    def iv(ci, nd):
        return "[n/a]" if not ci else f"[{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"

    lines = [
        "fast-scan check vs main study (planned sensitivity; estimates only, no tests)",
        f"{'condition':<24} {'compromised: subnet / fast / diff [95%]':<46} "
        f"{'RMST 300 s: subnet / fast / diff [95%]':<44}",
    ]
    for c, v in report["by_condition"].items():
        p, r = v["compromise_proportion"], v["rmst_300s_s"]
        lines.append(
            f"{c:<24} {fmt(p['subnet'], 3)} / {fmt(p['fast'], 3)} / "
            f"{fmt(p['fast_minus_subnet'], 3)} {iv(p['fast_minus_subnet_ci95'], 3):<18} "
            f"{fmt(r['subnet'], 1)} / {fmt(r['fast'], 1)} / {fmt(r['fast_minus_subnet'], 1)} "
            f"{iv(r['fast_minus_subnet_ci95'], 1)}"
        )
    b = report.get("moving_benefit_over_static") or {}
    if "compromise_proportion" in b:
        bp, br = b["compromise_proportion"], b["rmst_300s_s"]
        lines.append(
            "moving conditions' benefit over static (equal-weight mean): compromise "
            f"{fmt(bp['subnet'], 3)} -> {fmt(bp['fast'], 3)}, change "
            f"{fmt(bp['change_fast_minus_subnet'], 3)} {iv(bp['change_ci95'], 3)}; RMST "
            f"{fmt(br['subnet'], 1)} -> {fmt(br['fast'], 1)} s, change "
            f"{fmt(br['change_fast_minus_subnet'], 1)} {iv(br['change_ci95'], 1)}"
        )
    return lines
