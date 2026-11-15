"""RQ3 adaptation indicators: what the attacker did after an exploit failed (descriptive).

Re-run analysis update (2026-10-28; ``docs/rerun_runbook.md`` §6). Under Method C every
genuine service in the four moving conditions moves at each tick, during the attempt, and a
moved service is reported to the planner as ``unreachable``. RQ3's attempt-index models say
whether the attacker gets better across attempts; these indicators say how it adapts WITHIN
an attempt, from the recorded decisions. They are descriptive (no hypothesis test) and
reproduce ``scripts/attacker_behavior/traces.py`` exactly, from the same two columns:

- ``action_sequence`` (every planner decision, with each exploit's target) and
- ``action_timing`` (whether each exploit was stale, and how old its information was).

The outcome word the prompt showed for an executed exploit is reconstructed as in
``traces.py``: ``unreachable`` when the delivery was stale under the clock that drove the
attempt (on the wall clock ``stale_at_completion`` from timing format v3, 2026-10-31, when a
delivery is scored at its completion; ``stale_wall_observed`` in format v2, the October 2026
runs, scored at firing), else ``compromised`` for a genuine service and ``reached`` for a decoy. A row whose cells do not
decode, or whose two cells disagree in length, is counted in ``n_unreadable`` and skipped.

Indicators, per condition (the attempt-level shares carry a sequence cluster bootstrap 95%
interval: whole sequences resampled within the condition):

- ``after_first_unreachable``: the decision right after an attempt's first ``unreachable``
  (moving conditions), and the share of those attempts that scanned again;
- ``decision_after_exploit``: how often the decision right after an executed exploit was a
  scan, by the outcome that exploit showed (all conditions, pooled in ``all_conditions``);
- ``post_rescan_exploits``: in attempts whose first genuine exploit was stale, the exploits
  fired once the attempt had scanned more than once, by target (a decoy; a genuine service not
  yet tried; a genuine service shown unreachable at an earlier identity, i.e. the same service
  found again where it had moved; a genuine service already shown unreachable at this very
  identity), and the information age of the genuine ones (``gap_wall_observed_s``: seconds from
  the probe instant to the firing). Under prompt format 2 the attacker's earlier outcomes are
  keyed by the full identity, so a moved service read as one not yet tried; under format 1
  (every run before the second re-run) they were keyed by name, so it still read
  ``unreachable``. The split holds either reading;
- ``action_budget``: attempts that ended because they used the whole action budget
  (``planner_terminal_reason = action_budget_exhausted``), and every terminal reason;
- ``stops``: the model's own stops against actions the client forced (a decision recorded with
  a note, e.g. a reply with no tool call), by note;
- ``first_genuine_exploit_vs_compromise`` and ``recoveries``: whether the attempt's first
  genuine exploit was current or stale against whether the attempt compromised a genuine
  service (``exploitation_success``); a recovery is a compromise after a stale first genuine
  exploit.
"""

from __future__ import annotations

import collections

import numpy as np
import pandas as pd

from analysis import clustered
from analysis.separation import MOVING_TECHNIQUES
from measurement.action_trace import decode_trace
from measurement.models import CLOCK_MODELED, CLOCK_WALL
from measurement.timing import decode_action_timing

UNREACHABLE = "unreachable"
REACHED = "reached"
COMPROMISED = "compromised"
TERMINAL_BUDGET = "action_budget_exhausted"

CLASS_DECOY = "a decoy"
CLASS_ALREADY_FAILED = "a genuine service already shown unreachable at this identity"
CLASS_FAILED_EARLIER_IDENTITY = "a genuine service shown unreachable at an earlier identity"
CLASS_UNTRIED = "a genuine service not yet tried"
MODEL_OWN_STOP = "the model's own stop"

_NOTE = (
    "Descriptive indicators of within-attempt adaptation (no test), computed from "
    "action_sequence and action_timing exactly as scripts/attacker_behavior/traces.py does. "
    "Shares carry a percentile sequence cluster bootstrap 95% interval (whole sequences "
    "resampled within the condition; pooled rows resample within each condition). An attempt "
    "is the unit of every share; an exploit is the unit of the post-rescan counts and ages. "
    "after_first_unreachable, post_rescan_exploits and recoveries are defined for the four "
    "moving conditions (the deceptive presentation moves decoys only); decision_after_exploit, "
    "action_budget and stops cover every condition."
)


def _stale_as_shown(t, clock_mode: str) -> bool:
    """Whether this delivery was scored stale, on the attempt's own clock.

    On the second re-run's wall clock (timing format v3) this is ``stale_at_completion``: the
    delivery is scored when it completes (decision A1, 2026-10-31). On format-2 wall rows (the
    October 2026 runs, scored at firing) it is ``stale_wall_observed``; format-1 wall rows (the
    2026-10-26 pilot) used ``stale_wall``; modeled-clock rows use ``stale_modeled``. Exactly the
    rule ``traces.py`` applies. A v3 delivery that completed after the trial window
    (``completed_after_window``) was never shown at all: it is always its attempt's last action,
    so no indicator reads its word, and it is stale here only if its target had moved (a first
    genuine exploit that ran out of time reads as current and not compromised).
    """

    if t.stale_at_completion is not None:
        return bool(t.stale_at_completion)
    if clock_mode == CLOCK_MODELED and t.stale_modeled is not None:
        return bool(t.stale_modeled)
    if t.stale_wall_observed is not None:
        return bool(t.stale_wall_observed)
    if clock_mode == CLOCK_WALL and t.stale_wall is not None:
        return bool(t.stale_wall)
    return False


def _shown(t, clock_mode: str) -> str:
    if _stale_as_shown(t, clock_mode):
        return UNREACHABLE
    return COMPROMISED if t.service_genuine else REACHED


def _cell(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


def _attempt(row: dict, moving: bool) -> dict | None:
    """Everything one attempt contributes to the indicators, or None if its cells don't decode."""

    clock_mode = _cell(row.get("clock_mode")) or CLOCK_WALL
    try:
        acts = decode_trace(_cell(row.get("action_sequence")))
        tim = decode_action_timing(row.get("action_timing")) or ()
    except ValueError:
        return None
    if len(acts) != len(tim):
        return None
    stops: collections.Counter = collections.Counter()
    forced: collections.Counter = collections.Counter()
    for a in acts:
        if a.kind == "stop":
            stops[MODEL_OWN_STOP if not a.note else a.note] += 1
        elif a.note:
            forced[a.note] += 1
    scan_after: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    post_rescan: list[tuple[str, float | None, bool]] = []
    prior: list[tuple[object, str]] = []  # (target identity, outcome word shown)
    scans = 0
    first_gen_stale: bool | None = None
    first_unreachable_next: str | None = None
    for k, (a, t) in enumerate(zip(acts, tim, strict=True)):
        prev = acts[k - 1] if k else None
        if prior and prev is not None and prev.kind == "exploit" and prev.executed:
            scan_after[prior[-1][1]][1] += 1
            scan_after[prior[-1][1]][0] += a.kind == "recon"
        if a.kind == "recon" and a.executed:
            scans += 1
        elif a.kind == "exploit" and a.executed and a.target is not None:
            word = _shown(t, clock_mode)
            failed = {target for target, w in prior if w == UNREACHABLE}
            if moving and scans > 1 and first_gen_stale:
                if not t.service_genuine:
                    cls = CLASS_DECOY
                elif a.target in failed:
                    cls = CLASS_ALREADY_FAILED
                elif a.target.service in {f.service for f in failed}:
                    cls = CLASS_FAILED_EARLIER_IDENTITY
                else:
                    cls = CLASS_UNTRIED
                post_rescan.append((cls, t.gap_wall_observed_s, word == UNREACHABLE))
            if t.service_genuine and first_gen_stale is None:
                first_gen_stale = _stale_as_shown(t, clock_mode)
            prior.append((a.target, word))
            if moving and word == UNREACHABLE and first_unreachable_next is None:
                nxt = acts[k + 1] if k + 1 < len(acts) else None
                if nxt is None:
                    first_unreachable_next = "attempt ended"
                elif nxt.kind == "exploit" and nxt.target == a.target:
                    first_unreachable_next = "retried the same target"
                elif nxt.kind == "exploit":
                    first_unreachable_next = "exploited another service"
                elif nxt.kind == "recon":
                    first_unreachable_next = "scanned again"
                else:
                    first_unreachable_next = nxt.kind
    comp = pd.to_numeric(pd.Series([row.get("exploitation_success")]), errors="coerce").iloc[0]
    return {
        "sequence_id": _cell(row.get("sequence_id")),
        "terminal_reason": _cell(row.get("planner_terminal_reason")),
        "stops": stops,
        "forced": forced,
        "scan_after": dict(scan_after),
        "post_rescan": post_rescan,
        "first_gen_stale": first_gen_stale,
        "first_unreachable_next": first_unreachable_next,
        "compromised": None if pd.isna(comp) else bool(comp),
        "n_scans": scans,
    }


def _share(
    groups: list[list[tuple[str, int, int]]],
    *,
    n_boot: int,
    seed: int,
) -> dict:
    """sum(num) / sum(den) over attempts, with a sequence cluster bootstrap interval.

    ``groups`` holds one list per condition (stratum) of (sequence_id, num, den) per attempt.
    """

    strata = []
    for g in groups:
        by_seq: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
        for seq, num, den in g:
            by_seq[seq][0] += num
            by_seq[seq][1] += den
        if by_seq:
            arr = np.array(list(by_seq.values()), dtype=float)
            strata.append(arr)
    num = float(sum(a[:, 0].sum() for a in strata)) if strata else 0.0
    den = float(sum(a[:, 1].sum() for a in strata)) if strata else 0.0
    out: dict = {"count": int(num), "denominator": int(den)}
    if den == 0:
        return {**out, "share": None, "ci95": None}
    rng = np.random.default_rng(seed)
    draws = np.zeros(n_boot)
    dens = np.zeros(n_boot)
    for a in strata:
        idx = rng.integers(0, len(a), size=(n_boot, len(a)))
        draws += a[idx, 0].sum(axis=1)
        dens += a[idx, 1].sum(axis=1)
    ok = dens > 0
    ratio = draws[ok] / dens[ok]
    ci = [float(np.quantile(ratio, 0.025)), float(np.quantile(ratio, 0.975))] if ok.any() else None
    return {**out, "share": num / den, "ci95": ci}


def _ages(values: list[float | None]) -> dict:
    v = np.array([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if not v.size:
        return {"n": 0, "median_s": None, "q1_s": None, "q3_s": None}
    return {
        "n": int(v.size),
        "median_s": float(np.median(v)),
        "q1_s": float(np.quantile(v, 0.25)),
        "q3_s": float(np.quantile(v, 0.75)),
        "min_s": float(v.min()),
        "max_s": float(v.max()),
    }


def _moving_block(atts: list[dict], n_boot: int, seed: int) -> dict:
    after = collections.Counter(
        a["first_unreachable_next"] for a in atts if a["first_unreachable_next"]
    )
    classes = collections.Counter(cls for a in atts for cls, _age, _st in a["post_rescan"])
    genuine = [
        (age, stale) for a in atts for cls, age, stale in a["post_rescan"] if cls != CLASS_DECOY
    ]
    status = {"stale": [0, 0], "current": [0, 0], "none": [0, 0]}
    for a in atts:
        key = (
            "none"
            if a["first_gen_stale"] is None
            else "stale" if a["first_gen_stale"] else "current"
        )
        status[key][0] += 1
        status[key][1] += bool(a["compromised"])
    return {
        "after_first_unreachable": {
            "n_attempts_with_an_unreachable": int(sum(after.values())),
            "next_action": dict(after.most_common()),
            "scanned_again": _share(
                [
                    [
                        (a["sequence_id"], int(a["first_unreachable_next"] == "scanned again"), 1)
                        for a in atts
                        if a["first_unreachable_next"]
                    ]
                ],
                n_boot=n_boot,
                seed=seed,
            ),
        },
        "post_rescan_exploits": {
            "n_attempts_first_genuine_stale": status["stale"][0],
            "n_attempts_with_post_rescan_exploits": int(sum(1 for a in atts if a["post_rescan"])),
            "n_exploits": int(sum(classes.values())),
            "by_target_class": {
                c: int(classes.get(c, 0))
                for c in (
                    CLASS_DECOY,
                    CLASS_UNTRIED,
                    CLASS_FAILED_EARLIER_IDENTITY,
                    CLASS_ALREADY_FAILED,
                )
            },
            "genuine_information_age": {
                **_ages([age for age, _ in genuine]),
                "n_stale": int(sum(1 for _, st in genuine if st)),
            },
        },
        "first_genuine_exploit_vs_compromise": {
            k: {"n_attempts": v[0], "n_compromised": v[1]} for k, v in status.items()
        },
        "recoveries": _share(
            [
                [
                    (a["sequence_id"], int(bool(a["compromised"])), 1)
                    for a in atts
                    if a["first_gen_stale"] is True
                ]
            ],
            n_boot=n_boot,
            seed=seed,
        ),
    }


def _general_block(atts: list[dict], n_boot: int, seed: int) -> dict:
    stops: collections.Counter = collections.Counter()
    forced: collections.Counter = collections.Counter()
    scan_after: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    ends: collections.Counter = collections.Counter()
    for a in atts:
        stops.update(a["stops"])
        forced.update(a["forced"])
        ends[a["terminal_reason"]] += 1
        for word, (s, n) in a["scan_after"].items():
            scan_after[word][0] += s
            scan_after[word][1] += n
    return {
        "n_attempts": len(atts),
        "action_budget": {
            "used_whole_budget": _share(
                [
                    [
                        (a["sequence_id"], int(a["terminal_reason"] == TERMINAL_BUDGET), 1)
                        for a in atts
                    ]
                ],
                n_boot=n_boot,
                seed=seed,
            ),
            "terminal_reasons": dict(ends.most_common()),
        },
        "stops": {
            "model_own_stops": int(stops.get(MODEL_OWN_STOP, 0)),
            "client_forced_stops": {k: int(v) for k, v in stops.items() if k != MODEL_OWN_STOP},
            "client_forced_other_actions": dict(forced.most_common()),
        },
        "decision_after_exploit": {
            word: {
                "scans": int(scan_after[word][0]),
                "decisions": int(scan_after[word][1]),
                "share": (
                    (scan_after[word][0] / scan_after[word][1]) if scan_after[word][1] else None
                ),
            }
            for word in (UNREACHABLE, REACHED, COMPROMISED)
            if word in scan_after
        },
    }


def adaptation_indicators(
    df: pd.DataFrame,
    *,
    n_boot: int | None = None,
    seed: int | None = None,
) -> dict:
    """The RQ3 adaptation indicators per condition, plus the four moving conditions pooled."""

    n_boot = clustered.BOOT_DEFAULT if n_boot is None else int(n_boot)
    seed = clustered.BOOT_SEED_DEFAULT if seed is None else int(seed)
    need = {"condition", "sequence_id", "action_sequence", "action_timing"}
    if not need <= set(df.columns):
        return {
            "computed": False,
            "note": (
                f"needs {sorted(need)}; the dataset lacks {sorted(need - set(df.columns))} "
                "(runs before the action-timing instrumentation)"
            ),
        }
    by_cond: dict[str, list[dict]] = collections.defaultdict(list)
    unreadable: collections.Counter = collections.Counter()
    for row in df.to_dict("records"):
        cond = str(row.get("condition"))
        rec = _attempt(row, cond in MOVING_TECHNIQUES)
        if rec is None:
            unreadable[cond] += 1
        else:
            by_cond[cond].append(rec)
    if not by_cond:
        return {
            "computed": False,
            "n_unreadable": int(sum(unreadable.values())),
            "note": (
                "no row carries a decodable action_sequence and action_timing of the same "
                "length (a run that recorded the action trace but not the action timing, e.g. "
                "an offline run on the modeled clock without attacker.record_timing)"
            ),
        }
    out: dict = {}
    for cond in sorted(set(by_cond) | set(unreadable)):
        atts = by_cond.get(cond, [])
        block = {**_general_block(atts, n_boot, seed), "n_unreadable": int(unreadable[cond])}
        if cond in MOVING_TECHNIQUES:
            block.update(_moving_block(atts, n_boot, seed))
        out[cond] = block

    moving = [c for c in MOVING_TECHNIQUES if c in by_cond]
    pooled: dict = {}
    if moving:
        groups = {c: by_cond[c] for c in moving}
        pooled = {
            "conditions": moving,
            "n_attempts": int(sum(len(v) for v in groups.values())),
            "scanned_again_after_first_unreachable": _share(
                [
                    [
                        (a["sequence_id"], int(a["first_unreachable_next"] == "scanned again"), 1)
                        for a in g
                        if a["first_unreachable_next"]
                    ]
                    for g in groups.values()
                ],
                n_boot=n_boot,
                seed=seed,
            ),
            "used_whole_budget": _share(
                [
                    [(a["sequence_id"], int(a["terminal_reason"] == TERMINAL_BUDGET), 1) for a in g]
                    for g in groups.values()
                ],
                n_boot=n_boot,
                seed=seed,
            ),
            "recoveries": _share(
                [
                    [
                        (a["sequence_id"], int(bool(a["compromised"])), 1)
                        for a in g
                        if a["first_gen_stale"] is True
                    ]
                    for g in groups.values()
                ],
                n_boot=n_boot,
                seed=seed,
            ),
            "post_rescan_genuine_information_age": _ages(
                [
                    age
                    for g in groups.values()
                    for a in g
                    for cls, age, _st in a["post_rescan"]
                    if cls != CLASS_DECOY
                ]
            ),
        }
    all_scan_after: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for atts in by_cond.values():
        for a in atts:
            for word, (s, n) in a["scan_after"].items():
                all_scan_after[word][0] += s
                all_scan_after[word][1] += n
    return {
        "computed": True,
        "by_condition": out,
        "moving_conditions_pooled": pooled,
        "all_conditions": {
            "n_attempts": int(sum(len(v) for v in by_cond.values())),
            "n_unreadable": int(sum(unreadable.values())),
            "decision_after_exploit": {
                w: {
                    "scans": int(all_scan_after[w][0]),
                    "decisions": int(all_scan_after[w][1]),
                    "share": (
                        (all_scan_after[w][0] / all_scan_after[w][1])
                        if all_scan_after[w][1]
                        else None
                    ),
                }
                for w in (UNREACHABLE, REACHED, COMPROMISED)
                if w in all_scan_after
            },
        },
        "n_boot": int(n_boot),
        "bootstrap_seed": int(seed),
        "note": _NOTE,
    }
