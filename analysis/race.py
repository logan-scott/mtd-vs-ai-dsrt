"""The race between the attacker's scan-decide-deliver loop and the MTD ticks (descriptive).

Pre-specified on 2026-10-31, before the second re-run collected any data
(``docs/rerun_runbook.md`` §7p). In the October 2026 main study, whether an attempt in a
condition that moves genuine services was compromised depended almost entirely on how long its
first scan took: 41% of those attempts were compromised when the scan took at most 21 s, 0.4%
when it took more than 32 s, against a 35-s cadence. Scored at completion (decision A1), a
delivery must also finish before the next tick. This block describes that race. It tests
nothing and enters no FDR family.

Per condition, and pooled over the four conditions that move genuine services:

- ``compromise_by_first_scan``: the share of attempts compromised (``exploitation_success``) by
  how long the attempt's first scan took, in fixed bins (under 20 s, 20-25, 25-30, 30-35, 35 s
  or more), with a sequence cluster bootstrap 95% interval, and the durations themselves
  (``first_scan_s``). The static baseline and the deceptive presentation are the controls:
  nothing genuine moves there, so the share should not fall with the scan's duration.
- ``deliveries``: every executed exploit at a genuine service that was still in force when it
  was fired, by target: how many, how long the delivery took, how many a tick overtook before
  they completed (``stale_at_completion``, timing format 3; None before it) and how many
  completed after the window (``completed_after_window``).
- ``slack_at_firing_s`` (moving conditions): for those deliveries, the seconds from firing to
  the next tick. A delivery that completes inside the window is overtaken exactly when its
  slack is no longer than its delivery; ``overtaken_check`` counts the deliveries where the two
  disagree (expected: none).

The ticks are rebuilt from each attempt's seed (``harness.trial_runner.attempt_seed`` and
``mtd.orchestrator.rotation_phase_offset``: ``start + φ + k·cadence``, Method C) and checked
against the planner's own decision at firing (``stale_wall_observed``): in the four moving
conditions every tick moves every genuine service, so a genuine identity observed at ``t_o``
is stale when fired at ``t_f`` exactly when a tick falls in ``(t_o, t_f]``. ``tick_check``
counts the disagreements (expected: none; on the October 2026 data, 0 of 11,541).

Real seconds only: a row of the modeled clock, or without a decodable ``action_timing``, is
counted in ``n_without_timing`` and otherwise left out.
"""

from __future__ import annotations

import collections
import math
import re

import numpy as np
import pandas as pd

from analysis import clustered
from analysis.adaptation import _share
from measurement.action_trace import decode_trace
from measurement.timing import decode_action_timing
from testbed.profile import default_profile

#: The conditions whose techniques move every genuine service at every tick.
MOVING: tuple[str, ...] = (
    "combined_mtd_deception",
    "endpoint_mutation",
    "network_shuffling",
    "port_rotation",
)

#: First-scan duration bins (label, lower bound inclusive, upper bound exclusive), seconds.
SCAN_BINS: tuple[tuple[str, float, float], ...] = (
    ("under 20 s", 0.0, 20.0),
    ("20-25 s", 20.0, 25.0),
    ("25-30 s", 25.0, 30.0),
    ("30-35 s", 30.0, 35.0),
    ("35 s or more", 35.0, math.inf),
)

_REGEN = re.compile(r"-r(\d+)$")

_NOTE = (
    "Descriptive (no test, no FDR family); pre-specified 2026-10-31 before the second re-run "
    "collected data. Wall-clock rows only. compromise_by_first_scan: share of attempts "
    "compromised by the duration of the attempt's first scan, percentile sequence cluster "
    "bootstrap 95% interval (pooled rows resample within each condition). deliveries: executed "
    "exploits at genuine services still in force when fired, by target. slack_at_firing_s: "
    "seconds from firing to the next tick (moving conditions). Ticks are rebuilt from each "
    "attempt's seed and checked against the planner's own decision at firing (tick_check); a "
    "delivery that completes inside the window is overtaken exactly when slack <= delivery "
    "(overtaken_check). Both checks expect 0 disagreements."
)


def _cell(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


def _num(value) -> float | None:
    v = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(v) else float(v)


def _grid(row: dict) -> tuple[float, float, float] | None:
    """The attempt's tick grid as ``(phase, cadence, horizon)`` seconds, or None."""

    from harness.trial_runner import attempt_seed
    from mtd.orchestrator import rotation_phase_offset

    cadence = _num(row.get("mtd_cadence_s"))
    seed = _num(row.get("randomization_seed"))
    attempt = _num(row.get("attempt_index"))
    horizon = _num(row.get("trial_duration_limit_s"))
    if None in (cadence, seed, attempt, horizon) or cadence <= 0:
        return None
    m = _REGEN.search(_cell(row.get("trial_id")))
    base = attempt_seed(int(seed), int(attempt), int(m.group(1)) if m else 0)
    return rotation_phase_offset(base ^ 0x1, int(cadence)), cadence, horizon


def _next_tick(x: float, phase: float, cadence: float, horizon: float) -> float:
    """The first tick strictly after ``x`` (``inf`` past the window)."""

    tick = phase if x < phase else phase + (math.floor((x - phase) / cadence) + 1) * cadence
    if tick <= 0.0:  # a phase of 0: the grid starts one cadence in
        tick += cadence
    return tick if tick <= horizon else math.inf


def _attempt(row: dict, genuine: frozenset[str]) -> dict | None:
    """One wall-clock attempt's race facts, or None without usable timing."""

    if _cell(row.get("clock_mode")) != "wall":
        return None
    try:
        acts = decode_trace(_cell(row.get("action_sequence")))
        tim = decode_action_timing(row.get("action_timing"))
    except ValueError:
        return None
    if tim is None or len(tim) != len(acts):
        return None
    cond = _cell(row.get("condition"))
    grid = _grid(row) if cond in MOVING else None
    first_scan = next(
        (
            t.exec_s
            for a, t in zip(acts, tim, strict=True)
            if a.kind == "recon" and t.exec_s is not None
        ),
        None,
    )
    deliveries: list[dict] = []
    tick_checked = tick_disagree = over_checked = over_disagree = 0
    for a, t in zip(acts, tim, strict=True):
        if not (a.kind == "exploit" and a.executed and a.target is not None):
            continue
        if a.target.service not in genuine or t.exec_start_s is None or t.exec_s is None:
            continue
        fire = float(t.exec_start_s)
        if grid is not None and t.gap_wall_observed_s is not None:
            if t.stale_wall_observed is not None:
                observed = fire - float(t.gap_wall_observed_s)
                rebuilt = _next_tick(observed, *grid) <= fire
                tick_checked += 1
                tick_disagree += rebuilt != bool(t.stale_wall_observed)
        if t.stale_wall_observed is not False:
            continue  # stale when fired: the executor delivered nothing
        slack = None
        if grid is not None:
            slack = _next_tick(fire, *grid) - fire
        overtaken = None if t.stale_at_completion is None else bool(t.stale_at_completion)
        late = bool(t.completed_after_window)
        if overtaken is not None and slack is not None and not late:
            over_checked += 1
            over_disagree += (slack <= float(t.exec_s)) != overtaken
        deliveries.append(
            {
                "target": a.target.service,
                "delivery_s": float(t.exec_s),
                "overtaken": overtaken,
                "late": late,
                "slack_s": slack,
            }
        )
    success = _num(row.get("exploitation_success"))
    return {
        "condition": cond,
        "sequence_id": _cell(row.get("sequence_id")),
        "first_scan_s": None if first_scan is None else float(first_scan),
        "compromised": int(bool(success)),
        "deliveries": deliveries,
        "tick_checked": tick_checked,
        "tick_disagree": tick_disagree,
        "over_checked": over_checked,
        "over_disagree": over_disagree,
    }


def _quartiles(values: list[float]) -> dict:
    v = np.array([x for x in values if x is not None and math.isfinite(x)], dtype=float)
    if not v.size:
        return {"n": 0, "median": None, "q1": None, "q3": None, "min": None, "max": None}
    return {
        "n": int(v.size),
        "median": round(float(np.median(v)), 3),
        "q1": round(float(np.quantile(v, 0.25)), 3),
        "q3": round(float(np.quantile(v, 0.75)), 3),
        "min": round(float(v.min()), 3),
        "max": round(float(v.max()), 3),
    }


def _bin(seconds: float) -> str:
    return next(label for label, lo, hi in SCAN_BINS if lo <= seconds < hi)


def _by_first_scan(strata: list[list[dict]], n_boot: int, seed: int) -> dict:
    """The compromised share per first-scan bin. Every attempt of the stratum enters each bin's
    bootstrap, as 0/0 outside the bin, so whole sequences are resampled whether or not they have
    an attempt in that bin (a bin's attempt count varies across replicates, as it would across
    studies)."""

    out: dict = {}
    for label, _, _ in SCAN_BINS:
        groups = []
        for atts in strata:
            group = []
            for a in atts:
                inside = a["first_scan_s"] is not None and _bin(a["first_scan_s"]) == label
                group.append((a["sequence_id"], a["compromised"] if inside else 0, int(inside)))
            groups.append(group)
        out[label] = _share(groups, n_boot=n_boot, seed=seed)
    out["n_without_scan"] = int(sum(a["first_scan_s"] is None for atts in strata for a in atts))
    return out


def _by_target(atts: list[dict]) -> dict:
    by: dict[str, list[dict]] = collections.defaultdict(list)
    for a in atts:
        for d in a["deliveries"]:
            by[d["target"]].append(d)
    out: dict = {}
    for target, ds in sorted(by.items()):
        known = [d for d in ds if d["overtaken"] is not None]
        out[target] = {
            "n": len(ds),
            "delivery_s": _quartiles([d["delivery_s"] for d in ds]),
            "overtaken": (
                {"count": int(sum(d["overtaken"] for d in known)), "denominator": len(known)}
                if known
                else None
            ),
            "completed_after_window": int(sum(d["late"] for d in ds)),
        }
    return out


def _block(strata: list[list[dict]], moving: bool, n_boot: int, seed: int) -> dict:
    atts = [a for s in strata for a in s]
    out = {
        "n_attempts": len(atts),
        "first_scan_s": _quartiles([a["first_scan_s"] for a in atts]),
        "compromise_by_first_scan": _by_first_scan(strata, n_boot, seed),
        "deliveries": _by_target(atts),
    }
    if moving:
        out["slack_at_firing_s"] = _quartiles([d["slack_s"] for a in atts for d in a["deliveries"]])
        out["tick_check"] = {
            "n_exploits_checked": int(sum(a["tick_checked"] for a in atts)),
            "n_disagreements": int(sum(a["tick_disagree"] for a in atts)),
        }
        out["overtaken_check"] = {
            "n_deliveries_checked": int(sum(a["over_checked"] for a in atts)),
            "n_disagreements": int(sum(a["over_disagree"] for a in atts)),
        }
    return out


def race(df: pd.DataFrame, *, n_boot: int | None = None, seed: int | None = None) -> dict:
    """The scan-decide-deliver race against the ticks, by condition (descriptive)."""

    n_boot = clustered.BOOT_DEFAULT if n_boot is None else int(n_boot)
    seed = clustered.BOOT_SEED_DEFAULT if seed is None else int(seed)
    need = {"condition", "sequence_id", "action_sequence", "action_timing", "clock_mode"}
    if not need <= set(df.columns):
        return {
            "computed": False,
            "note": (
                f"needs {sorted(need)}; the dataset lacks {sorted(need - set(df.columns))} "
                "(runs before the real-clock timing instrumentation)"
            ),
        }
    genuine = frozenset(s.service_name for s in default_profile().services if s.genuine)
    by_cond: dict[str, list[dict]] = collections.defaultdict(list)
    without: collections.Counter = collections.Counter()
    for row in df.to_dict("records"):
        rec = _attempt(row, genuine)
        if rec is None:
            without[_cell(row.get("condition"))] += 1
        else:
            by_cond[rec["condition"]].append(rec)
    if not by_cond:
        return {
            "computed": False,
            "n_without_timing": int(sum(without.values())),
            "note": "no row carries wall-clock action_timing (the race is in real seconds)",
        }
    conditions = sorted(set(by_cond) | set(without))
    out = {
        c: {
            **_block([by_cond.get(c, [])], c in MOVING, n_boot, seed),
            "n_without_timing": int(without[c]),
        }
        for c in conditions
    }
    moving = [by_cond[c] for c in MOVING if c in by_cond]
    return {
        "computed": True,
        "scan_bins_s": [label for label, _, _ in SCAN_BINS],
        "by_condition": out,
        "moving_conditions_pooled": (
            {
                "conditions": [c for c in MOVING if c in by_cond],
                **_block(moving, True, n_boot, seed),
            }
            if moving
            else None
        ),
        "n_boot": int(n_boot),
        "bootstrap_seed": int(seed),
        "note": _NOTE,
    }
