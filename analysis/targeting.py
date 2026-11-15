"""What the attacker's exploits were aimed at, by service NAME (descriptive; decision A2).

Second re-run, decision A2 (2026-10-31; ``docs/rerun_runbook.md`` §7n). The row columns of
``measurement/decoy_engagement.py`` (metrics spec §4e) classify a delivery from its
``ExploitDelivery`` record: ``target_genuine`` is False both for a decoy and for an identity
MTD had already moved, so a stale delivery's target is recorded as ``unresolved``. Under the
real clock most first exploits in the moving conditions are stale, so ``first_exploit_target``
and the decoy counts say little there. Those columns stay as recorded; this block classifies
every executed exploit again, at analysis time, by the SERVICE NAME of its target in
``action_sequence``:

- a name of one of the vulnerability profile's decoys (``http-admin``, ``ssh-gateway``,
  ``ftp-backup``) is decoy-aimed;
- a name of one of its genuine services is genuine-aimed, whether or not that identity was
  still in force;
- any other name is counted as ``other`` (none is expected: the planner can only aim at a
  listed service).

No technique renames a service (``attacker/tools.py``; names are unique within an
inventory), so the name says what the planner aimed at in every condition, at any identity.
It is the rule the planner applies at run time for ``ActionTiming.service_genuine`` (``g``),
so where timing is recorded every classified exploit is checked against ``g``
(``timing_cross_check``; expected: no disagreement).

Per condition: the first exploit's target (genuine / decoy / other), the share of attempts
whose first exploit was decoy-aimed, the share that aimed any exploit at a decoy, the share of
all exploits that were decoy-aimed, the 1-based ordinal of the first decoy-aimed exploit (the
base of ``first_decoy_delivery_ordinal`` and ``attempts_per_success``), the same first-target
counts by attempt index (with memory, decoy-aimed exploits may fade across attempts: RQ3's
question). Shares carry a sequence cluster bootstrap 95% interval, as in
``rq3.adaptation_indicators``. Descriptive only: no test, no FDR family.

It reads the trace (and, for the cross-check, the timing) only, never the three decoy-engagement
row columns: those stay outside the analysis report altogether
(``tests/test_analysis.py::test_decoy_engagement_columns_cannot_enter_or_resize_any_fdr_family``).
"""

from __future__ import annotations

import collections

import numpy as np
import pandas as pd

from analysis import clustered
from analysis.adaptation import _share
from measurement.action_trace import decode_trace
from measurement.timing import decode_action_timing
from testbed.profile import default_profile

GENUINE = "genuine"
DECOY = "decoy"
OTHER = "other"

_NOTE = (
    "Descriptive (no test). Every executed exploit in action_sequence is classified by the "
    "service name of its target: a profile decoy name is decoy-aimed, a profile genuine name "
    "is genuine-aimed whether or not that identity was still in force (names never change "
    "under any technique). Decision A2 of 2026-10-31: the per-trial decoy-engagement columns "
    "(metrics spec 4e) stay as recorded and outside this report; they label a stale delivery "
    "'unresolved'. Attempt-level shares carry a percentile sequence cluster "
    "bootstrap 95% interval (whole sequences resampled within the condition); the exploit-"
    "level share resamples sequences too. first_decoy_ordinal is 1-based among the attempt's "
    "executed exploits. timing_cross_check compares the name rule with the planner's own "
    "service_genuine flag wherever action_timing was recorded (expected: 0 disagreements)."
)


def _names() -> tuple[frozenset[str], frozenset[str]]:
    services = default_profile().services
    genuine = frozenset(s.service_name for s in services if s.genuine)
    decoys = frozenset(s.service_name for s in services if not s.genuine)
    return genuine, decoys


def _cell(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


def _aimed(service: str, genuine: frozenset[str], decoys: frozenset[str]) -> str:
    if service in decoys:
        return DECOY
    if service in genuine:
        return GENUINE
    return OTHER


def _attempt(row: dict, genuine: frozenset[str], decoys: frozenset[str]) -> dict | None:
    """One attempt's exploit targets by name, or None if its trace does not decode.

    Where the row's ``action_timing`` decodes and matches the trace in length, each exploit
    classified genuine or decoy is checked against the planner's own ``service_genuine``.
    """

    try:
        acts = decode_trace(_cell(row.get("action_sequence")))
    except ValueError:
        return None
    try:
        tim = decode_action_timing(row.get("action_timing"))
    except ValueError:
        tim = None
    if tim is not None and len(tim) != len(acts):
        tim = None
    aimed: list[str] = []
    checked = disagree = 0
    for k, a in enumerate(acts):
        if not (a.kind == "exploit" and a.executed and a.target is not None):
            continue
        cls = _aimed(a.target.service, genuine, decoys)
        aimed.append(cls)
        if tim is not None and cls != OTHER and tim[k].service_genuine is not None:
            checked += 1
            disagree += (cls == GENUINE) != bool(tim[k].service_genuine)
    first_decoy = next((i for i, c in enumerate(aimed, start=1) if c == DECOY), None)
    idx = pd.to_numeric(pd.Series([row.get("attempt_index")]), errors="coerce").iloc[0]
    return {
        "sequence_id": _cell(row.get("sequence_id")),
        "attempt_index": None if pd.isna(idx) else int(idx),
        "aimed": aimed,
        "first": aimed[0] if aimed else None,
        "first_decoy_ordinal": first_decoy,
        "checked": checked,
        "disagree": disagree,
    }


def _ordinals(values: list[int]) -> dict:
    v = np.array(values, dtype=float)
    if not v.size:
        return {"n": 0, "median": None, "q1": None, "q3": None, "min": None, "max": None}
    return {
        "n": int(v.size),
        "median": float(np.median(v)),
        "q1": float(np.quantile(v, 0.25)),
        "q3": float(np.quantile(v, 0.75)),
        "min": int(v.min()),
        "max": int(v.max()),
    }


def _block(atts: list[dict], n_boot: int, seed: int) -> dict:
    with_exploit = [a for a in atts if a["aimed"]]
    first = collections.Counter(a["first"] for a in with_exploit)
    by_index: dict[int, list[int]] = collections.defaultdict(lambda: [0, 0, 0])
    for a in with_exploit:
        if a["attempt_index"] is None:
            continue
        cell = by_index[a["attempt_index"]]
        cell[0] += 1
        cell[1] += a["first"] == DECOY
        cell[2] += DECOY in a["aimed"]
    n_exploits = sum(len(a["aimed"]) for a in atts)
    n_decoy = sum(c == DECOY for a in atts for c in a["aimed"])
    return {
        "n_attempts": len(atts),
        "n_attempts_with_exploit": len(with_exploit),
        "first_exploit_aimed_at": {k: int(first.get(k, 0)) for k in (GENUINE, DECOY, OTHER)},
        "first_exploit_at_a_decoy": _share(
            [[(a["sequence_id"], int(a["first"] == DECOY), 1) for a in with_exploit]],
            n_boot=n_boot,
            seed=seed,
        ),
        "any_exploit_at_a_decoy": _share(
            [[(a["sequence_id"], int(DECOY in a["aimed"]), 1) for a in with_exploit]],
            n_boot=n_boot,
            seed=seed,
        ),
        "exploits": {
            "n": int(n_exploits),
            "at_a_decoy": int(n_decoy),
            "share_at_a_decoy": _share(
                [
                    [
                        (a["sequence_id"], sum(c == DECOY for c in a["aimed"]), len(a["aimed"]))
                        for a in with_exploit
                    ]
                ],
                n_boot=n_boot,
                seed=seed,
            ),
        },
        "first_decoy_ordinal": _ordinals(
            [a["first_decoy_ordinal"] for a in atts if a["first_decoy_ordinal"] is not None]
        ),
        "by_attempt_index": {
            str(i): {
                "n_attempts_with_exploit": v[0],
                "first_exploit_at_a_decoy": v[1],
                "any_exploit_at_a_decoy": v[2],
            }
            for i, v in sorted(by_index.items())
        },
    }


def decoy_targeting(
    df: pd.DataFrame,
    *,
    n_boot: int | None = None,
    seed: int | None = None,
) -> dict:
    """What each condition's exploits were aimed at, by service name (decision A2)."""

    n_boot = clustered.BOOT_DEFAULT if n_boot is None else int(n_boot)
    seed = clustered.BOOT_SEED_DEFAULT if seed is None else int(seed)
    need = {"condition", "sequence_id", "action_sequence"}
    if not need <= set(df.columns):
        return {
            "computed": False,
            "note": (
                f"needs {sorted(need)}; the dataset lacks {sorted(need - set(df.columns))} "
                "(runs before the action-trace instrumentation)"
            ),
        }
    genuine, decoys = _names()
    by_cond: dict[str, list[dict]] = collections.defaultdict(list)
    unreadable: collections.Counter = collections.Counter()
    for row in df.to_dict("records"):
        cond = str(row.get("condition"))
        rec = _attempt(row, genuine, decoys)
        if rec is None:
            unreadable[cond] += 1
        else:
            by_cond[cond].append(rec)
    if not by_cond:
        return {
            "computed": False,
            "n_unreadable": int(sum(unreadable.values())),
            "note": "no row carries a decodable action_sequence (the trace was not recorded)",
        }
    out = {
        cond: {**_block(by_cond.get(cond, []), n_boot, seed), "n_unreadable": int(unreadable[cond])}
        for cond in sorted(set(by_cond) | set(unreadable))
    }
    every = [a for atts in by_cond.values() for a in atts]
    with_exploit = [a for a in every if a["aimed"]]
    return {
        "computed": True,
        "rule": "by service name (decision A2, 2026-10-31)",
        "genuine_names": sorted(genuine),
        "decoy_names": sorted(decoys),
        "by_condition": out,
        "all_conditions": {
            "n_attempts": len(every),
            "n_unreadable": int(sum(unreadable.values())),
            "n_attempts_with_exploit": len(with_exploit),
            "n_first_exploit_at_a_decoy": int(sum(a["first"] == DECOY for a in with_exploit)),
            "n_exploits": int(sum(len(a["aimed"]) for a in every)),
            "n_exploits_at_a_decoy": int(sum(c == DECOY for a in every for c in a["aimed"])),
            "n_exploits_other": int(sum(c == OTHER for a in every for c in a["aimed"])),
        },
        "timing_cross_check": {
            "n_exploits_checked": int(sum(a["checked"] for a in every)),
            "n_disagreements": int(sum(a["disagree"] for a in every)),
        },
        "n_boot": int(n_boot),
        "bootstrap_seed": int(seed),
        "note": _NOTE,
    }
