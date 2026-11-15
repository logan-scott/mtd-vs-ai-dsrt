"""Real action timing and the attacker clock (PURE INSTRUMENTATION; never a dependent variable).

**Two clocks.** The planner stamps every action on an *attempt clock*: seconds since the trial
window opened. The MTD schedule it races is the same modeled one under both clocks
(``mtd.orchestrator``, Method C: ticks at ``start + phase + k*cadence``), and an exploit's
target is stale when ``ground_truth_at(<the exploit's stamp>)`` no longer contains it. The two
clocks differ ONLY in how much attempt time each action is charged:

``modeled`` (the default; the clock of every run archived under ``docs/results/``)
    A cursor advanced by fixed quanta from ``attacker/planner.py`` -- the recon command at
    ``cursor + 5`` s, the recon snapshot at ``cursor + 30`` s, an exploit at ``cursor + 10`` s
    -- whatever the real elapsed time. TTC/TTFVE therefore move in 5/10/30-s quanta and the
    first exploit of an attempt lands 10 s after its recon snapshot, always.

``wall``
    The real elapsed time on the control host's monotonic clock since the attempt began. A
    recon's command is stamped when the scan is issued and its snapshot when the scan RETURNS
    (when the attacker has the results); the snapshot's identities are labeled at the instant
    the scan actually PROBED the targets host (format version 2; see below), which the live
    courier reads from nmap's per-host XML start/end times. When the executor reports no probe
    instant (the offline executors) the identities are labeled at the scan's issue. The snapshot
    is scored (enumeration, recon accuracy) against the configuration it reflects
    (``ReconSnapshot.observed_at``), exactly as a modeled snapshot is, and enumeration is timed
    to its stamp. An exploit is stamped when it is FIRED (the instant the executor is invoked --
    the executor must know whether the target is still in force before it delivers). From format
    version 3 (2026-10-31, decision A1) the exploit is SCORED when its delivery COMPLETES: it
    counts only if its target is still in force at that instant, and it is stamped then, so
    time-to-compromise and time-to-first-valid-exploit end at completion. ``stale_at_completion``
    records whether the target had moved (the attacker was shown ``unreachable``);
    ``completed_after_window`` whether the delivery completed after the trial window (it counted
    for nothing and its result was never seen); ``executor_reached`` / ``executor_access`` keep
    the executor's own verdict, which the scoring may override.

    Format version 1 (the 2026-10-26 clock-validation pilot) labeled the identities at the
    scan's RETURN instead. A timing check on 2026-10-27 found nmap probes the targets host in
    the first second of a ~27-s subnet sweep, so return-labeling made the attacker's
    intelligence look ~26 s fresher than it was (``docs/clock_validation_runbook.md`` §5).

**The paired counterfactual.** With timing recorded, every exploit carries the staleness of its
target under THREE conventions, computed for the SAME delivery whichever clock drove the
attempt, so the clocks are compared on identical action sequences:

========================  ==========================================  ==========================
convention                intel instant (when the target was seen)    fire instant
========================  ==========================================  ==========================
``modeled``               the modeled snapshot of the latest recon     the modeled exploit stamp
``wall``                  the real instant that recon's scan RETURNED  the real firing instant
``wall_scan_start``       the real instant that recon's scan was       the real firing instant
                          ISSUED
``wall_observed``         the real instant the scan PROBED the targets the real firing instant
                          host (v2; the issue instant when unreported)
========================  ==========================================  ==========================

``wall`` and ``wall_scan_start`` bracket the real age of the attacker's intelligence: an nmap
scan observes each port at some moment between its issue and its return. ``wall_observed``
(format version 2) is that moment, as the scanner reports it, and it is the convention the
``wall`` clock now drives. For a target taken
from the attempt's latest snapshot ("snapshot-sourced" -- the planner only ever offers the
model the latest snapshot), a convention calls the delivery stale when the service's
observable identity at the intel instant differs from its identity at the fire instant; a
target the snapshot did not contain (retained cross-attempt knowledge) is stale when the tuple
is not in force at the fire instant. Under the clock that actually drove the attempt this is
exactly the executor's own staleness decision (``LiveTargetExecutor.exploit`` /
``OfflineToolExecutor.exploit``); the other conventions are counterfactuals on the same
sequence.

**What is recorded.** One :class:`ActionTiming` per planner decision -- including a stop and an
action the duration guard skipped -- carrying the real offset and duration of the planner call
(model latency, including any API retries) and of the executor call (the nmap scan or the
delivery), the action's effect instant on both clocks, and the exploit counterfactuals above.
:func:`timing_columns` derives the per-trial summary columns from it, and
:func:`encode_attempt_timing` serializes the whole record into the ``action_timing`` cell, from
which every summary can be recomputed (:func:`decode_action_timing`).

**Why it is off by default under the modeled clock.** Real durations differ run to run, and the
offline pipeline is contracted to be byte-deterministic given the seed. So the modeled clock
records timing only when ``attacker.record_timing`` is set (recommended for every live run);
the wall clock always records it, because under it the timing IS the clock. Nothing in this
module enters ``compute_trial``'s metrics: it fills instrumentation columns only.

**Layering.** Pure and stdlib-only, like the rest of ``measurement``: the attacker layer builds
these records (it owns the executor and the environment), this module only stores, summarizes
and (de)serializes them, and the analysis layer (``analysis.clock_validation``) reads them back.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from datetime import datetime, timezone

from measurement.models import CLOCK_MODELED, CLOCK_MODES, CLOCK_WALL, NOT_RECORDED

__all__ = [
    "CLOCK_MODELED",
    "CLOCK_MODES",
    "CLOCK_WALL",
    "KIND_EXPLOIT",
    "KIND_RECON",
    "SUPPORTED_TIMING_VERSIONS",
    "TIMING_COLUMNS",
    "TIMING_FORMAT_VERSION",
    "ActionTiming",
    "AttemptTiming",
    "decode_action_timing",
    "encode_attempt_timing",
    "timing_columns",
    "utc_iso",
]

#: Version of the ``action_timing`` JSON document (the ``"v"`` key). Bump on any change to the
#: key set or a key's meaning; :func:`decode_action_timing` refuses versions it does not know.
#: Version 2 adds the probe-time keys (``oo``, ``ao``, ``go``, ``so``); a version-1 document
#: (the 2026-10-26 pilot) carries none of them and still decodes. Version 3 (2026-10-31) adds
#: ``sc``, ``cw``, ``er`` and ``ea`` (under the wall clock an exploit is scored when its delivery
#: completes); version-1 and -2 documents (the October 2026 runs) carry none and still decode.
TIMING_FORMAT_VERSION = 3
SUPPORTED_TIMING_VERSIONS: tuple[int, ...] = (1, 2, 3)

#: Real offsets and durations are rounded to the millisecond. Finer resolution would only
#: record scheduler noise, and a fixed resolution keeps the stamps and the JSON consistent.
RESOLUTION_DECIMALS = 3

# Action kinds (values mirror ``attacker.base``; pinned by test).
KIND_RECON = "recon"
KIND_EXPLOIT = "exploit"

#: The ``TrialRow`` columns :func:`timing_columns` fills, in canonical (tail) order.
TIMING_COLUMNS: tuple[str, ...] = (
    "clock_mode",
    "wall_start_utc",
    "attempt_wall_s",
    "llm_latency_mean_s",
    "llm_latency_max_s",
    "scan_duration_mean_s",
    "first_exploit_gap_modeled_s",
    "first_exploit_gap_wall_s",
    "n_deliveries_clock_disagree",
    "first_delivery_clock_disagree",
    "action_timing",
)


def round_s(seconds: float) -> float:
    """Round a real duration/offset to the recording resolution (milliseconds)."""

    return round(float(seconds), RESOLUTION_DECIMALS)


def utc_iso(epoch_s: float) -> str:
    """Render a POSIX timestamp as ISO-8601 UTC with millisecond precision and a ``Z`` suffix.

    ``timezone.utc``, not ``datetime.UTC`` (3.11+): this module is on the 3.10 collection path.
    """

    return (
        datetime.fromtimestamp(float(epoch_s), timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


@dataclass(frozen=True)
class ActionTiming:
    """Real timing of one planner decision, its dispatch, and the clock counterfactuals.

    All ``*_s`` offsets are seconds since the attempt began (the real monotonic clock for
    ``decide_*``/``exec_*``/``wall_at_s``; the modeled cursor for ``modeled_at_s``). Fields that
    do not apply to the action are None: ``exec_*`` for an action that was never dispatched (a
    stop, an unrecognized action, one the duration guard skipped), the exploit fields for every
    non-exploit, ``relabeled`` for every non-recon.
    """

    ordinal: int
    kind: str
    decide_start_s: float
    decide_s: float
    # App-level API retries inside this planner call, when the client reports them (the SDK's
    # own retries are invisible and show up only as latency). None == not reported.
    api_retries: int | None = None
    exec_start_s: float | None = None
    exec_s: float | None = None
    # The action's EFFECT instant on each clock: a recon's snapshot, an exploit's firing.
    modeled_at_s: float | None = None
    wall_at_s: float | None = None
    # Recon only: True when an MTD tick changed a discovered identity while the real scan was
    # running, i.e. the snapshot's labels at the scan's issue and at its return differ.
    relabeled: bool | None = None
    # Recon only (v2): seconds after the scan's issue at which it probed the targets host, as
    # the scanner reported it (None when the executor reports none -- the offline executors).
    observed_offset_s: float | None = None
    # Recon only (v2): the real instant the snapshot's identities are dated to -- the issue
    # instant plus ``observed_offset_s`` (bounded by the scan's return), or the issue instant
    # when no offset was reported.
    observed_at_s: float | None = None
    # Exploit only.
    service_genuine: bool | None = None
    snapshot_sourced: bool | None = None
    gap_modeled_s: float | None = None
    gap_wall_s: float | None = None
    gap_wall_scan_start_s: float | None = None
    stale_modeled: bool | None = None
    stale_wall: bool | None = None
    stale_wall_scan_start: bool | None = None
    # Exploit only (v2): the same delivery under the ``wall_observed`` convention.
    gap_wall_observed_s: float | None = None
    stale_wall_observed: bool | None = None
    # Exploit only (v3), wall clock only; None under the modeled clock. The wall clock scores a
    # delivery when it COMPLETES (2026-10-31): it counts only if neither flag below is True.
    # ``stale_at_completion``: the target was not in force when the delivery was fired, or (for
    # a delivery that completed inside the window) when it completed. The delivery counted for
    # nothing and the attacker was shown ``unreachable``.
    stale_at_completion: bool | None = None
    # ``completed_after_window``: the delivery completed after the trial window closed. It
    # counted for nothing, the attacker never saw its result, and the attempt ended on it.
    completed_after_window: bool | None = None
    # The executor's own verdict for the delivery (reached and executed; access gained), before
    # the scoring at completion above may override it: kept so that an overtaken delivery's
    # real result on the testbed is not lost. The completion instant is
    # ``exec_start_s + exec_s``; ``wall_at_s`` stays the firing instant, the one the paired
    # counterfactual conventions use.
    executor_reached: bool | None = None
    executor_access: bool | None = None

    @property
    def executed(self) -> bool:
        """True iff the action was dispatched to the target executor."""

        return self.exec_start_s is not None


@dataclass(frozen=True)
class AttemptTiming:
    """The real timing of one attempt: its clock, real start, real length and decisions."""

    clock_mode: str
    wall_start_utc: str
    attempt_wall_s: float
    actions: tuple[ActionTiming, ...] = ()

    def __post_init__(self) -> None:
        if self.clock_mode not in CLOCK_MODES:
            raise ValueError(
                f"AttemptTiming.clock_mode must be one of {CLOCK_MODES}, got {self.clock_mode!r}"
            )


# --------------------------------------------------------------------------- #
# The action_timing cell (versioned JSON)
# --------------------------------------------------------------------------- #

# ActionTiming field -> JSON key. Short keys keep the cell compact (one document per row);
# the meaning of each key is documented in docs/metrics_specification.md §4f.
_KEYS: tuple[tuple[str, str], ...] = (
    ("ordinal", "i"),
    ("kind", "k"),
    ("decide_start_s", "d0"),
    ("decide_s", "d"),
    ("api_retries", "rt"),
    ("exec_start_s", "x0"),
    ("exec_s", "x"),
    ("modeled_at_s", "am"),
    ("wall_at_s", "aw"),
    ("relabeled", "rl"),
    ("observed_offset_s", "oo"),
    ("observed_at_s", "ao"),
    ("service_genuine", "g"),
    ("snapshot_sourced", "src"),
    ("gap_modeled_s", "gm"),
    ("gap_wall_s", "gw"),
    ("gap_wall_scan_start_s", "gs"),
    ("stale_modeled", "sm"),
    ("stale_wall", "sw"),
    ("stale_wall_scan_start", "ss"),
    ("gap_wall_observed_s", "go"),
    ("stale_wall_observed", "so"),
    ("stale_at_completion", "sc"),
    ("completed_after_window", "cw"),
    ("executor_reached", "er"),
    ("executor_access", "ea"),
)
_BOOL_FIELDS = frozenset(
    {
        "relabeled",
        "service_genuine",
        "snapshot_sourced",
        "stale_modeled",
        "stale_wall",
        "stale_wall_scan_start",
        "stale_wall_observed",
        "stale_at_completion",
        "completed_after_window",
        "executor_reached",
        "executor_access",
    }
)
_INT_FIELDS = frozenset({"ordinal", "api_retries"})
_STR_FIELDS = frozenset({"kind"})
_FIELD_BY_KEY = {key: name for name, key in _KEYS}


def _key_map_covers_every_field() -> bool:
    """True iff every ``ActionTiming`` field has exactly one JSON key (pinned by test)."""

    names = [name for name, _ in _KEYS]
    keys = [key for _, key in _KEYS]
    return sorted(names) == sorted(f.name for f in fields(ActionTiming)) and len(set(keys)) == len(
        keys
    )


def _encode_value(name: str, value: object) -> object:
    if name in _BOOL_FIELDS:
        return int(bool(value))
    if name in _INT_FIELDS or name in _STR_FIELDS:
        return value
    return round_s(value)  # a float offset/duration


def _action_to_obj(action: ActionTiming) -> dict:
    obj: dict = {}
    for name, key in _KEYS:
        value = getattr(action, name)
        if value is not None:
            obj[key] = _encode_value(name, value)
    return obj


def encode_attempt_timing(timing: AttemptTiming) -> str:
    """Serialize an attempt's decisions into the ``action_timing`` cell (compact JSON)."""

    doc = {
        "v": TIMING_FORMAT_VERSION,
        "actions": [_action_to_obj(a) for a in timing.actions],
    }
    return json.dumps(doc, separators=(",", ":"))


def _is_absent(cell: object) -> bool:
    if cell is None:
        return True
    if isinstance(cell, float) and math.isnan(cell):  # pandas renders an empty cell as NaN
        return True
    return isinstance(cell, str) and cell.strip() in ("", NOT_RECORDED)


def decode_action_timing(cell: object) -> tuple[ActionTiming, ...] | None:
    """Parse an ``action_timing`` cell back into :class:`ActionTiming` records.

    Returns None when the cell records no timing (``NOT_RECORDED``, empty, or NaN as pandas
    renders an empty cell). Raises ``ValueError`` on malformed JSON or an unknown version, so a
    corrupted cell can never be read as "no timing".
    """

    if _is_absent(cell):
        return None
    try:
        doc = json.loads(str(cell))
    except json.JSONDecodeError as exc:
        raise ValueError(f"action_timing is not valid JSON: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("v") not in SUPPORTED_TIMING_VERSIONS:
        version = doc.get("v") if isinstance(doc, dict) else None
        raise ValueError(
            f"action_timing version {version!r} is not supported "
            f"(this reader knows versions {SUPPORTED_TIMING_VERSIONS})"
        )
    v2_only = {"oo", "ao", "go", "so"}
    v3_only = {"sc", "cw", "er", "ea"}
    out: list[ActionTiming] = []
    for obj in doc.get("actions", []):
        kwargs: dict = {}
        for key, value in obj.items():
            name = _FIELD_BY_KEY.get(key)
            if (
                name is None
                or (doc["v"] < 2 and key in v2_only)
                or (doc["v"] < 3 and key in v3_only)
            ):
                raise ValueError(f"action_timing carries an unknown key {key!r}")
            if name in _BOOL_FIELDS:
                kwargs[name] = bool(value)
            elif name in _INT_FIELDS:
                kwargs[name] = int(value)
            elif name in _STR_FIELDS:
                kwargs[name] = str(value)
            else:
                kwargs[name] = float(value)
        out.append(ActionTiming(**kwargs))
    return tuple(out)


# --------------------------------------------------------------------------- #
# Per-trial summary columns
# --------------------------------------------------------------------------- #


def _mean(values: list[float]) -> float | None:
    return round_s(sum(values) / len(values)) if values else None


def timing_columns(timing: AttemptTiming | None, clock_mode: str) -> dict[str, object]:
    """The per-trial timing columns (:data:`TIMING_COLUMNS`) for one attempt.

    ``clock_mode`` is the clock the harness ran the attempt under; it is always recorded. With
    no timing record (``timing is None``) every other column takes its not-recorded sentinel --
    ``NOT_RECORDED`` for the two strings, null for the numbers -- never a fabricated 0, because
    "no exploit was delivered" (0 disagreements) and "nothing was measured" are different facts.
    """

    if clock_mode not in CLOCK_MODES:
        raise ValueError(f"clock_mode must be one of {CLOCK_MODES}, got {clock_mode!r}")
    if timing is None:
        return {
            "clock_mode": clock_mode,
            "wall_start_utc": NOT_RECORDED,
            "attempt_wall_s": None,
            "llm_latency_mean_s": None,
            "llm_latency_max_s": None,
            "scan_duration_mean_s": None,
            "first_exploit_gap_modeled_s": None,
            "first_exploit_gap_wall_s": None,
            "n_deliveries_clock_disagree": None,
            "first_delivery_clock_disagree": None,
            "action_timing": NOT_RECORDED,
        }
    if timing.clock_mode != clock_mode:
        raise ValueError(
            f"timing was recorded under the {timing.clock_mode!r} clock but the trial ran "
            f"under {clock_mode!r}"
        )

    latencies = [a.decide_s for a in timing.actions]
    scans = [a.exec_s for a in timing.actions if a.kind == KIND_RECON and a.exec_s is not None]
    exploits = [a for a in timing.actions if a.kind == KIND_EXPLOIT and a.executed]

    def _disagree(a: ActionTiming) -> bool | None:
        if a.stale_modeled is None or a.stale_wall is None:
            return None
        return a.stale_modeled != a.stale_wall

    flags = [_disagree(a) for a in exploits]
    first = exploits[0] if exploits else None
    first_flag = flags[0] if flags else None
    return {
        "clock_mode": clock_mode,
        "wall_start_utc": timing.wall_start_utc,
        "attempt_wall_s": round_s(timing.attempt_wall_s),
        "llm_latency_mean_s": _mean(latencies),
        "llm_latency_max_s": round_s(max(latencies)) if latencies else None,
        "scan_duration_mean_s": _mean(scans),
        "first_exploit_gap_modeled_s": first.gap_modeled_s if first else None,
        "first_exploit_gap_wall_s": first.gap_wall_s if first else None,
        "n_deliveries_clock_disagree": sum(1 for f in flags if f),
        "first_delivery_clock_disagree": None if first_flag is None else int(first_flag),
        "action_timing": encode_attempt_timing(timing),
    }
