"""LLM-driven attacker: a tool-use planning loop behind the Attacker interface.

The attacker executes the fixed offensive sequence per attempt — reconnaissance ->
vulnerability inference -> exploit selection -> delivery + verification — as a loop
in which a pluggable ``LLMClient`` chooses the next action and the offline tool
executor carries it out against the mock environment. Recon commands, recon
snapshots, and exploit deliveries are recorded as the loop runs, so the measurement
instrument scores an LLM attempt exactly as it scores the mock. What the attacker observed
in the earlier attempts of its sequence (``AttackerContext.history``) is passed to the client
on every planner call, so the model sees it (cross-attempt memory; ``attacker.base``).

The planning is model-driven; the *actions* never touch a live network — they are
resolved against the ground-truth inventory (``attacker.tools``).

Under the wall clock an exploit is scored when its delivery COMPLETES (decision A1,
2026-10-31): it counts only if its target is still in force then, and it is stamped then, so
time-to-compromise ends when the compromising exploit completes.

**The attempt clock.** Every action is stamped on an attempt clock (seconds since the trial
window opened), and the modeled MTD schedule decides at that stamp whether an exploit's target
is still in force. ``clock_mode="modeled"`` (the default, and the clock of every archived run)
charges fixed quanta -- ``_RECON_CMD_OFFSET_S`` / ``_RECON_DURATION_S`` / ``_EXPLOIT_DURATION_S``
-- whatever the real elapsed time; ``clock_mode="wall"`` charges the real elapsed time of each
planner call, scan and delivery, and dates what a scan saw to the instant it probed the targets
host (``executor.last_observed_offset_s``; the scan's issue when the executor reports none). With ``record_timing`` (forced on under ``wall``) the attempt
also returns its real action timing and, for every exploit, the staleness of the same delivery
under both clocks -- see ``measurement.timing``. With timing off under the modeled clock the
observation is exactly what it always was.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Protocol, runtime_checkable

from attacker.base import (
    CLOCK_MODELED,
    CLOCK_WALL,
    DISPOSITION_EXECUTED,
    DISPOSITION_HONORED,
    DISPOSITION_SKIPPED_DURATION_LIMIT,
    DISPOSITION_SKIPPED_UNRECOGNIZED,
    KIND_EXPLOIT,
    KIND_RECON,
    KIND_STOP,
    KIND_UNKNOWN,
    NOTE_NONE,
    TERMINAL_ACTION_BUDGET_EXHAUSTED,
    TERMINAL_DURATION_LIMIT,
    TERMINAL_NO_ACTION_BUDGET,
    TERMINAL_PLANNER_STOP,
    ActionTraceEntry,
    AttemptMemory,
    AttemptObservation,
    AttemptRequest,
    Environment,
    normalize_clock_mode,
)
from attacker.tools import (
    ExploitOutcome,
    OfflineToolExecutor,
    TargetExecutor,
    identity_at,
    relabel_at,
    service_is_genuine,
)
from measurement import clock
from measurement.models import ExploitDelivery, ReconCommand, ReconSnapshot, ServiceTuple
from measurement.timing import ActionTiming, AttemptTiming, round_s, utc_iso


def _observed_offset(executor: object) -> float | None:
    """Seconds after its issue at which the last scan probed the targets host, if reported.

    The live courier reads it from nmap's per-host XML times
    (``harness.live_targets.SubprocessTransport``); the offline executors report none.
    """

    value = getattr(executor, "last_observed_offset_s", None)
    if value is None:
        return None
    try:
        offset = float(value)
    except (TypeError, ValueError):
        return None
    return offset if offset == offset and offset >= 0.0 else None  # NaN / negative -> unknown


def _observation_instant(issue_s: float, return_s: float, offset_s: float | None) -> float:
    """The real instant a scan's observations are dated to (wall clock, format v2).

    The issue instant plus the reported probe offset, never later than the scan's return; the
    issue instant itself when no offset was reported (the measured probe lands within a second
    of the issue -- ``docs/clock_validation_runbook.md`` §5).
    """

    if offset_s is None:
        return round_s(issue_s)
    return round_s(min(issue_s + offset_s, max(return_s, issue_s)))


_RECON_CMD_OFFSET_S = 5
_RECON_DURATION_S = 30
_EXPLOIT_DURATION_S = 10


# --------------------------------------------------------------------------- #
# Actions the planner can choose
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ReconAction:
    pass


@dataclass(frozen=True)
class ExploitAction:
    target: ServiceTuple


@dataclass(frozen=True)
class StopAction:
    pass


Action = ReconAction | ExploitAction | StopAction


def _action_kind(action: Action) -> str:
    """Classify a chosen action for the trace (bounded vocabulary)."""

    if isinstance(action, ReconAction):
        return KIND_RECON
    if isinstance(action, ExploitAction):
        return KIND_EXPLOIT
    if isinstance(action, StopAction):
        return KIND_STOP
    return KIND_UNKNOWN


# Disposition an entry carries the moment it is appended, before dispatch. Recon and
# exploit start pessimistic: the ONLY path from here that does not execute them is the
# duration guard, and the branch overwrites this the instant that guard admits them.
# Stop and unrecognized are terminal on append — neither reaches an executor.
_PENDING_DISPOSITION = {
    KIND_RECON: DISPOSITION_SKIPPED_DURATION_LIMIT,
    KIND_EXPLOIT: DISPOSITION_SKIPPED_DURATION_LIMIT,
    KIND_STOP: DISPOSITION_HONORED,
    KIND_UNKNOWN: DISPOSITION_SKIPPED_UNRECOGNIZED,
}


@dataclass
class PlannerState:
    """The attacker's current view within one attempt (given to the LLM client)."""

    attempt_index: int
    reconned: bool
    discovered: frozenset[ServiceTuple]
    attempted: frozenset[ServiceTuple]
    outcomes: list[tuple[ServiceTuple, ExploitOutcome]]
    known_genuine: frozenset[ServiceTuple]
    known_decoys: frozenset[ServiceTuple]
    compromised_genuine: bool
    actions_used: int
    max_actions: int
    # What the attacker observed in the EARLIER attempts of its sequence (cross-attempt
    # memory; ``attacker.base.AttemptMemory``), oldest first. Empty on a sequence's first
    # attempt and whenever adaptation is disabled (repeatability runs).
    memory: tuple[AttemptMemory, ...] = ()


@runtime_checkable
class LLMClient(Protocol):
    """Chooses the next action given the planner state and system prompt."""

    def next_action(self, state: PlannerState, system_prompt: str) -> Action: ...


@dataclass(frozen=True)
class _Intel:
    """When the attempt's latest snapshot saw a service, on each clock (seconds into the window).

    ``modeled_s`` is the modeled snapshot instant; ``wall_s`` / ``wall_scan_start_s`` are the
    real instants the scan returned / was issued; ``wall_observed_s`` is the real instant it
    probed the targets host (the issue instant when unreported). Kept per service name for the
    latest snapshot only, because the planner only ever offers the model the latest snapshot.
    """

    modeled_s: float
    wall_s: float
    wall_scan_start_s: float
    wall_observed_s: float


def _api_retries(client: object) -> int | None:
    """App-level API retries inside the last planner call, if the client reports them."""

    value = getattr(client, "last_call_retries", None)
    return int(value) if value is not None else None


def _exploit_counterfactual(
    env: Environment,
    window_start: str,
    target: ServiceTuple,
    latest: frozenset[ServiceTuple],
    intel: dict[str, _Intel],
    modeled_fire_s: float,
    wall_fire_s: float,
) -> dict:
    """The staleness of ONE delivery under the three clock conventions (``measurement.timing``).

    A snapshot-sourced target is stale under a convention when the service's identity at that
    convention's intel instant differs from its identity at the fire instant -- under the clock
    that drove the attempt this is exactly the executor's own decision (the target IS the
    identity at the intel instant). A target the latest snapshot did not contain (retained
    knowledge) is stale when the tuple itself is not in force at the fire instant.
    """

    def at(offset_s: float) -> str:
        return clock.shift(window_start, offset_s)

    name = target.service
    genuine = service_is_genuine(env, name, at(wall_fire_s))
    src = intel.get(name) if target in latest else None
    if src is not None:

        def stale(intel_s: float, fire_s: float) -> bool:
            return identity_at(env, name, at(intel_s)) != identity_at(env, name, at(fire_s))

        return {
            "service_genuine": genuine,
            "snapshot_sourced": True,
            "gap_modeled_s": round_s(modeled_fire_s - src.modeled_s),
            "gap_wall_s": round_s(wall_fire_s - src.wall_s),
            "gap_wall_scan_start_s": round_s(wall_fire_s - src.wall_scan_start_s),
            "gap_wall_observed_s": round_s(wall_fire_s - src.wall_observed_s),
            "stale_modeled": stale(src.modeled_s, modeled_fire_s),
            "stale_wall": stale(src.wall_s, wall_fire_s),
            "stale_wall_scan_start": stale(src.wall_scan_start_s, wall_fire_s),
            "stale_wall_observed": stale(src.wall_observed_s, wall_fire_s),
        }

    def gone(fire_s: float) -> bool:
        gt = env.ground_truth_at(at(fire_s))
        return target not in gt.genuine and target not in gt.decoys

    stale_wall = gone(wall_fire_s)
    return {
        "service_genuine": genuine,
        "snapshot_sourced": False,
        "stale_modeled": gone(modeled_fire_s),
        "stale_wall": stale_wall,
        # No intel instant to vary for a target the snapshot did not contain.
        "stale_wall_scan_start": stale_wall,
        "stale_wall_observed": stale_wall,
    }


def _in_force(env: Environment, target: ServiceTuple, timestamp: str) -> bool:
    """Whether ``target`` (its full observable identity) is in force at ``timestamp``."""

    gt = env.ground_truth_at(timestamp)
    return target in gt.genuine or target in gt.decoys


@dataclass
class LLMAttacker:
    """Attacker adapter driven by a pluggable LLM client.

    ``executor_factory`` builds the ``TargetExecutor`` that carries out the chosen
    actions against ``req.environment``. It defaults to ``OfflineToolExecutor`` (the
    mock target), so offline behavior is unchanged; the live harness swaps in a
    factory that returns the environment's live executor (see ``harness.live_runner``).

    ``clock_mode`` selects the attempt clock (see the module docstring and
    ``measurement.timing``); ``record_timing`` returns the attempt's real action timing under
    the modeled clock too (it is always on under the wall clock). ``time_source`` (monotonic
    seconds) and ``wall_time_source`` (POSIX seconds, for the recorded real start) are
    injectable so the wall clock is unit-tested deterministically without real waits.
    """

    client: LLMClient
    model_id: str = "claude-llm"
    model_version: str = "unpinned"
    system_prompt: str = "You are an autonomous offensive workflow in an isolated lab."
    max_stops: int = field(default=0, repr=False)  # reserved
    executor_factory: Callable[[Environment], TargetExecutor] = OfflineToolExecutor
    clock_mode: str = CLOCK_MODELED
    record_timing: bool = False
    time_source: Callable[[], float] = field(default=time.monotonic, repr=False, compare=False)
    wall_time_source: Callable[[], float] = field(default=time.time, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Fail at construction, not mid-run, on a clock this planner does not implement.
        self.clock_mode = normalize_clock_mode(self.clock_mode)

    def attempt(self, req: AttemptRequest) -> AttemptObservation:
        mode = normalize_clock_mode(self.clock_mode)
        wall = mode == CLOCK_WALL
        # Under the wall clock the timing IS the clock, so it is always recorded. Under the
        # modeled clock the real clock is not even read unless asked for, so an untimed
        # attempt is exactly what it always was (and stays deterministic).
        timed = wall or self.record_timing
        executor = self.executor_factory(req.environment)
        env = req.environment
        ctx = req.context
        limit = req.trial_duration_limit_s

        t0 = self.time_source() if timed else 0.0
        wall_start_utc = utc_iso(self.wall_time_source()) if timed else ""

        def now() -> float:
            """Real seconds since the attempt began (ms resolution)."""

            return round_s(self.time_source() - t0)

        def at(offset_s: float) -> str:
            """The attempt-clock stamp ``offset_s`` seconds into the trial window."""

            return clock.shift(req.window_start, offset_s)

        recon_commands: list[ReconCommand] = []
        recon_snapshots: list[ReconSnapshot] = []
        deliveries: list[ExploitDelivery] = []
        outcomes: list[tuple[ServiceTuple, ExploitOutcome]] = []

        discovered: frozenset[ServiceTuple] = frozenset()
        attempted: set[ServiceTuple] = set()
        genuine_hits: set[ServiceTuple] = set()
        decoy_hits: set[ServiceTuple] = set()
        compromised = False

        # Instrumentation only. ``terminal_reason`` starts at the zero-iteration case
        # (``max_actions <= 0``: the loop body never runs and no planner call is made)
        # and is reset at the top of every iteration to the fall-through case, so a loop
        # that ends on the ``while`` condition is labeled without reconstruction after
        # the fact. Every ``break`` below sets its own reason.
        trace: list[ActionTraceEntry] = []
        terminal_reason = TERMINAL_NO_ACTION_BUDGET
        # Instrumentation only (timed attempts): one ActionTiming per planner call, and when
        # the latest snapshot saw each service, on both clocks.
        timings: list[ActionTiming] = []
        intel: dict[str, _Intel] = {}

        cursor = 0  # MODELED seconds into the trial window (advanced under both clocks)
        actions = 0
        while actions < req.max_actions:
            terminal_reason = TERMINAL_ACTION_BUDGET_EXHAUSTED
            state = PlannerState(
                attempt_index=req.attempt_index,
                reconned=bool(recon_snapshots),
                discovered=discovered,
                attempted=frozenset(attempted),
                outcomes=list(outcomes),
                known_genuine=ctx.known_genuine,
                known_decoys=ctx.known_decoys,
                compromised_genuine=compromised,
                actions_used=actions,
                max_actions=req.max_actions,
                memory=ctx.history,
            )
            decide_start = now() if timed else 0.0
            action = self.client.next_action(state, self.system_prompt)
            decide_end = now() if timed else 0.0

            # Record the decision BEFORE dispatch so that an action the guards reject
            # is still in the trace: every planner call appends exactly one entry, on
            # every path, and no chosen action can be silently dropped.
            kind = _action_kind(action)
            trace.append(
                ActionTraceEntry(
                    ordinal=actions,
                    kind=kind,
                    disposition=_PENDING_DISPOSITION[kind],
                    target=action.target if isinstance(action, ExploitAction) else None,
                    # Optional client signal: why THIS action, when the client
                    # substituted one for a malformed model response. Clients that
                    # do not publish it degrade to "" rather than raising.
                    note=str(getattr(self.client, "last_action_note", NOTE_NONE)),
                )
            )
            timing = (
                ActionTiming(
                    ordinal=actions,
                    kind=kind,
                    decide_start_s=decide_start,
                    decide_s=round_s(decide_end - decide_start),
                    api_retries=_api_retries(self.client),
                )
                if timed
                else None
            )

            if isinstance(action, StopAction):
                terminal_reason = TERMINAL_PLANNER_STOP
                if timing is not None:
                    timings.append(timing)
                break

            if isinstance(action, ReconAction):
                modeled_cmd = cursor + _RECON_CMD_OFFSET_S
                modeled_snap = cursor + _RECON_DURATION_S
                # ONE real-clock reading serves as both the duration guard and the issue stamp,
                # so a recon can never be admitted at one instant and stamped at a later one.
                issue = now() if timed else 0.0
                # Duration guard. The modeled clock knows the snapshot instant in advance and
                # refuses a recon that would complete past the limit; the wall clock cannot
                # know when a scan will return, so it admits a recon ISSUED within the limit
                # (and discards results that come back after it -- below).
                if (issue if wall else modeled_snap) > limit:
                    terminal_reason = TERMINAL_DURATION_LIMIT
                    if timing is not None:
                        timings.append(timing)
                    break
                trace[-1] = replace(trace[-1], disposition=DISPOSITION_EXECUTED)
                exec_start = issue
                # The executor labels what it finds with the identities in force at the instant
                # it is given. The modeled clock gives the (known) snapshot instant; the wall
                # clock does not know when the scan will see the target, so it scans at the
                # issue instant and relabels below at the instant the scan actually probed the
                # targets host (format v2; v1 relabeled at the scan's return).
                raw = executor.recon(at(exec_start if wall else modeled_snap))
                exec_end = now() if timed else 0.0
                observed_offset = _observed_offset(executor) if timed else None
                observed_at = (
                    _observation_instant(exec_start, exec_end, observed_offset) if timed else 0.0
                )
                if timing is not None:
                    timing = replace(
                        timing,
                        exec_start_s=exec_start,
                        exec_s=round_s(exec_end - exec_start),
                        modeled_at_s=float(modeled_snap),
                        wall_at_s=exec_end,
                        relabeled=relabel_at(raw, env, at(exec_start))
                        != relabel_at(raw, env, at(exec_end)),
                        observed_offset_s=(
                            None if observed_offset is None else round_s(observed_offset)
                        ),
                        observed_at_s=observed_at,
                    )
                cmd_at = float(exec_start) if wall else modeled_cmd
                recon_commands.append(ReconCommand(at(cmd_at), "scan --recon"))
                if wall and exec_end > limit:
                    # The scan was issued inside the window but its results came back after
                    # the window closed: the attacker never had them. The command stands (it
                    # was issued), no snapshot is recorded, nothing is learned, and the attempt
                    # ends here -- exactly what the modeled clock's pre-check achieves by
                    # refusing such a recon. No further (billed) planner call is made.
                    terminal_reason = TERMINAL_DURATION_LIMIT
                    if timing is not None:
                        timings.append(timing)
                    break
                if wall:
                    # The snapshot is STAMPED when the attacker has it (the scan's return: what
                    # time-to-enumeration keys on) and LABELED with the identities in force when
                    # the scan saw the target, so a tick during the rest of the scan leaves the
                    # attacker's picture stale for the exploits that use it, as it would be. It
                    # is SCORED (enumeration, recon accuracy) against the configuration it
                    # reflects -- ``observed_at`` -- exactly as a modeled snapshot is.
                    snap_at = float(exec_end)
                    discovered = relabel_at(raw, env, at(observed_at))
                    reflects = at(observed_at)
                else:
                    snap_at = modeled_snap
                    discovered = raw
                    reflects = None
                recon_snapshots.append(
                    ReconSnapshot(req.attempt_index, at(snap_at), discovered, reflects)
                )
                if timing is not None:
                    intel = {
                        t.service: _Intel(float(modeled_snap), exec_end, exec_start, observed_at)
                        for t in discovered
                    }
                cursor += _RECON_DURATION_S

            elif isinstance(action, ExploitAction):
                modeled_fire = cursor + _EXPLOIT_DURATION_S
                # ONE real-clock reading is both the guard and the firing stamp, so an exploit
                # admitted inside the window is never stamped past it.
                fire = now() if timed else 0.0
                if (fire if wall else modeled_fire) > limit:
                    terminal_reason = TERMINAL_DURATION_LIMIT
                    if timing is not None:
                        timings.append(timing)
                    break
                trace[-1] = replace(trace[-1], disposition=DISPOSITION_EXECUTED)
                exec_start = fire
                # The wall clock fires the exploit NOW: the executor must know whether the
                # target is still in force before it delivers anything, so it is given the
                # instant of firing.
                ts = at(exec_start if wall else modeled_fire)
                outcome = executor.exploit(ts, action.target)
                exec_end = now() if timed else 0.0
                stale_at_completion: bool | None = None
                late = False
                # The executor's own verdict, kept in the timing record (wall clock) because
                # scoring at completion may override it below.
                executor_verdict: dict = {}
                if wall:
                    # SCORED AT COMPLETION (decision A1, 2026-10-31). A delivery counts only if
                    # its target is still in force when the delivery COMPLETES, the moment the
                    # exploit has done its work. The Metasploit primitives run their modules in
                    # the resident msfd (a fresh msfconsole per delivery took 14-22 s, mostly
                    # Metasploit's own overhead, which then decided the race); the curl ones
                    # take about 1 s. If a tick moved the service in between, a real MTD would
                    # have moved it before the exploit arrived, so the delivery is scored -- and
                    # shown to the attacker -- as unreachable. The delivery is stamped at
                    # completion, so time-to-compromise ends when the compromising exploit
                    # completes. One that completes after the trial window closed never counts
                    # and its result is never seen, like a scan that returns after the window.
                    late = exec_end > limit
                    done = at(exec_end)
                    executor_verdict = {
                        "executor_reached": bool(outcome.reached_and_executed),
                        "executor_access": bool(outcome.unauthorized_access),
                    }
                    # Stale: not in force when fired (the executor delivered nothing), or moved
                    # before a delivery that completed inside the window completed. Lateness is
                    # its own verdict (``completed_after_window``): nothing need have moved.
                    stale_at_completion = not _in_force(env, action.target, ts) or (
                        not late and not _in_force(env, action.target, done)
                    )
                    if (stale_at_completion or late) and (
                        outcome.reached_and_executed or outcome.unauthorized_access
                    ):
                        outcome = ExploitOutcome(
                            reached_and_executed=False,
                            target_genuine=outcome.target_genuine,
                            unauthorized_access=False,
                        )
                    ts = done
                deliveries.append(
                    ExploitDelivery(
                        timestamp=ts,
                        target=action.target,
                        reached_and_executed=outcome.reached_and_executed,
                        target_genuine=outcome.target_genuine,
                        unauthorized_access=outcome.unauthorized_access,
                    )
                )
                outcomes.append((action.target, outcome))
                attempted.add(action.target)
                if outcome.unauthorized_access:
                    # What the prompt shows ("Genuine service compromised"): access gained, as
                    # the delivery reported it. Only a genuine service grants access (a decoy
                    # has no primitive, and the offline executors grant a decoy none), so this
                    # equals the labelled test below without reading the label.
                    compromised = True
                if outcome.target_genuine and outcome.unauthorized_access:
                    genuine_hits.add(action.target)
                elif outcome.reached_and_executed and not outcome.target_genuine:
                    decoy_hits.add(action.target)  # learned it is a decoy
                if timing is not None:
                    timing = replace(
                        timing,
                        exec_start_s=exec_start,
                        exec_s=round_s(exec_end - exec_start),
                        modeled_at_s=float(modeled_fire),
                        wall_at_s=exec_start,
                        **_exploit_counterfactual(
                            env,
                            req.window_start,
                            action.target,
                            discovered,
                            intel,
                            float(modeled_fire),
                            exec_start,
                        ),
                        stale_at_completion=stale_at_completion,
                        completed_after_window=late if wall else None,
                        **executor_verdict,
                    )
                cursor += _EXPLOIT_DURATION_S
                if late:
                    # The delivery completed after the window closed: it is recorded (it was
                    # fired) but counts for nothing, and the attempt ends here, with no further
                    # (billed) planner call -- as for a scan that returns after the window.
                    terminal_reason = TERMINAL_DURATION_LIMIT
                    if timing is not None:
                        timings.append(timing)
                    break

            if timing is not None:
                timings.append(timing)
            actions += 1

        attempt_timing = (
            AttemptTiming(
                clock_mode=mode,
                wall_start_utc=wall_start_utc,
                attempt_wall_s=now(),
                actions=tuple(timings),
            )
            if timed
            else None
        )
        return AttemptObservation(
            recon_commands=tuple(recon_commands),
            recon_snapshots=tuple(recon_snapshots),
            exploit_deliveries=tuple(deliveries),
            discovered=discovered,
            genuine_hits=frozenset(genuine_hits),
            decoy_hits=frozenset(decoy_hits),
            action_trace=tuple(trace),
            terminal_reason=terminal_reason,
            timing=attempt_timing,
        )
