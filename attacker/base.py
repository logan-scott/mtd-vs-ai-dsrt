"""Backend-agnostic attacker interface + retained context.

An attacker executes one *attempt* per call: it probes the ``Environment`` (in the
live system this is scanning/exploitation tooling; offline it is the mock's view
of ground truth), and returns the recon commands, recon snapshot, and exploit
deliveries the measurement instrument turns into a trial row. ``AttackerContext``
carries accumulated knowledge across the attempts of a sequence — the mechanism
of adaptation for RQ3. When adaptation is disabled (repeatability runs), the
context is not accumulated and attempts are independent.

**What reaches the model (cross-attempt memory, 2026-10-31).** ``AttackerContext.history``
holds one :class:`AttemptMemory` per completed attempt: what the attacker ITSELF observed
in it -- how many scans returned, what its last scan listed, and every exploit it sent with
its result. The planner passes it to the LLM client, which renders it into
every prompt of the later attempts (``attacker.backends``). Before this, no part of the
context reached the model: each attempt started with an empty listing and no record of
earlier outcomes, so the attacker could not learn across attempts (the 2026-10-30 finding).
Memory is built from observations only. ``known_genuine`` / ``known_decoys`` are derived
from the instrument's ground-truth labels (``ExploitOutcome.target_genuine``), which the
attacker cannot see, so they stay the instrument's record and are NEVER rendered.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol, runtime_checkable

from measurement.models import (
    CLOCK_MODELED,
    CLOCK_MODES,
    ExploitDelivery,
    GroundTruthInventory,
    ReconCommand,
    ReconSnapshot,
    ServiceTuple,
)
from measurement.models import CLOCK_WALL as CLOCK_WALL  # re-exported (see below)
from measurement.timing import AttemptTiming


@runtime_checkable
class Environment(Protocol):
    """What an attacker can probe. Offline: backed by the MTD orchestrator."""

    def ground_truth_at(self, timestamp: str) -> GroundTruthInventory:
        """The genuine/decoy inventory in force at ``timestamp``."""
        ...


#: The planner prompt's user-message format, recorded in each run's provenance
#: (``cli._collection_provenance`` -> ``attacker_prompt_format``) and, for the
#: ``anthropic`` backend, part of the run fingerprint (``harness.checkpoint``), so a run never
#: resumes across a change of format.
#:
#: 1 -- every archived run and the October 2026 re-run: the listing renders the four-field
#:      identity; the attempt's earlier exploit outcomes are keyed by service NAME (so a
#:      moved service kept its old outcome after a rescan); nothing from earlier attempts.
#: 2 -- 2026-10-31: earlier outcomes in the attempt are keyed by the FULL identity, and the
#:      attacker's own observations from the earlier attempts of its sequence are rendered
#:      (cross-attempt memory, :class:`AttemptMemory`). The system prompt is unchanged.
PROMPT_FORMAT_VERSION = 2


# The result of an exploit AS THE ATTACKER SEES IT. Derived only from what the delivery
# reported back to the attacker (``reached_and_executed`` / ``unauthorized_access``), never
# from the instrument's genuine/decoy label, which the attacker cannot observe. A decoy
# that answered reads "reached", exactly as it always has in the prompt.
OUTCOME_COMPROMISED = "compromised"
OUTCOME_REACHED = "reached"
OUTCOME_UNREACHABLE = "unreachable"
OBSERVED_OUTCOMES = (OUTCOME_COMPROMISED, OUTCOME_REACHED, OUTCOME_UNREACHABLE)


def observed_outcome(reached_and_executed: bool, unauthorized_access: bool) -> str:
    """The word the attacker is shown for one exploit result."""

    if unauthorized_access:
        return OUTCOME_COMPROMISED
    if reached_and_executed:
        return OUTCOME_REACHED
    return OUTCOME_UNREACHABLE


@dataclass(frozen=True)
class AttemptMemory:
    """What the attacker observed in one completed attempt of its sequence.

    Observations only: ``scans`` is how many scans returned results within the attempt,
    ``last_listing`` the services its last such scan listed (the identities it was shown,
    in listing order), and ``exploits`` every exploit it sent, in order, as
    ``(target as listed, observed outcome)``. Nothing here comes from the instrument's
    ground truth (in-force inventory, genuine/decoy labels).
    """

    attempt_index: int
    scans: int
    last_listing: tuple[ServiceTuple, ...]
    exploits: tuple[tuple[ServiceTuple, str], ...]


@dataclass(frozen=True)
class AttackerContext:
    """Accumulated cross-attempt knowledge (retained within a sequence).

    ``history`` is what reaches the model (see the module docstring); the other fields are
    the instrument's own record (``known_genuine`` / ``known_decoys`` use ground-truth
    labels and drive only the offline stub policy).
    """

    attempts_seen: int = 0
    last_discovered: frozenset[ServiceTuple] | None = None
    known_genuine: frozenset[ServiceTuple] = field(default_factory=frozenset)
    known_decoys: frozenset[ServiceTuple] = field(default_factory=frozenset)
    adaptation_enabled: bool = True
    history: tuple[AttemptMemory, ...] = ()

    def updated(
        self,
        discovered: frozenset[ServiceTuple],
        genuine_hits: frozenset[ServiceTuple],
        decoy_hits: frozenset[ServiceTuple],
        memory: AttemptMemory | None = None,
    ) -> AttackerContext:
        """Return the context after an attempt.

        ``memory`` is what the attacker observed in that attempt (:func:`attempt_memory`);
        it is appended to ``history`` and shown to the model on every later attempt.

        With adaptation disabled the context stays empty (attempts independent),
        which isolates measurement variance for the ICC repeatability runs: no memory is
        kept, so those prompts never carry any.
        """

        if not self.adaptation_enabled:
            return self
        return replace(
            self,
            attempts_seen=self.attempts_seen + 1,
            last_discovered=discovered,
            known_genuine=self.known_genuine | genuine_hits,
            known_decoys=self.known_decoys | decoy_hits,
            history=self.history + ((memory,) if memory is not None else ()),
        )


@dataclass(frozen=True)
class AttemptRequest:
    """Inputs for one attempt."""

    attempt_index: int
    window_start: str
    recon_ts: str
    environment: Environment
    context: AttackerContext
    seed: int
    max_actions: int
    trial_duration_limit_s: int


# --------------------------------------------------------------------------- #
# Action trace (pure instrumentation)
# --------------------------------------------------------------------------- #
#
# The instrument records the *consequences* of the planner's choices (recon
# snapshots, exploit deliveries) but not the choices themselves, so an attempt that
# never lands a valid exploit cannot be characterized except by inference from
# censoring. The action trace records the decision sequence as chosen. It is written
# by the attacker backends and consumed only by descriptive reporting: nothing here
# feeds ``measurement.compute_trial``, so no measured outcome depends on it.

# Action kinds. A bounded vocabulary: extending ``attacker.planner.Action`` requires
# extending this set and ``attacker.planner._action_kind``.
KIND_RECON = "recon"
KIND_EXPLOIT = "exploit"
KIND_STOP = "stop"
KIND_UNKNOWN = "unknown"  # an Action type the planner loop does not dispatch
ACTION_KINDS = frozenset({KIND_RECON, KIND_EXPLOIT, KIND_STOP, KIND_UNKNOWN})

# What became of a chosen action.
DISPOSITION_EXECUTED = "executed"  # admitted past the guards and dispatched
DISPOSITION_HONORED = "honored"  # a stop: no environment effect, loop ended
DISPOSITION_SKIPPED_DURATION_LIMIT = "skipped_duration_limit"  # chosen, never run
DISPOSITION_SKIPPED_UNRECOGNIZED = "skipped_unrecognized"  # fell through dispatch
DISPOSITIONS = frozenset(
    {
        DISPOSITION_EXECUTED,
        DISPOSITION_HONORED,
        DISPOSITION_SKIPPED_DURATION_LIMIT,
        DISPOSITION_SKIPPED_UNRECOGNIZED,
    }
)

# Why the CLIENT returned the action it returned, when the returned action is not the
# one the model asked for. ``AnthropicClient`` silently substitutes a stop for a
# malformed tool call, so without this a coerced stop is indistinguishable from a
# model that decided to stop (see ``attacker/backends.py``). "" == no substitution.
NOTE_NONE = ""
NOTE_COERCED_STOP_INDEX_OUT_OF_RANGE = "coerced_stop_index_out_of_range"
NOTE_COERCED_STOP_NO_TOOL_USE = "coerced_stop_no_tool_use"
NOTE_COERCED_RECON_NO_TOOL_USE = "coerced_recon_no_tool_use"
# Prompt format 2: the model asked for an exploit or a stop before the attempt's first scan.
# Each attempt runs a fixed workflow whose first step is a scan (system prompt, step 1), and
# before it there is no listing to exploit, so the scan is run instead. Without this, an
# exploit chosen from memory (index out of range) or a stop before any scan would end the
# attempt with no reconnaissance, which the harness excludes (``attacker_init_or_logging_
# incomplete``) -- losing the attempt. Under format 1 no attempt ever opened that way.
NOTE_COERCED_RECON_BEFORE_SCAN = "coerced_recon_before_scan"
CLIENT_NOTES = frozenset(
    {
        NOTE_NONE,
        NOTE_COERCED_STOP_INDEX_OUT_OF_RANGE,
        NOTE_COERCED_STOP_NO_TOOL_USE,
        NOTE_COERCED_RECON_NO_TOOL_USE,
        NOTE_COERCED_RECON_BEFORE_SCAN,
    }
)

# --------------------------------------------------------------------------- #
# Tool-choice request shaping (an experimental MANIPULATION, not an operational knob)
# --------------------------------------------------------------------------- #
#
# The Anthropic Messages API's ``tool_choice`` decides whether the model MAY answer in
# prose instead of calling a tool. Omitting the key is the API's documented default
# (equivalent to ``{"type": "auto"}``) and is what every run frozen under
# ``docs/results/`` did; ``{"type": "any"}`` forces some tool call. Because a response
# with no ``tool_use`` block is coerced into a stop by ``AnthropicClient.next_action``
# (see ``NOTE_COERCED_STOP_NO_TOOL_USE``), turning this on could change the measured
# behavior of the attacker -- so it is a manipulation whose effect must be MEASURED, not
# a fix to be applied. Hence three distinguishable states rather than a boolean:
#
#   None              -> the key is OMITTED from the payload entirely (the control, and
#                        the only setting that reproduces the frozen arms byte-for-byte)
#   TOOL_CHOICE_AUTO  -> ``{"type": "auto"}`` is SENT explicitly. Semantically the API
#                        default, but a different request -- available so "the key was
#                        absent" and "the key carried the default" can be separated,
#                        since only measurement can establish that they behave alike.
#   TOOL_CHOICE_ANY   -> ``{"type": "any"}`` is sent: the model must call some tool.
#
# The YAML surface is these exact lowercase strings (the API's own ``type`` vocabulary),
# NOT a bool: a bool cannot express three states, and it would have to spell the control
# condition ``false``, which reads like "auto" rather than "send nothing at all". Using
# the API's words also leaves room for ``{"type": "tool", ...}`` later without redesign.
# Values are matched EXACTLY and anything else raises (see ``normalize_tool_choice``):
# a typo must never degrade silently to the control condition, because the control
# condition is the thing under test.
TOOL_CHOICE_ANY = "any"
TOOL_CHOICE_AUTO = "auto"
TOOL_CHOICE_PAYLOADS: dict[str, dict[str, str]] = {
    TOOL_CHOICE_ANY: {"type": "any"},
    TOOL_CHOICE_AUTO: {"type": "auto"},
}
TOOL_CHOICE_SETTINGS = frozenset(TOOL_CHOICE_PAYLOADS)

#: Row value recorded when the request carried NO ``tool_choice`` key. A real recorded
#: observation ("we sent nothing"), which is why it is a word and not ``""``/null: the
#: dataset must keep it distinguishable from ``NOT_RECORDED`` ("this row predates the
#: column"). Mirrored as ``measurement.models.TOOL_CHOICE_OMITTED`` (pinned equal by
#: test) because ``measurement.models`` cannot import this module -- this one imports it.
TOOL_CHOICE_OMITTED = "omitted"


def normalize_tool_choice(raw: object) -> str | None:
    """Validate a configured ``tool_choice`` and return it, or ``None`` to omit the key.

    ``None`` (absent or explicitly nulled in YAML) passes through as ``None``. Every
    other value must be an exact member of :data:`TOOL_CHOICE_SETTINGS`; anything else
    -- a typo, a bool, a wrong-cased string -- raises ``ValueError`` at parse time.
    Failing loudly is the point: silently omitting an unrecognized value would produce
    the CONTROL condition under a config that asked for the treatment, which is exactly
    the confusion this setting exists to resolve.
    """

    if raw is None:
        return None
    if isinstance(raw, str) and raw in TOOL_CHOICE_SETTINGS:
        return raw
    allowed = ", ".join(sorted(TOOL_CHOICE_SETTINGS))
    raise ValueError(
        f"attacker.tool_choice must be null (omit the key entirely) or one of "
        f"{{{allowed}}}, got {raw!r}"
    )


def tool_choice_payload(setting: str | None) -> dict[str, str] | None:
    """The ``messages.create`` value for a setting, or ``None`` meaning "send no key"."""

    if setting is None:
        return None
    return dict(TOOL_CHOICE_PAYLOADS[normalize_tool_choice(setting)])


def tool_choice_row_value(setting: str | None) -> str:
    """The per-trial recorded value for a setting (``None`` -> :data:`TOOL_CHOICE_OMITTED`)."""

    return TOOL_CHOICE_OMITTED if setting is None else normalize_tool_choice(setting)


# --------------------------------------------------------------------------- #
# The attacker clock (``attacker.clock``)
# --------------------------------------------------------------------------- #
#
# Which clock charges attempt time to the planner's actions -- and therefore at which instants
# the modeled MTD schedule decides whether an exploit's target is still in force. The values
# live in ``measurement.models`` (the measurement core records them and must not import this
# layer); they are re-exported here so the attacker layer has one place to import them from.
#
#   CLOCK_MODELED  fixed per-action quanta (``attacker.planner``: recon command +5 s, snapshot
#                  +30 s, exploit +10 s). The default and the clock of every archived run.
#   CLOCK_WALL     the real elapsed time of each planner call, scan and delivery.
#
# See ``measurement.timing`` for exactly what each clock stamps, and the paired counterfactual
# both clocks record when timing is on. Like ``tool_choice`` this is matched EXACTLY and an
# unrecognized value raises: a typo must never silently run the modeled clock under a config
# that asked for the wall clock (or the reverse), since the clock is the thing under test.


def normalize_clock_mode(raw: object) -> str:
    """Validate a configured ``attacker.clock`` and return it (absent/null -> modeled).

    ``None`` (the key absent, or explicitly nulled in YAML) is the modeled clock, so every
    existing config keeps its behavior. Every other value must be an exact member of
    :data:`CLOCK_MODES`; anything else -- a typo, a bool, a wrong-cased string -- raises
    ``ValueError`` at parse time.
    """

    if raw is None:
        return CLOCK_MODELED
    if isinstance(raw, str) and raw in CLOCK_MODES:
        return raw
    allowed = ", ".join(CLOCK_MODES)
    raise ValueError(f"attacker.clock must be one of {{{allowed}}} (or absent), got {raw!r}")


# How an attempt's action loop ended. One shared vocabulary across backends so a
# single downstream parser handles every row; the applicable subsets differ and are
# published below (``targets_exhausted`` is reachable only by the mock's fixed
# policy, ``no_action_budget`` only by the planner's ``while`` guard).
#
# ``TERMINAL_NOT_RECORDED`` is SCREAMING_CASE while every real reason matches
# ``^[a-z][a-z_]*$``, and it is deliberately NOT a member of ``TERMINAL_REASONS``:
# the sentinel says "this attempt predates the instrumentation", which is not an
# outcome, and the case split makes it unforgeable by a legacy row.
TERMINAL_NOT_RECORDED = "NOT_RECORDED"  # sentinel: this attempt was not instrumented
TERMINAL_PLANNER_STOP = "planner_stop"
TERMINAL_DURATION_LIMIT = "duration_limit"
TERMINAL_ACTION_BUDGET_EXHAUSTED = "action_budget_exhausted"
TERMINAL_NO_ACTION_BUDGET = "no_action_budget"
TERMINAL_TARGETS_EXHAUSTED = "targets_exhausted"
TERMINAL_REASONS = frozenset(
    {
        TERMINAL_PLANNER_STOP,
        TERMINAL_DURATION_LIMIT,
        TERMINAL_ACTION_BUDGET_EXHAUSTED,
        TERMINAL_NO_ACTION_BUDGET,
        TERMINAL_TARGETS_EXHAUSTED,
    }
)
PLANNER_TERMINAL_REASONS = frozenset(
    {
        TERMINAL_PLANNER_STOP,
        TERMINAL_DURATION_LIMIT,
        TERMINAL_ACTION_BUDGET_EXHAUSTED,
        TERMINAL_NO_ACTION_BUDGET,
    }
)
MOCK_TERMINAL_REASONS = frozenset(
    {
        TERMINAL_DURATION_LIMIT,
        TERMINAL_ACTION_BUDGET_EXHAUSTED,
        TERMINAL_TARGETS_EXHAUSTED,
    }
)


@dataclass(frozen=True)
class ActionTraceEntry:
    """One action the attacker chose, recorded whether or not it was carried out.

    ``ordinal`` is the 0-based index of the decision within the attempt (for the
    planner, the index of the ``next_action`` call; it equals the planner's
    ``actions_used`` at the moment of the call). It duplicates the entry's position
    in ``AttemptObservation.action_trace`` on purpose, so a serialized or filtered
    subset of the trace stays interpretable.

    ``target`` is the service identity for an exploit — the same ``ServiceTuple``
    that appears in the trial's exploit deliveries, never a positional index into a
    discovered set, so the record survives any reordering of that set. It is None
    for every other kind.

    ``note`` records why the CLIENT returned this action when it is not the model's
    own choice — read from the client's ``last_action_note`` attribute if it has one,
    "" otherwise. Without it a stop the client fabricated from a malformed tool call
    is indistinguishable from a model that chose to stop.
    """

    ordinal: int
    kind: str
    disposition: str
    target: ServiceTuple | None = None
    note: str = NOTE_NONE

    @property
    def executed(self) -> bool:
        """True iff the action was dispatched to the target executor.

        Derived, not stored, so it can never drift from ``disposition``. Being a
        property (not a field) it is invisible to ``asdict``/equality.
        """

        return self.disposition == DISPOSITION_EXECUTED


@dataclass(frozen=True)
class AttemptObservation:
    """The offensive output of one attempt (fed to the measurement core)."""

    recon_commands: tuple[ReconCommand, ...]
    recon_snapshots: tuple[ReconSnapshot, ...]
    exploit_deliveries: tuple[ExploitDelivery, ...]
    discovered: frozenset[ServiceTuple]
    genuine_hits: frozenset[ServiceTuple]
    decoy_hits: frozenset[ServiceTuple]
    # Instrumentation only (see "Action trace" above). Defaults describe an attempt
    # from a backend that does not record a trace: the empty trace is meaningful ONLY
    # together with a terminal reason, so the default reason is the explicit
    # ``NOT_RECORDED`` sentinel and never a fabricated control-flow outcome.
    action_trace: tuple[ActionTraceEntry, ...] = ()
    terminal_reason: str = TERMINAL_NOT_RECORDED
    # Instrumentation only: the attempt's real action timing and clock counterfactuals
    # (``measurement.timing``). None == this backend did not record timing -- the mock, or the
    # planner under the modeled clock with ``record_timing`` off -- which keeps the default
    # observation deterministic (real durations differ run to run).
    timing: AttemptTiming | None = None


def attempt_memory(attempt_index: int, obs: AttemptObservation) -> AttemptMemory:
    """What the attacker observed in one attempt, for its later attempts (cross-attempt memory).

    Built from the observation's own record of what the attacker observed: the snapshots
    that returned (``recon_snapshots``; a scan whose results came back after the window
    closed was never seen and is not counted), the latest listing (``discovered``, the
    identities the listing showed), and each delivery's ``reached_and_executed`` /
    ``unauthorized_access``. ``ExploitDelivery.target_genuine`` -- the instrument's label --
    is deliberately not read.

    Every delivery is remembered, including the last one of an attempt that used its whole
    action budget: the tool returned its result, though the attempt ended before another
    decision could read it (as a real attacker sees its last command's output). The one
    exception is a delivery that completed after the trial window closed (wall clock, scored at
    completion from 2026-10-31): like a scan that returns after the window, its result was
    never seen, so it is not remembered. It is the attempt's last action, and the only way an
    attempt ends on the duration limit right after an executed exploit. Every attempt
    that was not lost to a fault is remembered, including one the analysis excludes
    (``trial_runner._inclusion``); a faulted attempt is regenerated and only its replacement
    is remembered.
    """

    deliveries = obs.exploit_deliveries
    trace = obs.action_trace
    if (
        deliveries
        and obs.terminal_reason == TERMINAL_DURATION_LIMIT
        and trace
        and trace[-1].kind == KIND_EXPLOIT
        and trace[-1].disposition == DISPOSITION_EXECUTED
    ):
        deliveries = deliveries[:-1]  # completed after the window: never seen
    return AttemptMemory(
        attempt_index=attempt_index,
        scans=len(obs.recon_snapshots),
        last_listing=tuple(sorted(obs.discovered)),
        exploits=tuple(
            (d.target, observed_outcome(d.reached_and_executed, d.unauthorized_access))
            for d in deliveries
        ),
    )


@runtime_checkable
class Attacker(Protocol):
    """Common interface for all attacker backends (mock and LLM-driven)."""

    model_id: str
    model_version: str

    def attempt(self, req: AttemptRequest) -> AttemptObservation:
        """Execute one attempt and return its observations."""
        ...
