"""Typed data models for the measurement instrument.

Every model is a frozen dataclass so that inputs are immutable and hashable
where needed (service tuples participate in set operations for recon metrics).
These models are the boundary between the collectors (which parse raw logs)
and the metric modules (which are pure functions over these types).

See ``docs/metrics_specification.md`` for the authoritative definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    # ``attacker`` depends on ``measurement``; this reference is annotation-only (the
    # module uses ``from __future__ import annotations``), so no runtime cycle exists
    # and the measurement core keeps zero runtime dependencies on the attacker layer.
    from attacker.base import ActionTraceEntry
    from measurement.timing import AttemptTiming

# --------------------------------------------------------------------------- #
# Conditions (mirrors harness/experiment_config.example.yaml)
# --------------------------------------------------------------------------- #

CONDITIONS: tuple[str, ...] = (
    "static_baseline",
    "network_shuffling",
    "port_rotation",
    "endpoint_mutation",
    "deceptive_presentation",
    "combined_mtd_deception",
)


# --------------------------------------------------------------------------- #
# Service identity
# --------------------------------------------------------------------------- #


#: Value recorded when the attacker request carried NO ``tool_choice`` key -- the API
#: default, and what every archived run sent. A WORD, not ``""``/null, because it is a
#: real recorded fact ("we sent nothing") that must stay distinguishable from
#: ``NOT_RECORDED`` ("this row predates the column"). Mirrors
#: ``attacker.base.TOOL_CHOICE_OMITTED`` (pinned equal by test); duplicated rather than
#: imported because ``attacker.base`` imports THIS module.
TOOL_CHOICE_OMITTED = "omitted"

#: Operational-cost provenance (the ``operational_cost_source`` column): how the FIVE
#: synthetic cost columns on a row -- ``latency_overhead_ms``, ``cpu_overhead_pct``,
#: ``memory_overhead_pct``, ``network_overhead_pct``, ``service_availability_impact`` -- were
#: produced. ``"synthetic"`` = the offline synthetic-telemetry generator (``harness.telemetry``)
#: supplied them; ``"not_measured"`` = a live run with the synthetic stand-in gated off, so
#: those five columns are null -- NEVER ``0``, because a value never measured is a different
#: fact from a value measured as zero. ``operational_event_volume`` is NOT governed by this
#: flag: it is a genuine count from the MTD orchestrator event log and is populated in both
#: modes. A dataset written before this column reads back ``NOT_RECORDED`` ("the dataset did
#: not record its cost provenance"), never ``"synthetic"`` -- the dataset states what the
#: dataset recorded, and the (synthetic) provenance of the runs archived under
#: ``docs/results/`` is documented in ``docs/metrics_specification.md`` §3. Measured
#: operational cost is produced at the TECHNIQUE level by the reconfiguration benchmark
#: (``harness.reconfig_benchmark``), a separate artifact, never as a per-trial column.
COST_SOURCE_SYNTHETIC = "synthetic"
COST_SOURCE_NOT_MEASURED = "not_measured"

#: The attacker clock (the ``clock_mode`` column; ``attacker.clock`` in the config). Which
#: clock charged attempt time to the attacker's actions -- and therefore which instants the
#: modeled MTD schedule was consulted at (``docs/metrics_specification.md`` §4f):
#:
#:   ``"modeled"``  fixed per-action quanta from ``attacker/planner.py`` (recon command +5 s,
#:                  recon snapshot +30 s, exploit +10 s), whatever the real elapsed time. The
#:                  default, and the clock of every run archived under ``docs/results/``.
#:   ``"wall"``     the real elapsed time of each planner call, scan and delivery, measured on
#:                  the control host's monotonic clock.
#:
#: Defined HERE (not in ``attacker.base``) because the measurement core needs the values and
#: must not import the attacker layer; ``attacker.base`` re-exports these same objects. A row
#: written before the column existed reads back ``NOT_RECORDED`` -- the archived runs used the
#: modeled clock, which §4f documents rather than back-writing it into rows that never said so.
CLOCK_MODELED = "modeled"
CLOCK_WALL = "wall"
CLOCK_MODES: tuple[str, ...] = (CLOCK_MODELED, CLOCK_WALL)


@dataclass(frozen=True, order=True)
class ServiceTuple:
    """A service–port–endpoint–address identity.

    This tuple is the unit of set-membership for reconnaissance accuracy and
    attack-path churn, so what it distinguishes decides what the study can measure.
    Equality/hashing is over ALL FOUR fields (frozen dataclass), which is what makes
    each of the four mutable observable attributes an evaluable manipulation.

    ``endpoint`` is "" for services with no path component (e.g., a raw TCP service).
    ``address`` is the **modeled observable** address — what the attacker fingerprints,
    NOT the concrete host a payload is delivered to. Live, delivery goes to the frozen
    physical host and the ``harness.safety.assert_within_cidr`` allowlist is asserted
    against THAT address only (``harness.live_targets.LiveService.delivery_host()``);
    a modeled address must never be fed to the allowlist, which would abort every live
    run (modeled addresses are ``10.10.x.x``; the live CIDR is ``172.31.66.0/24``).
    Adding it is what made ``network_shuffling`` evaluable: it mutates only the address,
    and while the address sat outside this tuple that condition was instantiated but
    never measured — see ``docs/null_condition_audit.md``.

    ``address`` is LAST and defaults to "" for two load-bearing reasons:

    1. This dataclass is ``order=True`` and ``attacker.mock`` / ``attacker.backends``
       choose exploit targets from ``sorted(...)``, so FIELD POSITION affects outcomes.
       Last means the address only ever breaks a tie among identical
       service/port/endpoint triples — which cannot occur in any arm that does not
       mutate the address — so the five non-shuffling conditions keep their exact
       target order.
    2. The default keeps every 3-positional construction site valid, including the
       ``[service, port, endpoint]`` fixture encoding read by
       ``measurement.collectors.parsers`` and the 3-field action-trace tokens in the
       frozen datasets. A missing address decodes to "" and is never fabricated into a
       guessed value.
    """

    service: str
    port: int
    endpoint: str = ""
    address: str = ""


# --------------------------------------------------------------------------- #
# Ground truth (regenerated on every configuration change)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GroundTruthInventory:
    """Genuine and decoy services for one configuration state.

    ``config_checksum`` ties the inventory to the exact configuration state it
    describes (baseline-equivalence checksum). ``genuine`` is the set scored for
    recall; ``decoys`` are deceptive services that count against precision when
    "discovered".
    """

    config_checksum: str
    genuine: frozenset[ServiceTuple]
    decoys: frozenset[ServiceTuple] = frozenset()


# --------------------------------------------------------------------------- #
# Attacker-side observations
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ReconSnapshot:
    """The attacker's discovered service set after one recon pass.

    ``timestamp`` is when the attacker HAS the results (the recon's completion; enumeration is
    timed to it). ``observed_at`` is the instant whose configuration the discovered identities
    reflect, when that is earlier -- under the real attacker clock, the instant the scan probed
    the targets host (``attacker.planner``, timing format v2). ``None`` (the modeled clock, the
    offline mock, every archived run) means the snapshot reflects its own ``timestamp``. A
    snapshot is scored against the ground truth ``as_of`` the instant it reflects.
    """

    attempt_index: int
    timestamp: str  # ISO-8601 on the synchronized clock
    discovered: frozenset[ServiceTuple]
    observed_at: str | None = None

    @property
    def as_of(self) -> str:
        """The instant whose configuration this snapshot reports."""

        return self.timestamp if self.observed_at is None else self.observed_at


@dataclass(frozen=True)
class ReconCommand:
    """A single reconnaissance command emitted by the workflow."""

    timestamp: str
    command: str


@dataclass(frozen=True)
class ExploitDelivery:
    """A discrete exploit-delivery attempt and its verified outcome.

    ``reached_and_executed`` marks a *technically valid* exploit (payload reached
    the target and executed) regardless of objective — this drives
    time-to-first-valid-exploit. ``target_genuine`` and ``unauthorized_access``
    are set by log correlation against ground truth and drive objective coding.
    """

    timestamp: str
    target: ServiceTuple
    reached_and_executed: bool
    target_genuine: bool
    unauthorized_access: bool  # command execution or data access (obj_b evidence)


# --------------------------------------------------------------------------- #
# Operational telemetry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MonitoringSample:
    """One utilization sample (percent units, 0–100) at a timestamp."""

    timestamp: str
    cpu_pct: float
    memory_pct: float
    network_pct: float


@dataclass(frozen=True)
class WorkloadSample:
    """One synthetic legitimate request result.

    ``answered_within_timeout`` is the availability criterion; ``latency_ms`` is
    used for latency-overhead comparison (only meaningful for answered requests).
    """

    timestamp: str
    answered_within_timeout: bool
    latency_ms: float | None = None


@dataclass(frozen=True)
class ReconfigEvent:
    """A timestamped MTD reconfiguration event (source of event volume)."""

    timestamp: str
    technique: str


# --------------------------------------------------------------------------- #
# Baseline reference (for overhead deltas)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BaselineReference:
    """Static-baseline reference means for the matching workload + profile.

    Overhead metrics are computed relative to this reference (see A4 in
    ``docs/open_items.md``). Tied to ``vuln_profile_version`` so the pairing is
    explicit.
    """

    vuln_profile_version: str
    mean_response_ms: float
    mean_cpu_pct: float
    mean_memory_pct: float
    mean_network_pct: float


# --------------------------------------------------------------------------- #
# Per-trial control/metadata (recorded on every row)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TrialMetadata:
    """Control variables and provenance recorded on every trial row.

    Mirrors ``docs/metrics_specification.md`` §4. ``workflow_start`` is the
    origin for time-to-first-valid-exploit; ``first_recon_ts`` (if absent) is
    derived from the offensive log.
    """

    trial_id: str
    condition: str
    technique: str
    attempt_index: int
    sequence_id: str
    block_id: str
    randomization_seed: int
    start_timestamp: str
    end_timestamp: str
    workflow_start: str
    attacker_model_id: str
    attacker_model_version: str
    # None == the request omitted ``temperature`` and the model used its own default.
    # Recorded as null (empty CSV cell), NEVER as 0.0: a reader must be able to tell
    # "we sent 0.7" from "we sent nothing" when reproducing a run.
    temperature: float | None
    context_window: int
    system_prompt_hash: str
    max_actions: int
    trial_duration_limit_s: int
    vuln_profile_version: str
    testbed_config_checksum: str
    tooling_versions: str
    mtd_cadence_s: int
    monitoring_interval_s: int
    attacker_max_retries: int = 0
    attacker_request_timeout_s: float = 0.0
    attacker_tool_choice: str = TOOL_CHOICE_OMITTED
    excluded: bool = False
    exclusion_reason: str = ""
    # How this trial's operational-cost columns were produced (COST_SOURCE_* above). Set by
    # the harness from the telemetry path actually taken; defaults to synthetic because the
    # offline path (the default, and every construction site that predates this field) uses
    # the synthetic generator. Copied verbatim onto the TrialRow by ``compute_trial``.
    operational_cost_source: str = COST_SOURCE_SYNTHETIC
    # Which attacker clock charged attempt time to the actions (CLOCK_* above). Set by the
    # harness from the attacker it ran; defaults to modeled, the clock of every construction
    # site that predates this field. Copied verbatim onto the TrialRow by ``compute_trial``.
    clock_mode: str = CLOCK_MODELED


# --------------------------------------------------------------------------- #
# Bundled inputs for one trial's metric computation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TrialInputs:
    """Everything the measurement core needs to compute one TrialRow.

    A *trial* is one attempt against one configuration state. ``recon_snapshots``
    holds this attempt's recon pass(es); the trial's discovered set is their union.
    ``prev_recon_discovered`` is the previous attempt's discovered set in the same
    sequence (None for ``attempt_index == 0``), used only for attack-path churn.
    """

    metadata: TrialMetadata
    ground_truth: GroundTruthInventory
    recon_snapshots: tuple[ReconSnapshot, ...] = ()
    recon_commands: tuple[ReconCommand, ...] = ()
    exploit_deliveries: tuple[ExploitDelivery, ...] = ()
    monitoring_samples: tuple[MonitoringSample, ...] = ()
    workload_samples: tuple[WorkloadSample, ...] = ()
    reconfig_events: tuple[ReconfigEvent, ...] = ()
    baseline_reference: BaselineReference | None = None
    prev_recon_discovered: frozenset[ServiceTuple] | None = None
    # --- planner instrumentation (pure; feeds no dependent variable) ---
    # ``action_trace`` is the decision sequence the attacker backend recorded, in
    # order. ``()`` is a REAL observation (the backend recorded zero decisions);
    # ``None`` means no trace object was supplied at all. Whether a trace was
    # recorded is decided by ``planner_terminal_reason``: a backend that records
    # nothing leaves it at the ``NOT_RECORDED`` sentinel (or None), because an empty
    # trace alone cannot distinguish "chose nothing" from "was never instrumented".
    action_trace: tuple[ActionTraceEntry, ...] | None = None
    planner_terminal_reason: str | None = None
    # --- real action timing (pure instrumentation; feeds no dependent variable) ---
    # ``None`` == the attacker backend did not record timing (the mock, or the planner with
    # ``record_timing`` off under the modeled clock); the timing columns then read
    # ``NOT_RECORDED`` / null. See ``measurement.timing``.
    timing: AttemptTiming | None = None


# --------------------------------------------------------------------------- #
# Availability survival events (separate artifact)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AvailabilityEvent:
    """A per-event failure/recovery record for service-uptime survival analysis."""

    failure_timestamp: str
    time_to_failure_s: float
    time_to_recovery_s: float | None  # None if not recovered within the trial
    recovery_censored: bool


# --------------------------------------------------------------------------- #
# The trial-level dataset row
# --------------------------------------------------------------------------- #


#: Metadata key marking a ``TrialRow`` field OPTIONAL-ON-READ: a dataset CSV written
#: before that column existed may omit it, and ``run_sheet.read_csv`` then substitutes
#: the value carried by this metadata entry instead of raising. Absence of the key means
#: the column is REQUIRED, so a typo in a field declaration fails CLOSED (strict) rather
#: than silently widening the tolerated set.
ABSENT_ON_READ = "absent_on_read"

#: Explicit "this row predates the instrumentation" sentinel. SCREAMING_CASE so it can
#: never collide with a real terminal reason (all of which match ``^[a-z][a-z_]*$``) and
#: never with an encoded action sequence. Distinct from ``""``, which means "the trace WAS
#: recorded and the planner made zero decisions". Mirrors
#: ``attacker.base.TERMINAL_NOT_RECORDED`` (pinned equal by test).
NOT_RECORDED = "NOT_RECORDED"


@dataclass
class TrialRow:
    """One row of the trial-level dataset.

    Field order here defines the canonical column order used by the run-sheet
    writer (``run_sheet.TRIAL_COLUMNS``). Metric fields default to sentinels so a
    partially-computed row is detectable; a *complete* row has all non-metadata
    fields populated or explicitly censored/flagged.
    """

    # --- metadata / control (from TrialMetadata) ---
    trial_id: str
    condition: str
    technique: str
    attempt_index: int
    sequence_id: str
    block_id: str
    randomization_seed: int
    start_timestamp: str
    end_timestamp: str
    attacker_model_id: str
    attacker_model_version: str
    # None == the request omitted ``temperature`` and the model used its own default.
    # Recorded as null (empty CSV cell), NEVER as 0.0: a reader must be able to tell
    # "we sent 0.7" from "we sent nothing" when reproducing a run.
    temperature: float | None
    context_window: int
    system_prompt_hash: str
    max_actions: int
    trial_duration_limit_s: int
    vuln_profile_version: str
    testbed_config_checksum: str
    tooling_versions: str
    mtd_cadence_s: int
    monitoring_interval_s: int
    # resilience provenance (operational; the API retry budget in force for this run).
    # PRE-EXISTING BUG FIX (unrelated to the action trace): these two columns postdate
    # docs/results/pilot/trial_dataset.csv, whose CSV omits them, so that frozen
    # dataset already failed ``run_sheet.read_csv`` / ``cli.py analyze``. Marking them
    # optional-on-read with a ``None`` substitute makes it readable again while stating
    # honestly that the retry budget is UNKNOWN for those rows -- never "0 retries / 0 s
    # timeout", which is a fabricated operational fact. The DEFAULT stays 0 / 0.0, so
    # nothing about a freshly computed row changes.
    attacker_max_retries: int | None = field(default=0, metadata={ABSENT_ON_READ: None})
    attacker_request_timeout_s: float | None = field(default=0.0, metadata={ABSENT_ON_READ: None})

    # --- primary security outcomes ---
    # exploit_success is enumeration-inclusive (>=1 objective, per v2); the
    # co-primary exploitation_success = obj_b (actual unauthorized exec/data access
    # on a genuine service). See docs/metrics_specification.md and open_items A1.
    exploit_success: int = 0
    exploitation_success: int = 0
    obj_a: int = 0
    obj_b: int = 0
    obj_c: int = 0
    objective_count: int = 0
    # PRIMARY survival outcome: time_to_compromise_s is anchored at the actual COMPROMISE
    # (obj_b: unauthorized exec/data access on a genuine service), right-censored when no
    # compromise occurred — the time-domain analogue of exploitation_success. It carries
    # the graded MTD effect (see metrics_spec §1.2).
    time_to_compromise_s: float | None = None
    ttc_censored: int = 0
    # SECONDARY/diagnostic: time_to_enumeration_s is the former enumeration-inclusive
    # (first-objective / obj_a) timing; it saturates near the recon offset and is largely
    # insensitive to MTD. Reported descriptively and FDR-controlled (metrics_spec §2.x).
    time_to_enumeration_s: float | None = None
    tten_censored: int = 0

    # --- secondary / diagnostic ---
    time_to_first_valid_exploit_s: float | None = None
    tfve_censored: int = 0
    recon_accuracy_f1: float | None = None
    attack_path_churn: float | None = None
    attempts_per_success: int | None = None
    aps_censored: int = 0

    # --- operational overhead ---
    latency_overhead_ms: float | None = None
    cpu_overhead_pct: float | None = None
    memory_overhead_pct: float | None = None
    network_overhead_pct: float | None = None
    operational_event_volume: int = 0
    service_availability_impact: float | None = None

    # --- inclusion / exclusion ---
    excluded: int = 0
    exclusion_reason: str = ""

    # --- planner action trace (PURE INSTRUMENTATION; never a dependent variable) ---
    # These five record what the attacker's planner CHOSE, in order, so a trial that
    # produced no compromise can be characterized directly instead of inferred from
    # censoring. They feed no hypothesis test, effect size or CI.
    #
    # ``action_sequence`` and ``planner_terminal_reason`` default to NOT_RECORDED, never
    # "": an empty sequence means the planner made zero decisions, which is a real
    # observation and must stay distinguishable from "no trace was captured". The counts
    # are ``int | None`` for the same reason -- a bare ``int`` could hold "not recorded"
    # only by fabricating a 0, and 0 is itself a real value.
    action_sequence: str = field(default=NOT_RECORDED, metadata={ABSENT_ON_READ: NOT_RECORDED})
    n_actions: int | None = field(default=None, metadata={ABSENT_ON_READ: None})
    n_recon_actions: int | None = field(default=None, metadata={ABSENT_ON_READ: None})
    n_exploit_actions: int | None = field(default=None, metadata={ABSENT_ON_READ: None})
    planner_terminal_reason: str = field(
        default=NOT_RECORDED, metadata={ABSENT_ON_READ: NOT_RECORDED}
    )

    # --- attacker request-shaping provenance ---
    # Which ``tool_choice`` the attacker's Messages-API request carried:
    # ``TOOL_CHOICE_OMITTED`` (the DEFAULT -- no key sent, the only setting whose request
    # is byte-identical to the frozen arms) or the value sent ("any"/"auto"). Rows written
    # before this column existed read back as ``NOT_RECORDED`` -- honestly "unknown", never
    # "omitted", which is itself a recorded observation about the request.
    #
    # APPENDED, not grouped with the other ``attacker_*`` provenance columns, because
    # ``TrialRow`` field order IS the CSV column order and ``read_csv`` -> ``write_csv``
    # is contracted to upgrade a legacy dataset PURELY ADDITIVELY: absent columns are
    # appended, present ones keep their position. Inserting mid-list would rewrite the
    # archived datasets' column order on any round trip.
    attacker_tool_choice: str = field(
        default=TOOL_CHOICE_OMITTED, metadata={ABSENT_ON_READ: NOT_RECORDED}
    )

    # --- decoy engagement (PURE INSTRUMENTATION; never a dependent variable) ---
    # Persisted so "the attacker AIMED at a decoy first" (bait-taking) and "the attacker
    # fell back onto a decoy after failing elsewhere" stay separable WITHOUT re-parsing
    # ``action_sequence``. Both readings were previously collapsed into one number: in the
    # rotating arms the attacker aims at a decoy in 36-49% of trials, but its FIRST exploit
    # hits one in 0-2.6% of them, and those are different claims about deception.
    #
    # Derived at row-computation time from the trial's own ``ExploitDelivery`` records --
    # ``target_genuine`` is already stamped from ground truth when the exploit is fired
    # (``harness/live_contract.py`` §2) -- so this is a DERIVED FACT, not a new measurement.
    # Under the wall clock a delivery overtaken before it completed reads "genuine" if aimed
    # at a genuine service and "unresolved" if aimed at a decoy (scored at completion,
    # 2026-10-31); ``rq2.decoy_targeting`` classifies by name instead.
    #
    # ``first_exploit_target`` is the ANCHOR of the group (the role
    # ``planner_terminal_reason`` plays for the action trace): it alone distinguishes every
    # state the other two columns cannot express on their own.
    #
    #   first_exploit_target
    #     "genuine"             the first delivery targeted a live genuine service
    #     "decoy"               the first delivery targeted a live decoy
    #     "unresolved"          the first delivery targeted an identity that was neither
    #                           (a stale/hallucinated tuple; nothing was there to reach)
    #     NO_EXPLOIT_DELIVERY   RECORDED, and the trial made zero exploit deliveries
    #     NOT_RECORDED          the row predates these columns; nothing is known
    #
    #   first_decoy_delivery_ordinal
    #     int >= 1  1-BASED position of the first decoy-aimed delivery within the trial's
    #               chronologically ordered deliveries -- the same sequence and the same
    #               base as ``attempts_per_success``, so the two are directly comparable
    #     None      no decoy-aimed delivery occurred, OR the row is not recorded; read the
    #               anchor to tell those apart
    #
    #   n_decoy_targeted_deliveries
    #     int >= 0  how many deliveries were decoy-aimed (0 is a real observation)
    #     None      not recorded
    #
    # APPENDED LAST, after ``attacker_tool_choice``, because ``TrialRow`` field order IS the
    # CSV column order and ``read_csv`` -> ``write_csv`` is contracted to widen a legacy
    # dataset PURELY ADDITIVELY.
    first_exploit_target: str = field(default=NOT_RECORDED, metadata={ABSENT_ON_READ: NOT_RECORDED})
    first_decoy_delivery_ordinal: int | None = field(default=None, metadata={ABSENT_ON_READ: None})
    n_decoy_targeted_deliveries: int | None = field(default=None, metadata={ABSENT_ON_READ: None})
    # Provenance for the operational-cost columns (COST_SOURCE_* above; metrics spec §3).
    # APPENDED LAST so the column order stays additive and the six archived datasets (which
    # lack it) upgrade purely. Optional-on-read -> NOT_RECORDED for those datasets: the
    # dataset records what the dataset recorded, and the synthetic provenance of the runs that
    # predate this column is documented in the metrics spec, not asserted into their rows.
    operational_cost_source: str = field(
        default=COST_SOURCE_SYNTHETIC, metadata={ABSENT_ON_READ: NOT_RECORDED}
    )

    # --- attacker clock + real action timing (PURE INSTRUMENTATION; never a DV) ---
    # Filled by ``measurement.timing.timing_columns`` (docs/metrics_specification.md §4f).
    # ``clock_mode`` records which clock charged attempt time to the attacker's actions and is
    # set on every freshly computed row. The ten columns after it summarize the REAL timing of
    # the attempt and are populated only when timing was recorded (``attacker.record_timing``,
    # forced on under ``clock: wall``); otherwise they hold the not-recorded sentinels, so the
    # offline pipeline stays byte-deterministic. ``action_timing`` carries the per-decision
    # detail (versioned JSON) from which every summary can be recomputed.
    #
    # APPENDED LAST, after ``operational_cost_source``, for the same reason every instrumentation
    # group before it was: ``TrialRow`` field order IS the CSV column order and ``read_csv`` ->
    # ``write_csv`` widens a legacy dataset PURELY ADDITIVELY. Every archived dataset predates
    # them and reads back ``NOT_RECORDED`` / null.
    clock_mode: str = field(default=CLOCK_MODELED, metadata={ABSENT_ON_READ: NOT_RECORDED})
    wall_start_utc: str = field(default=NOT_RECORDED, metadata={ABSENT_ON_READ: NOT_RECORDED})
    attempt_wall_s: float | None = field(default=None, metadata={ABSENT_ON_READ: None})
    llm_latency_mean_s: float | None = field(default=None, metadata={ABSENT_ON_READ: None})
    llm_latency_max_s: float | None = field(default=None, metadata={ABSENT_ON_READ: None})
    scan_duration_mean_s: float | None = field(default=None, metadata={ABSENT_ON_READ: None})
    first_exploit_gap_modeled_s: float | None = field(default=None, metadata={ABSENT_ON_READ: None})
    first_exploit_gap_wall_s: float | None = field(default=None, metadata={ABSENT_ON_READ: None})
    n_deliveries_clock_disagree: int | None = field(default=None, metadata={ABSENT_ON_READ: None})
    first_delivery_clock_disagree: int | None = field(default=None, metadata={ABSENT_ON_READ: None})
    action_timing: str = field(default=NOT_RECORDED, metadata={ABSENT_ON_READ: NOT_RECORDED})
