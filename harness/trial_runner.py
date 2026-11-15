"""Run one sequence of trials end-to-end.

A *sequence* is ``attempts_per_sequence`` trials against the same evolving
environment, with the attacker retaining context across attempts (the adaptation
mechanism for RQ3). Each attempt: orchestrate MTD over the trial window → verify
the baseline-equivalence checksum (inclusion gate) → run the attacker → attach
operational-cost telemetry → ``compute_trial`` → emit a TrialRow. Deterministic given
the sequence seed.

The cost telemetry is the one place offline and live diverge here. Offline runs attach
the synthetic stand-in from ``harness.telemetry`` (fixtures + determinism). A LIVE run
(an injected non-offline ``env_factory``) attaches it ONLY when
``config.live.synthetic_telemetry`` is explicitly set; otherwise the five synthetic cost
columns are left unmeasured (null, ``operational_cost_source="not_measured"``) and
reconfiguration cost is benchmarked separately at the technique level
(``harness.reconfig_benchmark``). The security outcomes are live measurements in either
mode.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from attacker.base import (
    CLOCK_MODELED,
    Attacker,
    AttackerContext,
    AttemptRequest,
    Environment,
    attempt_memory,
    normalize_clock_mode,
    tool_choice_row_value,
)
from harness import telemetry
from harness.config import ExperimentConfig
from harness.environment import OfflineEnvironment
from measurement import clock
from measurement.compute_trial import compute_availability_events, compute_trial
from measurement.models import (
    COST_SOURCE_NOT_MEASURED,
    COST_SOURCE_SYNTHETIC,
    ServiceTuple,
    TrialInputs,
    TrialMetadata,
    TrialRow,
)
from mtd.checksums import profile_checksum
from mtd.models import ConfigState
from mtd.orchestrator import OrchestrationResult, run_orchestration

_RECON_OFFSET_S = 30

# Builds the attacker's environment from a trial's orchestration. Defaults to the
# offline mock environment; live runs inject a factory that builds a LiveEnvironment
# (and provisions the testbed) — see harness.live_runner.
EnvFactory = Callable[[OrchestrationResult], Environment]


@dataclass(frozen=True)
class SequenceSpec:
    """Everything needed to run one within-subjects sequence."""

    condition: str
    sequence_id: str
    block_id: str
    initial_state: ConfigState
    sequence_start_ts: str
    sequence_seed: int
    adaptation_enabled: bool = True


@dataclass(frozen=True)
class SequenceResult:
    """A sequence's trial rows plus its service-uptime event records.

    ``uptime_events`` are per-failure (or per-trial-censored) records feeding the
    survival analysis of service uptime; one censored record is emitted for a
    trial that suffered no failure (it survived the whole window).
    """

    rows: list[TrialRow]
    uptime_events: list[dict]


def attempt_seed(sequence_seed: int, attempt_index: int, regen: int = 0) -> int:
    """The base seed of one attempt: its MTD schedule is seeded ``^ 0x1``, its attacker ``^ 0x2``
    and its synthetic telemetry ``^ 0x3`` / ``^ 0x4``. ``regen`` counts the attempt's
    regenerations (a regenerated trial's id ends ``-r<regen>``). The analysis rebuilds the MTD
    tick grid from it (``analysis/race.py``)."""

    return (sequence_seed ^ (attempt_index * 0x9E3779B1) ^ (regen * 0x85EBCA77)) & 0xFFFFFFFF


def _system_prompt_hash(config: ExperimentConfig) -> str:
    path = config.attacker.system_prompt_path
    material = ""
    if path and Path(path).exists():
        material = Path(path).read_text(encoding="utf-8")
    else:
        material = f"{config.attacker.model_id}:{config.attacker.model_version}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _inclusion(
    state_profile_checksum: str,
    expected_profile_checksum: str,
    recon_commands: tuple,
) -> tuple[bool, str]:
    """A-priori inclusion gate: checksum passed, attacker init'd, logging complete."""

    if state_profile_checksum != expected_profile_checksum:
        return False, "checksum_mismatch"
    if not recon_commands:
        return False, "attacker_init_or_logging_incomplete"
    return True, ""


def run_sequence(
    spec: SequenceSpec,
    attacker: Attacker,
    config: ExperimentConfig,
    n_attempts: int | None = None,
    fault_rate: float = 0.0,
    max_regen: int = 2,
    env_factory: EnvFactory = OfflineEnvironment,
) -> list[TrialRow]:
    """Run a sequence and return its trial rows (see ``run_sequence_with_uptime``)."""

    return run_sequence_with_uptime(
        spec,
        attacker,
        config,
        n_attempts=n_attempts,
        fault_rate=fault_rate,
        max_regen=max_regen,
        env_factory=env_factory,
    ).rows


def run_sequence_with_uptime(
    spec: SequenceSpec,
    attacker: Attacker,
    config: ExperimentConfig,
    n_attempts: int | None = None,
    fault_rate: float = 0.0,
    max_regen: int = 2,
    env_factory: EnvFactory = OfflineEnvironment,
) -> SequenceResult:
    """Run the attempts of a sequence, returning trial rows + uptime events.

    ``n_attempts`` overrides ``config.attempts_per_sequence`` (used by
    repeatability runs, which execute single-attempt trials). ``fault_rate``
    injects simulated infrastructure faults: a faulted trial is recorded with
    ``excluded=1`` and the attempt is regenerated (state/context not advanced),
    up to ``max_regen`` times, mirroring the a-priori exclude-and-regenerate rule.
    """

    expected_checksum = profile_checksum(spec.initial_state)
    horizon = config.max_trial_duration_s
    sph = _system_prompt_hash(config)
    attempts = n_attempts if n_attempts is not None else config.attempts_per_sequence
    # The clock the ATTACKER actually runs (the planner's ``clock_mode``; an attacker without
    # one -- the offline mock -- runs modeled timings). Read from the attacker, not the config,
    # because the clock-validation pilot runs two attackers that differ only in this.
    clock_mode = normalize_clock_mode(getattr(attacker, "clock_mode", CLOCK_MODELED))

    state = spec.initial_state
    ctx = AttackerContext(adaptation_enabled=spec.adaptation_enabled)
    prev_discovered: frozenset[ServiceTuple] | None = None
    fault_rng = random.Random((spec.sequence_seed ^ 0xFA017) & 0xFFFFFFFF)
    rows: list[TrialRow] = []
    uptime_events: list[dict] = []

    for attempt_index in range(attempts):
        regen = 0
        while True:
            window_start = clock.shift(spec.sequence_start_ts, attempt_index * horizon)
            end_ts = clock.shift(window_start, horizon)
            recon_hint = clock.shift(window_start, _RECON_OFFSET_S)

            base_seed = attempt_seed(spec.sequence_seed, attempt_index, regen)
            orchestration = run_orchestration(
                spec.condition,
                state,
                window_start,
                config.mtd_cadence_s,
                horizon,
                seed=base_seed ^ 0x1,
                rotation_phase=config.mtd_rotation_phase,
            )
            env = env_factory(orchestration)

            req = AttemptRequest(
                attempt_index=attempt_index,
                window_start=window_start,
                recon_ts=recon_hint,
                environment=env,
                context=ctx,
                seed=base_seed ^ 0x2,
                max_actions=config.max_actions,
                trial_duration_limit_s=horizon,
            )
            obs = attacker.attempt(req)

            # The ground truth the attempt's recon is scored against: the configuration in force
            # at the instant the first snapshot REFLECTS (its timestamp under the modeled clock;
            # when the scan probed the targets host under the real clock). Enumeration is still
            # timed to the snapshot's timestamp, when the attacker has the results.
            recon_ts = obs.recon_snapshots[0].as_of if obs.recon_snapshots else recon_hint
            gt_in_force = env.ground_truth_at(recon_ts)
            state_checksum = profile_checksum(state)

            faulted = fault_rate > 0.0 and regen < max_regen and fault_rng.random() < fault_rate
            included, reason = _inclusion(state_checksum, expected_checksum, obs.recon_commands)
            if faulted:
                included, reason = False, "infrastructure_fault"

            event_ts = tuple(e.timestamp for e in orchestration.events)
            # Operational-cost telemetry source. ``harness.telemetry`` is the OFFLINE
            # synthetic stand-in for real monitoring/workload collectors; it must never
            # SILENTLY supply a live run (that is exactly how the confirmatory runs shipped a
            # fully synthetic cost side). Offline runs always use it (fixtures + determinism);
            # a live run uses it ONLY when explicitly opted in via config.live.synthetic_telemetry.
            # Otherwise the five synthetic cost columns are left unmeasured (null) and the row
            # records operational_cost_source="not_measured" -- a value never measured, which
            # is a different fact from one measured as zero. ``operational_event_volume`` is a
            # genuine orchestrator count (reconfig_events below) and is populated either way.
            # Live-ness is read the way the CLI labels it (cli._write_run_meta): the injected
            # env_factory, not the unwired config.live.enabled flag.
            live_run = env_factory is not OfflineEnvironment
            if (not live_run) or config.live.synthetic_telemetry:
                monitoring = telemetry.synth_monitoring(
                    len(orchestration.events),
                    window_start,
                    horizon,
                    config.monitoring_interval_s,
                    seed=base_seed ^ 0x3,
                )
                workload = telemetry.synth_workload(
                    event_ts,
                    window_start,
                    horizon,
                    config.workload.legitimate_rate_rps,
                    seed=base_seed ^ 0x4,
                )
                baseline = telemetry.baseline_reference(spec.initial_state.version)
                cost_source = COST_SOURCE_SYNTHETIC
            else:
                monitoring = ()
                workload = ()
                baseline = None
                cost_source = COST_SOURCE_NOT_MEASURED

            metadata = TrialMetadata(
                trial_id=f"{spec.sequence_id}-a{attempt_index}" + (f"-r{regen}" if regen else ""),
                condition=spec.condition,
                technique=spec.condition,
                attempt_index=attempt_index,
                sequence_id=spec.sequence_id,
                block_id=spec.block_id,
                randomization_seed=spec.sequence_seed,
                start_timestamp=window_start,
                end_timestamp=end_ts,
                workflow_start=(
                    obs.recon_commands[0].timestamp if obs.recon_commands else window_start
                ),
                attacker_model_id=attacker.model_id,
                attacker_model_version=attacker.model_version,
                temperature=config.attacker.temperature,
                context_window=config.attacker.context_window,
                system_prompt_hash=sph,
                max_actions=config.max_actions,
                trial_duration_limit_s=horizon,
                vuln_profile_version=spec.initial_state.version,
                testbed_config_checksum=state_checksum,
                tooling_versions=config.tooling_versions,
                mtd_cadence_s=config.mtd_cadence_s,
                monitoring_interval_s=config.monitoring_interval_s,
                attacker_max_retries=config.attacker.max_retries,
                attacker_request_timeout_s=config.attacker.request_timeout_s,
                # None (the key was omitted from the request) is recorded as the explicit
                # "omitted" word, never as an empty cell -- see measurement.models.
                attacker_tool_choice=tool_choice_row_value(config.attacker.tool_choice),
                excluded=not included,
                exclusion_reason=reason,
                operational_cost_source=cost_source,
                clock_mode=clock_mode,
            )

            inputs = TrialInputs(
                metadata=metadata,
                ground_truth=gt_in_force,
                recon_snapshots=obs.recon_snapshots,
                recon_commands=obs.recon_commands,
                exploit_deliveries=obs.exploit_deliveries,
                monitoring_samples=monitoring,
                workload_samples=workload,
                reconfig_events=orchestration.events,
                baseline_reference=baseline,
                prev_recon_discovered=prev_discovered,
                # instrumentation only: the planner's decision sequence, verbatim
                action_trace=obs.action_trace,
                planner_terminal_reason=obs.terminal_reason,
                # instrumentation only: real action timing + clock counterfactuals
                timing=obs.timing,
            )
            row = compute_trial(inputs)
            rows.append(row)

            if faulted:
                regen += 1
                continue  # regenerate: do not advance state/context

            # uptime events for the survival analysis (included trials only)
            uptime_events.extend(_uptime_records(row, compute_availability_events(inputs), horizon))

            # Not a fault: thread state + context (the attacker's memory) to the next attempt.
            # An attempt the analysis excludes (``_inclusion``) is still remembered; a faulted
            # attempt was regenerated above, and only its replacement is remembered.
            state = orchestration.final_state
            ctx = ctx.updated(
                obs.discovered,
                obs.genuine_hits,
                obs.decoy_hits,
                memory=attempt_memory(attempt_index, obs),
            )
            # Churn compares this attempt's discovered set with the PREVIOUS attempt's, and an
            # attempt's discovered set is the union of all its snapshots (compute_trial,
            # TrialInputs). Until 2026-10-31 the previous set was only its LAST snapshot, so an
            # attempt that rescanned was compared on unlike sets.
            prev_discovered = frozenset().union(*(s.discovered for s in obs.recon_snapshots))
            break

    return SequenceResult(rows=rows, uptime_events=uptime_events)


def _uptime_records(row: TrialRow, events: list, horizon: int) -> list[dict]:
    """Build service-uptime survival records for one trial.

    One record per observed failure (``failure_observed=1`` with its
    time-to-recovery); if the trial had no failure, a single right-censored
    record spanning the whole window (the service never went down).
    """

    common = {
        "trial_id": row.trial_id,
        "condition": row.condition,
        "technique": row.technique,
        "mtd_cadence_s": row.mtd_cadence_s,
    }
    if not events:
        return [
            {
                **common,
                "time_to_failure_s": float(horizon),
                "failure_observed": 0,
                "time_to_recovery_s": None,
                "recovery_censored": 1,
            }
        ]
    records: list[dict] = []
    for e in events:
        records.append(
            {
                **common,
                "time_to_failure_s": e.time_to_failure_s,
                "failure_observed": 1,
                "time_to_recovery_s": e.time_to_recovery_s,
                "recovery_censored": int(e.recovery_censored),
            }
        )
    return records
