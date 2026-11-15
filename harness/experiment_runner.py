"""Experiment runner: condition blocks, randomized order, pilot/repeatability.

Manages the full experiment: condition-block order is randomized with the
recorded seed; each condition runs ``sequences_per_condition`` within-subjects
sequences. Two special modes:

- **pilot** — baseline + one MTD condition, fewer sequences (IRB pilot).
- **repeatability** — single-attempt trials with configuration fixed and
  adaptation disabled, to isolate measurement variance for the ICC.
- **clock_validation** — the real-timing validation pilot: every (condition, sequence) is
  run once per attacker *arm* (the modeled and the wall clock) with the SAME seed and
  timeline slot, so both arms face the identical MTD schedule and phase; the arm order
  alternates by sequence index. See ``docs/clock_validation_runbook.md``.

Deterministic given the config seed. Returns the trial-level dataset (list of
TrialRow); the CLI writes it via ``measurement.run_sheet``.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from attacker.base import Attacker
from attacker.errors import AttackerApiError
from harness.checkpoint import RunCheckpoint
from harness.config import RUN_ORDER_BLOCKED, RUN_ORDER_INTERLEAVED, ExperimentConfig
from harness.environment import OfflineEnvironment
from harness.safety import ProvisioningError
from harness.trial_runner import EnvFactory, SequenceSpec, run_sequence, run_sequence_with_uptime
from measurement import clock
from measurement.models import TrialRow
from mtd.models import ConfigState

_BASE_START = "2026-07-01T00:00:00Z"

# Faults that a full run recovers from (exclude the sequence and continue, or halt to
# resume) rather than crashing: a transient API outage that outlasted all retries, or an
# infra/provisioning fault. Anything else propagates (it is a bug, not a transient fault).
_EXCLUDABLE_FAULTS = (AttackerApiError, ProvisioningError)


class RunHalted(RuntimeError):
    """Raised under ``on_trial_failure='halt'`` so the CLI can checkpoint + exit non-zero."""

    def __init__(self, sequence_id: str, reason: str):
        super().__init__(f"halted at sequence {sequence_id!r} ({reason})")
        self.sequence_id = sequence_id
        self.reason = reason


def _fault_reason(exc: BaseException) -> str:
    if isinstance(exc, AttackerApiError):
        return "api_overload"
    if isinstance(exc, ProvisioningError):
        return "infrastructure_fault"
    return "unrecoverable_fault"  # pragma: no cover - defensive


def _fault_detail(exc: BaseException) -> str:
    """What failed, in the fault's own words (one line, bounded): the audit trail for an exclusion.

    ``reason`` says which kind of fault; this says which one -- a readiness timeout, a scan that
    failed twice, a delivery without a verdict, a decoy that did not answer -- so the dissertation can report
    every lost sequence's cause rather than a category.
    """

    text = " ".join(str(exc).split())
    return text[:300] if text else type(exc).__name__


def _hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


@dataclass(frozen=True)
class ExperimentDataset:
    """The full experiment output: trial-level rows + service-uptime event log.

    ``excluded_sequences`` records whole sequences lost to an unrecoverable API/infra
    fault under the ``exclude`` policy (each: sequence_id, block_index, seq_i, reason,
    n_lost_attempts) — surfaced in ``run_meta.json``.
    """

    trial_rows: list[TrialRow]
    uptime_events: list[dict]
    excluded_sequences: list[dict] = field(default_factory=list)


def randomized_block_order(conditions: tuple[str, ...], seed: int) -> list[str]:
    """Condition-block order randomized with the recorded seed."""

    rng = random.Random(seed)
    order = list(conditions)
    rng.shuffle(order)
    return order


def run_schedule(
    order: Sequence[str], sequences: int, run_order: str, seed: int
) -> list[tuple[str, int]]:
    """The (condition, sequence index) units of a run, in the order they execute.

    ``blocked`` (every archived run): each condition's sequences back to back, conditions in
    ``order``. ``interleaved``: round r runs sequence r of EVERY condition, in an order shuffled
    per round from the master seed, so no condition is tied to one stretch of the run (C209).
    The set of units -- and hence every seed, keyed on (condition, sequence) -- is identical.
    """

    if run_order == RUN_ORDER_BLOCKED:
        return [(condition, seq_i) for condition in order for seq_i in range(sequences)]
    if run_order != RUN_ORDER_INTERLEAVED:  # pragma: no cover - validated at config load
        raise ValueError(f"unknown run order {run_order!r}")
    schedule: list[tuple[str, int]] = []
    for seq_i in range(sequences):
        rng = random.Random(((seed & 0xFFFFFFFF) * 1000003 + 0x5EED + seq_i) & 0xFFFFFFFFFFFF)
        round_order = list(order)
        rng.shuffle(round_order)
        schedule.extend((condition, seq_i) for condition in round_order)
    return schedule


def _sequence_start(base_ts: str, global_index: int, attempts: int, horizon: int) -> str:
    """Give each sequence its own non-overlapping slot on the shared timeline."""

    return clock.shift(base_ts, global_index * attempts * horizon)


def arm_order(arms: Sequence[tuple[str, Attacker]], seq_i: int) -> list[tuple[str, Attacker]]:
    """The order a (condition, sequence)'s arms run in: as given for even ``seq_i``, reversed
    for odd -- an ABBA alternation, so neither arm always runs first (and so is never
    systematically the one that meets a warmer cache, a later hour or a drifting API latency).
    """

    ordered = list(arms)
    return ordered if seq_i % 2 == 0 else ordered[::-1]


def run_experiment(
    config: ExperimentConfig,
    attacker: Attacker,
    initial_state: ConfigState,
    mode: str = "full",
    base_start_ts: str = _BASE_START,
    fault_rate: float = 0.0,
    env_factory: EnvFactory = OfflineEnvironment,
) -> list[TrialRow]:
    """Run the experiment and return its trial rows (see ``run_experiment_dataset``)."""

    return run_experiment_dataset(
        config,
        attacker,
        initial_state,
        mode=mode,
        base_start_ts=base_start_ts,
        fault_rate=fault_rate,
        env_factory=env_factory,
    ).trial_rows


def run_experiment_dataset(
    config: ExperimentConfig,
    attacker: Attacker,
    initial_state: ConfigState,
    mode: str = "full",
    base_start_ts: str = _BASE_START,
    fault_rate: float = 0.0,
    env_factory: EnvFactory = OfflineEnvironment,
    checkpoint: RunCheckpoint | None = None,
    on_trial_failure: str = "exclude",
    max_consecutive_exclusions: int = 5,
    progress: Callable[[str], None] | None = None,
    arms: Sequence[tuple[str, Attacker]] | None = None,
    usage_snapshot: Callable[[], dict] | None = None,
) -> ExperimentDataset:
    """Run the full, pilot or clock-validation experiment across randomized condition blocks.

    ``fault_rate`` (0 by default) injects a small fraction of excluded/regenerated
    trials to exercise the a-priori inclusion/exclusion handling downstream.

    The full run executes its (condition, sequence) units in ``config.run_order``
    (:func:`run_schedule`): condition blocks, or interleaved rounds. Seeds, sequence ids and block
    ids depend only on the (condition, sequence) unit, never on the order.

    When a ``checkpoint`` is given the run is crash-safe and resumable at sequence
    granularity: each completed sequence is appended to the checkpoint and, on a restart,
    already-completed sequences are **skipped** while the seeded order is replayed
    identically (so the remaining sequences get byte-identical seeds/timestamps). The
    ``global_seq`` counter advances for skipped sequences too, which is what preserves the
    timeline. ``on_trial_failure`` controls what happens when a sequence still fails after
    all retries (``exclude`` = record + skip + continue; ``halt`` = raise ``RunHalted`` so
    the operator resumes later). ``progress`` receives one line per completed/excluded
    sequence. None of this affects the science.

    ``arms`` (``[(label, attacker), ...]``) runs every (condition, sequence) once PER ARM, in
    :func:`arm_order`, with the identical seed and timeline slot -- the paired design of the
    clock-validation pilot. Each arm's sequence gets its own ``sequence_id``
    (``<condition>-s<i>-<label>``), so it is its own checkpoint unit and its own analysis
    cluster; ``attacker`` is then unused. Without ``arms`` the loop is exactly the single-
    attacker loop it always was (one pass per sequence, unsuffixed ids).

    ``usage_snapshot`` returns this process's cumulative API usage; with a checkpoint it is
    recorded after every sequence, so the run's provenance covers every process of a resumed run.
    """

    if mode == "pilot":
        conditions = config.pilot.conditions
        sequences = config.pilot.sequences_per_condition
    elif mode == "full":
        conditions = config.conditions
        sequences = config.sequences_per_condition
    elif mode == "clock_validation":
        conditions = config.clock_validation.conditions
        sequences = config.clock_validation.sequences_per_condition
        if not arms:
            raise ValueError("mode 'clock_validation' needs the clock arms to run")
    else:  # pragma: no cover - defensive
        raise ValueError(
            f"unknown experiment mode: {mode!r} (use 'full', 'pilot' or 'clock_validation')"
        )

    arm_list: list[tuple[str | None, Attacker]] = list(arms) if arms else [(None, attacker)]
    labels = [label for label, _ in arm_list]
    if arms and (len(set(labels)) != len(labels) or not all(labels)):
        raise ValueError(f"arm labels must be unique and non-empty, got {labels!r}")

    order = randomized_block_order(conditions, config.randomization_seed)
    # A condition's block index stays its position in the seeded block order whatever the run
    # order, so seeds and block ids per (condition, sequence) do not depend on the run order.
    block_index_of = {condition: i for i, condition in enumerate(order)}
    # The full run honors ``design.run_order``; the pilot and the clock-validation pilot run
    # blocked, as they always have.
    run_order = getattr(config, "run_order", RUN_ORDER_BLOCKED) if mode == "full" else None
    run_order = run_order or RUN_ORDER_BLOCKED
    schedule = run_schedule(order, sequences, run_order, config.randomization_seed)
    total = len(order) * sequences * len(arm_list)
    completed = checkpoint.completed_ids() if checkpoint else set()
    if progress and completed:
        progress(f"[resume] skipping {len(completed)}/{total} already-completed sequences")

    rows: list[TrialRow] = []
    uptime_events: list[dict] = []
    excluded_sequences: list[dict] = []
    run_start = time.monotonic()
    global_seq = 0
    run_index = 0  # sequences run (or skipped) so far, counting every arm -- progress only
    consecutive_exclusions = 0
    # Per condition too: under the interleaved order a fault confined to ONE condition fails
    # only every k-th sequence, and the successes in between reset the global count, so the
    # global breaker alone would let a broken arm lose every remaining sequence silently.
    consecutive_by_condition: dict[str, int] = {}
    try:
        for condition, seq_i in schedule:
            block_index = block_index_of[condition]
            base_sequence_id = f"{condition}-s{seq_i}"
            seq_seed = (
                config.randomization_seed * 1000003 + block_index * 101 + seq_i
            ) & 0xFFFFFFFF
            start = _sequence_start(
                base_start_ts, global_seq, config.attempts_per_sequence, config.max_trial_duration_s
            )
            # Advance regardless of skip -> identical timeline on resume. ONE slot per
            # (condition, sequence): every arm of a paired sequence shares it, with its seed.
            global_seq += 1
            for arm_label, arm_attacker in arm_order(arm_list, seq_i):
                run_index += 1
                sequence_id = base_sequence_id + (f"-{arm_label}" if arm_label else "")
                arm_note = f" arm={arm_label}" if arm_label else ""
                if sequence_id in completed:
                    continue

                spec = SequenceSpec(
                    condition=condition,
                    sequence_id=sequence_id,
                    block_id=f"block-{block_index}",
                    initial_state=initial_state,
                    sequence_start_ts=start,
                    sequence_seed=seq_seed,
                    adaptation_enabled=True,
                )
                try:
                    result = run_sequence_with_uptime(
                        spec, arm_attacker, config, fault_rate=fault_rate, env_factory=env_factory
                    )
                except _EXCLUDABLE_FAULTS as exc:
                    reason = _fault_reason(exc)
                    if on_trial_failure == "halt":
                        raise RunHalted(sequence_id, reason) from exc
                    detail = _fault_detail(exc)
                    rec = {
                        "sequence_id": sequence_id,
                        "block_index": block_index,
                        "seq_i": seq_i,
                        "reason": reason,
                        "n_lost_attempts": config.attempts_per_sequence,
                        "detail": detail,
                    }
                    excluded_sequences.append(rec)
                    if checkpoint:
                        checkpoint.record_excluded_sequence(**rec)
                        if usage_snapshot is not None:
                            checkpoint.record_usage(usage_snapshot())
                    consecutive_exclusions += 1
                    in_condition = consecutive_by_condition.get(condition, 0) + 1
                    consecutive_by_condition[condition] = in_condition
                    if progress:
                        progress(
                            f"[seq {run_index}/{total}] condition={condition} block={block_index} "
                            f"seq={seq_i}{arm_note} EXCLUDED ({reason}: {detail[:160]}) "
                            f"elapsed={_hms(time.monotonic() - run_start)} "
                            f"excluded={len(excluded_sequences)}"
                        )
                    # Circuit breaker: back-to-back failures signal a systemic outage/bug, not
                    # the rare loss the exclusion buffer covers — halt so the operator resumes.
                    # Either kind trips it: N in a row overall, or N in a row of one condition.
                    if 0 < max_consecutive_exclusions <= consecutive_exclusions:
                        raise RunHalted(
                            sequence_id,
                            f"circuit_breaker:{consecutive_exclusions}_consecutive:{reason}",
                        ) from exc
                    if 0 < max_consecutive_exclusions <= in_condition:
                        raise RunHalted(
                            sequence_id,
                            f"circuit_breaker:{in_condition}_consecutive_in_{condition}:{reason}",
                        ) from exc
                    continue

                consecutive_exclusions = 0
                consecutive_by_condition[condition] = 0
                if checkpoint:
                    checkpoint.append_sequence(
                        sequence_id, block_index, seq_i, result.rows, result.uptime_events
                    )
                    if usage_snapshot is not None:
                        checkpoint.record_usage(usage_snapshot())
                else:
                    rows.extend(result.rows)
                    uptime_events.extend(result.uptime_events)
                if progress:
                    progress(
                        f"[seq {run_index}/{total}] condition={condition} block={block_index} "
                        f"seq={seq_i}{arm_note} rows={len(result.rows)} "
                        f"elapsed={_hms(time.monotonic() - run_start)} "
                        f"excluded={len(excluded_sequences)}"
                    )
    finally:
        # Whatever ends the loop (Ctrl-C, a halt, a crash), record this process's usage once
        # more, so the calls of a sequence that did not finish are counted too (the last
        # record of each process wins; cli._collection_provenance sums them).
        if checkpoint and usage_snapshot is not None:
            try:
                checkpoint.record_usage(usage_snapshot())
            except Exception:  # pragma: no cover - never mask the fault that ended the loop
                pass

    if checkpoint:
        rows, uptime_events, excluded_sequences = checkpoint.assemble()
    return ExperimentDataset(
        trial_rows=rows, uptime_events=uptime_events, excluded_sequences=excluded_sequences
    )


def run_repeatability(
    config: ExperimentConfig,
    attacker: Attacker,
    initial_state: ConfigState,
    base_start_ts: str = _BASE_START,
    env_factory: EnvFactory = OfflineEnvironment,
) -> list[TrialRow]:
    """Repeatability runs for the ICC: fixed config, adaptation disabled.

    Each trial is an independent single-attempt run on the same fixed
    configuration; attacker stochasticity varies per trial so the intraclass
    correlation can isolate measurement consistency from experimental variation.

    Like the main experiment, an unrecoverable API/infra fault (after all retries) on a
    trial does not crash the run — that trial is skipped (this short phase has no
    checkpoint; a subsequent resume simply re-runs it). Non-fault errors still propagate.
    """

    condition = config.repeatability.condition
    rows: list[TrialRow] = []
    for i in range(config.repeatability.n_trials):
        seq_seed = (config.randomization_seed * 7919 + i) & 0xFFFFFFFF
        start = _sequence_start(base_start_ts, i, 1, config.max_trial_duration_s)
        spec = SequenceSpec(
            condition=condition,
            sequence_id=f"rep-s{i}",
            block_id="block-rep",
            initial_state=initial_state,
            sequence_start_ts=start,
            sequence_seed=seq_seed,
            adaptation_enabled=config.repeatability.adaptation_enabled,
        )
        try:
            rows.extend(run_sequence(spec, attacker, config, n_attempts=1, env_factory=env_factory))
        except _EXCLUDABLE_FAULTS:
            continue  # skip this repeatability trial; the ICC tolerates fewer occasions
    return rows
