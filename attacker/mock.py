"""Deterministic mock attacker for offline dataset generation.

A stand-in for the LLM-driven attacker (``attacker.planner``): it produces realistic recon and
exploit observations without any network access or model calls, and it **adapts**
across a sequence's attempts via the retained ``AttackerContext`` — discovery and
exploitation improve, and it learns to ignore decoys. Against a static baseline
(ground truth never changes) it converges to full compromise; against MTD it is
repeatedly forced to re-derive its target picture. All behavior is a pure function
of the seed, so datasets are reproducible.

This is an offline instrument-validation tool, not an attack capability: it calls
nothing and only reasons over the ground-truth inventory the environment exposes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from attacker.base import (
    DISPOSITION_EXECUTED,
    DISPOSITION_SKIPPED_DURATION_LIMIT,
    KIND_EXPLOIT,
    KIND_RECON,
    TERMINAL_ACTION_BUDGET_EXHAUSTED,
    TERMINAL_DURATION_LIMIT,
    TERMINAL_TARGETS_EXHAUSTED,
    ActionTraceEntry,
    AttemptObservation,
    AttemptRequest,
)
from measurement import clock
from measurement.models import ExploitDelivery, ReconCommand, ReconSnapshot

_MIX = 2654435761  # Knuth multiplicative hash constant for per-attempt seeding


@dataclass(frozen=True)
class MockAttacker:
    """A config-driven, adaptive, deterministic offline attacker."""

    model_id: str = "mock-attacker"
    model_version: str = "0.1.0"
    base_discovery_prob: float = 0.5
    discovery_gain: float = 0.10
    max_discovery_prob: float = 0.95
    base_exploit_prob: float = 0.30
    exploit_gain: float = 0.12
    max_exploit_prob: float = 0.90
    base_decoy_confusion: float = 0.40
    decoy_learn_rate: float = 0.10
    disruption_weight: float = 0.6
    base_recon_s: float = 45.0
    recon_delay_weight: float = 6.0
    recon_noise_s: float = 15.0
    recon_offset_s: int = 5
    delivery_interval_s: int = 10

    def attempt(self, req: AttemptRequest) -> AttemptObservation:
        rng = random.Random((req.seed ^ (req.attempt_index * _MIX)) & 0xFFFFFFFF)
        ctx = req.context
        adapt = ctx.attempts_seen if ctx.adaptation_enabled else 0
        limit = float(req.trial_duration_limit_s)

        # How much of the attacker's retained knowledge has gone stale (a proxy for
        # how disruptive the current MTD condition is): 0 under a static baseline,
        # high when mutation has invalidated the observable tuples it learned. Measured
        # against the state at window start (before this window's mutations).
        gt0 = req.environment.ground_truth_at(req.window_start)
        stale_fraction = (
            1.0 - len(ctx.known_genuine & gt0.genuine) / len(ctx.known_genuine)
            if ctx.known_genuine
            else 0.0
        )
        disruption = self.disruption_weight * stale_fraction

        # Reconnaissance takes longer under disruption — the attacker must re-derive a
        # moving target picture. This delays enumeration (obj_a) and thus TTC, and can
        # exceed the trial window (enumeration fails -> right-censored trial).
        enum_time = max(
            self.base_recon_s * (1.0 + self.recon_delay_weight * disruption)
            + rng.uniform(-self.recon_noise_s, self.recon_noise_s),
            1.0,
        )
        enumerated_in_time = enum_time <= limit
        recon_ts = clock.shift(req.window_start, min(enum_time, limit))

        gt = req.environment.ground_truth_at(recon_ts)
        genuine = gt.genuine
        decoys = gt.decoys
        known_valid = ctx.known_genuine & genuine

        discovery_prob = min(
            self.base_discovery_prob + self.discovery_gain * adapt, self.max_discovery_prob
        ) * (1.0 - disruption)
        exploit_prob = min(
            self.base_exploit_prob + self.exploit_gain * adapt, self.max_exploit_prob
        ) * (1.0 - disruption)
        decoy_confusion = max(self.base_decoy_confusion - self.decoy_learn_rate * adapt, 0.0)

        # The attacker reports everything it *believes* is genuine: newly
        # discovered current services plus everything retained from prior attempts.
        # Under MTD, retained knowledge goes stale (endpoints/ports mutated), so the
        # report accrues false positives — the modeled cost of acting on old intel.
        # If recon never completed within the window, nothing new is discovered.
        newly = (
            frozenset(g for g in sorted(genuine) if rng.random() < discovery_prob)
            if enumerated_in_time
            else frozenset()
        )
        current_confirmed = known_valid | newly
        reported_genuine = ctx.known_genuine | newly

        # Decoys the attacker is fooled by this attempt (skips ones already learned).
        discovered_decoys = frozenset(
            d
            for d in sorted(decoys)
            if d not in ctx.known_decoys and rng.random() < decoy_confusion
        )

        discovered = reported_genuine | discovered_decoys

        recon_commands = (
            ReconCommand(clock.shift(req.window_start, self.recon_offset_s), "recon --scan"),
        )
        recon_snapshots = (ReconSnapshot(req.attempt_index, recon_ts, discovered),)

        # Exploit deliveries against everything discovered, capped by max_actions.
        # Current genuine -> reachable (may succeed); decoys -> reached but futile;
        # stale believed-genuine -> unreachable (a wasted, invalid attempt).
        budget = max(req.max_actions - len(recon_commands), 0)
        deliveries: list[ExploitDelivery] = []

        # Instrumentation only: a faithful trace of THIS policy, not the planner's.
        # The recon above is unconditional (it is emitted even when ``max_actions`` is
        # 0 or less — a pre-existing property of the mock, reported rather than hidden),
        # so it is always entry 0 and always executed; whether it enumerated inside the
        # window is an outcome, already measured by obj_a/TTEN, not an execution fact.
        # The reason below is the run-to-completion case and is overwritten by the
        # duration break: the mock is either cut off by the budget slice or it simply
        # runs out of discovered targets — a terminal condition the planner cannot have.
        trace: list[ActionTraceEntry] = [
            ActionTraceEntry(ordinal=0, kind=KIND_RECON, disposition=DISPOSITION_EXECUTED)
        ]
        terminal_reason = (
            TERMINAL_ACTION_BUDGET_EXHAUSTED
            if len(discovered) > budget
            else TERMINAL_TARGETS_EXHAUSTED
        )

        for i, tgt in enumerate(sorted(discovered)[:budget]):
            ts = clock.shift(recon_ts, (i + 1) * self.delivery_interval_s)
            if clock.elapsed_seconds(req.window_start, ts) > req.trial_duration_limit_s:
                trace.append(
                    ActionTraceEntry(
                        ordinal=i + 1,
                        kind=KIND_EXPLOIT,
                        disposition=DISPOSITION_SKIPPED_DURATION_LIMIT,
                        target=tgt,
                    )
                )
                terminal_reason = TERMINAL_DURATION_LIMIT
                break
            trace.append(
                ActionTraceEntry(
                    ordinal=i + 1,
                    kind=KIND_EXPLOIT,
                    disposition=DISPOSITION_EXECUTED,
                    target=tgt,
                )
            )
            if tgt in genuine:
                reached, is_genuine = True, True
                unauthorized = rng.random() < exploit_prob
            elif tgt in decoys:
                reached, is_genuine, unauthorized = True, False, False
            else:  # stale believed-genuine: nothing there to reach
                reached, is_genuine, unauthorized = False, False, False
            deliveries.append(
                ExploitDelivery(
                    timestamp=ts,
                    target=tgt,
                    reached_and_executed=reached,
                    target_genuine=is_genuine,
                    unauthorized_access=unauthorized,
                )
            )

        return AttemptObservation(
            recon_commands=recon_commands,
            recon_snapshots=recon_snapshots,
            exploit_deliveries=tuple(deliveries),
            discovered=discovered,
            genuine_hits=current_confirmed,
            decoy_hits=discovered_decoys,
            action_trace=tuple(trace),
            terminal_reason=terminal_reason,
        )
