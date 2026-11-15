"""Live attacker environment backed by the real testbed.

The offline ``OfflineEnvironment`` answers ``ground_truth_at`` from the orchestrator
and lets ``OfflineToolExecutor`` resolve actions against it. The live environment
answers ``ground_truth_at`` from the *same* orchestrator (ground truth is a
provisioning fact — what the instrument stood up) and additionally carries the
``LiveTargetExecutor`` that resolves the attacker's recon/exploit actions against
real services on the testbed subnet.

Ground truth comes from the orchestrator in both modes because it defines the
experimental condition; only the *executor* differs (mock target vs. real
services). This is what lets the identical ``verify_objectives`` scoring apply to a
live trial — see ``harness.live_contract``.
"""

from __future__ import annotations

from dataclasses import dataclass

from attacker.tools import TargetExecutor
from measurement.models import GroundTruthInventory
from mtd.orchestrator import OrchestrationResult


@dataclass(frozen=True)
class LiveEnvironment:
    """Wraps one trial's orchestration + its live executor as a probeable environment.

    Implements the ``attacker.base.Environment`` protocol (``ground_truth_at``). The
    live executor is carried on ``.executor`` so the harness's live executor factory
    can hand it to the planner (``lambda env: env.executor``).
    """

    orchestration: OrchestrationResult
    executor: TargetExecutor

    def ground_truth_at(self, timestamp: str) -> GroundTruthInventory:
        return self.orchestration.ground_truth_at(timestamp)
