"""Offline attacker environment backed by the MTD orchestrator.

Implements the ``Environment`` protocol by delegating to an ``OrchestrationResult``:
the in-force ground truth at a timestamp is the latest snapshot at or before it.
The live environment (``harness.live_environment``) implements the same protocol via real
scanning.
"""

from __future__ import annotations

from dataclasses import dataclass

from measurement.models import GroundTruthInventory
from mtd.orchestrator import OrchestrationResult


@dataclass(frozen=True)
class OfflineEnvironment:
    """Wraps one trial's orchestration output as a probeable environment."""

    orchestration: OrchestrationResult

    def ground_truth_at(self, timestamp: str) -> GroundTruthInventory:
        return self.orchestration.ground_truth_at(timestamp)
