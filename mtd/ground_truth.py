"""Ground-truth inventory generator.

Produces the genuine (non-deceptive) and decoy service–port–endpoint sets for the
*current* configuration state. The orchestrator regenerates this on **every**
configuration change so reconnaissance accuracy is always scored against the
inventory in force at the time of the attacker's recon pass.
"""

from __future__ import annotations

from measurement.models import GroundTruthInventory
from mtd.checksums import state_checksum
from mtd.models import ConfigState


def generate_ground_truth(state: ConfigState) -> GroundTruthInventory:
    """Derive the ground-truth inventory from a configuration state.

    Genuine services form the recall set; decoys are the deceptive set that
    counts against precision when "discovered". Keyed by ``state_checksum`` so the
    inventory is tied to the exact state it describes.
    """

    genuine = frozenset(s.observable() for s in state.genuine())
    decoys = frozenset(s.observable() for s in state.decoys())
    return GroundTruthInventory(
        config_checksum=state_checksum(state),
        genuine=genuine,
        decoys=decoys,
    )
