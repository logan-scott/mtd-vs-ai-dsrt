"""MTD layer: config-state model, checksums, ground-truth generator, techniques,
and the reconfiguration orchestrator.

This layer is deterministic and offline-testable: given an initial configuration
state and a seed, it produces the reconfiguration event log and ground-truth
inventory snapshots that the measurement instrument consumes — with no live
infrastructure. The same technique and orchestrator logic drives the live testbed.
"""
