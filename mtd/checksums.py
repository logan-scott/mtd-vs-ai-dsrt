"""Deterministic configuration checksums.

Two checksums serve different purposes:

- ``profile_checksum`` — the **baseline-equivalence** checksum. It hashes only the
  genuine services' stable identity and vulnerability profile (``service_id``,
  ``vuln_id``, ``software_version``), which are held constant across all six
  conditions. MTD mutation of observable attributes does **not** change it, so it
  can be verified before every trial set to confirm the standardized vulnerability
  profile is intact (control variable). This is the value recorded as
  ``testbed_config_checksum`` on each trial row.
- ``state_checksum`` — a fingerprint of the **full observable state** (all
  services, all attributes). It changes on every reconfiguration and tags each
  ground-truth inventory snapshot to the exact configuration state it describes.

Both are pure and deterministic (identical states → identical checksums).
"""

from __future__ import annotations

import hashlib
import json

from mtd.models import ConfigState


def _digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def profile_checksum(state: ConfigState) -> str:
    """Baseline-equivalence checksum over the standardized vulnerability profile.

    Invariant under MTD mutation of observable attributes (ip/port/endpoint/
    service_name). Two states with the same genuine services + vulnerabilities
    produce the same checksum regardless of their current mutation.
    """

    profile = sorted((s.service_id, s.vuln_id, s.software_version) for s in state.genuine())
    return _digest({"version": state.version, "profile": profile})


def state_checksum(state: ConfigState) -> str:
    """Fingerprint of the full observable configuration state (mutates each change)."""

    observable = sorted(
        (s.service_id, s.service_name, s.ip, s.port, s.endpoint, s.genuine) for s in state.services
    )
    return _digest({"version": state.version, "state": observable})
