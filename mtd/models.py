"""Configuration-state model for the testbed.

A ``ConfigState`` is the current configuration of the testbed: a set of services,
each with a **stable identity** (``service_id``) and **vulnerability profile**
(``vuln_id``, ``software_version``) that are invariant across MTD conditions, plus
**mutable observable attributes** (``service_name`` fingerprint, ``ip``, ``port``,
``endpoint``) that MTD techniques reconfigure.

Two derived views matter:

- The **vulnerability profile** (genuine services' stable identity + vulnerability)
  must be identical across all six conditions — verified by ``profile_checksum``.
- The **observable state** (what the attacker can fingerprint) is a set of
  service–port–endpoint tuples — the basis of the ground-truth inventory.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from measurement.models import ServiceTuple


@dataclass(frozen=True)
class ServiceConfig:
    """One service in the testbed configuration.

    ``service_id`` is a stable internal identity that persists across mutations
    (so a service can be tracked as its observable attributes change).
    ``genuine`` distinguishes real targets from decoys. For decoys, ``vuln_id``
    is empty (they expose no real vulnerability).
    """

    service_id: str
    service_name: str  # observable fingerprint (mutable, e.g., "http-dvwa")
    # Observable AND part of ``ServiceTuple`` (mutated by network shuffling). It was
    # not in the tuple until 2026-08; while it sat outside, network shuffling mutated
    # this attribute and changed nothing the study measured — see
    # ``docs/null_condition_audit.md``. Modeled address only: the live transport
    # delivers to the frozen physical host, never to this value.
    ip: str
    port: int  # observable (mutated by port rotation)
    endpoint: str  # observable (mutated by endpoint mutation)
    genuine: bool
    vuln_id: str = ""  # stable vulnerability identity (genuine only)
    software_version: str = ""  # stable (part of the standardized profile)

    def observable(self) -> ServiceTuple:
        """The attacker-observable identity: service–port–endpoint–address tuple.

        All four mutable observable attributes are carried, so every MTD technique
        mutates something the ground-truth inventory (and therefore recon accuracy and
        attack-path churn) can actually see.
        """

        return ServiceTuple(self.service_name, self.port, self.endpoint, self.ip)

    def with_attrs(self, **changes) -> ServiceConfig:
        """Return a copy with the given attributes replaced (immutably)."""

        return replace(self, **changes)


@dataclass(frozen=True)
class ConfigState:
    """The full testbed configuration at a point in time."""

    version: str  # vulnerability-profile version tag (control variable)
    services: tuple[ServiceConfig, ...]

    def genuine(self) -> tuple[ServiceConfig, ...]:
        return tuple(s for s in self.services if s.genuine)

    def decoys(self) -> tuple[ServiceConfig, ...]:
        return tuple(s for s in self.services if not s.genuine)

    def replace_service(self, service_id: str, new: ServiceConfig) -> ConfigState:
        """Return a new state with one service swapped by ``service_id``."""

        services = tuple(new if s.service_id == service_id else s for s in self.services)
        return ConfigState(version=self.version, services=services)

    def with_services(self, services: tuple[ServiceConfig, ...]) -> ConfigState:
        return ConfigState(version=self.version, services=services)
