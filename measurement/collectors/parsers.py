"""JSONL/JSON fixture parsers → measurement models.

A trial directory contains:

- ``metadata.json``            (object) — TrialMetadata fields
- ``ground_truth.json``        (object) — {config_checksum, genuine[], decoys[]}
- ``recon_snapshots.jsonl``    — {attempt_index, timestamp, discovered[]}
- ``recon_commands.jsonl``     — {timestamp, command}
- ``exploit_deliveries.jsonl`` — {timestamp, target, reached_and_executed, target_genuine, unauthorized_access}
- ``monitoring.jsonl``         — {timestamp, cpu_pct, memory_pct, network_pct}
- ``workload.jsonl``           — {timestamp, answered_within_timeout, latency_ms?}
- ``reconfig_events.jsonl``    — {timestamp, technique}
- ``baseline_reference.json``  (object, optional)
- ``prev_recon_discovered.json`` (object, optional) — {discovered[]}

A service is encoded as ``[service, port, endpoint]`` (endpoint optional).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from measurement.models import (
    BaselineReference,
    ExploitDelivery,
    GroundTruthInventory,
    MonitoringSample,
    ReconCommand,
    ReconfigEvent,
    ReconSnapshot,
    ServiceTuple,
    TrialInputs,
    TrialMetadata,
    WorkloadSample,
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _svc(raw: list) -> ServiceTuple:
    """Decode ``[service, port, endpoint?, address?]`` into a ServiceTuple.

    A record written before the address became observable carries three elements and
    decodes with ``address=""`` -- honestly "not recorded", never a fabricated address.
    Both ``parse_ground_truth`` and the recon-snapshot parser go through ``_svc_set``, so
    a 3-field record decodes consistently on BOTH sides of a set comparison and an old
    fixture still matches itself. Left at three fields this function would silently
    revert the log-ingestion path to the very defect the address was added to fix.
    """

    service = raw[0]
    port = int(raw[1])
    endpoint = raw[2] if len(raw) > 2 else ""
    address = raw[3] if len(raw) > 3 else ""
    return ServiceTuple(service=service, port=port, endpoint=endpoint, address=address)


def _svc_set(rows: list[list]) -> frozenset[ServiceTuple]:
    return frozenset(_svc(r) for r in rows)


def parse_ground_truth(path: Path) -> GroundTruthInventory:
    d = _read_json(path)
    return GroundTruthInventory(
        config_checksum=d["config_checksum"],
        genuine=_svc_set(d.get("genuine", [])),
        decoys=_svc_set(d.get("decoys", [])),
    )


def parse_metadata(path: Path) -> TrialMetadata:
    d = _read_json(path)
    return TrialMetadata(**d)


def parse_recon_snapshots(path: Path) -> tuple[ReconSnapshot, ...]:
    return tuple(
        ReconSnapshot(
            attempt_index=int(r["attempt_index"]),
            timestamp=r["timestamp"],
            discovered=_svc_set(r.get("discovered", [])),
        )
        for r in _read_jsonl(path)
    )


def parse_recon_commands(path: Path) -> tuple[ReconCommand, ...]:
    return tuple(
        ReconCommand(timestamp=r["timestamp"], command=r["command"]) for r in _read_jsonl(path)
    )


def parse_exploit_deliveries(path: Path) -> tuple[ExploitDelivery, ...]:
    return tuple(
        ExploitDelivery(
            timestamp=r["timestamp"],
            target=_svc(r["target"]),
            reached_and_executed=bool(r["reached_and_executed"]),
            target_genuine=bool(r["target_genuine"]),
            unauthorized_access=bool(r["unauthorized_access"]),
        )
        for r in _read_jsonl(path)
    )


def parse_monitoring(path: Path) -> tuple[MonitoringSample, ...]:
    return tuple(
        MonitoringSample(
            timestamp=r["timestamp"],
            cpu_pct=float(r["cpu_pct"]),
            memory_pct=float(r["memory_pct"]),
            network_pct=float(r["network_pct"]),
        )
        for r in _read_jsonl(path)
    )


def parse_workload(path: Path) -> tuple[WorkloadSample, ...]:
    return tuple(
        WorkloadSample(
            timestamp=r["timestamp"],
            answered_within_timeout=bool(r["answered_within_timeout"]),
            latency_ms=(None if r.get("latency_ms") is None else float(r["latency_ms"])),
        )
        for r in _read_jsonl(path)
    )


def parse_reconfig_events(path: Path) -> tuple[ReconfigEvent, ...]:
    return tuple(
        ReconfigEvent(timestamp=r["timestamp"], technique=r["technique"]) for r in _read_jsonl(path)
    )


def parse_baseline_reference(path: Path) -> BaselineReference | None:
    if not path.exists():
        return None
    d = _read_json(path)
    return BaselineReference(
        vuln_profile_version=d["vuln_profile_version"],
        mean_response_ms=float(d["mean_response_ms"]),
        mean_cpu_pct=float(d["mean_cpu_pct"]),
        mean_memory_pct=float(d["mean_memory_pct"]),
        mean_network_pct=float(d["mean_network_pct"]),
    )


def parse_prev_recon(path: Path) -> frozenset[ServiceTuple] | None:
    if not path.exists():
        return None
    d = _read_json(path)
    return _svc_set(d.get("discovered", []))


def load_trial_inputs(trial_dir: str | Path) -> TrialInputs:
    """Assemble ``TrialInputs`` from a fixture trial directory."""

    d = Path(trial_dir)
    return TrialInputs(
        metadata=parse_metadata(d / "metadata.json"),
        ground_truth=parse_ground_truth(d / "ground_truth.json"),
        recon_snapshots=parse_recon_snapshots(d / "recon_snapshots.jsonl"),
        recon_commands=parse_recon_commands(d / "recon_commands.jsonl"),
        exploit_deliveries=parse_exploit_deliveries(d / "exploit_deliveries.jsonl"),
        monitoring_samples=parse_monitoring(d / "monitoring.jsonl"),
        workload_samples=parse_workload(d / "workload.jsonl"),
        reconfig_events=parse_reconfig_events(d / "reconfig_events.jsonl"),
        baseline_reference=parse_baseline_reference(d / "baseline_reference.json"),
        prev_recon_discovered=parse_prev_recon(d / "prev_recon_discovered.json"),
    )
