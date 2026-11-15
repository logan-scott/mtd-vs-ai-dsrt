"""Control-host isolation preflight — the runtime egress-split guard for ``--live``.

The management path is already provisioned (``infra/control.tf`` dual-homes the
control host on ``vmbr9`` + the LAN leg). What remains is a runtime guard that
refuses to start a live run unless the containment invariants actually hold on the
built topology. This module is that guard; it is wired into ``--live`` startup and
is also runnable on its own so an operator can check isolation before a run:

    python -m harness.preflight --config harness/experiment_config.yaml           # real probes
    python -m harness.preflight --config harness/experiment_config.yaml --dry-run  # on-paper

It asserts, fail-closed (raising ``SafetyError``, which aborts the run):

  1. a ``testbed_cidr`` allowlist is configured;
  2. every non-control testbed node has **no route off the testbed** (cannot reach
     the model API or any generic internet host);
  3. the control host **reaches the model API** (the planner needs it);
  4. the control host has **IP forwarding disabled** (``net.ipv4.ip_forward = 0``),
     so it never routes the testbed to the internet; and
  5. every non-control testbed node has **no default route** (a structural belt-and-
     suspenders check on top of the reachability probe).

Checks 2–4 are the core egress split (``harness.safety.assert_egress_split``); this
module adds the CIDR gate and the no-default-route check and renders a report.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from harness.config import ExperimentConfig, LiveSettings
from harness.live_runner import egress_map
from harness.safety import (
    EgressMap,
    EgressViolation,
    RouteChecker,
    SafetyError,
    assert_egress_split,
    parse_cidr,
)


@dataclass(frozen=True)
class IsolationReport:
    """The checks that passed (the preflight raises on the first violation)."""

    ok: bool
    egress_map: EgressMap
    checks: list[dict]


def run_isolation_preflight(settings: LiveSettings, checker: RouteChecker) -> IsolationReport:
    """Assert the containment invariants for a live run; raise ``SafetyError`` on any.

    Returns an ``IsolationReport`` of what passed. This is the function ``--live``
    startup calls (with the real ``SshRouteChecker``) before any offensive action.
    """

    parse_cidr(settings.testbed_cidr)  # ConfigurationError if unset/blank
    emap = egress_map(settings)

    # Core egress split: node egress-lessness, planner reaches API, control ip_forward=0.
    assert_egress_split(emap, checker)

    # Structural belt-and-suspenders: non-control nodes must have no default route.
    has_default_route = getattr(checker, "has_default_route", None)
    if callable(has_default_route):
        for node in emap.testbed_nodes:
            if has_default_route(node):
                raise EgressViolation(
                    f"testbed node {node} has an IPv4 default route — a non-control "
                    "testbed node must have none (its isolation is structural)"
                )

    # A node must not still resolve external DNS (a residual provisioning-DNS leak).
    resolves_external_dns = getattr(checker, "resolves_external_dns", None)
    if callable(resolves_external_dns):
        for node in emap.testbed_nodes:
            if resolves_external_dns(node):
                raise EgressViolation(
                    f"testbed node {node} can still resolve external DNS — the "
                    "provisioning resolver was not reverted (must be sealed before a trial)"
                )

    checks = [
        {"name": "testbed_cidr allowlist configured", "detail": settings.testbed_cidr},
        {
            "name": "non-control nodes have no off-testbed egress",
            "detail": ", ".join(emap.testbed_nodes),
        },
        {"name": "control host reaches the model API", "detail": emap.control_host},
        {"name": "control host IP forwarding disabled", "detail": "net.ipv4.ip_forward = 0"},
        {
            "name": "non-control nodes have no default route",
            "detail": (
                "checked" if callable(has_default_route) else "skipped (checker cannot probe)"
            ),
        },
        {
            "name": "non-control nodes cannot resolve external DNS",
            "detail": (
                "checked" if callable(resolves_external_dns) else "skipped (checker cannot probe)"
            ),
        },
    ]
    return IsolationReport(ok=True, egress_map=emap, checks=checks)


def format_report(report: IsolationReport) -> str:
    lines = ["ISOLATION PREFLIGHT — PASS (fail-closed)"]
    for check in report.checks:
        lines.append(f"  [ok] {check['name']}: {check['detail']}")
    lines.append("egress map:")
    for host, spec in report.egress_map.as_dict().items():
        lines.append(f"  {host} [{spec['role']}]")
        lines.append(f"    may reach:      {', '.join(spec['may_reach'])}")
        lines.append(f"    must not reach: {', '.join(spec['must_not_reach'])}")
    return "\n".join(lines)


def build_checker(settings: LiveSettings, dry_run: bool) -> RouteChecker:
    if dry_run:
        from harness.live_runner import PlannedEgressChecker

        return PlannedEgressChecker(settings)
    from harness.live_runner import build_remote_runner
    from harness.safety import SshRouteChecker

    return SshRouteChecker(
        build_remote_runner(settings), dns_check_host=settings.provisioning_dns_check_host
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Control-host isolation preflight for --live")
    parser.add_argument("--config", default="harness/experiment_config.example.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="validate the intended egress map on paper (no live probing)",
    )
    args = parser.parse_args(argv)

    settings = ExperimentConfig.load(args.config).live
    checker = build_checker(settings, args.dry_run)
    try:
        report = run_isolation_preflight(settings, checker)
    except SafetyError as exc:
        print(f"ABORT (isolation preflight): {exc}")
        return 2
    print(format_report(report))
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry
    raise SystemExit(main())
