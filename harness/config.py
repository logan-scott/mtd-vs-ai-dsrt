"""Declarative experiment configuration loader.

Parses ``experiment_config.yaml`` (schema in
``harness/experiment_config.example.yaml``) into typed objects. Nothing in the
harness is hardcoded — every experiment parameter flows from here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from attacker.base import CLOCK_MODELED, normalize_clock_mode, normalize_tool_choice


def _optional_float(d: dict, key: str, default: float) -> float | None:
    """Parse a numeric field that may be explicitly nulled to mean "do not send this".

    Three cases must stay distinguishable, because they are three different requests:

    - key ABSENT          -> ``default`` (existing configs keep their current behavior)
    - key present, a number -> that number, parsed exactly as before
    - key present, ``null`` -> ``None``, meaning the caller omits the parameter entirely

    ``d.get(key, default)`` alone cannot express this: it collapses an explicit ``null``
    onto the absent case, and ``float(None)`` would raise. The distinction matters because
    ``None`` is not a value we send -- it is the instruction NOT to send one.
    """

    if key not in d:
        return default
    raw = d[key]
    return None if raw is None else float(raw)


def _strict_bool(d: dict, key: str, default: bool) -> bool:
    """Parse a YAML boolean, refusing anything that is not one.

    ``bool(d.get(key))`` would read the STRING ``"false"`` as True; a switch that changes what
    a run records must fail loudly on a malformed value instead.
    """

    if key not in d or d[key] is None:
        return default
    raw = d[key]
    if isinstance(raw, bool):
        return raw
    raise ValueError(f"{key} must be true or false, got {raw!r}")


#: What the live reconnaissance scan targets (``live.scan_scope``). ``subnet`` (the default, and
#: the scope of every archived run) sweeps the whole ``testbed_cidr`` with ``-Pn``, so most of
#: its ~27 s is nmap waiting on unused addresses; ``target_host`` scans only ``targets_host`` --
#: an attacker who already knows where the services live (about a second). The scope sets how
#: old the attacker's intelligence is when it strikes, so it is run-defining.
SCAN_SCOPE_SUBNET = "subnet"
SCAN_SCOPE_TARGET_HOST = "target_host"
SCAN_SCOPES: tuple[str, ...] = (SCAN_SCOPE_SUBNET, SCAN_SCOPE_TARGET_HOST)


def normalize_scan_scope(value: object) -> str:
    """Validate ``live.scan_scope`` (absent/None == ``subnet``); anything else raises."""

    if value is None:
        return SCAN_SCOPE_SUBNET
    scope = str(value).strip().lower()
    if scope not in SCAN_SCOPES:
        raise ValueError(f"live.scan_scope must be one of {SCAN_SCOPES}, got {value!r}")
    return scope


#: Whether a live attempt may start, and be scored, against an incomplete testbed surface
#: (``live.surface_check``). ``off`` (the default, and every archived run) scores whatever the
#: scan finds. ``strict`` makes the per-attempt provisioner WAIT until every in-force service
#: answers on its published port (up to ``live.readiness_timeout_s``) before the attacker starts,
#: and makes a scan that cannot reach an in-force service an infrastructure fault instead of an
#: observation. Under Method C every in-force service is physically up by design, so a missing
#: one is a testbed failure (a container still starting after a port change, or a crash), never
#: an MTD effect. Run-defining.
SURFACE_CHECK_OFF = "off"
SURFACE_CHECK_STRICT = "strict"
SURFACE_CHECKS: tuple[str, ...] = (SURFACE_CHECK_OFF, SURFACE_CHECK_STRICT)


def normalize_surface_check(value: object) -> str:
    """Validate ``live.surface_check`` (absent/None == ``off``); anything else raises."""

    if value is None:
        return SURFACE_CHECK_OFF
    mode = str(value).strip().lower()
    if mode not in SURFACE_CHECKS:
        raise ValueError(f"live.surface_check must be one of {SURFACE_CHECKS}, got {value!r}")
    return mode


#: The order the full run executes its (condition, sequence) units in (``design.run_order``).
#: ``blocked`` (the default, and the order of every archived run) runs each condition as one
#: contiguous block, block order randomized once from the master seed -- so condition and
#: execution period are confounded. ``interleaved`` runs in ROUNDS: round r runs sequence r of
#: every condition, in an order shuffled per round from the master seed, so every condition is
#: spread evenly over the run. Seeds per (condition, sequence) are identical under both orders.
RUN_ORDER_BLOCKED = "blocked"
RUN_ORDER_INTERLEAVED = "interleaved"
RUN_ORDERS: tuple[str, ...] = (RUN_ORDER_BLOCKED, RUN_ORDER_INTERLEAVED)


def normalize_run_order(value: object) -> str:
    """Validate ``design.run_order`` (absent/None == ``blocked``); anything else raises."""

    if value is None:
        return RUN_ORDER_BLOCKED
    order = str(value).strip().lower()
    if order not in RUN_ORDERS:
        raise ValueError(f"design.run_order must be one of {RUN_ORDERS}, got {value!r}")
    return order


@dataclass(frozen=True)
class LiveSettings:
    """Live-testbed settings + the fail-closed egress split (see harness.safety).

    Offline runs ignore these. Live runs require ``testbed_cidr`` (the allowlist);
    ``control_host`` runs the LLM planner (model-API egress) and ``executor_node``
    runs the offensive actions from the no-egress testbed node. ``targets_host`` is
    where the genuine + decoy services are provisioned (the scan matches ports
    there); ``testbed_nodes`` is every non-control node the egress preflight probes
    for isolation. Both default (blank/empty) to the infra convention derived from
    ``testbed_cidr`` — see ``harness.live_runner.resolve_targets_host`` /
    ``resolve_testbed_nodes``. All host values must be SSH-reachable from the
    control host (use IPs, or add host entries — the isolated net has no DNS).
    """

    enabled: bool = False
    # Operational-cost telemetry in a LIVE run. ``harness.telemetry`` is an OFFLINE synthetic
    # stand-in for real monitoring/workload collectors; a live run must not silently emit
    # synthetic cost. Left False (the default), a live run records the five synthetic cost
    # columns as not-measured (null, ``operational_cost_source="not_measured"``) and
    # reconfiguration cost is benchmarked separately at the technique level
    # (``harness.reconfig_benchmark``; see docs/metrics_specification.md §3). Set True ONLY to
    # deliberately keep the synthetic stand-in in a live run, in which case the rows are
    # labeled ``operational_cost_source="synthetic"`` -- honestly, not as a measurement.
    synthetic_telemetry: bool = False
    testbed_cidr: str = ""
    control_host: str = "control-host"
    executor_node: str = "mtd-attacker"
    targets_host: str = ""
    testbed_nodes: tuple[str, ...] = ()
    # Per-host admin SSH port overrides (host, port). The targets node's admin sshd
    # is moved to 2222 so the SSH decoy can own 22 (default derived in live_runner).
    node_ssh_ports: tuple[tuple[str, int], ...] = ()
    ssh_user: str = "labadmin"
    # Provisioning-only DNS: during the temporary egress window the nodes' resolver
    # is pointed at `provisioning_dns` on `provisioning_iface`, verified by resolving
    # `provisioning_dns_check_host`, and reverted when the window closes. Trial-phase
    # sealing is unaffected (reverted before any trial). `provisioning_dns` blank ("")
    # means auto-detect the control host's own upstream resolver (via `resolvectl
    # status`) — a hardcoded public IP like 1.1.1.1 fails on networks that block
    # external port 53. Set it to override the auto-detected value.
    provisioning_dns: str = ""
    provisioning_iface: str = "eth0"
    provisioning_dns_check_host: str = "archive.ubuntu.com"
    # What the reconnaissance scan targets (``SCAN_SCOPES``). Run-defining when not the default.
    scan_scope: str = SCAN_SCOPE_SUBNET
    # Whether the attacker may meet an incomplete surface (``SURFACE_CHECKS``), and how long the
    # provisioner waits for every in-force service to answer under ``strict``.
    surface_check: str = SURFACE_CHECK_OFF
    readiness_timeout_s: float = 300.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "scan_scope", normalize_scan_scope(self.scan_scope))
        object.__setattr__(self, "surface_check", normalize_surface_check(self.surface_check))
        if not float(self.readiness_timeout_s) > 0:
            raise ValueError(
                f"live.readiness_timeout_s must be > 0, got {self.readiness_timeout_s!r}"
            )


@dataclass(frozen=True)
class AttackerParams:
    backend: str = "mock"
    model_id: str = "mock-attacker"
    model_version: str = "0.1.0"
    # None == omit the sampling parameter from the request entirely, letting the model
    # apply its own default. Claude 4.7+ (e.g. claude-sonnet-5) REJECTS any explicitly
    # supplied temperature/top_p/top_k with a 400, so such a model can only be driven
    # with this unset. Absent in YAML still means 0.0, so existing configs are untouched.
    temperature: float | None = 0.0
    # None == omit ``tool_choice`` from the request entirely: the API default, and what
    # every archived run under docs/results/ sent. "any" forces the model to call a tool,
    # "auto" sends the API default explicitly. An EXPERIMENTAL MANIPULATION, not an
    # operational knob (see attacker.base): whether forcing a tool call changes the
    # attacker's behavior is a question to be measured, so the default must stay the
    # omitted control condition and an unrecognized value must raise, never degrade to it.
    tool_choice: str | None = None
    context_window: int = 200000
    system_prompt_path: str = ""
    tool_access: tuple[str, ...] = ()
    # Resilience (operational; never affects sampled outputs). SDK-level retry budget +
    # request timeout, and an app-level backoff layer on top — see attacker.backends.
    max_retries: int = 8
    request_timeout_s: float = 120.0
    api_retry_attempts: int = 6
    # The attempt clock (see ``measurement.timing`` / metrics spec §4f). "modeled" (absent ==
    # modeled) charges the planner's fixed per-action quanta and is the clock of every archived
    # run; "wall" charges the real elapsed time of each planner call, scan and delivery. An
    # unrecognized value raises at load time. Part of the checkpoint fingerprint when it is not
    # the default, so a run never resumes across a change of clock.
    clock: str = CLOCK_MODELED
    # Record the attempt's real action timing under the modeled clock too (always on under
    # "wall"). Instrumentation only -- it changes no outcome -- but real durations differ run
    # to run, so it is off by default to keep the offline pipeline byte-deterministic.
    # Recommended ON for every live run: it yields the paired modeled-vs-wall staleness
    # counterfactual on every delivery at no cost.
    record_timing: bool = False


@dataclass(frozen=True)
class RunParams:
    """Full-run operational policy (does not affect the science)."""

    # 'exclude': a sequence lost to an unrecoverable API/infra fault is recorded as an
    # exclusion and skipped (the ~15% exclusion buffer covers it), the run continues.
    # 'halt': checkpoint and exit non-zero so the operator resumes after the incident.
    on_trial_failure: str = "exclude"
    # Circuit breaker: even under 'exclude', halt if this many sequences fail back-to-back
    # (a systemic outage/bug, not the rare loss the buffer covers). 0 disables it.
    max_consecutive_exclusions: int = 5
    # Rate of the SIMULATED row-level fault injector when ``--fault-rate`` is not given on the
    # command line. None (key absent) keeps the CLI's historical default of 0.03; a study run
    # sets 0.0 so that every exclusion it records is a real fault (C278).
    fault_rate: float | None = None


@dataclass(frozen=True)
class WorkloadParams:
    legitimate_rate_rps: float = 1.0
    availability_timeout_s: int = 2


@dataclass(frozen=True)
class PilotMode:
    conditions: tuple[str, ...] = ("static_baseline", "endpoint_mutation")
    sequences_per_condition: int = 2


@dataclass(frozen=True)
class RepeatabilityMode:
    condition: str = "static_baseline"
    adaptation_enabled: bool = False
    n_trials: int = 15


#: The two arms of the clock-validation pilot, in their base (even-sequence) order.
CLOCK_VALIDATION_ARMS: tuple[str, ...] = ("modeled", "wall")

_ALL_CONDITIONS: tuple[str, ...] = (
    "static_baseline",
    "network_shuffling",
    "port_rotation",
    "endpoint_mutation",
    "deceptive_presentation",
    "combined_mtd_deception",
)


#: The conditions whose technique moves a GENUINE service on every tick -- the race the attacker
#: clock decides. The pilot's default design; ``analysis.clock_validation`` restates the same
#: set (pinned equal by test).
_RACE_SENSITIVE: tuple[str, ...] = (
    "network_shuffling",
    "port_rotation",
    "endpoint_mutation",
    "combined_mtd_deception",
)

#: The keys ``modes.clock_validation.criteria`` accepts. Anything else raises: a misspelled
#: decision criterion must never silently fall back to its default.
CLOCK_VALIDATION_CRITERIA_KEYS: tuple[str, ...] = (
    "equivalence_margin",
    "min_first_genuine_deliveries",
    "min_pairs_per_condition",
)


@dataclass(frozen=True)
class ClockValidationMode:
    """The real-timing validation pilot (``cli.py validate-clock``).

    Every (condition, sequence) runs TWICE with the same seed -- once per clock arm, in
    alternating order -- so both arms face the identical MTD schedule and phase. The default
    design is the four race-sensitive conditions x 6 sequences (x 8 attempts x 2 arms = 384
    trials); ``static_baseline`` and ``deceptive_presentation`` can be added but never enter
    the primary statistic (no genuine service of theirs ever goes stale). The criteria are
    fixed BEFORE the run: ``validate-clock`` locks them into the checkpoint manifest at the
    first start and refuses to resume under different ones (docs/clock_validation_runbook.md).
    """

    conditions: tuple[str, ...] = _RACE_SENSITIVE
    sequences_per_condition: int = 6
    # Equivalence margin (absolute) on the paired difference in the first genuine delivery's
    # stale probability, wall minus modeled.
    equivalence_margin: float = 0.10
    # The verdict is "insufficient_data" with fewer first genuine deliveries than this in the
    # race-sensitive conditions, or fewer pairs (independent sequences) than
    # ``min_pairs_per_condition`` in any race-sensitive condition present.
    min_first_genuine_deliveries: int = 30
    min_pairs_per_condition: int = 4

    def __post_init__(self) -> None:
        unknown = [c for c in self.conditions if c not in _ALL_CONDITIONS]
        if unknown or not self.conditions:
            raise ValueError(
                f"modes.clock_validation.conditions must be a non-empty subset of "
                f"{_ALL_CONDITIONS}, got {self.conditions!r}"
            )
        if len(set(self.conditions)) != len(self.conditions):
            # A repeated condition would collide on sequence and trial ids and silently run
            # (and keep) only half of what was announced.
            raise ValueError(
                f"modes.clock_validation.conditions has duplicates: {self.conditions!r}"
            )
        if self.sequences_per_condition < 1:
            raise ValueError("modes.clock_validation.sequences_per_condition must be >= 1")
        if not 0.0 < self.equivalence_margin < 1.0:
            raise ValueError("modes.clock_validation.criteria.equivalence_margin must be in (0, 1)")
        if self.min_first_genuine_deliveries < 1:
            raise ValueError(
                "modes.clock_validation.criteria.min_first_genuine_deliveries must be >= 1"
            )
        if self.min_pairs_per_condition < 2:
            raise ValueError(
                "modes.clock_validation.criteria.min_pairs_per_condition must be >= 2 (one "
                "sequence cannot estimate between-sequence variance)"
            )

    def criteria_dict(self) -> dict:
        """The decision criteria as recorded (and locked) for a run."""

        return {
            "equivalence_margin": float(self.equivalence_margin),
            "min_first_genuine_deliveries": int(self.min_first_genuine_deliveries),
            "min_pairs_per_condition": int(self.min_pairs_per_condition),
        }


@dataclass(frozen=True)
class ExperimentConfig:
    conditions: tuple[str, ...]
    attempts_per_sequence: int
    sequences_per_condition: int
    randomization_seed: int
    max_trial_duration_s: int
    max_actions: int
    mtd_cadence_s: int
    mtd_rotation_phase: str
    monitoring_interval_s: int
    vuln_profile_version: str
    attacker: AttackerParams
    workload: WorkloadParams
    pilot: PilotMode
    repeatability: RepeatabilityMode
    run: RunParams = field(default_factory=RunParams)
    clock_validation: ClockValidationMode = field(default_factory=ClockValidationMode)
    tooling_versions: str = "mock-tools=0.1.0"
    trigger_mode: str = "time"
    config_version: str = "0.1.0"
    live: LiveSettings = field(default_factory=LiveSettings)
    # Execution order of the full run's (condition, sequence) units (``RUN_ORDERS``).
    run_order: str = RUN_ORDER_BLOCKED

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_order", normalize_run_order(self.run_order))

    @staticmethod
    def from_dict(d: dict) -> ExperimentConfig:
        design = d.get("design", {})
        limits = d.get("limits", {})
        mtd = d.get("mtd", {})
        mon = d.get("monitoring", {})
        wl = d.get("workload", {})
        atk = d.get("attacker", {})
        modes = d.get("modes", {})
        meta = d.get("meta", {})
        live = d.get("live", {})

        pilot = modes.get("pilot", {})
        rep = modes.get("repeatability", {})
        cval = modes.get("clock_validation") or {}
        criteria = cval.get("criteria") or {}
        bad_keys = sorted(set(criteria) - set(CLOCK_VALIDATION_CRITERIA_KEYS))
        if bad_keys:
            raise ValueError(
                f"modes.clock_validation.criteria has unknown keys {bad_keys}; accepted: "
                f"{list(CLOCK_VALIDATION_CRITERIA_KEYS)}"
            )
        cval_defaults = ClockValidationMode()

        rotation_phase = str(mtd.get("rotation_phase", "fixed"))
        if rotation_phase not in ("fixed", "randomized"):
            raise ValueError(
                f"mtd.rotation_phase must be 'fixed' or 'randomized', got {rotation_phase!r}"
            )

        run = d.get("run", {})
        on_trial_failure = str(run.get("on_trial_failure", "exclude"))
        if on_trial_failure not in ("exclude", "halt"):
            raise ValueError(
                f"run.on_trial_failure must be 'exclude' or 'halt', got {on_trial_failure!r}"
            )
        max_consecutive_exclusions = int(run.get("max_consecutive_exclusions", 5))
        fault_rate = _optional_float(run, "fault_rate", None)
        if fault_rate is not None and not 0.0 <= fault_rate < 1.0:
            raise ValueError(f"run.fault_rate must be in [0, 1), got {fault_rate!r}")

        return ExperimentConfig(
            conditions=tuple(design.get("conditions", ())),
            attempts_per_sequence=int(design.get("attempts_per_sequence", 8)),
            sequences_per_condition=int(design.get("sequences_per_condition", 10)),
            randomization_seed=int(meta.get("randomization_seed", 0)),
            max_trial_duration_s=int(limits.get("max_trial_duration_s", 1800)),
            max_actions=int(limits.get("max_actions", 50)),
            mtd_cadence_s=int(mtd.get("cadence_s", 90)),
            mtd_rotation_phase=rotation_phase,
            trigger_mode=str(mtd.get("trigger_mode", "time")),
            monitoring_interval_s=int(mon.get("sample_interval_s", 5)),
            vuln_profile_version=str(d.get("vuln_profile_version", "vp-0.2.0")),
            attacker=AttackerParams(
                backend=str(atk.get("backend", "mock")),
                model_id=str(atk.get("model_id", "mock-attacker")),
                model_version=str(atk.get("model_version", "0.1.0")),
                temperature=_optional_float(atk, "temperature", 0.0),
                # Absent and explicit ``null`` both mean "omit the key" here, so unlike
                # ``temperature`` there is no third state to preserve. Validation is
                # eager: a typo raises at load time rather than silently running control.
                tool_choice=normalize_tool_choice(atk.get("tool_choice")),
                context_window=int(atk.get("context_window", 200000)),
                system_prompt_path=str(atk.get("system_prompt_path", "")),
                tool_access=tuple(atk.get("tool_access", ())),
                max_retries=int(atk.get("max_retries", 8)),
                request_timeout_s=float(atk.get("request_timeout_s", 120.0)),
                api_retry_attempts=int(atk.get("api_retry_attempts", 6)),
                # Eager validation, like tool_choice: a typo raises here rather than running
                # the other clock.
                clock=normalize_clock_mode(atk.get("clock")),
                record_timing=_strict_bool(atk, "record_timing", False),
            ),
            workload=WorkloadParams(
                legitimate_rate_rps=float(wl.get("legitimate_rate_rps", 1.0)),
                availability_timeout_s=int(wl.get("availability_timeout_s", 2)),
            ),
            pilot=PilotMode(
                conditions=tuple(pilot.get("conditions", ("static_baseline", "endpoint_mutation"))),
                sequences_per_condition=int(pilot.get("sequences_per_condition", 2)),
            ),
            repeatability=RepeatabilityMode(
                condition=str(rep.get("condition", "static_baseline")),
                adaptation_enabled=bool(rep.get("adaptation_enabled", False)),
                n_trials=int(rep.get("n_trials", 15)),
            ),
            run=RunParams(
                on_trial_failure=on_trial_failure,
                max_consecutive_exclusions=max_consecutive_exclusions,
                fault_rate=fault_rate,
            ),
            run_order=normalize_run_order(design.get("run_order")),
            clock_validation=ClockValidationMode(
                conditions=tuple(cval.get("conditions", cval_defaults.conditions)),
                sequences_per_condition=int(
                    cval.get("sequences_per_condition", cval_defaults.sequences_per_condition)
                ),
                equivalence_margin=float(
                    criteria.get("equivalence_margin", cval_defaults.equivalence_margin)
                ),
                min_first_genuine_deliveries=int(
                    criteria.get(
                        "min_first_genuine_deliveries", cval_defaults.min_first_genuine_deliveries
                    )
                ),
                min_pairs_per_condition=int(
                    criteria.get("min_pairs_per_condition", cval_defaults.min_pairs_per_condition)
                ),
            ),
            config_version=str(meta.get("config_version", "0.1.0")),
            live=LiveSettings(
                enabled=bool(live.get("enabled", False)),
                synthetic_telemetry=bool(live.get("synthetic_telemetry", False)),
                testbed_cidr=str(live.get("testbed_cidr", "")),
                control_host=str(live.get("control_host", "control-host")),
                executor_node=str(live.get("executor_node", "mtd-attacker")),
                targets_host=str(live.get("targets_host", "")),
                testbed_nodes=tuple(str(n) for n in live.get("testbed_nodes", ())),
                node_ssh_ports=tuple(
                    (str(h), int(p)) for h, p in (live.get("node_ssh_ports") or {}).items()
                ),
                ssh_user=str(live.get("ssh_user", "labadmin")),
                provisioning_dns=str(live.get("provisioning_dns", "")),
                provisioning_iface=str(live.get("provisioning_iface", "eth0")),
                provisioning_dns_check_host=str(
                    live.get("provisioning_dns_check_host", "archive.ubuntu.com")
                ),
                scan_scope=normalize_scan_scope(live.get("scan_scope")),
                surface_check=normalize_surface_check(live.get("surface_check")),
                readiness_timeout_s=float(live.get("readiness_timeout_s", 300.0)),
            ),
        )

    @staticmethod
    def load(path: str | Path) -> ExperimentConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return ExperimentConfig.from_dict(data)
