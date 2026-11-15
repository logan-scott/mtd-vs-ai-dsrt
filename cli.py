"""Reproducible command-line entry point for the instrument.

Generates a trial-level dataset and runs the analysis pipeline, writing all artifacts
under ``data/``. By default the run is offline, with the mock attacker; ``--live
--backend anthropic`` drives the model-planned attacker against the deployed testbed
with the same commands.

    python cli.py run      --config harness/experiment_config.example.yaml
    python cli.py pilot    --config harness/experiment_config.example.yaml
    python cli.py analyze  --dataset data/full/trial_dataset.csv   # re-analyze, no re-run
    python cli.py validate-clock --backend stub --sequences 1   # clock-validation pilot (offline smoke)
    python cli.py clock-report --dataset data/clock_validation/trial_dataset.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings

# timezone.utc (not datetime.UTC, which is 3.11+): the collection path — `run`, `pilot`,
# `reconfig-bench` — must import on the targets node's Python 3.10. timezone.utc is the SAME
# object datetime.UTC aliases, so emitted timestamps are byte-identical. See pyproject.toml
# `requires-python` and tests/test_collection_path_py310_clean.py.
from datetime import datetime, timezone
from pathlib import Path

# NOTE: `analysis` (pandas / numpy / scipy / statsmodels / lifelines) is imported LAZILY,
# inside the subcommands that analyze data (`run`, `pilot`, `analyze`), NOT at module scope.
# Data collection — `reconfig-bench` on the sealed targets node, a live `run`, a `preflight`
# — must not require the analysis stack: the targets node has only Python 3 + PyYAML and no
# egress to install more. A module-scope `from analysis import ...` made `python cli.py
# reconfig-bench` fail at import before the subcommand was even reached. Keep it lazy; the
# regression test `tests/test_cli_import_without_analysis_stack.py` fails if it returns here.
from attacker.backends import AnthropicClient, build_attacker
from attacker.base import PROMPT_FORMAT_VERSION
from attacker.planner import LLMAttacker
from harness.config import ExperimentConfig
from harness.environment import OfflineEnvironment
from harness.experiment_runner import run_experiment_dataset, run_repeatability
from harness.safety import SafetyError
from harness.tooling import ToolingError
from measurement import run_sheet
from testbed.profile import default_profile

warnings.filterwarnings("ignore")


def _wire_live(config: ExperimentConfig):
    """Build the live env/executor factories, failing closed at the isolation preflight.

    Returns ``(config, env_factory, executor_factory, provisioner)`` for a real live run —
    ``config`` is returned with ``tooling_versions`` refreshed from the executor node (the
    real ``nmap``/``curl``/``metasploit`` versions the primitives drive), so every trial row
    records the actual standard tooling used. The preflight actively probes the built topology
    (CIDR allowlist, no node egress, planner reaches the API, control-host forwarding off, no
    node default route) and aborts on any violation. A live run therefore refuses to start
    until a real, correctly-isolated testbed is deployed. It also refuses to start unless the
    resident Metasploit daemon answers on the executor node (``harness.tooling.require_msfd``):
    without it every Struts and vsftpd delivery would end its sequence as an infrastructure
    fault. Use ``--dry-run`` to exercise the path offline with the mock executor.
    """

    import dataclasses

    from harness.live_runner import (
        AnsibleProvisioner,
        build_env_factory,
        build_remote_runner,
        live_executor_factory,
        resolve_targets_host,
        subprocess_transport_factory,
    )
    from harness.preflight import run_isolation_preflight
    from harness.safety import SshRouteChecker
    from harness.tooling import capture_tooling_versions, require_msfd
    from testbed.roster import load_roster

    settings = config.live
    runner = build_remote_runner(settings)
    run_isolation_preflight(settings, SshRouteChecker(runner))  # fail closed before anything runs
    msfd_version = require_msfd(runner, settings.executor_node)  # fail closed: no msfd, no run
    print(f"msfd on {settings.executor_node}: framework {msfd_version}")
    roster = load_roster()
    # Record the real tooling versions from the executor node (provenance; per-trial control).
    tooling = capture_tooling_versions(
        runner, settings.executor_node, roster_version=roster.vuln_profile_version
    )
    config = dataclasses.replace(config, tooling_versions=tooling)
    provisioner = AnsibleProvisioner(
        runner,
        settings.control_host,
        ssh_user=settings.ssh_user,
        # Wiring for the deception manipulation check: after reconfiguring a condition
        # that rotates decoy ports, the provisioner probes the targets host from the
        # executor node and fails the attempt if any decoy is not reachable on its
        # modeled observable port.
        executor_node=settings.executor_node,
        targets_host=resolve_targets_host(settings),
        testbed_cidr=settings.testbed_cidr,
        # live.surface_check: strict -- every attempt waits until every in-force service
        # answers before the attacker starts (harness.config.SURFACE_CHECKS).
        readiness_timeout_s=(
            settings.readiness_timeout_s if settings.surface_check == "strict" else 0.0
        ),
    )
    transport_factory = subprocess_transport_factory(settings, runner, roster.primitives())
    env_factory = build_env_factory(settings, transport_factory, provisioner)
    return config, env_factory, live_executor_factory, provisioner


def _generate(
    config: ExperimentConfig,
    mode: str,
    outdir: Path,
    fault_rate: float,
    backend: str,
    *,
    env_factory=OfflineEnvironment,
    executor_factory=None,
    provisioner=None,
    checkpoint=None,
):
    # 'mock' and 'stub' run fully offline; 'anthropic' calls the model (needs a key)
    # while its actions still resolve against the (offline mock or live) target.
    attacker = build_attacker(config, backend=backend)
    if executor_factory is not None and hasattr(attacker, "executor_factory"):
        attacker.executor_factory = executor_factory  # live: resolve via env.executor
    profile = default_profile(config.vuln_profile_version)
    resumed = bool(checkpoint and checkpoint.completed_ids())
    started_utc = _utc_now()
    # The code THIS process loaded, read before any trial runs: a pull made while the run is in
    # progress must not be recorded as the code that collected it (C235).
    code = _git_provenance()
    started = time.perf_counter()
    data = run_experiment_dataset(
        config,
        attacker,
        profile,
        mode=mode,
        fault_rate=fault_rate,
        env_factory=env_factory,
        checkpoint=checkpoint,
        on_trial_failure=config.run.on_trial_failure,
        max_consecutive_exclusions=config.run.max_consecutive_exclusions,
        progress=(lambda line: print(line, flush=True)) if checkpoint else None,
        usage_snapshot=_usage_snapshot_fn(attacker),
    )
    rep = run_repeatability(config, attacker, profile, env_factory=env_factory)
    wall_s = time.perf_counter() - started

    outdir.mkdir(parents=True, exist_ok=True)
    run_sheet.write_csv(data.trial_rows, outdir / "trial_dataset.csv")
    run_sheet.write_jsonl(data.trial_rows, outdir / "trial_dataset.jsonl")
    run_sheet.write_csv(rep, outdir / "repeatability.csv")
    run_sheet.write_schema(outdir / "schema.json")
    (outdir / "uptime_events.json").write_text(json.dumps(data.uptime_events), encoding="utf-8")
    _write_run_meta(
        outdir,
        data,
        backend,
        env_factory,
        wall_s,
        attacker,
        provisioner,
        config,
        resumed,
        fault_rate=fault_rate,
        started_utc=started_utc,
        finished_utc=_utc_now(),
        code=code,
        checkpoint=checkpoint,
    )
    print(f"wrote {len(data.trial_rows)} trials + {len(rep)} repeatability rows -> {outdir}")
    return data, rep


def _utc_now() -> str:
    """The current UTC time (real run dates; C235), in the representation every ``run_meta.json``
    and ``_run_manifest.json`` timestamp uses: ISO-8601, seconds, ``+00:00`` (pinned by
    ``tests/test_timestamp_format.py``)."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_provenance(root: Path | None = None) -> dict:
    """The instrument's git commit and uncommitted changes at collection time (fail-soft).

    Recorded in ``run_meta.json`` so every dataset names the exact code that produced it
    (C235). ``data/`` is gitignored, so the run's own outputs never appear as changes.
    ``root`` defaults to the instrument directory (tests point it at a scratch repository).
    """

    import subprocess

    here = str(root if root is not None else Path(__file__).resolve().parent)
    out: dict = {"git_commit": None, "git_uncommitted": None}
    try:
        head = subprocess.run(
            ["git", "-C", here, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if head.returncode == 0:
            out["git_commit"] = head.stdout.strip() or None
        status = subprocess.run(
            ["git", "-C", here, "status", "--porcelain", "--", "."],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if status.returncode == 0:
            changed = [ln.rstrip() for ln in status.stdout.splitlines() if ln.strip()]
            out["git_uncommitted"] = changed[:50]
            if len(changed) > 50:
                out["git_uncommitted_truncated"] = len(changed)
            # The CONTENT of the uncommitted state, not just the file names: a second edit to an
            # already-modified file changes this hash (it is what code_changed_since_first_start
            # compares). Untracked files contribute their names.
            diff = subprocess.run(
                ["git", "-C", here, "diff", "HEAD", "--", "."],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if diff.returncode == 0:
                import hashlib

                digest = hashlib.sha256(diff.stdout.encode("utf-8"))
                untracked = sorted(ln[3:] for ln in changed if ln.startswith("?? "))
                digest.update("\n".join(untracked).encode("utf-8"))
                out["git_uncommitted_sha256"] = digest.hexdigest()[:16]
    except (OSError, subprocess.SubprocessError):
        pass
    return out


#: ``_run_manifest.json`` key holding the code version and UTC time of a checkpointed run's
#: FIRST start. Recorded once, outside the fingerprint, so a resume after a pull is visible.
_CODE_AT_FIRST_START_KEY = "code_at_first_start"


def _first_start_record() -> dict:
    """The manifest's record of when, and on which code, a checkpointed run first started."""

    return {_CODE_AT_FIRST_START_KEY: {"started_utc": _utc_now(), **_git_provenance()}}


def _haiku_cost_usd(usage: dict) -> float:
    """Token usage priced on the Haiku 4.5 table (USD per million tokens; see run_meta)."""

    return (
        usage.get("input_tokens", 0) * 1.00
        + usage.get("cache_read_input_tokens", 0) * 0.10
        + usage.get("cache_creation_input_tokens", 0) * 1.25
        + usage.get("output_tokens", 0) * 5.00
    ) / 1_000_000.0


def _usage_snapshot_fn(attacker):
    """A callable returning this process's cumulative API usage, or None (no API client)."""

    client = getattr(attacker, "client", None)
    if client is None or not hasattr(client, "served_models"):
        return None

    def snapshot() -> dict:
        totals = client.usage_totals() if hasattr(client, "usage_totals") else {}
        return {
            "served_models": dict(client.served_models),
            "llm_calls": int(getattr(client, "call_count", 0) or 0),
            "token_usage": {k: int(v) for k, v in dict(totals).items()},
        }

    return snapshot


def _collection_provenance(
    config, attacker, started_utc, finished_utc, code=None, checkpoint=None
) -> dict:
    """What a replicator needs to know about how the data were collected (C235).

    ``code`` is the git record read when this process started (read here, at write time, only
    when a caller did not pass one). For a checkpointed run, the code and time of its FIRST
    start are copied from the manifest, with a flag when this process ran different code -- a
    resume after a pull mixes code versions within one dataset, and the archive must say so.
    """

    import platform

    prov: dict = {
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "python": platform.python_version(),
        **(code if code is not None else _git_provenance()),
    }
    manifest = checkpoint.manifest() if checkpoint is not None else None
    first = (manifest or {}).get(_CODE_AT_FIRST_START_KEY)
    if isinstance(first, dict):
        prov[_CODE_AT_FIRST_START_KEY] = first
        prov["code_changed_since_first_start"] = (
            first.get("git_commit"),
            first.get("git_uncommitted"),
            first.get("git_uncommitted_sha256"),
        ) != (
            prov.get("git_commit"),
            prov.get("git_uncommitted"),
            prov.get("git_uncommitted_sha256"),
        )
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            prov["anthropic_sdk"] = version("anthropic")
        except PackageNotFoundError:
            prov["anthropic_sdk"] = None
    except ImportError:  # pragma: no cover - stdlib since 3.8
        prov["anthropic_sdk"] = None
    client = getattr(attacker, "client", None)
    if isinstance(client, AnthropicClient):
        prov["max_tokens"] = client.max_tokens
        prov["api_served_models"] = dict(sorted(client.served_models.items()))
        # Which user-message format the model was sent (``attacker.backends``): 2 = full-identity
        # outcomes plus the attacker's observations from earlier attempts (2026-10-31). Runs
        # collected before this key existed used format 1.
        prov["attacker_prompt_format"] = PROMPT_FORMAT_VERSION
    snapshot = _usage_snapshot_fn(attacker)
    if checkpoint is not None and snapshot is not None:
        # A resumed run was collected by several processes; each recorded its cumulative usage
        # after every sequence. Sum the last record of each, with THIS process's final counts
        # (they include the repeatability trials that follow the last sequence).
        by_process = checkpoint.usage_by_process()
        by_process[checkpoint.process_id] = snapshot()
        served: dict[str, int] = {}
        tokens: dict[str, int] = {}
        calls = 0
        for snap in by_process.values():
            for model, n in (snap.get("served_models") or {}).items():
                served[model] = served.get(model, 0) + int(n)
            for key, n in (snap.get("token_usage") or {}).items():
                tokens[key] = tokens.get(key, 0) + int(n)
            calls += int(snap.get("llm_calls") or 0)
        if "api_served_models" in prov:
            prov["api_served_models_this_process"] = prov["api_served_models"]
        prov["api_served_models"] = dict(sorted(served.items()))
        prov["usage_all_processes"] = {
            "processes": len(by_process),
            "llm_calls": calls,
            "token_usage": tokens,
            "est_cost_usd": round(_haiku_cost_usd(tokens), 4),
        }
    if config is not None:
        from measurement.timing import TIMING_FORMAT_VERSION

        prov["run_order"] = getattr(config, "run_order", "blocked")
        prov["scan_scope"] = getattr(getattr(config, "live", None), "scan_scope", "subnet")
        prov["surface_check"] = getattr(getattr(config, "live", None), "surface_check", "off")
        prov["attacker_model_version"] = getattr(attacker, "model_version", None)
        if getattr(attacker, "clock_mode", "modeled") == "wall":
            prov["wall_clock_dating"] = "probe_time"
            prov["wall_clock_scoring"] = "completion"
            prov["timing_format_version"] = TIMING_FORMAT_VERSION
    return prov


def _write_run_meta(
    outdir,
    data,
    backend,
    env_factory,
    wall_s,
    attacker,
    provisioner=None,
    config=None,
    resumed=False,
    fault_rate=None,
    extra=None,
    started_utc=None,
    finished_utc=None,
    code=None,
    checkpoint=None,
):
    """Sidecar: timings, live/offline mode, fault exclusions, provisioning, and cost.

    ``extra`` (a dict) is merged in last -- the clock-validation pilot records its arms and
    pre-specified criteria there. ``provenance`` records the code version, run dates, SDK and
    request settings (C235); ``code`` is the git record taken when the run started.
    """

    excluded = [r for r in data.trial_rows if getattr(r, "excluded", 0)]
    excluded_seqs = getattr(data, "excluded_sequences", []) or []
    seq_reasons = sorted({s.get("reason", "") for s in excluded_seqs if s.get("reason")})
    meta = {
        "backend": backend,
        "mode": "live" if env_factory is not OfflineEnvironment else "offline",
        "wall_clock_s": round(wall_s, 3),
        "resumed": bool(resumed),
        "n_trials": len(data.trial_rows),
        # Trial-level exclusions (a-priori: checksum / simulated infra fault) ...
        "n_excluded": len(excluded),
        "exclusion_reasons": sorted({r.exclusion_reason for r in excluded if r.exclusion_reason}),
        # ... plus whole sequences lost to an unrecoverable API/infra fault (buffered).
        "n_excluded_sequences": len(excluded_seqs),
        "n_lost_trials": sum(int(s.get("n_lost_attempts", 0)) for s in excluded_seqs),
        "excluded_sequence_reasons": seq_reasons,
        # Each lost sequence and what failed (the fault's own message; absent in runs collected
        # before 2026-10-27), so every exclusion can be reported by cause, not just by category.
        "excluded_sequence_details": [
            {k: s.get(k) for k in ("sequence_id", "reason", "detail")} for s in excluded_seqs
        ],
        # Rate of the SIMULATED row-level fault injector (cli --fault-rate). Recorded so a
        # reader can tell injected 'infrastructure_fault' rows from real testbed failures
        # without reading the harness source. Runs collected before this key existed report
        # it as absent, and the report's exclusions.run_meta_agreement says so explicitly.
        "fault_rate": (float(fault_rate) if fault_rate is not None else None),
    }
    if config is not None:
        meta["resilience"] = {
            "max_retries": config.attacker.max_retries,
            "request_timeout_s": config.attacker.request_timeout_s,
            "api_retry_attempts": config.attacker.api_retry_attempts,
            "on_trial_failure": config.run.on_trial_failure,
        }
        # Which attacker clock the run used and whether real action timing was recorded
        # (always under "wall"). Read from the ATTACKER that ran, not the config: the mock
        # records no timing whatever the config says, and the clock-validation pilot runs two
        # attackers that differ only in their clock (it overrides this block via ``extra``).
        # Per-row truth is the ``clock_mode`` column (metrics spec §4f).
        is_planner = isinstance(attacker, LLMAttacker)
        clock_used = attacker.clock_mode if is_planner else "modeled"
        meta["attacker_clock"] = {
            "clock": clock_used,
            "record_timing": bool(is_planner and (attacker.record_timing or clock_used == "wall")),
        }
    meta["provenance"] = _collection_provenance(
        config, attacker, started_utc, finished_utc, code=code, checkpoint=checkpoint
    )
    if provisioner is not None and getattr(provisioner, "records", None):
        meta["provisioning"] = provisioner.summary()
    client = getattr(attacker, "client", None)
    if isinstance(client, AnthropicClient):
        usage = client.usage_totals()
        # Haiku 4.5 USD/M defaults, matching the cost-probe. These are NOT model-agnostic:
        # the same arithmetic is applied to whatever Anthropic model is configured, so on a
        # non-Haiku arm est_cost_usd is priced on the wrong table. Rather than guess another
        # model's prices, the basis is recorded next to the figure so a reader can tell.
        cost = _haiku_cost_usd(usage)
        meta["llm_calls"] = client.call_count
        meta["token_usage"] = usage
        meta["est_cost_usd"] = round(cost, 4)
        meta["est_cost_basis"] = {
            "price_table": "claude-haiku-4-5 (USD/M: in 1.00, cache_read 0.10, "
            "cache_write 1.25, out 5.00)",
            # startswith, not equality: every config in the repo pins a DATED SNAPSHOT
            # ("claude-haiku-4-5-20251001"), so comparing to the bare alias made this a
            # permanent false negative -- it disclaimed an exact figure on every Haiku run,
            # including the frozen 2026-08-24 archive's $22.05.
            "applies_to_configured_model": str(
                getattr(getattr(config, "attacker", None), "model_id", None) or ""
            ).startswith("claude-haiku-4-5"),
            "note": (
                "est_cost_usd is computed with the Haiku 4.5 price table regardless of the "
                "configured model. When applies_to_configured_model is false the figure is "
                "a token-weighted Haiku-equivalent cost, not this model's actual spend."
            ),
        }
    if extra:
        meta.update(extra)
    (outdir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _setup_checkpoint(
    config: ExperimentConfig, outdir: Path, fresh: bool, payload=None, extra=None, backend=None
):
    """Build the run checkpoint; fail closed on a config-fingerprint mismatch.

    Returns a ``RunCheckpoint`` (resuming if a matching one exists) or exits with a clear
    message if the existing checkpoint was written for a different run-defining config and
    ``--fresh`` was not given. ``payload`` replaces the config's fingerprint payload (the
    clock-validation pilot fingerprints its arms and design too); ``extra`` is recorded in the
    manifest at the first start, outside the fingerprint (see ``RunCheckpoint.init``);
    ``backend`` is the attacker backend the run uses (``--backend``), which decides whether the
    prompt format enters the fingerprint.
    """

    from harness import checkpoint as ckpt_mod

    if payload is None:
        payload = ckpt_mod.fingerprint_payload(config, backend=backend)
    fingerprint = ckpt_mod.payload_fingerprint(payload)
    cp = ckpt_mod.RunCheckpoint(outdir, fingerprint)
    if fresh:
        cp.clear()
    if cp.exists():
        stored = cp.stored_fingerprint()
        if stored != fingerprint:
            print(
                f"ABORT: {outdir / '_run_manifest.json'} was written for a DIFFERENT "
                f"run-defining config (stored fingerprint {stored}, current {fingerprint}). "
                "Resuming would mix incompatible runs. Re-run with --fresh to discard the old "
                "checkpoint and start clean."
            )
            raise SystemExit(4)
    else:
        cp.init(payload, extra=extra)
    return cp


#: The fault-injection rate a run uses when neither ``--fault-rate`` nor ``run.fault_rate`` sets
#: one: the historical default of every archived run (C278).
_LEGACY_FAULT_RATE = 0.03


def _resolve_fault_rate(cli_value, config) -> float:
    """``--fault-rate`` if given, else the config's ``run.fault_rate``, else the legacy 0.03."""

    if cli_value is not None:
        return float(cli_value)
    configured = getattr(getattr(config, "run", None), "fault_rate", None)
    return _LEGACY_FAULT_RATE if configured is None else float(configured)


def _apply_run_overrides(config: ExperimentConfig, args) -> ExperimentConfig:
    """``run --sequences/--conditions``: shrink the full design (a smoke run) explicitly.

    Both are run-defining, so they enter the checkpoint fingerprint through the config.
    """

    import dataclasses

    from harness.config import _ALL_CONDITIONS

    overrides: dict = {}
    sequences = getattr(args, "sequences", None)
    if sequences is not None:
        if sequences < 1:
            print("ABORT: --sequences must be >= 1")
            raise SystemExit(2)
        overrides["sequences_per_condition"] = int(sequences)
    conditions = getattr(args, "conditions", None)
    if conditions:
        unknown = [c for c in conditions if c not in _ALL_CONDITIONS]
        if unknown or len(set(conditions)) != len(conditions):
            print(
                f"ABORT: --conditions must be distinct names from {list(_ALL_CONDITIONS)}, "
                f"got {conditions}"
            )
            raise SystemExit(2)
        overrides["conditions"] = tuple(conditions)
    return dataclasses.replace(config, **overrides) if overrides else config


def _run_pipeline(config: ExperimentConfig, mode: str, args, default_outdir: str):
    """Shared body for run/pilot: dry-run plan, live wiring (fail closed), or offline."""

    if mode == "full":
        config = _apply_run_overrides(config, args)
    fault_rate = _resolve_fault_rate(getattr(args, "fault_rate", None), config)

    if getattr(args, "dry_run", False):
        from harness import live_runner

        plan = live_runner.dry_run_plan(config, mode=mode)
        print(_format_plan(plan))
        print(
            _format_design_settings(
                config, mode, fault_rate, backend=getattr(args, "backend", None)
            )
        )
        return None

    env_factory, executor_factory, provisioner = OfflineEnvironment, None, None
    if getattr(args, "live", False):
        config, env_factory, executor_factory, provisioner = _wire_live(config)

    outdir = Path(args.outdir or default_outdir)
    # The full confirmatory run is checkpointed + resumable; the short pilot is not.
    checkpoint = None
    if mode == "full":
        checkpoint = _setup_checkpoint(
            config,
            outdir,
            getattr(args, "fresh", False),
            extra=_first_start_record(),
            backend=args.backend,
        )
    return _generate(
        config,
        mode,
        outdir,
        fault_rate,
        args.backend,
        env_factory=env_factory,
        executor_factory=executor_factory,
        provisioner=provisioner,
        checkpoint=checkpoint,
    )


def _surface_line(config: ExperimentConfig) -> str:
    live = config.live
    if live.surface_check != "strict":
        return "off (the attacker may meet a partly started testbed)"
    return (
        f"strict (each attempt waits up to {live.readiness_timeout_s:g} s for every service; "
        "a scan that misses one is an infrastructure fault)"
    )


def _format_design_settings(
    config: ExperimentConfig, mode: str, fault_rate: float, backend: str | None = None
) -> str:
    """The run-defining settings a dry run should show before a multi-day run starts."""

    atk = config.attacker
    backend = backend or atk.backend
    prompt = (
        f"format {PROMPT_FORMAT_VERSION} (the attacker sees its earlier attempts in the sequence)"
        if backend == "anthropic"
        else f"not used (the {backend} backend chooses without a model)"
    )
    timing = atk.record_timing or atk.clock == "wall"
    dating = (
        " (each scan dated to when it probed the targets host; each exploit scored when it"
        " completes)"
        if atk.clock == "wall"
        else ""
    )
    run_order = config.run_order if mode == "full" else "blocked (only the full run interleaves)"
    return "\n".join(
        [
            "design settings:",
            f"  sequences/condition: {config.sequences_per_condition} "
            f"x {config.attempts_per_sequence} attempts",
            f"  run order:           {run_order}",
            f"  attacker clock:      {atk.clock}{dating}; timing recorded: {timing}",
            f"  model:               {atk.model_id} (version {atk.model_version})",
            f"  attacker prompt:     {prompt}",
            f"  scan scope:          {config.live.scan_scope}",
            f"  surface check:       {_surface_line(config)}",
            f"  simulated faults:    {fault_rate}",
        ]
    )


def _format_plan(plan: dict) -> str:
    lines = [
        f"DRY RUN — {plan['mode']} (live={plan['live']}, executor: {plan['executor']})",
        f"conditions: {', '.join(plan['conditions'])}",
        f"trials:     {plan['trials']}",
        f"testbed_cidr: {plan['testbed_cidr']}",
        "sample targets (observable identity -> in-CIDR delivery host, no network touched):",
    ]
    lines += [
        f"  {t['address']:>15}  {t['service']} port={t['port']} path={t['endpoint'] or '/'}"
        f"  -> {t['delivery_host']}"
        for t in plan["sample_targets"]
    ]
    lines.append("egress map:")
    for host, spec in plan["egress_map"].items():
        lines.append(f"  {host} [{spec['role']}]")
        lines.append(f"    may reach:      {', '.join(spec['may_reach'])}")
        lines.append(f"    must not reach: {', '.join(spec['must_not_reach'])}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Report provenance (what produced this report, from what inputs, on which stack)
# --------------------------------------------------------------------------- #

_ANALYSIS_DISTS = (
    "numpy",
    "pandas",
    "scipy",
    "statsmodels",
    "lifelines",
    "pingouin",
    "scikit-learn",
    "scikit-posthocs",
)
_LOCKFILE = Path(__file__).resolve().parent / "requirements-analysis.lock"


def _library_versions() -> dict:
    """Installed versions of the analysis stack, recorded in every report.

    The mixed-model layer is version-sensitive on rank-deficient designs: the same code on
    the same CSV returns a "Singular matrix" note under the statsmodels build that produced
    the archived report and a boundary-solution fit under 0.14.6. A report is therefore only
    reproducible against a recorded stack, so the stack is part of the result.
    """

    from importlib.metadata import PackageNotFoundError, version

    out: dict = {"python": sys.version.split()[0]}
    for dist in _ANALYSIS_DISTS:
        try:
            out[dist] = version(dist)
        except PackageNotFoundError:
            out[dist] = None
    return out


def _sha256(path: Path) -> str:
    """Streamed SHA-256 of a file (inputs are up to a few MB; do not slurp)."""

    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_lockfile(path: Path) -> dict:
    """Parse ``name==version`` pins out of a pip-freeze lockfile (names normalized)."""

    pinned: dict = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, _, ver = line.partition("==")
        pinned[name.strip().lower().replace("_", "-")] = ver.strip()
    return pinned


def _lockfile_provenance() -> dict:
    """Whether the running analysis stack matches the committed pin set.

    ``stack_matches_lock`` is None when no lockfile is committed (nothing to compare
    against), True when every pinned analysis distribution equals the installed version,
    and False with the offending names in ``lock_mismatches`` otherwise -- so a reader can
    tell at a glance whether the numbers in this report were produced on the stack of
    record, without having to trust that nobody upgraded anything.
    """

    if not _LOCKFILE.exists():
        return {
            "lockfile": None,
            "lockfile_sha256": None,
            "stack_matches_lock": None,
            "lock_mismatches": [],
        }
    pinned = _read_lockfile(_LOCKFILE)
    live = _library_versions()
    mismatches = sorted(d for d in _ANALYSIS_DISTS if d in pinned and pinned[d] != live.get(d))
    return {
        "lockfile": _LOCKFILE.name,
        "lockfile_sha256": _sha256(_LOCKFILE),
        "stack_matches_lock": not mismatches,
        "lock_mismatches": mismatches,
    }


def _provenance(source: str, **extra) -> dict:
    """Build the provenance block recorded at ``analysis_report.json:provenance``."""

    import platform

    prov = {
        "source": source,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    prov.update(extra)
    # Mixed-model fits are platform-sensitive (BLAS / optimizer), so the OS is part of the
    # result as well as the library versions.
    prov["platform"] = platform.platform()
    prov["library_versions"] = _library_versions()
    prov.update(_lockfile_provenance())
    return prov


def _resolve_sidecar(explicit, skip: bool, base: Path, name: str, flag: str):
    """Resolve an optional analysis input; returns ``(path or None, provenance note)``.

    ``--no-<flag>`` is a deliberate skip. An EXPLICIT path must exist -- a typo must never
    silently drop a whole report section. Otherwise the sidecar is auto-discovered beside
    the dataset; when it is absent the section is omitted and the report records why.
    """

    skip_flag = flag.replace("--", "--no-", 1)
    if skip:
        return None, f"{skip_flag} -> section skipped by request"
    if explicit is not None:
        path = Path(explicit)
        if not path.exists():
            print(f"ABORT: {flag} {path} not found")
            raise SystemExit(1)
        return path, f"{flag} {path}"
    path = base / name
    if path.exists():
        return path, f"auto-discovered {path}"
    return None, f"{name} not found beside the dataset -> section omitted (not reconstructible)"


def _run_provenance(source: str, outdir: Path) -> dict:
    """Provenance for an in-process run (rows are NOT round-tripped through the CSV).

    All three inputs are written by ``_generate`` BEFORE ``run_all`` is called, so they are
    hashed here exactly as ``cmd_analyze`` hashes them: a report from ``cli.py run`` and a
    report regenerated later from the same directory carry the same three digests, which is
    what makes the two comparable. A ``null`` digest therefore means the file genuinely is
    not on disk, never merely "not computed on this path".
    """

    dataset = outdir / "trial_dataset.csv"
    uptime = outdir / "uptime_events.json"
    rep = outdir / "repeatability.csv"
    return _provenance(
        source,
        dataset=str(dataset),
        dataset_sha256=_sha256(dataset) if dataset.exists() else None,
        uptime_events=str(uptime),
        uptime_events_sha256=_sha256(uptime) if uptime.exists() else None,
        repeatability=str(rep),
        repeatability_sha256=_sha256(rep) if rep.exists() else None,
        sections_omitted=[],
        notes=[f"generated in-process by {source} from the just-written run"],
    )


def cmd_run(args):
    from analysis.pipeline import run_all, write_report  # lazy: analysis stack (see top-of-file)
    from harness.experiment_runner import RunHalted

    config = ExperimentConfig.load(args.config)
    try:
        result = _run_pipeline(config, "full", args, "data/full")
    except RunHalted as exc:
        # on_trial_failure='halt': the checkpoint is durable; exit non-zero so the operator
        # resumes (re-run the same command) once the incident clears.
        print(
            f"HALTED at sequence {exc.sequence_id} ({exc.reason}). Checkpoint written to "
            f"{Path(args.outdir or 'data/full')}. Re-run the same command to resume after the "
            "incident clears."
        )
        raise SystemExit(5) from exc
    if result is None:  # dry-run printed the plan
        return
    data, rep = result
    outdir = Path(args.outdir or "data/full")
    results = run_all(
        data.trial_rows,
        data.uptime_events,
        rep,
        provenance=_run_provenance("cli.run", outdir),
        sequence_exclusions=data.excluded_sequences,
    )
    report = write_report(results, outdir / "analysis_report.json")
    print(f"analysis report -> {report}")


def cmd_pilot(args):
    from analysis.pipeline import run_all, write_report  # lazy: analysis stack (see top-of-file)

    config = ExperimentConfig.load(args.config)
    result = _run_pipeline(config, "pilot", args, "data/pilot")
    if result is None:  # dry-run printed the plan
        return
    data, rep = result
    outdir = Path(args.outdir or "data/pilot")
    results = run_all(
        data.trial_rows,
        data.uptime_events,
        rep,
        provenance=_run_provenance("cli.pilot", outdir),
        sequence_exclusions=data.excluded_sequences,
    )
    report = write_report(results, outdir / "analysis_report.json")
    print(f"pilot analysis report -> {report}")


def cmd_cost_probe(args):
    import os

    from harness import cost_probe

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "cost-probe needs ANTHROPIC_API_KEY set — it makes real Anthropic API calls "
            "(a few dollars of usage). Set the key and re-run."
        )
        raise SystemExit(1)

    config = ExperimentConfig.load(args.config)
    prices = {
        "input": args.price_input,
        "cache_read": args.price_cache_read,
        "cache_write": args.price_cache_write,
        "output": args.price_output,
    }
    projection = cost_probe.Projection(
        conditions=args.conditions, sequences=args.sequences, attempts=args.attempts
    )

    print(
        f"running {args.trials} real {config.attacker.model_id} trials "
        "against the offline mock ..."
    )
    records = cost_probe.run_probe(config, args.trials, prices, seed=args.seed)
    summary = cost_probe.summarize(
        records, prices, projection, floor=args.floor, model_id=config.attacker.model_id
    )

    outdir = Path(args.outdir or "data/cost_probe")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "trials.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print(cost_probe.format_summary(summary))
    print()
    print(f"raw records + summary -> {outdir}")


def cmd_benchmark(args):
    """Measure the technique-intrinsic cost of ONE reconfiguration, with replicates.

    Runs on the TARGETS NODE inside the sealed testbed (touches only the testbed CIDR; reuses
    ``assert_within_cidr``). Writes a tidy per-replicate CSV + a metadata sidecar
    (host/kernel/docker/git/image-digests/roster) + a summary, and restores the pristine
    profile on exit so a subsequent study run is not contaminated. This is the MEASURED
    replacement for the synthetic cost telemetry (see docs/metrics_specification.md §3).
    """

    from harness import reconfig_benchmark as rb

    config = ExperimentConfig.load(args.config)
    techniques = tuple(args.techniques) if args.techniques else config.conditions
    cfg = rb.BenchmarkConfig(
        techniques=techniques,
        replicates=args.replicates,
        testbed_dir=args.testbed_dir,
        reconfigure_sh=args.reconfigure_sh or str(Path(args.testbed_dir) / "reconfigure.sh"),
        targets_host=args.targets_host or config.live.targets_host or "172.31.66.10",
        bind_host=args.bind_host or args.targets_host or "172.31.66.10",
        testbed_cidr=args.testbed_cidr or config.live.testbed_cidr or "172.31.66.0/24",
        vuln_profile_version=config.vuln_profile_version,
        monitored_service_id=args.monitored_service,
        probe_interval_s=args.probe_interval,
        readiness_interval_s=args.readiness_interval,
        readiness_timeout_s=args.readiness_timeout,
        seed=args.seed,
    )
    base_compose = (
        Path(args.base_compose)
        if args.base_compose
        else Path(cfg.testbed_dir) / "docker-compose.yml"
    )

    if getattr(args, "dry_run", False):
        # Validate reachability + config WITHOUT touching Docker/SSH: this is the reachability
        # proof for the sealed node (no analysis stack) and lets the operator confirm the
        # command resolves before a real run. It asserts containment and prints the plan.
        plan = rb.dry_run_plan(cfg)
        print(
            f"[dry-run] {len(cfg.techniques)} techniques x {cfg.replicates} replicates on "
            f"{plan['targets_host']} (cidr {plan['testbed_cidr']}, testbed {plan['testbed_dir']}); "
            f"monitored service={plan['monitored_service_id']}; representative seed="
            f"{plan['representative_seed']}"
        )
        for t in plan["techniques"]:
            print(
                f"  {t['technique']:24s} noop={str(t['checksum_noop']):5s} "
                f"genuine_ports_move={str(t['genuine_ports_move']):5s} "
                f"decoy_ports_move={str(t['decoy_ports_move']):5s} "
                f"monitored_ports={t['monitored_ports']}"
            )
        print(
            f"\n[dry-run] would write reconfig_cost.csv + summary.json + metadata.json -> "
            f"{Path(args.outdir or 'data/reconfig_benchmark')}. No Docker/SSH touched."
        )
        return

    print(
        f"benchmarking {len(cfg.techniques)} techniques x {cfg.replicates} replicates "
        f"on {cfg.targets_host} (testbed {cfg.testbed_dir}) ..."
    )
    records = rb.run_benchmark(cfg)
    summary = rb.summarize(records)
    metadata = rb.build_metadata(
        cfg,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        base_compose=base_compose,
    )

    outdir = Path(args.outdir or "data/reconfig_benchmark")
    outdir.mkdir(parents=True, exist_ok=True)
    rb.write_records_csv(records, outdir / "reconfig_cost.csv")
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print()
    for tech, s in summary["per_technique"].items():
        w = s["wall_clock_s"]
        av = s["availability"]
        settle = s["settle_wait_s"]
        print(
            f"  {tech:24s} n={s['n']:<3d} median={w['median']:>8.3f}s "
            f"mean={w['mean']:>8.3f}s  avail(mean={av['mean']} n={av['n_measured']} "
            f"na={av['n_not_applicable_noop']} unusable={av['n_unusable_not_ready']}) "
            f"settle_max={settle['max']}s  restarts={s['restart_count_distribution']}"
        )
    bt = summary["between_technique_test"]
    bt_detail = bt.get("note") or f"H={bt.get('H')} p={bt.get('p_value')}"
    print(f"\n  between-technique {bt.get('test')}: {bt_detail}")
    print(f"\nper-replicate CSV + summary + metadata -> {outdir}")


def cmd_preflight(args):
    from harness import preflight as preflight_mod

    config = ExperimentConfig.load(args.config)
    checker = preflight_mod.build_checker(config.live, getattr(args, "dry_run", False))
    report = preflight_mod.run_isolation_preflight(config.live, checker)
    print(preflight_mod.format_report(report))


def cmd_deploy(args):
    from harness import provision
    from harness.live_runner import build_remote_runner

    config = ExperimentConfig.load(args.config)
    runner = build_remote_runner(config.live)
    result = provision.deploy_sealed(config.live, runner, mode=args.mode, inventory=args.inventory)
    print(
        f"deploy OK (mode={result.mode}, site rc={result.site_returncode}, "
        f"sealed={result.sealed}). Run a trial only now that the preflight passed."
    )


def _load_reconfig_bench(path_arg):
    """Load ``--reconfig-bench``; returns ``(frame or None, provenance dict or None)``.

    The reconfiguration-cost benchmark lives outside the run directory
    (``docs/results/benchmark/reconfig_cost.csv``), so it is never auto-discovered: without the
    flag, ``rq5.benchmark_profile`` records that the H05 profile test was not run. An explicit
    path must exist and parse -- a typo must never silently drop the test.
    """

    if not path_arg:
        return None, None
    from analysis import rq5_profile

    path = Path(path_arg)
    if not path.exists():
        print(f"ABORT: --reconfig-bench {path} not found")
        raise SystemExit(1)
    try:
        bench = rq5_profile.load_benchmark(path)
    except ValueError as exc:
        print(f"ABORT: {exc}")
        raise SystemExit(1) from exc
    prov = {"path": str(path), "sha256": _sha256(path), "n_rows_used": int(len(bench))}
    return bench, prov


def cmd_analyze(args):
    """Regenerate the FULL analysis report from a written trial dataset CSV.

    Runs exactly the same ``analysis.pipeline.run_all`` as ``cli.py run``, so a corrected
    report can be produced from an archived dataset without re-running a single trial.
    ``uptime_events.json`` and ``repeatability.csv`` are picked up from beside the dataset
    when present; when they are not, ``survival_uptime`` / ``reliability`` are OMITTED and
    listed in ``provenance.sections_omitted`` rather than fabricated. ``run_meta.json`` is
    picked up the same way and supplies the SEQUENCE-level exclusion counter that
    ``exclusions`` reconciles the row-level count against; without it that reconciliation
    reports the sequence count as unavailable rather than assuming zero.

    The frame handed to ``run_all`` is deliberately UNSCREENED: screening here would zero
    out the whole ``exclusions`` block silently, with no error and no failing test.
    """

    # Lazy: the analysis stack (pandas / scipy / statsmodels / lifelines) loads only here,
    # where it is actually used, so data-collection subcommands never require it (top-of-file).
    from analysis.pipeline import write_report

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"ABORT: dataset {dataset_path} not found")
        raise SystemExit(1)
    out_path = Path(args.out) if args.out else dataset_path.parent / "analysis_report.json"
    if out_path.exists() and not args.overwrite:
        print(
            f"ABORT: {out_path} already exists. Re-run with --overwrite to replace it, or "
            "pass --out to write elsewhere."
        )
        raise SystemExit(1)

    results, omitted = _analysis_results(args, "cli.analyze")
    report = write_report(results, out_path)
    print(f"analysis report -> {report}")
    if omitted:
        print(f"NOTE: sections omitted (inputs unavailable): {', '.join(omitted)}")


def _analysis_results(args, source: str):
    """Load and validate ``args.dataset`` plus its inputs and run the full pipeline.

    Shared by ``analyze`` and ``analysis-corrections`` so both produce the report the same
    way. Returns ``(results, sections_omitted)``; any invalid input aborts with exit 1.
    """

    from analysis import dataset as ds_mod
    from analysis.pipeline import run_all

    dataset_path = Path(args.dataset)
    base = dataset_path.parent
    frame = ds_mod.load_csv(dataset_path)
    # REQUIRED_COLUMNS, not TRIAL_COLUMNS: an archived dataset written before a column
    # was added must stay analyzable. Optional-on-read columns are simply ABSENT from the
    # pandas frame here (pandas cannot substitute a sentinel for a column that is not in
    # the file); ``analysis.dataset.action_trace_recorded`` is the guard for that case.
    missing = [c for c in run_sheet.REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        print(f"ABORT: {dataset_path} is not a run-sheet CSV; missing columns {missing}")
        raise SystemExit(1)
    # A clock-validation run writes BOTH attacker clocks into one combined dataset. Pooling the
    # arms would analyze every sequence twice under two different clocks as if they were one
    # design, so refuse and point at the per-arm subsets it also wrote.
    if "clock_mode" in frame.columns:
        from measurement.models import NOT_RECORDED

        clocks = sorted({str(m) for m in frame["clock_mode"].dropna().unique()} - {NOT_RECORDED})
        if len(clocks) > 1:
            print(
                f"ABORT: {dataset_path} mixes attacker clocks {clocks} -- the combined output of "
                "a clock-validation run. Analyze one arm at a time (arm_modeled/"
                "trial_dataset.csv, arm_wall/trial_dataset.csv beside it), or compare the arms "
                "with `cli.py clock-report`."
            )
            raise SystemExit(1)

    uptime_path, uptime_note = _resolve_sidecar(
        args.uptime_events,
        args.no_uptime_events,
        base,
        "uptime_events.json",
        "--uptime-events",
    )
    rep_path, rep_note = _resolve_sidecar(
        args.repeatability,
        args.no_repeatability,
        base,
        "repeatability.csv",
        "--repeatability",
    )
    meta_path, meta_note = _resolve_sidecar(
        args.run_meta,
        args.no_run_meta,
        base,
        "run_meta.json",
        "--run-meta",
    )

    uptime_events = None
    if uptime_path is not None:
        parsed = json.loads(uptime_path.read_text(encoding="utf-8"))
        if not isinstance(parsed, list) or not all(isinstance(e, dict) for e in parsed):
            print(f"ABORT: {uptime_path} is not a JSON list of uptime-event objects")
            raise SystemExit(1)
        uptime_events = parsed
        if not uptime_events:
            uptime_note = f"{uptime_path} parsed to 0 events -> section omitted"

    rep_rows = None
    if rep_path is not None:
        try:
            rep_rows = run_sheet.read_csv(rep_path)
        except ValueError as exc:
            print(f"ABORT: {exc}")
            raise SystemExit(1) from exc
        if not rep_rows:
            rep_note = f"{rep_path} has 0 data rows -> section omitted"

    run_meta = None
    if meta_path is not None:
        try:
            parsed_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"ABORT: {meta_path} is not valid JSON ({exc})")
            raise SystemExit(1) from exc
        if not isinstance(parsed_meta, dict):
            print(f"ABORT: {meta_path} is not a run_meta object")
            raise SystemExit(1)
        run_meta = parsed_meta

    bench, bench_prov = _load_reconfig_bench(args.reconfig_bench)
    design_basis, manifest_path, manifest_note = _design_basis_from_manifest(base)

    omitted = []
    if not uptime_events:
        omitted.append("survival_uptime")
    if not rep_rows:
        omitted.append("reliability")

    provenance = _provenance(
        source,
        dataset=str(dataset_path),
        dataset_sha256=_sha256(dataset_path),
        uptime_events=str(uptime_path) if uptime_path else None,
        uptime_events_sha256=_sha256(uptime_path) if uptime_path else None,
        repeatability=str(rep_path) if rep_path else None,
        repeatability_sha256=_sha256(rep_path) if rep_path else None,
        # run_meta.json is not a report SECTION -- it only corroborates the exclusion
        # accounting -- so it is recorded on its own keys rather than folded into
        # ``notes``/``sections_omitted``, which describe the two reconstructible sections.
        run_meta=str(meta_path) if meta_path else None,
        run_meta_sha256=_sha256(meta_path) if meta_path else None,
        run_meta_note=meta_note,
        reconfig_benchmark=bench_prov["path"] if bench_prov else None,
        reconfig_benchmark_sha256=bench_prov["sha256"] if bench_prov else None,
        # The run's design (N per condition, run order) for the power and drift blocks.
        run_manifest=str(manifest_path) if manifest_path else None,
        run_manifest_sha256=_sha256(manifest_path) if manifest_path else None,
        run_manifest_note=manifest_note,
        sections_omitted=omitted,
        notes=[uptime_note, rep_note],
    )
    results = run_all(
        frame,
        uptime_events,
        rep_rows,
        provenance=provenance,
        sequence_exclusions=run_meta,
        reconfig_benchmark=bench,
        reconfig_benchmark_provenance=bench_prov,
        design_basis=design_basis,
    )
    return results, omitted


def _design_basis_from_manifest(base: Path):
    """The run's design from ``_run_manifest.json`` beside the dataset: (basis, path, note).

    Read for the power block's locked N and planning check and for the drift check's run
    order (the manifest records ``run_order`` only when it is not ``blocked``). A missing or
    unreadable manifest leaves the basis out, and the report says which.
    """

    path = base / "_run_manifest.json"
    if not path.exists():
        return None, None, "_run_manifest.json not found beside the dataset -> design basis omitted"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        cfg = doc["config"]
        basis = {
            "sequences_per_condition": int(cfg["sequences_per_condition"]),
            "attempts_per_sequence": int(cfg["attempts_per_sequence"]),
            "run_order": str(cfg.get("run_order") or "blocked"),
            "attacker_clock": str(cfg.get("attacker_clock") or "modeled"),
            "source": str(path),
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return (
            None,
            path,
            f"{path} could not be read as a run manifest ({exc}) -> design basis omitted",
        )
    return basis, path, f"auto-discovered {path}"


def _git_state() -> dict:
    """Best-effort commit of the code that produced a report (read-only git calls)."""

    import subprocess

    here = Path(__file__).resolve().parent

    def git(*cmd: str) -> str:
        return subprocess.run(
            ["git", "--no-optional-locks", *cmd],
            cwd=here,
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout.strip()

    try:
        return {
            "git_commit": git("rev-parse", "HEAD"),
            "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty_analysis_code": bool(
                git("status", "--porcelain", "--", "analysis", "cli.py")
            ),
        }
    except Exception:  # noqa: BLE001 - not a git checkout, or git unavailable
        return {"git_commit": None, "git_branch": None, "git_dirty_analysis_code": None}


def _code_fingerprint() -> dict:
    """SHA-256 of every analysis module and cli.py: identifies the code without git."""

    here = Path(__file__).resolve().parent
    files = sorted((here / "analysis").glob("*.py")) + [here / "cli.py"]
    per = {str(f.relative_to(here)): _sha256(f) for f in files}
    combined = hashlib.sha256("".join(f"{k}:{v}\n" for k, v in per.items()).encode()).hexdigest()
    return {"combined_sha256": combined, "files": per}


def cmd_analysis_corrections(args):
    """Corrected report + before -> after record for an archived run. Never overwrites.

    Re-runs the CURRENT pipeline on ``--dataset`` exactly as ``analyze`` does (same sidecar
    discovery, same ``--reconfig-bench``), writes the result as
    ``analysis_report_corrected_<stamp>.json`` NEXT TO the archived report, and compares it
    with ``--archived-report`` (the report of record) in ``analysis_corrections_<stamp>.json``
    and ``.md``: a before -> after table for every number the 2026-10-27 corrections change
    or add, plus a consistency check over every other number. The archived report is only
    read. Any of the three outputs that already exists aborts the command before any work;
    all three are then reserved by exclusive creation before the analysis runs, and removed
    again if anything fails, so a run either writes all three or leaves nothing behind.
    """

    dataset_path, archived_path = Path(args.dataset), Path(args.archived_report)
    for path, flag in ((dataset_path, "--dataset"), (archived_path, "--archived-report")):
        if not path.exists():
            print(f"ABORT: {flag} {path} not found")
            raise SystemExit(1)
    out_dir = Path(args.out_dir) if args.out_dir else dataset_path.parent
    stamp = args.stamp or datetime.now(timezone.utc).strftime("%Y%m%d")
    targets = {
        "corrected_report": out_dir / f"analysis_report_corrected_{stamp}.json",
        "corrections_json": out_dir / f"analysis_corrections_{stamp}.json",
        "corrections_md": out_dir / f"analysis_corrections_{stamp}.md",
    }
    existing = [str(t) for t in targets.values() if t.exists()]
    if existing:
        print(
            f"ABORT: {', '.join(existing)} already exist(s); this command never overwrites. "
            "Pass a different --stamp or --out-dir."
        )
        raise SystemExit(1)
    if archived_path.resolve() in {t.resolve() for t in targets.values()}:
        print("ABORT: --archived-report must not be one of this command's outputs")
        raise SystemExit(1)
    try:
        archived = json.loads(archived_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"ABORT: {archived_path} is not valid JSON ({exc})")
        raise SystemExit(1) from exc
    dataset_sha = _sha256(dataset_path)
    archived_sha = (archived.get("provenance") or {}).get("dataset_sha256")
    if archived_sha and archived_sha != dataset_sha:
        print(
            f"ABORT: {archived_path} was computed from a dataset with SHA-256 {archived_sha}, "
            f"not {dataset_path} ({dataset_sha}); a before -> after table across datasets "
            "would be meaningless."
        )
        raise SystemExit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    try:
        # Reserve all three names atomically ("x" fails if a file appeared since the check).
        for target in targets.values():
            with target.open("x", encoding="utf-8"):
                pass
            created.append(target)
        _write_corrections(args, archived, archived_path, dataset_path, dataset_sha, targets)
    except BaseException:
        for target in created:
            target.unlink(missing_ok=True)
        raise


def _write_corrections(args, archived, archived_path, dataset_path, dataset_sha, targets):
    """Fill the three reserved outputs of ``analysis-corrections`` (see that command)."""

    from analysis import corrections
    from analysis.pipeline import _sanitize, write_report

    results, omitted = _analysis_results(args, "cli.analysis-corrections")
    write_report(results, targets["corrected_report"])
    corrected = json.loads(targets["corrected_report"].read_text(encoding="utf-8"))

    prov = corrected.get("provenance") or {}
    inputs = {
        "dataset": {"path": str(dataset_path), "sha256": dataset_sha},
        "archived_report": {"path": str(archived_path), "sha256": _sha256(archived_path)},
        "corrected_report": {
            "path": str(targets["corrected_report"]),
            "sha256": _sha256(targets["corrected_report"]),
        },
    }
    for key in ("uptime_events", "repeatability", "run_meta", "reconfig_benchmark"):
        if prov.get(key):
            inputs[key] = {"path": prov[key], "sha256": prov.get(f"{key}_sha256")}
    record = corrections.build(
        archived,
        corrected,
        dataset_sha256=dataset_sha,
        inputs=inputs,
        provenance={
            "generated_at": prov.get("generated_at"),
            "platform": prov.get("platform"),
            **_git_state(),
            "analysis_code": _code_fingerprint(),
            "library_versions": prov.get("library_versions"),
            "stack_matches_lock": prov.get("stack_matches_lock"),
            "lock_mismatches": prov.get("lock_mismatches"),
            "archived_report_platform": (archived.get("provenance") or {}).get("platform"),
            "archived_report_library_versions": (archived.get("provenance") or {}).get(
                "library_versions"
            ),
        },
    )
    findings: list = []
    safe = _sanitize(record, "", findings)
    targets["corrections_json"].write_text(
        json.dumps(safe, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8"
    )
    targets["corrections_md"].write_text(corrections.render_markdown(safe), encoding="utf-8")
    for name, path in targets.items():
        print(f"{name} -> {path}")
    cc = record["consistency_check"]
    print(
        f"consistency check: {cc['n_changed_unexpected']} value(s) changed outside the "
        f"intended corrections ({cc['n_numbers_compared']} compared)"
    )
    if cc["n_changed_unexpected"]:
        print(
            "NOTE: see the Consistency check section; on a platform other than the archived "
            "report's, mixed-model blocks differ numerically."
        )
    if omitted:
        print(f"NOTE: sections omitted (inputs unavailable): {', '.join(omitted)}")


# --------------------------------------------------------------------------- #
# Clock validation (real-timing pilot): modeled vs wall attacker clock, paired
# --------------------------------------------------------------------------- #

_CLOCK_REPORT_JSON = "clock_validation_report.json"
_CLOCK_REPORT_MD = "clock_validation_summary.md"


def _clock_validation_payload(config: ExperimentConfig, backend: str, live: bool) -> dict:
    """Fingerprint of a clock-validation run: the config's run-defining fields + its design.

    Backend and live/offline mode are part of it, so a stub smoke run can never be "resumed"
    as the real pilot (or a halted real pilot finished on the stub). The decision criteria are
    NOT part of it: they are locked separately in the manifest at the first start (see
    ``_lock_clock_criteria``), so a changed criterion is refused with its own message.
    """

    from harness import checkpoint as ckpt_mod
    from harness.config import CLOCK_VALIDATION_ARMS

    payload = ckpt_mod.fingerprint_payload(config)
    payload["clock_validation"] = {
        "arms": list(CLOCK_VALIDATION_ARMS),
        "arm_order": "alternating_by_sequence_index",
        "conditions": list(config.clock_validation.conditions),
        "sequences_per_condition": config.clock_validation.sequences_per_condition,
        "record_timing": True,
        "backend": backend,
        "live": bool(live),
        # The wall arm dates a scan's observations to the instant it probed the targets host
        # (timing format v2); the 2026-10-26 pilot (v1) dated them to the scan's return, so a
        # checkpoint written by that pilot can never be resumed by this code.
        "wall_dating": "probe_time",
    }
    return payload


def _clock_criteria(config: ExperimentConfig):
    from analysis.clock_validation import Criteria  # stdlib-only module (no analysis stack)

    return Criteria.from_dict(config.clock_validation.criteria_dict())


_LOCKED_CRITERIA_KEY = "clock_validation_criteria"


def _lock_clock_criteria(checkpoint, criteria) -> dict:
    """Return the criteria locked at the run's first start; refuse to resume under others.

    ``_setup_checkpoint`` writes them into the manifest when the run first starts (before any
    data exist). On every later invocation the configured criteria must equal the locked ones:
    the decision rule of a validation must not move after its data are in.
    """

    locked = (checkpoint.manifest() or {}).get(_LOCKED_CRITERIA_KEY)
    current = criteria.as_dict()
    if locked != current:
        print(
            f"ABORT: the decision criteria in the config {current} differ from those locked when "
            f"this run first started {locked} ({checkpoint.manifest_path}). The criteria are "
            "fixed before data collection and cannot change mid-run. Restore them in the "
            "config, or start a separate run with --fresh in a NEW --outdir."
        )
        raise SystemExit(4)
    return locked


def _find_recorded_criteria(dataset: Path):
    """Criteria recorded for a dataset's run, and where: manifest first, then run_meta.

    Looked for beside the dataset and one level up, so an arm subset (``arm_wall/``) finds the
    paired run's record. Returns ``(mapping, source)`` or ``(None, None)``.
    """

    for folder in (dataset.parent, dataset.parent.parent):
        manifest = folder / "_run_manifest.json"
        if manifest.exists():
            try:
                found = json.loads(manifest.read_text(encoding="utf-8")).get(_LOCKED_CRITERIA_KEY)
            except (json.JSONDecodeError, AttributeError):
                found = None
            if found is not None:
                return found, f"locked in the checkpoint manifest at first start ({manifest})"
    for folder in (dataset.parent, dataset.parent.parent):
        meta = folder / "run_meta.json"
        if meta.exists():
            try:
                found = (
                    json.loads(meta.read_text(encoding="utf-8"))
                    .get("clock_validation", {})
                    .get("criteria")
                )
            except (json.JSONDecodeError, AttributeError):
                found = None
            if found is not None:
                return found, f"recorded in {meta}"
    return None, None


def _recorded_run_facts(dataset: Path) -> dict:
    """Backend / live mode / model of the run that wrote ``dataset`` (for report provenance)."""

    for folder in (dataset.parent, dataset.parent.parent):
        manifest = folder / "_run_manifest.json"
        if not manifest.exists():
            continue
        try:
            cfg = json.loads(manifest.read_text(encoding="utf-8")).get("config", {})
        except (json.JSONDecodeError, AttributeError):
            continue
        cv = cfg.get("clock_validation", {}) if isinstance(cfg, dict) else {}
        if cv:
            return {
                "backend": cv.get("backend"),
                "live": cv.get("live"),
                "model_id": cfg.get("attacker_model_id"),
            }
    return {}


def _write_dataset_files(outdir: Path, rows, uptime_events) -> None:
    """The trial dataset (CSV + JSONL), its schema, and the uptime-event sidecar."""

    outdir.mkdir(parents=True, exist_ok=True)
    run_sheet.write_csv(rows, outdir / "trial_dataset.csv")
    run_sheet.write_jsonl(rows, outdir / "trial_dataset.jsonl")
    run_sheet.write_schema(outdir / "schema.json")
    (outdir / "uptime_events.json").write_text(json.dumps(uptime_events), encoding="utf-8")


def _write_clock_report(rows, criteria, outdir: Path, provenance: dict, out_json=None):
    """Build + write the clock-validation report and its Markdown summary; return both."""

    from analysis.clock_validation import build_report, format_summary

    report = build_report(rows, criteria, provenance=provenance)
    out_json = Path(out_json) if out_json else outdir / _CLOCK_REPORT_JSON
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = format_summary(report)
    out_md = out_json.with_name(out_json.stem.replace("_report", "_summary") + ".md")
    if out_md == out_json:  # pragma: no cover - defensive (an --out without "_report")
        out_md = out_json.with_suffix(".md")
    out_md.write_text(summary, encoding="utf-8")
    return report, summary, out_json, out_md


def cmd_validate_clock(args):
    """Run the real-timing validation pilot: every sequence under BOTH attacker clocks.

    Each (condition, sequence) runs once with the modeled clock and once with the wall clock,
    same seed and timeline slot (identical MTD schedule and phase), arm order alternating by
    sequence index; real action timing is recorded on both arms. The run is checkpointed and
    resumable exactly like ``run`` (re-run the same command; ``--fresh`` to start over); the
    design, backend, live/offline mode and decision criteria are all locked at the first
    start. Writes the combined dataset, per-arm subsets (``arm_modeled/``, ``arm_wall/`` --
    each analyzable with ``cli.py analyze``), ``run_meta.json``, and the clock-validation
    report. The report needs only the standard library, so it is produced on the control host.
    """

    import dataclasses

    from harness.config import CLOCK_VALIDATION_ARMS
    from harness.experiment_runner import RunHalted

    config = ExperimentConfig.load(args.config)
    cv = config.clock_validation
    overrides = {}
    if args.sequences is not None:
        overrides["sequences_per_condition"] = args.sequences
    if args.conditions:
        overrides["conditions"] = tuple(args.conditions)
    if overrides:
        try:
            config = dataclasses.replace(
                config, clock_validation=dataclasses.replace(cv, **overrides)
            )
        except ValueError as exc:
            print(f"ABORT: {exc}")
            raise SystemExit(2) from exc
    cv = config.clock_validation
    if args.backend == "mock":
        print(
            "ABORT: the clock-validation pilot needs the planner (--backend stub offline, or "
            "--backend anthropic for the real pilot); the mock attacker has no real actions "
            "to time."
        )
        raise SystemExit(2)
    criteria = _clock_criteria(config)

    n_trials = (
        len(cv.conditions)
        * cv.sequences_per_condition
        * config.attempts_per_sequence
        * len(CLOCK_VALIDATION_ARMS)
    )
    if args.dry_run:
        if args.live:
            from harness import live_runner

            print(_format_plan(live_runner.dry_run_plan(config, mode="clock_validation")))
        print(
            f"clock validation: arms {', '.join(CLOCK_VALIDATION_ARMS)} (same seed per sequence, "
            f"order alternating by sequence index); {len(cv.conditions)} conditions x "
            f"{cv.sequences_per_condition} sequences x {config.attempts_per_sequence} attempts x "
            f"{len(CLOCK_VALIDATION_ARMS)} arms = {n_trials} trials; backend={args.backend}; "
            f"live={bool(args.live)}"
        )
        print(f"criteria: {criteria.as_dict()}")
        return

    env_factory, executor_factory, provisioner = OfflineEnvironment, None, None
    if args.live:
        config, env_factory, executor_factory, provisioner = _wire_live(config)

    outdir = Path(args.outdir or "data/clock_validation")
    checkpoint = _setup_checkpoint(
        config,
        outdir,
        args.fresh,
        payload=_clock_validation_payload(config, args.backend, args.live),
        extra={_LOCKED_CRITERIA_KEY: criteria.as_dict(), **_first_start_record()},
    )
    locked = _lock_clock_criteria(checkpoint, criteria)
    resumed = bool(checkpoint.completed_ids())
    base = build_attacker(config, backend=args.backend)
    if executor_factory is not None:
        base.executor_factory = executor_factory  # live: resolve via env.executor
    arms = [
        (label, dataclasses.replace(base, clock_mode=label, record_timing=True))
        for label in CLOCK_VALIDATION_ARMS
    ]
    profile = default_profile(config.vuln_profile_version)
    print(
        f"clock validation: {n_trials} trials ({len(cv.conditions)} conditions x "
        f"{cv.sequences_per_condition} sequences x {config.attempts_per_sequence} attempts x "
        f"2 arms), backend={args.backend}, live={bool(args.live)} -> {outdir}",
        flush=True,
    )
    started_utc = _utc_now()
    code = _git_provenance()  # the code this process loaded (see _generate)
    started = time.perf_counter()
    try:
        data = run_experiment_dataset(
            config,
            base,
            profile,
            mode="clock_validation",
            fault_rate=args.fault_rate,
            env_factory=env_factory,
            checkpoint=checkpoint,
            on_trial_failure=config.run.on_trial_failure,
            max_consecutive_exclusions=config.run.max_consecutive_exclusions,
            progress=lambda line: print(line, flush=True),
            arms=arms,
            usage_snapshot=_usage_snapshot_fn(base),
        )
    except RunHalted as exc:
        print(
            f"HALTED at sequence {exc.sequence_id} ({exc.reason}). Checkpoint written to "
            f"{outdir}. Re-run the same command to resume after the incident clears."
        )
        raise SystemExit(5) from exc
    wall_s = time.perf_counter() - started
    finished_utc = _utc_now()

    run_facts = {
        "backend": args.backend,
        "live": bool(args.live),
        "model_id": config.attacker.model_id,
    }
    design = {
        "arms": list(CLOCK_VALIDATION_ARMS),
        "arm_order": "alternating_by_sequence_index (even: modeled first; odd: wall first)",
        "conditions": list(cv.conditions),
        "sequences_per_condition": cv.sequences_per_condition,
        "criteria": locked,
        "criteria_locked_in": str(checkpoint.manifest_path),
        **run_facts,
    }
    _write_dataset_files(outdir, data.trial_rows, data.uptime_events)
    _write_run_meta(
        outdir,
        data,
        args.backend,
        env_factory,
        wall_s,
        base,
        provisioner,
        config,
        resumed,
        fault_rate=args.fault_rate,
        started_utc=started_utc,
        finished_utc=finished_utc,
        code=code,
        checkpoint=checkpoint,
        extra={
            "clock_validation": design,
            # Both arms ran, both timed -- not the config's single clock.
            "attacker_clock": {"arms": list(CLOCK_VALIDATION_ARMS), "record_timing": True},
        },
    )
    from harness.experiment_runner import ExperimentDataset

    for label in CLOCK_VALIDATION_ARMS:
        arm_rows = [r for r in data.trial_rows if r.clock_mode == label]
        tids = {r.trial_id for r in arm_rows}
        arm_data = ExperimentDataset(
            trial_rows=arm_rows,
            uptime_events=[e for e in data.uptime_events if e.get("trial_id") in tids],
            excluded_sequences=[
                x
                for x in (data.excluded_sequences or [])
                if str(x.get("sequence_id", "")).endswith(f"-{label}")
            ],
        )
        arm_dir = outdir / f"arm_{label}"
        _write_dataset_files(arm_dir, arm_data.trial_rows, arm_data.uptime_events)
        _write_run_meta(
            arm_dir,
            arm_data,
            args.backend,
            env_factory,
            wall_s,
            None,  # token usage is shared by both arms: recorded once, in ../run_meta.json
            None,
            config,
            resumed,
            fault_rate=args.fault_rate,
            started_utc=started_utc,
            finished_utc=finished_utc,
            code=code,
            checkpoint=checkpoint,
            extra={
                "clock_validation_arm": label,
                "clock_validation": design,
                "attacker_clock": {"clock": label, "record_timing": True},
                "note": "one arm of a paired clock-validation run; wall_clock_s, token usage "
                "and provisioning are for the paired run as a whole (see ../run_meta.json)",
            },
        )
    print(
        f"wrote {len(data.trial_rows)} trials (both arms) -> {outdir}; per-arm subsets in "
        f"{outdir / 'arm_modeled'} and {outdir / 'arm_wall'}"
    )
    if args.no_report:
        return
    from analysis.clock_validation import Criteria

    provenance = {
        "source": "cli.validate-clock",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": str(outdir / "trial_dataset.csv"),
        "dataset_sha256": _sha256(outdir / "trial_dataset.csv"),
        "criteria_source": (
            f"locked in the checkpoint manifest at first start ({checkpoint.manifest_path})"
        ),
        **run_facts,
    }
    _report, summary, out_json, out_md = _write_clock_report(
        data.trial_rows, Criteria.from_dict(locked), outdir, provenance
    )
    print()
    print(summary)
    print(f"clock-validation report -> {out_json} (+ {out_md.name})")


def cmd_clock_report(args):
    """Regenerate the clock-validation report from a written trial dataset (no re-run).

    The criteria are the ones locked in the run's checkpoint manifest at its first start
    (looked for beside the dataset and one level up, so an arm subset finds them), else those
    recorded in run_meta.json; ``--config`` supplies them only when no record exists, and the
    report says which source was used. Standard library only (runs on the control host).
    """

    dataset = Path(args.dataset)
    if not dataset.exists():
        print(f"ABORT: dataset {dataset} not found")
        raise SystemExit(1)
    out_json = Path(args.out) if args.out else dataset.parent / _CLOCK_REPORT_JSON
    if out_json.exists() and not args.overwrite:
        print(
            f"ABORT: {out_json} already exists. Re-run with --overwrite to replace it, or pass "
            "--out to write elsewhere."
        )
        raise SystemExit(1)
    try:
        rows = run_sheet.read_csv(dataset)
    except ValueError as exc:
        print(f"ABORT: {exc}")
        raise SystemExit(1) from exc

    from analysis.clock_validation import Criteria

    recorded, source = _find_recorded_criteria(dataset)
    try:
        if recorded is not None:
            criteria = Criteria.from_dict(recorded)
        else:
            criteria = _clock_criteria(ExperimentConfig.load(args.config))
            source = f"config {args.config} (no criteria recorded for this dataset)"
    except ValueError as exc:
        print(f"ABORT: unusable decision criteria ({source or args.config}): {exc}")
        raise SystemExit(1) from exc
    provenance = {
        "source": "cli.clock-report",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": str(dataset),
        "dataset_sha256": _sha256(dataset),
        "criteria_source": source,
        **_recorded_run_facts(dataset),
    }
    _report, summary, out_json, out_md = _write_clock_report(
        rows, criteria, dataset.parent, provenance, out_json=out_json
    )
    print(summary)
    print(f"clock-validation report -> {out_json} (+ {out_md.name})")


_FASTSCAN_JSON = "fastscan_comparison.json"


def _run_scan_facts(dataset: Path) -> dict:
    """The scan scope and code version a run recorded in its run_meta.json, when present."""

    meta_path = dataset.parent / "run_meta.json"
    if not meta_path.is_file():
        return {"run_meta": None}
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"run_meta": str(meta_path), "note": "run_meta.json could not be read"}
    if not isinstance(meta, dict):
        return {"run_meta": str(meta_path), "note": "run_meta.json is not a JSON object"}
    prov = meta.get("provenance")
    prov = prov if isinstance(prov, dict) else {}
    return {
        "run_meta": str(meta_path),
        "scan_scope": prov.get("scan_scope"),
        "git_commit": prov.get("git_commit"),
        "run_order": prov.get("run_order"),
    }


#: The scan scope each side of the comparison must have recorded, when it recorded one.
_FASTSCAN_EXPECTED_SCOPE = {"main": "subnet", "fast": "target_host"}


def cmd_fastscan_compare(args):
    """Set the fast-scan check against the main study (planned sensitivity; estimates only).

    Decision D1 of 2026-10-29 (``docs/rerun_runbook.md`` §1 and §4e): per shared condition, the
    compromise proportion and the 300-s restricted mean time to compromise under each scan,
    the fast - subnet difference, and the moving conditions' benefit over static under each
    scan, all with 95% sequence-cluster-bootstrap intervals. No hypothesis is decided and no
    p-value is reported. Reads each dataset's run_meta.json (when present) and refuses a run
    whose recorded scan scope contradicts its flag (--main must be the subnet sweep, --fast the
    targets-host scan), so swapped or duplicated datasets cannot give reversed differences.
    Writes nothing and exits 1 when the comparison cannot be computed.
    """

    from analysis import dataset as ds_mod
    from analysis import fastscan
    from analysis.pipeline import write_report

    main_path, fast_path = Path(args.main), Path(args.fast)
    for label, path in (("--main", main_path), ("--fast", fast_path)):
        if not path.is_file():
            what = "is a directory; pass its trial_dataset.csv" if path.is_dir() else "not found"
            print(f"ABORT: {label} dataset {path} {what}")
            raise SystemExit(1)
    out_path = Path(args.out) if args.out else fast_path.parent / _FASTSCAN_JSON
    if out_path.exists() and not args.overwrite:
        print(
            f"ABORT: {out_path} already exists. Re-run with --overwrite to replace it, or pass "
            "--out to write elsewhere."
        )
        raise SystemExit(1)
    facts = {"main": _run_scan_facts(main_path), "fast": _run_scan_facts(fast_path)}
    for side, flag in (("main", "--main"), ("fast", "--fast")):
        scope, want = facts[side].get("scan_scope"), _FASTSCAN_EXPECTED_SCOPE[side]
        if scope is not None and scope != want:
            print(
                f"ABORT: the {flag} dataset recorded scan scope {scope!r}; {flag} takes the "
                f"run with scan scope {want!r} (--main the main study, --fast the fast-scan "
                "check). Check that the two datasets are not swapped."
            )
            raise SystemExit(1)
    unrecorded = [f"--{side}" for side in ("main", "fast") if facts[side].get("scan_scope") is None]
    frames = {}
    for label, path in (("main", main_path), ("fast", fast_path)):
        try:
            frame = ds_mod.load_csv(path)
        except (OSError, ValueError) as exc:
            print(f"ABORT: {path} could not be read as a CSV: {exc}")
            raise SystemExit(1) from exc
        missing = [c for c in run_sheet.REQUIRED_COLUMNS if c not in frame.columns]
        if missing:
            print(f"ABORT: {path} is not a run-sheet CSV; missing columns {missing}")
            raise SystemExit(1)
        frames[label] = ds_mod.prepare(frame)
    report = fastscan.compare(frames["main"], frames["fast"])
    if "by_condition" not in report:
        print(f"ABORT: the comparison could not be computed: {report.get('note')}")
        raise SystemExit(1)
    report["provenance"] = {
        "source": "cli.fastscan-compare",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "main_dataset": str(main_path),
        "main_dataset_sha256": _sha256(main_path),
        "fast_dataset": str(fast_path),
        "fast_dataset_sha256": _sha256(fast_path),
        "runs": facts,
        "scan_scope_unrecorded": unrecorded,
    }
    written = write_report(report, out_path)
    for line in fastscan.summary_lines(report):
        print(line)
    if unrecorded:
        print(
            f"NOTE: no scan scope recorded for {', '.join(unrecorded)}; the comparison trusts "
            "the flags."
        )
    print(f"fast-scan comparison -> {written}")


def main(argv=None):
    # Front-load a gitignored .secrets file (e.g. ANTHROPIC_API_KEY) so the
    # operator need not export it in the shell. Real env vars still take precedence.
    from harness.secrets import load_secrets

    load_secrets()

    parser = argparse.ArgumentParser(description="MTD instrument offline pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, fn in (("run", cmd_run), ("pilot", cmd_pilot)):
        p = sub.add_parser(name)
        p.add_argument("--config", default="harness/experiment_config.example.yaml")
        p.add_argument("--outdir", default=None)
        p.add_argument(
            "--fault-rate",
            type=float,
            default=None,
            dest="fault_rate",
            help="SIMULATED row-level fault injection rate (default: the config's run.fault_rate, "
            f"else {_LEGACY_FAULT_RATE}, the rate of every archived run)",
        )
        if name == "run":
            p.add_argument(
                "--sequences",
                type=int,
                default=None,
                help="override design.sequences_per_condition (e.g. a smoke run)",
            )
            p.add_argument(
                "--conditions",
                nargs="+",
                default=None,
                help="override design.conditions (e.g. a smoke run)",
            )
        p.add_argument(
            "--backend",
            default="mock",
            choices=["mock", "stub", "anthropic"],
            help="attacker backend (mock/stub run offline; anthropic calls the model)",
        )
        p.add_argument(
            "--live",
            action="store_true",
            help="run against the real testbed (opt-in; fails closed without a deployed "
            "testbed + verified egress split). Offline is the default.",
        )
        p.add_argument(
            "--dry-run",
            action="store_true",
            dest="dry_run",
            help="exercise the live path with the mock executor and print the plan "
            "(conditions, trials, targets, egress map) without touching the network",
        )
        p.add_argument(
            "--fresh",
            action="store_true",
            help="discard any existing data/full checkpoint and start the run clean "
            "(full run only; required to start over after the run-defining config changed)",
        )
        p.set_defaults(func=fn)

    pv = sub.add_parser(
        "validate-clock",
        help="real-timing validation pilot: run every sequence under BOTH attacker clocks "
        "(modeled + wall, same seed, alternating order) and report whether the modeled clock "
        "reproduces the real race (docs/clock_validation_runbook.md)",
    )
    pv.add_argument("--config", default="harness/experiment_config.example.yaml")
    pv.add_argument("--outdir", default=None, help="default: data/clock_validation")
    pv.add_argument(
        "--backend",
        default="stub",
        choices=["stub", "anthropic"],
        help="planner backend (stub runs offline; anthropic calls the model). The mock "
        "attacker has no real actions to time and is not accepted.",
    )
    pv.add_argument(
        "--live",
        action="store_true",
        help="run against the real testbed (fails closed at the isolation preflight)",
    )
    pv.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="print the plan (with --live: the live plan + egress map) without running",
    )
    pv.add_argument(
        "--fresh",
        action="store_true",
        help="discard an existing checkpoint in --outdir and start clean",
    )
    pv.add_argument(
        "--fault-rate",
        type=float,
        default=0.0,
        dest="fault_rate",
        help="SIMULATED row-level fault injection (default 0: none -- this pilot measures timing)",
    )
    pv.add_argument(
        "--sequences",
        type=int,
        default=None,
        help="override modes.clock_validation.sequences_per_condition",
    )
    pv.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help="override modes.clock_validation.conditions",
    )
    pv.add_argument(
        "--no-report",
        action="store_true",
        dest="no_report",
        help="collect only; run `cli.py clock-report` later",
    )
    pv.set_defaults(func=cmd_validate_clock)

    pr = sub.add_parser(
        "clock-report",
        help="regenerate the clock-validation report from a written trial dataset (no re-run; "
        "standard library only)",
    )
    pr.add_argument("--dataset", required=True, help="path to a run-sheet trial_dataset.csv")
    pr.add_argument(
        "--config",
        default="harness/experiment_config.example.yaml",
        help="criteria source ONLY when no run_meta.json beside the dataset records them",
    )
    pr.add_argument(
        "--out",
        default=None,
        help=f"report path (default: {_CLOCK_REPORT_JSON} beside the dataset)",
    )
    pr.add_argument("--overwrite", action="store_true", help="replace an existing report")
    pr.set_defaults(func=cmd_clock_report)

    pf = sub.add_parser(
        "fastscan-compare",
        help="set the fast-scan check against the main study (planned sensitivity; estimates "
        "with 95%% intervals, no tests)",
    )
    pf.add_argument("--main", required=True, help="the main study's trial_dataset.csv")
    pf.add_argument("--fast", required=True, help="the fast-scan check's trial_dataset.csv")
    pf.add_argument(
        "--out",
        default=None,
        help=f"report path (default: {_FASTSCAN_JSON} beside the fast-scan dataset)",
    )
    pf.add_argument("--overwrite", action="store_true", help="replace an existing report")
    pf.set_defaults(func=cmd_fastscan_compare)

    pa = sub.add_parser(
        "analyze",
        help="regenerate the FULL analysis report from a written trial dataset CSV "
        "(no trials are re-run)",
    )
    pa.add_argument("--dataset", required=True, help="path to a run-sheet trial_dataset.csv")
    pa.add_argument(
        "--out",
        default=None,
        help="report path (default: analysis_report.json beside the dataset)",
    )
    pa.add_argument(
        "--uptime-events",
        default=None,
        dest="uptime_events",
        help="service-uptime event log for the survival_uptime section "
        "(default: uptime_events.json beside the dataset)",
    )
    pa.add_argument(
        "--no-uptime-events",
        action="store_true",
        dest="no_uptime_events",
        help="skip the survival_uptime section even if uptime_events.json is present",
    )
    pa.add_argument(
        "--repeatability",
        default=None,
        help="repeatability run sheet for the reliability section "
        "(default: repeatability.csv beside the dataset)",
    )
    pa.add_argument(
        "--no-repeatability",
        action="store_true",
        dest="no_repeatability",
        help="skip the reliability section even if repeatability.csv is present",
    )
    pa.add_argument(
        "--run-meta",
        default=None,
        dest="run_meta",
        help="run_meta.json to reconcile the SEQUENCE-level exclusion counter against the "
        "row-level one (default: run_meta.json beside the dataset)",
    )
    pa.add_argument(
        "--no-run-meta",
        action="store_true",
        dest="no_run_meta",
        help="ignore run_meta.json; exclusions.sequence_level then reports the "
        "sequence-level count as unavailable instead of reconciling it",
    )
    pa.add_argument(
        "--reconfig-bench",
        default=None,
        dest="reconfig_bench",
        help="reconfiguration-cost benchmark CSV (docs/results/benchmark/reconfig_cost.csv) for "
        "the H05 benefit/cost profile test in rq5.benchmark_profile (not auto-discovered)",
    )
    pa.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing report at the output path (refuses by default so an "
        "archived report of record is never clobbered accidentally)",
    )
    pa.set_defaults(func=cmd_analyze)

    pc = sub.add_parser(
        "analysis-corrections",
        help="re-analyze an archived run with the current code and write a before -> after "
        "record against its archived report (never overwrites)",
    )
    pc.add_argument("--dataset", required=True, help="the archived run's trial_dataset.csv")
    pc.add_argument(
        "--archived-report",
        required=True,
        dest="archived_report",
        help="the archived report of record to compare against (only read), e.g. "
        "analysis_report_lockstack.json",
    )
    pc.add_argument(
        "--reconfig-bench",
        default=None,
        dest="reconfig_bench",
        help="reconfiguration-cost benchmark CSV for the RQ5 profile test "
        "(docs/results/benchmark/reconfig_cost.csv)",
    )
    pc.add_argument(
        "--out-dir",
        default=None,
        dest="out_dir",
        help="output directory (default: beside the dataset)",
    )
    pc.add_argument(
        "--stamp",
        default=None,
        help="date stamp in the output names (default: today, UTC, YYYYMMDD)",
    )
    # Sidecars are auto-discovered beside the dataset exactly as `analyze` does.
    pc.set_defaults(
        func=cmd_analysis_corrections,
        uptime_events=None,
        no_uptime_events=False,
        repeatability=None,
        no_repeatability=False,
        run_meta=None,
        no_run_meta=False,
    )

    pp = sub.add_parser(
        "preflight",
        help="assert the live-run isolation invariants (egress split, control-host "
        "forwarding off, no node egress) — fails closed",
    )
    pp.add_argument("--config", default="harness/experiment_config.example.yaml")
    pp.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="validate the intended egress map on paper (no live probing)",
    )
    pp.set_defaults(func=cmd_preflight)

    pd = sub.add_parser(
        "deploy",
        help="provision the testbed (site.yml) then re-seal + verify isolation — fails "
        "closed. Run on the control host with a filled-in --config + --inventory.",
    )
    pd.add_argument("--config", default="harness/experiment_config.example.yaml")
    pd.add_argument(
        "--mode",
        default="egress",
        choices=["egress", "mirror"],
        help="egress: open temporary provisioning egress around site.yml then re-seal; "
        "mirror: no egress (nodes use the monitoring-node mirror)",
    )
    pd.add_argument("--inventory", default="infra/ansible/inventory.ini")
    pd.set_defaults(func=cmd_deploy)

    pc = sub.add_parser(
        "cost-probe",
        help="profile the configured Anthropic attacker model's cost/latency/capability "
        "against the offline mock",
    )
    pc.add_argument("--config", default="harness/experiment_config.example.yaml")
    pc.add_argument("--outdir", default=None)
    pc.add_argument("--trials", type=int, default=24, help="total profiling trials")
    pc.add_argument("--seed", type=int, default=20260712)
    # Full-study projection multipliers.
    pc.add_argument("--conditions", type=int, default=6, help="conditions for cost projection")
    pc.add_argument(
        "--sequences",
        type=int,
        default=20,
        help="sequences/condition for projection (default 20 = conservative end of the "
        "16-20 provisional power target; override once pilot rho finalizes it)",
    )
    pc.add_argument("--attempts", type=int, default=8, help="attempts/sequence for projection")
    # Capability floor.
    pc.add_argument(
        "--floor",
        type=float,
        default=0.5,
        help="min static-baseline exploitation_success rate before flagging the "
        "attacker model as underpowered",
    )
    # Prices (USD per million tokens), Haiku 4.5 defaults.
    pc.add_argument("--price-input", type=float, default=1.00, dest="price_input")
    pc.add_argument("--price-cache-read", type=float, default=0.10, dest="price_cache_read")
    pc.add_argument("--price-cache-write", type=float, default=1.25, dest="price_cache_write")
    pc.add_argument("--price-output", type=float, default=5.00, dest="price_output")
    pc.set_defaults(func=cmd_cost_probe)

    pb = sub.add_parser(
        "reconfig-bench",
        help="measure the technique-intrinsic cost of ONE reconfiguration on the targets "
        "node, with replicates (the MEASURED replacement for the synthetic cost telemetry)",
    )
    pb.add_argument("--config", default="harness/experiment_config.example.yaml")
    pb.add_argument("--outdir", default=None)
    pb.add_argument(
        "--techniques",
        nargs="+",
        default=None,
        help="techniques to benchmark (default: the config's six conditions, in the documented "
        "order static_baseline, network_shuffling, port_rotation, endpoint_mutation, "
        "deceptive_presentation, combined_mtd_deception)",
    )
    pb.add_argument("--replicates", type=int, default=20, help="replicates per technique (>=20)")
    pb.add_argument("--testbed-dir", default="/opt/testbed", dest="testbed_dir")
    pb.add_argument(
        "--reconfigure-sh",
        default=None,
        dest="reconfigure_sh",
        help="path to reconfigure.sh (default: <testbed-dir>/reconfigure.sh)",
    )
    pb.add_argument(
        "--base-compose",
        default=None,
        dest="base_compose",
        help="base compose file, for image-digest provenance (default: "
        "<testbed-dir>/docker-compose.yml)",
    )
    pb.add_argument("--targets-host", default=None, dest="targets_host")
    pb.add_argument("--bind-host", default=None, dest="bind_host")
    pb.add_argument("--testbed-cidr", default=None, dest="testbed_cidr")
    pb.add_argument("--monitored-service", default="dvwa", dest="monitored_service")
    pb.add_argument("--probe-interval", type=float, default=0.1, dest="probe_interval")
    pb.add_argument(
        "--readiness-interval",
        type=float,
        default=0.5,
        dest="readiness_interval",
        help="poll interval (s) while waiting for every published service to answer after the "
        "untimed setup apply (outside the timed region)",
    )
    pb.add_argument(
        "--readiness-timeout",
        type=float,
        default=300.0,
        dest="readiness_timeout",
        help="bound (s) on the setup settle wait; a replicate that exceeds it is marked "
        "availability-unusable rather than recording a contaminated zero",
    )
    pb.add_argument("--seed", type=int, default=1)
    pb.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="validate config + containment and print the plan WITHOUT touching Docker/SSH "
        "(use to confirm the command resolves on the node before a real run)",
    )
    pb.set_defaults(func=cmd_benchmark)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except SafetyError as exc:
        # Fail closed: a missing CIDR allowlist or a violated egress split aborts.
        print(f"ABORT (safety): {exc}")
        raise SystemExit(2) from exc
    except ToolingError as exc:
        # Fail closed: a primitive's standard tooling is not ready on the executor node.
        print(f"ABORT (tooling): {exc}")
        raise SystemExit(2) from exc
    except NotImplementedError as exc:
        # Live components (real transport/route-check/provisioner) are staged.
        print(
            f"ABORT: live component not available in this environment: {exc}\n"
            "Use '--dry-run' to exercise the live path with the mock executor offline."
        )
        raise SystemExit(3) from exc


if __name__ == "__main__":
    main()
