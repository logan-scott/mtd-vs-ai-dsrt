"""Measurement core: turn one trial's inputs into one TrialRow.

This is the deterministic entry point the harness calls after a trial completes.
It correlates offensive output with ground truth, computes every dependent
variable defined in ``docs/metrics_specification.md``, and returns a fully
populated (or correctly censored/flagged) ``TrialRow``.
"""

from __future__ import annotations

from measurement.action_trace import count_actions, encode_trace
from measurement.correlation import verify_objectives
from measurement.decoy_engagement import decoy_engagement
from measurement.metrics import (
    availability_metrics,
    overhead_metrics,
    recon_metrics,
    security_outcomes,
)
from measurement.models import (
    NOT_RECORDED,
    AvailabilityEvent,
    ServiceTuple,
    TrialInputs,
    TrialRow,
)
from measurement.timing import timing_columns


def _union_discovered(inputs: TrialInputs) -> frozenset[ServiceTuple]:
    """The trial's discovered set = union over its recon-pass snapshots."""

    out: set[ServiceTuple] = set()
    for snap in inputs.recon_snapshots:
        out |= snap.discovered
    return frozenset(out)


def compute_trial(inputs: TrialInputs) -> TrialRow:
    """Compute the full trial-level row from bundled trial inputs."""

    md = inputs.metadata
    limit = md.trial_duration_limit_s

    # --- objective verification (drives exploit_success + TTC) ---
    objectives = verify_objectives(
        inputs.ground_truth, inputs.recon_snapshots, inputs.exploit_deliveries
    )

    # Per the research design, both TTC and TFVE are anchored at workflow
    # initiation == the first reconnaissance command.
    first_recon_ts = security_outcomes.first_recon_timestamp(inputs.recon_commands)
    ttc = security_outcomes.time_to_compromise(first_recon_ts, objectives, limit)
    tten = security_outcomes.time_to_enumeration(first_recon_ts, objectives, limit)
    tfve = security_outcomes.time_to_first_valid_exploit(
        first_recon_ts, inputs.exploit_deliveries, limit
    )
    aps_count, aps_censored = security_outcomes.attempts_per_success(inputs.exploit_deliveries)

    # --- reconnaissance metrics ---
    discovered = _union_discovered(inputs)
    f1 = (
        recon_metrics.recon_accuracy_f1(discovered, inputs.ground_truth)
        if inputs.recon_snapshots
        else None
    )
    churn = (
        recon_metrics.jaccard_distance(inputs.prev_recon_discovered, discovered)
        if inputs.prev_recon_discovered is not None
        else None
    )

    # --- operational overhead (relative to matching baseline reference) ---
    base = inputs.baseline_reference
    if base is not None:
        latency = overhead_metrics.latency_overhead_ms(inputs.workload_samples, base)
        cpu = overhead_metrics.cpu_overhead_pct(inputs.monitoring_samples, base)
        mem = overhead_metrics.memory_overhead_pct(inputs.monitoring_samples, base)
        net = overhead_metrics.network_overhead_pct(inputs.monitoring_samples, base)
    else:
        latency = cpu = mem = net = None

    event_volume = overhead_metrics.operational_event_volume(
        inputs.reconfig_events, md.start_timestamp, md.end_timestamp
    )
    availability = availability_metrics.service_availability_impact(inputs.workload_samples)

    # --- planner action trace (PURE INSTRUMENTATION; nothing below feeds a metric) ---
    # The terminal reason -- not the trace's emptiness -- decides whether a trace was
    # recorded: an attacker backend that records nothing hands over an EMPTY tuple with
    # the NOT_RECORDED sentinel, while a planner that genuinely made zero decisions hands
    # over an empty tuple with a real reason (e.g. ``no_action_budget``). Branching on
    # emptiness would collapse those two into one cell.
    terminal_reason = inputs.planner_terminal_reason or NOT_RECORDED
    if inputs.action_trace is None or terminal_reason == NOT_RECORDED:
        action_sequence = NOT_RECORDED
        n_actions = n_recon_actions = n_exploit_actions = None
    else:
        action_sequence = encode_trace(inputs.action_trace)
        counts = count_actions(inputs.action_trace)
        n_actions = counts.n_actions
        n_recon_actions = counts.n_recon_actions
        n_exploit_actions = counts.n_exploit_actions

    # --- decoy engagement (PURE INSTRUMENTATION; nothing below feeds a metric) ---
    # Derived from the deliveries this row already carries; see
    # ``measurement/decoy_engagement.py`` for the classification rule and the sentinels.
    decoys = decoy_engagement(inputs.exploit_deliveries)

    # --- attacker clock + real action timing (PURE INSTRUMENTATION; nothing below feeds a
    # metric). ``clock_mode`` is always recorded; the rest is null/NOT_RECORDED unless the
    # attacker recorded timing. See ``measurement/timing.py``.
    timing = timing_columns(inputs.timing, md.clock_mode)

    return TrialRow(
        # metadata / control
        trial_id=md.trial_id,
        condition=md.condition,
        technique=md.technique,
        attempt_index=md.attempt_index,
        sequence_id=md.sequence_id,
        block_id=md.block_id,
        randomization_seed=md.randomization_seed,
        start_timestamp=md.start_timestamp,
        end_timestamp=md.end_timestamp,
        attacker_model_id=md.attacker_model_id,
        attacker_model_version=md.attacker_model_version,
        temperature=md.temperature,
        context_window=md.context_window,
        system_prompt_hash=md.system_prompt_hash,
        max_actions=md.max_actions,
        trial_duration_limit_s=md.trial_duration_limit_s,
        vuln_profile_version=md.vuln_profile_version,
        testbed_config_checksum=md.testbed_config_checksum,
        tooling_versions=md.tooling_versions,
        mtd_cadence_s=md.mtd_cadence_s,
        monitoring_interval_s=md.monitoring_interval_s,
        attacker_max_retries=md.attacker_max_retries,
        attacker_request_timeout_s=md.attacker_request_timeout_s,
        attacker_tool_choice=md.attacker_tool_choice,
        # primary security outcomes
        exploit_success=int(objectives.exploit_success),
        exploitation_success=int(objectives.obj_b),  # co-primary: actual exploitation
        obj_a=int(objectives.obj_a),
        obj_b=int(objectives.obj_b),
        obj_c=int(objectives.obj_c),
        objective_count=objectives.objective_count,
        time_to_compromise_s=ttc.value,
        ttc_censored=int(ttc.censored),
        time_to_enumeration_s=tten.value,
        tten_censored=int(tten.censored),
        # secondary / diagnostic
        time_to_first_valid_exploit_s=tfve.value,
        tfve_censored=int(tfve.censored),
        recon_accuracy_f1=f1,
        attack_path_churn=churn,
        attempts_per_success=aps_count,
        aps_censored=int(aps_censored),
        # operational overhead
        latency_overhead_ms=latency,
        cpu_overhead_pct=cpu,
        memory_overhead_pct=mem,
        network_overhead_pct=net,
        operational_event_volume=event_volume,
        service_availability_impact=availability,
        # inclusion / exclusion
        excluded=int(md.excluded),
        exclusion_reason=md.exclusion_reason,
        # planner action trace (instrumentation only)
        action_sequence=action_sequence,
        n_actions=n_actions,
        n_recon_actions=n_recon_actions,
        n_exploit_actions=n_exploit_actions,
        planner_terminal_reason=terminal_reason,
        # decoy engagement (instrumentation only)
        first_exploit_target=decoys.first_exploit_target,
        first_decoy_delivery_ordinal=decoys.first_decoy_delivery_ordinal,
        n_decoy_targeted_deliveries=decoys.n_decoy_targeted_deliveries,
        # provenance: how the operational-cost columns above were produced (set by the harness)
        operational_cost_source=md.operational_cost_source,
        # attacker clock + real action timing (instrumentation only)
        **timing,
    )


def compute_availability_events(inputs: TrialInputs) -> list[AvailabilityEvent]:
    """Per-event uptime records for the service-uptime survival analysis."""

    return availability_metrics.availability_events(
        inputs.workload_samples, inputs.metadata.start_timestamp
    )
