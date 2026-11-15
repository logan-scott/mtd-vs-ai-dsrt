"""Cost / capability probe for the real Anthropic attacker against the offline mock.

Runs a small profiling batch of **real** Anthropic calls for whichever model the
config names as the attacker, with the attacker's actions still resolving against
the offline mock target (no live testbed, no network beyond the model API). It
records per-trial LLM call counts, the token-usage breakdown (uncached input,
cache-read, cache-write, output), wall-clock latency, and the ``exploit_success``
/ ``exploitation_success`` outcomes, then reports cost, latency, capability, and
a full-study cost projection.

Aggregation, projection, and the capability-floor check are pure functions of the
per-trial records so they can be unit-tested on synthetic usage without any API
calls. Only ``run_probe`` (invoked by the ``cost-probe`` CLI command) makes real
calls, and only when the user runs it with ``ANTHROPIC_API_KEY`` set.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from attacker.backends import AnthropicClient, build_attacker
from attacker.base import AttackerContext, AttemptRequest
from harness.config import ExperimentConfig
from harness.environment import OfflineEnvironment
from measurement import clock
from measurement.correlation import verify_objectives
from mtd.orchestrator import run_orchestration
from testbed.profile import default_profile

# Conditions profiled: the static baseline (capability floor) + one MTD arm.
PROBE_CONDITIONS = ("static_baseline", "endpoint_mutation")

_BASE_START = "2026-07-01T00:00:00Z"
_RECON_OFFSET_S = 30

# Default Haiku 4.5 prices, USD per million tokens (all overridable via CLI flags).
DEFAULT_PRICES = {
    "input": 1.00,  # uncached input
    "cache_read": 0.10,
    "cache_write": 1.25,
    "output": 5.00,
}


@dataclass(frozen=True)
class Projection:
    """Full-study design multipliers used for the cost projection."""

    conditions: int = 6
    sequences: int = 20  # conservative end of the 16-20 provisional power target
    attempts: int = 8

    @property
    def total_trials(self) -> int:
        return self.conditions * self.sequences * self.attempts


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile (q in 0..100). Empty -> 0.0."""

    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    rank = (q / 100.0) * (len(xs) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return float(xs[lo])
    frac = rank - lo
    return float(xs[lo] + (xs[hi] - xs[lo]) * frac)


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def trial_cost(record: dict, prices: dict) -> float:
    """Per-trial cost (USD) from the token breakdown and per-type prices."""

    return (
        record["uncached_input"] * prices["input"]
        + record["cache_read"] * prices["cache_read"]
        + record["cache_write"] * prices["cache_write"]
        + record["output"] * prices["output"]
    ) / 1_000_000.0


def _record_from_usage(
    trial_id: str,
    condition: str,
    usage: dict,
    llm_calls: int,
    latency_s: float,
    exploit_success: int,
    exploitation_success: int,
    prices: dict,
) -> dict:
    record = {
        "trial_id": trial_id,
        "condition": condition,
        "llm_calls": int(llm_calls),
        "uncached_input": int(usage.get("input_tokens", 0)),
        "cache_read": int(usage.get("cache_read_input_tokens", 0)),
        "cache_write": int(usage.get("cache_creation_input_tokens", 0)),
        "output": int(usage.get("output_tokens", 0)),
        "latency_s": float(latency_s),
        "exploit_success": int(exploit_success),
        "exploitation_success": int(exploitation_success),
    }
    record["cost_usd"] = trial_cost(record, prices)
    return record


def _run_one_trial(
    attacker,
    condition: str,
    config: ExperimentConfig,
    profile,
    trial_index: int,
    seed: int,
    prices: dict,
) -> dict:
    """Run a single offline attempt against a freshly orchestrated environment."""

    horizon = config.max_trial_duration_s
    window_start = clock.shift(_BASE_START, trial_index * horizon)
    recon_hint = clock.shift(window_start, _RECON_OFFSET_S)

    orchestration = run_orchestration(
        condition,
        profile,
        window_start,
        config.mtd_cadence_s,
        horizon,
        seed=seed ^ 0x1,
        rotation_phase=config.mtd_rotation_phase,
    )
    env = OfflineEnvironment(orchestration)
    req = AttemptRequest(
        attempt_index=0,
        window_start=window_start,
        recon_ts=recon_hint,
        environment=env,
        context=AttackerContext(),
        seed=seed ^ 0x2,
        max_actions=config.max_actions,
        trial_duration_limit_s=horizon,
    )

    client = attacker.client
    if isinstance(client, AnthropicClient):
        client.reset_usage()

    start = time.perf_counter()
    obs = attacker.attempt(req)
    latency_s = time.perf_counter() - start

    recon_ts = obs.recon_snapshots[0].as_of if obs.recon_snapshots else recon_hint
    ground_truth = env.ground_truth_at(recon_ts)
    objectives = verify_objectives(ground_truth, obs.recon_snapshots, obs.exploit_deliveries)

    usage = client.usage_totals() if isinstance(client, AnthropicClient) else {}
    calls = client.call_count if isinstance(client, AnthropicClient) else 0

    return _record_from_usage(
        trial_id=f"{condition}-t{trial_index}",
        condition=condition,
        usage=usage,
        llm_calls=calls,
        latency_s=latency_s,
        exploit_success=int(objectives.exploit_success),
        exploitation_success=int(objectives.obj_b),
        prices=prices,
    )


def run_probe(
    config: ExperimentConfig,
    trials: int,
    prices: dict,
    seed: int = 20260712,
    conditions: tuple[str, ...] = PROBE_CONDITIONS,
) -> list[dict]:
    """Run ``trials`` real profiling attempts, balanced across ``conditions``.

    Uses the configured (anthropic) backend + pinned model against the offline
    mock. Makes real API calls — callers must ensure ``ANTHROPIC_API_KEY`` is set.
    """

    attacker = build_attacker(config, backend="anthropic")
    profile = default_profile(config.vuln_profile_version)
    records: list[dict] = []
    for i in range(trials):
        condition = conditions[i % len(conditions)]
        records.append(
            _run_one_trial(
                attacker,
                condition,
                config,
                profile,
                trial_index=i,
                seed=(seed + i) & 0xFFFFFFFF,
                prices=prices,
            )
        )
    return records


def _token_stats(records: list[dict], key: str) -> dict:
    values = [float(r[key]) for r in records]
    return {"mean": _mean(values), "p90": percentile(values, 90)}


def cache_hit_rate(records: list[dict]) -> float:
    """Fraction of input-side tokens served from cache (aggregate over all trials)."""

    read = sum(r["cache_read"] for r in records)
    denom = sum(r["cache_read"] + r["uncached_input"] + r["cache_write"] for r in records)
    return float(read / denom) if denom else 0.0


def _rate(records: list[dict], key: str) -> float:
    return _mean([float(r[key]) for r in records]) if records else 0.0


def summarize(
    records: list[dict],
    prices: dict | None = None,
    projection: Projection | None = None,
    floor: float = 0.5,
    model_id: str | None = None,
) -> dict:
    """Aggregate per-trial records into the probe summary (pure function)."""

    prices = prices if prices is not None else DEFAULT_PRICES
    projection = projection if projection is not None else Projection()
    costs = [trial_cost(r, prices) for r in records]
    latencies = [float(r["latency_s"]) for r in records]
    calls = [float(r["llm_calls"]) for r in records]

    mean_cost = _mean(costs)
    static = [r for r in records if r["condition"] == "static_baseline"]
    floor_success = _rate(static, "exploitation_success")
    adequate = floor_success >= floor

    return {
        "n_trials": len(records),
        "conditions_run": sorted({r["condition"] for r in records}),
        "tokens_per_trial": {
            "uncached_input": _token_stats(records, "uncached_input"),
            "cache_read": _token_stats(records, "cache_read"),
            "cache_write": _token_stats(records, "cache_write"),
            "output": _token_stats(records, "output"),
        },
        "cache_hit_rate": cache_hit_rate(records),
        "cost_per_trial_usd": {"mean": mean_cost, "p90": percentile(costs, 90)},
        "full_study_projection": {
            "conditions": projection.conditions,
            "sequences": projection.sequences,
            "attempts": projection.attempts,
            "total_trials": projection.total_trials,
            "projected_cost_usd": mean_cost * projection.total_trials,
        },
        "latency_per_trial_s": {"mean": _mean(latencies), "p90": percentile(latencies, 90)},
        "llm_calls_per_trial_mean": _mean(calls),
        "capability_floor": {
            "floor": floor,
            "static_baseline_trials": len(static),
            "static_exploit_success_rate": _rate(static, "exploit_success"),
            "static_exploitation_success_rate": floor_success,
            "verdict": (
                "capability adequate"
                if adequate
                else (
                    f"{model_id or 'the configured attacker model'} may be underpowered "
                    "as the attacker — consider a more capable model"
                )
            ),
        },
    }


def format_summary(summary: dict) -> str:
    """Render the summary dict as a human-readable report."""

    tk = summary["tokens_per_trial"]
    proj = summary["full_study_projection"]
    cap = summary["capability_floor"]
    cost = summary["cost_per_trial_usd"]
    lat = summary["latency_per_trial_s"]
    lines = [
        f"cost-probe summary ({summary['n_trials']} trials; "
        f"conditions: {', '.join(summary['conditions_run'])})",
        "",
        "tokens/trial (mean | p90):",
        f"  uncached input : {tk['uncached_input']['mean']:10.1f} | {tk['uncached_input']['p90']:10.1f}",
        f"  cache read     : {tk['cache_read']['mean']:10.1f} | {tk['cache_read']['p90']:10.1f}",
        f"  cache write    : {tk['cache_write']['mean']:10.1f} | {tk['cache_write']['p90']:10.1f}",
        f"  output         : {tk['output']['mean']:10.1f} | {tk['output']['p90']:10.1f}",
        f"  cache-hit rate : {summary['cache_hit_rate'] * 100:.1f}%",
        "",
        f"cost/trial USD : mean ${cost['mean']:.4f} | p90 ${cost['p90']:.4f}",
        f"full-study proj: {proj['total_trials']} trials "
        f"({proj['conditions']}x{proj['sequences']}x{proj['attempts']}) "
        f"-> ${proj['projected_cost_usd']:.2f}",
        "",
        f"latency/trial s: mean {lat['mean']:.2f} | p90 {lat['p90']:.2f}",
        f"LLM calls/trial: mean {summary['llm_calls_per_trial_mean']:.2f}",
        "",
        "capability floor (static baseline):",
        f"  exploit_success       rate: {cap['static_exploit_success_rate'] * 100:.1f}%",
        f"  exploitation_success  rate: {cap['static_exploitation_success_rate'] * 100:.1f}% "
        f"(floor {cap['floor'] * 100:.0f}%)",
        f"  verdict: {cap['verdict']}",
    ]
    return "\n".join(lines)
