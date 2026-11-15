"""Clock-validation report: does the modeled attacker clock reproduce the real race?

Reads a trial dataset that recorded real action timing (``measurement.timing``) -- normally the
paired output of ``cli.py validate-clock``, where every (condition, sequence) ran once under the
modeled clock and once under the wall clock with the same seed -- and answers one question with
pre-specified criteria (``docs/clock_validation_runbook.md``):

    Is the probability that the attacker's FIRST GENUINE exploit lands after an MTD tick
    (its intelligence went stale) materially different under measured action timing than
    under the instrument's modeled clock?

That probability is what the reconfiguration-vs-attack race turns into the efficacy estimates:
with a uniformly random rotation phase, an exploit fired ``g`` seconds after the snapshot it
relies on is stale with probability ``min(g, cadence) / cadence``. The modeled clock fixes the
gap at 10 s for an exploit that directly follows its recon (10/35 = .286 at the study cadence);
the real gap is the model's decision latency (if the scan's observation is dated to its return)
up to the scan's own duration plus that latency (if dated to its issue). Every timed delivery
carries its gap under all three conventions -- ``modeled``, ``wall`` and ``wall_scan_start`` --
so the comparison is PAIRED on identical action sequences, and the primary statistic uses the
analytic stale probability (no phase noise). The realized staleness flags and the two arms'
outcomes are reported alongside as secondary, descriptive evidence.

**``wall_observed`` (report version 3, 2026-10-28).** Timing format v2 also records each
delivery under ``wall_observed``: the scan's observations dated to the instant nmap probed the
targets host, which is the convention the wall clock drives from 2026-10-27 on and therefore the
attacker's own staleness in the re-run (``docs/rerun_runbook.md`` §2, §6). It is reported beside
the other three conventions, with its own paired deltas against the modeled clock; v1 rows (the
2026-10-26 pilot) carry no value for it and are left out of its means. The pre-specified verdict
is unchanged: it still reads ``delta_wall_minus_modeled``.

**Inference.** The independent unit is the PAIR -- the modeled and wall runs of one
(condition, sequence), which share a seed and an MTD schedule -- not the delivery: the eight
attempts of a sequence share an attacker context, so deliveries within a pair are correlated.
Every interval is therefore a cluster-robust (Taylor-linearization) interval for a ratio mean,
clustered on the pair and stratified by condition, with the small-sample factor G_h/(G_h - 1)
per stratum and a Student t reference on sum(G_h - 1) degrees of freedom. A condition with a
single pair makes the variance non-estimable, and the verdict additionally requires a minimum
number of pairs in EVERY race-sensitive condition present, so a handful of sequences can never
yield a confident verdict.

**Stdlib only, on purpose.** This module imports nothing outside the standard library and
``measurement``, so ``cli.py validate-clock`` can produce its verdict on the control host right
after collection, without the pandas/statsmodels analysis stack (and on Python 3.10). It is not
part of ``analysis.pipeline`` and nothing here enters an RQ model or an FDR family.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, fields

from measurement.models import CLOCK_MODELED, CLOCK_WALL, NOT_RECORDED, TrialRow
from measurement.timing import KIND_EXPLOIT, KIND_RECON, ActionTiming, decode_action_timing

__all__ = [
    "MODELED_CONSTANTS",
    "RACE_SENSITIVE_CONDITIONS",
    "REPORT_VERSION",
    "Criteria",
    "build_report",
    "format_summary",
    "implied_stale_probability",
]

#: 3 (2026-10-28): adds the ``wall_observed`` convention and ``attacker_convention``.
REPORT_VERSION = 3

#: The modeled clock's constants (``attacker/planner.py``), restated for the report. Pinned
#: equal to the planner's by test, so the report can never describe a clock the code no longer
#: runs.
MODELED_CONSTANTS: dict[str, float] = {
    "recon_command_offset_s": 5.0,
    "recon_action_s": 30.0,
    "exploit_action_s": 10.0,
    "first_exploit_gap_s": 10.0,
}

#: Conditions whose technique changes a GENUINE service's observable identity on every tick,
#: so a genuine exploit can go stale: the race the clock decides. ``static_baseline`` never
#: ticks and ``deceptive_presentation`` rotates decoys only, so their genuine deliveries can
#: never be stale under any clock; they are reported for timing but are not in the primary
#: population. Pinned against ``mtd.techniques`` by test.
RACE_SENSITIVE_CONDITIONS: tuple[str, ...] = (
    "network_shuffling",
    "port_rotation",
    "endpoint_mutation",
    "combined_mtd_deception",
)

_CONVENTIONS = ("modeled", "wall", "wall_scan_start", "wall_observed")


#: Two-sided confidence level of every interval in the report (fixed, not a criterion knob).
CONFIDENCE = 0.95


@dataclass(frozen=True)
class Criteria:
    """Pre-specified decision criteria (``modes.clock_validation.criteria`` in the config).

    ``cli.py validate-clock`` locks them into the run's checkpoint manifest when the run first
    starts and refuses to resume under different ones, so they cannot drift after data exist.
    """

    equivalence_margin: float = 0.10
    min_first_genuine_deliveries: int = 30
    min_pairs_per_condition: int = 4

    def __post_init__(self) -> None:
        if not 0.0 < float(self.equivalence_margin) < 1.0:
            raise ValueError("criteria.equivalence_margin must be in (0, 1)")
        if int(self.min_first_genuine_deliveries) < 1:
            raise ValueError("criteria.min_first_genuine_deliveries must be >= 1")
        if int(self.min_pairs_per_condition) < 2:
            raise ValueError(
                "criteria.min_pairs_per_condition must be >= 2 (one pair cannot estimate "
                "between-sequence variance)"
            )

    def as_dict(self) -> dict:
        return {
            "equivalence_margin": float(self.equivalence_margin),
            "min_first_genuine_deliveries": int(self.min_first_genuine_deliveries),
            "min_pairs_per_condition": int(self.min_pairs_per_condition),
        }

    @classmethod
    def from_dict(cls, raw: dict) -> Criteria:
        """Build from a recorded/configured mapping; unknown or missing keys raise."""

        if not isinstance(raw, dict):
            raise ValueError(f"criteria must be a mapping, got {type(raw).__name__}")
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - known)
        missing = sorted(known - set(raw))
        if unknown or missing:
            raise ValueError(f"criteria keys: unknown {unknown}, missing {missing}")
        return cls(
            equivalence_margin=float(raw["equivalence_margin"]),
            min_first_genuine_deliveries=int(raw["min_first_genuine_deliveries"]),
            min_pairs_per_condition=int(raw["min_pairs_per_condition"]),
        )


def implied_stale_probability(gap_s: float, cadence_s: float) -> float:
    """P(at least one tick in a ``gap_s``-second window) under a uniform rotation phase.

    Ticks fall every ``cadence_s`` seconds at a uniformly random phase, so a window of length
    ``g`` contains a tick with probability ``min(g, cadence) / cadence`` (a negative gap --
    which cannot occur -- is treated as 0).
    """

    if cadence_s <= 0:
        return 0.0
    return min(max(float(gap_s), 0.0), float(cadence_s)) / float(cadence_s)


# --------------------------------------------------------------------------- #
# Small stdlib statistics
# --------------------------------------------------------------------------- #


def _r(x: float | None, nd: int = 4) -> float | None:
    return None if x is None else round(float(x), nd)


def _quantile(sorted_vals: Sequence[float], q: float) -> float | None:
    """Linear-interpolation quantile (numpy's default) of an ascending sequence."""

    if not sorted_vals:
        return None
    pos = (len(sorted_vals) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return float(sorted_vals[lo])
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo))


def _describe(values: Iterable[float]) -> dict:
    vals = sorted(float(v) for v in values)
    n = len(vals)
    if n == 0:
        return {"n": 0}
    mean = sum(vals) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else None
    return {
        "n": n,
        "mean": _r(mean),
        "sd": _r(sd),
        "median": _r(_quantile(vals, 0.5)),
        "p10": _r(_quantile(vals, 0.10)),
        "p90": _r(_quantile(vals, 0.90)),
        "min": _r(vals[0]),
        "max": _r(vals[-1]),
    }


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the regularized incomplete beta (modified Lentz)."""

    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 500):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-15:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""

    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _t_cdf(t: float, df: float) -> float:
    """Student t cumulative distribution function."""

    tail = 0.5 * _betainc(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - tail if t > 0 else tail


def t_quantile(p: float, df: float) -> float:
    """Upper quantile of Student t (``p`` in (0.5, 1)), by bisection on the exact CDF."""

    lo, hi = 0.0, 1.0
    while _t_cdf(hi, df) < p:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _cr_result(estimate: float | None, z: dict[str, dict[str, float]], n_units: int) -> dict:
    """Interval from per-cluster linearized totals ``z[stratum][cluster]`` (see module doc)."""

    clusters_by_condition = {h: len(c) for h, c in sorted(z.items())}
    out = {
        "estimate": _r(estimate),
        "ci95": [None, None],
        "se": None,
        "df": None,
        "n_units": n_units,
        "n_pairs": sum(clusters_by_condition.values()),
        "pairs_by_condition": clusters_by_condition,
    }
    if estimate is None or not z or any(g < 2 for g in clusters_by_condition.values()):
        return out  # a single pair in any stratum: between-pair variance is not estimable
    var = 0.0
    df = 0
    for clusters in z.values():
        g = len(clusters)
        zs = list(clusters.values())
        zbar = sum(zs) / g
        var += g / (g - 1) * sum((v - zbar) ** 2 for v in zs)
        df += g - 1
    se = math.sqrt(var)
    t = t_quantile(0.5 + CONFIDENCE / 2.0, df)
    out.update(ci95=[_r(estimate - t * se), _r(estimate + t * se)], se=_r(se), df=df)
    return out


def _cr_mean(units: Sequence[dict], value: Callable[[dict], float | None]) -> dict:
    """Cluster-robust interval for the mean of ``value`` over the units that have one."""

    pts = [(u["condition"], u["pair"], value(u)) for u in units]
    pts = [(h, g, float(v)) for h, g, v in pts if v is not None]
    if not pts:
        return _cr_result(None, {}, 0)
    n = len(pts)
    est = sum(v for _, _, v in pts) / n
    z: dict[str, dict[str, float]] = {}
    for h, g, v in pts:
        z.setdefault(h, {}).setdefault(g, 0.0)
        z[h][g] += (v - est) / n
    return _cr_result(est, z, n)


def _cr_arm_diff(units: Sequence[dict], field: str, arm_a: str, arm_b: str) -> dict:
    """Cluster-robust interval for mean(arm_a) - mean(arm_b), pairs as clusters."""

    a = [(u["condition"], u["pair"], float(u[field])) for u in units if u["arm"] == arm_a]
    b = [(u["condition"], u["pair"], float(u[field])) for u in units if u["arm"] == arm_b]
    if not a or not b:
        return _cr_result(None, {}, len(a) + len(b))
    pa = sum(v for _, _, v in a) / len(a)
    pb = sum(v for _, _, v in b) / len(b)
    z: dict[str, dict[str, float]] = {}
    for h, g, v in a:
        z.setdefault(h, {}).setdefault(g, 0.0)
        z[h][g] += (v - pa) / len(a)
    for h, g, v in b:
        z.setdefault(h, {}).setdefault(g, 0.0)
        z[h][g] -= (v - pb) / len(b)
    return _cr_result(pa - pb, z, len(a) + len(b))


def _classify(ci: Sequence[float | None], margin: float, gate_ok: bool) -> str:
    lo, hi = ci
    if not gate_ok or lo is None or hi is None:
        return "insufficient_data"
    if -margin < lo and hi < margin:
        return "equivalent"
    if lo > margin or hi < -margin:
        return "not_equivalent"
    return "inconclusive"


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #


def _pair_id(row: TrialRow) -> str:
    """The paired-design cluster: the sequence id without its clock-arm suffix."""

    suffix = f"-{row.clock_mode}"
    sid = row.sequence_id
    return sid[: -len(suffix)] if row.clock_mode and sid.endswith(suffix) else sid


def _actions(row: TrialRow) -> tuple[ActionTiming, ...] | None:
    return decode_action_timing(row.action_timing)


def _timing_version(cell: object) -> int | None:
    """The ``action_timing`` document's format version (``"v"``), or None if unreadable."""

    try:
        doc = json.loads(str(cell))
    except (TypeError, ValueError):
        return None
    v = doc.get("v") if isinstance(doc, dict) else None
    return v if isinstance(v, int) else None


def _first(actions: Sequence[ActionTiming], pred: Callable[[ActionTiming], bool]):
    for a in actions:
        if pred(a):
            return a
    return None


def attacker_convention(clock_mode: str, a: ActionTiming) -> str:
    """The convention whose staleness the executor acted on for this delivery.

    The modeled clock acts on ``modeled``. The wall clock acts on ``wall_observed`` when the
    delivery records it (timing format v2, from 2026-10-27) and on ``wall`` otherwise (v1, the
    2026-10-26 pilot, which dated a scan to its return). From format v3 (2026-10-31) the wall
    clock also scores the delivery when it completes (``stale_at_completion``); the convention
    here is still the one that decided, at firing, whether anything was delivered at all, which
    is what the intel-age comparison and the delivery-duration filter below need.
    """

    if clock_mode == CLOCK_MODELED:
        return "modeled"
    return "wall_observed" if a.stale_wall_observed is not None else "wall"


def _actual_stale(clock_mode: str, a: ActionTiming) -> bool | None:
    conv = attacker_convention(clock_mode, a)
    return {
        "modeled": a.stale_modeled,
        "wall": a.stale_wall,
        "wall_observed": a.stale_wall_observed,
    }[conv]


def _delivery_unit(row: TrialRow, a: ActionTiming, first_in_trial: bool) -> dict:
    cadence = float(row.mtd_cadence_s)
    gaps = {
        "modeled": a.gap_modeled_s,
        "wall": a.gap_wall_s,
        "wall_scan_start": a.gap_wall_scan_start_s,
        "wall_observed": a.gap_wall_observed_s,
    }
    stale = {
        "modeled": a.stale_modeled,
        "wall": a.stale_wall,
        "wall_scan_start": a.stale_wall_scan_start,
        "wall_observed": a.stale_wall_observed,
    }
    actual = _actual_stale(row.clock_mode, a)
    return {
        "condition": row.condition,
        "arm": row.clock_mode,
        "pair": f"{row.condition}|{_pair_id(row)}",
        "trial_id": row.trial_id,
        "cadence": cadence,
        "first_in_trial": first_in_trial,
        "gap": gaps,
        "q": {
            c: (None if g is None else implied_stale_probability(g, cadence))
            for c, g in gaps.items()
        },
        "stale": stale,
        "stale_actual": actual,
        "attacker_convention": attacker_convention(row.clock_mode, a),
        "delivery_s": a.exec_s,
        "api_retries": a.api_retries,
    }


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #


def _diff(
    key_a: Callable[[dict], float | None], key_b: Callable[[dict], float | None]
) -> Callable[[dict], float | None]:
    """Per-unit paired difference ``key_a - key_b`` (None when either is missing)."""

    def value(u: dict) -> float | None:
        a, b = key_a(u), key_b(u)
        return None if a is None or b is None else a - b

    return value


def _q(conv: str) -> Callable[[dict], float | None]:
    return lambda u: u["q"][conv]


def _stale(conv: str) -> Callable[[dict], float | None]:
    def key(u: dict) -> float | None:
        v = u["stale"][conv]
        return None if v is None else float(v)

    return key


def _agree(u: dict) -> float | None:
    m, w = u["stale"]["modeled"], u["stale"]["wall"]
    return None if m is None or w is None else float(m == w)


def _staleness_block(units: list[dict]) -> dict:
    """Implied (analytic) and realized staleness of a delivery population, paired."""

    implied = {c: _cr_mean(units, _q(c)) for c in _CONVENTIONS}
    implied["delta_wall_minus_modeled"] = _cr_mean(units, _diff(_q("wall"), _q("modeled")))
    implied["delta_wall_scan_start_minus_modeled"] = _cr_mean(
        units, _diff(_q("wall_scan_start"), _q("modeled"))
    )
    implied["delta_wall_observed_minus_modeled"] = _cr_mean(
        units, _diff(_q("wall_observed"), _q("modeled"))
    )
    realized = {c: _cr_mean(units, _stale(c)) for c in _CONVENTIONS}
    realized["delta_wall_minus_modeled"] = _cr_mean(units, _diff(_stale("wall"), _stale("modeled")))
    realized["delta_wall_observed_minus_modeled"] = _cr_mean(
        units, _diff(_stale("wall_observed"), _stale("modeled"))
    )
    both = [u for u in units if _agree(u) is not None]
    realized["agreement_modeled_vs_wall"] = {
        **_cr_mean(units, _agree),
        "agree": sum(1 for u in both if _agree(u)),
        "stale_modeled_only": sum(
            1 for u in both if u["stale"]["modeled"] and not u["stale"]["wall"]
        ),
        "stale_wall_only": sum(1 for u in both if u["stale"]["wall"] and not u["stale"]["modeled"]),
    }
    gaps = {
        c: _describe(u["gap"][c] for u in units if u["gap"][c] is not None) for c in _CONVENTIONS
    }
    conv_counts: dict[str, int] = {}
    for u in units:
        conv_counts[u["attacker_convention"]] = conv_counts.get(u["attacker_convention"], 0) + 1
    return {
        "n": len(units),
        "attacker_convention": dict(sorted(conv_counts.items())),
        "gap_s": gaps,
        "implied": implied,
        "realized": realized,
    }


def _timing_block(rows: Sequence[TrialRow]) -> dict:
    decide_by_kind: dict[str, list[float]] = {}
    first_decide: list[float] = []
    scans: list[float] = []
    deliveries: list[float] = []
    recon_actions: list[float] = []
    exploit_actions: list[float] = []
    relabels = n_scans = retried_calls = n_calls = 0
    over_limit = 0
    attempt_wall = []
    for row in rows:
        acts = _actions(row)
        if acts is None:
            continue
        if row.attempt_wall_s is not None:
            attempt_wall.append(row.attempt_wall_s)
            # Only the wall clock consults real time for the duration limit; a modeled-arm
            # attempt's real length says nothing about its (modeled) window.
            if row.clock_mode == CLOCK_WALL and row.attempt_wall_s > row.trial_duration_limit_s:
                over_limit += 1
        if acts:
            first_decide.append(acts[0].decide_s)
        for a in acts:
            n_calls += 1
            decide_by_kind.setdefault(a.kind, []).append(a.decide_s)
            if a.api_retries:
                retried_calls += 1
            if not a.executed:
                continue
            if a.kind == KIND_RECON:
                n_scans += 1
                scans.append(a.exec_s)
                recon_actions.append(a.decide_s + a.exec_s)
                relabels += int(bool(a.relabeled))
            elif a.kind == KIND_EXPLOIT:
                # A delivery the ACTUAL clock judged stale was short-circuited before any
                # physical delivery, so its duration is not a delivery time (and the action's
                # length would understate a real exploit action).
                actual = _actual_stale(row.clock_mode, a)
                if not actual:
                    deliveries.append(a.exec_s)
                    exploit_actions.append(a.decide_s + a.exec_s)
    all_decide = [v for vals in decide_by_kind.values() for v in vals]
    return {
        "planner_call_latency_s": _describe(all_decide),
        "planner_call_latency_s_by_kind": {
            k: _describe(v) for k, v in sorted(decide_by_kind.items())
        },
        "first_planner_call_latency_s": _describe(first_decide),
        "scan_duration_s": _describe(scans),
        "delivery_duration_s": _describe(deliveries),
        "recon_action_s": _describe(recon_actions),
        "exploit_action_s": _describe(exploit_actions),
        "attempt_wall_s": _describe(attempt_wall),
        "n_planner_calls": n_calls,
        "n_planner_calls_with_api_retries": retried_calls,
        "n_scans": n_scans,
        "n_scans_relabeled_by_a_tick": relabels,
        "n_attempts_over_duration_limit": over_limit,
    }


def _outcome_units(rows: Sequence[TrialRow]) -> list[dict]:
    return [
        {
            "condition": r.condition,
            "arm": r.clock_mode,
            "pair": f"{r.condition}|{_pair_id(r)}",
            "exploitation_success": float(r.exploitation_success),
            "exploit_success": float(r.exploit_success),
            "ttc_uncensored_s": None if r.ttc_censored else r.time_to_compromise_s,
            "ttc_censored": float(r.ttc_censored),
        }
        for r in rows
    ]


def _outcomes_block(rows: Sequence[TrialRow]) -> dict:
    units = _outcome_units(rows)
    out: dict = {}
    for cond in sorted({u["condition"] for u in units}):
        cu = [u for u in units if u["condition"] == cond]
        per_arm = {}
        for arm in (CLOCK_MODELED, CLOCK_WALL):
            au = [u for u in cu if u["arm"] == arm]
            if not au:
                continue
            k = int(sum(u["exploitation_success"] for u in au))
            per_arm[arm] = {
                "n_trials": len(au),
                "exploitation_success": _r(k / len(au)),
                "exploitation_success_interval": _cr_mean(au, lambda u: u["exploitation_success"]),
                "exploit_success": _r(_mean([u["exploit_success"] for u in au])),
                "ttc_censored_rate": _r(_mean([u["ttc_censored"] for u in au])),
                "ttc_uncensored_s": _describe(
                    u["ttc_uncensored_s"] for u in au if u["ttc_uncensored_s"] is not None
                ),
            }
        entry: dict = {"by_arm": per_arm}
        if CLOCK_MODELED in per_arm and CLOCK_WALL in per_arm:
            entry["delta_exploitation_success_wall_minus_modeled"] = _cr_arm_diff(
                cu, "exploitation_success", CLOCK_WALL, CLOCK_MODELED
            )
        out[cond] = entry
    return out


def build_report(
    rows: Sequence[TrialRow],
    criteria: Criteria | None = None,
    provenance: dict | None = None,
) -> dict:
    """Build the clock-validation report from trial rows (see the module docstring)."""

    criteria = criteria or Criteria()
    included = [r for r in rows if not r.excluded]
    timed = [r for r in included if r.action_timing not in (NOT_RECORDED, "", None)]
    arms_present = sorted({r.clock_mode for r in included})

    inventory: dict = {
        "n_rows": len(rows),
        "n_included": len(included),
        "n_excluded": len(rows) - len(included),
        "n_included_with_timing": len(timed),
        "coverage": ("none" if not timed else "full" if len(timed) == len(included) else "partial"),
        "arms_present": arms_present,
        "by_condition_arm": {},
    }
    for r in included:
        key = f"{r.condition}|{r.clock_mode}"
        inventory["by_condition_arm"][key] = inventory["by_condition_arm"].get(key, 0) + 1
    inventory["by_condition_arm"] = dict(sorted(inventory["by_condition_arm"].items()))
    versions: dict[str, int] = {}
    for r in timed:
        v = _timing_version(r.action_timing)
        versions[str(v)] = versions.get(str(v), 0) + 1
    inventory["timing_format_versions"] = dict(sorted(versions.items()))

    # Delivery populations (executed exploits with timing).
    first_genuine: list[dict] = []
    all_sourced: list[dict] = []
    for row in timed:
        acts = _actions(row) or ()
        exploits = [a for a in acts if a.kind == KIND_EXPLOIT and a.executed]
        fg = _first(exploits, lambda a: bool(a.service_genuine) and bool(a.snapshot_sourced))
        for a in exploits:
            if a.snapshot_sourced:
                all_sourced.append(_delivery_unit(row, a, a is exploits[0]))
        if fg is not None:
            first_genuine.append(_delivery_unit(row, fg, fg is exploits[0]))

    primary_units = [u for u in first_genuine if u["condition"] in RACE_SENSITIVE_CONDITIONS]
    # Any planner call in the trial that needed app-level API retries: its latency includes
    # backoff sleeps, which is infrastructure, not attacker speed.
    retried_trials = {
        row.trial_id for row in timed if any(a.api_retries for a in (_actions(row) or ()))
    }
    primary = _staleness_block(primary_units)
    by_condition = {
        cond: _staleness_block([u for u in primary_units if u["condition"] == cond])
        for cond in RACE_SENSITIVE_CONDITIONS
        if any(u["condition"] == cond for u in primary_units)
    }
    sens_units = [u for u in primary_units if u["trial_id"] not in retried_trials]

    margin = criteria.equivalence_margin
    min_n = criteria.min_first_genuine_deliveries
    min_pairs = criteria.min_pairs_per_condition
    d_wall = primary["implied"]["delta_wall_minus_modeled"]
    d_start = primary["implied"]["delta_wall_scan_start_minus_modeled"]
    m_hat = primary["implied"]["modeled"]["estimate"]
    w_hat = primary["implied"]["wall"]["estimate"]
    s_hat = primary["implied"]["wall_scan_start"]["estimate"]
    pairs_by_condition = d_wall["pairs_by_condition"]
    gate = {
        "n_first_genuine_deliveries": len(primary_units),
        "min_first_genuine_deliveries": min_n,
        "pairs_by_condition": pairs_by_condition,
        "min_pairs_per_condition": min_pairs,
        "passed": bool(
            primary_units
            and len(primary_units) >= min_n
            and all(g >= min_pairs for g in pairs_by_condition.values())
        ),
    }
    verdict = {
        "question": (
            "Is the first genuine exploit's stale probability under measured timing within "
            f"+/-{margin} of the modeled clock's (race-sensitive conditions, paired)?"
        ),
        "primary_statistic": "implied.delta_wall_minus_modeled",
        "interval": (
            f"{CONFIDENCE:.0%} cluster-robust t interval (pairs as clusters, stratified by "
            "condition)"
        ),
        "n_first_genuine_deliveries": len(primary_units),
        "n_pairs": d_wall["n_pairs"],
        "df": d_wall["df"],
        "gate": gate,
        "primary": _classify(d_wall["ci95"], margin, gate["passed"]),
        "sensitivity_scan_start_labels": _classify(d_start["ci95"], margin, gate["passed"]),
        "by_condition": {
            cond: _classify(
                blk["implied"]["delta_wall_minus_modeled"]["ci95"],
                margin,
                blk["implied"]["delta_wall_minus_modeled"]["n_pairs"] >= min_pairs,
            )
            for cond, blk in by_condition.items()
        },
        "modeled_within_measured_bracket": (
            None
            if None in (m_hat, w_hat, s_hat)
            else bool(min(w_hat, s_hat) <= m_hat <= max(w_hat, s_hat))
        ),
    }

    return {
        "report": "clock_validation",
        "version": REPORT_VERSION,
        "provenance": dict(provenance or {}),
        "criteria": criteria.as_dict(),
        "modeled_constants": dict(MODELED_CONSTANTS),
        "race_sensitive_conditions": list(RACE_SENSITIVE_CONDITIONS),
        "inventory": inventory,
        "verdict": verdict,
        "first_genuine_delivery": {"pooled": primary, "by_condition": by_condition},
        "sensitivity_excluding_api_retry_trials": _staleness_block(sens_units),
        "all_snapshot_sourced_deliveries": _staleness_block(
            [u for u in all_sourced if u["condition"] in RACE_SENSITIVE_CONDITIONS]
        ),
        "measured_timing": _timing_block(timed),
        "arm_outcomes": _outcomes_block(included),
    }


# --------------------------------------------------------------------------- #
# Human-readable summary
# --------------------------------------------------------------------------- #

_VERDICT_TEXT = {
    "equivalent": (
        "EQUIVALENT: under measured timing the first genuine exploit's stale probability is "
        "within the margin of the modeled clock's. The modeled clock is an adequate "
        "approximation of the race at this cadence."
    ),
    "not_equivalent": (
        "NOT EQUIVALENT: the modeled clock mis-states the race by more than the margin. The "
        "efficacy estimates are conditional on the modeled clock; report the measured "
        "stale probability and the wall arm's outcomes alongside them."
    ),
    "inconclusive": (
        "INCONCLUSIVE: the interval straddles the margin. More sequences would be needed to "
        "establish or rule out equivalence."
    ),
    "insufficient_data": (
        "INSUFFICIENT DATA: too few first genuine deliveries, or too few independent sequences "
        "in a race-sensitive condition, to apply the criterion."
    ),
}


def _fmt(x: float | None, nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def _fmt_ci(block: dict, nd: int = 3) -> str:
    lo, hi = block.get("ci95", [None, None])
    return f"{_fmt(block.get('estimate'), nd)} [{_fmt(lo, nd)}, {_fmt(hi, nd)}]"


def format_summary(report: dict) -> str:
    """A short Markdown summary of a :func:`build_report` result."""

    v = report["verdict"]
    crit = report["criteria"]
    inv = report["inventory"]
    pooled = report["first_genuine_delivery"]["pooled"]
    imp = pooled["implied"]
    real = pooled["realized"]
    t = report["measured_timing"]
    mc = report["modeled_constants"]
    prov = report.get("provenance", {})
    pairs = ", ".join(f"{c} {g}" for c, g in v["gate"]["pairs_by_condition"].items()) or "none"
    run_line = (
        f"- Run: backend {prov.get('backend', 'n/a')}, "
        f"{'live testbed' if prov.get('live') else 'offline' if prov.get('live') is False else 'mode n/a'}"
        f", model {prov.get('model_id', 'n/a')}; criteria {prov.get('criteria_source', 'n/a')}."
    )
    lines = [
        "# Clock validation — modeled vs. measured attacker timing",
        "",
        f"**Verdict (primary, pre-specified): {v['primary'].upper()}** — "
        + _VERDICT_TEXT[v["primary"]],
        "",
        f"- Criterion: paired difference in the first genuine exploit's implied stale "
        f"probability (wall − modeled); its {v['interval']} must lie inside "
        f"±{crit['equivalence_margin']}. Requires at least "
        f"{crit['min_first_genuine_deliveries']} deliveries and "
        f"{crit['min_pairs_per_condition']} pairs (independent sequences) in every "
        f"race-sensitive condition.",
        f"- Data behind it: {v['n_first_genuine_deliveries']} first genuine deliveries in "
        f"{v['n_pairs']} pairs ({pairs}); df = {v['df']}; gate "
        f"{'passed' if v['gate']['passed'] else 'NOT passed'}.",
        f"- Sensitivity (scan observations dated to the scan's issue): "
        f"**{v['sensitivity_scan_start_labels']}**.",
        f"- Modeled stale probability inside the measured bracket: "
        f"{v['modeled_within_measured_bracket']}.",
        f"- Trials: {inv['n_included']} included ({inv['n_excluded']} excluded); timing recorded "
        f"on {inv['n_included_with_timing']} ({inv['coverage']}); arms: "
        f"{', '.join(inv['arms_present']) or 'none'}.",
        run_line,
        "",
        "## First genuine exploit, race-sensitive conditions (paired)",
        "",
        "| convention | gap median (s) | implied stale prob. [95% CI] | realized stale rate [95% CI] |",
        "|---|---|---|---|",
    ]
    for conv in _CONVENTIONS:
        lines.append(
            f"| {conv} | {_fmt(pooled['gap_s'][conv].get('median'), 2)} | "
            f"{_fmt_ci(imp[conv])} | {_fmt_ci(real[conv])} |"
        )
    conv = pooled.get("attacker_convention") or {}
    lines += [
        "",
        f"- Δ implied (wall − modeled): {_fmt_ci(imp['delta_wall_minus_modeled'])}",
        f"- Δ implied (wall_scan_start − modeled): "
        f"{_fmt_ci(imp['delta_wall_scan_start_minus_modeled'])}",
        f"- Δ implied (wall_observed − modeled): "
        f"{_fmt_ci(imp['delta_wall_observed_minus_modeled'])}",
        "- The convention the attacker's own clock acted on, by delivery: "
        + (", ".join(f"{k} {v}" for k, v in conv.items()) or "n/a")
        + ".",
        f"- Realized agreement, modeled vs wall: "
        f"{_fmt_ci(real['agreement_modeled_vs_wall'])} "
        f"(n = {real['agreement_modeled_vs_wall']['n_units']}; stale under modeled only: "
        f"{real['agreement_modeled_vs_wall']['stale_modeled_only']}, under wall only: "
        f"{real['agreement_modeled_vs_wall']['stale_wall_only']})",
        "",
        "## Measured timing vs. the modeled clock",
        "",
        "| quantity | modeled | measured median | measured p10–p90 | n |",
        "|---|---|---|---|---|",
    ]

    def row(label: str, modeled: float | None, block: dict) -> str:
        return (
            f"| {label} | {_fmt(modeled, 1) if modeled is not None else '—'} | "
            f"{_fmt(block.get('median'), 2)} | {_fmt(block.get('p10'), 2)}–"
            f"{_fmt(block.get('p90'), 2)} | {block.get('n', 0)} |"
        )

    lines += [
        row(
            "first planner call (recon command offset)",
            mc["recon_command_offset_s"],
            t["first_planner_call_latency_s"],
        ),
        row("recon action (decide + scan)", mc["recon_action_s"], t["recon_action_s"]),
        row("exploit action (decide + deliver)", mc["exploit_action_s"], t["exploit_action_s"]),
        row("planner call (any)", None, t["planner_call_latency_s"]),
        row("nmap scan", None, t["scan_duration_s"]),
        row("delivery (physically delivered)", None, t["delivery_duration_s"]),
        row("attempt", None, t["attempt_wall_s"]),
        "",
        f"- Planner calls with API retries: {t['n_planner_calls_with_api_retries']} of "
        f"{t['n_planner_calls']}; scans during which a tick relabeled a discovered service: "
        f"{t['n_scans_relabeled_by_a_tick']} of {t['n_scans']}; attempts over the duration "
        f"limit: {t['n_attempts_over_duration_limit']}.",
        "",
        "## Exploitation success by arm (secondary, descriptive)",
        "",
        "| condition | modeled | wall | Δ wall − modeled [95% CI] |",
        "|---|---|---|---|",
    ]
    for cond, entry in report["arm_outcomes"].items():
        by_arm = entry["by_arm"]
        m = by_arm.get(CLOCK_MODELED, {})
        w = by_arm.get(CLOCK_WALL, {})
        delta = entry.get("delta_exploitation_success_wall_minus_modeled")
        lines.append(
            f"| {cond} | {_fmt(m.get('exploitation_success'))} (n={m.get('n_trials', 0)}) | "
            f"{_fmt(w.get('exploitation_success'))} (n={w.get('n_trials', 0)}) | "
            f"{_fmt_ci(delta) if delta else 'n/a'} |"
        )
    lines += [
        "",
        "Implied stale probability = min(gap, cadence)/cadence (uniform rotation phase). "
        "`wall` dates a scan's observations to its return; `wall_scan_start` to its issue; "
        "the real value lies between them. `wall_observed` dates them to the instant the scan "
        "probed the targets host (timing format v2): the attacker's own convention from "
        "2026-10-27 on (the re-run), absent from v1 rows. Intervals are cluster-robust t "
        "intervals with the pair (one sequence under both clocks) as the cluster; in a run "
        "with only the wall clock the pair is the sequence itself. See "
        "docs/clock_validation_runbook.md.",
        "",
    ]
    return "\n".join(lines)
