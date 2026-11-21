"""Values the dissertation reports from the main study's data that the analysis report does not hold.

Reads ``trial_dataset.csv`` and ``run_meta.json`` from a results folder (by default
``docs/results/full``) and prints:

- treatment delivery: deliveries to a genuine service that stayed in place from firing to
  completion, deliveries the tooling reported as successful whose target moved before they
  completed, and the scans the client ran before an attempt's first scan;
- the replies that carried no action, which the client records as a stop: how many, where, how
  long those decisions took, the output tokens per decision, and the compromises they likely
  cost;
- the exploratory observations reported with the third research question: compromise in later
  attempts by the target of the first exploit, with and without the attempts that ended in such
  a reply, and the static baseline and deceptive presentation misses by what the record of
  earlier attempts showed;
- the cross-validation folds of the fourth research question's regularized selection.

From the repository root, with the analysis environment:

    .venv-analysis/bin/python scripts/supplementary_checks.py [results folder]
"""

from __future__ import annotations

import collections
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import KFold, StratifiedKFold  # noqa: E402

from analysis.dataset import load_csv, prepare  # noqa: E402
from analysis.rq4 import METRIC_SET, _numeric  # noqa: E402
from measurement.action_trace import decode_trace  # noqa: E402
from measurement.timing import decode_action_timing  # noqa: E402
from testbed.profile import default_profile  # noqa: E402

CONDITIONS = [
    "static_baseline",
    "deceptive_presentation",
    "network_shuffling",
    "port_rotation",
    "endpoint_mutation",
    "combined_mtd_deception",
]
SINGLE_MOVING = ["network_shuffling", "port_rotation", "endpoint_mutation"]
NO_ACTION = "coerced_stop_no_tool_use"
GENUINE = {s.service_name for s in default_profile().services if s.genuine}
DECOYS = {s.service_name for s in default_profile().services if not s.genuine}


def compromised(t) -> bool:
    """A delivery that counted: genuine target in place at firing and at completion, access."""
    return bool(
        t is not None
        and t.service_genuine
        and t.stale_wall_observed is False
        and t.stale_at_completion is False
        and not t.completed_after_window
        and t.executor_access
    )


def load(folder: Path) -> tuple[pd.DataFrame, list[dict]]:
    df = prepare(load_csv(folder / "trial_dataset.csv"))
    trials = []
    for r in df.to_dict("records"):
        acts = decode_trace(r["action_sequence"])
        timing = {t.ordinal: t for t in (decode_action_timing(r["action_timing"]) or ())}
        sent = [a for a in acts if a.kind == "exploit" and a.executed and a.target is not None]
        first_win = next((a.ordinal for a in sent if compromised(timing.get(a.ordinal))), None)
        trials.append(
            {
                "condition": r["condition"],
                "sequence": r["sequence_id"],
                "attempt": int(r["attempt_index"]),
                "won": int(r["exploitation_success"]),
                "acts": acts,
                "timing": timing,
                "sent": sent,
                "first_win": first_win,
                "no_action": bool(acts) and acts[-1].kind == "stop" and acts[-1].note == NO_ACTION,
                "model_stop": bool(acts) and acts[-1].kind == "stop" and not acts[-1].note,
            }
        )
    return df, trials


def delivery(trials: list[dict]) -> None:
    print("Treatment delivery")
    in_place = won = 0
    overtaken = collections.Counter()
    overtaken_trials = {}
    client_scans = 0
    for tr in trials:
        client_scans += sum(
            1 for a in tr["acts"] if a.kind == "recon" and a.note == "coerced_recon_before_scan"
        )
        for a in tr["sent"]:
            t = tr["timing"].get(a.ordinal)
            if t is None or not t.service_genuine:
                continue
            if (
                t.stale_wall_observed is False
                and t.stale_at_completion is False
                and not t.completed_after_window
            ):
                in_place += 1
                won += bool(t.executor_access)
            if t.stale_wall_observed is False and t.stale_at_completion and t.executor_access:
                overtaken[a.target.service] += 1
                overtaken_trials[(tr["sequence"], tr["attempt"])] = tr
    print(
        f"  deliveries to a genuine service in place from firing to completion: {in_place}, compromised {won}"
    )
    print(
        f"  deliveries reported successful whose target moved before completion: {sum(overtaken.values())} "
        + str(dict(overtaken.most_common()))
    )
    missed = collections.Counter(
        tr["condition"] for tr in overtaken_trials.values() if not tr["won"]
    )
    print(
        f"  trials with such a delivery: {len(overtaken_trials)}, uncompromised {sum(missed.values())} "
        + str(dict(missed.most_common()))
    )
    print(
        f"  scans the client ran in place of an action requested before the attempt's first scan: {client_scans}"
    )


def no_action_replies(trials: list[dict], folder: Path) -> None:
    print("\nReplies that carried no action")
    n = collections.Counter(tr["condition"] for tr in trials)
    stops = collections.Counter(tr["condition"] for tr in trials if tr["no_action"])
    total = sum(stops.values())
    print(
        f"  attempts: {total} of {len(trials)} ({total / len(trials):.1%}); "
        + ", ".join(f"{c} {stops[c]}" for c in CONDITIONS)
    )
    print("  share of attempts, by attempt index 0-7:")
    for c in CONDITIONS:
        cells = []
        for k in range(8):
            at = [tr for tr in trials if tr["condition"] == c and tr["attempt"] == k]
            cells.append(f"{sum(tr['no_action'] for tr in at) / len(at):4.0%}")
        print(f"    {c:24s} " + " ".join(cells))

    decide = collections.defaultdict(list)
    for tr in trials:
        for a in tr["acts"]:
            t = tr["timing"].get(a.ordinal)
            if t is None or t.decide_s is None:
                continue
            if a.kind == "stop":
                key = "reply with no action" if a.note == NO_ACTION else "stop"
            else:
                key = a.kind
            decide[key].append(float(t.decide_s))
    print(
        "  median decision time: "
        + ", ".join(
            f"{k} {statistics.median(v):.2f} s (n = {len(v)})" for k, v in sorted(decide.items())
        )
    )

    usage = json.loads((folder / "run_meta.json").read_text())["provenance"]["usage_all_processes"]
    print(
        f"  output tokens per decision: {usage['token_usage']['output_tokens'] / usage['llm_calls']:.0f}"
    )

    # Compromises they likely cost: an attempt that ended in such a reply after p actions, with
    # no compromise yet, is credited with the share of the condition's other attempts that had
    # taken more than p actions without a compromise before action p, kept going, and were
    # compromised at action p or later.
    reached = collections.Counter()
    went_on = collections.Counter()
    for tr in trials:
        if tr["no_action"]:
            continue
        for p in range(1, len(tr["acts"])):
            if tr["first_win"] is not None and tr["first_win"] < p:
                break
            reached[(tr["condition"], p)] += 1
            went_on[(tr["condition"], p)] += tr["first_win"] is not None
    print("  compromises they likely cost:")
    for c in CONDITIONS:
        lost = 0.0
        for tr in trials:
            if tr["condition"] != c or not tr["no_action"] or tr["first_win"] is not None:
                continue
            p = len(tr["acts"]) - 1
            lost += went_on[(c, p)] / reached[(c, p)] if reached[(c, p)] else 0.0
        print(
            f"    {c:24s} {lost:4.1f}  (compromised trials {sum(tr['won'] for tr in trials if tr['condition'] == c)} of {n[c]})"
        )
    early = [tr for tr in trials if tr["attempt"] >= 1 and tr["model_stop"] and not tr["sent"]]
    print(
        f"  later attempts the model itself stopped before sending any exploit: {len(early)} "
        + str(dict(collections.Counter(tr["condition"] for tr in early)))
    )


def first_target(trials: list[dict]) -> None:
    print(
        "\nLater attempts (attempt index 1-7): compromised / attempts, by the target of the first exploit"
    )
    for label, keep in (
        ("all", lambda tr: True),
        ("without attempts that ended in a reply with no action", lambda tr: not tr["no_action"]),
    ):
        print(f"  {label}:")
        for c in CONDITIONS:
            cell = {True: [0, 0], False: [0, 0]}
            for tr in trials:
                if tr["condition"] != c or tr["attempt"] == 0 or not tr["sent"] or not keep(tr):
                    continue
                decoy = tr["sent"][0].target.service in DECOYS
                cell[decoy][0] += tr["won"]
                cell[decoy][1] += 1
            d, g = cell[True], cell[False]
            print(
                f"    {c:24s} decoy first {d[0]} / {d[1]}   genuine service first {g[0]} / {g[1]}"
            )


def record_of_earlier_attempts(trials: list[dict]) -> None:
    print(
        "\nStatic baseline and deceptive presentation, later attempts, by whether the record of earlier"
        "\nattempts showed all five genuine services compromised"
    )
    by_sequence = collections.defaultdict(list)
    for tr in trials:
        by_sequence[tr["sequence"]].append(tr)
    table = collections.defaultdict(collections.Counter)
    for runs in by_sequence.values():
        seen: set[str] = set()
        for tr in sorted(runs, key=lambda x: x["attempt"]):
            if tr["attempt"] >= 1:
                cell = table[(tr["condition"], GENUINE <= seen)]
                cell["attempts"] += 1
                if not tr["won"]:
                    if not tr["sent"]:
                        kind = (
                            "stopped before any exploit"
                            if tr["model_stop"]
                            else ("reply with no action" if tr["no_action"] else "other")
                        )
                    elif not any(
                        tr["timing"].get(a.ordinal) and tr["timing"][a.ordinal].service_genuine
                        for a in tr["sent"]
                    ):
                        kind = "exploits at decoys only"
                    else:
                        kind = "exploits at genuine services"
                    cell["uncompromised"] += 1
                    cell[kind] += 1
            seen |= {
                a.target.service for a in tr["sent"] if compromised(tr["timing"].get(a.ordinal))
            }
    for c in ("static_baseline", "deceptive_presentation"):
        for full in (True, False):
            cell = table[(c, full)]
            detail = {k: v for k, v in cell.items() if k not in ("attempts", "uncompromised")}
            print(
                f"  {c:24s} {'all five' if full else 'fewer':8s} attempts {cell['attempts']:4d}  "
                f"uncompromised {cell['uncompromised']:3d}  {detail if detail else ''}"
            )


def folds(df: pd.DataFrame) -> None:
    print(
        "\nRQ4 regularized selection: sequences whose attempts fall in more than one of the five folds"
    )

    def split(d: pd.DataFrame, splitter, y=None) -> str:
        fold = np.empty(len(d), dtype=int)
        for k, (_, test) in enumerate(splitter.split(np.zeros(len(d)), y)):
            fold[test] = k
        per_sequence = collections.defaultdict(set)
        for s, f in zip(df.loc[d.index, "sequence_id"], fold, strict=True):
            per_sequence[s].add(f)
        n_split = sum(1 for v in per_sequence.values() if len(v) > 1)
        return f"{n_split} of {len(per_sequence)} sequences ({len(d)} trials)"

    target = "time_to_compromise_s"
    d = _numeric(df, [target, *[p for p in METRIC_SET if p != target]]).dropna()
    print(f"  elastic net, time-to-compromise (consecutive folds): {split(d, KFold(5))}")
    target = "exploitation_success"
    d = _numeric(df, [p for p in METRIC_SET if p != target]).copy()
    d[target] = pd.to_numeric(df.loc[d.index, target], errors="coerce")
    d = d.dropna()
    y = d[target].astype(int).to_numpy()
    print(
        f"  L1 logistic, exploitation success (folds stratified by outcome): {split(d, StratifiedKFold(5), y)}"
    )


def main() -> None:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs" / "results" / "full"
    df, trials = load(folder)
    shown = folder.resolve().relative_to(ROOT) if folder.resolve().is_relative_to(ROOT) else folder
    print(f"{shown}: {len(trials)} trials\n")
    delivery(trials)
    no_action_replies(trials, folder)
    first_target(trials)
    record_of_earlier_attempts(trials)
    folds(df)


if __name__ == "__main__":
    main()
