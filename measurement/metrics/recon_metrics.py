"""Reconnaissance metrics: recon accuracy (F1) and attack-path churn.

Definitions: ``docs/metrics_specification.md`` §2.3–§2.4.

- recon_accuracy_f1: F1 of the attacker's discovered service inventory vs. the
  current ground truth. **Precision** excludes deceptive services from the
  true-positive set (a discovered decoy is a false positive). **Recall** is over
  genuine services only.
- attack_path_churn: Jaccard distance (1 − Jaccard similarity) between successive
  recon snapshots, where a snapshot is a set of service–port–endpoint tuples.
"""

from __future__ import annotations

from dataclasses import dataclass

from measurement.models import GroundTruthInventory, ReconSnapshot, ServiceTuple


@dataclass(frozen=True)
class ReconAccuracy:
    """Components of the recon-accuracy computation (all deterministic)."""

    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def recon_accuracy(
    discovered: frozenset[ServiceTuple], ground_truth: GroundTruthInventory
) -> ReconAccuracy:
    """Compute precision/recall/F1 of ``discovered`` vs. ``ground_truth``.

    - TP = genuine services correctly discovered.
    - FP = discovered items that are decoys or non-existent (``discovered − genuine``).
    - FN = genuine services missed (``genuine − discovered``).

    F1 is 0.0 when precision + recall == 0 (including the degenerate all-empty
    case). Deceptive services are never counted as true positives.
    """

    genuine = ground_truth.genuine
    tp = len(discovered & genuine)
    fp = len(discovered - genuine)
    fn = len(genuine - discovered)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return ReconAccuracy(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn)


def recon_accuracy_f1(
    discovered: frozenset[ServiceTuple], ground_truth: GroundTruthInventory
) -> float:
    """Convenience wrapper returning only the F1 score (0–1)."""

    return recon_accuracy(discovered, ground_truth).f1


def jaccard_distance(a: frozenset[ServiceTuple], b: frozenset[ServiceTuple]) -> float:
    """Jaccard distance ``1 − |a ∩ b| / |a ∪ b|``. Zero for two empty sets."""

    union = a | b
    if not union:
        return 0.0
    return 1.0 - len(a & b) / len(union)


def attack_path_churn_series(
    snapshots: tuple[ReconSnapshot, ...],
) -> list[float | None]:
    """Churn per attempt, ordered by ``attempt_index``.

    The first snapshot in a sequence has no predecessor → ``None`` (recorded as
    null with reason ``first_in_sequence`` in the dataset).
    """

    ordered = sorted(snapshots, key=lambda s: s.attempt_index)
    out: list[float | None] = []
    for i, snap in enumerate(ordered):
        if i == 0:
            out.append(None)
        else:
            out.append(jaccard_distance(ordered[i - 1].discovered, snap.discovered))
    return out
