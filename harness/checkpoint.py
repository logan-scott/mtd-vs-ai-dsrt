"""Crash-safe checkpointing + resume for the full confirmatory run.

The experiment loop is deterministic: ``randomized_block_order`` fixes the block order
and each sequence has a deterministic ``seq_seed`` and non-overlapping timeline slot. That
lets the run resume at **sequence** granularity — each completed sequence's rows and uptime
events are appended (flushed + ``fsync``'d) to on-disk partials and the ``sequence_id`` is
recorded in ``_completed.txt``, so a crash loses at most the in-flight sequence, and a
restart **skips** already-done sequences while replaying the identical seeded order. The
remaining sequences therefore get byte-identical seeds/timestamps to an uninterrupted run.

This is operational only: it changes nothing about the science (seeds, order, metrics, the
offline<->live parity contract are all untouched).

File layout under the run's output dir (``data/full/`` by default; gitignored):
  - ``_run_manifest.json``            — config fingerprint + the run-defining fields
  - ``trial_dataset.partial.jsonl``   — lossless per-sequence rows (assembly source)
  - ``trial_dataset.partial.csv``     — human-visible RAW append log (tail-able); it is not
        deduplicated, so after a crash it may contain a re-run sequence's rows twice — the
        authoritative, deduplicated dataset is the final ``trial_dataset.csv``
  - ``uptime_events.partial.jsonl``   — per-sequence service-uptime event records
  - ``_completed.txt``                — one line per processed sequence (skip-on-resume)
  - ``_excluded_sequences.jsonl``     — sequences lost to an unrecoverable fault
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from measurement import run_sheet
from measurement.models import TrialRow

MANIFEST = "_run_manifest.json"
COMPLETED = "_completed.txt"
ROWS_JSONL = "trial_dataset.partial.jsonl"
ROWS_CSV = "trial_dataset.partial.csv"
UPTIME_JSONL = "uptime_events.partial.jsonl"
EXCLUDED_JSONL = "_excluded_sequences.jsonl"
#: Cumulative API usage per collection process (one line per finished sequence), so a resumed
#: run's provenance covers every process, not just the one that finished it.
USAGE_JSONL = "_usage.jsonl"

CHECKPOINT_FILES = (
    MANIFEST,
    COMPLETED,
    ROWS_JSONL,
    ROWS_CSV,
    UPTIME_JSONL,
    EXCLUDED_JSONL,
    USAGE_JSONL,
)


def fingerprint_payload(config, backend: str | None = None) -> dict:
    """The run-defining fields whose change must invalidate a resume (science-relevant).

    ``attacker_clock`` is included ONLY when it is not the default modeled clock: the clock
    decides at which instants the MTD schedule is consulted, so a run must never resume across
    a change of clock, yet every checkpoint written before the setting existed ran the modeled
    clock and must keep its fingerprint (a run in flight across this upgrade still resumes).
    ``record_timing`` is deliberately absent: it adds instrumentation columns and changes no
    outcome, so toggling it only changes how many rows carry timing. ``run_order`` and the live
    ``scan_scope`` follow the clock's rule: present only when not their default.
    ``backend`` is the attacker backend the run uses (``cli.py run --backend``); None reads the
    config's own ``attacker.backend``.
    """

    payload = _base_fingerprint_payload(config)
    clock_mode = getattr(getattr(config, "attacker", None), "clock", "modeled")
    if clock_mode != "modeled":
        payload["attacker_clock"] = clock_mode
        # How the wall clock dates what a scan saw. Version 2 dates it to the instant the scan
        # probed the targets host; the 2026-10-26 pilot (v1) dated it to the scan's return. A
        # wall-clock run never resumes across that change.
        payload["attacker_clock_dating"] = "probe_time"
        # How the wall clock scores an exploit: when its delivery COMPLETES (decision A1,
        # 2026-10-31; timing format 3), no longer only when it was fired. A wall-clock run never
        # resumes across that change.
        payload["attacker_clock_scoring"] = "completion"
    # Like the clock, the next two enter the fingerprint only when not their default, so every
    # checkpoint written before they existed keeps its fingerprint.
    run_order = getattr(config, "run_order", "blocked")
    if run_order != "blocked":
        payload["run_order"] = run_order
    scan_scope = getattr(getattr(config, "live", None), "scan_scope", "subnet")
    if scan_scope != "subnet":
        payload["scan_scope"] = scan_scope
    # Whether an attempt may start (and a scan be scored) against an incomplete surface: it
    # decides which trials are measured at all, so a run never resumes across it.
    surface_check = getattr(getattr(config, "live", None), "surface_check", "off")
    if surface_check != "off":
        payload["surface_check"] = surface_check
    # The attacker's prompt format (``attacker.base.PROMPT_FORMAT_VERSION``). Format 2
    # (2026-10-31) shows the attacker what it observed in earlier attempts, which changes what
    # the attacker is, so a run never resumes across a change of format: a checkpoint written
    # under format 1 (every earlier run) refuses to resume under this code, and the reverse.
    # Only the anthropic backend renders the prompt, so the offline backends' fingerprints
    # are unchanged. It is the backend the run USES that counts (the CLI's ``--backend``), not
    # the one the config names: the CLI does not read the config's.
    if backend is None:
        backend = getattr(getattr(config, "attacker", None), "backend", "mock")
    if backend == "anthropic":
        from attacker.base import PROMPT_FORMAT_VERSION

        payload["attacker_prompt_format"] = PROMPT_FORMAT_VERSION
    return payload


def _base_fingerprint_payload(config) -> dict:
    """The fingerprint fields every run has carried since the manifest existed."""

    return {
        "conditions": list(config.conditions),
        "sequences_per_condition": config.sequences_per_condition,
        "attempts_per_sequence": config.attempts_per_sequence,
        "randomization_seed": config.randomization_seed,
        "vuln_profile_version": config.vuln_profile_version,
        "attacker_model_id": config.attacker.model_id,
        "attacker_temperature": config.attacker.temperature,
        # Request shaping that can change what the model does (forcing a tool call
        # removes the model's option to answer in prose, which the client would coerce
        # into a stop). A run must never resume across a change to it.
        "attacker_tool_choice": config.attacker.tool_choice,
        "mtd_cadence_s": config.mtd_cadence_s,
        "mtd_rotation_phase": config.mtd_rotation_phase,
        # Also science-relevant: the action budget and the right-censoring bound. Changing
        # either mid-run would make resumed sequences incomparable to the completed ones.
        "max_actions": config.max_actions,
        "max_trial_duration_s": config.max_trial_duration_s,
        # Anything else that changes an attacker action, the synthetic telemetry, or a
        # per-row provenance value, so a mid-run change is refused rather than mixed in.
        "attacker_model_version": config.attacker.model_version,
        "system_prompt_path": config.attacker.system_prompt_path,
        "monitoring_interval_s": config.monitoring_interval_s,
        "legitimate_rate_rps": config.workload.legitimate_rate_rps,
    }


def payload_fingerprint(payload: dict) -> str:
    """Stable short hash of a fingerprint payload."""

    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def config_fingerprint(config) -> str:
    """Stable short hash of the run-defining config (see ``fingerprint_payload``)."""

    return payload_fingerprint(fingerprint_payload(config))


def _fsync_dir(directory: Path) -> None:
    """fsync a directory so a newly-created file's directory entry is durable."""

    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:  # pragma: no cover - platform without dir fds
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_append(path: Path, text: str) -> None:
    """Append ``text`` and durably flush it (file data + directory entry) to disk."""

    if not text:
        return
    new = not path.exists()
    with path.open("a", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    if new:
        _fsync_dir(path.parent)


def _fsync_write(path: Path, text: str) -> None:
    new = not path.exists()
    with path.open("w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    if new:
        _fsync_dir(path.parent)


def _row_from_record(obj: dict) -> TrialRow:
    """Rebuild a ``TrialRow`` from a checkpoint JSONL record, honouring optional columns.

    A checkpoint written BEFORE a column existed must not silently receive that column's
    dataclass default on resume: for ``attacker_max_retries`` the default ``0`` would
    assert a retry budget that was never recorded. Absent keys get the column's declared
    not-recorded value instead -- the same substitution ``run_sheet.read_csv`` performs
    for a legacy CSV.

    (The human-visible ``rows.csv`` mirror written by ``_append_rows`` writes its header
    once, so a resume ACROSS an upgrade appends wider rows under the older header. Nothing
    reads that file -- ``assemble`` uses the JSONL -- but it is why the JSONL, not the CSV
    mirror, is the checkpoint's authoritative record.)
    """

    filled = dict(obj)
    for column, absent in run_sheet.OPTIONAL_ON_READ.items():
        if column not in filled:
            filled[column] = absent
    return TrialRow(**filled)


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file, tolerating a trailing partial/blank line (crash-torn write)."""

    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a torn last line from a crash mid-write; the seq re-runs on resume
    return out


@dataclass
class RunCheckpoint:
    """Append-only, fsync'd sequence checkpoint for one run output directory."""

    dir: Path
    fingerprint: str
    # Identifies THIS collection process in the usage record (a resume is a new process).
    process_id: str = field(default_factory=lambda: uuid.uuid4().hex, compare=False)

    def __post_init__(self) -> None:
        self.dir = Path(self.dir)

    # --- paths --------------------------------------------------------------- #
    @property
    def manifest_path(self) -> Path:
        return self.dir / MANIFEST

    @property
    def completed_path(self) -> Path:
        return self.dir / COMPLETED

    @property
    def rows_jsonl_path(self) -> Path:
        return self.dir / ROWS_JSONL

    @property
    def rows_csv_path(self) -> Path:
        return self.dir / ROWS_CSV

    @property
    def uptime_path(self) -> Path:
        return self.dir / UPTIME_JSONL

    @property
    def usage_path(self) -> Path:
        return self.dir / USAGE_JSONL

    @property
    def excluded_path(self) -> Path:
        return self.dir / EXCLUDED_JSONL

    # --- lifecycle ----------------------------------------------------------- #
    def exists(self) -> bool:
        return self.manifest_path.exists()

    def stored_fingerprint(self) -> str | None:
        if not self.manifest_path.exists():
            return None
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8")).get(
                "config_fingerprint"
            )
        except (json.JSONDecodeError, OSError):
            return None

    def init(self, payload: dict, extra: dict | None = None) -> None:
        """Write the manifest on first start (idempotent — kept if it already matches).

        ``extra`` adds keys that are recorded ONCE, at the first start, outside the
        fingerprint -- e.g. the clock-validation pilot's decision criteria, which must be fixed
        before any data exist yet must not change the fingerprint of the run they govern.
        """

        self.dir.mkdir(parents=True, exist_ok=True)
        if not self.manifest_path.exists():
            doc = {"config_fingerprint": self.fingerprint, "config": payload}
            for key, value in (extra or {}).items():
                if key in doc:
                    raise ValueError(f"manifest key {key!r} is reserved")
                doc[key] = value
            _fsync_write(self.manifest_path, json.dumps(doc, indent=2) + "\n")

    def manifest(self) -> dict | None:
        """The manifest as written at the first start (None if absent or unreadable)."""

        if not self.manifest_path.exists():
            return None
        try:
            doc = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return doc if isinstance(doc, dict) else None

    def clear(self) -> None:
        """Remove all checkpoint files (``--fresh``).

        The manifest is deleted **last** (commit-marker semantics): the manifest's presence
        is what triggers resume, so as long as it exists it must describe valid partials. A
        torn ``--fresh`` (crash mid-clear) then leaves either a manifest with its own (now
        emptied) state or no manifest at all — never a manifest paired with a stale
        ``_completed.txt`` from a different run.
        """

        for name in CHECKPOINT_FILES:
            if name != MANIFEST:
                (self.dir / name).unlink(missing_ok=True)
        (self.dir / MANIFEST).unlink(missing_ok=True)
        if self.dir.exists():
            _fsync_dir(self.dir)

    # --- resume state -------------------------------------------------------- #
    def completed_ids(self) -> set[str]:
        if not self.completed_path.exists():
            return set()
        return {
            line.split("\t", 1)[0]
            for line in self.completed_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    # --- writes (rows first, completed marker LAST so a mid-write crash re-runs) --- #
    def append_sequence(
        self,
        sequence_id: str,
        block_index: int,
        seq_i: int,
        rows: list[TrialRow],
        uptime_events: list[dict],
    ) -> None:
        self._append_rows(rows)
        _fsync_append(
            self.uptime_path,
            "".join(json.dumps(e, sort_keys=True) + "\n" for e in uptime_events),
        )
        self._mark_completed(sequence_id, block_index, seq_i, "ok")

    def record_excluded_sequence(
        self,
        sequence_id: str,
        block_index: int,
        seq_i: int,
        reason: str,
        n_lost_attempts: int,
        detail: str | None = None,
    ) -> None:
        record = {
            "sequence_id": sequence_id,
            "block_index": block_index,
            "seq_i": seq_i,
            "reason": reason,
            "n_lost_attempts": n_lost_attempts,
        }
        if detail is not None:
            record["detail"] = detail  # what failed, in the fault's own words
        _fsync_append(self.excluded_path, json.dumps(record, sort_keys=True) + "\n")
        self._mark_completed(sequence_id, block_index, seq_i, f"excluded:{reason}")

    def record_usage(self, snapshot: dict) -> None:
        """Append this process's CUMULATIVE API usage (after a sequence; last line wins)."""

        _fsync_append(
            self.usage_path,
            json.dumps({"process": self.process_id, **snapshot}, sort_keys=True) + "\n",
        )

    def usage_by_process(self) -> dict[str, dict]:
        """The latest cumulative usage snapshot of every process that recorded one."""

        out: dict[str, dict] = {}
        for rec in _read_jsonl(self.usage_path):
            pid = rec.get("process")
            if isinstance(pid, str):
                out[pid] = {k: v for k, v in rec.items() if k != "process"}
        return out

    def _mark_completed(self, sequence_id: str, block_index: int, seq_i: int, status: str) -> None:
        _fsync_append(self.completed_path, f"{sequence_id}\t{block_index}\t{seq_i}\t{status}\n")

    def _append_rows(self, rows: list[TrialRow]) -> None:
        if not rows:
            return
        _fsync_append(
            self.rows_jsonl_path,
            "".join(json.dumps(run_sheet.row_to_dict(r)) + "\n" for r in rows),
        )
        # Human-visible CSV mirror (header written once).
        header = not self.rows_csv_path.exists() or self.rows_csv_path.stat().st_size == 0
        buf = io.StringIO()
        writer = csv.writer(buf)
        if header:
            writer.writerow(run_sheet.TRIAL_COLUMNS)
        for r in rows:
            d = run_sheet.row_to_dict(r)
            writer.writerow(["" if d[c] is None else d[c] for c in run_sheet.TRIAL_COLUMNS])
        _fsync_append(self.rows_csv_path, buf.getvalue())

    # --- assembly ------------------------------------------------------------ #
    def assemble(self) -> tuple[list[TrialRow], list[dict], list[dict]]:
        """Reconstruct (rows, uptime_events, excluded_sequences) for completed sequences.

        Only a sequence that was written but not marked before a crash — and therefore
        re-run on resume — can appear twice; the normal clean-stop/resume path skips
        completed sequences so has no duplicates. Deduplication must NOT lose legitimately
        identical records, so it is keyed on structure, never on content:

        - **rows**: keyed on the unique ``trial_id`` (last write wins).
        - **uptime**: a trial's events are written as one contiguous block per append (a
          re-run appends the block again), so we keep, per ``trial_id``, the events from
          its LAST contiguous block. This preserves genuinely-distinct events that happen
          to share content (e.g. equal-duration periodic outages) while dropping a
          re-appended duplicate block — matching the non-checkpoint path exactly.
        """

        completed = self.completed_ids()
        rows_by_tid: dict[str, TrialRow] = {}
        for obj in _read_jsonl(self.rows_jsonl_path):
            if obj.get("sequence_id") in completed:
                rows_by_tid[obj["trial_id"]] = _row_from_record(obj)
        rows = list(rows_by_tid.values())
        kept_tids = set(rows_by_tid)

        # Per-trial last-contiguous-block dedup (see docstring). Reset a trial's block
        # whenever the stream's trial_id changes to it, so a re-appended block replaces
        # (not extends) the prior one, but consecutive same-trial events are all kept.
        uptime_by_tid: dict[str, list[dict]] = {}
        prev_tid: str | None = None
        for obj in _read_jsonl(self.uptime_path):
            tid = obj.get("trial_id")
            if tid not in kept_tids:
                prev_tid = tid
                continue
            if tid != prev_tid:
                uptime_by_tid[tid] = []
            uptime_by_tid[tid].append(obj)
            prev_tid = tid
        uptime = [e for events in uptime_by_tid.values() for e in events]

        # A sequence that ultimately produced rows is NOT an exclusion, even if a crash
        # left a stale excluded record before it succeeded on resume (avoids double-count).
        row_seqs = {r.sequence_id for r in rows}
        excluded_by_seq: dict[str, dict] = {}
        for obj in _read_jsonl(self.excluded_path):
            seq = obj.get("sequence_id")
            if seq in completed and seq not in row_seqs:
                excluded_by_seq[seq] = obj
        return rows, uptime, list(excluded_by_seq.values())
