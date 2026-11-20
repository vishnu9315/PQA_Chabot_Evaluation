"""Golden dataset loading/saving for the capture pipeline: resumability and strict
ground-truth-field preservation.

Nothing here ever writes to the golden dataset path it reads from - `save_captured_dataset`
always writes to a separate output path supplied by the caller.
"""
from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger("capture_new_chatbot")

# Fields this script must NEVER modify (ground truth / scoring metadata). Mirrors
# ragas_eval/dataset.py's _REQUIRED_ROW plus the extra fields the capture task calls out
# (question_origin isn't in _REQUIRED_ROW but is a required Sample field there too).
PROTECTED_FIELDS = (
    "id", "user_input", "reference", "reference_contexts", "must_include_facts",
    "expected_sources", "expected_behavior", "ragas_eligible", "stratum", "category",
    "difficulty", "question_origin", "house",
)

# Fields this script is responsible for populating from a live capture.
CAPTURE_FIELDS = ("response", "retrieved_contexts", "cited_sources", "cited_sources_raw")

# Defaults are this project's dataset (meity-pq-SHORT-2part-v1: 66 rows, all
# ragas_eligible, no judge-only rows); override with these env vars if the dataset changes.
EXPECTED_TOTAL_ROWS = int(os.environ.get("GOLDEN_EXPECTED_TOTAL_ROWS", "66"))
EXPECTED_RAGAS_ELIGIBLE = int(os.environ.get("GOLDEN_EXPECTED_RAGAS_ELIGIBLE", "66"))
_judge_only_raw = os.environ.get("GOLDEN_EXPECTED_JUDGE_ONLY_IDS", "")
EXPECTED_JUDGE_ONLY_IDS = tuple(x.strip() for x in _judge_only_raw.split(",") if x.strip())


class DatasetIntegrityError(RuntimeError):
    """The golden dataset (or a captured copy of it) failed a structural/count check."""


def load_golden_dataset(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "rows" not in data:
        raise DatasetIntegrityError(f"{path}: missing top-level 'rows' key")
    return data


def validate_row_counts(data: dict, *, where: str) -> None:
    """Raises DatasetIntegrityError unless the dataset has exactly the expected shape:
    EXPECTED_TOTAL_ROWS rows, EXPECTED_RAGAS_ELIGIBLE ragas_eligible=True, and exactly the
    EXPECTED_JUDGE_ONLY_IDS
    with ragas_eligible=False. Called both before capture (on the golden input) and
    after (on the saved output), per the task's "validate before and after" requirement.
    """
    rows = data["rows"]
    if len(rows) != EXPECTED_TOTAL_ROWS:
        raise DatasetIntegrityError(
            f"{where}: expected {EXPECTED_TOTAL_ROWS} rows, found {len(rows)}")

    ids = [r.get("id") for r in rows]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise DatasetIntegrityError(f"{where}: duplicate row ids present: {dupes}")

    eligible = [r["id"] for r in rows if r.get("ragas_eligible") is True]
    if len(eligible) != EXPECTED_RAGAS_ELIGIBLE:
        raise DatasetIntegrityError(
            f"{where}: expected {EXPECTED_RAGAS_ELIGIBLE} ragas_eligible rows, "
            f"found {len(eligible)}")

    judge_only = sorted(r["id"] for r in rows if r.get("ragas_eligible") is False)
    if judge_only != sorted(EXPECTED_JUDGE_ONLY_IDS):
        raise DatasetIntegrityError(
            f"{where}: expected judge-only row(s) {sorted(EXPECTED_JUDGE_ONLY_IDS)}, "
            f"found {judge_only}")


def load_existing_capture(path: Path) -> Optional[dict]:
    """The output of a previous capture run, if present - used to resume."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def row_already_captured(row: dict) -> bool:
    """True if this row already holds a genuine successful capture that --resume should
    skip: a non-empty response and no recorded capture_error."""
    return bool(row.get("response")) and not row.get("capture_error")


def assert_protected_fields_unchanged(original_row: dict, new_row: dict) -> None:
    """Defensive runtime check: raises if any ground-truth field was touched. This should
    be structurally impossible given how build_output_rows constructs rows (deepcopy +
    only CAPTURE_FIELDS/bookkeeping fields mutated), but it is asserted on every row
    anyway, since the task treats this as a strict, must-never-happen requirement."""
    for f in PROTECTED_FIELDS:
        if original_row.get(f) != new_row.get(f):
            raise DatasetIntegrityError(
                f"row {original_row.get('id')!r}: protected field {f!r} was modified - "
                f"refusing to save")


def build_output_rows(golden_data: dict, resume_from: Optional[dict]) -> list[dict]:
    """Start from a deep copy of every golden row (so ground truth is untouched by
    construction), carrying over any already-successful capture from `resume_from` so
    resuming is a true no-op for rows that already succeeded.

    `resume_from` being None (no prior capture file, or capture fields intentionally
    ignored) simply means every row starts with empty capture fields.
    """
    resume_by_id = {r["id"]: r for r in (resume_from or {}).get("rows", [])}
    out: list[dict] = []
    for golden_row in golden_data["rows"]:
        rid = golden_row["id"]
        row = copy.deepcopy(golden_row)
        row.setdefault("response", "")
        row.setdefault("retrieved_contexts", [])
        row.setdefault("cited_sources", [])
        row.setdefault("cited_sources_raw", [])
        row.setdefault("capture_status", "not_captured")
        row.setdefault("capture_error", None)

        prior = resume_by_id.get(rid)
        if prior and row_already_captured(prior):
            for f in CAPTURE_FIELDS:
                if f in prior:
                    row[f] = prior[f]
            row["capture_status"] = prior.get("capture_status", "captured")
            row["capture_error"] = None
        out.append(row)
    return out


def save_captured_dataset(path: Path, golden_data: dict, rows: list[dict], *, meta: dict) -> None:
    """Write `rows` (golden ground truth + captured fields) to `path`. Writes to a
    temporary file first and renames it into place, so a crash mid-write never leaves a
    half-written output file behind. Never writes to the golden dataset's own path -
    callers must pass a distinct output path."""
    out = copy.deepcopy(golden_data)
    out["rows"] = rows
    out["capture_meta"] = meta
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)
    logger.info("saved %d rows -> %s", len(rows), path)
