"""Tests for capture/dataset_io.py: validation, resumability, and the strict
ground-truth-preservation guarantee. No network calls for the synthetic-fixture tests below.

A handful of tests also run read-only against YOUR real golden dataset once you've added one
(see README.md) - they auto-skip until GOLDEN_DATASET_PATH points at a real file, so this
suite passes out of the box on a fresh clone with no dataset yet.
"""
import copy
import json
import os
from pathlib import Path

import pytest

from capture.dataset_io import (
    DatasetIntegrityError,
    PROTECTED_FIELDS,
    assert_protected_fields_unchanged,
    build_output_rows,
    load_golden_dataset,
    row_already_captured,
    save_captured_dataset,
    validate_row_counts,
)

# Customize via GOLDEN_DATASET_PATH (same env var capture_new_chatbot.py reads) once you have
# a real dataset in place; defaults to a sibling file that doesn't exist yet on a fresh clone.
REAL_GOLDEN_PATH = Path(os.environ.get(
    "GOLDEN_DATASET_PATH", str(Path(__file__).resolve().parent.parent / "golden_dataset.json")))
_requires_real_dataset = pytest.mark.skipif(
    not REAL_GOLDEN_PATH.exists(),
    reason=f"no real golden dataset at {REAL_GOLDEN_PATH} yet - see README.md")


@pytest.fixture(scope="module")
def real_golden_bytes() -> bytes:
    return REAL_GOLDEN_PATH.read_bytes()


@pytest.fixture(scope="module")
def real_golden_data() -> dict:
    return load_golden_dataset(REAL_GOLDEN_PATH)


# --- validate_row_counts against YOUR real dataset, once one exists -----------------

@_requires_real_dataset
def test_real_golden_dataset_passes_validation(real_golden_data):
    validate_row_counts(real_golden_data, where=REAL_GOLDEN_PATH.name)


@_requires_real_dataset
def test_loading_real_golden_dataset_never_writes_to_it(real_golden_bytes):
    """Loading + validating must be read-only."""
    load_golden_dataset(REAL_GOLDEN_PATH)
    assert REAL_GOLDEN_PATH.read_bytes() == real_golden_bytes


# --- validate_row_counts: synthetic failure cases -----------------------------------
# These read the module's own (env-configurable - see dataset_io.py's TEMPLATE note)
# EXPECTED_* constants rather than hardcoding 45/44/GD-0165, so they stay correct however
# GOLDEN_EXPECTED_TOTAL_ROWS/etc. are set for your own dataset.

from capture.dataset_io import (  # noqa: E402
    EXPECTED_JUDGE_ONLY_IDS,
    EXPECTED_RAGAS_ELIGIBLE,
    EXPECTED_TOTAL_ROWS,
)

_A_JUDGE_ONLY_ID = EXPECTED_JUDGE_ONLY_IDS[0] if EXPECTED_JUDGE_ONLY_IDS else "JUDGE-ONLY-0"


def _minimal_dataset(n_rows, n_eligible, judge_only_id=_A_JUDGE_ONLY_ID):
    rows = []
    for i in range(n_rows):
        eligible = i < n_eligible
        rid = judge_only_id if not eligible and i == n_rows - 1 else f"ROW-{i}"
        rows.append({"id": rid, "ragas_eligible": eligible})
    return {"rows": rows}


def test_validate_row_counts_rejects_wrong_total():
    data = _minimal_dataset(n_rows=EXPECTED_TOTAL_ROWS - 1, n_eligible=EXPECTED_RAGAS_ELIGIBLE)
    with pytest.raises(DatasetIntegrityError, match=f"expected {EXPECTED_TOTAL_ROWS} rows"):
        validate_row_counts(data, where="test")


def test_validate_row_counts_rejects_wrong_eligible_count():
    data = _minimal_dataset(n_rows=EXPECTED_TOTAL_ROWS, n_eligible=0)
    with pytest.raises(DatasetIntegrityError,
                        match=f"expected {EXPECTED_RAGAS_ELIGIBLE} ragas_eligible"):
        validate_row_counts(data, where="test")


def test_validate_row_counts_rejects_wrong_judge_only_id():
    data = _minimal_dataset(n_rows=EXPECTED_TOTAL_ROWS, n_eligible=EXPECTED_RAGAS_ELIGIBLE,
                             judge_only_id="SOME-OTHER-ID-NOT-EXPECTED")
    if EXPECTED_TOTAL_ROWS == EXPECTED_RAGAS_ELIGIBLE:
        pytest.skip("no judge-only rows expected for this dataset shape - nothing to mismatch")
    with pytest.raises(DatasetIntegrityError, match="judge-only row"):
        validate_row_counts(data, where="test")


def test_validate_row_counts_rejects_duplicate_ids():
    data = _minimal_dataset(n_rows=EXPECTED_TOTAL_ROWS, n_eligible=EXPECTED_RAGAS_ELIGIBLE)
    data["rows"][1]["id"] = data["rows"][0]["id"]
    with pytest.raises(DatasetIntegrityError, match="duplicate row ids"):
        validate_row_counts(data, where="test")


# --- build_output_rows: ground-truth preservation + resumability -------------------

@pytest.fixture
def tiny_golden():
    return {
        "n_rows": 2,
        "rows": [
            {
                "id": "ROW-A", "user_input": "question A", "reference": "ref A",
                "reference_contexts": ["ctx A"], "must_include_facts": [],
                "expected_sources": [], "expected_behavior": "answer_full",
                "ragas_eligible": True, "stratum": "s", "category": "c",
                "difficulty": "easy", "question_origin": "paraphrased_past", "house": "ls",
                "response": "", "retrieved_contexts": [], "cited_sources": [],
                "cited_sources_raw": [],
            },
            {
                "id": "ROW-B", "user_input": "question B", "reference": "ref B",
                "reference_contexts": [], "must_include_facts": [],
                "expected_sources": [], "expected_behavior": "abstain",
                "ragas_eligible": False, "stratum": "s2", "category": "c2",
                "difficulty": "hard", "question_origin": "novel_session", "house": "rs",
                "response": "", "retrieved_contexts": [], "cited_sources": [],
                "cited_sources_raw": [],
            },
        ],
    }


def test_build_output_rows_preserves_protected_fields(tiny_golden):
    out = build_output_rows(tiny_golden, resume_from=None)
    for golden_row, new_row in zip(tiny_golden["rows"], out):
        for f in PROTECTED_FIELDS:
            assert golden_row[f] == new_row[f], f"{f} changed for {golden_row['id']}"


def test_build_output_rows_starts_with_empty_capture_fields_when_no_resume(tiny_golden):
    out = build_output_rows(tiny_golden, resume_from=None)
    for row in out:
        assert row["response"] == ""
        assert row["retrieved_contexts"] == []
        assert row["cited_sources"] == []
        assert row["capture_status"] == "not_captured"


def test_build_output_rows_resumes_prior_successful_capture(tiny_golden):
    resume_from = {
        "rows": [
            {"id": "ROW-A", "response": "a real answer", "retrieved_contexts": [],
             "cited_sources": ["doc.md"], "cited_sources_raw": [{"filename": "doc.md"}],
             "capture_status": "captured", "capture_error": None},
        ]
    }
    out = build_output_rows(tiny_golden, resume_from=resume_from)
    row_a = next(r for r in out if r["id"] == "ROW-A")
    row_b = next(r for r in out if r["id"] == "ROW-B")
    assert row_a["response"] == "a real answer"
    assert row_a["cited_sources"] == ["doc.md"]
    assert row_b["response"] == ""  # untouched - no prior capture for ROW-B


def test_build_output_rows_does_not_resume_failed_rows(tiny_golden):
    resume_from = {
        "rows": [
            {"id": "ROW-A", "response": "", "retrieved_contexts": [], "cited_sources": [],
             "cited_sources_raw": [], "capture_status": "failed",
             "capture_error": "HTTP 500"},
        ]
    }
    out = build_output_rows(tiny_golden, resume_from=resume_from)
    row_a = next(r for r in out if r["id"] == "ROW-A")
    assert row_a["response"] == ""
    assert row_already_captured(row_a) is False


def test_row_already_captured():
    assert row_already_captured({"response": "something", "capture_error": None}) is True
    assert row_already_captured({"response": "", "capture_error": None}) is False
    assert row_already_captured({"response": "x", "capture_error": "boom"}) is False
    assert row_already_captured({}) is False


# --- assert_protected_fields_unchanged ----------------------------------------------

def test_assert_protected_fields_unchanged_passes_on_identical_rows(tiny_golden):
    row = copy.deepcopy(tiny_golden["rows"][0])
    assert_protected_fields_unchanged(tiny_golden["rows"][0], row)  # no raise


def test_assert_protected_fields_unchanged_raises_on_mutated_field(tiny_golden):
    original = tiny_golden["rows"][0]
    mutated = copy.deepcopy(original)
    mutated["reference"] = "a different reference entirely"
    with pytest.raises(DatasetIntegrityError, match="reference"):
        assert_protected_fields_unchanged(original, mutated)


def test_assert_protected_fields_unchanged_ignores_capture_fields(tiny_golden):
    original = tiny_golden["rows"][0]
    mutated = copy.deepcopy(original)
    mutated["response"] = "a captured answer"  # allowed to change
    mutated["cited_sources"] = ["doc.md"]
    assert_protected_fields_unchanged(original, mutated)  # no raise


# --- save_captured_dataset: isolation from the golden file --------------------------

def test_save_captured_dataset_writes_loadable_file(tmp_path, tiny_golden):
    out_path = tmp_path / "captured.json"
    rows = build_output_rows(tiny_golden, resume_from=None)
    save_captured_dataset(out_path, tiny_golden, rows, meta={"note": "test"})
    assert out_path.exists()
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert len(saved["rows"]) == 2
    assert saved["capture_meta"]["note"] == "test"


@_requires_real_dataset
def test_save_captured_dataset_never_touches_the_golden_file(tmp_path, real_golden_bytes):
    """Saving a captured copy must never write to the real golden dataset path, even
    when the in-memory golden_data dict originated from it."""
    golden_data = load_golden_dataset(REAL_GOLDEN_PATH)
    out_path = tmp_path / "captured.json"
    rows = build_output_rows(golden_data, resume_from=None)
    save_captured_dataset(out_path, golden_data, rows, meta={})
    assert REAL_GOLDEN_PATH.read_bytes() == real_golden_bytes
    assert out_path.exists()
    assert out_path != REAL_GOLDEN_PATH


def test_save_captured_dataset_writes_atomically_no_leftover_tmp_file(tmp_path, tiny_golden):
    out_path = tmp_path / "captured.json"
    rows = build_output_rows(tiny_golden, resume_from=None)
    save_captured_dataset(out_path, tiny_golden, rows, meta={})
    assert not (tmp_path / "captured.json.tmp").exists()
