"""Tests for capture/capture_log.py - the optional debugging/traceability sidecar log.

No network calls here; this module is pure file I/O plus dict-building.
"""
import json

from capture import capture_log

SECRET_COOKIE_VALUE = "super-secret-session-token-do-not-leak-99999"


def test_build_log_entry_contains_no_header_or_secret_fields():
    entry = capture_log.build_log_entry(
        row_id="GD-V1-0001", query="what is the FTA count?", query_field="query",
        endpoint_url="https://example.invalid/api/generate/stream",
        http_status=200, complete_event={"answer": "the answer", "sources": []},
        answer="the answer", source_filenames=["doc.md"], started_at=1000.0,
        duration_seconds=3.2, success=True, error=None,
    )
    # the request payload is exactly {query_field: query} - no headers, no cookie
    assert entry["request_payload"] == {"query": "what is the FTA count?"}
    assert "headers" not in entry
    assert "cookie" not in json.dumps(entry).lower()
    assert SECRET_COOKIE_VALUE not in json.dumps(entry)


def test_build_log_entry_success_fields():
    entry = capture_log.build_log_entry(
        row_id="GD-V1-0001", query="q", query_field="query",
        endpoint_url="https://example.invalid/api/generate/stream",
        http_status=200, complete_event={"answer": "a", "sources": [{"filename": "x.md"}]},
        answer="a", source_filenames=["x.md"], started_at=0.0,
        duration_seconds=1.5, success=True, error=None,
    )
    assert entry["id"] == "GD-V1-0001"
    assert entry["http_status"] == 200
    assert entry["captured_answer"] == "a"
    assert entry["source_filenames"] == ["x.md"]
    assert entry["duration_seconds"] == 1.5
    assert entry["success"] is True
    assert entry["error"] is None
    assert entry["capture_timestamp"] == "1970-01-01T00:00:00Z"


def test_build_log_entry_failure_fields():
    entry = capture_log.build_log_entry(
        row_id="GD-0001", query="q", query_field="query",
        endpoint_url="https://example.invalid/api/generate/stream",
        http_status=401, complete_event=None, answer=None, source_filenames=[],
        started_at=0.0, duration_seconds=0.4, success=False,
        error="HTTP 401 - session rejected",
    )
    assert entry["success"] is False
    assert entry["http_status"] == 401
    assert entry["captured_answer"] is None
    assert entry["source_filenames"] == []
    assert "401" in entry["error"]


def test_derive_log_path_mirrors_historical_naming_convention(tmp_path):
    captured_path = tmp_path / "golden_dataset_new_chatbot_captured.json"
    log_path = capture_log.derive_log_path(captured_path)
    assert log_path.name == "golden_dataset_new_chatbot_captured.capture_log.json"


def test_save_log_then_load_round_trips(tmp_path):
    path = tmp_path / "out.capture_log.json"
    entries = [
        capture_log.build_log_entry(
            row_id="A", query="q1", query_field="query", endpoint_url="u",
            http_status=200, complete_event={"answer": "a1"}, answer="a1",
            source_filenames=[], started_at=0.0, duration_seconds=1.0,
            success=True, error=None),
    ]
    capture_log.save_log(path, entries)
    loaded = capture_log.load_existing_log(path)
    assert len(loaded) == 1
    assert loaded[0]["id"] == "A"


def test_save_log_merges_by_id_without_dropping_other_rows(tmp_path):
    path = tmp_path / "out.capture_log.json"
    capture_log.save_log(path, [
        capture_log.build_log_entry(
            row_id="A", query="q1", query_field="query", endpoint_url="u",
            http_status=200, complete_event=None, answer="a1", source_filenames=[],
            started_at=0.0, duration_seconds=1.0, success=True, error=None),
        capture_log.build_log_entry(
            row_id="B", query="q2", query_field="query", endpoint_url="u",
            http_status=200, complete_event=None, answer="a2", source_filenames=[],
            started_at=0.0, duration_seconds=1.0, success=True, error=None),
    ])
    # A second run only re-captures row A - row B's earlier entry must survive the merge
    capture_log.save_log(path, [
        capture_log.build_log_entry(
            row_id="A", query="q1", query_field="query", endpoint_url="u",
            http_status=200, complete_event=None, answer="a1-v2", source_filenames=[],
            started_at=0.0, duration_seconds=1.0, success=True, error=None),
    ])
    loaded = {e["id"]: e for e in capture_log.load_existing_log(path)}
    assert set(loaded) == {"A", "B"}
    assert loaded["A"]["captured_answer"] == "a1-v2"
    assert loaded["B"]["captured_answer"] == "a2"


def test_load_existing_log_missing_file_returns_empty_list(tmp_path):
    assert capture_log.load_existing_log(tmp_path / "does-not-exist.json") == []
