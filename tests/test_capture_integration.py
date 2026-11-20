"""End-to-end tests of capture_new_chatbot.main(), with the network layer mocked out
(capture.client._http_post is monkeypatched - no real HTTP call is ever made). Runs
against a small synthetic golden dataset written to tmp_path. One test also touches YOUR
real golden dataset (read-only, to confirm it's untouched) once you've added one - see
README.md; it auto-skips until then.
"""
import json
import os
import uuid
from pathlib import Path

import pytest

import capture_new_chatbot as cnc
from capture import client as client_mod

REAL_GOLDEN_PATH = Path(os.environ.get(
    "GOLDEN_DATASET_PATH", str(Path(__file__).resolve().parent.parent / "golden_dataset.json")))


def _make_golden(tmp_path: Path, n_rows: int = 3) -> Path:
    rows = []
    for i in range(n_rows):
        rows.append({
            "id": f"ROW-{i}", "user_input": f"question {i}", "reference": f"ref {i}",
            "reference_contexts": [], "must_include_facts": [], "expected_sources": [],
            "expected_behavior": "answer_full", "ragas_eligible": True,
            "stratum": "s", "category": "c", "difficulty": "easy",
            "question_origin": "paraphrased_past", "house": "ls",
            "response": "", "retrieved_contexts": [], "cited_sources": [],
            "cited_sources_raw": [],
        })
    data = {"note": "synthetic test dataset", "n_rows": n_rows, "rows": rows}
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _patch_expected_counts(monkeypatch):
    """The synthetic fixtures here use 3 rows, not the real 45/44/1 shape - relax the
    module-level expectations for just these tests rather than faking a 45-row dataset."""
    monkeypatch.setattr(cnc, "validate_row_counts", lambda data, where: None)


@pytest.fixture
def fake_env(monkeypatch):
    monkeypatch.setenv("PQA_CHATBOT_BASE_URL", "https://example.invalid/api")
    # A real .env on disk (PQA_CHATBOT_SESSION_FILE/COOKIE_HEADER, set for the live test)
    # gets loaded into os.environ as soon as capture_new_chatbot is imported, regardless
    # of this fixture - clear them explicitly so these tests see a clean synthetic env.
    monkeypatch.delenv("PQA_CHATBOT_SESSION_FILE", raising=False)
    monkeypatch.delenv("PQA_CHATBOT_COOKIE_HEADER", raising=False)


def _sse_ok(answer: str) -> str:
    payload = json.dumps({"answer": answer, "sources": [{"filename": "doc.md", "text_preview": "x"}]})
    return f"event: complete\ndata: {payload}\n\n"


def test_dry_run_makes_no_network_calls(tmp_path, fake_env, monkeypatch):
    golden_path = _make_golden(tmp_path)
    calls = {"n": 0}
    monkeypatch.setattr(client_mod, "_http_post", lambda req, timeout: calls.__setitem__("n", calls["n"] + 1) or _sse_ok("x"))
    out_path = tmp_path / "out.json"
    rc = cnc.main(["--golden", str(golden_path), "--out", str(out_path), "--dry-run"])
    assert rc == 0
    assert calls["n"] == 0
    assert not out_path.exists()


def test_full_capture_succeeds_for_all_rows(tmp_path, fake_env, monkeypatch):
    golden_path = _make_golden(tmp_path, n_rows=3)
    monkeypatch.setattr(client_mod, "_http_post", lambda req, timeout: _sse_ok("a real answer"))
    out_path = tmp_path / "out.json"
    rc = cnc.main(["--golden", str(golden_path), "--out", str(out_path)])
    assert rc == 0
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert len(saved["rows"]) == 3
    assert all(r["response"] == "a real answer" for r in saved["rows"])
    assert all(r["capture_status"] == "captured_preview_only" for r in saved["rows"])
    assert saved["capture_meta"]["n_captured_this_run"] == 3


def test_original_golden_dataset_is_never_modified(tmp_path, fake_env, monkeypatch):
    golden_path = _make_golden(tmp_path, n_rows=2)
    before = golden_path.read_bytes()
    monkeypatch.setattr(client_mod, "_http_post", lambda req, timeout: _sse_ok("answer"))
    out_path = tmp_path / "out.json"
    cnc.main(["--golden", str(golden_path), "--out", str(out_path)])
    assert golden_path.read_bytes() == before
    assert out_path != golden_path


@pytest.mark.skipif(not REAL_GOLDEN_PATH.exists(),
                     reason=f"no real golden dataset at {REAL_GOLDEN_PATH} yet - see README.md")
def test_real_golden_dataset_is_never_written_by_a_dry_run(tmp_path, monkeypatch, fake_env):
    """Sanity check against YOUR real dataset: a dry run against it must not write
    anything, anywhere, including to its own path."""
    before = REAL_GOLDEN_PATH.read_bytes()
    out_path = tmp_path / "out.json"
    rc = cnc.main(["--golden", str(REAL_GOLDEN_PATH), "--out", str(out_path), "--dry-run"])
    assert rc == 0
    assert REAL_GOLDEN_PATH.read_bytes() == before
    assert not out_path.exists()


def test_resume_skips_already_captured_rows_on_second_run(tmp_path, fake_env, monkeypatch):
    golden_path = _make_golden(tmp_path, n_rows=3)
    out_path = tmp_path / "out.json"

    calls = {"n": 0}

    def first_pass(req, timeout):
        calls["n"] += 1
        return _sse_ok("first pass answer")

    monkeypatch.setattr(client_mod, "_http_post", first_pass)
    rc1 = cnc.main(["--golden", str(golden_path), "--out", str(out_path)])
    assert rc1 == 0
    assert calls["n"] == 3

    def second_pass(req, timeout):
        calls["n"] += 1
        return _sse_ok("second pass answer - should not appear")

    monkeypatch.setattr(client_mod, "_http_post", second_pass)
    rc2 = cnc.main(["--golden", str(golden_path), "--out", str(out_path)])
    assert rc2 == 0
    # no new network calls - every row was already successfully captured
    assert calls["n"] == 3
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert all(r["response"] == "first pass answer" for r in saved["rows"])


def test_force_recaptures_even_already_successful_rows(tmp_path, fake_env, monkeypatch):
    golden_path = _make_golden(tmp_path, n_rows=2)
    out_path = tmp_path / "out.json"
    monkeypatch.setattr(client_mod, "_http_post", lambda req, timeout: _sse_ok("v1"))
    cnc.main(["--golden", str(golden_path), "--out", str(out_path)])

    monkeypatch.setattr(client_mod, "_http_post", lambda req, timeout: _sse_ok("v2"))
    cnc.main(["--golden", str(golden_path), "--out", str(out_path), "--force"])

    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert all(r["response"] == "v2" for r in saved["rows"])


def test_force_with_ids_only_recaptures_the_targeted_row(tmp_path, fake_env, monkeypatch):
    golden_path = _make_golden(tmp_path, n_rows=3)
    out_path = tmp_path / "out.json"
    monkeypatch.setattr(client_mod, "_http_post", lambda req, timeout: _sse_ok("v1"))
    cnc.main(["--golden", str(golden_path), "--out", str(out_path)])

    monkeypatch.setattr(client_mod, "_http_post", lambda req, timeout: _sse_ok("v2-row1-only"))
    cnc.main(["--golden", str(golden_path), "--out", str(out_path), "--force", "--ids", "ROW-1"])

    saved = json.loads(out_path.read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in saved["rows"]}
    assert by_id["ROW-0"]["response"] == "v1"          # untouched
    assert by_id["ROW-1"]["response"] == "v2-row1-only"  # re-captured
    assert by_id["ROW-2"]["response"] == "v1"          # untouched


def test_failed_row_is_recorded_without_touching_ground_truth(tmp_path, fake_env, monkeypatch):
    golden_path = _make_golden(tmp_path, n_rows=1)
    out_path = tmp_path / "out.json"

    def failing_post(req, timeout):
        return "event: status\ndata: {\"message\": \"stuck\"}\n\n"  # no complete event

    monkeypatch.setattr(client_mod, "_http_post", failing_post)
    rc = cnc.main(["--golden", str(golden_path), "--out", str(out_path)])
    assert rc == 1  # non-zero because something failed

    saved = json.loads(out_path.read_text(encoding="utf-8"))
    row = saved["rows"][0]
    assert row["capture_status"] == "failed"
    assert "complete" in row["capture_error"]
    assert row["response"] == ""
    assert row["reference"] == "ref 0"  # ground truth untouched

    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    assert golden["rows"][0]["reference"] == "ref 0"


def test_missing_base_url_exits_cleanly_without_raising(tmp_path, monkeypatch):
    monkeypatch.delenv("PQA_CHATBOT_BASE_URL", raising=False)
    golden_path = _make_golden(tmp_path, n_rows=1)
    out_path = tmp_path / "out.json"
    rc = cnc.main(["--golden", str(golden_path), "--out", str(out_path)])
    assert rc == 2
    assert not out_path.exists()


def test_missing_golden_dataset_exits_cleanly(tmp_path, fake_env):
    out_path = tmp_path / "out.json"
    rc = cnc.main(["--golden", str(tmp_path / "does-not-exist.json"), "--out", str(out_path)])
    assert rc == 2


# --- the bare question is sent verbatim, matching the real MeitY app's own contract ---

def test_request_body_carries_the_bare_user_input_with_no_wrapper(tmp_path, fake_env, monkeypatch):
    """Confirmed live via genuine browser requests to the MeitY 'AI Assistant' chat
    (Playwright network capture, 2026-11-20): this project's chatbot has no wrapper
    prompt - the outgoing `query` field must be exactly row["user_input"], verbatim."""
    golden_path = _make_golden(tmp_path, n_rows=1)
    sent_bodies = []

    def fake_post(req, timeout):
        sent_bodies.append(json.loads(req.data.decode("utf-8")))
        return _sse_ok("answer")

    monkeypatch.setattr(client_mod, "_http_post", fake_post)
    out_path = tmp_path / "out.json"
    rc = cnc.main(["--golden", str(golden_path), "--out", str(out_path)])
    assert rc == 0
    assert len(sent_bodies) == 1
    assert sent_bodies[0]["query"] == "question 0"


def test_request_body_matches_the_real_apps_confirmed_contract(tmp_path, fake_env, monkeypatch):
    """Confirmed live via genuine browser requests to the MeitY 'AI Assistant' chat
    (Playwright network capture, 2026-11-20) - see chatbot_endpoint_config.json's
    "_confirmed" note. capture_new_chatbot.py must send the same envelope: query, a
    literal type="generate", messages, and a user_id/session_id pair (no
    draft_context/sub_question/part_label - that was a different project's contract)."""
    golden_path = _make_golden(tmp_path, n_rows=1)
    sent_bodies = []

    def fake_post(req, timeout):
        sent_bodies.append(json.loads(req.data.decode("utf-8")))
        return _sse_ok("answer")

    monkeypatch.setattr(client_mod, "_http_post", fake_post)
    out_path = tmp_path / "out.json"
    rc = cnc.main(["--golden", str(golden_path), "--out", str(out_path)])
    assert rc == 0
    body = sent_bodies[0]
    assert body["type"] == "generate"
    assert body["messages"] == [{"role": "user", "content": body["query"]}]
    assert "draft_context" not in body
    assert "sub_question" not in body
    assert "part_label" not in body
    uuid.UUID(body["user_id"])  # a real UUID, not asserting a specific value
    uuid.UUID(body["session_id"])


def test_captured_row_user_input_field_itself_is_never_wrapped(tmp_path, fake_env, monkeypatch):
    """The prompt wrapping must only affect what's POSTed - the saved row's own
    user_input ground-truth field must stay exactly the bare question."""
    golden_path = _make_golden(tmp_path, n_rows=1)
    monkeypatch.setattr(client_mod, "_http_post", lambda req, timeout: _sse_ok("answer"))
    out_path = tmp_path / "out.json"
    cnc.main(["--golden", str(golden_path), "--out", str(out_path)])
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["rows"][0]["user_input"] == "question 0"


# --- follow-up pairs (provenance.follow_up_of) - golden_dataset_README.md section 3.2 --

def _make_golden_with_followup(tmp_path: Path) -> Path:
    """Row 1 (F1) is a normal standalone question; row 2 (F2) follows up on it, matching
    the short-2part dataset's -F1/-F2 convention."""
    rows = [
        {
            "id": "GD-F1", "user_input": "F1 question", "reference": "ref",
            "reference_contexts": [], "must_include_facts": [], "expected_sources": [],
            "expected_behavior": "answer_full", "ragas_eligible": True,
            "stratum": "s", "category": "c", "difficulty": "easy",
            "question_origin": "paraphrased_past", "house": "ls",
            "response": "", "retrieved_contexts": [], "cited_sources": [],
            "cited_sources_raw": [], "provenance": {},
        },
        {
            "id": "GD-F2", "user_input": "F2 follow-up question", "reference": "ref",
            "reference_contexts": [], "must_include_facts": [], "expected_sources": [],
            "expected_behavior": "answer_full", "ragas_eligible": True,
            "stratum": "s", "category": "c", "difficulty": "easy",
            "question_origin": "paraphrased_past", "house": "ls",
            "response": "", "retrieved_contexts": [], "cited_sources": [],
            "cited_sources_raw": [], "provenance": {"follow_up_of": "GD-F1"},
        },
    ]
    data = {"note": "synthetic follow-up test dataset", "n_rows": 2, "rows": rows}
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_follow_up_row_sends_the_prior_turns_prompt_and_answer_as_history(
        tmp_path, fake_env, monkeypatch):
    golden_path = _make_golden_with_followup(tmp_path)
    sent_bodies = []
    answers = iter(["the F1 answer", "the F2 answer"])

    def fake_post(req, timeout):
        sent_bodies.append(json.loads(req.data.decode("utf-8")))
        return _sse_ok(next(answers))

    monkeypatch.setattr(client_mod, "_http_post", fake_post)
    out_path = tmp_path / "out.json"
    rc = cnc.main(["--golden", str(golden_path), "--out", str(out_path)])
    assert rc == 0

    f1_body, f2_body = sent_bodies
    # F1 is a normal standalone call - no history.
    assert f1_body["messages"] == [{"role": "user", "content": f1_body["query"]}]
    # F2 carries F1's actual prompt + F1's actual captured answer ahead of its own turn.
    assert f2_body["messages"][0] == {"role": "user", "content": f1_body["query"]}
    assert f2_body["messages"][1] == {"role": "assistant", "content": "the F1 answer"}
    assert f2_body["messages"][2] == {"role": "user", "content": f2_body["query"]}

    saved = json.loads(out_path.read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in saved["rows"]}
    assert by_id["GD-F1"]["capture_status"] == "captured_preview_only"
    assert by_id["GD-F2"]["capture_status"] == "captured_preview_only"  # continuity, no suffix


def test_follow_up_row_sent_standalone_and_flagged_when_prior_turn_unavailable(
        tmp_path, fake_env, monkeypatch):
    """If F1 hasn't been successfully captured (e.g. --ids scoped to just F2, or F1
    failed), F2 must still get a valid standalone request - never crash - but its
    capture_status is flagged so reporting can exclude it, per README section 3.2."""
    golden_path = _make_golden_with_followup(tmp_path)
    sent_bodies = []

    def fake_post(req, timeout):
        sent_bodies.append(json.loads(req.data.decode("utf-8")))
        return _sse_ok("the F2 answer")

    monkeypatch.setattr(client_mod, "_http_post", fake_post)
    out_path = tmp_path / "out.json"
    # --ids GD-F2 only: F1 is never called, so it has no captured response to build on.
    rc = cnc.main(["--golden", str(golden_path), "--out", str(out_path), "--ids", "GD-F2"])
    assert rc == 0

    assert len(sent_bodies) == 1
    f2_body = sent_bodies[0]
    assert f2_body["messages"] == [{"role": "user", "content": f2_body["query"]}]  # standalone

    saved = json.loads(out_path.read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in saved["rows"]}
    assert by_id["GD-F2"]["capture_status"] == "captured_preview_only_no_continuity"
    assert by_id["GD-F1"]["response"] == ""  # untouched - never selected by --ids
