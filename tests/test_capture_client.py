"""Tests for capture/client.py. No real network call is ever made here - `_http_post`
(the only function that opens a socket) is monkeypatched in every test.
"""
import json
import urllib.error

import pytest

from capture import client as client_mod
from capture.client import (
    CaptureAuthError,
    CaptureConfigError,
    CaptureRequestError,
    ChatbotClientConfig,
    call_chatbot,
)

SECRET_COOKIE_VALUE = "super-secret-session-token-do-not-leak-67890"

ENDPOINT_CONFIG = {
    "endpoint_path": "/chat/stream",
    "request": {"query_field": "query"},
    "response": {"terminal_event": "complete", "answer_field": "answer", "sources_field": "sources"},
    "source_object_fields": {"identifier_field": "filename", "full_text_field": None},
}


def _sse(answer="The answer.", sources=None):
    sources = sources if sources is not None else []
    payload = json.dumps({"answer": answer, "sources": sources})
    return f"event: status\ndata: {{\"message\": \"go\"}}\n\nevent: complete\ndata: {payload}\n\n"


def _cfg(**overrides):
    base = dict(base_url="https://example.invalid/meityy/api/assistant",
                endpoint_path="/chat/stream", timeout_seconds=5.0,
                max_retries=3, retry_backoff_seconds=0.0)
    base.update(overrides)
    return ChatbotClientConfig(**base)


# --- ChatbotClientConfig.from_env -------------------------------------------------

def test_from_env_raises_capture_config_error_when_base_url_missing():
    with pytest.raises(CaptureConfigError, match="PQA_CHATBOT_BASE_URL"):
        ChatbotClientConfig.from_env(ENDPOINT_CONFIG, env={})


def test_from_env_builds_config_from_env_dict():
    env = {"PQA_CHATBOT_BASE_URL": "https://example.invalid/api", "PQA_CHATBOT_TIMEOUT": "30"}
    cfg = ChatbotClientConfig.from_env(ENDPOINT_CONFIG, env=env)
    assert cfg.base_url == "https://example.invalid/api"
    assert cfg.timeout_seconds == 30.0
    assert cfg.url == "https://example.invalid/api/chat/stream"
    assert cfg.api_key is None


def test_from_env_rejects_non_numeric_timeout():
    env = {"PQA_CHATBOT_BASE_URL": "https://example.invalid/api", "PQA_CHATBOT_TIMEOUT": "soon"}
    with pytest.raises(CaptureConfigError, match="PQA_CHATBOT_TIMEOUT"):
        ChatbotClientConfig.from_env(ENDPOINT_CONFIG, env=env)


def test_from_env_rejects_malformed_extra_headers_json():
    env = {"PQA_CHATBOT_BASE_URL": "https://example.invalid/api",
           "PQA_CHATBOT_EXTRA_HEADERS_JSON": "{not json"}
    with pytest.raises(CaptureConfigError, match="EXTRA_HEADERS_JSON"):
        ChatbotClientConfig.from_env(ENDPOINT_CONFIG, env=env)


def test_from_env_parses_extra_headers_and_api_key():
    env = {"PQA_CHATBOT_BASE_URL": "https://example.invalid/api",
           "PQA_CHATBOT_API_KEY": "secret123",
           "PQA_CHATBOT_EXTRA_HEADERS_JSON": '{"X-Trace": "abc"}'}
    cfg = ChatbotClientConfig.from_env(ENDPOINT_CONFIG, env=env)
    assert cfg.api_key == "secret123"
    assert cfg.extra_headers == {"X-Trace": "abc"}


# --- call_chatbot: success paths ---------------------------------------------------

def test_successful_capture_extracts_answer_and_sources(monkeypatch):
    sources = [{"filename": "rs_unstarred_2282.md", "text_preview": "snippet"}]
    monkeypatch.setattr(client_mod, "_http_post", lambda req, timeout: _sse("The answer.", sources))
    result = call_chatbot(_cfg(), "some question", ENDPOINT_CONFIG)
    assert result.answer == "The answer."
    assert result.cited_sources == ["rs_unstarred_2282.md"]
    assert result.sources_raw == sources
    assert result.retrieved_contexts == []
    assert result.had_full_context is False


def test_missing_sources_defaults_to_empty_list_not_an_error(monkeypatch):
    payload = json.dumps({"answer": "ok"})  # no 'sources' key at all
    stream = f"event: complete\ndata: {payload}\n\n"
    monkeypatch.setattr(client_mod, "_http_post", lambda req, timeout: stream)
    result = call_chatbot(_cfg(), "q", ENDPOINT_CONFIG)
    assert result.answer == "ok"
    assert result.sources_raw == []
    assert result.cited_sources == []


def test_full_text_field_populates_retrieved_contexts_when_available(monkeypatch):
    cfg_with_full_text = dict(ENDPOINT_CONFIG)
    cfg_with_full_text["source_object_fields"] = {
        "identifier_field": "filename", "full_text_field": "full_chunk_text",
    }
    sources = [{"filename": "rs_unstarred_2282.md", "full_chunk_text": "the entire chunk..."}]
    monkeypatch.setattr(client_mod, "_http_post", lambda req, timeout: _sse("ans", sources))
    result = call_chatbot(_cfg(), "q", cfg_with_full_text)
    assert result.retrieved_contexts == ["the entire chunk..."]
    assert result.had_full_context is True


def test_previews_only_never_populate_retrieved_contexts(monkeypatch, caplog):
    sources = [{"filename": "a.md", "text_preview": "short snippet"}]
    monkeypatch.setattr(client_mod, "_http_post", lambda req, timeout: _sse("ans", sources))
    import logging
    with caplog.at_level(logging.WARNING, logger="capture_new_chatbot"):
        result = call_chatbot(_cfg(), "q", ENDPOINT_CONFIG)
    assert result.retrieved_contexts == []
    assert any("retrieved_contexts left EMPTY" in r.message for r in caplog.records)


# --- call_chatbot: malformed-response failures (never retried) --------------------

def test_missing_complete_event_raises_without_retry(monkeypatch):
    calls = {"n": 0}

    def fake_post(req, timeout):
        calls["n"] += 1
        return "event: status\ndata: {\"message\": \"still going\"}\n\n"

    monkeypatch.setattr(client_mod, "_http_post", fake_post)
    with pytest.raises(CaptureRequestError, match="no 'complete' event"):
        call_chatbot(_cfg(), "q", ENDPOINT_CONFIG)
    assert calls["n"] == 1  # not retried


def test_malformed_json_in_complete_event_raises(monkeypatch):
    stream = "event: complete\ndata: {not valid json\n\n"
    monkeypatch.setattr(client_mod, "_http_post", lambda req, timeout: stream)
    with pytest.raises(CaptureRequestError, match="not valid JSON"):
        call_chatbot(_cfg(), "q", ENDPOINT_CONFIG)


def test_empty_answer_raises(monkeypatch):
    monkeypatch.setattr(client_mod, "_http_post", lambda req, timeout: _sse(answer=""))
    with pytest.raises(CaptureRequestError, match="empty or missing"):
        call_chatbot(_cfg(), "q", ENDPOINT_CONFIG)


def test_no_events_at_all_raises(monkeypatch):
    monkeypatch.setattr(client_mod, "_http_post", lambda req, timeout: "")
    with pytest.raises(CaptureRequestError, match="no parseable SSE events"):
        call_chatbot(_cfg(), "q", ENDPOINT_CONFIG)


def test_sources_not_a_list_raises(monkeypatch):
    payload = json.dumps({"answer": "ok", "sources": "not-a-list"})
    stream = f"event: complete\ndata: {payload}\n\n"
    monkeypatch.setattr(client_mod, "_http_post", lambda req, timeout: stream)
    with pytest.raises(CaptureRequestError, match="not a list"):
        call_chatbot(_cfg(), "q", ENDPOINT_CONFIG)


# --- call_chatbot: transport/HTTP errors and retries -------------------------------

def _http_error(code: int) -> urllib.error.HTTPError:
    import io
    return urllib.error.HTTPError(
        url="https://example.invalid", code=code, msg="err", hdrs=None,
        fp=io.BytesIO(b"error body"))


def test_client_error_4xx_does_not_retry(monkeypatch):
    calls = {"n": 0}

    def fake_post(req, timeout):
        calls["n"] += 1
        raise _http_error(400)

    monkeypatch.setattr(client_mod, "_http_post", fake_post)
    with pytest.raises(CaptureRequestError, match="HTTP 400"):
        call_chatbot(_cfg(), "q", ENDPOINT_CONFIG)
    assert calls["n"] == 1


def test_server_error_5xx_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def fake_post(req, timeout):
        calls["n"] += 1
        if calls["n"] < 3:
            raise _http_error(503)
        return _sse("recovered")

    monkeypatch.setattr(client_mod, "_http_post", fake_post)
    result = call_chatbot(_cfg(max_retries=3, retry_backoff_seconds=0.0), "q", ENDPOINT_CONFIG)
    assert result.answer == "recovered"
    assert calls["n"] == 3


def test_server_error_5xx_exhausts_retries_and_raises(monkeypatch):
    calls = {"n": 0}

    def fake_post(req, timeout):
        calls["n"] += 1
        raise _http_error(500)

    monkeypatch.setattr(client_mod, "_http_post", fake_post)
    with pytest.raises(CaptureRequestError, match="HTTP 500"):
        call_chatbot(_cfg(max_retries=3, retry_backoff_seconds=0.0), "q", ENDPOINT_CONFIG)
    assert calls["n"] == 3


def test_timeout_is_retried_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def fake_post(req, timeout):
        calls["n"] += 1
        if calls["n"] < 2:
            raise TimeoutError("timed out")
        return _sse("recovered after timeout")

    monkeypatch.setattr(client_mod, "_http_post", fake_post)
    result = call_chatbot(_cfg(max_retries=3, retry_backoff_seconds=0.0), "q", ENDPOINT_CONFIG)
    assert result.answer == "recovered after timeout"
    assert calls["n"] == 2


def test_connection_error_exhausts_retries_and_raises(monkeypatch):
    def fake_post(req, timeout):
        raise ConnectionError("refused")

    monkeypatch.setattr(client_mod, "_http_post", fake_post)
    with pytest.raises(CaptureRequestError, match="transport error"):
        call_chatbot(_cfg(max_retries=2, retry_backoff_seconds=0.0), "q", ENDPOINT_CONFIG)


# --- request building ---------------------------------------------------------------

def test_api_key_sets_authorization_header():
    cfg = _cfg(api_key="secret123")
    req = client_mod._build_request(cfg, "hello", "query")
    assert req.get_header("Authorization") == "Bearer secret123"
    body = json.loads(req.data.decode("utf-8"))
    assert body["query"] == "hello"


def test_no_api_key_means_no_authorization_header():
    cfg = _cfg(api_key=None)
    req = client_mod._build_request(cfg, "hello", "query")
    assert req.get_header("Authorization") is None


def test_extra_headers_are_merged_in():
    cfg = _cfg(extra_headers={"X-Trace": "abc"})
    req = client_mod._build_request(cfg, "hello", "query")
    assert req.get_header("X-trace") == "abc"  # urllib title-cases header names


def test_cookie_header_is_sent_when_configured():
    cfg = _cfg(cookie_header=f"pqa_session={SECRET_COOKIE_VALUE}")
    req = client_mod._build_request(cfg, "hello", "query")
    assert req.get_header("Cookie") == f"pqa_session={SECRET_COOKIE_VALUE}"


def test_no_cookie_header_when_not_configured():
    cfg = _cfg(cookie_header=None)
    req = client_mod._build_request(cfg, "hello", "query")
    assert req.get_header("Cookie") is None


# --- request body matches the real app's confirmed contract (2026-10-30) -----------

def test_body_always_includes_type_and_messages():
    """Confirmed live via genuine browser requests to the MeitY 'AI Assistant' chat
    (Playwright network capture, 2026-11-20): the real app always sends these two
    fields alongside 'query'. type is always the literal string "generate"."""
    req = client_mod._build_request(_cfg(), "hello", "query")
    body = json.loads(req.data.decode("utf-8"))
    assert body["type"] == "generate"
    assert body["messages"] == [{"role": "user", "content": "hello"}]


def test_body_omits_user_id_and_session_id_when_not_given():
    req = client_mod._build_request(_cfg(), "hello", "query")
    body = json.loads(req.data.decode("utf-8"))
    assert "user_id" not in body
    assert "session_id" not in body


def test_body_includes_user_id_and_session_id_when_given():
    req = client_mod._build_request(
        _cfg(), "hello", "query", user_id="user-123", session_id="session-456")
    body = json.loads(req.data.decode("utf-8"))
    assert body["user_id"] == "user-123"
    assert body["session_id"] == "session-456"


def test_call_chatbot_forwards_user_id_and_session_id_into_the_body(monkeypatch):
    sent_bodies = []

    def fake_post(req, timeout):
        sent_bodies.append(json.loads(req.data.decode("utf-8")))
        return _sse("ans")

    monkeypatch.setattr(client_mod, "_http_post", fake_post)
    call_chatbot(_cfg(), "the question", ENDPOINT_CONFIG,
                 user_id="user-123", session_id="session-456")
    assert sent_bodies[0]["user_id"] == "user-123"
    assert sent_bodies[0]["session_id"] == "session-456"


# --- multi-turn `history` (follow-up questions in the same chat) -------------------

def test_body_has_single_turn_messages_when_no_history_given():
    req = client_mod._build_request(_cfg(), "hello", "query")
    body = json.loads(req.data.decode("utf-8"))
    assert body["messages"] == [{"role": "user", "content": "hello"}]


def test_body_prepends_history_turns_ahead_of_the_current_message():
    history = [
        {"role": "user", "content": "the F1 prompt"},
        {"role": "assistant", "content": "the F1 answer"},
    ]
    req = client_mod._build_request(_cfg(), "the F2 prompt", "query", history=history)
    body = json.loads(req.data.decode("utf-8"))
    assert body["messages"] == [
        {"role": "user", "content": "the F1 prompt"},
        {"role": "assistant", "content": "the F1 answer"},
        {"role": "user", "content": "the F2 prompt"},
    ]
    # `query` itself stays just the current turn - history lives only in `messages`.
    assert body["query"] == "the F2 prompt"


def test_call_chatbot_forwards_history_into_the_body(monkeypatch):
    sent_bodies = []

    def fake_post(req, timeout):
        sent_bodies.append(json.loads(req.data.decode("utf-8")))
        return _sse("ans")

    monkeypatch.setattr(client_mod, "_http_post", fake_post)
    history = [{"role": "user", "content": "F1 q"}, {"role": "assistant", "content": "F1 a"}]
    call_chatbot(_cfg(), "F2 q", ENDPOINT_CONFIG, history=history)
    assert sent_bodies[0]["messages"][:2] == history
    assert sent_bodies[0]["messages"][-1] == {"role": "user", "content": "F2 q"}


# --- session-cookie auth: HTTP 401/403 handling -------------------------------------

def test_401_raises_capture_auth_error_without_retry(monkeypatch):
    calls = {"n": 0}

    def fake_post(req, timeout):
        calls["n"] += 1
        raise _http_error(401)

    monkeypatch.setattr(client_mod, "_http_post", fake_post)
    cfg = _cfg(cookie_header=f"pqa_session={SECRET_COOKIE_VALUE}")
    with pytest.raises(CaptureAuthError, match="HTTP 401"):
        call_chatbot(cfg, "q", ENDPOINT_CONFIG)
    assert calls["n"] == 1  # not retried - a stale cookie won't fix itself


def test_403_raises_capture_auth_error_without_retry(monkeypatch):
    calls = {"n": 0}

    def fake_post(req, timeout):
        calls["n"] += 1
        raise _http_error(403)

    monkeypatch.setattr(client_mod, "_http_post", fake_post)
    with pytest.raises(CaptureAuthError, match="HTTP 403"):
        call_chatbot(_cfg(), "q", ENDPOINT_CONFIG)
    assert calls["n"] == 1


def test_capture_auth_error_is_a_capture_request_error():
    assert issubclass(CaptureAuthError, CaptureRequestError)


def test_401_error_message_never_contains_the_cookie_value(monkeypatch):
    def fake_post(req, timeout):
        raise _http_error(401)

    monkeypatch.setattr(client_mod, "_http_post", fake_post)
    cfg = _cfg(cookie_header=f"pqa_session={SECRET_COOKIE_VALUE}")
    with pytest.raises(CaptureAuthError) as excinfo:
        call_chatbot(cfg, "q", ENDPOINT_CONFIG)
    assert SECRET_COOKIE_VALUE not in str(excinfo.value)


def test_other_4xx_errors_are_not_treated_as_auth_errors(monkeypatch):
    """A plain 400 is a request problem, not a session problem - must not be
    misreported as CaptureAuthError (which the caller treats as a whole-run abort)."""
    def fake_post(req, timeout):
        raise _http_error(400)

    monkeypatch.setattr(client_mod, "_http_post", fake_post)
    with pytest.raises(CaptureRequestError) as excinfo:
        call_chatbot(_cfg(), "q", ENDPOINT_CONFIG)
    assert not isinstance(excinfo.value, CaptureAuthError)


# --- ChatbotClientConfig.from_env: session-cookie precedence and secrecy -----------

def test_from_env_prefers_explicit_cookie_header_over_session_file(tmp_path):
    session_path = tmp_path / "session.json"
    session_path.write_text(json.dumps({"cookies": [
        {"name": "pqa_session", "value": "from-file-not-used", "domain": "example.invalid"},
    ]}), encoding="utf-8")
    env = {
        "PQA_CHATBOT_BASE_URL": "https://example.invalid/api",
        "PQA_CHATBOT_COOKIE_HEADER": f"pqa_session={SECRET_COOKIE_VALUE}",
        "PQA_CHATBOT_SESSION_FILE": str(session_path),
    }
    cfg = ChatbotClientConfig.from_env(ENDPOINT_CONFIG, env=env)
    assert cfg.cookie_header == f"pqa_session={SECRET_COOKIE_VALUE}"


def test_from_env_derives_cookie_header_from_session_file(tmp_path):
    session_path = tmp_path / "session.json"
    session_path.write_text(json.dumps({"cookies": [
        {"name": "pqa_session", "value": SECRET_COOKIE_VALUE, "domain": "example.invalid"},
    ]}), encoding="utf-8")
    env = {
        "PQA_CHATBOT_BASE_URL": "https://example.invalid/api",
        "PQA_CHATBOT_SESSION_FILE": str(session_path),
    }
    cfg = ChatbotClientConfig.from_env(ENDPOINT_CONFIG, env=env)
    assert cfg.cookie_header == f"pqa_session={SECRET_COOKIE_VALUE}"


def test_from_env_raises_capture_config_error_for_bad_session_file(tmp_path):
    env = {
        "PQA_CHATBOT_BASE_URL": "https://example.invalid/api",
        "PQA_CHATBOT_SESSION_FILE": str(tmp_path / "missing.json"),
    }
    with pytest.raises(CaptureConfigError, match="PQA_CHATBOT_SESSION_FILE"):
        ChatbotClientConfig.from_env(ENDPOINT_CONFIG, env=env)


def test_from_env_with_neither_cookie_source_set_has_no_cookie_header():
    env = {"PQA_CHATBOT_BASE_URL": "https://example.invalid/api"}
    cfg = ChatbotClientConfig.from_env(ENDPOINT_CONFIG, env=env)
    assert cfg.cookie_header is None


# --- secrets never leak via repr/str -------------------------------------------------

def test_repr_redacts_api_key_cookie_header_and_extra_headers():
    cfg = _cfg(api_key="api-secret", cookie_header=f"pqa_session={SECRET_COOKIE_VALUE}",
               extra_headers={"X-Session": "also-secret"})
    text = repr(cfg)
    assert "api-secret" not in text
    assert SECRET_COOKIE_VALUE not in text
    assert "also-secret" not in text
    assert "***redacted***" in text


def test_repr_shows_none_when_no_secrets_configured():
    cfg = _cfg(api_key=None, cookie_header=None, extra_headers={})
    text = repr(cfg)
    assert "api_key=None" in text
    assert "cookie_header=None" in text
