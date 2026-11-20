"""Tests for capture/session.py: building a Cookie header from a Playwright storageState
file. No network calls. Every test that uses a real-looking cookie value asserts that
value never appears in any exception message this module raises.
"""
import json

import pytest

from capture.session import SessionFileError, domain_matches, load_cookie_header

SECRET_VALUE = "super-secret-session-token-do-not-leak-12345"


def _write_storage_state(path, cookies):
    path.write_text(json.dumps({"cookies": cookies, "origins": []}), encoding="utf-8")
    return path


# --- domain_matches ------------------------------------------------------------------

@pytest.mark.parametrize("cookie_domain,host,expected", [
    ("pqa.myscheme.in", "pqa.myscheme.in", True),
    ("PQA.MYSCHEME.IN", "pqa.myscheme.in", True),          # case-insensitive
    (".myscheme.in", "pqa.myscheme.in", True),             # leading-dot suffix match
    (".myscheme.in", "myscheme.in", True),                 # leading-dot exact (bare) match
    ("myscheme.in", "pqa.myscheme.in", True),               # bare parent domain
    ("pqa.myscheme.in", "other.myscheme.in", False),
    ("pqa.myscheme.in", "evil-pqa.myscheme.in", False),
    ("example.com", "pqa.myscheme.in", False),
])
def test_domain_matches(cookie_domain, host, expected):
    assert domain_matches(cookie_domain, host) is expected


# --- load_cookie_header: success -----------------------------------------------------

def test_loads_cookie_header_from_real_shaped_storage_state(tmp_path):
    path = _write_storage_state(tmp_path / "session.json", [
        {"name": "pqa_session", "value": SECRET_VALUE, "domain": "pqa.myscheme.in"},
        {"name": "pqa_refresh", "value": "another-secret", "domain": "pqa.myscheme.in"},
        {"name": "accessibilitySettings", "value": "x", "domain": ".myscheme.in"},
    ])
    header = load_cookie_header(path, "https://pqa.myscheme.in/meityy/api/assistant")
    assert f"pqa_session={SECRET_VALUE}" in header
    assert "pqa_refresh=another-secret" in header
    assert "accessibilitySettings=x" in header
    assert header.count(";") == 2  # three cookies joined


def test_cookies_for_other_hosts_are_excluded(tmp_path):
    path = _write_storage_state(tmp_path / "session.json", [
        {"name": "pqa_session", "value": SECRET_VALUE, "domain": "pqa.myscheme.in"},
        {"name": "unrelated_cookie", "value": "zzz", "domain": "someother.example.com"},
    ])
    header = load_cookie_header(path, "https://pqa.myscheme.in/meityy/api/assistant")
    assert "pqa_session" in header
    assert "unrelated_cookie" not in header


# --- load_cookie_header: failures, and that they never leak the secret -------------

def test_missing_file_raises_session_file_error(tmp_path):
    with pytest.raises(SessionFileError, match="not found"):
        load_cookie_header(tmp_path / "does-not-exist.json", "https://pqa.myscheme.in/api")


def test_malformed_json_raises_session_file_error(tmp_path):
    path = tmp_path / "session.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(SessionFileError, match="not valid JSON"):
        load_cookie_header(path, "https://pqa.myscheme.in/api")


def test_missing_cookies_key_raises_session_file_error(tmp_path):
    path = tmp_path / "session.json"
    path.write_text(json.dumps({"origins": []}), encoding="utf-8")
    with pytest.raises(SessionFileError, match="cookies"):
        load_cookie_header(path, "https://pqa.myscheme.in/api")


def test_no_matching_cookies_raises_session_file_error(tmp_path):
    path = _write_storage_state(tmp_path / "session.json", [
        {"name": "pqa_session", "value": SECRET_VALUE, "domain": "some-other-host.example.com"},
    ])
    with pytest.raises(SessionFileError, match="no cookies .* matched host"):
        load_cookie_header(path, "https://pqa.myscheme.in/api")


def test_no_matching_cookies_error_does_not_leak_the_secret_value(tmp_path):
    path = _write_storage_state(tmp_path / "session.json", [
        {"name": "pqa_session", "value": SECRET_VALUE, "domain": "wrong-host.example.com"},
    ])
    with pytest.raises(SessionFileError) as excinfo:
        load_cookie_header(path, "https://pqa.myscheme.in/api")
    assert SECRET_VALUE not in str(excinfo.value)


def test_malformed_json_error_does_not_leak_file_contents(tmp_path):
    path = tmp_path / "session.json"
    path.write_text(f'{{"cookies": [{{"value": "{SECRET_VALUE}"', encoding="utf-8")
    with pytest.raises(SessionFileError) as excinfo:
        load_cookie_header(path, "https://pqa.myscheme.in/api")
    assert SECRET_VALUE not in str(excinfo.value)


def test_cookie_entries_missing_required_keys_are_skipped_not_fatal(tmp_path):
    path = _write_storage_state(tmp_path / "session.json", [
        {"name": "pqa_session", "value": SECRET_VALUE, "domain": "pqa.myscheme.in"},
        {"name": "incomplete_cookie", "domain": "pqa.myscheme.in"},  # no 'value'
        "not even a dict",
    ])
    header = load_cookie_header(path, "https://pqa.myscheme.in/api")
    assert header == f"pqa_session={SECRET_VALUE}"


def test_unresolvable_host_raises_session_file_error(tmp_path):
    path = _write_storage_state(tmp_path / "session.json", [
        {"name": "pqa_session", "value": SECRET_VALUE, "domain": "pqa.myscheme.in"},
    ])
    with pytest.raises(SessionFileError, match="hostname"):
        load_cookie_header(path, "not-a-url-at-all")
