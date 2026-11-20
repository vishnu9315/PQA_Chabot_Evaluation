"""HTTP client for the new PQA chatbot's streaming generate endpoint.

Uses only the Python standard library (urllib) so this capture pipeline adds zero new
runtime dependencies beyond what PQA_ragas already pins in requirements.lock.txt.

No network call happens at import time. The only functions that touch the network are
`call_chatbot` (and the `_http_post` it calls) - everything else here is pure
configuration/parsing logic, which is what tests/test_capture_client.py exercises by
monkeypatching `_http_post`.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .session import SessionFileError, load_cookie_header
from .sse import find_event, parse_sse

logger = logging.getLogger("capture_new_chatbot")


class CaptureConfigError(RuntimeError):
    """Required configuration (an environment variable) is missing or invalid.

    Raised before any network call - the caller should print the message and exit(2),
    not retry.
    """


class CaptureRequestError(RuntimeError):
    """A single row's capture ultimately failed: transport error after retries, an HTTP
    client/server error, or a response that could not be turned into an answer (missing
    'complete' event, malformed JSON, empty answer). The caller records this against the
    row's capture_status/capture_error fields and moves on - it must never raise past
    the per-row loop in capture_new_chatbot.py.

    `status_code` is the HTTP status that caused this (when known - e.g. an HTTPError's
    code), for the optional capture log; it is None for transport-level failures that
    never got an HTTP response at all. Never included in the message text itself.
    """

    def __init__(self, message: str, *, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class CaptureAuthError(CaptureRequestError):
    """The session was rejected (HTTP 401/403). Unlike other CaptureRequestErrors, this
    is NOT a per-row problem - the same expired/invalid cookie will reject every
    remaining row identically, so capture_new_chatbot.py aborts the whole run on this
    error rather than ploughing through 44 more guaranteed failures. Never retried.
    """


_REDACTED = "***redacted***"


@dataclass(frozen=True)
class ChatbotClientConfig:
    base_url: str
    endpoint_path: str
    timeout_seconds: float
    api_key: Optional[str] = None
    cookie_header: Optional[str] = None
    extra_headers: dict = field(default_factory=dict)
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0

    def __repr__(self) -> str:
        """Never show secret values, even if something accidentally logs/prints this
        config directly (the dataclass-generated __repr__ would otherwise include
        api_key, cookie_header and extra_headers verbatim)."""
        redacted_api_key = _REDACTED if self.api_key else None
        redacted_cookie = _REDACTED if self.cookie_header else None
        redacted_extra = {k: _REDACTED for k in self.extra_headers}
        return (
            f"ChatbotClientConfig(base_url={self.base_url!r}, "
            f"endpoint_path={self.endpoint_path!r}, "
            f"timeout_seconds={self.timeout_seconds!r}, "
            f"api_key={redacted_api_key!r}, "
            f"cookie_header={redacted_cookie!r}, "
            f"extra_headers={redacted_extra!r}, "
            f"max_retries={self.max_retries!r}, "
            f"retry_backoff_seconds={self.retry_backoff_seconds!r})"
        )

    @property
    def url(self) -> str:
        return self.base_url.rstrip("/") + self.endpoint_path

    @classmethod
    def from_env(cls, endpoint_config: dict, *, env: Optional[dict] = None) -> "ChatbotClientConfig":
        """Build config from environment variables (or `env`, for tests).

        Raises CaptureConfigError with a clear setup message if PQA_CHATBOT_BASE_URL is
        unset - this is the "display a clear setup message and exit safely" requirement.
        """
        src = env if env is not None else os.environ
        base_url = str(src.get("PQA_CHATBOT_BASE_URL", "")).strip()
        if not base_url:
            raise CaptureConfigError(
                "PQA_CHATBOT_BASE_URL is not set.\n\n"
                "Set it in this project's .env, e.g.:\n"
                "  PQA_CHATBOT_BASE_URL=https://pqa.myscheme.in/meityy/api/assistant\n\n"
                "See chatbot_endpoint_config.json and README.md's capture section for the "
                "full request/response contract this script expects."
            )
        timeout_raw = str(src.get("PQA_CHATBOT_TIMEOUT", "120")).strip() or "120"
        try:
            timeout = float(timeout_raw)
        except ValueError as exc:
            raise CaptureConfigError(
                f"PQA_CHATBOT_TIMEOUT must be a number of seconds, got {timeout_raw!r}") from exc

        api_key = str(src.get("PQA_CHATBOT_API_KEY", "")).strip() or None

        # Session-cookie auth: PQA_CHATBOT_COOKIE_HEADER (explicit raw header) wins over
        # PQA_CHATBOT_SESSION_FILE (derived from a Playwright storageState file) if both
        # are set. Neither is required - with neither set, no Cookie header is sent.
        cookie_header = str(src.get("PQA_CHATBOT_COOKIE_HEADER", "")).strip() or None
        if cookie_header is None:
            session_file_raw = str(src.get("PQA_CHATBOT_SESSION_FILE", "")).strip()
            if session_file_raw:
                try:
                    cookie_header = load_cookie_header(Path(session_file_raw), base_url)
                except SessionFileError as exc:
                    raise CaptureConfigError(
                        f"Could not build a session cookie from "
                        f"PQA_CHATBOT_SESSION_FILE: {exc}\n\n"
                        "See README.md's 'Supplying a session cookie' section."
                    ) from exc

        extra_headers: dict = {}
        raw_extra = str(src.get("PQA_CHATBOT_EXTRA_HEADERS_JSON", "")).strip()
        if raw_extra:
            try:
                parsed = json.loads(raw_extra)
            except json.JSONDecodeError as exc:
                raise CaptureConfigError(
                    f"PQA_CHATBOT_EXTRA_HEADERS_JSON is not valid JSON: {exc}") from exc
            if not isinstance(parsed, dict):
                raise CaptureConfigError(
                    "PQA_CHATBOT_EXTRA_HEADERS_JSON must be a JSON object, e.g. "
                    '\'{"X-Trace": "..."}\'')
            extra_headers = parsed

        return cls(
            base_url=base_url,
            endpoint_path=endpoint_config["endpoint_path"],
            timeout_seconds=timeout,
            api_key=api_key,
            cookie_header=cookie_header,
            extra_headers=extra_headers,
        )


@dataclass(frozen=True)
class CaptureResult:
    answer: str
    sources_raw: list       # raw source objects exactly as returned by the API (for audit)
    cited_sources: list     # list[str] of source identifiers - matches the dataset schema
    retrieved_contexts: list  # list[str]; empty unless the API genuinely exposes full chunk text
    had_full_context: bool
    http_status: int = 200  # _http_post only returns on a 2xx response; see its docstring
    raw_payload: Optional[dict] = None  # the literal 'complete' event JSON body, for the
    # optional capture log (capture/capture_log.py) - never written to the dataset itself


def _build_request(
    cfg: ChatbotClientConfig,
    query: str,
    query_field: str,
    *,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    history: Optional[list] = None,
) -> urllib.request.Request:
    """Builds the POST body.

    Confirmed 2026-11-20 by inspecting genuine browser requests to the MeitY "AI
    Assistant" chat (via Playwright network capture, no secrets involved): the body is
    `query` + `type: "generate"` + `messages` + `user_id` + `session_id`. No
    `draft_context`/`sub_question`/`part_label` here - that was the PREVIOUS project's
    different, per-sub-question "Draft with AI" contract, not this one's general-purpose
    chat contract. `user_id`/`session_id` are optional so existing callers that only have
    a query keep working; when provided, they're sent as-is (no validation of their shape
    here).

    `history`, if given, is a list of prior `{"role": ..., "content": ...}` turns
    (typically one prior user turn + its assistant answer) prepended to `messages` ahead
    of the current turn - for a multi-turn conversation (e.g. a follow-up question that
    depends on an earlier one in the same chat). CONFIRMED 2026-11-20 live: sending the
    same `session_id` across two calls did NOT by itself carry context - the real frontend
    replays the full prior turn into `messages` every time regardless, and that replay
    (not `session_id`) is what the backend actually uses for continuity. So this growing-
    `messages` mechanism is the right one here, independently confirmed, not a guess.
    """
    body_dict: dict = {
        query_field: query,
        "type": "generate",
        "messages": [*(history or []), {"role": "user", "content": query}],
    }
    if user_id is not None:
        body_dict["user_id"] = user_id
    if session_id is not None:
        body_dict["session_id"] = session_id
    body = json.dumps(body_dict).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    if cfg.cookie_header:
        headers["Cookie"] = cfg.cookie_header
    headers.update(cfg.extra_headers)
    return urllib.request.Request(cfg.url, data=body, headers=headers, method="POST")


def _http_post(req: urllib.request.Request, timeout: float) -> str:
    """The only function in this module that actually opens a socket. Isolated so tests
    can monkeypatch it instead of running a real HTTP stack."""
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (internal tool, trusted config)
        return resp.read().decode("utf-8", errors="replace")


def call_chatbot(
    cfg: ChatbotClientConfig,
    query: str,
    endpoint_config: dict,
    *,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    history: Optional[list] = None,
) -> CaptureResult:
    """POST `query` to the chatbot's streaming endpoint and extract the terminal event.

    `user_id`/`session_id` and `history` (prior conversation turns, for a follow-up
    question) are optional extras matching the real app's request shape - see
    _build_request's docstring. Omitting them still produces a valid request.

    Retries transient failures (timeouts, connection errors, 5xx) with exponential
    backoff up to cfg.max_retries attempts. 4xx errors and malformed-response errors are
    NOT retried - they are raised immediately, since retrying a bad request or a parse
    failure cannot change the outcome. HTTP 401/403 specifically raise CaptureAuthError
    (a CaptureRequestError subclass) rather than the generic client-error message, and
    without echoing the response body - see CaptureAuthError's docstring for why the
    caller treats this as a whole-run abort signal, not a per-row failure.
    """
    query_field = endpoint_config["request"]["query_field"]
    last_exc: Optional[Exception] = None

    for attempt in range(1, cfg.max_retries + 1):
        try:
            req = _build_request(
                cfg, query, query_field, user_id=user_id, session_id=session_id,
                history=history)
            raw = _http_post(req, cfg.timeout_seconds)
            return _parse_response(raw, endpoint_config)
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise CaptureAuthError(
                    f"HTTP {exc.code} - session rejected by {cfg.url}. The configured "
                    f"session cookie is missing, invalid or expired (response body "
                    f"omitted deliberately). Re-run the PQA OTP auth capture and "
                    f"refresh PQA_CHATBOT_SESSION_FILE / PQA_CHATBOT_COOKIE_HEADER - "
                    f"see README.md's 'Supplying a session cookie' section.",
                    status_code=exc.code,
                ) from exc
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:  # pragma: no cover - best-effort error body read
                pass
            if exc.code < 500:
                raise CaptureRequestError(
                    f"HTTP {exc.code} (client error) from {cfg.url}: {body[:500]}",
                    status_code=exc.code) from exc
            last_exc = CaptureRequestError(
                f"HTTP {exc.code} (server error): {body[:500]}", status_code=exc.code)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_exc = CaptureRequestError(f"transport error calling {cfg.url}: {exc}")
        except CaptureRequestError:
            raise  # malformed-response errors from _parse_response are never retried

        if attempt < cfg.max_retries:
            sleep_s = cfg.retry_backoff_seconds * (2 ** (attempt - 1))
            logger.warning("attempt %d/%d failed (%s) - retrying in %.1fs",
                           attempt, cfg.max_retries, last_exc, sleep_s)
            time.sleep(sleep_s)

    assert last_exc is not None
    raise last_exc


def _parse_response(raw: str, endpoint_config: dict) -> CaptureResult:
    events = parse_sse(raw)
    if not events:
        raise CaptureRequestError(
            "response body contained no parseable SSE events", status_code=200)

    terminal = endpoint_config["response"]["terminal_event"]
    complete_event = find_event(events, terminal)
    if complete_event is None:
        seen_types = sorted({e.event for e in events})
        raise CaptureRequestError(
            f"no '{terminal}' event found in the stream (saw event types: {seen_types})",
            status_code=200)

    try:
        payload = complete_event.json()
    except json.JSONDecodeError as exc:
        raise CaptureRequestError(
            f"'{terminal}' event data was not valid JSON: {exc}; "
            f"raw={complete_event.data[:500]!r}", status_code=200) from exc

    if not isinstance(payload, dict):
        raise CaptureRequestError(
            f"'{terminal}' event data was valid JSON but not an object: {type(payload).__name__}",
            status_code=200)

    answer_field = endpoint_config["response"]["answer_field"]
    sources_field = endpoint_config["response"]["sources_field"]

    answer = payload.get(answer_field)
    if not isinstance(answer, str) or not answer.strip():
        raise CaptureRequestError(
            f"'{terminal}' event had an empty or missing '{answer_field}' field",
            status_code=200)

    sources_raw = payload.get(sources_field)
    if sources_raw is None:
        sources_raw = []
    if not isinstance(sources_raw, list):
        raise CaptureRequestError(
            f"'{sources_field}' was present but not a list: {type(sources_raw).__name__}",
            status_code=200)

    source_cfg = endpoint_config["source_object_fields"]
    id_field = source_cfg["identifier_field"]
    full_text_field = source_cfg.get("full_text_field")  # None today; forward-compatible

    cited_sources: list = []
    retrieved_contexts: list = []
    for src in sources_raw:
        if not isinstance(src, dict):
            continue
        identifier = src.get(id_field)
        if isinstance(identifier, str) and identifier:
            cited_sources.append(identifier)
        if full_text_field:
            chunk = src.get(full_text_field)
            if isinstance(chunk, str) and chunk.strip():
                retrieved_contexts.append(chunk)

    had_full_context = bool(full_text_field) and len(retrieved_contexts) > 0
    if sources_raw and not had_full_context:
        logger.warning(
            "only short previews available for %d source(s) - retrieved_contexts left "
            "EMPTY (not populated with previews); Faithfulness/Context Precision/"
            "Context Recall will be NA for this row",
            len(sources_raw))

    return CaptureResult(
        answer=answer,
        sources_raw=sources_raw,
        cited_sources=cited_sources,
        retrieved_contexts=retrieved_contexts,
        had_full_context=had_full_context,
        http_status=200,
        raw_payload=payload,
    )
