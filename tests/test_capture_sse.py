"""Pure, network-free tests for capture/sse.py."""
import json

import pytest

from capture.sse import SSEEvent, find_event, parse_sse


def test_parses_single_event():
    text = "event: status\ndata: {\"message\": \"Generating...\"}\n\n"
    events = parse_sse(text)
    assert len(events) == 1
    assert events[0].event == "status"
    assert events[0].json() == {"message": "Generating..."}


def test_parses_multiple_events_in_order():
    text = (
        "event: status\ndata: {\"message\": \"a\"}\n\n"
        "event: token\ndata: {\"token\": \"b\"}\n\n"
        "event: complete\ndata: {\"answer\": \"c\", \"sources\": []}\n\n"
    )
    events = parse_sse(text)
    assert [e.event for e in events] == ["status", "token", "complete"]
    assert events[-1].json()["answer"] == "c"


def test_multiline_data_joined_with_newline():
    text = "event: complete\ndata: line one\ndata: line two\n\n"
    events = parse_sse(text)
    assert len(events) == 1
    assert events[0].data == "line one\nline two"


def test_comment_lines_are_ignored():
    text = ": this is a keep-alive comment\nevent: status\ndata: {\"message\": \"ok\"}\n\n"
    events = parse_sse(text)
    assert len(events) == 1
    assert events[0].event == "status"


def test_default_event_type_is_message_when_no_event_field():
    text = "data: {\"a\": 1}\n\n"
    events = parse_sse(text)
    assert len(events) == 1
    assert events[0].event == "message"


def test_trailing_event_with_no_final_blank_line_is_still_flushed():
    text = "event: complete\ndata: {\"answer\": \"x\", \"sources\": []}"
    events = parse_sse(text)
    assert len(events) == 1
    assert events[0].event == "complete"


def test_empty_data_block_produces_no_event():
    text = "event: status\n\ndata: {\"a\": 1}\n\n"
    events = parse_sse(text)
    # the first block has an event: line but no data: line - nothing to dispatch
    assert len(events) == 1
    assert events[0].event == "message"


def test_crlf_line_endings_are_handled():
    text = "event: complete\r\ndata: {\"answer\": \"x\", \"sources\": []}\r\n\r\n"
    events = parse_sse(text)
    assert len(events) == 1
    assert events[0].json()["answer"] == "x"


def test_value_with_colon_in_it_is_preserved():
    text = "event: complete\ndata: {\"answer\": \"ratio is 3:1\", \"sources\": []}\n\n"
    events = parse_sse(text)
    assert events[0].json()["answer"] == "ratio is 3:1"


def test_json_raises_jsondecodeerror_on_malformed_data():
    event = SSEEvent(event="complete", data="{not valid json")
    with pytest.raises(json.JSONDecodeError):
        event.json()


def test_find_event_returns_none_when_absent():
    events = parse_sse("event: status\ndata: {}\n\n")
    assert find_event(events, "complete") is None


def test_find_event_returns_last_match_when_duplicated():
    text = (
        "event: complete\ndata: {\"answer\": \"first\", \"sources\": []}\n\n"
        "event: complete\ndata: {\"answer\": \"second\", \"sources\": []}\n\n"
    )
    events = parse_sse(text)
    last = find_event(events, "complete")
    assert last is not None
    assert last.json()["answer"] == "second"


def test_real_captured_stream_shape_from_live_probe():
    """Mirrors the exact shape observed during manual probing of the real endpoint
    (status x2, one token event, then complete) - see chatbot_endpoint_config.json's
    'known_from_live_probe' block."""
    text = (
        "event: status\ndata: {\"message\": \"Generating...\"}\n\n"
        "event: status\ndata: {\"message\": \"Retrieving sources...\"}\n\n"
        "event: status\ndata: {\"message\": \"Generating answer...\"}\n\n"
        "event: token\ndata: {\"token\": \"The answer text.\"}\n\n"
        "event: complete\n"
        "data: {\"answer\": \"The answer text.\", \"sources\": "
        "[{\"filename\": \"rs_unstarred_2282.md\", \"text_preview\": \"snippet...\"}]}\n\n"
    )
    events = parse_sse(text)
    assert [e.event for e in events] == ["status", "status", "status", "token", "complete"]
    complete = find_event(events, "complete")
    assert complete is not None
    payload = complete.json()
    assert payload["answer"] == "The answer text."
    assert payload["sources"][0]["filename"] == "rs_unstarred_2282.md"
