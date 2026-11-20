"""Pure, network-free parsing of Server-Sent Events (SSE) streams.

Implements enough of the SSE wire format for this project's needs: `event:` and `data:`
fields, multi-line data (joined with '\\n' per spec), blank-line-terminated events,
comment lines (a leading ':') ignored, and the spec's default event type 'message' when
no `event:` line appears in a block. No network I/O happens here - this module only
turns an already-fetched response body (str) into a list of SSEEvent.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class SSEEvent:
    event: str
    data: str  # raw, un-parsed data text (multi-line data: fields joined with '\n')

    def json(self) -> Any:
        """Parse `data` as JSON. Raises json.JSONDecodeError on malformed payloads -
        callers decide how a malformed event should be handled (see capture/client.py)."""
        return json.loads(self.data)


def parse_sse(text: str) -> list[SSEEvent]:
    """Parse a complete SSE stream body into a list of SSEEvent, in stream order.

    - Events are separated by a blank line.
    - Lines starting with ':' are comments and are ignored.
    - An `event:` line sets that event block's type; if none is present, the type
      defaults to 'message' (the SSE spec's default).
    - Multiple `data:` lines within one event block are joined with '\\n'.
    - A trailing event with no blank line at end-of-stream is still flushed.
    - A block with no `data:` lines at all produces no event (matches the spec: an
      event with an empty data buffer is not dispatched).
    """
    events: list[SSEEvent] = []
    event_type = "message"
    data_lines: list[str] = []

    def _flush() -> None:
        nonlocal event_type, data_lines
        if data_lines:
            events.append(SSEEvent(event=event_type, data="\n".join(data_lines)))
        event_type = "message"
        data_lines = []

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    for line in normalized.split("\n"):
        if line == "":
            _flush()
            continue
        if line.startswith(":"):
            continue
        if ":" in line:
            field, _, value = line.partition(":")
            if value.startswith(" "):
                value = value[1:]
        else:
            field, value = line, ""
        if field == "event":
            event_type = value
        elif field == "data":
            data_lines.append(value)
        # id: and retry: are valid SSE fields but unused by this project - ignored.
    _flush()
    return events


def find_event(events: list[SSEEvent], event_type: str) -> Optional[SSEEvent]:
    """The LAST event of the given type, or None if absent.

    Defensively returns the last match (not the first) in case a provider ever emits
    more than one event of the same type in a single stream, so a late correction wins
    rather than a stale earlier one.
    """
    matches = [e for e in events if e.event == event_type]
    return matches[-1] if matches else None
