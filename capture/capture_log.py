"""Optional, separate debugging/traceability log for capture_new_chatbot.py.

This is a sidecar JSON file for a specific capture run - it never touches the unified
golden/captured dataset schema (capture/dataset_io.py). One entry per question attempted,
holding: the request payload sent, the HTTP status, the parsed terminal ('complete') SSE
event, the captured answer, cited source filenames, a timestamp, duration, and
success/failure with an error message when it failed.

Secrets never appear here: no header (Cookie, Authorization, or any extra header) is ever
written to a log entry - only the JSON body `{query_field: query}` that was POSTed, which
contains the row's question text and nothing auth-related.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional


def build_log_entry(
    *,
    row_id: str,
    query: str,
    query_field: str,
    endpoint_url: str,
    http_status: Optional[int],
    complete_event: Optional[dict],
    answer: Optional[str],
    source_filenames: list,
    started_at: float,
    duration_seconds: float,
    success: bool,
    error: Optional[str],
    history_turns: int = 0,
    note: Optional[str] = None,
) -> dict:
    """Build one log entry.

    `complete_event` is the literal JSON payload of the terminal SSE event as parsed by
    capture/client.py (e.g. {"answer": ..., "sources": [...]})  - nothing reconstructed,
    nothing auth-bearing. `request_payload` is the exact JSON body sent; headers are
    deliberately never included here.

    `history_turns` is how many prior conversation turns (see capture/client.py's
    `history` param) were included in this request - 0 for an ordinary standalone
    capture, >0 for a follow-up question asked in the same chat as an earlier row.
    `note` is free-text traceability for anything noteworthy that isn't itself a failure
    (e.g. "follow-up row sent standalone - prior turn not available") - never a secret.
    """
    return {
        "id": row_id,
        "request_payload": {query_field: query},
        "endpoint_url": endpoint_url,
        "http_status": http_status,
        "complete_sse_event": complete_event,
        "captured_answer": answer,
        "source_filenames": list(source_filenames) if source_filenames else [],
        "capture_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_at)),
        "duration_seconds": round(duration_seconds, 3),
        "success": success,
        "error": error,
        "history_turns": history_turns,
        "note": note,
    }


def derive_log_path(captured_dataset_path: Path) -> Path:
    """Mirrors the historical naming convention already present in this dataset's own
    provenance metadata (golden_dataset_final_unified_eval.capture_log.json): the
    captured dataset's filename with '.json' replaced by '.capture_log.json'."""
    name = captured_dataset_path.name
    if name.endswith(".json"):
        name = name[: -len(".json")]
    return captured_dataset_path.with_name(name + ".capture_log.json")


def load_existing_log(path: Path) -> list:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def save_log(path: Path, entries: list) -> None:
    """Atomic write (temp file + replace), same pattern as dataset_io.save_captured_dataset.
    Entries are merged by id with whatever is already on disk, so a run scoped with
    --ids doesn't discard other rows' previously logged entries."""
    existing = load_existing_log(path)
    by_id = {e["id"]: e for e in existing if isinstance(e, dict) and "id" in e}
    for e in entries:
        by_id[e["id"]] = e
    merged = list(by_id.values())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)
