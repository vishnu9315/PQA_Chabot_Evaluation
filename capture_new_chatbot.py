#!/usr/bin/env python3
"""Capture the MeitY PQA chatbot's responses for the golden evaluation dataset
(meity-pq-SHORT-2part-v1, 66 rows - see golden_dataset_README.md).

This script ONLY captures: it calls the live chatbot for each row's `user_input` and
records what it says. It does NOT run any RAGAS metric or LLM judge, and does NOT need
AWS/Bedrock credentials - those are read later, by cli.py, from the captured output file
this script produces. The golden dataset it reads from is never modified; every row's
ground-truth fields (see capture/dataset_io.py PROTECTED_FIELDS) are asserted unchanged
before every save.

Confirmed 2026-11-20 via live Playwright network capture against the real MeitY "AI
Assistant" chat: this project's chatbot has NO wrapper/drafting prompt - row["user_input"]
is sent verbatim as `query`, with no Instruction/Division framing (unlike the previous
project this pipeline was copied from). See chatbot_endpoint_config.json for the full
confirmed contract.

Usage:
    python3 capture_new_chatbot.py --dry-run         # validate config + dataset, no calls
    python3 capture_new_chatbot.py                   # capture all rows, skip already-done
    python3 capture_new_chatbot.py --force            # re-capture every selected row
    python3 capture_new_chatbot.py --ids GD-MTY-LS17-2-U2536-V
    python3 capture_new_chatbot.py --force --ids GD-MTY-RS-249-U642-F1   # re-capture one row

Configuration is read from environment variables (see .env.example and README.md):
    PQA_CHATBOT_BASE_URL         required - no default, see chatbot_endpoint_config.json
    PQA_CHATBOT_SESSION_FILE     optional - path to a Playwright storageState JSON
                                  (e.g. auth-capture/.auth/default.json);
                                  its cookies for PQA_CHATBOT_BASE_URL's host are sent
                                  as a Cookie header. See README's "Supplying a session
                                  cookie" section - this is the confirmed auth mechanism.
    PQA_CHATBOT_COOKIE_HEADER    optional - a raw 'Cookie:' header value, if you'd
                                  rather paste one directly than point at a session file.
                                  Takes precedence over PQA_CHATBOT_SESSION_FILE if both
                                  are set.
    PQA_CHATBOT_API_KEY          optional - sent as 'Authorization: Bearer <value>'
                                  (placeholder only - not the confirmed mechanism)
    PQA_CHATBOT_EXTRA_HEADERS_JSON  optional - JSON object of extra request headers
    PQA_CHATBOT_TIMEOUT          optional - seconds, default 120
    GOLDEN_DATASET_PATH          optional - default ./golden_dataset.json
    CAPTURED_DATASET_PATH        optional - default ./data/artifacts/golden_dataset_new_chatbot_captured.json
    GOLDEN_EXPECTED_TOTAL_ROWS / GOLDEN_EXPECTED_RAGAS_ELIGIBLE / GOLDEN_EXPECTED_JUDGE_ONLY_IDS
                                  optional - see capture/dataset_io.py; already set in
                                  .env for this dataset's actual shape (66/66/none)

Cookie values are never logged or printed by this script - only safe metadata (paths,
cookie counts/names) ever appears in log output.

Ctrl-C saves progress made so far and exits with status 1 rather than losing captured rows.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_dotenv(env_path: Path) -> None:
    """Load ROOT/.env into os.environ, same as config.py does for the scoring side -
    without this, setting values in .env would silently have no effect on this script.
    Prefers python-dotenv; falls back to a tiny parser if it isn't installed. Never
    overrides a variable already set in the real environment.
    """
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(env_path)
        return
    except ModuleNotFoundError:
        pass
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        val = val.split(" #", 1)[0].strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), val)


_load_dotenv(ROOT / ".env")

from capture.client import (  # noqa: E402
    CaptureAuthError,
    CaptureConfigError,
    CaptureRequestError,
    ChatbotClientConfig,
    call_chatbot,
)
from capture.dataset_io import (  # noqa: E402
    DatasetIntegrityError,
    assert_protected_fields_unchanged,
    build_output_rows,
    load_existing_capture,
    load_golden_dataset,
    row_already_captured,
    save_captured_dataset,
    validate_row_counts,
)
from capture import capture_log  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("capture_new_chatbot")

DEFAULT_GOLDEN_PATH = ROOT / "golden_dataset.json"
DEFAULT_OUTPUT_PATH = ROOT / "data" / "artifacts" / "golden_dataset_new_chatbot_captured.json"
DEFAULT_ENDPOINT_CONFIG_PATH = ROOT / "chatbot_endpoint_config.json"


class _Interrupted(RuntimeError):
    """Raised by the SIGINT handler so the main loop can save progress before exiting."""


def _install_sigint_handler() -> None:
    def _handler(signum, frame):  # noqa: ARG001
        raise _Interrupted()

    signal.signal(signal.SIGINT, _handler)


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--golden", type=Path, default=None,
                   help=f"golden dataset path (default: {DEFAULT_GOLDEN_PATH}, "
                        f"or $GOLDEN_DATASET_PATH)")
    p.add_argument("--out", type=Path, default=None,
                   help=f"captured output path (default: {DEFAULT_OUTPUT_PATH}, "
                        f"or $CAPTURED_DATASET_PATH)")
    p.add_argument("--endpoint-config", type=Path, default=DEFAULT_ENDPOINT_CONFIG_PATH)
    p.add_argument("--ids", type=str, default=None,
                   help="comma-separated row ids to capture (default: all rows)")
    p.add_argument("--force", action="store_true",
                   help="re-capture selected rows even if already successfully captured")
    p.add_argument("--dry-run", action="store_true",
                   help="validate config and dataset, print the plan, make no network calls")
    p.add_argument("--save-every", type=int, default=5,
                   help="write the output file after every N captured rows (default: 5)")
    p.add_argument("--capture-log", type=Path, default=None,
                   help="optional debug/traceability log path (default: derived from "
                        "--out, or $CAPTURE_LOG_PATH); see capture/capture_log.py")
    p.add_argument("--no-capture-log", action="store_true",
                   help="disable writing the optional capture log")
    p.add_argument("--user-id", type=str, default=None,
                   help="user_id sent with every request this run (default: a fresh "
                        "random UUID, generated once per run - see "
                        "chatbot_endpoint_config.json)")
    return p.parse_args(argv)


def resolve_path(cli_value: Optional[Path], env_var: str, default: Path) -> Path:
    if cli_value is not None:
        return cli_value
    env_value = os.environ.get(env_var, "").strip()
    if env_value:
        return Path(env_value)
    return default


def _build_meta(n_done: int, n_failed: int, n_skipped: int, *, interrupted: bool) -> dict:
    return {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_captured_this_run": n_done,
        "n_failed_this_run": n_failed,
        "n_skipped_this_run": n_skipped,
        "interrupted": interrupted,
        "note": (
            "This file is OUTPUT of capture_new_chatbot.py, produced by calling the "
            "MeitY PQA chatbot - it is a separate file from the golden dataset and is "
            "never read by cli.py's scoring pipeline unless MOC_ARTIFACTS_DIR/"
            "GOLDEN_DATASET_PATH-style configuration is deliberately pointed at it. "
            "The original golden dataset (golden_dataset.json) is never modified by "
            "this script."
        ),
    }


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    golden_path = resolve_path(args.golden, "GOLDEN_DATASET_PATH", DEFAULT_GOLDEN_PATH)
    out_path = resolve_path(args.out, "CAPTURED_DATASET_PATH", DEFAULT_OUTPUT_PATH)
    log_path = None
    if not args.no_capture_log:
        log_path = resolve_path(
            args.capture_log, "CAPTURE_LOG_PATH", capture_log.derive_log_path(out_path))

    if not args.endpoint_config.exists():
        logger.error("endpoint config not found: %s", args.endpoint_config)
        return 2
    endpoint_config = json.loads(args.endpoint_config.read_text(encoding="utf-8"))

    try:
        client_cfg = ChatbotClientConfig.from_env(endpoint_config)
    except CaptureConfigError as exc:
        logger.error("%s", exc)
        return 2

    if not golden_path.exists():
        logger.error("golden dataset not found: %s", golden_path)
        return 2

    try:
        golden_data = load_golden_dataset(golden_path)
        validate_row_counts(golden_data, where=f"{golden_path.name} (before capture)")
    except DatasetIntegrityError as exc:
        logger.error("golden dataset failed validation: %s", exc)
        return 2

    logger.info("golden dataset OK: %d rows from %s", len(golden_data["rows"]), golden_path)

    resume_from = load_existing_capture(out_path)
    if resume_from:
        logger.info("found existing capture at %s - will skip already-successful rows "
                    "unless --force is passed", out_path)

    rows = build_output_rows(golden_data, resume_from)
    golden_by_id = {r["id"]: r for r in golden_data["rows"]}
    # Built once, read (never replaced) as the loop below mutates individual row dicts in
    # place - so a follow-up row later in file order sees its prior turn's fresh capture.
    rows_by_id = {r["id"]: r for r in rows}
    wanted_ids = set(args.ids.split(",")) if args.ids else None
    user_id = args.user_id if args.user_id is not None else str(uuid.uuid4())

    if args.dry_run:
        todo = [r for r in rows
                if (wanted_ids is None or r["id"] in wanted_ids)
                and (args.force or not row_already_captured(r))]
        logger.info("[dry-run] endpoint: %s", client_cfg.url)
        logger.info("[dry-run] user_id: %s", user_id)
        logger.info("[dry-run] would capture %d/%d row(s); no network calls made",
                    len(todo), len(rows))
        for r in todo:
            logger.info("[dry-run]   %s: %r", r["id"], r["user_input"][:80])
        return 0

    _install_sigint_handler()

    n_done = n_failed = n_skipped = 0
    captured_since_save = 0
    interrupted = False
    auth_failed = False
    query_field = endpoint_config["request"]["query_field"]
    log_entries: list = []

    def _log_entry(rid: str, query: str, t0: float, *, http_status, complete_event,
                    answer, source_filenames, success: bool, error,
                    history_turns: int = 0, note: Optional[str] = None) -> None:
        if log_path is None:
            return
        log_entries.append(capture_log.build_log_entry(
            row_id=rid, query=query, query_field=query_field, endpoint_url=client_cfg.url,
            http_status=http_status, complete_event=complete_event, answer=answer,
            source_filenames=source_filenames, started_at=t0,
            duration_seconds=time.time() - t0, success=success, error=error,
            history_turns=history_turns, note=note))

    _SUCCESSFUL_STATUSES = ("captured", "captured_preview_only")

    def _build_history_for(row: dict) -> tuple[Optional[list], Optional[str]]:
        """If `row` is a follow-up (provenance.follow_up_of set - see
        golden_dataset_README.md section 3.2), builds the prior turn's {"role",
        "content"} pair from the prior row's own already-captured query+answer, found in
        `rows_by_id` regardless of whether that prior row was captured in THIS run or a
        previous one (both end up in `rows`/`rows_by_id` the same way). Returns
        (history, warning) - history is None (send standalone) when the prior row isn't
        successfully captured yet, with `warning` explaining why; `warning` is also None
        for an ordinary non-follow-up row.

        Confirmed live 2026-11-20: this replay-the-prior-turn mechanism is what the real
        frontend does and is what actually carries context (not session_id alone) - see
        chatbot_endpoint_config.json's "verified_continuity" note.
        """
        follow_up_of = (row.get("provenance") or {}).get("follow_up_of")
        if not follow_up_of:
            return None, None
        prior = rows_by_id.get(follow_up_of)
        if prior is None:
            return None, (
                f"follow-up of {follow_up_of!r}, but that id is not in the golden "
                f"dataset at all - sent standalone; per golden_dataset_README.md 3.2, "
                f"treat this row's score as invalid")
        if prior.get("capture_status") not in _SUCCESSFUL_STATUSES or not prior.get("response"):
            return None, (
                f"follow-up of {follow_up_of!r}, but that row has no successful capture "
                f"yet (capture_status={prior.get('capture_status')!r}) - sent standalone "
                f"without conversation history; per golden_dataset_README.md 3.2, treat "
                f"this row's score as invalid")
        # No wrapper prompt for this project (see module docstring) - the prior turn's
        # "prompt" is simply its own verbatim user_input.
        return [
            {"role": "user", "content": prior["user_input"]},
            {"role": "assistant", "content": prior["response"]},
        ], None

    try:
        for row in rows:
            rid = row["id"]
            if wanted_ids is not None and rid not in wanted_ids:
                continue
            if not args.force and row_already_captured(row):
                n_skipped += 1
                logger.info("[%s] SKIP (already captured)", rid)
                continue

            logger.info("[%s] capturing...", rid)
            t0 = time.time()
            # No wrapper prompt for this project - row["user_input"] goes straight to
            # the chatbot verbatim, confirmed against the real app's own request. See
            # chatbot_endpoint_config.json and this module's docstring.
            query = row["user_input"]
            session_id = str(uuid.uuid4())  # fresh per row; see _build_history_for's note
            history, continuity_warning = _build_history_for(row)
            if continuity_warning:
                logger.warning("[%s] %s", rid, continuity_warning)
            elif history:
                logger.info("[%s] follow-up - including prior turn as chat history", rid)
            try:
                result = call_chatbot(
                    client_cfg, query, endpoint_config,
                    user_id=user_id, session_id=session_id, history=history)
            except CaptureAuthError as exc:
                # Not a per-row failure: the same rejected session will fail every
                # remaining row identically, so record this one and stop the run
                # rather than burning through guaranteed failures.
                n_failed += 1
                row["response"] = ""
                row["retrieved_contexts"] = []
                row["cited_sources"] = []
                row["cited_sources_raw"] = []
                row["capture_status"] = "failed"
                row["capture_error"] = str(exc)
                logger.error("[%s] AUTH FAILED after %.1fs: %s", rid, time.time() - t0, exc)
                _log_entry(rid, query, t0,
                           http_status=getattr(exc, "status_code", None), complete_event=None,
                           answer=None, source_filenames=[], success=False, error=str(exc),
                           history_turns=len(history or []), note=continuity_warning)
                assert_protected_fields_unchanged(golden_by_id[rid], row)
                auth_failed = True
                break
            except CaptureRequestError as exc:
                n_failed += 1
                row["response"] = ""
                row["retrieved_contexts"] = []
                row["cited_sources"] = []
                row["cited_sources_raw"] = []
                row["capture_status"] = "failed"
                row["capture_error"] = str(exc)
                logger.error("[%s] FAILED after %.1fs: %s", rid, time.time() - t0, exc)
                _log_entry(rid, query, t0,
                           http_status=getattr(exc, "status_code", None), complete_event=None,
                           answer=None, source_filenames=[], success=False, error=str(exc),
                           history_turns=len(history or []), note=continuity_warning)
            else:
                row["response"] = result.answer
                row["retrieved_contexts"] = result.retrieved_contexts
                row["cited_sources"] = result.cited_sources
                row["cited_sources_raw"] = result.sources_raw
                status = "captured" if result.had_full_context else "captured_preview_only"
                if continuity_warning:
                    # Still a successful call, but a follow-up row sent without its
                    # prior turn's context - per golden_dataset_README.md 3.2, its score
                    # is not meaningful; flagged distinctly so reporting can exclude it.
                    status += "_no_continuity"
                row["capture_status"] = status
                row["capture_error"] = None
                n_done += 1
                logger.info(
                    "[%s] OK in %.1fs (%d source(s), full_context=%s, history_turns=%d)",
                    rid, time.time() - t0, len(result.sources_raw), result.had_full_context,
                    len(history or []))
                _log_entry(rid, query, t0,
                           http_status=result.http_status, complete_event=result.raw_payload,
                           answer=result.answer, source_filenames=result.cited_sources,
                           success=True, error=None,
                           history_turns=len(history or []), note=continuity_warning)

            assert_protected_fields_unchanged(golden_by_id[rid], row)
            captured_since_save += 1
            if captured_since_save >= args.save_every:
                save_captured_dataset(
                    out_path, golden_data, rows,
                    meta=_build_meta(n_done, n_failed, n_skipped, interrupted=False))
                if log_path is not None:
                    capture_log.save_log(log_path, log_entries)
                    log_entries = []
                captured_since_save = 0
    except _Interrupted:
        interrupted = True
        logger.warning(
            "interrupted - saving progress so far (%d captured, %d failed, %d skipped)",
            n_done, n_failed, n_skipped)

    save_captured_dataset(
        out_path, golden_data, rows,
        meta=_build_meta(n_done, n_failed, n_skipped, interrupted=interrupted))
    if log_path is not None:
        capture_log.save_log(log_path, log_entries)

    saved = load_existing_capture(out_path)
    assert saved is not None
    validate_row_counts(saved, where=f"{out_path.name} (after capture)")

    if auth_failed:
        logger.error(
            "aborted: authentication failed - the remaining row(s) were not attempted, "
            "since they would fail identically. Fix PQA_CHATBOT_SESSION_FILE / "
            "PQA_CHATBOT_COOKIE_HEADER and re-run (already-captured rows will be "
            "skipped automatically).")
        return 3

    logger.info("done: %d captured, %d failed, %d skipped (interrupted=%s) -> %s",
                n_done, n_failed, n_skipped, interrupted, out_path)
    return 1 if (interrupted or n_failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
