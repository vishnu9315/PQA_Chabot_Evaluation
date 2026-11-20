#!/usr/bin/env python3
"""Promote capture_new_chatbot.py's captured-dataset output into the unified eval file
cli.py / ragas_eval/dataset.py actually score from (config.py's `unified_eval_json`).

Why this script exists: capture_new_chatbot.py's output (CAPTURED_DATASET_PATH) is
ALREADY structurally a unified file - every row carries both the golden scoring key
and the captured response/retrieved_contexts/cited_sources in one place, since
capture/dataset_io.py's build_output_rows() merges them per row. The only thing missing
is a top-level `amendments` key, which ragas_eval/dataset.py's load_payload() requires
(TEMPLATE: the previous project used `amendments` to track a history of post-freeze
corrections - this starts as an empty list for a first run; append to it by hand if you
ever hand-correct a row after scoring, the same way that project did).

This is a deliberate, explicit "promote" step rather than scoring straight off the raw
capture output, so a capture that's still in progress (or was interrupted) never gets
scored by accident - you run this only when you're satisfied the capture is complete.

Usage:
    python3 tools/build_unified_eval.py
    python3 tools/build_unified_eval.py --captured path/to/captured.json --out path/to/unified.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import load_config  # noqa: E402


def build(captured_path: Path, out_path: Path) -> dict:
    data = json.loads(captured_path.read_text(encoding="utf-8"))
    if "rows" not in data:
        raise ValueError(f"{captured_path}: missing top-level 'rows' key - is this really "
                          f"a capture_new_chatbot.py output file?")

    not_captured = [r["id"] for r in data["rows"]
                    if r.get("capture_status") in (None, "not_captured")]
    failed = [r["id"] for r in data["rows"] if r.get("capture_status") == "failed"]
    if not_captured:
        raise ValueError(
            f"{len(not_captured)} row(s) were never captured: {not_captured[:5]}"
            f"{'...' if len(not_captured) > 5 else ''} - run capture_new_chatbot.py "
            f"again (it resumes automatically) before building the unified eval file.")
    if failed:
        print(f"WARNING: {len(failed)} row(s) have capture_status='failed' and will be "
              f"scored with an empty response: {failed}", file=sys.stderr)

    unified = {
        "name": data.get("name"),
        "built": data.get("built"),
        "note": (
            f"Unified eval file built by tools/build_unified_eval.py from "
            f"{captured_path.name} ({data.get('note', '')})"
        ),
        "amendments": data.get("amendments", []),
        "n_rows": data["n_rows"],
        "rows": data["rows"],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(unified, indent=2, ensure_ascii=False), encoding="utf-8")
    return unified


def main(argv=None) -> int:
    cfg = load_config()
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--captured", type=Path,
                   default=ROOT / "data" / "artifacts" / "golden_dataset_new_chatbot_captured.json")
    p.add_argument("--out", type=Path, default=cfg.unified_eval_json)
    args = p.parse_args(argv)

    if not args.captured.exists():
        print(f"ERROR: captured dataset not found: {args.captured}", file=sys.stderr)
        return 2

    unified = build(args.captured, args.out)
    print(f"wrote {len(unified['rows'])} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
