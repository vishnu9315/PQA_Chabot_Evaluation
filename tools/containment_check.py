#!/usr/bin/env python3
"""Non-LLM cross-check for Context Precision: does each retrieved chunk contain a verbatim
fragment (6-word shingle) of the row's reference? Prints per-chunk verdicts and the
rank-weighted precision ragas would compute from them. No API calls.

Use it to sanity-check the LLM judge: where the judge scores a row 0 but this finds
reference fragments in the chunks, the judge is suspect (that is how the stock prompt's
false-zero on GD-MTY-MULTI-01-N was caught). Generic shared phrasing (e.g. a Minister's
title) can inflate it, so treat it as a guide, not ground truth.

    .venv/Scripts/python.exe tools/containment_check.py --ids GD-MTY-MULTI-01-N,GD-MTY-MULTI-04-N
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def shingles(text: str, n: int = 6) -> set:
    w = re.findall(r"[a-z0-9.,]+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def rank_weighted_precision(verdicts: list) -> float:
    hits, total = 0, 0.0
    for k, v in enumerate(verdicts, 1):
        if v:
            hits += 1
            total += hits / k
    return total / hits if hits else 0.0


def main() -> int:
    from config import load_config
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="comma-separated row ids")
    args = ap.parse_args()
    rows = {r["id"]: r for r in json.loads(
        load_config().unified_eval_json.read_text(encoding="utf-8"))["rows"]}
    for rid in args.ids.split(","):
        r = rows[rid]
        ref = shingles(r["reference"])
        v = [1 if shingles(c) & ref else 0 for c in r["retrieved_contexts"]]
        print(f"{rid:28s} chunks={len(v):2d} with-ref-fragment={sum(v)} verdicts={v} "
              f"precision={rank_weighted_precision(v):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
