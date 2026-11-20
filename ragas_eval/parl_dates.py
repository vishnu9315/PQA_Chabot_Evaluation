"""Parliamentary-reply metadata parsed from the KB documents themselves.

Single source of truth for two things the dataset's `expected_sources` entries carry:

  * `answered_on(doc, kb_dir)` — the date the reply was answered, from the document header.
    Two header forms exist in this corpus and BOTH must be handled: the numeric Rajya Sabha
    style (`ANSWERED ON 31/07/2026`) and the worded Lok Sabha style (`TO BE ANSWERED ON
    TUESDAY, THE 28TH JULY, 2026.`). A parser that only did the numeric form left 57% of
    answer_dates null and silently disabled the client's newest-answered conflict rule.

  * `from_filename(doc)` — house / parl_type / question number. The FILENAME is authoritative,
    not the header: ls_unstarred_1423.md contains the extraction typo "USTARRED QUESTION NO.
    1423", which defeats a `(?<!UN)STARRED` regex and mis-tagged that document as starred.

This module lives in the framework so `PQA_ragas/` is self-contained and can be shared and
run on its own; a golden-dataset-building script can import it rather than keeping a second
copy of the same regexes.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

MONTHS = {m.upper(): i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"], 1)}

# `ANSWERED ON 31/07/2026`  (Rajya Sabha style, and some Lok Sabha)
_NUMERIC = re.compile(r"ANSWERED\s+ON\s*:?\s*(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})", re.I)
# `TO BE ANSWERED ON TUESDAY, THE 28TH JULY, 2026.`  (worded Lok Sabha style)
_WORDED = re.compile(r"ANSWERED\s+ON[^\n,]*,?\s*THE\s+(\d{1,2})\s*(?:ST|ND|RD|TH)?\s+"
                     r"([A-Za-z]+)\s*,?\s*(\d{4})", re.I)
_NAME = re.compile(r"^(ls|rs)_(starred|unstarred)_(\d+)\.md$")


def answered_on(doc: str, kb_dir) -> str | None:
    """ISO date the reply was answered, or None. Reads the KB document read-only."""
    path = Path(kb_dir) / doc
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    m = _NUMERIC.search(text)
    if m:
        return datetime(int(m.group(3)), int(m.group(2)), int(m.group(1))).date().isoformat()
    m = _WORDED.search(text)
    if m and m.group(2).upper() in MONTHS:
        return datetime(int(m.group(3)), MONTHS[m.group(2).upper()],
                        int(m.group(1))).date().isoformat()
    return None


def from_filename(doc: str) -> dict | None:
    """house / parl_type / qno per the corpus naming convention, or None if unparseable."""
    m = _NAME.match(doc)
    if not m:
        return None
    return {"house": m.group(1), "parl_type": m.group(2), "qno": int(m.group(3))}
