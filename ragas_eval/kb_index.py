"""Knowledge-base document index built from `documents/` (the chatbot's ingested corpus).

Every file under `documents/` is ingested into the chatbot (operator-confirmed 2026-11-20).
The chatbot cites those same files by a different spelling, so this module maps both sides
onto ONE canonical id — the chatbot's own citation form:

    documents/Lok Sabha/Lok Sabha 18/Session VIII/AU2878_rN86Hc.pdf
        -> lok_sabha/lok_sabha_18/Session_VIII/AU2878_rN86Hc.md
    documents/Rajya Sabha/Session 270/AS239_CYnH3S.pdf
        -> rajya_sabha/Session_270/AS239_CYnH3S.md

The golden dataset's `expected_sources[].document` uses yet another spelling (the file names
as QA received them, e.g. `.../Unstarred Question_2878_050826.docx`), so expected sources are
resolved through each session's `metadata.csv` on (house, assembly, session, question no,
question type) — fields every expected_sources entry carries. An entry that does not resolve
(e.g. Part 2's not-ingested replies, deliberately absent from the KB) keeps its authored
name and simply cannot be matched by any citation.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8,
          "ix": 9, "x": 10, "xi": 11, "xii": 12, "xiii": 13, "xiv": 14, "xv": 15}


def canonical_id(path: Path, documents_dir: Path) -> Optional[str]:
    parts = path.relative_to(documents_dir).parts
    if len(parts) == 4 and parts[0] == "Lok Sabha":
        assembly = re.search(r"\d+", parts[1]).group()
        session = parts[2].split()[-1]
        return f"lok_sabha/lok_sabha_{assembly}/Session_{session}/{path.stem}.md"
    if len(parts) == 3 and parts[0] == "Rajya Sabha":
        session = re.search(r"\d+", parts[1]).group()
        return f"rajya_sabha/Session_{session}/{path.stem}.md"
    return None


def _session_key(session: str) -> Optional[tuple]:
    """'ls18-viii' -> ('ls', 18, 8); 'rs-270' -> ('rs', None, 270)."""
    m = re.fullmatch(r"ls(\d+)-([ivx]+)", session or "")
    if m and m.group(2) in _ROMAN:
        return ("ls", int(m.group(1)), _ROMAN[m.group(2)])
    m = re.fullmatch(r"rs-(\d+)", session or "")
    if m:
        return ("rs", None, int(m.group(1)))
    return None


def citation_prefix(session: str) -> Optional[str]:
    """'ls18-viii' -> 'lok_sabha/lok_sabha_18/Session_VIII/'; 'rs-271' -> 'rajya_sabha/Session_271/'."""
    m = re.fullmatch(r"ls(\d+)-([ivx]+)", session or "")
    if m:
        return f"lok_sabha/lok_sabha_{m.group(1)}/Session_{m.group(2).upper()}/"
    m = re.fullmatch(r"rs-(\d+)", session or "")
    return f"rajya_sabha/Session_{m.group(1)}/" if m else None


def session_label(session: str) -> str:
    """'ls18-viii' -> 'Lok Sabha 18 Session VIII'; 'rs-271' -> 'Rajya Sabha Session 271'."""
    m = re.fullmatch(r"ls(\d+)-([ivx]+)", session or "")
    if m:
        return f"Lok Sabha {m.group(1)} Session {m.group(2).upper()}"
    m = re.fullmatch(r"rs-(\d+)", session or "")
    return f"Rajya Sabha Session {m.group(1)}" if m else str(session)


@dataclass
class KbIndex:
    names: set = field(default_factory=set)
    by_question: dict = field(default_factory=dict)

    def resolve(self, expected: dict) -> Optional[str]:
        sk = _session_key(str(expected.get("session", "")))
        try:
            qno = int(expected.get("qno"))
        except (TypeError, ValueError):
            return None
        if sk is None:
            return None
        return self.by_question.get((*sk, qno, str(expected.get("parl_type", "")).lower()))


def load_kb_index(documents_dir) -> Optional[KbIndex]:
    """None when `documents_dir` doesn't exist (no KB configured — see attribution.py)."""
    d = Path(documents_dir)
    if not d.is_dir():
        return None
    idx = KbIndex()
    for p in d.rglob("*.pdf"):
        cid = canonical_id(p, d)
        if cid:
            idx.names.add(cid)
    for meta in d.rglob("metadata.csv"):
        with open(meta, encoding="utf-8", newline="") as fh:
            for m in csv.DictReader(fh):
                cid = canonical_id(meta.parent / m["filename"], d)
                if cid is None or cid not in idx.names:
                    continue
                house = "ls" if m["house"].strip().lower().startswith("lok") else "rs"
                assembly = int(m["house_no"]) if house == "ls" else None
                key = (house, assembly, int(m["session"]), int(m["q_no"]),
                       m["qtype"].strip().lower())
                idx.by_question[key] = cid
    return idx
