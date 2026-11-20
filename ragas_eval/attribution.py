"""Deterministic cited-source matcher for the Sources-Validation judge. No API calls.

cited_sources are already document ids normalised to `<doc>.md`, and expected_sources
carry `{document, necessity}` — so matching is exact-name, not filename<->title fuzzing.

Facts computed here rather than left to the LLM, because they are mechanical:

  * `exists_in_kb`   — per cited document, membership in a local knowledge-base document
                       union, IF ONE IS CONFIGURED (see `kb_names` below). A cited
                       document absent from it is a FABRICATION (automatic 0 - see
                       judge_runner.py's coerce()).
  * `required_hit` / `acceptable_hit` — per expected source, whether it was cited.
  * counts the judge needs for recall/precision arithmetic.

The row's expected_sources remains the scoring truth for required/acceptable; KB existence
is only the fabrication floor. The judge still decides claim-level support ("does each
citation point at a document that actually contains the claim") — that part is judgement,
not string matching, and stays in the prompt.

`kb_names` comes from dataset.kb_document_names(): this project's `documents/` corpus, in
the chatbot's citation form (see kb_index.py), and expected_sources are resolved to that
same form at load time. `kb_names=None` means no KB corpus is configured at all - distinct
from `kb_names=set()` (a real, empty corpus) - so the fabrication check is SKIPPED rather
than falsely triggered on every citation. See judge_runner.py's two `_SOURCES_AMENDMENT_*`
texts for the matching judge-prompt wording.
"""
from __future__ import annotations

from typing import Optional


def _norm(name: str) -> str:
    name = (name or "").strip()
    return name if name.endswith(".md") else name + ".md"


def build_hint(sample, kb_names: Optional[set]) -> dict:
    """The deterministic pre-pass shown to the Sources-Validation judge."""
    cited = [_norm(c) for c in sample.cited_sources]
    expected = [{"document": _norm(e.get("document", "")),
                 "necessity": e.get("necessity", "required")}
                for e in sample.expected_sources]
    required = [e["document"] for e in expected if e["necessity"] == "required"]
    acceptable = [e["document"] for e in expected if e["necessity"] != "required"]

    required_cited = sorted([d for d in required if d in cited])
    required_missing = sorted([d for d in required if d not in cited])
    acceptable_cited = sorted([d for d in acceptable if d in cited])
    unexpected_cited = sorted([c for c in cited
                               if c not in required and c not in acceptable])

    if kb_names is None:
        fabricated: Optional[list] = None
        note = ("No local KB document corpus is configured for this project, so "
                "fabrication CANNOT be mechanically checked - fabricated_not_in_kb is "
                "null (not an empty list) and must NOT be treated as 'nothing is "
                "fabricated'. Score recall/precision normally from "
                "expected_required/expected_acceptable vs cited_sources; do not set the "
                "fabrication flag on KB-membership grounds. Whether an `acceptable` twin "
                "carries the same claim as a missing `required` source — and whether each "
                "citation actually supports the claim it is attached to — is YOUR "
                "judgement, not this hint's.")
    else:
        exists = {c: (c in kb_names) for c in cited}
        fabricated = sorted([c for c, ok in exists.items() if not ok])
        note = ("fabricated_not_in_kb is authoritative (mechanical membership check "
                "against the configured KB document union). Whether an `acceptable` twin "
                "carries the same claim as a missing `required` source — and whether each "
                "citation actually supports the claim it is attached to — is YOUR "
                "judgement, not this hint's.")

    return {
        "cited_sources": cited,
        "expected_required": sorted(required),
        "expected_acceptable": sorted(acceptable),
        "required_cited_directly": required_cited,
        "required_not_cited_directly": required_missing,
        "acceptable_cited": acceptable_cited,
        "cited_but_not_expected": unexpected_cited,
        "fabricated_not_in_kb": fabricated,
        "kb_size": len(kb_names) if kb_names is not None else None,
        "note": note,
    }
