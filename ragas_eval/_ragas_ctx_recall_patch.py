"""Calibration hardening for ragas 0.4.3 `ContextRecallClassificationPrompt` (ISSUE-P11).

NOT ADOPTED — this arm failed the Phase 4 validation gate. See `_DEFAULT` below and the
ISSUE-P11 section of KNOWN_ISSUES.md. (An earlier draft of this file cited "ISSUE-P12";
that number now belongs to the PER_CHUNK_TIMEOUT mis-calibration, and the recall result has
always been recorded under ISSUE-P11 alongside the precision arm it was gated with.)

The false-low
-------------
Context Recall decomposes the reference into statements and asks, per statement, whether it
is attributable to the retrieved context. The stock prompt (and its single few-shot
example) works sentence-by-sentence, so on this corpus — where authored references are long
COMPOUND sentences stitched from several quoted facts — the judge classifies each compound
sentence all-or-nothing: a sentence with 2-of-3 facts retrieved scores 0. The trailing
`Sources:` line is emitted as a statement and is a guaranteed dead entry (nothing in the
chunks "supports" a citation list), deflating every record's denominator.

Measured on this corpus (full run 2026-08-04): TC-006's reference is 6 compound sentences +
a Sources line; roughly 7 of its 11 atomic facts are present in the chunks (fair ≈ 0.6),
yet the judge returned 0.125 — exactly 1/8, the all-or-nothing signature. Divergences vs
the independent string cross-check: TC-006 0.125 vs 0.75, TC-031 0.20 vs 0.75, TC-013 0.00
vs 0.40, while TC-016 agrees (0.286 vs 0.20 — a genuine retrieval gap, which must NOT
inflate).

The fix — prompt text only
--------------------------
Scoring math, parsing and metric class are untouched (hard rule). When hardened, this
module appends a split-compound-sentences directive to the class-level `instruction` and
adds ONE extra few-shot example showing a compound sentence split into atomic statements
with mixed verdicts, citation markers stripped, and the Sources line not emitted as a
statement. `examples` is a class-level list shared across instances, so a new list is
assigned, never `.append()`.

Default OFF (validation gate)
-----------------------------
Env knob `CTX_RECALL_JUDGE`: `stock` (default) | `hardened`. Adopted as default only if the
Phase 3 pilot passes the Phase 4 gate in `AGENT_PLAN_judge_calibration.md`. The resolved
mode is recorded in `config_snapshot.json`. `apply()` is idempotent; `revert()` restores
the captured stock prompt for test isolation.
"""
from __future__ import annotations

import logging
import os

_log = logging.getLogger(__name__)

KNOB = "CTX_RECALL_JUDGE"
_VALID = ("stock", "hardened", "no_apparatus")
# Default OFF pending the `no_apparatus` validation gate (RCA_CONTEXT_RECALL.md option 2).
#
# `hardened` is NOT ADOPTED and must not become the default: the Phase 4 pilot FAILED that arm —
# the atomic-split rule grows the statement denominator faster than it wins attributions, so
# scores fall. TC-016 (a healthy control) dropped 0.2500 -> 0.1306 and TC-031 scored WORSE than
# matched-effort stock. The RCA explains why it must lose: the stock prompt credits a compound
# sentence with one `attributed=1` when ANY part is supported, so splitting turns 1/1 into 1/3.
# Kept behind the knob for future work only. See ISSUE-P11.
_DEFAULT = "stock"

_PATCHED = False
_MODE = "stock"
_APPLIED = "stock"          # which mode is PHYSICALLY installed on the ragas class right now
_ORIG_INSTRUCTION = None
_ORIG_EXAMPLES = None

# Marker phrases guarding each instruction append (idempotency across repeated apply()).
_MARKER = "split it into separate atomic statements"                 # `hardened`
_APPARATUS_MARKER = "citation apparatus, not a factual claim"        # `no_apparatus`

# ---------------------------------------------------------------------------
# `no_apparatus` (ISSUE-P6) — the narrowest possible change, and the only recall option with a
# measured, exactly-priced benefit.
#
# Measured over 64 captured draws (RCA_CONTEXT_RECALL.md section 4): the judge emits the
# reference's trailing `Sources:` line — or a bare citation marker — as a statement in **31% of
# draws at both efforts**, and **all 16 such statements scored 0**. Not most: all. It is a
# guaranteed dead denominator entry, because no retrieved chunk can "support" a bibliography.
# Every one of the 16 references carries such a line, so every record is exposed. Cost when it
# fires: mean 0.0865 (medium) / 0.0674 (max) score points; amortised over all draws,
# +0.0270 / +0.0211 on the column mean.
#
# This directive changes only WHICH statements are emitted, never how a statement is judged, so
# it cannot raise a genuinely-low record — it can only delete entries that were certain zeros.
# That is why it is separable from, and far lower-risk than, the failed `hardened` arm.
# ---------------------------------------------------------------------------
_APPARATUS_APPEND = (
    " Two rules about citation apparatus. They affect only WHICH statements you emit, never how "
    "you judge them: (a) do not emit the trailing \"Sources:\" line as a statement, and do not "
    "emit a bare citation marker such as [C1] as a statement — a citation list is " +
    _APPARATUS_MARKER + " about the subject matter, and no context can support it; (b) strip "
    "citation markers such as [C1] out of the statements you do emit. Decompose and classify "
    "every remaining statement exactly as you otherwise would — do not split sentences "
    "differently, and do not change any verdict because of this instruction."
)

_INSTRUCTION_APPEND = (
    " If a sentence in the answer contains multiple independent facts, " + _MARKER +
    " and classify each independently — never give one combined verdict for a compound "
    "sentence. Ignore citation markers such as [C1], and do not emit the trailing "
    '"Sources:" line as a statement — they are citation apparatus, not content. Classify '
    "a statement as attributed (1) if its factual content is supported anywhere in the "
    "context, including paraphrase and equivalent abbreviations; use 0 only when the "
    "content is absent."
)

# The added example's question doubles as the examples-list marker.
#
# DELIBERATELY FICTIONAL DOMAIN (municipal transit, invented facts): the example must teach
# the atomic-split RULE without quoting anything from the evaluation corpus — real EIA
# phrasing here would contaminate the pilot cells the validation gate judges.
_EXAMPLE_QUESTION = "What are the requirements for adding a new electric bus route?"


def mode() -> str:
    """The resolved knob value for this process (recorded in the run snapshot)."""
    return _MODE


def _prompt_cls():
    from ragas.metrics._context_recall import ContextRecallClassificationPrompt
    return ContextRecallClassificationPrompt


def _ensure_originals() -> None:
    """Capture the stock prompt once, before any mutation (used by revert() and tests)."""
    global _ORIG_INSTRUCTION, _ORIG_EXAMPLES
    if _ORIG_INSTRUCTION is None:
        cls = _prompt_cls()
        _ORIG_INSTRUCTION = cls.instruction
        _ORIG_EXAMPLES = list(cls.examples)


def _strip_apparatus() -> bool:
    """Install the `no_apparatus` directive. Instruction text ONLY — no new few-shot example.

    Deliberately minimal: the stock example contains no citation apparatus, so there is nothing
    for a counter-example to disambiguate, and adding one would risk perturbing decomposition —
    the exact coupling that sank the `hardened` arm. If validation shows the instruction alone is
    not obeyed, an example is the next lever, not the first.
    """
    global _PATCHED
    _ensure_originals()
    cls = _prompt_cls()
    if _APPARATUS_MARKER not in (cls.instruction or ""):
        cls.instruction = cls.instruction.rstrip() + _APPARATUS_APPEND
    _PATCHED = True
    _log.info("ISSUE-P6 context-recall citation-apparatus directive installed (no_apparatus).")
    return True


def _harden() -> bool:
    """Mutate the class-level prompt. Returns True if anything changed (idempotent)."""
    global _PATCHED
    from ragas.metrics._context_recall import (ContextRecallClassification,
                                               ContextRecallClassifications, QCA)

    _ensure_originals()
    cls = _prompt_cls()

    if _MARKER not in (cls.instruction or ""):
        cls.instruction = cls.instruction.rstrip() + _INSTRUCTION_APPEND

    if not any(inp.question == _EXAMPLE_QUESTION for inp, _ in cls.examples):
        example = (
            QCA(
                question=_EXAMPLE_QUESTION,
                context=(
                    "New routes require approval by the transit authority's route "
                    "committee. Operators shall file a service plan on the transit "
                    "portal. The committee shall issue its decision within twelve weeks of "
                    "the filing of the service plan."
                ),
                answer=(
                    "A new electric bus route requires route-committee approval, must "
                    "file a service plan on the transit portal, and is exempt from the "
                    "annual fare review [C1]. The decision is issued within twelve weeks "
                    "[C2]. Sources: [C1] Transit Act s. 14; [C2] Committee rules."
                ),
            ),
            ContextRecallClassifications(classifications=[
                # the first compound sentence splits into three ATOMIC statements with
                # mixed verdicts; [C1]/[C2] markers are stripped from the statements
                ContextRecallClassification(
                    statement="A new electric bus route requires route-committee approval.",
                    reason=("Directly supported: the context states new routes require "
                            "approval by the route committee."),
                    attributed=1,
                ),
                ContextRecallClassification(
                    statement=("A new electric bus route must file a service plan on the "
                               "transit portal."),
                    reason=("Supported: the context says operators shall file a service "
                            "plan on the transit portal."),
                    attributed=1,
                ),
                ContextRecallClassification(
                    statement=("A new electric bus route is exempt from the annual fare "
                               "review."),
                    reason=("Not supported: the context says nothing about a fare-review "
                            "exemption, so this fact is absent even though the rest of "
                            "the same sentence is supported."),
                    attributed=0,
                ),
                ContextRecallClassification(
                    statement="The decision is issued within twelve weeks.",
                    reason=("Supported: the context states the committee shall issue its "
                            "decision within twelve weeks of the filing."),
                    attributed=1,
                ),
                # note: the trailing "Sources:" line is deliberately NOT a statement
            ]),
        )
        # class-level list is shared across instances: assign a NEW list, never .append()
        cls.examples = cls.examples + [example]

    _PATCHED = True
    _log.info("ISSUE-P11 context-recall prompt hardened (atomic-statement calibration).")
    return True


def apply(mode_override: str | None = None) -> bool:
    """Resolve the knob (env or explicit override) and apply. Returns True if this call
    changed the installed prompts. Raises ValueError on an unknown knob value.

    Review findings addressed here:
    * `apply("stock")` after a prior `apply("hardened")` must actually RESTORE the stock
      prompt — leaving the hardened text live while `mode()` (and therefore
      config_snapshot.json) reports "stock" would corrupt the stock-vs-hardened
      comparability the whole validation exists for.
    * An empty/blank env value means "default" (repo convention, cf. NONLLM_SIMILARITY),
      not a ValueError at `import ragas_eval` time.
    """
    global _MODE, _APPLIED
    raw = mode_override if mode_override is not None else os.getenv(KNOB)
    resolved = (raw or _DEFAULT).strip().lower() or _DEFAULT
    if resolved not in _VALID:
        raise ValueError(f"{KNOB}={resolved!r} is not one of {_VALID}")

    if resolved == _APPLIED:            # already physically installed — true no-op
        _MODE = resolved
        return False

    # Switching between two non-stock modes must go through stock first, or the two directives
    # stack on one `instruction` and the judge receives both rule sets at once.
    reverted = revert() if _APPLIED != "stock" else False

    _MODE = resolved
    if resolved == "stock":
        _APPLIED = "stock"
        return reverted
    if resolved == "hardened":
        _harden()
    else:
        _strip_apparatus()
    _APPLIED = resolved
    return True


def revert() -> bool:
    """Restore the captured stock prompt (test isolation). Returns True if restored."""
    global _PATCHED, _MODE, _APPLIED
    _MODE = "stock"
    if _ORIG_INSTRUCTION is None:
        _PATCHED = False
        _APPLIED = "stock"
        return False
    cls = _prompt_cls()
    cls.instruction = _ORIG_INSTRUCTION
    cls.examples = list(_ORIG_EXAMPLES)
    _PATCHED = False
    _APPLIED = "stock"
    return True
