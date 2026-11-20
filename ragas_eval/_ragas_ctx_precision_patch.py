"""Calibration hardening for ragas 0.4.3 `ContextPrecisionPrompt` (ISSUE-P11).

The false-low
-------------
Context Precision asks, per retrieved chunk, "was this context useful in arriving at the
given answer?" — where "the answer" is our authored `reference`: a long compound text
synthesized from many quoted facts, carrying `[C1]`-style citation markers and a trailing
`Sources:` line. The stock few-shot examples are all short single-fact answers, so the
judge learns an implicit all-or-nothing standard and zeroes chunks that support only *part*
of the reference.

Measured on this corpus (full run 2026-08-04): TC-006 scored 0.0 although chunks at ranks
16/22/23/32 (among others) contain **verbatim** fragments of the reference — the string
"scrutinized in office within 30 days" appears verbatim in the rank-32 chunk. A fair judge
on those ranks gives ≈0.22, which is exactly what the independent string-containment
cross-check produced (0.2215). TC-016 likewise scored 0.0 with 4–6 genuinely supporting
chunks.

The fix — prompt text only
--------------------------
Scoring math, parsing and metric class are untouched (hard rule). When hardened, this
module appends a partial-support directive to the class-level `instruction` and adds ONE
extra few-shot example demonstrating verdict=1 for a chunk that supports a single clause of
a long multi-fact answer. `examples` is a class-level list shared across instances, so a
new list is assigned (`cls.examples = cls.examples + [new]`), never `.append()`.

Code default stock; this project runs hardened
-----------------------------------------------
Env knob `CTX_PRECISION_JUDGE`: `stock` | `hardened`. The code default stays `stock`; this
project sets `hardened` in `.env` after its own 2026-11-20 calibration (runs_calib/): stock
false-zeroed the two-document row GD-MTY-MULTI-01-N (0.00, vs 0.50 from both the non-LLM
containment check and context recall), while hardened matched containment on 6/6 rows. The
recall arm stays stock. The resolved mode is recorded per cell and in
`config_snapshot.json`.

`apply()` is idempotent (marker-guarded, `_PATCHED` flag) and wired into
`ragas_eval/__init__.py` after the existing patches. `revert()` restores the captured stock
prompt — needed by the offline tests so hardened assertions cannot leak into stock ones.
"""
from __future__ import annotations

import logging
import os

_log = logging.getLogger(__name__)

KNOB = "CTX_PRECISION_JUDGE"
_VALID = ("stock", "hardened")
# Code default is the stock prompt; a project opts into `hardened` via .env only after its
# own calibration (this project did on 2026-11-20 - see the module docstring).
_DEFAULT = "stock"

_PATCHED = False
_MODE = "stock"
_ORIG_INSTRUCTION = None
_ORIG_EXAMPLES = None

# Marker phrase guarding the instruction append (idempotency across repeated apply()).
_MARKER = "The answer may be long and synthesized from multiple sources"

_INSTRUCTION_APPEND = (
    " " + _MARKER + ": give verdict 1 if the context supports or contains ANY statement, "
    "fact, figure, or quotation that appears in the answer, even if it covers only a small "
    "part and other contexts would be needed for the rest. Give verdict 0 only if nothing "
    "in the context supports any part of the answer. Ignore citation markers such as [C1] "
    'and any trailing "Sources:" line in the answer — they are citation apparatus; do not '
    "require the context to match them or to come from a document named there. Equivalent "
    'paraphrases and abbreviations (e.g. "MTPA" vs "million tonnes per annum") count as '
    "support."
)

# The added example's question doubles as the examples-list marker.
#
# DELIBERATELY FICTIONAL DOMAIN (municipal transit, invented facts): the example must teach
# the partial-support RULE without quoting anything from the evaluation corpus. An earlier
# draft reused real EIA phrasing — including a verbatim TC-006 reference fragment — which
# would have contaminated the very cells the validation pilot judges, rigging the gate.
_EXAMPLE_QUESTION = ("What approvals and timelines apply to expanding a city bus depot "
                     "for electric buses?")


def mode() -> str:
    """The resolved knob value for this process (recorded in the run snapshot)."""
    return _MODE


def _prompt_cls():
    from ragas.metrics._context_precision import ContextPrecisionPrompt
    return ContextPrecisionPrompt


def _ensure_originals() -> None:
    """Capture the stock prompt once, before any mutation (used by revert() and tests)."""
    global _ORIG_INSTRUCTION, _ORIG_EXAMPLES
    if _ORIG_INSTRUCTION is None:
        cls = _prompt_cls()
        _ORIG_INSTRUCTION = cls.instruction
        _ORIG_EXAMPLES = list(cls.examples)


def _harden() -> bool:
    """Mutate the class-level prompt. Returns True if anything changed (idempotent)."""
    global _PATCHED
    if _PATCHED:
        return False
    from ragas.metrics._context_precision import QAC, Verification

    _ensure_originals()
    cls = _prompt_cls()

    if _MARKER not in (cls.instruction or ""):
        cls.instruction = cls.instruction.rstrip() + _INSTRUCTION_APPEND

    if not any(inp.question == _EXAMPLE_QUESTION for inp, _ in cls.examples):
        example = (
            QAC(
                question=_EXAMPLE_QUESTION,
                context=(
                    "Under the transit authority's procurement code, fleet expansions of "
                    "up to 40 buses may reuse the previous tender's technical evaluation, "
                    "provided the vehicle specification remains unchanged."
                ),
                answer=(
                    "Depot expansion requires a land-use variance from the municipal "
                    "planning board before construction [C1]. Fleet expansions of up to "
                    "40 buses may reuse the previous tender's technical evaluation if the "
                    "vehicle specification is unchanged [C2]. A safety audit must be "
                    "completed before the first revenue service [C3], and night-charging "
                    "bays additionally require a separate electrical permit [C4]. "
                    "Sources: [C1] Municipal Code ch. 12; [C2] Procurement Code s. 7; "
                    "[C3] Safety Manual; [C4] Electrical Code."
                ),
            ),
            Verification(
                reason=(
                    "The context supports the tender-reuse clause (expansions up to 40 "
                    "buses with an unchanged specification) that appears in the answer. "
                    "It covers only that one part of a long, multi-source answer and "
                    "other contexts would be needed for the remaining facts, but "
                    "supporting any part is sufficient for verdict 1. The [C2] marker and "
                    "the trailing Sources line are citation apparatus and are ignored."
                ),
                verdict=1,
            ),
        )
        # class-level list is shared across instances: assign a NEW list, never .append()
        cls.examples = cls.examples + [example]

    _PATCHED = True
    _log.info("ISSUE-P11 context-precision prompt hardened (partial-support calibration).")
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
    global _MODE
    raw = mode_override if mode_override is not None else os.getenv(KNOB)
    resolved = (raw or _DEFAULT).strip().lower() or _DEFAULT
    if resolved not in _VALID:
        raise ValueError(f"{KNOB}={resolved!r} is not one of {_VALID}")
    _MODE = resolved
    if resolved == "stock":
        return revert() if _PATCHED else False
    return _harden()


def revert() -> bool:
    """Restore the captured stock prompt (test isolation). Returns True if restored."""
    global _PATCHED, _MODE
    _MODE = "stock"
    if _ORIG_INSTRUCTION is None:
        _PATCHED = False
        return False
    cls = _prompt_cls()
    cls.instruction = _ORIG_INSTRUCTION
    cls.examples = list(_ORIG_EXAMPLES)
    _PATCHED = False
    return True
