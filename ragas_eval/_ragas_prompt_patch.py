"""Prompt hardening for RAGAS 0.4.3's answer_correctness classifier (ISSUE-9, see
KNOWN_ISSUES.md) — fix the *cause* of the omitted-empty-category failure without fabricating
any score.

The failure (confirmed live on PQ-02): `CorrectnessClassifier` (in
`ragas/metrics/_answer_correctness.py`) classifies statements into TP/FP/FN, whose schema
`ClassificationWithReason` requires all three. Claude sometimes returns complete, valid JSON
containing only the non-empty category (e.g. `{"TP": [...]}`), omitting the empty `FP`/`FN`
arrays; the all-required schema then rejects it and, after retries, the cell dies as a
`scoring_failure`.

The safe fix is NOT to default the missing fields to `[]` in the parser — an omitted `FN`
(ground-truth statements absent from the answer) is not reliably empty, and coercing it would
systematically inflate recall to 1.0 (adversarial review, KNOWN_ISSUES ISSUE-9). Instead we
address the cause: append an explicit completeness directive to the classifier's instruction so
the model reliably emits all three keys (using `[]` only for genuinely empty categories) and
actually performs the ground-truth→answer pass needed to populate `FN`. If the model still
omits despite this, the cell remains an honest NA — we never manufacture a score.

`apply()` mutates the class-level `instruction` of `CorrectnessClassifier` (read dynamically at
prompt-build time, so it reaches every metric instance created afterward). Idempotent; called
once from `ragas_eval/__init__.py` after the vertexai compat shim.


PORTED VERBATIM from ../Ragas_Skill_Ministry_PSQ_Chatbot/ragas_eval/ (Session 06 spec:
"Port the patches; don't rediscover the bugs"). Sample ids such as PQ-02 in the notes below
refer to that project's dataset; the ragas 0.4.3 bugs they describe are version bugs, not
dataset-specific ones, and PARIVESH's longer answers make them MORE likely, not less.
"""
from __future__ import annotations

import logging

_log = logging.getLogger(__name__)

_PATCHED = False

# Marker so we never double-append (idempotency across repeated apply() calls).
_MARKER = "Your JSON output must always include all three keys"

_COMPLETENESS_DIRECTIVE = (
    " " + _MARKER + " — \"TP\", \"FP\" and \"FN\" — each as a JSON array, and use an empty "
    "array [] for any category that has no statements (never omit a key). In particular, before "
    "finishing, review every ground truth statement and add to \"FN\" each ground truth "
    "statement that is not present in the answer; only use an empty FN array if every ground "
    "truth statement is genuinely covered by the answer."
)


def apply() -> bool:
    """Append the completeness directive to `CorrectnessClassifier.instruction`. Returns True if
    applied, False if already applied / unavailable (idempotent)."""
    global _PATCHED
    if _PATCHED:
        return False
    try:
        from ragas.metrics._answer_correctness import CorrectnessClassifier
    except Exception as e:                      # pragma: no cover - import guarded
        _log.warning("ISSUE-9 prompt patch skipped: could not import CorrectnessClassifier (%s)", e)
        return False

    if _MARKER not in (CorrectnessClassifier.instruction or ""):
        CorrectnessClassifier.instruction = CorrectnessClassifier.instruction.rstrip() + _COMPLETENESS_DIRECTIVE
        _log.info("ISSUE-9 prompt patch applied: CorrectnessClassifier instruction hardened to "
                  "require complete TP/FP/FN output.")
    _PATCHED = True
    return True
