"""Two-call fallback for RAGAS 0.4.3 answer_correctness when the single-call TP/FP/FN classifier
won't produce a complete classification (ISSUE-9 follow-up, see KNOWN_ISSUES.md).

Background
----------
`answer_correctness` decomposes the response and the reference into statement lists, then makes a
SINGLE `CorrectnessClassifier` call that must return all three of TP/FP/FN
(`ClassificationWithReason`, all-required). On some samples (confirmed live on PQ-02) Claude
returns a complete, valid classification containing only `TP` — it never performs the
ground-truth→answer pass that populates `FN`. Prompt hardening (`_ragas_prompt_patch.py`) reduces
this but does not eliminate it; for PQ-02 the model omitted `FN` across all rounds, so the cell
was an NA. PQ-02's answer genuinely omits many reference facts, so the true `FN` is large and the
correct factual score is well below 1.0 — defaulting `FN=[]` would inflate recall to 1.0, so we
must *actually compute* `FN`.

The fix
-------
Patch `CorrectnessClassifier.generate` so that, if the standard single call fails
(`RagasOutputParserException`), it falls back to a robust **two-call** classification:

  * Call 1 — for each ANSWER statement (referenced by its 0-based INDEX), return whether it is
    supported (TP) or not (FP).
  * Call 2 — return the 0-based INDICES of GROUND-TRUTH statements NOT present in the answer (FN).

Both outputs are **compact index lists** — no verbatim statement text or reasons are echoed —
because RAGAS's scorer (`_compute_statement_presence` → `fbeta_score(len(TP),len(FP),len(FN))`)
only needs the COUNTS. This keeps each sub-call's output tiny, which (a) avoids the max_tokens
truncation that a full statement-echoing classification hits on long answers, and (b) removes any
multi-field "empty category" the model could omit. `_combine` maps the indices back to
`StatementsWithReason` objects (statements filled in, reason left blank) so RAGAS scores it
unchanged — a *real* FN (non-empty when the answer is genuinely incomplete), never fabricated.

Fires ONLY on failure (success path byte-for-byte unchanged). The primary attempt is capped to
`_PRIMARY_FIX_RETRIES` fix-rounds so its slow, never-helpful-for-omission retries don't starve the
fallback of the per-cell timeout. If the fallback still can't produce a usable classification
(sub-call fails, or the result is degenerate) the exception propagates → the cell stays an honest
NA. Every fallback logs a warning with the resulting TP/FP/FN counts (auditable).

Semantic fidelity: RAGAS counts TP/FP over ANSWER statements and FN over GROUND-TRUTH statements
as INDEPENDENT populations. The index split preserves exactly that. Few-shot examples are RAGAS's
own (sun + boiling-point), expressed as indices; the boiling-point one shows a *covered*
ground-truth statement is NOT an FN (guards against over-listing / deflation).

`apply()` is idempotent; called once from `ragas_eval/__init__.py` after the vertexai shim.


PORTED VERBATIM from ../Ragas_Skill_Ministry_PSQ_Chatbot/ragas_eval/ (Session 06 spec:
"Port the patches; don't rediscover the bugs"). Sample ids such as PQ-02 in the notes below
refer to that project's dataset; the ragas 0.4.3 bugs they describe are version bugs, not
dataset-specific ones, and PARIVESH's longer answers make them MORE likely, not less.
"""
from __future__ import annotations

import logging
import typing as t

from pydantic import BaseModel, Field

_log = logging.getLogger(__name__)

_PATCHED = False

# Cap on the primary single-call classifier's fix_output_format retries before dropping to the
# two-call fallback. Kept low because those retries never recover the FN-omission this patch
# targets and each is a slow throttled Bedrock call that would starve the fallback of the
# per-cell timeout budget. 1 = give genuinely-transient malformed JSON one fix shot on the
# standard path; persistent failures fall through to the fallback immediately.
_PRIMARY_FIX_RETRIES = 1


# --- sub-call schemas: compact INDEX-based outputs (tiny → no truncation, no omissible field) ---
class _AnswerLabel(BaseModel):
    i: int = Field(..., description="0-based index of the answer statement")
    supported: bool = Field(..., description="true if this answer statement is directly supported "
                                            "by >=1 ground truth statement (TP); false otherwise (FP)")


class _AnswerLabelsOutput(BaseModel):
    labels: t.List[_AnswerLabel] = Field(
        ..., description="exactly one entry per answer statement, referenced by 0-based index")


class _ClassifyAnswerInput(BaseModel):
    question: str
    answer: t.List[str]
    ground_truth: t.List[str]


class _FalseNegativesOutput(BaseModel):
    uncovered_indices: t.List[int] = Field(
        ..., description="0-based indices of ground truth statements NOT present in the answer "
                         "(false negatives); empty only if every ground truth statement is covered")


class _FalseNegativesInput(BaseModel):
    question: str
    answer: t.List[str]
    ground_truth: t.List[str]


def _clean_indices(idxs, n: int):
    """Dedupe (preserve order) and keep only indices in range [0, n)."""
    seen = set()
    out = []
    for i in idxs:
        if isinstance(i, bool):            # bools are ints in python; reject them
            continue
        if isinstance(i, int) and 0 <= i < n and i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _tp_fp_indices(labels, n_answer: int):
    """Pure helper: from index labels, return (tp_indices, fp_indices), deduped by index
    (first occurrence wins) and clamped to valid answer indices."""
    tp, fp, seen = [], [], set()
    for l in labels:
        i = getattr(l, "i", None)
        if isinstance(i, bool) or not isinstance(i, int):
            continue
        if not (0 <= i < n_answer) or i in seen:
            continue
        seen.add(i)
        (tp if l.supported else fp).append(i)
    return tp, fp


# apply() resolves ragas classes/prompts lazily (needs the vertexai shim first) and stores them
# here so module-level helpers/override can use them and tests can monkeypatch them.
_classify_answer_prompt = None
_false_negatives_prompt = None
_ClassificationWithReason = None
_StatementsWithReason = None
_RagasOutputParserException = None
_orig_generate = None

_FALLBACK_REASON = "(recovered via two-call index-based fallback)"


def _is_degenerate_classification(result, answer_nonempty: bool) -> bool:
    """True if the combined classification carries no usable information, so scoring it would
    produce a spurious value (e.g. fbeta(0,0,0)=0.0) rather than an honest NA:
      * nothing classified at all (TP=FP=FN all empty), or
      * the answer HAD statements to label but Call 1 labelled none (TP+FP==0) — a Call-1
        malfunction, which must not silently become a 0.0.
    A genuinely empty answer (no answer statements) with real FN is NOT degenerate — recall 0 /
    a low score is the correct outcome there, so it is allowed through."""
    n_tp, n_fp, n_fn = len(result.TP), len(result.FP), len(result.FN)
    if n_tp + n_fp + n_fn == 0:
        return True
    if answer_nonempty and (n_tp + n_fp) == 0:
        return True
    return False


def _combine(answer, ground_truth, labels_out, fn_out):
    """Map the two sub-call index outputs back to a ragas `ClassificationWithReason`. TP/FP come
    from the answer-labeling pass (indices into `answer`); FN from the ground-truth-coverage pass
    (indices into `ground_truth`). Only counts matter for scoring; statements are filled in for
    fidelity, reason left as a marker."""
    tp_idx, fp_idx = _tp_fp_indices(labels_out.labels, len(answer))
    fn_idx = _clean_indices(fn_out.uncovered_indices, len(ground_truth))
    tp = [_StatementsWithReason(statement=answer[i], reason=_FALLBACK_REASON) for i in tp_idx]
    fp = [_StatementsWithReason(statement=answer[i], reason=_FALLBACK_REASON) for i in fp_idx]
    fn = [_StatementsWithReason(statement=ground_truth[i], reason=_FALLBACK_REASON) for i in fn_idx]
    return _ClassificationWithReason(TP=tp, FP=fp, FN=fn)


async def _two_call_classify(llm, data, callbacks):
    """Recompute TP/FP/FN via two focused, compact index sub-calls; combine into a
    ClassificationWithReason. Raises (→ honest NA) if a sub-call fails or the result is
    degenerate — never fabricates a score."""
    answer = list(data.answer)
    ground_truth = list(data.ground_truth)
    labels_out = await _classify_answer_prompt.generate(
        llm=llm,
        data=_ClassifyAnswerInput(question=data.question, answer=answer, ground_truth=ground_truth),
        callbacks=callbacks,
    )
    fn_out = await _false_negatives_prompt.generate(
        llm=llm,
        data=_FalseNegativesInput(question=data.question, answer=answer, ground_truth=ground_truth),
        callbacks=callbacks,
    )
    result = _combine(answer, ground_truth, labels_out, fn_out)
    if _is_degenerate_classification(result, answer_nonempty=len(answer) > 0):
        _log.warning(
            "answer_correctness two-call fallback produced a DEGENERATE classification "
            "(TP=%d FP=%d FN=%d, answer_nonempty=%s) — refusing to emit a spurious score; cell "
            "stays an honest NA.", len(result.TP), len(result.FP), len(result.FN), len(answer) > 0)
        raise _RagasOutputParserException()
    _log.warning(
        "answer_correctness two-call fallback fired: single-call classifier did not return a "
        "complete TP/FP/FN; recomputed via compact index-based answer-labeling + "
        "ground-truth-coverage passes -> TP=%d FP=%d FN=%d.",
        len(result.TP), len(result.FP), len(result.FN))
    return result


def apply() -> bool:
    """Patch `CorrectnessClassifier.generate` with the two-call fallback. Idempotent."""
    global _PATCHED, _classify_answer_prompt, _false_negatives_prompt
    global _ClassificationWithReason, _StatementsWithReason, _RagasOutputParserException, _orig_generate
    if _PATCHED:
        return False
    try:
        from ragas.exceptions import RagasOutputParserException
        from ragas.metrics._answer_correctness import (ClassificationWithReason,
                                                       CorrectnessClassifier, StatementsWithReason)
        from ragas.prompt import PydanticPrompt
    except Exception as e:                                    # pragma: no cover
        _log.warning("answer_correctness two-call fallback skipped: import failed (%s)", e)
        return False

    _ClassificationWithReason = ClassificationWithReason
    _StatementsWithReason = StatementsWithReason
    _RagasOutputParserException = RagasOutputParserException

    class _ClassifyAnswerPrompt(PydanticPrompt[_ClassifyAnswerInput, _AnswerLabelsOutput]):
        instruction = (
            "You are given a question, a list of ANSWER statements, and a list of GROUND TRUTH "
            "statements. The answer statements are indexed from 0 by their position in the "
            "'answer' array. For EVERY answer statement, decide whether it is directly supported "
            "by one or more of the ground truth statements. Return one label {i, supported} per "
            "answer statement, where i is its 0-based index and supported is true (a true "
            "positive) or false (present in the answer but not supported — a false positive). "
            "Return only indices and booleans; do not echo the statement text.")
        input_model = _ClassifyAnswerInput
        output_model = _AnswerLabelsOutput
        examples = [
            (
                _ClassifyAnswerInput(
                    question="What powers the sun and what is its primary function?",
                    answer=[
                        "The sun is powered by nuclear fission, similar to nuclear reactors on Earth.",
                        "The primary function of the sun is to provide light to the solar system.",
                    ],
                    ground_truth=[
                        "The sun is powered by nuclear fusion, where hydrogen atoms fuse to form helium.",
                        "This fusion process in the sun's core releases a tremendous amount of energy.",
                        "The energy from the sun provides heat and light, which are essential for life on Earth.",
                    ],
                ),
                _AnswerLabelsOutput(labels=[
                    _AnswerLabel(i=0, supported=False),   # fission — contradicts ground truth (fusion)
                    _AnswerLabel(i=1, supported=True),    # light — supported
                ]),
            ),
            (
                _ClassifyAnswerInput(
                    question="What is the boiling point of water?",
                    answer=["The boiling point of water is 100 degrees Celsius at sea level."],
                    ground_truth=[
                        "The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at sea level.",
                        "The boiling point of water can change with altitude.",
                    ],
                ),
                _AnswerLabelsOutput(labels=[_AnswerLabel(i=0, supported=True)]),
            ),
        ]

    class _FalseNegativesPrompt(PydanticPrompt[_FalseNegativesInput, _FalseNegativesOutput]):
        instruction = (
            "You are given a question, a list of ANSWER statements, and a list of GROUND TRUTH "
            "statements. The ground truth statements are indexed from 0 by their position in the "
            "'ground_truth' array. Examine EACH ground truth statement and determine whether it "
            "is present in (covered by) the answer statements. Return uncovered_indices: the "
            "0-based indices of the ground truth statements that are NOT present in the answer "
            "(these are false negatives). Return an empty list only if every ground truth "
            "statement is genuinely covered by the answer. Return only indices; do not echo text.")
        input_model = _FalseNegativesInput
        output_model = _FalseNegativesOutput
        examples = [
            (
                _FalseNegativesInput(
                    question="What powers the sun and what is its primary function?",
                    answer=[
                        "The sun is powered by nuclear fission, similar to nuclear reactors on Earth.",
                        "The primary function of the sun is to provide light to the solar system.",
                    ],
                    ground_truth=[
                        "The sun is powered by nuclear fusion, where hydrogen atoms fuse to form helium.",
                        "This fusion process in the sun's core releases a tremendous amount of energy.",
                        "The energy from the sun provides heat and light, which are essential for life on Earth.",
                    ],
                ),
                # all three ground-truth statements are absent from the answer
                _FalseNegativesOutput(uncovered_indices=[0, 1, 2]),
            ),
            (
                _FalseNegativesInput(
                    question="What is the boiling point of water?",
                    answer=["The boiling point of water is 100 degrees Celsius at sea level."],
                    ground_truth=[
                        "The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at sea level.",
                        "The boiling point of water can change with altitude.",
                    ],
                ),
                # GT[0] (boiling point) IS covered → NOT an FN; only GT[1] (altitude) is uncovered
                _FalseNegativesOutput(uncovered_indices=[1]),
            ),
        ]

    _classify_answer_prompt = _ClassifyAnswerPrompt()
    _false_negatives_prompt = _FalseNegativesPrompt()
    _orig_generate = PydanticPrompt.generate       # unbound; the standard single-call path

    async def _cc_generate(self, llm, data, temperature=None, stop=None, callbacks=None,
                           retries_left: int = 3):
        primary_retries = min(retries_left, _PRIMARY_FIX_RETRIES)
        try:
            return await _orig_generate(self, llm, data, temperature=temperature, stop=stop,
                                        callbacks=callbacks, retries_left=primary_retries)
        except RagasOutputParserException:
            return await _two_call_classify(llm, data, callbacks)

    CorrectnessClassifier.generate = _cc_generate
    _PATCHED = True
    return True
