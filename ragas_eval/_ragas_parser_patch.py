"""Robustness patch for `ragas.prompt.pydantic_prompt.RagasOutputParser.parse_output_string`
(RAGAS 0.4.3) against a Claude-on-Bedrock output habit (ISSUE-8, see KNOWN_ISSUES.md).

ISSUE-8 — fix_output_format's `StringIO` wrapper is dropped by the model
------------------------------------------------------------------------
When a metric's classification parse fails, ragas asks the LLM to repair the output via
`fix_output_format_prompt`, whose output schema is `StringIO` (`{"text": "<fixed JSON>"}`).
Claude routinely returns the corrected object *directly* (e.g. `{"TP": [...], ...}`), not
wrapped in a `text` field, so the repair round itself fails to validate against `StringIO` and
the cell is abandoned as a `scoring_failure`. Fix: when the target schema is `StringIO`, the
offending text is valid, non-degenerate JSON, and it merely lacks a `"text"` key, wrap it
(`{"text": <json>}`) and use it directly. Hardened after adversarial review: refuse the shortcut
for degenerate all-empty JSON, and log every rewrap for auditability.

Related — ISSUE-9 (metric schema rejects omitted *empty* list categories, e.g.
answer_correctness returning only `{"TP": [...]}`) is addressed SEPARATELY in
`_ragas_prompt_patch.py` by hardening the classifier prompt to emit complete TP/FP/FN. An
earlier attempt to fix ISSUE-9 *here* — defaulting omitted required list-fields to `[]` — was
**removed** after adversarial review showed it systematically inflates the score: an omitted
`FN` (ground-truth statements missing from the answer) is NOT reliably "empty", so coercing it
to `[]` forces recall to 1.0 and turns an honest NA into an upper-bound score. We do not
fabricate scores; a classification the model won't complete stays an honest NA.

`apply()` is idempotent and called once from `ragas_eval/__init__.py`, after the vertexai
compat shim (`_compat.apply()`), since this module imports `ragas.prompt.pydantic_prompt`,
which imports `ragas.llms.base`, which needs that shim in place first.


PORTED VERBATIM from ../Ragas_Skill_Ministry_PSQ_Chatbot/ragas_eval/ (Session 06 spec:
"Port the patches; don't rediscover the bugs"). Sample ids such as PQ-02 in the notes below
refer to that project's dataset; the ragas 0.4.3 bugs they describe are version bugs, not
dataset-specific ones, and PARIVESH's longer answers make them MORE likely, not less.
"""
from __future__ import annotations

import logging

_log = logging.getLogger(__name__)

_PATCHED = False


def _is_degenerate(obj) -> bool:
    """True if `obj` (recursively) contains no non-empty/non-falsy value anywhere — i.e. it's
    entirely made of "", None, 0, False, [] and {}. Used to refuse auto-accepting a response
    that carries no actual information (a total give-up)."""
    if isinstance(obj, dict):
        return all(_is_degenerate(v) for v in obj.values())
    if isinstance(obj, list):
        return all(_is_degenerate(v) for v in obj)
    return not obj


def apply() -> bool:
    """Patch `RagasOutputParser.parse_output_string`. Returns True if applied, False if
    already patched (idempotent)."""
    global _PATCHED
    if _PATCHED:
        return False

    import json

    from langchain_core.exceptions import OutputParserException
    from ragas.callbacks import new_group
    from ragas.exceptions import RagasOutputParserException
    from ragas.prompt.base import StringIO
    from ragas.prompt.pydantic_prompt import (OutputStringAndPrompt, RagasOutputParser,
                                              fix_output_format_prompt)
    from ragas.prompt.utils import extract_json

    def _try_rewrap_as_stringio(output_string: str):
        """ISSUE-8: if `output_string` is valid, non-degenerate JSON without a top-level
        "text" key, return it wrapped as a `StringIO` (and log). Returns None (never raises)
        otherwise, so the caller falls through to ragas's normal fix/retry path unchanged."""
        try:
            jsonstr = extract_json(output_string)
            obj = json.loads(jsonstr)
        except Exception:
            return None
        if not isinstance(obj, dict) or "text" in obj:
            return None
        if _is_degenerate(obj):
            return None
        try:
            wrapped = StringIO(text=jsonstr)
        except Exception:
            return None
        _log.warning(
            "ISSUE-8 StringIO-repair patch fired: fix_output_format response was valid JSON "
            "missing the 'text' wrapper key; auto-wrapped and accepted without an extra LLM "
            "round-trip. Recovered payload (truncated): %s", jsonstr[:300])
        return wrapped

    async def _patched_parse_output_string(self, output_string, prompt_value, llm, callbacks,
                                            retries_left: int = 1):
        callbacks = callbacks or []
        try:
            jsonstr = extract_json(output_string)
            result = super(RagasOutputParser, self).parse(jsonstr)
        except OutputParserException:
            # ISSUE-8: the fix_output_format round dropped the StringIO wrapper — recover it
            # (cheap, no LLM) before spending a retry.
            if self.pydantic_object is StringIO:
                rewrapped = _try_rewrap_as_stringio(output_string)
                if rewrapped is not None:
                    return rewrapped
            if retries_left != 0:
                retry_rm, retry_cb = new_group(
                    name="fix_output_format",
                    inputs={"output_string": output_string},
                    callbacks=callbacks,
                )
                fixed_output_string = await fix_output_format_prompt.generate(
                    llm=llm,
                    data=OutputStringAndPrompt(
                        output_string=output_string,
                        prompt_value=prompt_value.to_string(),
                    ),
                    callbacks=retry_cb,
                    retries_left=retries_left - 1,
                )
                retry_rm.on_chain_end({"fixed_output_string": fixed_output_string})
                result = super(RagasOutputParser, self).parse(fixed_output_string.text)
            else:
                raise RagasOutputParserException()
        return result

    RagasOutputParser.parse_output_string = _patched_parse_output_string
    _PATCHED = True
    return True
