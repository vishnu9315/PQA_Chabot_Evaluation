"""Judge runner mechanics — verbatim prompt extraction, ISSUE-1/6 fixes, leak-proofing,
score coercion. No API calls."""
import pytest

from config import load_config
from ragas_eval import judge_runner as J
from tests.fixtures import make_sample


# --- verbatim rubric extraction ----------------------------------------------
def test_rubrics_parse_verbatim_from_the_frozen_file():
    cfg = load_config()
    md = cfg.judge_prompts_md.read_text(encoding="utf-8")
    rubrics = J.parse_judge_prompts(md)
    assert set(rubrics) == set(J.JUDGE_METRICS)
    # verbatim: the extracted text must be a substring of the file, byte-for-byte
    for canon, text in rubrics.items():
        assert text in md, f"{canon} rubric is not verbatim"
    # the dropped Hallucination judge is not extracted
    assert "hallucination" not in rubrics
    # spot anchors from the frozen prompts
    assert "PRESENT, PARTIAL or ABSENT" in rubrics["completeness"]
    assert "SATISFIES" in rubrics["sources_validation"]
    assert "Register" in rubrics["response_quality"]


def test_parse_raises_when_a_section_is_missing():
    with pytest.raises(ValueError, match="missing rubric"):
        J.parse_judge_prompts("## Completeness\n\n```\nx\n```\n")


# --- ISSUE-1: robust JSON extraction ------------------------------------------
def test_extract_json_survives_brace_in_reasoning():
    text = 'Reasoning: the set {a, b} matters.\n{"score": 0.75, "justification": "ok"}'
    assert J._extract_json(text)["score"] == 0.75


def test_extract_json_strips_fences():
    text = '```json\n{"score": 1.0, "justification": "x"}\n```'
    assert J._extract_json(text)["score"] == 1.0


def test_extract_json_prefers_the_verdict_object():
    text = '{"note": "prose"} then the verdict {"recall": 0.5, "precision": 1.0}'
    assert "recall" in J._extract_json(text)


def test_extract_json_raises_on_no_object():
    with pytest.raises(ValueError):
        J._extract_json("no json here at all")


# --- ISSUE-6: content is a list of blocks --------------------------------------
def test_content_to_text_drops_reasoning_blocks_and_survives_apostrophes():
    content = [
        {"type": "reasoning_content", "reasoning_content": {"text": "thinking… India's"}},
        {"type": "text", "text": '{"score": 0.5, "justification": "India\'s reply"}'},
    ]
    text = J._content_to_text(content)
    assert "reasoning" not in text.lower() or "India's reply" in text
    assert J._extract_json(text)["score"] == 0.5
    # the old str(list) path would have produced a repr that cannot parse
    with pytest.raises(ValueError):
        J._extract_json(str(content).replace('"', ""))


# --- leak-proof field views -----------------------------------------------------
def test_completeness_judge_never_sees_reference_or_sources():
    s = make_sample()
    names = [fn for fn, _ in J.fields_for("completeness", s.judge_item(), None)]
    assert "response" in names and "must_include_facts" in names
    assert not any("reference" in n for n in names)
    assert not any("sources" in n for n in names)


def test_rq_sees_reference_only_for_partial_rows_without_aspects():
    partial_no_aspects = make_sample(behavior="answer_partial")
    names = [fn for fn, _ in J.fields_for("response_quality",
                                          partial_no_aspects.judge_item(), None)]
    assert any("reference" in n for n in names)

    partial_with_aspects = make_sample(behavior="answer_partial",
                                       aspects=["State-wise split not itemised"])
    names = [fn for fn, _ in J.fields_for("response_quality",
                                          partial_with_aspects.judge_item(), None)]
    assert not any("reference" in n for n in names)

    full = make_sample(behavior="answer_full")
    names = [fn for fn, _ in J.fields_for("response_quality", full.judge_item(), None)]
    assert not any("reference" in n for n in names)


def test_assemble_prompt_carries_the_verbatim_rubric_and_the_with_kb_amendment():
    """fabricated_not_in_kb=[] (not None) means a KB corpus IS configured and simply
    found nothing fabricated - the judge gets the "KB is authoritative" amendment."""
    s = make_sample()
    prompt = J.assemble_prompt("sources_validation", "RUBRIC-SENTINEL", s.judge_item(),
                               {"fabricated_not_in_kb": []})
    assert "RUBRIC-SENTINEL" in prompt
    assert "authoritative" in prompt               # the KB-is-configured amendment
    assert "OUTPUT (JSON only):" in prompt


def test_assemble_prompt_carries_the_no_kb_amendment_when_kb_is_unconfigured():
    """fabricated_not_in_kb=None means no local KB corpus is configured at all - the
    judge must be told NOT to zero the score on that basis (see attribution.py)."""
    s = make_sample()
    prompt = J.assemble_prompt("sources_validation", "RUBRIC-SENTINEL", s.judge_item(),
                               {"fabricated_not_in_kb": None})
    assert "RUBRIC-SENTINEL" in prompt
    assert "no local KB document corpus is configured" in prompt
    assert "do not zero the score" in prompt.lower()


# --- score coercion --------------------------------------------------------------
def test_sources_fabrication_forces_zero_even_if_judge_disagrees():
    hint = {"fabricated_not_in_kb": ["ghost_document.md"]}
    rec = J.coerce({"recall": 1.0, "precision": 1.0, "fabrication": False},
                   "sources_validation", "GD-X", hint)
    assert rec["score"] == 0.0
    assert rec["detail"]["fabrication"] is True


def test_sources_score_is_mean_of_recall_and_precision():
    rec = J.coerce({"recall": 1.0, "precision": 0.5, "fabrication": False},
                   "sources_validation", "GD-X", {"fabricated_not_in_kb": []})
    assert rec["score"] == 0.75


def test_rq_mean_is_recomputed_from_dimensions():
    rec = J.coerce({"dimensions": {"register": 1, "structure": 1, "behaviour": 0,
                                   "figures": 0}, "score": 0.99},
                   "response_quality", "GD-X", None)
    assert rec["score"] == 0.5                      # never trusts the reported mean


def test_out_of_range_score_becomes_na():
    rec = J.coerce({"score": 3.7}, "completeness", "GD-X", None)
    assert rec["score"] is None
    assert rec["na_reason"] == "unparseable_or_out_of_range_score"
