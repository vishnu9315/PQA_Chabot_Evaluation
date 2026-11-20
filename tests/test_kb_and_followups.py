"""KB index (documents/ -> chatbot citation ids), expected-source resolution, and follow-up
context for -F2 rows. Synthetic corpus in tmp_path; no API calls."""
from dataclasses import replace

from ragas_eval import judge_runner as J
from ragas_eval.attribution import build_hint
from ragas_eval.kb_index import load_kb_index
from tests.fixtures import make_sample

_META_HEADER = "house,house_no,session,q_no,qtype,starred,date,subject,members,ministry,filename\n"


def _corpus(tmp_path):
    ls = tmp_path / "Lok Sabha" / "Lok Sabha 18" / "Session VIII"
    rs = tmp_path / "Rajya Sabha" / "Session 270"
    for d in (ls, rs):
        d.mkdir(parents=True)
    (ls / "AU2878_rN86Hc.pdf").write_bytes(b"%PDF")
    (rs / "AS239_CYnH3S.pdf").write_bytes(b"%PDF")
    (ls / "metadata.csv").write_text(
        _META_HEADER + "Lok Sabha,18,8,2878,UNSTARRED,False,2026-08-05,s,m,MEITY,AU2878_rN86Hc.pdf\n",
        encoding="utf-8")
    (rs / "metadata.csv").write_text(
        _META_HEADER + "Rajya Sabha,,270,239,STARRED,True,2026-03-01,s,m,MEITY,AS239_CYnH3S.pdf\n",
        encoding="utf-8")
    return tmp_path


def test_index_uses_the_chatbots_citation_spelling(tmp_path):
    idx = load_kb_index(_corpus(tmp_path))
    assert idx.names == {"lok_sabha/lok_sabha_18/Session_VIII/AU2878_rN86Hc.md",
                         "rajya_sabha/Session_270/AS239_CYnH3S.md"}


def test_expected_sources_resolve_through_metadata(tmp_path):
    idx = load_kb_index(_corpus(tmp_path))
    ls = {"document": "Lok Sabha/.../Unstarred Question_2878_050826.docx",
          "session": "ls18-viii", "qno": 2878, "parl_type": "unstarred"}
    rs = {"document": "Rajya Sabha/.../AS239.pdf", "session": "rs-270", "qno": 239,
          "parl_type": "starred"}
    assert idx.resolve(ls) == "lok_sabha/lok_sabha_18/Session_VIII/AU2878_rN86Hc.md"
    assert idx.resolve(rs) == "rajya_sabha/Session_270/AS239_CYnH3S.md"


def test_unresolvable_expected_source_returns_none(tmp_path):
    idx = load_kb_index(_corpus(tmp_path))
    assert idx.resolve({"session": "rs-271", "qno": 713, "parl_type": "unstarred"}) is None
    assert idx.resolve({"document": "x.md"}) is None


def test_missing_documents_dir_means_no_kb(tmp_path):
    assert load_kb_index(tmp_path / "absent") is None


def test_resolved_citation_counts_as_hit_and_not_fabricated(tmp_path):
    idx = load_kb_index(_corpus(tmp_path))
    cid = "lok_sabha/lok_sabha_18/Session_VIII/AU2878_rN86Hc.md"
    s = make_sample(cited=[cid], expected=[{"document": cid, "necessity": "required"}])
    h = build_hint(s, idx.names)
    assert h["required_cited_directly"] == [cid]
    assert h["fabricated_not_in_kb"] == []


def _followup():
    first = make_sample("GD-X-F1")
    second = make_sample("GD-X-F2")
    return replace(second, follow_up_of=first.id,
                   prior_turn={"id": first.id, "user_input": "How many sites were defaced?",
                               "response": "12 sites."})


def test_followup_ragas_input_carries_the_earlier_question():
    s = _followup()
    ui = s.to_ragas_kwargs()["user_input"]
    assert "How many sites were defaced?" in ui and s.user_input in ui
    assert make_sample().to_ragas_kwargs()["user_input"] == make_sample().user_input


def test_followup_judges_see_the_earlier_turn_but_sources_judge_does_not():
    item = _followup().judge_item()
    for canon in ("completeness", "response_quality"):
        names = [fn for fn, _ in J.fields_for(canon, item, None)]
        assert names[0].startswith("earlier_turn_in_this_conversation")
    names = [fn for fn, _ in J.fields_for("sources_validation", item, {})]
    assert not any(n.startswith("earlier_turn") for n in names)
    plain = [fn for fn, _ in J.fields_for("completeness", make_sample().judge_item(), None)]
    assert not any(n.startswith("earlier_turn") for n in plain)


def test_part2_block_reports_readme_section_4_4_items():
    from ragas_eval.report import part2_block

    def rec(rid, part, faith, resp="Figures as of March 2026 are 15.", cited=()):
        return {"id": rid, "provenance": part, "response": resp, "cited_sources": list(cited),
                "metrics": {"faithfulness": {"score": faith}, "completeness": {"score": 0.5}}}
    recs = [rec("P1", "part1-ingested", 0.9),
            rec("P2-ok", "part2-not-ingested", 0.95),
            rec("P2-bad", "part2-not-ingested", 0.4),
            rec("P2-refuse", "part2-not-ingested", None,
                resp="I could not find an answer to that question.")]
    text = "\n".join(part2_block(recs))
    assert "1/2 — P2-bad" in text          # refusal excluded from the answered-row count
    assert "1/3 — P2-refuse" in text
    assert "none — Session 271 is absent" in text
    cited = recs + [rec("P2-271", "part2-not-ingested", 0.9,
                        cited=["rajya_sabha/Session_271/AU713_x.md"])]
    assert "YES on P2-271" in "\n".join(part2_block(cited))
    assert part2_block([recs[0]]) == []
