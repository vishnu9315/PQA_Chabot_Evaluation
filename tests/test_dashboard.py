"""dashboard.html rendering from synthetic per-sample records. No API calls."""
from dataclasses import replace

from ragas_eval import aggregate as agg
from ragas_eval.dashboard import TITLE, build_html
from tests.fixtures import FakeCfg, make_sample

CID = "lok_sabha/lok_sabha_18/Session_VIII/AU2878_rN86Hc.md"


class _Cache:
    def get(self, sid, metric):
        cell = {"score": 0.9 if metric != "faithfulness" else 0.4}
        if metric == "completeness":
            cell.update(justification="Both core facts present.",
                        detail={"per_fact": [{"fact": "MeitY issued the MTS", "weight": "core",
                                              "verdict": "PRESENT"},
                                             {"fact": "latest revision date", "weight": "core",
                                              "verdict": "ABSENT"}]})
        if metric == "sources_validation":
            cell.update(justification="Required source cited.",
                        detail={"recall": 1.0, "precision": 0.5})
        return cell


def _records():
    p1 = replace(make_sample("GD-MTY-A-F1", cited=[CID, "rajya_sabha/Session_260/AS180.md"],
                             expected=[{"document": CID, "necessity": "required",
                                        "document_as_authored": "Unstarred Question_2878.docx"}],
                             response="<script>alert(1)</script> The MTS was issued."),
                 part="1-ingested", query_style="Follow-up")
    p2 = replace(make_sample("GD-MTY-A-F2", response="I could not find an answer to that question."),
                 part="2-not-ingested", query_style="Follow-up",
                 follow_up_of=p1.id, prior_turn={"id": p1.id, "user_input": "Earlier Q?",
                                                 "response": "x"})
    return agg.build_per_sample([p1, p2], _Cache())


def _cfg():
    cfg = FakeCfg()
    cfg.segments = ("provenance", "expected_behavior")
    return cfg


def test_dashboard_is_shareable_and_carries_row_detail():
    out = build_html(_records(), _cfg(), run_meta={"run_id": "run_x"})
    assert out.startswith(f"<title>{TITLE}</title>")
    assert 'prefers-color-scheme:dark' in out and ':root[data-theme="dark"]' in out
    assert '<details class="row" id="GD-MTY-A-F1">' in out and 'href="#GD-MTY-A-F1"' in out
    assert "Earlier question in the same chat" in out and "Earlier Q?" in out
    assert "authored as Unstarred Question_2878.docx" in out
    assert '<span class="chip good">cited</span>' in out
    assert "Also listed by the chatbot (1)" in out
    assert '<span class="chip bad">absent</span>' in out
    assert "Completeness (0.90): Both core facts present." in out
    assert "recall 1.00, precision 0.50" in out
    assert "Part 2 (not ingested)" in out and "Reading notes" in out


def test_dashboard_escapes_chatbot_text():
    out = build_html(_records(), _cfg(), run_meta={"run_id": "run_x"})
    assert "<script>alert(1)</script>" not in out
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in out
