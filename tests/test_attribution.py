"""Deterministic sources matcher. No API calls."""
from ragas_eval.attribution import build_hint
from tests.fixtures import make_sample

KB = {"ls_unstarred_1.md", "rs_unstarred_2.md", "ls_starred_3.md"}


def test_required_and_acceptable_split():
    s = make_sample(cited=["ls_unstarred_1.md"])
    h = build_hint(s, KB)
    assert h["expected_required"] == ["ls_unstarred_1.md"]
    assert h["expected_acceptable"] == ["rs_unstarred_2.md"]
    assert h["required_cited_directly"] == ["ls_unstarred_1.md"]
    assert h["fabricated_not_in_kb"] == []


def test_fabrication_is_membership_in_the_union_kb():
    s = make_sample(cited=["ghost_9999.md", "ls_unstarred_1.md"])
    h = build_hint(s, KB)
    assert h["fabricated_not_in_kb"] == ["ghost_9999.md"]


def test_acceptable_twin_cited_is_not_unexpected():
    s = make_sample(cited=["rs_unstarred_2.md"])
    h = build_hint(s, KB)
    assert h["acceptable_cited"] == ["rs_unstarred_2.md"]
    assert h["cited_but_not_expected"] == []
    assert h["required_not_cited_directly"] == ["ls_unstarred_1.md"]


def test_names_normalised_to_md():
    s = make_sample(cited=["ls_unstarred_1"])
    h = build_hint(s, KB)
    assert h["cited_sources"] == ["ls_unstarred_1.md"]
