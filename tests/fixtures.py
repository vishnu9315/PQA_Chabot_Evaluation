"""Synthetic fixtures for the no-API tests. Nothing here touches the real artifacts."""
from __future__ import annotations

from ragas_eval.dataset import Sample


def make_sample(sid="GD-0900", *, behavior="answer_full", n_ctx=3, eligible=True,
                cited=None, expected=None, facts=None, aspects=None,
                response="An official reply with the figure ₹1,234 crore.",
                reference="The reply states the figure is ₹1,234 crore.") -> Sample:
    return Sample(
        id=sid,
        user_input=f"the details of scheme X for {sid}?",
        response=response,
        retrieved_contexts=[f"chunk {i} for {sid}" for i in range(n_ctx)],
        reference=reference,
        reference_contexts=[f"reference span {i} for {sid}" for i in range(2)],
        expected_behavior=behavior,
        stratum="single_doc_lookup", category="Trade", difficulty="medium",
        house="ls", question_origin="verbatim_past",
        must_include_facts=facts if facts is not None else
            [{"fact": "the figure is ₹1,234 crore", "weight": "core"}],
        expected_sources=expected if expected is not None else
            [{"document": "ls_unstarred_1.md", "necessity": "required"},
             {"document": "rs_unstarred_2.md", "necessity": "acceptable"}],
        cited_sources=cited if cited is not None else ["ls_unstarred_1.md"],
        ragas_eligible=eligible,
        unsupported_aspects=aspects or [],
    )


class FakeCfg:
    """Minimal config stand-in for pure-function tests (no .env, no AWS)."""
    qa_pass = 0.80
    segments = ("provenance", "expected_behavior", "difficulty", "stratum",
                "category", "house")
    rate_llm_in_per_1k = 0.003
    rate_llm_out_per_1k = 0.015
    rate_embed_per_1k = 0.00002
    bedrock_model_id = "global.anthropic.claude-sonnet-5"
    bedrock_embed_model_id = "amazon.titan-embed-text-v2:0"
    aws_region = "ap-south-1"
    thinking_mode = "adaptive"
    thinking_effort = "medium"
    thinking_budget = 24576
    ragas_max_tokens = 64000
    ragas_timeout = 1800
    per_chunk_timeout = 300
    ragas_retry_rounds = 3
    judge_retry_rounds = 2
    stability_tolerance = 0.05
    cell_concurrency = 4
    executor_threads = 8
    judge_max_workers = 2
    pace_concurrency_start = 2
    pace_concurrency_max = 4
    pace_concurrency_min = 1
    pace_rpm = 600.0
    pace_tpm = 10_000_000.0
    pace_aimd_successes = 2
    pace_cooldown_s = 0.01
    pace_max_backoff_s = 0.05
    pace_max_attempts = 5
