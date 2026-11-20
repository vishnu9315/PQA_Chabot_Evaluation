"""Cost estimation and measured actuals — PER PHASE (RAGAS vs judge), never gated.

No code path changes behaviour on spend: there is no budget ceiling, no sampling, no
"skip the retry to save tokens". `--estimate-cost` and `cost.json` exist so the number is
visible, and that is all they do.

The estimate is deliberately coarse and says so. RAGAS metrics are multi-step (claim
decomposition, per-chunk verdicts, NLI) and extended thinking adds an input-dependent
reasoning tail. Reference forensics: B's first estimate came in 3.3x under (ISSUE-3);
A's `_OUT_TOK_PER_CALL=3000` came in 18x OVER on its highest-volume metric and inverted
the direction of alarm (ISSUE-P14). The per-call output constants below are per-metric,
seeded from A's measured shapes at effort=medium, and clearly labelled ROUGH.
"""
from __future__ import annotations

from . import judge_runner as J
from . import ragas_runner as R

# LLM calls per (sample, metric); context_precision is per retrieved chunk.
_CALLS_PER_CELL = {
    "context_recall": 1,
    "faithfulness": 2,          # statement generation + NLI
    "answer_relevancy": 1,      # question generation (+ embeddings)
    "answer_correctness": 3,    # 2 statement decompositions + 1 classification
}
# Output tokens per call, per metric family (ISSUE-P14: one scalar was 18x wrong for the
# highest-volume metric). Chunk verdicts are tiny; classification lists are not.
_OUT_TOK = {"context_precision": 250, "context_recall": 900, "faithfulness": 1200,
            "answer_relevancy": 400, "answer_correctness": 1500}
_JUDGE_OUT_TOK = 700            # verdict JSON + justification, thinking tail at medium


def _tok(text: str) -> float:
    return len(text) / 4.0


def estimate(cfg, samples, ragas_metrics=None, judge_metrics=None) -> dict:
    """Rough, no-API pre-run estimate with a RAGAS/judge phase split."""
    ragas_metrics = ragas_metrics if ragas_metrics is not None else list(R.RAGAS_METRICS)
    judge_metrics = judge_metrics if judge_metrics is not None else list(J.JUDGE_METRICS)

    # --- RAGAS phase (routing-aware: context metrics skip empty-retrieval rows) ---
    r_calls = r_in = r_out = 0.0
    by_metric: dict = {}
    for m in ragas_metrics:
        spec = R.BY_NAME[m]
        eligible = [s for s in samples if s.route != "judge_only"
                    and R.missing_inputs(spec, s) is None]
        if m == "semantic_similarity":
            by_metric[m] = 0
            continue                       # embeddings only
        if m == "context_precision":
            calls = sum(len(s.retrieved_contexts) for s in eligible)
        else:
            calls = _CALLS_PER_CELL[m] * len(eligible)
        by_metric[m] = calls
        r_calls += calls
        for s in eligible:
            base = sum(
                (sum(_tok(c) for c in s.retrieved_contexts) if f == "retrieved_contexts"
                 else _tok(getattr(s, f)))
                for f in spec.reads)
            mult = len(s.retrieved_contexts) if m == "context_precision" else \
                _CALLS_PER_CELL.get(m, 1)
            r_in += base * max(1, mult) * 0.6   # each call re-reads a subset of the fields
        r_out += calls * _OUT_TOK[m]

    # --- judge phase (all 45 rows x 3 judges, 1 call each) ---
    j_calls = j_in = j_out = 0.0
    for s in samples:
        item_tok = (_tok(s.user_input) + _tok(s.response) + _tok(s.reference)
                    + sum(_tok(c) for c in s.retrieved_contexts)
                    + _tok(str(s.must_include_facts)) + _tok(str(s.expected_sources)))
        for m in judge_metrics:
            j_calls += 1
            j_in += item_tok * 0.7 + 900       # rubric + schema + hint overhead
            j_out += _JUDGE_OUT_TOK

    embed_tok = sum(_tok(s.response) + _tok(s.reference) + _tok(s.user_input)
                    for s in samples if s.route != "judge_only") * 2

    def usd(i, o):
        return i / 1000 * cfg.rate_llm_in_per_1k + o / 1000 * cfg.rate_llm_out_per_1k

    ragas_usd, judge_usd = usd(r_in, r_out), usd(j_in, j_out)
    embed_usd = embed_tok / 1000 * cfg.rate_embed_per_1k
    return {
        "basis": ("ROUGH — multi-step metric prompts + a variable extended-thinking tail. "
                  "Informational only; never gates a run."),
        "samples": len(samples),
        "ragas": {"llm_calls": int(r_calls), "calls_by_metric": by_metric,
                  "in_tokens": int(r_in), "out_tokens": int(r_out),
                  "usd": round(ragas_usd, 4)},
        "judge": {"llm_calls": int(j_calls), "in_tokens": int(j_in), "out_tokens": int(j_out),
                  "usd": round(judge_usd, 4)},
        "embed_tokens": int(embed_tok), "embed_usd": round(embed_usd, 6),
        "total_usd": round(ragas_usd + judge_usd + embed_usd, 4),
    }


def actual(cfg, usage: dict, phase: str) -> dict:
    in_tok = int(usage.get("input_tokens", 0))
    out_tok = int(usage.get("output_tokens", 0))
    llm_usd = in_tok / 1000 * cfg.rate_llm_in_per_1k + out_tok / 1000 * cfg.rate_llm_out_per_1k
    return {"phase": phase, "source": "measured (usage_metadata on every response)",
            "llm_calls": int(usage.get("calls", 0)),
            "in_tokens": in_tok, "out_tokens": out_tok,
            "responses_with_reasoning_block": int(usage.get("reasoning_blocks", 0)),
            "truncated_responses": int(usage.get("truncations", 0)),
            "usd": round(llm_usd, 4)}


def build_cost_report(cfg, run_id: str, est: dict, ragas_usage: dict, judge_usage: dict) -> dict:
    ra = actual(cfg, ragas_usage, "ragas")
    ja = actual(cfg, judge_usage, "judge")
    total = round(ra["usd"] + ja["usd"], 4)
    return {
        "run_id": run_id,
        "gating": "NONE — cost is reported, never enforced",
        "rates_usd_per_1k": {"llm_in": cfg.rate_llm_in_per_1k,
                             "llm_out": cfg.rate_llm_out_per_1k,
                             "embed": cfg.rate_embed_per_1k},
        "estimate": est,
        "actual": {"ragas": ra, "judge": ja, "total_usd": total},
        "delta_usd": round(total - est["total_usd"], 4),
        "note": "embeddings token counts are not exposed by Bedrock; embed cost is estimate-only",
    }


def config_snapshot(cfg, run_id: str, dhash: str, ids, ragas_metrics, judge_metrics,
                    parity: dict, versions: dict) -> dict:
    """The full resolved run configuration (NO secrets) — S10 pins: unified-file sha, both
    golden shas, judge-prompts sha, union KB fingerprint, judge model id + thinking config."""
    from . import _ragas_ctx_precision_patch as _cp
    from . import _ragas_ctx_recall_patch as _cr
    import json as _json
    payload = _json.loads(cfg.unified_eval_json.read_text(encoding="utf-8"))
    return {
        "run_id": run_id,
        "dataset_path": str(cfg.unified_eval_json),
        "dataset_sha256_16": dhash,
        "unified_file_sha256": parity.get("unified eval file sha256", {}).get("sha"),
        "golden_v2_sha256": payload.get("golden_sha256"),
        "legacy_golden_sha256": payload.get("legacy_golden_sha256"),
        "judge_prompts_sha256": parity.get("judge_prompts_v1.md sha256", {}).get("sha"),
        "kb_union_fingerprint": parity.get("KB union fingerprint", {}).get("sha"),
        "judge_model_id": cfg.bedrock_model_id,
        "embeddings_model_id": cfg.bedrock_embed_model_id,
        "thinking": {"mode": cfg.thinking_mode, "effort": cfg.thinking_effort,
                     "additional_model_request_fields": cfg.additional_model_request_fields()},
        # OWNER OVERRIDE of ragas defaults ([0.75,0.25], beta=1.0) — recorded so a finished
        # run is always attributable to the weighting that produced it
        "answer_correctness_weighting": {
            "weights_factuality_semantic": list(cfg.ac_weights),
            "recall_weight_within_f_score": cfg.ac_recall_weight,
            "precision_weight_within_f_score": round(1.0 - cfg.ac_recall_weight, 6),
            "beta": round(cfg.ac_beta, 6),
            "ragas_defaults_would_be": {"weights": [0.75, 0.25], "beta": 1.0}},
        "sample_ids": list(ids),
        "metrics": {"ragas": list(ragas_metrics), "judge": list(judge_metrics)},
        "hallucination_judge": "dropped (2026-08-13 decision) — forbidden_facts unconsumed",
        "ctx_precision_judge": _cp.mode(),
        "ctx_recall_judge": _cr.mode(),
        "versions": versions,
        "config": cfg.snapshot(),
    }


def library_versions() -> dict:
    import platform
    out = {"python": platform.python_version()}
    for mod in ("ragas", "langchain_core", "langchain_aws", "boto3", "botocore", "pandas",
                "numpy", "openpyxl"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            out[mod] = "not installed"
    return out
