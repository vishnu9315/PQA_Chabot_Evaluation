"""Runner routing, NA semantics, timeout shaping, aggregation and costs. No API calls."""
from ragas_eval import aggregate as agg
from ragas_eval import costs as C
from ragas_eval import ragas_runner as R
from ragas_eval.cache import CellCache
from tests.fixtures import FakeCfg, make_sample


# --- routing / NA semantics -------------------------------------------------
def test_empty_retrieval_partitions_context_metrics_to_terminal_na():
    s = make_sample(n_ctx=0)
    for m in R.CONTEXT_METRICS:
        reason = R.missing_inputs(R.BY_NAME[m], s)
        assert reason == "skipped: empty_retrieved_contexts"
        assert not R.is_retryable_na(reason)       # terminal — never retried, never 0
    for m in R.ANSWER_SIDE_METRICS:
        assert R.missing_inputs(R.BY_NAME[m], s) is None


def test_timeout_scales_per_chunk_for_context_precision():
    cfg = FakeCfg()
    runner = R.CellRunner(cfg, [make_sample(n_ctx=10)], cache=None)
    s = runner.samples["GD-0900"]
    assert runner.timeout_for(R.BY_NAME["context_precision"], s) == max(1800, 300 * 10)
    assert runner.timeout_for(R.BY_NAME["faithfulness"], s) == 1800


def test_retryable_vs_terminal_na_classification():
    assert R.is_retryable_na("TimeoutError after 1800s (5 retrieved chunks)")
    assert R.is_retryable_na("OutputParserException: …")
    assert not R.is_retryable_na("skipped: empty_retrieved_contexts")
    assert not R.is_retryable_na(None)


def test_pending_and_retryable_sweeps(tmp_path):
    cfg = FakeCfg()
    cache = CellCache(tmp_path / "c.jsonl", model_id="m", dhash="d")
    s = make_sample()
    runner = R.CellRunner(cfg, [s], cache)
    names = ["faithfulness", "semantic_similarity"]
    assert len(runner.pending(names)) == 2
    cache.set(s.id, "faithfulness", None, "TimeoutError after 1800s")
    cache.set(s.id, "semantic_similarity", 0.9, None)
    assert len(runner.pending(names)) == 0
    retry = runner.retryable(names)
    assert [(x.id, sp.name) for x, sp in retry] == [(s.id, "faithfulness")]


# --- aggregation --------------------------------------------------------------
def _per_sample(tmp_path):
    cache = CellCache(tmp_path / "c.jsonl", model_id="m", dhash="d")
    s_full = make_sample("GD-0001")
    s_empty = make_sample("GD-0083", n_ctx=0)
    for m in agg.ALL_METRICS:
        cache.set("GD-0001", m, 0.8, None)
    for m in agg.ALL_METRICS:
        if m in R.CONTEXT_METRICS:
            cache.set("GD-0083", m, None, "skipped: empty_retrieved_contexts",
                      failure_class="skipped")
        else:
            cache.set("GD-0083", m, 0.6, None)
    return agg.build_per_sample([s_full, s_empty], cache)


def test_partial_composite_never_counts_na_as_zero(tmp_path):
    per = _per_sample(tmp_path)
    empty = next(r for r in per if r["id"] == "GD-0083")
    assert empty["composite"] == 0.6               # mean over the 6 scored, not 9
    assert empty["n_components"] == 6
    assert empty["composite_partial"] is True
    assert set(empty["composite_missing"]) == set(R.CONTEXT_METRICS)
    full = next(r for r in per if r["id"] == "GD-0001")
    assert full["composite_full"] == 0.8


def test_aggregate_states_denominators(tmp_path):
    per = _per_sample(tmp_path)
    rows = agg.aggregate(per, FakeCfg.segments)
    faith = next(r for r in rows
                 if r["metric"] == "faithfulness" and r["segment_type"] == "overall")
    assert faith["n"] == 2 and faith["n_scored"] == 1 and faith["na_skipped"] == 1
    assert faith["mean"] == 0.8                    # over scored cells only


def test_na_manifest_normalises_the_empty_retrieval_reason(tmp_path):
    per = _per_sample(tmp_path)
    manifest = agg.na_manifest(per)
    reasons = {(m["id"], m["metric"]): m["reason"] for m in manifest}
    assert reasons[("GD-0083", "faithfulness")] == "empty retrieved_contexts"
    assert len(manifest) == 3                      # exactly the three context metrics


def test_provenance_is_the_first_segment():
    assert FakeCfg.segments[0] == "provenance"


# --- costs ---------------------------------------------------------------------
def test_estimate_is_routing_aware_and_per_phase():
    cfg = FakeCfg()
    full = make_sample("GD-0001", n_ctx=5)
    empty = make_sample("GD-0083", n_ctx=0)
    abstain = make_sample("GD-0165", behavior="abstain", eligible=False, cited=[])
    est = C.estimate(cfg, [full, empty, abstain])
    # context_precision counts only the full row's 5 chunks (empty + abstain excluded)
    assert est["ragas"]["calls_by_metric"]["context_precision"] == 5
    # answer_correctness runs on full + empty (not the abstain)
    assert est["ragas"]["calls_by_metric"]["answer_correctness"] == 6
    # judges cover ALL rows
    assert est["judge"]["llm_calls"] == 9
    assert est["total_usd"] > 0
    assert set(est) >= {"ragas", "judge", "total_usd", "basis"}


def test_judge_only_rows_get_structural_skip_not_not_computed(tmp_path):
    cache = CellCache(tmp_path / "c2.jsonl", model_id="m", dhash="d")
    abstain = make_sample("GD-0165", behavior="abstain", eligible=False, cited=[])
    for m in agg.JUDGE_METRICS:
        cache.set("GD-0165", m, 1.0, None)
    per = agg.build_per_sample([abstain], cache)
    rec = per[0]
    for m in R.RAGAS_METRICS:
        assert rec["metrics"][m]["na_reason"] == "skipped: judge_only_row (abstain)"
        assert rec["metrics"][m]["failure_class"] == "skipped"
    # composite is over the three judge scores only, flagged partial
    assert rec["composite"] == 1.0 and rec["n_components"] == 3


# --- Answer Correctness weighting: OWNER OVERRIDE of the ragas defaults (2026-08-20) --------

def test_ac_weights_and_beta_are_the_owner_values_not_ragas_defaults():
    """ragas ships weights [0.75, 0.25] and beta=1.0. Both are overridden: claim-F1 and
    semantic similarity weigh equally, and inside the F-score claim recall carries 0.6 against
    precision 0.4. If these ever silently revert, every Answer Correctness figure shifts."""
    import math
    from config import load_config
    cfg = load_config()
    assert cfg.ac_weights == [0.5, 0.5]
    assert cfg.ac_recall_weight == 0.6
    assert math.isclose(cfg.ac_beta, math.sqrt(1.5), rel_tol=1e-12)
    # ragas' __post_init__ rejects a beta whose `type(...) is not float`
    assert type(cfg.ac_beta) is float


def test_ac_beta_really_gives_recall_60_precision_40():
    """F_beta = (1+b^2)PR / (b^2 P + R) is the weighted harmonic mean of precision and recall
    with recall share b^2/(1+b^2). Assert the identity rather than trusting the algebra: the
    F-score must equal the harmonic mean weighted 0.4 precision / 0.6 recall."""
    from config import load_config
    from ragas.metrics.utils import fbeta_score
    beta = load_config().ac_beta
    assert abs(beta ** 2 / (1 + beta ** 2) - 0.6) < 1e-12
    for tp, fp, fn in [(13, 57, 3), (9, 0, 11), (5, 5, 5), (20, 1, 4), (1, 9, 1)]:
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        expected = 1.0 / (0.4 / p + 0.6 / r)          # weighted harmonic mean
        assert abs(fbeta_score(tp, fp, fn, beta) - expected) < 1e-9, (tp, fp, fn)


def test_ac_metric_is_constructed_with_the_owner_weighting():
    """The values must reach the metric object, not just the config."""
    from config import load_config
    from ragas_eval import ragas_runner as RR
    cfg = load_config()
    spec = next(s for s in RR.ALL_SPECS if s.name == "answer_correctness")
    m = spec.build(None, None, cfg)
    assert m.weights == [0.5, 0.5]
    assert abs(m.beta ** 2 / (1 + m.beta ** 2) - 0.6) < 1e-12


def test_ac_weighting_is_stamped_on_every_cell(tmp_path):
    """The cell-cache key does NOT cover the weighting, so a cell scored under one weighting
    would otherwise be indistinguishable from another. The values must travel in the cell."""
    from config import load_config
    from ragas_eval.ragas_runner import CellRunner
    cfg = load_config()
    prov = CellRunner.__dict__["_judge_provenance"](type("S", (), {"cfg": cfg})())
    assert prov["ac_weights"] == [0.5, 0.5]
    assert abs(prov["ac_beta"] - 1.224745) < 1e-5
    assert prov["ac_recall_weight"] == 0.6
