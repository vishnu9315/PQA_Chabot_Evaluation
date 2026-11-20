"""The RAGAS phase: six metrics, driven one (sample, metric) cell at a time, crash-safely.

Module name follows project B's layout; the engine inside is project A's `runner.CellRunner`
— B's `ragas.evaluate()` batching was replaced because evaluate() swallows per-cell
exceptions into bare NaN, returns nothing until the last cell lands, and cannot express
WHY a cell failed (A's ISSUE-P4 forensics). Per-cell driving gives fsync'd incremental
results, a reason on every NA, and independent retries.

Metric registry (S10 spec — exactly these six):
    Answer Correctness   -> AnswerCorrectness   (needs answer_similarity passed explicitly —
                                                 ragas 0.4.3 leaves it None and asserts)
    Answer Relevancy     -> ResponseRelevancy
    Semantic Similarity  -> SemanticSimilarity  (embeddings only)
    Faithfulness         -> Faithfulness
    Context Precision    -> LLMContextPrecisionWithReference
    Context Recall       -> LLMContextRecall

Routing is enforced by `requires`: the 5 partial-RAGAS rows (empty retrieved_contexts) get
their three context metrics recorded as NA "skipped: empty_retrieved_contexts" — a terminal
NA that is never retried and never 0.0 (S10: "never 0.0, never silently dropped").

Ported fixes, cited at the site of each: ISSUE-2 (retry pass, reimplemented per-cell),
ISSUE-P7/P12 (per-chunk timeout shaping for Context Precision), ISSUE-P4 (failure classes),
ISSUE-P10 (TruncatedOutputError surfaces via providers), ISSUE-P8 (watchdog heartbeats).
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import math
import time
import traceback
from dataclasses import dataclass, field
from typing import Dict, Set

import ragas_eval  # noqa: F401  — applies the compat shim + upstream patches on import

from .providers import TruncatedOutputError
from .watchdog import heartbeat


# ---------------------------------------------------------------------------
# metric registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MetricSpec:
    name: str
    label: str
    ragas_class: str
    reads: tuple
    requires: tuple          # sample attributes that must be non-empty
    needs_llm: bool
    needs_embeddings: bool
    note: str = ""

    def build(self, llm, emb, cfg=None):
        return _build(self.name, llm, emb, cfg)


ANSWER_CORRECTNESS = MetricSpec(
    "answer_correctness", "Answer Correctness", "AnswerCorrectness",
    ("user_input", "response", "reference"), ("response", "reference"), True, True,
    "weighted mean of claim-F1 and semantic similarity; weights and the F-score beta are "
    "owner-set via AC_FACTUALITY_WEIGHT / AC_RECALL_WEIGHT (defaults 0.5/0.5 and recall 0.6 "
    "vs precision 0.4), NOT the ragas defaults of [0.75, 0.25] and beta=1. "
    "answer_similarity must be constructed explicitly on ragas 0.4.3.")
ANSWER_RELEVANCY = MetricSpec(
    "answer_relevancy", "Answer Relevancy", "ResponseRelevancy",
    ("user_input", "response"), ("response",), True, True,
    "Generates questions from the answer and embeds them against the real question.")
SEMANTIC_SIMILARITY = MetricSpec(
    "semantic_similarity", "Semantic Similarity", "SemanticSimilarity",
    ("response", "reference"), ("response", "reference"), False, True,
    "Embeddings only, no LLM call.")
FAITHFULNESS = MetricSpec(
    "faithfulness", "Faithfulness", "Faithfulness",
    ("user_input", "response", "retrieved_contexts"), ("retrieved_contexts", "response"),
    True, False,
    "Claim decomposition + NLI against the retrieved set. NA (never 0) on empty retrieval.")
CONTEXT_PRECISION = MetricSpec(
    "context_precision", "Context Precision", "LLMContextPrecisionWithReference",
    ("user_input", "retrieved_contexts", "reference"), ("retrieved_contexts",), True, False,
    "One LLM verdict per retrieved chunk, serially — timeout scales with chunk count "
    "(ISSUE-P7/P12). Rank-sensitive: chunk order is never altered.")
CONTEXT_RECALL = MetricSpec(
    "context_recall", "Context Recall", "LLMContextRecall",
    ("user_input", "retrieved_contexts", "reference"), ("retrieved_contexts",), True, False,
    "Attributes each reference statement to the retrieved set. NA (never 0) on empty retrieval.")

ALL_SPECS = (ANSWER_CORRECTNESS, ANSWER_RELEVANCY, SEMANTIC_SIMILARITY, FAITHFULNESS,
             CONTEXT_PRECISION, CONTEXT_RECALL)
BY_NAME = {s.name: s for s in ALL_SPECS}
RAGAS_METRICS = [s.name for s in ALL_SPECS]
CONTEXT_METRICS = ["faithfulness", "context_precision", "context_recall"]
ANSWER_SIDE_METRICS = [m for m in RAGAS_METRICS if m not in CONTEXT_METRICS]


def _build(name, llm, emb, cfg=None):
    if cfg is None:                     # only reached by ad-hoc callers; the phase passes cfg
        from config import load_config
        cfg = load_config()
    from ragas.metrics import (AnswerCorrectness, AnswerSimilarity, Faithfulness,
                               LLMContextPrecisionWithReference, LLMContextRecall,
                               ResponseRelevancy, SemanticSimilarity)
    if name == "answer_correctness":
        # ragas 0.4.3 never constructs answer_similarity itself; passing `embeddings=` alone
        # leaves it None and every call dies on `AssertionError: AnswerSimilarity must be set`.
        # weights and beta are OWNER-OVERRIDDEN (2026-08-20), not ragas defaults: equal
        # weight to claim-F1 and semantic similarity, and inside the F-score claim recall
        # carries 0.6 against precision 0.4. Both come from config so they are recorded in
        # config_snapshot.json and stamped on every cell; the cache key does NOT cover them,
        # so changing either needs `--clear-cells --metrics answer_correctness` first.
        return AnswerCorrectness(llm=llm, embeddings=emb,
                                 weights=cfg.ac_weights, beta=cfg.ac_beta,
                                 answer_similarity=AnswerSimilarity(embeddings=emb))
    if name == "answer_relevancy":
        return ResponseRelevancy(llm=llm, embeddings=emb)
    if name == "semantic_similarity":
        return SemanticSimilarity(embeddings=emb)
    if name == "faithfulness":
        return Faithfulness(llm=llm)
    if name == "context_precision":
        return LLMContextPrecisionWithReference(llm=llm)
    if name == "context_recall":
        return LLMContextRecall(llm=llm)
    raise KeyError(name)


def missing_inputs(spec: MetricSpec, sample) -> str | None:
    """Return an NA reason if this sample cannot support this metric, else None.
    This is what routes the 5 empty-retrieval rows to partial-RAGAS."""
    for attr in spec.requires:
        val = getattr(sample, attr, None)
        if val is None or (hasattr(val, "__len__") and len(val) == 0) or \
           (isinstance(val, str) and not val.strip()):
            return f"skipped: empty_{attr}"
    return None


def resolve(names) -> list:
    if not names:
        return list(RAGAS_METRICS)
    unknown = [n for n in names if n not in BY_NAME]
    if unknown:
        raise ValueError(f"unknown RAGAS metric(s): {unknown}; known: {RAGAS_METRICS}")
    return [n for n in RAGAS_METRICS if n in set(names)]


# ---------------------------------------------------------------------------
# NA classification (A's ISSUE-P4: every failure carries its family and reason)
# ---------------------------------------------------------------------------
TERMINAL_NA_PREFIXES = ("skipped:",)


def is_retryable_na(na_reason) -> bool:
    if not na_reason:
        return False
    return not str(na_reason).startswith(TERMINAL_NA_PREFIXES)


def _fmt_exc(exc: BaseException) -> str:
    msg = str(exc).strip().replace("\n", " ")
    if len(msg) > 400:
        msg = msg[:400] + "…"
    return f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__


def classify_failure(exc: BaseException) -> str:
    if isinstance(exc, TruncatedOutputError):
        return "truncated_output"
    if isinstance(exc, asyncio.TimeoutError):
        return "timeout"
    name = type(exc).__name__
    blob = f"{name} {exc}".lower()
    if "ragasoutputparser" in blob or "outputparser" in blob or "failed to parse" in blob:
        return "parse_failure"
    if "throttl" in blob or "too many requests" in blob:
        return "throttled"
    if "validation error" in blob or "field required" in blob:
        return "schema_rejection"
    if "timeout" in blob:
        return "timeout"
    return "error"


# ---------------------------------------------------------------------------
class CellRunner:
    """Scores RAGAS cells for one (dataset, model) pair into one crash-safe cache."""

    def __init__(self, cfg, samples, cache, llm=None, emb=None, log=None):
        self.cfg = cfg
        self.samples = {s.id: s for s in samples}
        self.order = [s.id for s in samples]
        self.cache = cache
        self.log = log or (lambda m: None)
        self._llm = llm
        self._emb = emb

    def _providers(self):
        if self._llm is None or self._emb is None:
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from ragas.llms import LangchainLLMWrapper
            from .providers import get_embeddings, get_llm
            # bypass_temperature: ragas mutates the model's temperature at call time, which
            # Sonnet 5 rejects outright (and thinking mode implies temperature=1 anyway).
            self._llm = LangchainLLMWrapper(get_llm(self.cfg), bypass_temperature=True)
            self._emb = LangchainEmbeddingsWrapper(get_embeddings(self.cfg))
        return self._llm, self._emb

    def _judge_provenance(self) -> dict:
        """Stamp judge settings onto every cell — the cache key deliberately excludes the
        thinking effort (adding it would invalidate paid cells), so the effort a cell was
        scored at is recorded IN the cell and the report surfaces the mix (A's ISSUE-P10)."""
        from . import _ragas_ctx_precision_patch as _cp
        from . import _ragas_ctx_recall_patch as _cr
        return {"judge_effort": getattr(self.cfg, "thinking_effort", None),
                "judge_max_tokens": getattr(self.cfg, "ragas_max_tokens", None),
                "judge_ctx_precision": _cp.mode(),
                "judge_ctx_recall": _cr.mode(),
                # owner-overridden Answer Correctness weighting; the cache key excludes it,
                # so the values a cell was scored under are recorded IN the cell
                "ac_weights": list(self.cfg.ac_weights),
                "ac_beta": round(self.cfg.ac_beta, 6),
                "ac_recall_weight": self.cfg.ac_recall_weight}

    def timeout_for(self, spec, sample) -> float:
        """Per-cell deadline, scaled for Context Precision whose cost is one serial LLM
        verdict per retrieved chunk (A's ISSUE-P7; constant recalibrated per ISSUE-P12)."""
        base = float(self.cfg.ragas_timeout)
        per_chunk = float(getattr(self.cfg, "per_chunk_timeout", 300))
        if spec.name == "context_precision":
            return max(base, per_chunk * max(1, len(sample.retrieved_contexts)))
        return base

    async def _score_one(self, sample, spec, attempt_tag=""):
        """Score one cell. Returns a cache record; never raises for scoring failures."""
        from ragas import SingleTurnSample

        na = missing_inputs(spec, sample)
        if na:
            return self.cache.set(sample.id, spec.name, None, na,
                                  failure_class="skipped", attempts=0, seconds=0.0)

        llm, emb = self._providers()
        t0 = time.monotonic()
        try:
            metric = spec.build(llm, emb, self.cfg)
            st = SingleTurnSample(**sample.to_ragas_kwargs())
            score = await asyncio.wait_for(metric.single_turn_ascore(st),
                                           timeout=self.timeout_for(spec, sample))
        except BaseException as exc:                          # noqa: BLE001
            if isinstance(exc, (KeyboardInterrupt, SystemExit, asyncio.CancelledError)):
                raise
            dt = time.monotonic() - t0
            reason = _fmt_exc(exc)
            if isinstance(exc, asyncio.TimeoutError):
                # a bare TimeoutError is not actionable (A's ISSUE-P7 post-mortem)
                reason = (f"TimeoutError after {self.timeout_for(spec, sample):.0f}s "
                          f"({len(sample.retrieved_contexts)} retrieved chunks)")
            fclass = classify_failure(exc)
            heartbeat(f"{sample.id}/{spec.name} NA ({fclass})", cell=True)
            self.log(f"  NA  {sample.id:11} {spec.name:22} {fclass}: {reason[:140]}")
            return self.cache.set(sample.id, spec.name, None, reason,
                                  failure_class=fclass, seconds=round(dt, 2),
                                  traceback_tail=traceback.format_exc()[-600:],
                                  attempt_tag=attempt_tag, **self._judge_provenance())
        dt = time.monotonic() - t0
        if score is None or (isinstance(score, float) and math.isnan(score)):
            # ragas returns NaN when its own averaging has nothing to average over —
            # an unscorable cell, not a zero (A's ISSUE-P4 distinction).
            self.log(f"  NaN {sample.id:11} {spec.name:22} ragas returned NaN in {dt:.1f}s")
            return self.cache.set(sample.id, spec.name, None, "ragas_returned_nan",
                                  failure_class="nan", seconds=round(dt, 2),
                                  attempt_tag=attempt_tag)
        heartbeat(f"{sample.id}/{spec.name} scored", cell=True)
        self.log(f"  ok  {sample.id:11} {spec.name:22} {float(score):.4f}  ({dt:.1f}s)")
        return self.cache.set(sample.id, spec.name, float(score), None,
                              failure_class=None, seconds=round(dt, 2), attempt_tag=attempt_tag,
                              **self._judge_provenance())

    # -- sweeps -------------------------------------------------------------
    def pending(self, metric_names) -> list:
        return [(self.samples[sid], BY_NAME[name])
                for sid in self.order for name in metric_names
                if not self.cache.has(sid, name)]

    def retryable(self, metric_names) -> list:
        out = []
        for sid in self.order:
            for name in metric_names:
                rec = self.cache.get(sid, name)
                if rec and rec.get("score") is None and is_retryable_na(rec.get("na_reason")):
                    out.append((self.samples[sid], BY_NAME[name]))
        return out

    async def _gather(self, work, attempt_tag="") -> int:
        sem = asyncio.Semaphore(max(1, self.cfg.cell_concurrency))

        async def one(sample, spec):
            async with sem:
                return await self._score_one(sample, spec, attempt_tag=attempt_tag)

        results = await asyncio.gather(*(one(s, sp) for s, sp in work), return_exceptions=True)
        for r in results:
            if isinstance(r, BaseException):
                raise r
        return len(results)

    def run(self, metric_names, retry_rounds=None) -> dict:
        """Main sweep + automatic retry pass (B's ISSUE-2, per-cell). Returns statistics."""
        retry_rounds = self.cfg.ragas_retry_rounds if retry_rounds is None else retry_rounds
        work = self.pending(metric_names)
        total_cells = len(self.order) * len(metric_names)
        cached = total_cells - len(work)
        self.log(f"cells: {total_cells} total, {cached} cached, {len(work)} to compute")

        stats = {"total_cells": total_cells, "cached": cached, "computed": 0,
                 "retry_rounds_used": 0, "recovered_on_retry": 0}
        if work:
            stats["computed"] += _run_async(self._gather(work, "main"), self.cfg)

        for rnd in range(1, retry_rounds + 1):
            todo = self.retryable(metric_names)
            if not todo:
                break
            stats["retry_rounds_used"] = rnd
            before = {(s.id, sp.name) for s, sp in todo}
            self.log(f"retry round {rnd}/{retry_rounds}: re-scoring {len(todo)} NA cell(s)")
            # A retried cell must actually be recomputed — drop the cached NA first
            # (B's ISSUE-8 lesson: a failure cell still counts as "present" to a resume).
            for s, sp in todo:
                self.cache.drop(s.id, sp.name)
            stats["computed"] += _run_async(self._gather(todo, f"retry{rnd}"), self.cfg)
            still = {(s.id, sp.name) for s, sp in self.retryable(metric_names)}
            stats["recovered_on_retry"] += len(before - still)

        residual = self.retryable(metric_names)
        if residual:
            self.log(f"after {stats['retry_rounds_used']} retry round(s), {len(residual)} "
                     f"cell(s) remain NA: {sorted({sp.name for _, sp in residual})}")
        else:
            self.log("no non-terminal NA cells remain")
        stats["residual_na"] = [{"id": s.id, "metric": sp.name} for s, sp in residual]
        self.cache.compact()
        return stats

    # -- stability repeats (smoke gate) --------------------------------------
    def run_repeats(self, metric_names, repeats: int) -> dict:
        """Score the same cells `repeats` times WITHOUT the cache. Determinism is
        unavailable (thinking forces temperature=1), so spread is the only honest
        trustworthiness measure. All repeats fan out under the same limiter."""
        triples = [(r, self.samples[sid], BY_NAME[name])
                   for r in range(1, repeats + 1)
                   for sid in self.order
                   for name in metric_names]

        async def _all():
            sem = asyncio.Semaphore(max(1, self.cfg.cell_concurrency))

            async def one(r, sample, spec):
                async with sem:
                    rec = await self._score_one_uncached(sample, spec, r)
                    return (sample.id, spec.name, r, rec)

            return await asyncio.gather(*(one(*t) for t in triples))

        out: dict = {}
        for sid, name, r, rec in _run_async(_all(), self.cfg):
            out.setdefault((sid, name), []).append((r, rec))
        return {k: [rec for _, rec in sorted(v)] for k, v in out.items()}

    async def _score_one_uncached(self, sample, spec, repeat_idx):
        from ragas import SingleTurnSample
        na = missing_inputs(spec, sample)
        if na:
            return {"score": None, "na_reason": na, "failure_class": "skipped", "seconds": 0.0}
        llm, emb = self._providers()
        t0 = time.monotonic()
        try:
            metric = spec.build(llm, emb, self.cfg)
            st = SingleTurnSample(**sample.to_ragas_kwargs())
            score = await asyncio.wait_for(metric.single_turn_ascore(st),
                                           timeout=self.timeout_for(spec, sample))
        except BaseException as exc:                          # noqa: BLE001
            if isinstance(exc, (KeyboardInterrupt, SystemExit, asyncio.CancelledError)):
                raise
            dt = time.monotonic() - t0
            self.log(f"  rep{repeat_idx} NA  {sample.id:11} {spec.name:22} {_fmt_exc(exc)[:120]}")
            return {"score": None, "na_reason": _fmt_exc(exc),
                    "failure_class": classify_failure(exc), "seconds": round(dt, 2)}
        dt = time.monotonic() - t0
        if score is None or (isinstance(score, float) and math.isnan(score)):
            return {"score": None, "na_reason": "ragas_returned_nan", "failure_class": "nan",
                    "seconds": round(dt, 2)}
        self.log(f"  rep{repeat_idx} ok  {sample.id:11} {spec.name:22} {float(score):.4f} ({dt:.1f}s)")
        return {"score": float(score), "na_reason": None, "seconds": round(dt, 2)}


# ---------------------------------------------------------------------------
def _run_async(coro, cfg):
    """Run a coroutine on a fresh loop whose default executor is big enough. LangChain
    dispatches the sync `_generate` through `loop.run_in_executor(None, …)`; Python's
    default executor (min(32, cpu+4) threads) silently becomes the run's concurrency
    ceiling otherwise (A's ISSUE-P4 throughput note)."""
    loop = asyncio.new_event_loop()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(8, cfg.executor_threads),
                                                 thread_name_prefix="bedrock")
    try:
        loop.set_default_executor(pool)
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
        asyncio.set_event_loop(None)
        loop.close()
        pool.shutdown(wait=False, cancel_futures=True)
