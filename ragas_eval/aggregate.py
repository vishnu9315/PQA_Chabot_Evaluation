"""Assembly and aggregation. No API calls.

`per_sample.jsonl` is the single canonical artefact: every table, the composite, the
coverage panel and the NA manifest derive from it in one pass, so no two views of a run
can disagree.

NA semantics are pinned and never relaxed: an unscored cell is `None`, never `0`; means
are over scored cells only and are `None` when nothing scored; every NA carries its reason
and lands in the run manifest as {id, metric, reason} (S10 spec); and every aggregate
states its denominator — context-metric means may cover fewer rows than the dataset holds.

Slicing (S10 REPORTING): provenance (pilot-v2 / legacy-v1 / rebuilt-s12) FIRST — the subsets have
different lineage and must be readable separately — then expected_behavior, difficulty,
stratum, category, house. Per-stratum cells can be as small as 1 row: n is printed beside
every mean and cells with n_scored < MIN_N are flagged directional. A single overall mean
is never the headline.

Composite: equal weights over the nine metrics (6 RAGAS + 3 judges, all 0–1). A sample's
composite is the mean of the components that DID score (with n_components and a partial
flag), following project A's fix for B's all-or-nothing composite that once covered 7/20;
`composite_full` (all nine present) is reported alongside so the stricter number stays
visible.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from . import judge_runner as J
from . import ragas_runner as R

RAGAS_METRICS = list(R.RAGAS_METRICS)
JUDGE_METRICS = list(J.JUDGE_METRICS)
ALL_METRICS = RAGAS_METRICS + JUDGE_METRICS
COMPOSITE_METRICS = list(ALL_METRICS)          # equal weights, all nine
COMPOSITE_WEIGHTS = {m: 1.0 / len(COMPOSITE_METRICS) for m in COMPOSITE_METRICS}
LABEL = {**{s.name: s.label for s in R.ALL_SPECS}, **J.JUDGE_LABEL}
MIN_N = 5


def load_per_sample(path) -> list:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def write_per_sample(per_sample: list, path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in per_sample) + "\n",
                 encoding="utf-8")


_NOT_COMPUTED = ("not computed — no cached cell for this (sample, metric); re-run with "
                 "--resume to fill only the gap")


def _cell(rec: Optional[dict]) -> dict:
    if rec is None:
        return {"score": None, "scored": False, "na_reason": _NOT_COMPUTED,
                "failure_class": "not_computed"}
    out = {"score": rec.get("score"), "scored": rec.get("score") is not None,
           "na_reason": rec.get("na_reason"), "failure_class": rec.get("failure_class"),
           "seconds": rec.get("seconds"), "judge_effort": rec.get("judge_effort")}
    for k in ("justification", "detail"):
        if rec.get(k):
            out[k] = rec[k]
    return out


def present_metrics(per_sample: list) -> list:
    """Metrics actually present in the run, in canonical order — a --metrics subset must
    not make unrequested metrics look like failures (project A's coverage lesson)."""
    seen = set()
    for rec in per_sample:
        seen.update((rec.get("metrics") or {}).keys())
    return [m for m in ALL_METRICS if m in seen] or list(ALL_METRICS)


def build_per_sample(samples, cache, metric_names=None) -> list:
    """Assemble per-sample records from the cell cache + dataset tags. No API calls."""
    metric_names = metric_names or ALL_METRICS
    out = []
    for s in samples:
        rec = {"id": s.id,
               "question": s.user_input,
               "provenance": s.provenance,
               "expected_behavior": s.expected_behavior,
               "difficulty": s.difficulty,
               "stratum": s.stratum,
               "category": s.category,
               "house": s.house,
               "route": s.route,
               "n_retrieved_contexts": len(s.retrieved_contexts),
               "response_chars": len(s.response),
               # The scored text itself travels with the scores. per_sample.jsonl is the
               # canonical artefact, so a dashboard/report rebuilt from a finished run shows
               # exactly what was evaluated — it can never pair old scores with a newer
               # capture of the dataset.
               "response": s.response,
               "reference": s.reference,
               "query_style": s.query_style,
               "prior_question": (s.prior_turn or {}).get("user_input"),
               "cited_sources": list(s.cited_sources),
               "expected_sources": [e.get("document") for e in s.expected_sources],
               "expected_sources_detail": [
                   {"document": e.get("document"), "necessity": e.get("necessity", "required"),
                    "authored_as": e.get("document_as_authored"), "session": e.get("session")}
                   for e in s.expected_sources if isinstance(e, dict)],
               "unsupported_aspects": list(s.unsupported_aspects),
               "metrics": {}}
        for m in metric_names:
            cached = cache.get(s.id, m)
            if cached is None and m in RAGAS_METRICS and s.route == "judge_only":
                # abstains are routed out of the RAGAS phase entirely — that is a
                # structural skip, not a computation gap
                rec["metrics"][m] = {"score": None, "scored": False,
                                     "na_reason": "skipped: judge_only_row (abstain)",
                                     "failure_class": "skipped"}
                continue
            rec["metrics"][m] = _cell(cached)
        rec.update(composite_for(rec))
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
def composite_for(rec: dict) -> dict:
    requested = [m for m in COMPOSITE_METRICS if m in (rec.get("metrics") or {})]
    vals, missing = [], []
    for m in requested:
        c = rec.get("metrics", {}).get(m) or {}
        if c.get("score") is None:
            missing.append(m)
        else:
            vals.append(float(c["score"]))
    n, total = len(vals), len(requested)
    return {"composite": round(sum(vals) / n, 4) if n else None,
            "composite_full": round(sum(vals) / n, 4) if n and n == total else None,
            "n_components": n, "n_components_total": total,
            "composite_partial": bool(n and n < total),
            "composite_missing": missing}


def _mean(vals):
    return round(sum(vals) / len(vals), 4) if vals else None


def aggregate(per_sample: list, segments) -> list:
    """One-pass reduction to (metric x segment_type x segment_value) rows. Every row
    carries n AND n_scored — the denominator is never implicit."""
    metric_names = present_metrics(per_sample)
    groups: dict = {}
    for rec in per_sample:
        segs = [("overall", "all")] + [(s, str(rec.get(s))) for s in segments]
        for m in metric_names:
            c = rec.get("metrics", {}).get(m) or {}
            score = c.get("score")
            bucket = "scored" if score is not None else (
                "na_skipped" if str(c.get("failure_class")) == "skipped" else "na_failed")
            for st, sv in segs:
                g = groups.setdefault((m, st, sv),
                                      {"scores": [], "n": 0, "na_skipped": 0, "na_failed": 0})
                g["n"] += 1
                if bucket == "scored":
                    g["scores"].append(float(score))
                else:
                    g[bucket] += 1

    rows = []
    order = {"overall": 0}
    for i, s in enumerate(segments, 1):
        order[s] = i
    for (m, st, sv), g in groups.items():
        sc = g["scores"]
        rows.append({
            "metric": m, "label": LABEL[m],
            "phase": "ragas" if m in RAGAS_METRICS else "judge",
            "segment_type": st, "segment_value": sv,
            "n": g["n"], "n_scored": len(sc), "mean": _mean(sc),
            "min": round(min(sc), 4) if sc else None,
            "max": round(max(sc), 4) if sc else None,
            "low_n": len(sc) < MIN_N,
            "na_skipped": g["na_skipped"], "na_failed": g["na_failed"],
        })
    rows.sort(key=lambda r: (ALL_METRICS.index(r["metric"]),
                             order.get(r["segment_type"], 99), str(r["segment_value"])))
    return rows


def na_manifest(per_sample: list) -> list:
    """Every NA in the run as {id, metric, reason} (S10: 'Every NA lands in the run
    manifest')."""
    out = []
    for rec in per_sample:
        for m in present_metrics(per_sample):
            c = rec.get("metrics", {}).get(m) or {}
            if c.get("score") is None:
                out.append({"id": rec["id"], "metric": m,
                            "reason": ("empty retrieved_contexts"
                                       if str(c.get("na_reason", "")).startswith(
                                           "skipped: empty_retrieved_contexts")
                                       else c.get("na_reason")),
                            "failure_class": c.get("failure_class")})
    return out


def coverage(per_sample: list) -> dict:
    out: dict = {}
    for m in present_metrics(per_sample):
        scored, na = 0, []
        for rec in per_sample:
            c = rec.get("metrics", {}).get(m) or {}
            if c.get("score") is not None:
                scored += 1
            else:
                na.append({"id": rec["id"], "failure_class": c.get("failure_class"),
                           "na_reason": c.get("na_reason")})
        by_class: dict = {}
        for item in na:
            by_class[str(item["failure_class"])] = by_class.get(str(item["failure_class"]), 0) + 1
        out[m] = {"label": LABEL[m], "scored": scored, "na": len(na),
                  "total": len(per_sample), "na_by_class": by_class, "na_detail": na}
    return out


def composite_summary(per_sample: list, segments) -> dict:
    def agg(recs):
        vals = [r["composite"] for r in recs if r.get("composite") is not None]
        full = [r["composite_full"] for r in recs if r.get("composite_full") is not None]
        return {"composite_mean": _mean(vals), "n_scored": len(vals), "n_total": len(recs),
                "composite_full_mean": _mean(full), "n_full": len(full),
                "low_n": len(vals) < MIN_N}

    out = {"overall": agg(per_sample), "by_segment": {}}
    for st in segments:
        buckets: dict = {}
        for rec in per_sample:
            buckets.setdefault(str(rec.get(st)), []).append(rec)
        out["by_segment"][st] = {sv: agg(v) for sv, v in sorted(buckets.items())}
    return out


def worst_samples(per_sample: list, k: int = 5) -> list:
    scored = [r for r in per_sample if r.get("composite") is not None]
    scored.sort(key=lambda r: (r["composite"], r["id"]))
    out = []
    for rec in scored[:k]:
        worst_m, worst_s = None, None
        for m in [x for x in COMPOSITE_METRICS if x in (rec.get("metrics") or {})]:
            s = (rec["metrics"].get(m) or {}).get("score")
            if s is not None and (worst_s is None or s < worst_s):
                worst_m, worst_s = m, s
        out.append({"id": rec["id"], "provenance": rec["provenance"],
                    "composite": rec["composite"], "weakest_metric": worst_m,
                    "weakest_score": None if worst_s is None else round(worst_s, 4),
                    "n_components": rec["n_components"]})
    return out


def summary_json(run_id, per_sample, agg_rows, cov, comp, stats, pacing, cost,
                 routing=None) -> dict:
    return {
        "run_id": run_id,
        "n_samples": len(per_sample),
        "metrics": present_metrics(per_sample),
        "composite_metrics": COMPOSITE_METRICS,
        "composite_weights": COMPOSITE_WEIGHTS,
        "routing": routing,
        "composite": comp,
        "coverage": {m: {k: v for k, v in c.items() if k != "na_detail"}
                     for m, c in cov.items()},
        "per_metric_overall": {r["metric"]: {"mean": r["mean"], "n_scored": r["n_scored"],
                                             "n": r["n"]}
                               for r in agg_rows if r["segment_type"] == "overall"},
        "na_manifest": na_manifest(per_sample),
        "run_stats": stats,
        "pacing": pacing,
        "cost": cost,
        "worst_samples": worst_samples(per_sample, 5),
    }
