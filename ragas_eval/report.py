"""Reporting. No API calls.

Writes into the run dir, from `per_sample.jsonl` (the canonical source):
  * aggregates.csv — the canonical tidy long table (metric x segment), n beside every mean
  * summary.json   — machine-readable summary (dashboard / --compare read this)
  * report.md      — provenance-sliced tables FIRST (Part 1 ingested vs Part 2 not-ingested
                     must be readable separately), then expected_behavior / difficulty /
                     stratum / category / house; the coverage panel with the NA manifest;
                     the routing table; the Part 2 section golden_dataset_README.md 4.4 asks
                     for; denominators stated on every aggregate. A single overall mean is
                     never the headline.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import aggregate as agg

_AGG_COLUMNS = ["metric", "label", "phase", "segment_type", "segment_value", "n", "n_scored",
                "mean", "min", "max", "low_n", "na_skipped", "na_failed"]


def _md_table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join("—" if c is None else str(c) for c in r) + " |")
    return "\n".join(out)


def _fmt(v):
    return "—" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def write_aggregates_csv(agg_rows: list, path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_AGG_COLUMNS)
        w.writeheader()
        for r in agg_rows:
            w.writerow({k: r.get(k) for k in _AGG_COLUMNS})


def _seg_block(L, agg_rows, comp, st, metrics):
    by = {}
    for r in agg_rows:
        if r["segment_type"] == st:
            by.setdefault(r["segment_value"], {})[r["metric"]] = r
    headers = [st] + [agg.LABEL[m] for m in metrics] + ["composite (n)"]
    rows = []
    for sv in sorted(by):
        cells = []
        for m in metrics:
            r = by[sv].get(m)
            if not r or r["mean"] is None:
                cells.append("—")
            else:
                flag = "*" if r["low_n"] else ""
                cells.append(f"{r['mean']:.2f}{flag} (n={r['n_scored']})")
        c = comp["by_segment"].get(st, {}).get(sv, {})
        cm = c.get("composite_mean")
        cells.append("—" if cm is None else
                     f"{cm:.3f} ({c.get('n_scored')}/{c.get('n_total')})")
        rows.append([sv] + cells)
    L.append(f"**By {st}:**")
    L.append("")
    L.append(_md_table(headers, rows))
    L.append("")


INTERPRETATION_NOTES = [
    ("No Hallucination judge.", "Faithfulness covers grounding-to-retrieval; Answer "
     "Correctness catches wrong figures vs reference."),
    ("Sources Validation", "honours `acceptable` satisfying `required` (cross-House twins) "
     "and uses the `documents/` corpus as the fabrication floor. Its precision counts every "
     "document the chatbot lists, so a row listing several related replies scores low "
     "precision even when the right one is cited; read recall and precision separately."),
    ("Context Precision", "never sees the response; it measures the retriever over up to 10 "
     "captured chunks (hardened prompt, see docs/RAGAS_SCORING.md section 5)."),
    ("Refusals", "(\"I could not find an answer to that question.\") make no claims, so "
     "Faithfulness on them is undefined and ragas returns 0 or 1 arbitrarily; the Part 2 "
     "section reports faithfulness with refusals excluded."),
    ("Faithfulness under-scores answers that reproduce large tables:", "the retrieved text "
     "holds them as HTML cells far from their column headers, so per-cell claims are hard "
     "for the judge to verify. Check any low-faithfulness row with a table by hand."),
    ("The composite", "is an equal-weight mean of 9 metrics that mixes retrieval, grounding "
     "and answer quality; use it to rank rows, not as a quality score."),
]

PART1, PART2 = "part1-ingested", "part2-not-ingested"
# Faithfulness below this on a Part 2 row = claims the retrieved text doesn't support.
PART2_FAITHFULNESS_FLAG = 0.8
_LACKS_INFO = re.compile(
    r"(do(es)? not|don't|doesn't) (have|contain|include)|not available|no (specific )?"
    r"(information|data|details)|(could|can)( ?not|n't) (find|locate)|unable to (find|locate)|"
    r"not (found|present|covered) in", re.IGNORECASE)


def _score(rec, metric):
    return ((rec.get("metrics") or {}).get(metric) or {}).get("score")


def _mean_n(recs, metric):
    vals = [v for v in (_score(r, metric) for r in recs) if v is not None]
    return ("—" if not vals else f"{sum(vals) / len(vals):.3f}") + f" (n={len(vals)}/{len(recs)})"


def part2_facts(per_sample) -> Optional[dict]:
    """The facts golden_dataset_README.md section 4.4 asks for; None without Part 2 rows.
    Shared by report.md and dashboard.html so the two never disagree."""
    p1 = [r for r in per_sample if r.get("provenance") == PART1]
    p2 = [r for r in per_sample if r.get("provenance") == PART2]
    if not p2:
        return None
    refusals = {r["id"] for r in per_sample if _LACKS_INFO.search(r.get("response") or "")}
    p2_ans = [r for r in p2 if r["id"] not in refusals]
    return {
        "p1": p1, "p2": p2,
        "p1_ans": [r for r in p1 if r["id"] not in refusals], "p2_ans": p2_ans,
        "lacks": sorted(r["id"] for r in p2 if r["id"] in refusals),
        "flagged": sorted(r["id"] for r in p2_ans
                          if (_score(r, "faithfulness") is not None
                              and _score(r, "faithfulness") < PART2_FAITHFULNESS_FLAG)),
        "rs271": sorted(r["id"] for r in p2
                        if any("Session_271" in c for c in r.get("cited_sources") or [])),
    }


def part2_block(per_sample) -> list:
    """golden_dataset_README.md section 4.4 — what to report for the not-ingested rows."""
    f = part2_facts(per_sample)
    if f is None:
        return []
    p1, p2, p1_ans, p2_ans = f["p1"], f["p2"], f["p1_ans"], f["p2_ans"]
    lacks, flagged, rs271 = f["lacks"], f["flagged"], f["rs271"]
    L = ["## 9. Part 2 (not ingested) — how the assistant behaves without the document", "",
         "Part 2's answer documents (Rajya Sabha 271) are not in the KB, so source recall / "
         "precision are 0 **by design** and are not failures. Faithfulness is the main "
         "measure here (golden_dataset_README.md section 4).", ""]
    L.append(_md_table(["metric", "Part 1 (ingested)", "Part 2 (not ingested)"],
                       [["faithfulness, answered rows (refusals make no claims, excluded)",
                         _mean_n(p1_ans, "faithfulness"), _mean_n(p2_ans, "faithfulness")],
                        ["faithfulness, all rows", _mean_n(p1, "faithfulness"),
                         _mean_n(p2, "faithfulness")],
                        ["completeness (Part 2: from earlier sessions only, no pass mark)",
                         _mean_n(p1, "completeness"), _mean_n(p2, "completeness")]]))
    L.append("")
    L.append(f"- **Answered rows with claims not supported by the retrieved text** "
             f"(faithfulness < {PART2_FAITHFULNESS_FLAG}): {len(flagged)}/{len(p2_ans)}"
             + (f" — {', '.join(flagged)}" if flagged else "")
             + ". Review each for invented or undated-old figures (README 4.3).")
    L.append(f"- **Rows where the assistant said it lacked the information** (keyword match — "
             f"spot-check): {len(lacks)}/{len(p2)}" + (f" — {', '.join(lacks)}" if lacks else ""))
    L.append(f"- **Rajya Sabha 271 documents cited** (README 4.1): "
             + (f"YES on {', '.join(rs271)} — Session 271 may be ingested; report Part 2 as "
                f"ingested questions instead" if rs271 else "none — Session 271 is absent "
                "from the KB, as Part 2 assumes"))
    L.append("")
    return L


def build_report_md(per_sample, agg_rows, comp, cov, meta, routing=None, stats=None,
                    pacing=None, cost=None, stability=None) -> str:
    metrics = agg.present_metrics(per_sample)
    overall = {r["metric"]: r for r in agg_rows if r["segment_type"] == "overall"}
    L: list = []
    w = L.append

    w("# MeitY Parliamentary Q&A Chatbot — Evaluation Report")
    w("")
    w(f"- **Run:** {meta.get('run_id', '?')} ({meta.get('mode', '?')})  ")
    w(f"- **Judge:** {meta.get('llm_model_id', '?')} · thinking "
      f"{meta.get('thinking', '?')} · **Embeddings:** {meta.get('embeddings_model_id', '?')}  ")
    # Counts are DERIVED, never written in. The hard-coded "41-row / 25 pilot-v2 +
    # 16 legacy-v1 / cells are 1-8 rows" header survived the S12 rebuild and the S15
    # re-capture and was wrong on all three numbers by the time anyone read it.
    n_total = len(per_sample)
    prov = Counter(r.get("provenance") or "unknown" for r in per_sample)
    prov_str = " + ".join(f"{v} {k}" for k, v in sorted(prov.items(), key=lambda kv: (-kv[1], kv[0])))
    seg_ns = [int(r["n"]) for r in agg_rows
              if r.get("segment_type") == "stratum" and str(r.get("n", "")).isdigit()]
    cells = (f"{min(seg_ns)}–{max(seg_ns)}" if seg_ns else "?")
    w(f"- **Dataset:** {n_total}-row unified eval (sha16 `{meta.get('dataset_hash', '?')}`) · "
      f"**Generated:** {meta.get('generated', '?')}")
    w("")
    w(f"> **Read with care — small cells.** n={n_total} overall ({prov_str}); "
      f"per-stratum cells are {cells} rows. `*` marks cells with n_scored < 5 — directional "
      "only. Every mean states its denominator. A single overall mean is never "
      "the headline: read the provenance split first — the subsets have different "
      "lineage (golden_v2 60f0b1f… vs legacy b16be30…) and are not one population.")
    w("")

    if routing:
        w("## 1. Routing (derived from the data, verified against the amendments block)")
        w("")
        c = routing["counts"]
        w(_md_table(["route", "n", "ids"],
                    [["full-RAGAS (6 metrics + 3 judges)", c["full_ragas"],
                      "…" if c["full_ragas"] > 8 else ", ".join(routing["routes"]["full_ragas"])],
                     ["partial-RAGAS (3 answer-side metrics + 3 judges)", c["partial_ragas"],
                      ", ".join(routing["routes"]["partial_ragas"]) or "—"],
                     ["judge-only (abstains)", c["judge_only"],
                      ", ".join(routing["routes"]["judge_only"])]]))
        w("")
        for note in routing.get("notes", []):
            w(f"> ⚠ {note}")
            w("")

    w("## 2. Provenance split — the primary slice")
    w("")
    _seg_block(L, agg_rows, comp, "provenance", metrics)

    o = comp["overall"]
    w("## 3. Composite (equal weights over the 9 metrics; partial composites flagged)")
    w("")
    w(f"Overall: **{_fmt(o['composite_mean'])}** over {o['n_scored']}/{o['n_total']} samples "
      f"(`composite_full`, all nine components present: {_fmt(o['composite_full_mean'])} over "
      f"{o['n_full']}). A sample's composite is the mean of the components that scored; "
      f"missing components shrink the numerator's basis, never count as 0.")
    w("")

    w("## 4. Per-metric (overall — denominators stated)")
    w("")
    rows = []
    for m in metrics:
        r = overall.get(m, {})
        rows.append([agg.LABEL[m], r.get("phase"),
                     f"{r.get('n_scored', 0)}/{r.get('n', 0)}",
                     _fmt(r.get("mean")), _fmt(r.get("min")), _fmt(r.get("max")),
                     ("skipped:" + str(r.get("na_skipped")) + " failed:" + str(r.get("na_failed")))
                     if (r.get("na_skipped") or r.get("na_failed")) else ""])
    w(_md_table(["metric", "phase", "n_scored/n", "mean", "min", "max", "NA"], rows))
    w("")

    w("## 5. Per-segment tables")
    w("")
    for st in ("expected_behavior", "difficulty", "stratum", "category", "house"):
        _seg_block(L, agg_rows, comp, st, metrics)

    w("## 6. Coverage and the NA manifest")
    w("")
    rows = [[c["label"], f"{c['scored']}/{c['total']}",
             json.dumps(c["na_by_class"]) if c["na_by_class"] else "—"]
            for m, c in cov.items()]
    w(_md_table(["metric", "scored/total", "NA by class"], rows))
    w("")
    manifest = agg.na_manifest(per_sample)
    if manifest:
        w(f"All {len(manifest)} NA cells ({{id, metric, reason}}):")
        w("")
        for item in manifest:
            w(f"- `{item['id']}` / {item['metric']} — {str(item['reason'])[:140]}")
    else:
        w("No NA cells — every metric scored on every routed row.")
    w("")
    w("*A NA is excluded from every mean, never counted as 0.*")
    w("")

    if stability:
        w("## 7. Stability (smoke repeats)")
        w("")
        w(f"{stability['repeats']} uncached repeats; tolerance {stability['tolerance']} on "
          f"composite metrics; **{stability['n_exceeding']} cell(s) exceed it.**")
        w("")
        rows = [[r["id"], r["metric"],
                 " / ".join("—" if s is None else f"{s:.3f}" for s in r["scores"]),
                 _fmt(r["spread"]), "⚠" if r["exceeds"] else ""]
                for r in stability["rows"][:20]]
        w(_md_table(["id", "metric", "scores", "spread", "flag"], rows))
        w("")

    w("## 8. Worst samples (by composite)")
    w("")
    rows = [[x["id"], x["provenance"], _fmt(x["composite"]),
             f"{x['weakest_metric']} = {_fmt(x['weakest_score'])}", x["n_components"]]
            for x in agg.worst_samples(per_sample, 5)]
    w(_md_table(["id", "provenance", "composite", "weakest metric", "components"], rows))
    w("")

    L.extend(part2_block(per_sample))

    if cost:
        w("## 10. Cost (tracked, never gated)")
        w("")
        w(f"- estimate: ${cost['estimate']['total_usd']:.2f} "
          f"(RAGAS ${cost['estimate']['ragas']['usd']:.2f} + "
          f"judge ${cost['estimate']['judge']['usd']:.2f})")
        a = cost["actual"]
        w(f"- measured: ${a['total_usd']:.2f} "
          f"(RAGAS ${a['ragas']['usd']:.2f} over {a['ragas']['llm_calls']} calls; "
          f"judge ${a['judge']['usd']:.2f} over {a['judge']['llm_calls']} calls)")
        tr = a["ragas"]["truncated_responses"] + a["judge"]["truncated_responses"]
        w(f"- truncated responses refused (never scored): {tr}")
        w("")

    if pacing:
        w("## 11. Pacing")
        w("")
        w(f"- {pacing.get('requests', 0)} calls, {pacing.get('throttles', 0)} throttles, "
          f"{pacing.get('transient_errors', 0)} transient, total wait "
          f"{pacing.get('total_wait_s', 0)}s, max concurrency "
          f"{pacing.get('max_concurrency_reached', 0)}")
        w("")

    w("## Interpretation notes")
    w("")
    for title, text in INTERPRETATION_NOTES:
        w(f"- **{title}** {text}")
    from .dataset import FULL_ANSWER_CONTEXT_IDS
    present = [r["id"] for r in per_sample if r["id"] in FULL_ANSWER_CONTEXT_IDS]
    if present:
        w(f"- **{len(present)} row(s) take their retrieved context from the chatbot's "
          f"`full_answer` rather than reranked `extracts`** ({', '.join(present)}): on "
          f"verbatim-gate rows the server emits no extracts but populates `full_answer` with the "
          f"source-derived passage. These rows' **Faithfulness, Context Precision and Context "
          f"Recall are INCLUDED in every mean above by operator decision**, but read them with "
          f"care: `full_answer` is ~99.8% similar to the row's own response, so those three "
          f"metrics are partly self-referential there — Faithfulness in particular tends to ~1.0 "
          f"by construction rather than by measurement. The other "
          f"{len(per_sample) - len(present)} rows use genuine reranked extracts. Per-row values "
          f"are in the All-rows table if you want to read the two groups separately.")
    return "\n".join(L)


def write_reports(run_dir, per_sample, cfg, run_meta: Optional[dict] = None,
                  routing=None, stats=None, pacing=None, cost=None, snapshot=None,
                  stability=None) -> dict:
    from .dataset import dataset_hash
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    agg_rows = agg.aggregate(per_sample, cfg.segments)
    comp = agg.composite_summary(per_sample, cfg.segments)
    cov = agg.coverage(per_sample)

    meta = {"run_id": run_dir.name, "llm_model_id": cfg.bedrock_model_id,
            "embeddings_model_id": cfg.bedrock_embed_model_id,
            "thinking": f"{cfg.thinking_mode}@{cfg.thinking_effort}",
            "dataset_hash": dataset_hash(cfg.unified_eval_json),
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    if run_meta:
        meta.update(run_meta)

    agg_csv = run_dir / "aggregates.csv"
    write_aggregates_csv(agg_rows, agg_csv)
    summ = agg.summary_json(meta["run_id"], per_sample, agg_rows, cov, comp, stats,
                            pacing, cost, routing=routing)
    summ_json = run_dir / "summary.json"
    summ_json.write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
    report_md = run_dir / "report.md"
    report_md.write_text(build_report_md(per_sample, agg_rows, comp, cov, meta,
                                         routing=routing, stats=stats, pacing=pacing,
                                         cost=cost, stability=stability), encoding="utf-8")
    out = {"aggregates_csv": agg_csv, "summary_json": summ_json, "report_md": report_md,
           "composite": comp}
    if cost is not None:
        p = run_dir / "cost.json"
        p.write_text(json.dumps(cost, ensure_ascii=False, indent=2), encoding="utf-8")
        out["cost_json"] = p
    if snapshot is not None:
        p = run_dir / "config_snapshot.json"
        p.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        out["config_snapshot"] = p
    return out
