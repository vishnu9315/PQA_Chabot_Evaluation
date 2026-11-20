"""evaluation_report.xlsx — the run as a workbook (openpyxl, no API calls).

Sheets: Summary | Per Sample | Aggregates | NA Manifest | Judge Detail | Config.
Provenance is the first column of every per-sample surface (S10 slicing rule).
"""
from __future__ import annotations

import json
from pathlib import Path

from . import aggregate as agg

_HDR_FILL = "D9E2EC"
_BANDS = ((0.8, "C8E6C9"), (0.5, "FFF3C4"), (0.0, "FFCDD2"))


def _fill(ws, cell, color):
    from openpyxl.styles import PatternFill
    ws[cell].fill = PatternFill(start_color=color, end_color=color, fill_type="solid")


def _score_fill(ws, cell, v):
    if v is None:
        _fill(ws, cell, "ECEFF1")
        return
    for floor, color in _BANDS:
        if v >= floor:
            _fill(ws, cell, color)
            return


def _header(ws, row, headers):
    from openpyxl.styles import Font
    for j, h in enumerate(headers, 1):
        c = ws.cell(row=row, column=j, value=h)
        c.font = Font(bold=True)
        _fill(ws, c.coordinate, _HDR_FILL)


def write_excel_report(run_dir, per_sample, samples, cfg, run_meta=None, cost=None,
                       snapshot=None, pacing=None, routing=None, stability=None,
                       stats=None) -> Path:
    from openpyxl import Workbook

    metrics = agg.present_metrics(per_sample)
    agg_rows = agg.aggregate(per_sample, cfg.segments)
    comp = agg.composite_summary(per_sample, cfg.segments)
    manifest = agg.na_manifest(per_sample)
    meta = run_meta or {}

    wb = Workbook()

    # --- Summary ---
    ws = wb.active
    ws.title = "Summary"
    ws.append(["MeitY PQA evaluation", str(meta.get("run_id", ""))])
    ws.append(["judge", cfg.bedrock_model_id])
    ws.append(["thinking", f"{cfg.thinking_mode}@{cfg.thinking_effort}"])
    ws.append(["rows", len(per_sample)])
    if routing:
        ws.append(["routing", f"{routing['counts']['full_ragas']} full-RAGAS / "
                              f"{routing['counts']['partial_ragas']} partial-RAGAS / "
                              f"{routing['counts']['judge_only']} judge-only"])
    o = comp["overall"]
    ws.append(["composite (equal-weight, 9 metrics)",
               o["composite_mean"], f"over {o['n_scored']}/{o['n_total']}"])
    ws.append([])
    _header(ws, ws.max_row + 1, ["metric", "phase", "mean", "n_scored", "n",
                                 "NA skipped", "NA failed"])
    overall = {r["metric"]: r for r in agg_rows if r["segment_type"] == "overall"}
    for m in metrics:
        r = overall.get(m, {})
        ws.append([agg.LABEL[m], r.get("phase"), r.get("mean"), r.get("n_scored"),
                   r.get("n"), r.get("na_skipped"), r.get("na_failed")])
        _score_fill(ws, f"C{ws.max_row}", r.get("mean"))
    ws.append([])
    _header(ws, ws.max_row + 1, ["provenance", "composite", "n_scored", "n_total"])
    for sv, c in sorted(comp["by_segment"].get("provenance", {}).items()):
        ws.append([sv, c["composite_mean"], c["n_scored"], c["n_total"]])
        _score_fill(ws, f"B{ws.max_row}", c["composite_mean"])
    if cost:
        ws.append([])
        ws.append(["cost estimate ($)", cost["estimate"]["total_usd"],
                   "actual ($)", cost["actual"]["total_usd"], "never gated"])

    # --- Per Sample ---
    ws = wb.create_sheet("Per Sample")
    headers = (["id", "provenance", "route", "expected_behavior", "difficulty", "stratum",
                "category", "house"] + [agg.LABEL[m] for m in metrics]
               + ["composite", "components"])
    _header(ws, 1, headers)
    for rec in per_sample:
        row = [rec["id"], rec["provenance"], rec.get("route"), rec["expected_behavior"],
               rec["difficulty"], rec["stratum"], rec["category"], rec["house"]]
        for m in metrics:
            row.append((rec.get("metrics", {}).get(m) or {}).get("score"))
        row += [rec.get("composite"),
                f"{rec.get('n_components')}/{rec.get('n_components_total')}"]
        ws.append(row)
        base = 9
        for k, m in enumerate(metrics):
            col = ws.cell(row=ws.max_row, column=base + k).coordinate
            _score_fill(ws, col, (rec.get("metrics", {}).get(m) or {}).get("score"))
        _score_fill(ws, ws.cell(row=ws.max_row, column=base + len(metrics)).coordinate,
                    rec.get("composite"))
    ws.freeze_panes = "B2"

    # --- Aggregates (the canonical long table) ---
    ws = wb.create_sheet("Aggregates")
    cols = ["metric", "label", "phase", "segment_type", "segment_value", "n", "n_scored",
            "mean", "min", "max", "low_n", "na_skipped", "na_failed"]
    _header(ws, 1, cols)
    for r in agg_rows:
        ws.append([r.get(c) for c in cols])
    ws.freeze_panes = "A2"

    # --- NA Manifest ---
    ws = wb.create_sheet("NA Manifest")
    _header(ws, 1, ["id", "metric", "reason", "failure_class"])
    for item in manifest:
        ws.append([item["id"], item["metric"], str(item["reason"])[:500],
                   item.get("failure_class")])

    # --- Judge Detail ---
    ws = wb.create_sheet("Judge Detail")
    _header(ws, 1, ["id", "metric", "score", "justification", "detail"])
    for rec in per_sample:
        for m in [x for x in metrics if x in agg.JUDGE_METRICS]:
            c = rec.get("metrics", {}).get(m) or {}
            ws.append([rec["id"], agg.LABEL[m], c.get("score"),
                       str(c.get("justification", ""))[:900],
                       json.dumps(c.get("detail", {}), ensure_ascii=False)[:900]])
            _score_fill(ws, f"C{ws.max_row}", c.get("score"))
    ws.freeze_panes = "A2"

    # --- Config ---
    ws = wb.create_sheet("Config")
    _header(ws, 1, ["key", "value"])
    for k, v in (snapshot or {}).items():
        ws.append([k, json.dumps(v, ensure_ascii=False, default=str)[:900]
                   if isinstance(v, (dict, list)) else str(v)])

    out = Path(run_dir) / "evaluation_report.xlsx"
    wb.save(out)
    return out
