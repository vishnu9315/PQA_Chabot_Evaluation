"""Compare two runs (kept from project B). No API calls.

Reads each run's `summary.json` and diffs the composite and the per-metric overall means.
Runs are only comparable like-for-like: if the comparable key (unified-file sha, judge
prompts sha, KB fingerprint, model id, metric set, thinking effort) differs, that is
surfaced as a loud warning — numbers are still shown, but a delta across different data,
prompts or models is not meaningful.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .aggregate import ALL_METRICS


def _load(run_dir):
    p = Path(run_dir) / "summary.json"
    if not p.exists():
        raise FileNotFoundError(f"no summary.json in {run_dir} (run --all/--report there first)")
    summ = json.loads(p.read_text(encoding="utf-8"))
    snap_p = Path(run_dir) / "config_snapshot.json"
    snap = json.loads(snap_p.read_text(encoding="utf-8")) if snap_p.exists() else {}
    return summ, snap


def _key(snap: dict) -> dict:
    return {k: snap.get(k) for k in
            ("unified_file_sha256", "judge_prompts_sha256", "kb_union_fingerprint",
             "judge_model_id", "ctx_precision_judge", "ctx_recall_judge")} | {
            "thinking": (snap.get("thinking") or {}).get("effort"),
            "metrics": json.dumps(snap.get("metrics", {}), sort_keys=True)}


def _delta(a, b) -> Optional[float]:
    if a is None or b is None:
        return None
    return round(b - a, 4)


def _fmt(v):
    return "—" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def compare_runs(dir_a, dir_b) -> dict:
    (a, sa), (b, sb) = _load(dir_a), _load(dir_b)
    ka, kb = _key(sa), _key(sb)
    comparable = ka == kb
    warnings = [f"{k}: {ka.get(k)!r} != {kb.get(k)!r}"
                for k in sorted(set(ka) | set(kb)) if ka.get(k) != kb.get(k)]

    ca = (a.get("composite") or {}).get("overall", {})
    cb = (b.get("composite") or {}).get("overall", {})
    composite = {"a": ca.get("composite_mean"), "b": cb.get("composite_mean"),
                 "delta": _delta(ca.get("composite_mean"), cb.get("composite_mean")),
                 "a_covered": f"{ca.get('n_scored')}/{ca.get('n_total')}",
                 "b_covered": f"{cb.get('n_scored')}/{cb.get('n_total')}"}

    metrics = {}
    for m in ALL_METRICS:
        ma = (a.get("per_metric_overall") or {}).get(m, {})
        mb = (b.get("per_metric_overall") or {}).get(m, {})
        metrics[m] = {"a": ma.get("mean"), "b": mb.get("mean"),
                      "delta": _delta(ma.get("mean"), mb.get("mean")),
                      "a_n": ma.get("n_scored"), "b_n": mb.get("n_scored")}

    return {"run_a": a.get("run_id"), "run_b": b.get("run_id"),
            "comparable": comparable, "warnings": warnings,
            "composite": composite, "metrics": metrics}


def format_compare(diff: dict) -> str:
    L = [f"# Run comparison — A `{diff['run_a']}`  vs  B `{diff['run_b']}`", ""]
    if not diff["comparable"]:
        L.append("> ⚠️  **Runs are NOT comparable** — deltas below span different "
                 "data/prompts/models:")
        for w in diff["warnings"]:
            L.append(f">   - {w}")
        L.append("")
    else:
        L.append("Runs share the unified-file sha, judge-prompts sha, KB fingerprint, "
                 "model id, prompt modes and thinking effort (comparable).")
        L.append("")
    c = diff["composite"]
    L += ["## Composite",
          f"- A: {_fmt(c['a'])}  ({c['a_covered']} covered)",
          f"- B: {_fmt(c['b'])}  ({c['b_covered']} covered)",
          f"- Δ (B−A): {_fmt(c['delta'])}", "",
          "## Per-metric overall mean — Δ = B−A",
          "| metric | A (n) | B (n) | Δ |", "|---|---|---|---|"]
    for m in ALL_METRICS:
        d = diff["metrics"][m]
        L.append(f"| {m} | {_fmt(d['a'])} ({d['a_n']}) | {_fmt(d['b'])} ({d['b_n']}) "
                 f"| {_fmt(d['delta'])} |")
    return "\n".join(L)
