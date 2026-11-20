"""dashboard.html — a self-contained, no-API, shareable view of one run (from per_sample.jsonl).

Provenance (Part 1 / Part 2) split first, then the metric tiles (each stating its
denominator), the Part 2 section, the per-row matrix, per-row detail (question, chatbot
response, ground truth, sources, required facts, judge notes), reading notes and the NA
manifest. Score chips are colour banded: good >= 0.8, warn >= 0.5, bad < 0.5, na = grey.
Renders in light and dark themes; each row is linkable as dashboard.html#<row id>.
"""
from __future__ import annotations

import html
import json
import re
from collections import Counter
from pathlib import Path

from . import aggregate as agg
from .report import (INTERPRETATION_NOTES, PART1, PART2, PART2_FAITHFULNESS_FLAG,
                     missing_sessions_text, part2_facts)

TITLE = "MeitY PQA Chatbot Evaluation"

_DARK = ("--bg:#0f151b;--card:#151e26;--text:#e3e9ef;--muted:#93a2b0;--line:#26323d;"
         "--th:#1b252e;--txt:#111920;--notice:#2b2510;--notice-line:#5a4b1c;--link:#7fb2dd;"
         "--good:#17331f;--good-t:#8fd3a0;--warn:#3a3011;--warn-t:#efc65c;--bad:#3e1b18;"
         "--bad-t:#f29b91;--na:#243039;--na-t:#a3b3c0;color-scheme:dark")

_CSS = """
:root{--bg:#f6f7f9;--card:#fff;--text:#1c2733;--muted:#5b6b7b;--line:#e3e8ee;--th:#f0f3f6;
--txt:#fbfcfd;--notice:#fff8e1;--notice-line:#f0e0a0;--link:#1f5f8b;
--good:#e8f5e9;--good-t:#1b5e20;--warn:#fff8e1;--warn-t:#8d6e00;--bad:#fdecea;--bad-t:#b3261e;
--na:#eceff1;--na-t:#546e7a}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){DARK}}
:root[data-theme="dark"]{DARK}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);
font:14px/1.5 system-ui,"Segoe UI",Roboto,sans-serif;padding-block:28px 40px;
padding-left:max(clamp(16px,4vw,28px),env(safe-area-inset-left,0px));
padding-right:max(clamp(16px,4vw,28px),env(safe-area-inset-right,0px))}
a{color:var(--link)}code{font-size:.92em}
h1{font-size:21px;margin:0 0 4px;text-wrap:balance}h2{font-size:16px;margin:28px 0 10px}
.sub{color:var(--muted);font-size:12.5px;margin-bottom:18px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.kpi .v{font-size:26px;font-weight:700}.kpi .n{color:var(--muted);font-size:12px}
table{border-collapse:collapse;width:100%;background:var(--card);border:1px solid var(--line);
border-radius:10px;overflow:hidden;font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{background:var(--th);font-weight:600;white-space:nowrap}
.chip{display:inline-block;min-width:44px;text-align:center;border-radius:6px;
padding:2px 7px;font-weight:600;font-size:12.5px}
.good{background:var(--good);color:var(--good-t)}.warn{background:var(--warn);color:var(--warn-t)}
.bad{background:var(--bad);color:var(--bad-t)}.na{background:var(--na);color:var(--na-t)}
.tablewrap{overflow-x:auto}.small{font-size:12px;color:var(--muted)}
.notice{background:var(--notice);border:1px solid var(--notice-line);border-radius:10px;
padding:10px 14px;font-size:13px;margin:14px 0}
details.row{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:10px 14px;margin:9px 0;scroll-margin-top:12px}
details.row summary{cursor:pointer;font-weight:600;font-size:13.5px;list-style:none}
details.row summary::-webkit-details-marker{display:none}
details.row summary:before{content:"\\25B8 ";color:var(--muted)}
details.row[open] summary:before{content:"\\25BE "}
details.row summary:focus-visible{outline:2px solid var(--link);outline-offset:3px}
details.row .meta{color:var(--muted);font-weight:400;font-size:12px}
.txt{white-space:pre-wrap;word-break:break-word;font-size:12.5px;line-height:1.5;
background:var(--txt);border:1px solid var(--line);border-radius:8px;padding:9px 11px;
margin:5px 0 12px;max-height:420px;overflow:auto}
.lbl{font-size:11.5px;font-weight:700;letter-spacing:.04em;text-transform:uppercase;
color:var(--muted);margin-top:9px}
.scorestrip{margin:6px 0 2px}.scorestrip span.chip{margin-right:5px}
table.inner{margin:5px 0 12px;font-size:12.5px}table.inner td{word-break:break-word}
.doc{font-family:ui-monospace,Consolas,monospace;font-size:12px;word-break:break-all}
ul.notes{margin:0;padding-left:20px;max-width:80ch}ul.notes li{margin:5px 0}
""".replace("DARK", _DARK)

_HASH_JS = """<script>
(function(){
  function openFromHash(){
    var id=decodeURIComponent(location.hash.slice(1));
    var d=id&&document.getElementById(id);
    if(d&&d.tagName==="DETAILS"){d.open=true;d.scrollIntoView({block:"start"});}
  }
  window.addEventListener("hashchange",openFromHash);
  document.addEventListener("click",function(e){
    var a=e.target.closest&&e.target.closest('a[href^="#"]');
    if(a&&a.getAttribute("href")===location.hash){e.preventDefault();openFromHash();}
  });
  openFromHash();
})();
</script>"""

_VERDICT = {"PRESENT": "good", "PARTIAL": "warn", "ABSENT": "bad"}


def _chip(v) -> str:
    if v is None:
        return '<span class="chip na">NA</span>'
    band = "good" if v >= 0.8 else ("warn" if v >= 0.5 else "bad")
    return f'<span class="chip {band}">{v:.2f}</span>'


def _f2(v) -> str:
    return "—" if v is None else f"{v:.2f}"


def _mean(recs, metric):
    vals = [s for s in (((r.get("metrics") or {}).get(metric) or {}).get("score")
                        for r in recs) if s is not None]
    return (sum(vals) / len(vals) if vals else None), len(vals)


def _md_inline(text: str) -> str:
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", html.escape(text))


def _dataset_name(cfg) -> str:
    try:
        return json.loads(Path(cfg.unified_eval_json).read_text(encoding="utf-8")).get("name") or ""
    except (OSError, ValueError, AttributeError):
        return ""


def _part2_section(per_sample, esc) -> list:
    f = part2_facts(per_sample)
    if f is None:
        return []
    out = ['<h2>Part 2 (not ingested) — behaviour without the document</h2>',
           '<div class="small">Part 2 answer documents are not in the knowledge base, so source '
           'recall and precision are 0 by design and are not failures. Faithfulness is the main '
           'measure (golden_dataset_README.md section 4).</div>',
           '<div class="tablewrap" style="margin-top:8px"><table><tr><th>metric</th>'
           '<th>Part 1 (ingested)</th><th>Part 2 (not ingested)</th></tr>']
    rows = [("faithfulness, answered rows (refusals make no claims, excluded)",
             f["p1_ans"], f["p2_ans"], "faithfulness"),
            ("faithfulness, all rows", f["p1"], f["p2"], "faithfulness"),
            ("completeness (Part 2: from earlier sessions only, no pass mark)",
             f["p1"], f["p2"], "completeness")]
    for label, a, b, m in rows:
        (ma, na), (mb, nb) = _mean(a, m), _mean(b, m)
        out.append(f'<tr><td>{esc(label)}</td><td>{_chip(ma)}<span class="small"> n={na}</span>'
                   f'</td><td>{_chip(mb)}<span class="small"> n={nb}</span></td></tr>')
    out.append("</table></div>")

    def ids(lst):
        return ", ".join(f'<a href="#{esc(i)}">{esc(i)}</a>' for i in lst)

    out.append('<ul class="notes" style="margin-top:12px">')
    out.append(f'<li><b>Answered rows with claims not supported by the retrieved text</b> '
               f'(faithfulness &lt; {PART2_FAITHFULNESS_FLAG}): {len(f["flagged"])}/'
               f'{len(f["p2_ans"])}' + (f' — {ids(f["flagged"])}' if f["flagged"] else "") + '</li>')
    out.append(f'<li><b>Rows where the assistant said it lacked the information:</b> '
               f'{len(f["lacks"])}/{len(f["p2"])}' + (f' — {ids(f["lacks"])}' if f["lacks"] else "")
               + '</li>')
    sessions = esc(missing_sessions_text(f))
    if not f["missing_sessions"]:
        out.append('<li><b>Documents from the not-ingested sessions cited:</b> not checked, '
                   'because the Part 2 rows carry no session field.</li>')
    else:
        out.append(f'<li><b>Documents from {sessions} cited:</b> ' +
                   (f'YES on {ids(f["cited_missing"])}; that session may be ingested'
                    if f["cited_missing"] else
                    'none, so it is absent from the knowledge base as Part 2 assumes') + '</li>')
    out.append("</ul>")
    return out


def _row_detail(rec, metrics, esc) -> list:
    out = []
    m = rec.get("metrics") or {}
    comp = rec.get("composite")
    style = rec.get("query_style")
    head = (f'{esc(rec["id"])} <span class="meta">· {esc(str(rec.get("provenance")))}'
            + (f' · {esc(style)}' if style else "") +
            f' · {esc(str(rec.get("expected_behavior")))} · {esc(str(rec.get("stratum")))} · '
            f'composite {"—" if comp is None else f"{comp:.3f}"}</span>')
    out.append(f'<details class="row" id="{esc(rec["id"])}"><summary>{head}</summary>')
    out.append('<div class="scorestrip">')
    for k in metrics:
        out.append(f'<span class="small">{esc(agg.LABEL[k])}</span> '
                   f'{_chip((m.get(k) or {}).get("score"))} ')
    out.append('</div>')
    if rec.get("prior_question"):
        out.append('<div class="lbl">Earlier question in the same chat</div>')
        out.append(f'<div class="txt">{esc(rec["prior_question"])}</div>')
    out.append('<div class="lbl">Query</div>')
    out.append(f'<div class="txt">{esc(str(rec.get("question", "")))}</div>')
    cited = rec.get("cited_sources") or []
    out.append(f'<div class="lbl">Chatbot response <span class="meta">'
               f'({rec.get("response_chars", 0):,} chars · '
               f'{rec.get("n_retrieved_contexts", 0)} retrieved context(s) · '
               f'{len(cited)} source(s) listed)</span></div>')
    out.append(f'<div class="txt">{esc(str(rec.get("response", "")))}</div>')
    out.append('<div class="lbl">Ground truth (official reply)</div>')
    out.append(f'<div class="txt">{esc(str(rec.get("reference", "")))}</div>')

    detail = rec.get("expected_sources_detail") or [
        {"document": d, "necessity": "required", "authored_as": None}
        for d in rec.get("expected_sources") or []]
    expected = {e["document"] for e in detail}
    out.append('<div class="lbl">Sources · expected vs cited</div>')
    out.append('<div class="tablewrap"><table class="inner"><tr><th>expected document</th>'
               '<th>necessity</th><th>cited?</th></tr>')
    for e in detail:
        note = (f'<br><span class="small">authored as {esc(e["authored_as"])}</span>'
                if e.get("authored_as") and e["authored_as"] != e["document"] else "")
        hit = ('<span class="chip good">cited</span>' if e["document"] in cited
               else '<span class="chip bad">not cited</span>')
        out.append(f'<tr><td class="doc">{esc(str(e["document"]))}{note}</td>'
                   f'<td>{esc(str(e.get("necessity")))}</td><td>{hit}</td></tr>')
    out.append("</table></div>")
    extra = [c for c in cited if c not in expected]
    if extra:
        out.append(f'<div class="small">Also listed by the chatbot ({len(extra)}): '
                   f'<span class="doc">{esc(" · ".join(extra))}</span></div>')

    facts = ((m.get("completeness") or {}).get("detail") or {}).get("per_fact") or []
    if facts:
        out.append('<div class="lbl">Required facts · completeness judge</div>')
        out.append('<div class="tablewrap"><table class="inner"><tr><th>fact</th><th>weight</th>'
                   '<th>verdict</th></tr>')
        for fct in facts:
            v = str(fct.get("verdict") or "")
            chip = (f'<span class="chip {_VERDICT.get(v, "na")}">{esc(v.lower() or "—")}</span>')
            out.append(f'<tr><td>{esc(str(fct.get("fact", "")))}</td>'
                       f'<td>{esc(str(fct.get("weight", "")))}</td><td>{chip}</td></tr>')
        out.append("</table></div>")
    ua = rec.get("unsupported_aspects") or []
    if ua:
        out.append('<div class="lbl">Unsupported aspects (expected to be flagged, not answered)</div>')
        out.append('<div class="txt">' + esc("\n".join("• " + a for a in ua)) + '</div>')

    notes = []
    for k, label in (("completeness", "Completeness"), ("response_quality", "Response quality"),
                     ("sources_validation", "Sources")):
        c = m.get(k) or {}
        if not c.get("justification"):
            continue
        d = c.get("detail") or {}
        extra_bits = ""
        if k == "response_quality" and d.get("dimensions"):
            extra_bits = "; " + ", ".join(f"{dk} {_f2(float(dv))}" for dk, dv in d["dimensions"].items())
        if k == "sources_validation" and d:
            extra_bits = f"; recall {_f2(d.get('recall'))}, precision {_f2(d.get('precision'))}"
        notes.append(f"{label} ({_f2(c.get('score'))}{extra_bits}): {c['justification']}")
    if notes:
        out.append('<div class="lbl">Judge notes</div>')
        out.append(f'<div class="txt">{esc(chr(10).join(notes))}</div>')
    out.append("</details>")
    return out


def build_html(per_sample, cfg, run_meta=None, cost=None, snapshot=None, pacing=None,
               routing=None, stability=None) -> str:
    esc = html.escape
    metrics = agg.present_metrics(per_sample)
    agg_rows = agg.aggregate(per_sample, cfg.segments)
    comp = agg.composite_summary(per_sample, cfg.segments)
    overall = {r["metric"]: r for r in agg_rows if r["segment_type"] == "overall"}
    manifest = agg.na_manifest(per_sample)
    meta = run_meta or {}
    has_parts = any(r.get("provenance") in (PART1, PART2) for r in per_sample)

    P = []
    w = P.append
    w(f"<title>{esc(TITLE)}</title>")
    w('<meta name="claude:viewport-fit-sides" content="cover">')
    w(f"<style>{_CSS}</style>")
    w('<div>')
    w(f"<h1>MeitY PQA — evaluation run {esc(str(meta.get('run_id', '')))}</h1>")
    think = f"{cfg.thinking_mode}@{cfg.thinking_effort}"
    _prov = Counter(r.get("provenance") or "unknown" for r in per_sample)
    _prov_str = " + ".join(f"{v} {k}" for k, v in sorted(_prov.items(),
                                                         key=lambda kv: (-kv[1], kv[0])))
    name = _dataset_name(cfg)
    w(f'<div class="sub">MeitY Parliament AI chatbot · '
      + (f'dataset {esc(name)} · ' if name else "")
      + f'judge {esc(cfg.bedrock_model_id)} · thinking {esc(think)} · '
      f'{len(per_sample)} rows ({esc(_prov_str)}) · '
      f'generated {esc(str(meta.get("finished", meta.get("generated", ""))))}</div>')

    o = comp["overall"]
    lineage = ("read the Part 1 / Part 2 split first: Part 2 answer documents are not in the "
               "knowledge base, so several of its scores are low by design." if has_parts else
               "read the provenance split first: the subsets have different lineage.")
    w(f'<div class="notice">A single overall mean is never the headline — {lineage} Composite '
      f'(equal-weight, 9 metrics): <b>{"—" if o["composite_mean"] is None else o["composite_mean"]}'
      f'</b> over {o["n_scored"]}/{o["n_total"]} rows '
      f'(strict all-components: {"—" if o["composite_full_mean"] is None else o["composite_full_mean"]} '
      f'over {o["n_full"]}).</div>')

    from .dataset import FULL_ANSWER_CONTEXT_IDS
    fa = [r["id"] for r in per_sample if r["id"] in FULL_ANSWER_CONTEXT_IDS]
    if fa:
        w(f'<div class="notice"><b>{len(fa)} row(s) use <code>full_answer</code> as their '
          f'retrieved context</b> ({esc(", ".join(fa))}). Their Faithfulness / Context Precision / '
          f'Context Recall are included in the means but partly self-referential.</div>')

    if routing:
        c = routing["counts"]
        w("<h2>Routing</h2>")
        w(f'<div class="card">full-RAGAS <b>{c["full_ragas"]}</b> · partial-RAGAS '
          f'<b>{c["partial_ragas"]}</b> ({esc(", ".join(routing["routes"]["partial_ragas"]) or "—")}) '
          f'· judge-only <b>{c["judge_only"]}</b> '
          f'({esc(", ".join(routing["routes"]["judge_only"]) or "—")})</div>')

    w("<h2>Provenance split (primary slice)</h2>")
    if has_parts:
        w('<div class="small" style="margin-bottom:8px"><b>part1-ingested</b>: answer documents '
          'are in the knowledge base, so these rows measure answer quality. '
          '<b>part2-not-ingested</b>: documents not ingested, so these rows measure behaviour '
          'without the document. Never average the two.</div>')
    w('<div class="tablewrap"><table><tr><th>provenance</th>')
    for m in metrics:
        w(f"<th>{esc(agg.LABEL[m])}</th>")
    w("<th>composite</th></tr>")
    by = {}
    for r in agg_rows:
        if r["segment_type"] == "provenance":
            by.setdefault(r["segment_value"], {})[r["metric"]] = r
    for sv in sorted(by):
        w(f"<tr><td><b>{esc(sv)}</b></td>")
        for m in metrics:
            r = by[sv].get(m) or {}
            w(f'<td>{_chip(r.get("mean"))}<span class="small"> n={r.get("n_scored", 0)}</span></td>')
        c = comp["by_segment"].get("provenance", {}).get(sv, {})
        w(f"<td>{_chip(c.get('composite_mean'))}"
          f"<span class='small'> {c.get('n_scored')}/{c.get('n_total')}</span></td></tr>")
    w("</table></div>")

    w("<h2>Per-metric (overall, denominator on every tile)</h2>")
    w('<div class="grid">')
    for m in metrics:
        r = overall.get(m, {})
        w(f'<div class="card kpi"><div>{esc(agg.LABEL[m])} '
          f'<span class="small">[{esc(r.get("phase", ""))}]</span></div>'
          f'<div class="v">{_chip(r.get("mean"))}</div>'
          f'<div class="n">n_scored {r.get("n_scored", 0)}/{r.get("n", 0)}'
          + (f' · NA skipped {r.get("na_skipped")}' if r.get("na_skipped") else "")
          + (f' · NA failed {r.get("na_failed")}' if r.get("na_failed") else "")
          + "</div></div>")
    w("</div>")

    P.extend(_part2_section(per_sample, esc))

    w("<h2>All rows</h2>")
    w('<div class="small" style="margin-bottom:8px">Click an id to open its detail below.</div>')
    w('<div class="tablewrap"><table><tr><th>id</th><th>provenance</th><th>style</th>'
      '<th>behaviour</th><th>stratum</th>')
    for m in metrics:
        w(f"<th>{esc(agg.LABEL[m])}</th>")
    w("<th>composite</th></tr>")
    for rec in per_sample:
        w(f'<tr><td><b><a href="#{esc(rec["id"])}">{esc(rec["id"])}</a></b></td>'
          f"<td>{esc(str(rec['provenance']))}</td><td>{esc(str(rec.get('query_style') or '—'))}</td>"
          f"<td>{esc(str(rec['expected_behavior']))}</td><td>{esc(str(rec['stratum']))}</td>")
        for m in metrics:
            c = rec.get("metrics", {}).get(m) or {}
            w(f"<td>{_chip(c.get('score'))}</td>")
        w(f"<td>{_chip(rec.get('composite'))}"
          f"<span class='small'> {rec.get('n_components')}/{rec.get('n_components_total')}"
          f"</span></td></tr>")
    w("</table></div>")

    w("<h2>Row detail — query, response and ground truth</h2>")
    w('<div class="small">Collapsed by default; click a row to expand. The text shown is the '
      'text that was scored, carried in per_sample.jsonl alongside the scores. Each row can be '
      'linked directly: add #&lt;row id&gt; to the page address.</div>')
    for rec in per_sample:
        P.extend(_row_detail(rec, metrics, esc))

    w("<h2>Reading notes</h2>")
    w('<ul class="notes">' + "".join(f"<li><b>{esc(t)}</b> {_md_inline(x)}</li>"
                                     for t, x in INTERPRETATION_NOTES) + "</ul>")

    w(f"<h2>NA manifest ({len(manifest)})</h2>")
    if manifest:
        w('<div class="tablewrap"><table><tr><th>id</th><th>metric</th><th>reason</th></tr>')
        for item in manifest:
            w(f"<tr><td>{esc(item['id'])}</td><td>{esc(item['metric'])}</td>"
              f"<td class='small'>{esc(str(item['reason'])[:200])}</td></tr>")
        w("</table></div>")
        w('<div class="small">An NA is excluded from every mean, never counted as 0.</div>')
    else:
        w('<div class="card">No NA cells: every metric scored on every routed row.</div>')

    if stability:
        w(f"<h2>Stability ({stability['repeats']} repeats, tolerance "
          f"{stability['tolerance']})</h2>")
        w('<div class="tablewrap"><table><tr><th>id</th><th>metric</th><th>scores</th>'
          '<th>spread</th></tr>')
        for r in stability["rows"][:25]:
            scores = " / ".join("NA" if s is None else f"{s:.3f}" for s in r["scores"])
            flag = " ⚠" if r["exceeds"] else ""
            w(f"<tr><td>{esc(r['id'])}</td><td>{esc(r['metric'])}</td><td>{esc(scores)}</td>"
              f"<td>{'—' if r['spread'] is None else r['spread']}{flag}</td></tr>")
        w("</table></div>")

    foot = []
    if cost:
        a = cost["actual"]
        foot.append(f"cost measured ${a['total_usd']:.2f} "
                    f"(RAGAS ${a['ragas']['usd']:.2f} + judge ${a['judge']['usd']:.2f}) vs "
                    f"estimate ${cost['estimate']['total_usd']:.2f} — reported, never gated")
    if pacing:
        foot.append(f"pacing: {pacing.get('requests', 0)} calls, "
                    f"{pacing.get('throttles', 0)} throttles, "
                    f"{pacing.get('total_wait_s', 0)}s waited")
    if snapshot:
        shas = [(label, snapshot.get(key)) for label, key in
                (("dataset sha", "dataset_sha256_16"), ("unified sha", "unified_file_sha256"),
                 ("KB", "kb_union_fingerprint"), ("prompts", "judge_prompts_sha256"))]
        foot.extend(f"{label} {str(v)[:16]}" for label, v in shas if v)
        if snapshot.get("ctx_precision_judge"):
            foot.append(f"context precision judge: {snapshot['ctx_precision_judge']}")
    if foot:
        w(f'<div class="sub" style="margin-top:22px">{esc(" · ".join(foot))}</div>')
    w("</div>")
    w(_HASH_JS)
    return "\n".join(P)


def write_dashboard(run_dir, per_sample, cfg, run_meta=None, cost=None, snapshot=None,
                    pacing=None, routing=None, stability=None, stats=None) -> Path:
    out = Path(run_dir) / "dashboard.html"
    out.write_text(build_html(per_sample, cfg, run_meta=run_meta, cost=cost,
                              snapshot=snapshot, pacing=pacing, routing=routing,
                              stability=stability), encoding="utf-8")
    return out


def build_dashboard(run_dir, cfg) -> Path:
    """Rebuild dashboard.html for an existing run dir from its own artefacts. No API."""
    d = Path(run_dir)
    per_path = d / "per_sample.jsonl"
    if not per_path.exists():
        raise FileNotFoundError(f"no per_sample.jsonl in {d} (run --all/--report there first)")
    per_sample = agg.load_per_sample(per_path)

    def _load(name):
        p = d / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    summ = _load("summary.json") or {}
    return write_dashboard(d, per_sample, cfg,
                           run_meta={"run_id": d.name},
                           cost=_load("cost.json"), snapshot=_load("config_snapshot.json"),
                           pacing=summ.get("pacing"), routing=summ.get("routing"),
                           stability=_load("stability.json"))
