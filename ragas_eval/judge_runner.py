"""LLM-judge runner: three judges on the 0–1 scale, driven by the prompts in
`data/artifacts/judge_prompts_v1.md` (parsed at runtime, used VERBATIM — once this project
freezes, the parity gate pins that file's sha256, so a prompt edit fails --validate before
it can silently change scores).

Judges:
  completeness        response vs must_include_facts (core/supporting)
  response_quality    vs expected_behavior + unsupported_aspects + register
  sources_validation  cited_sources vs expected_sources, honouring `acceptable` satisfying
                      `required`; fabrication = cited document absent from the KB document
                      union (`documents/`, deterministic, computed in attribution.py) —
                      automatic 0. Skipped, not falsely triggered, if no KB is configured.

Follow-up (-F2) rows also show completeness / response_quality their earlier (-F1) turn,
since they were asked in the same chat (golden_dataset_README.md section 3.2).

The Hallucination judge in judge_prompts_v1.md is NOT used (dropped 2026-08-13):
Faithfulness covers grounding-to-retrieval and Answer Correctness catches wrong figures vs
reference. forbidden_facts is therefore unconsumed (README documents this).

Guarantees carried from project B's judge_runner, with the issue ids at each site:
  * leak-proof — each judge is shown ONLY the fields its rubric needs;
  * ISSUE-1 — JSON extraction scans for the last balanced object preferring one with a
    "score" key (never the first-{...}-to-last-} slice);
  * ISSUE-6 — response.content is a LIST of blocks under extended thinking; only real
    `text` blocks are concatenated, never str()'d. providers.py already collapses content
    structurally, so this is defence in depth;
  * judge retry (cfg.judge_retry_rounds) for transient parse/no-score failures;
  * per-(sample, metric) caching in the same crash-safe CellCache as the RAGAS phase.

Score semantics (all 0–1, no rescaling):
  completeness        the rubric's own weighted score, as reported by the judge;
  response_quality    mean of the four 0–1 dimensions (recomputed from the dimensions
                      defensively — the reported mean is not trusted blindly);
  sources_validation  0.0 if fabrication, else (recall + precision) / 2 — the two
                      sub-scores and the fabrication flag are kept on the record.
"""
from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from .attribution import build_hint
from .watchdog import heartbeat

JUDGE_METRICS = ["completeness", "response_quality", "sources_validation"]
JUDGE_LABEL = {"completeness": "Completeness", "response_quality": "Response Quality",
               "sources_validation": "Sources Validation"}
# md section header -> canonical name; Hallucination deliberately absent (dropped).
SECTION_TO_CANON = {"Completeness": "completeness",
                    "Response Quality": "response_quality",
                    "Sources Validation": "sources_validation"}

_SYSTEM = (
    "You are a meticulous, impartial evaluator of an official Government of India ministry "
    "chatbot (Ministry of Electronics and Information Technology, parliamentary Q&A). Score "
    "exactly one metric for one item using the frozen rubric provided, judging only what "
    "the rubric asks and basing every judgement solely on the fields shown. All scores are "
    "on a 0-1 scale. Return ONLY a single JSON object matching the OUTPUT schema and "
    "nothing else.")

# Shown to the Sources-Validation judge when a local KB document corpus IS configured
# (cfg.kb_dir exists) - the mechanical membership check is authoritative.
_SOURCES_AMENDMENT_WITH_KB = (
    "AMENDMENT to the rubric above: the attribution_hint's `fabricated_not_in_kb` list is "
    "a mechanical membership check against the configured KB document union and is "
    "authoritative — if it is non-empty the fabrication flag is true and the score is 0. "
    "The row's expected_sources remains the scoring truth for required/acceptable. When "
    "judging whether a citation's document actually contains the claim it is attached to, "
    "use the retrieved context excerpts shown; if they give no evidence either way for a "
    "citation, do not penalise on that sub-check.")

# Shown instead when no KB document corpus is configured. The judge cannot mechanically
# check fabrication, so it must not zero the score on that basis; recall/precision still
# score normally from expected_sources vs cited_sources.
_SOURCES_AMENDMENT_NO_KB = (
    "AMENDMENT to the rubric above: no local KB document corpus is configured for this "
    "project, so `fabricated_not_in_kb` is null, not an empty list - this does NOT mean "
    "nothing is fabricated, it means fabrication cannot be mechanically checked this run. "
    "Set the `fabrication` field to false and score recall/precision normally from "
    "expected_sources vs cited_sources; do not zero the score for a missing KB. When "
    "judging whether a citation's document actually contains the claim it is attached to, "
    "use the retrieved context excerpts shown; if they give no evidence either way for a "
    "citation, do not penalise on that sub-check.")


# ---------------------------------------------------------------------------
# verbatim rubric extraction from judge_prompts_v1.md
# ---------------------------------------------------------------------------
def parse_judge_prompts(md: str) -> dict:
    """{canonical: verbatim rubric text} for the three ACTIVE judges. The rubric is the
    fenced block under each `## <name>` heading, byte-for-byte (only the fence markers
    stripped). Raises if any active judge's section or fence is missing."""
    out: dict = {}
    for sec in re.split(r"(?m)^## ", md)[1:]:
        header = sec.splitlines()[0].strip()
        # normalise "Hallucination (absence of)" -> not in map -> skipped (dropped judge)
        canon = SECTION_TO_CANON.get(header)
        if canon is None:
            continue
        m = re.search(r"```\n(.*?)```", sec, re.DOTALL)
        if not m:
            raise ValueError(f"judge_prompts_v1.md: no fenced rubric under '## {header}'")
        out[canon] = m.group(1).rstrip("\n")
    missing = [m for m in JUDGE_METRICS if m not in out]
    if missing:
        raise ValueError(f"judge_prompts_v1.md is missing rubric section(s): {missing}")
    return out


# ---------------------------------------------------------------------------
# per-metric field views (leak-proof) + output schemas
# ---------------------------------------------------------------------------
def fields_for(canon: str, item: dict, hint: Optional[dict]) -> list:
    """[(field_name, value)] shown to this judge — nothing more."""
    prior = item.get("prior_turn")
    context = ([("earlier_turn_in_this_conversation (user_input below is a follow-up to it, "
                 "asked in the same chat; judge the response as an answer to the follow-up)",
                 {"user_input": prior["user_input"], "response": prior["response"]})]
               if prior else [])
    if canon == "completeness":
        return context + [("user_input", item["user_input"]),
                          ("expected_behavior", item["expected_behavior"]),
                          ("must_include_facts", item["must_include_facts"]),
                          ("response", item["response"])]
    if canon == "response_quality":
        fields = context + [("user_input", item["user_input"]),
                  ("expected_behavior", item["expected_behavior"]),
                  ("unsupported_aspects", item["unsupported_aspects"]),
                  ("response", item["response"])]
        if item["expected_behavior"] == "answer_partial" and not item["unsupported_aspects"]:
            # GD-V1 partial rows never had unsupported_aspects; the gap statements inside
            # `reference` carry the expectation (S10 ROUTING) — show it only then.
            fields.append(("reference (carries the gap statements for this partial row)",
                           item["reference"]))
        return fields
    if canon == "sources_validation":
        return [("expected_behavior", item["expected_behavior"]),
                ("expected_sources", item["expected_sources"]),
                ("cited_sources", item["cited_sources"]),
                ("attribution_hint", hint or {}),
                ("response", item["response"]),
                ("retrieved_contexts (evidence for the claim-support sub-check)",
                 item["retrieved_contexts"])]
    raise KeyError(canon)


_SCHEMAS = {
    "completeness": {
        "metric": "Completeness", "id": "<id>", "score": "<0-1 float per the rubric formula>",
        "per_fact": [{"fact": "<first 12 words>", "weight": "core|supporting",
                      "verdict": "PRESENT|PARTIAL|ABSENT"}],
        "absent_core": ["<each ABSENT core fact, quoted>"],
        "justification": "<2-4 sentences>"},
    "response_quality": {
        "metric": "Response Quality", "id": "<id>",
        "dimensions": {"register": "<0-1>", "structure": "<0-1>", "behaviour": "<0-1>",
                       "figures": "<0-1>"},
        "score": "<mean of the four dimensions>", "weakest_dimension": "<name>",
        "justification": "<2-4 sentences>"},
    "sources_validation": {
        "metric": "Sources Validation", "id": "<id>", "recall": "<0-1>", "precision": "<0-1>",
        "fabrication": "<true|false>",
        "claim_support_issues": ["<citation that does not contain its claim, if evidenced>"],
        "justification": "<2-4 sentences>"},
}


def assemble_prompt(canon: str, rubric: str, item: dict, hint: Optional[dict]) -> str:
    lines = [f"METRIC: {JUDGE_LABEL[canon]}", "RUBRIC (frozen, judge_prompts v1):", rubric]
    if canon == "sources_validation":
        kb_configured = bool(hint) and hint.get("fabricated_not_in_kb") is not None
        lines += ["", _SOURCES_AMENDMENT_WITH_KB if kb_configured else _SOURCES_AMENDMENT_NO_KB]
    lines += ["", f"ITEM ID: {item['id']}", "", "FIELDS SHOWN:"]
    for fn, val in fields_for(canon, item, hint):
        lines.append(f"{fn}:")
        if isinstance(val, (list, dict)):
            lines.append(json.dumps(val, ensure_ascii=False, indent=1))
        else:
            lines.append(str(val))
        lines.append("")
    lines += ["OUTPUT (JSON only):", json.dumps(_SCHEMAS[canon], ensure_ascii=False)]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ISSUE-6 / ISSUE-1 (project B) — content flattening + robust JSON extraction
# ---------------------------------------------------------------------------
def _content_to_text(content) -> str:
    """Flatten a LangChain message `content` to the model's TEXT (B's ISSUE-6 fix).
    providers.py already collapses content to a plain string structurally; this stays as
    defence in depth for any path that hands us raw block lists."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "reasoning_content" or "reasoning_content" in block:
                    continue
                if isinstance(block.get("text"), str):
                    parts.append(block["text"])
            else:
                t = getattr(block, "text", None)
                if isinstance(t, str):
                    parts.append(t)
        return "\n".join(p for p in parts if p)
    return str(content)


def _extract_json(text: str) -> dict:
    """Extract the verdict JSON robustly (B's ISSUE-1 fix): strip code fences, scan every
    `{` with raw_decode keeping the last object that parses, preferring one carrying a
    'score' or 'recall' key; guarded single-quote fallback as last resort."""
    if not text or not text.strip():
        raise ValueError("empty judge output")
    cleaned = re.sub(r"```[a-zA-Z]*", "", text).replace("```", "")

    dec = json.JSONDecoder()
    best = None
    i = 0
    while True:
        i = cleaned.find("{", i)
        if i == -1:
            break
        try:
            obj, end = dec.raw_decode(cleaned, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(obj, dict):
            if "score" in obj or "recall" in obj:
                best = obj
            elif best is None or not ("score" in best or "recall" in best):
                best = obj
        i = max(end, i + 1)
    if best is not None:
        return best

    a, b = cleaned.find("{"), cleaned.rfind("}")
    if a != -1 and b > a:
        frag = cleaned[a:b + 1]
        if '"' not in frag and "'" in frag:
            try:
                got = json.loads(frag.replace("'", '"'))
                if isinstance(got, dict):
                    return got
            except json.JSONDecodeError:
                pass
    raise ValueError("no JSON object in judge output")


def _as_unit(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if 0.0 <= x <= 1.0 else None


def coerce(raw: dict, canon: str, sid: str, hint: Optional[dict]) -> dict:
    """Validate + derive the cell score from the judge's JSON. Never raises."""
    rec = {"id": sid, "metric": canon, "score": None, "na_reason": None,
           "justification": str(raw.get("justification") or ""), "detail": {}}
    if canon == "completeness":
        rec["score"] = _as_unit(raw.get("score"))
        rec["detail"] = {"per_fact": raw.get("per_fact") or [],
                         "absent_core": raw.get("absent_core") or []}
    elif canon == "response_quality":
        dims = raw.get("dimensions") or {}
        vals = [_as_unit(dims.get(k)) for k in ("register", "structure", "behaviour", "figures")]
        if all(v is not None for v in vals):
            rec["score"] = round(sum(vals) / 4.0, 4)   # recompute; never trust the mean blindly
        else:
            rec["score"] = _as_unit(raw.get("score"))
        rec["detail"] = {"dimensions": dims,
                         "weakest_dimension": raw.get("weakest_dimension")}
    elif canon == "sources_validation":
        recall, precision = _as_unit(raw.get("recall")), _as_unit(raw.get("precision"))
        fab = raw.get("fabrication")
        fab = bool(fab) if isinstance(fab, bool) else str(fab).strip().lower() == "true"
        # The deterministic KB check is authoritative — a judge cannot un-fabricate.
        # `fabricated_not_in_kb` is None (falsy) when no local KB corpus is configured
        # (attribution.py's build_hint) - this `if` correctly does NOT fire in that case,
        # since None is distinct from "a real KB confirms nothing is fabricated" ([]).
        if hint and hint.get("fabricated_not_in_kb"):
            fab = True
        if fab:
            rec["score"] = 0.0
        elif recall is not None and precision is not None:
            rec["score"] = round((recall + precision) / 2.0, 4)
        rec["detail"] = {"recall": recall, "precision": precision, "fabrication": fab,
                         "claim_support_issues": raw.get("claim_support_issues") or [],
                         "fabricated_not_in_kb": (hint or {}).get("fabricated_not_in_kb", [])}
    if rec["score"] is None:
        rec["na_reason"] = "unparseable_or_out_of_range_score"
    return rec


# ---------------------------------------------------------------------------
# one judgement + the run
# ---------------------------------------------------------------------------
def _judge_one(llm, item: dict, canon: str, rubric: str, hint: Optional[dict],
               rounds: int) -> dict:
    """One judgement, retried up to `rounds` times on parse/no-score failure (B's ISSUE-6
    retry). Never raises — returns the last error record if all attempts fail."""
    from langchain_core.messages import HumanMessage, SystemMessage
    prompt = assemble_prompt(canon, rubric, item, hint)
    last = None
    for _ in range(max(1, rounds)):
        t0 = time.monotonic()
        try:
            resp = llm.invoke([SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)])
            text = _content_to_text(resp.content)
            rec = coerce(_extract_json(text), canon, item["id"], hint)
            usage = getattr(resp, "usage_metadata", None) or {}
            rec["in_tokens"] = int(usage.get("input_tokens", 0))
            rec["out_tokens"] = int(usage.get("output_tokens", 0))
            rec["seconds"] = round(time.monotonic() - t0, 2)
            if rec["score"] is not None:
                heartbeat(f"{item['id']}/{canon} judged", cell=True)
                return rec
            last = rec
        except Exception as e:                    # transient / parse failure — retry
            last = {"id": item["id"], "metric": canon, "score": None,
                    "na_reason": f"error: {e}", "justification": f"error: {e}",
                    "detail": {}, "in_tokens": 0, "out_tokens": 0,
                    "seconds": round(time.monotonic() - t0, 2)}
    heartbeat(f"{item['id']}/{canon} NA", cell=True)
    return last


def load_rubrics(cfg) -> dict:
    return parse_judge_prompts(cfg.judge_prompts_md.read_text(encoding="utf-8"))


def run_judge(cfg, samples, cache, kb_names: Optional[set], metrics=None, progress=None,
              max_workers: Optional[int] = None):
    """Run the three judges over `samples` (ALL routes — judges see every row).
    Returns (records, n_computed). Cached cells are free."""
    metrics = metrics or JUDGE_METRICS
    if max_workers is None:
        max_workers = getattr(cfg, "judge_max_workers", 6)
    rubrics = load_rubrics(cfg)

    hints = {s.id: build_hint(s, kb_names) for s in samples}
    items = {s.id: s.judge_item() for s in samples}

    results: dict = {}
    tasks = []
    for s in samples:
        for m in metrics:
            cached = cache.get(s.id, m)
            if cached is not None:
                results[(s.id, m)] = cached
                continue
            tasks.append((s.id, m))

    if tasks:
        from .providers import get_llm
        llm = get_llm(cfg)
        lock = threading.Lock()
        done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_judge_one, llm, items[sid], m, rubrics[m], hints[sid],
                              cfg.judge_retry_rounds): (sid, m) for sid, m in tasks}
            for fut in as_completed(futs):
                rec = fut.result()
                stored = cache.set(rec["id"], rec["metric"], rec["score"], rec.get("na_reason"),
                                   failure_class=None if rec["score"] is not None else "judge_failure",
                                   justification=rec.get("justification", ""),
                                   detail=rec.get("detail", {}),
                                   in_tokens=rec.get("in_tokens", 0),
                                   out_tokens=rec.get("out_tokens", 0),
                                   seconds=rec.get("seconds"),
                                   judge_effort=getattr(cfg, "thinking_effort", None))
                results[(rec["id"], rec["metric"])] = stored
                with lock:
                    done += 1
                    if progress:
                        progress(done, len(tasks), stored)

    ordered = [results[(s.id, m)] for s in samples for m in metrics]
    return ordered, len(tasks)


def run_repeats(cfg, samples, kb_names: Optional[set], metrics=None, repeats: int = 3, log=None):
    """Uncached judge repeats for the smoke stability check."""
    metrics = metrics or JUDGE_METRICS
    rubrics = load_rubrics(cfg)
    from .providers import get_llm
    llm = get_llm(cfg)
    log = log or (lambda m: None)
    out: dict = {}
    work = [(r, s, m) for r in range(1, repeats + 1) for s in samples for m in metrics]
    with ThreadPoolExecutor(max_workers=getattr(cfg, "judge_max_workers", 6)) as ex:
        futs = {ex.submit(_judge_one, llm, s.judge_item(), m, rubrics[m],
                          build_hint(s, kb_names), cfg.judge_retry_rounds): (r, s.id, m)
                for r, s, m in work}
        for fut in as_completed(futs):
            r, sid, m = futs[fut]
            rec = fut.result()
            sc = rec.get("score")
            log(f"  rep{r} {'ok ' if sc is not None else 'NA '} {sid:11} {m:22} "
                f"{'' if sc is None else f'{sc:.4f}'}")
            out.setdefault((sid, m), []).append((r, rec))
    return {k: [rec for _, rec in sorted(v)] for k, v in out.items()}
