"""MeitY PQA RAGAS + LLM-judge evaluation framework — CLI.

    .venv/Scripts/python.exe cli.py --validate         # schema + routing + parity gate (no API; parity
                                                        # self-skips until this project freezes one)
    .venv/Scripts/python.exe cli.py --estimate-cost    # per-phase token/$ estimate (no API)
    .venv/Scripts/python.exe cli.py --connectivity     # 1 small call: creds, model, thinking on?
    .venv/Scripts/python.exe cli.py --smoke --ids GD-MTY-LS17-2-U2536-V --repeats 3   # gated smoke
    .venv/Scripts/python.exe cli.py --all              # full 66-row run (paid — launch only when asked)
    .venv/Scripts/python.exe cli.py --resume           # same as --all; cached cells cost nothing
    .venv/Scripts/python.exe cli.py --ragas-only | --judge-only
    .venv/Scripts/python.exe cli.py --report RUN_DIR   # rebuild reports from the cache (no API)
    .venv/Scripts/python.exe cli.py --clear-na         # drop retryable NA cells, then --resume
    .venv/Scripts/python.exe cli.py --compare RUN_A RUN_B | --dashboard RUN_DIR | --excel RUN_DIR

Any paid command takes `--ids GD-MTY-LS17-2-U2536-V,GD-MTY-RS-252-U1340-V` and `--metrics a,b`
to subset; `--log FILE` tees stdout+stderr live. Cost is tracked and reported, never gated.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# This file prints Unicode (checkmarks, em-dashes, ellipses) throughout. On Windows,
# stdout/stderr otherwise default to the console's legacy codepage (cp1252), which
# raises UnicodeEncodeError on the first such character - not a hypothetical, this was
# hit for real running --validate on Windows. reconfigure() is a no-op where stdout is
# already UTF-8 capable (e.g. most Unix terminals), so this is safe everywhere.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:  # pragma: no cover - best-effort; never block CLI startup on this
            pass

from config import load_config                                    # noqa: E402
import ragas_eval                                                 # noqa: E402,F401 (applies patches)
from ragas_eval import aggregate as agg                           # noqa: E402
from ragas_eval import costs as C                                 # noqa: E402
from ragas_eval import dataset as ds                              # noqa: E402
from ragas_eval import judge_runner as J                          # noqa: E402
from ragas_eval import ragas_runner as R                          # noqa: E402
from ragas_eval import watchdog                                   # noqa: E402
from ragas_eval.cache import CellCache                            # noqa: E402
from ragas_eval.dashboard import build_dashboard, write_dashboard  # noqa: E402
from ragas_eval.excel_report import write_excel_report            # noqa: E402
from ragas_eval.pacing import AdaptiveLimiter                     # noqa: E402
from ragas_eval.report import write_reports                       # noqa: E402


# ---------------------------------------------------------------------------
class _Tee:
    def __init__(self, stream, fh):
        self._s, self._f = stream, fh

    def write(self, s):
        self._s.write(s); self._s.flush(); self._f.write(s); self._f.flush()
        return len(s)

    def flush(self):
        self._s.flush(); self._f.flush()

    def isatty(self):
        return getattr(self._s, "isatty", lambda: False)()


def _setup_log(path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fh = open(p, "w", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, fh)
    sys.stderr = _Tee(sys.__stderr__, fh)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
def _split_metrics(names):
    """Split a --metrics list into (ragas_subset, judge_subset); None -> full sets."""
    if not names:
        return list(R.RAGAS_METRICS), list(J.JUDGE_METRICS)
    unknown = [n for n in names if n not in R.BY_NAME and n not in J.JUDGE_METRICS]
    if unknown:
        raise SystemExit(f"[error] unknown metric(s): {unknown}; known: "
                         f"{R.RAGAS_METRICS + J.JUDGE_METRICS}")
    return ([m for m in R.RAGAS_METRICS if m in names],
            [m for m in J.JUDGE_METRICS if m in names])


def load_and_check(cfg, only_ids=None, skip_parity=False):
    """Parity gate + load + routing. Raises on any breach — never scores stale inputs.

    TEMPLATE: also skips the gate (regardless of `skip_parity`) when
    cfg.parity_gate_available is False, i.e. this project has never done its own freeze
    (no freeze_manifest.json) - there's nothing to compare against yet, so the gate would
    otherwise crash on a FileNotFoundError for every single run. The moment a real
    freeze_manifest.json exists, this starts enforcing automatically.
    """
    parity = {} if (skip_parity or not cfg.parity_gate_available) else ds.parity_gate(cfg)
    samples = ds.load_samples()
    routing = ds.routing(samples)
    if only_ids:
        known = {s.id for s in samples}
        unknown = [i for i in only_ids if i not in known]
        if unknown:
            raise SystemExit(f"[error] unknown ids: {unknown}")
        samples = [s for s in samples if s.id in set(only_ids)]
    return samples, parity, routing


def _run_dir(cfg, smoke=False, name=None) -> Path:
    import re
    if smoke:
        d = cfg.runs_dir / "smoke" / (name or datetime.now().strftime("run_%Y%m%d_%H%M%S"))
        d.mkdir(parents=True, exist_ok=True)
        return d
    mx = 0
    if cfg.runs_dir.exists():
        for p in cfg.runs_dir.iterdir():
            m = re.match(r"run_v(\d+)$", p.name)
            if p.is_dir() and m:
                mx = max(mx, int(m.group(1)))
    d = cfg.runs_dir / f"run_v{mx + 1}" / (name or datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_for(cfg, dhash) -> CellCache:
    return CellCache(cfg.cache_dir / "cells.jsonl",
                     model_id=cfg.bedrock_model_id, dhash=dhash, variant="full")


def _print_routing(routing):
    c = routing["counts"]
    print(f"[route]  {c['full_ragas']} full-RAGAS / {c['partial_ragas']} partial-RAGAS / "
          f"{c['judge_only']} judge-only")
    if c["partial_ragas"]:
        print(f"[route]  ⚠ partial-RAGAS rows (EMPTY retrieved_contexts — Faithfulness / "
              f"Context Precision / Context Recall will be NA 'empty retrieved_contexts', "
              f"never 0.0):")
        for rid in routing["routes"]["partial_ragas"]:
            print(f"           - {rid}")
        print(f"[route]    (re-capture retrieval for these to lift the carve-out; "
              f"validate would then print "
              f"{c['full_ragas'] + c['partial_ragas']}/0/{c['judge_only']})")
    print(f"[route]  judge-only abstains: {', '.join(routing['routes']['judge_only'])}")
    for note in routing.get("notes", []):
        print(f"[route]  ⚠ {note}")


# ---------------------------------------------------------------------------
def cmd_validate(cfg, args) -> int:
    print("=" * 78)
    print("VALIDATE — schema, routing, joins, and the parity gate (no API calls)")
    print("=" * 78)

    payload = ds.load_payload(cfg.unified_eval_json)
    print(f"[file]   note + amendments blocks present; n_rows={payload['n_rows']}")

    samples = ds.load_samples()
    print(f"[load]   {len(samples)} rows parsed "
          f"({sum(1 for s in samples if s.provenance == 'part1-ingested')} part1-ingested + "
          f"{sum(1 for s in samples if s.provenance == 'part2-not-ingested')} part2-not-ingested)")
    print("[load]   structural invariants OK (keys, types, id uniqueness)")

    routing = ds.routing(samples)
    _print_routing(routing)

    # TEMPLATE: sourced from Sample.unsupported_aspects_source (set explicitly in
    # load_samples), not inferred from "has unsupported_aspects + no erratum" - that
    # inference broke the moment a project's rows started carrying the field directly
    # (this project's do). See dataset.py's Sample dataclass for why.
    from_own_row = sorted(s.id for s in samples if s.unsupported_aspects_source == "own_row")
    from_join = sorted(s.id for s in samples if s.unsupported_aspects_source == "golden_v2_join")
    from_erratum = sorted(s.id for s in samples if s.unsupported_aspects_source == "erratum")
    if from_own_row:
        print(f"[join]   unsupported_aspects read directly from the row for: "
              f"{len(from_own_row)} row(s) (this project's dataset carries them inline, "
              f"no join needed)")
    # A row can be BOTH in the documented join list and erratum-corrected (GD-0005); the erratum
    # wins, so it is expected to appear as erratum-sourced rather than join-sourced.
    expected_join = set(ds.UNSUPPORTED_JOIN_IDS) - set(from_erratum)
    print(f"[join]   unsupported_aspects joined from golden_v2 for: {from_join} "
          f"(expected {sorted(expected_join)}"
          + (f"; {sorted(set(ds.UNSUPPORTED_JOIN_IDS) & set(from_erratum))} superseded by an "
             f"erratum" if set(ds.UNSUPPORTED_JOIN_IDS) & set(from_erratum) else "") + ")")
    if set(from_join) != expected_join:
        print("[join]   ✗ JOIN MISMATCH")
        return 1
    for rid in from_erratum:
        s = next(x for x in samples if x.id == rid)
        print(f"[join]   {rid}: {len(s.unsupported_aspects)} unsupported_aspects read from its "
              f"REGISTERED ERRATUM block ({s.erratum.get('date')}) — golden_v2 cannot supply "
              f"them for a row it still records differently")
    errata = [s for s in samples if s.erratum]
    if errata:
        print(f"[erratum] {len(errata)} registered erratum row(s) — deliberate, documented "
              f"correction recorded on the row:")
        for s in errata:
            # Two erratum shapes exist. The S11-era blocks carry a `was` snapshot of the fields
            # that changed; the S13-era blocks (2026-08-20) carry `defect`/`correction` prose
            # instead, because what they fix — unweighted facts, a stale figure, an untrue
            # stratum label — is not a behaviour/stratum swap. Print whichever is present.
            was = s.erratum.get("was")
            if was:
                detail = (f"now {s.expected_behavior}/{s.stratum}; "
                          f"was {was.get('expected_behavior')}/{was.get('stratum')}")
            else:
                detail = str(s.erratum.get("defect") or
                             s.erratum.get("correction") or "see row")
                detail = detail[:96] + ("…" if len(detail) > 96 else "")
            print(f"           - {s.id} ({s.erratum.get('date')}): {detail}")

    if not cfg.parity_gate_available:
        print("[parity] SKIPPED — no freeze_manifest.json yet for this project, so there's "
              "nothing to compare against (see config.py's parity_gate_available). This is "
              "expected until this project does its own freeze; not a failure.")
    else:
        print("[parity] recomputing the 4-point gate …")
        try:
            parity = ds.parity_gate(cfg)
        except ds.ParityError as e:
            print(f"\n{e}")
            return 1
        for name, info in parity.items():
            print(f"[parity] ✓ {name}: {info['sha'][:16]}…")

    warns = ds.validate(samples, expected_n=cfg.expected_sample_count)

    # This folder carries its own copy of every input (data/), so it can be shared and run
    # standalone. The parity gate above pins the sha recorded in THAT copy's freeze manifest,
    # so a copy left behind by later edits in the authoring kit would still pass it. Compare
    # against the kit when it happens to be present; say nothing when it is not.
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))
        import sync_inputs
        if sync_inputs.upstream_present():
            for what, ours, theirs in sync_inputs.drift():
                warns.append(f"data/ is BEHIND the authoring kit for {what}: this copy "
                             f"{ours[:12]}… vs kit {theirs[:12]}… — the run would evaluate "
                             f"the older inputs; `python3 tools/sync_inputs.py` to update")
    except Exception as exc:                                        # noqa: BLE001
        warns.append(f"could not compare data/ against the authoring kit: "
                     f"{type(exc).__name__}: {exc}")

    print(f"[warn]   {len(warns)} soft warning(s)")
    for w in warns:
        print(f"           - {w}")

    st = ds.stats(samples)
    print(f"[shape]  retrieved chunks {st['retrieved_chunks_total']} total "
          f"(max {st['retrieved_chunks_max']}/row); longest response "
          f"{st['response_chars_max']:,} chars; longest reference {st['reference_chars_max']:,}")

    print(f"[metric] RAGAS: {', '.join(R.RAGAS_METRICS)}")
    print(f"[metric] judges: {', '.join(J.JUDGE_METRICS)}  "
          f"(Hallucination dropped — forbidden_facts unconsumed)")
    rubrics = J.load_rubrics(cfg)
    print(f"[judge]  {len(rubrics)} verbatim rubrics parsed from {cfg.judge_prompts_md.name}")

    issues = cfg.issues()
    for i in issues:
        print(f"[config] note: {i}")
    print("\nVALIDATE PASSED" + (" (with config notes above)" if issues else ""))
    return 0


def cmd_estimate(cfg, args) -> int:
    samples, _, routing = load_and_check(cfg, args.ids, skip_parity=True)
    rm, jm = _split_metrics(args.metrics)
    est = C.estimate(cfg, samples, rm, jm)
    print(json.dumps(est, indent=2))
    print(f"\nPer-phase: RAGAS ${est['ragas']['usd']:.2f} over ~{est['ragas']['llm_calls']} "
          f"calls; judges ${est['judge']['usd']:.2f} over {est['judge']['llm_calls']} calls; "
          f"total ${est['total_usd']:.2f} for {est['samples']} row(s).")
    print("Informational only — no run is ever refused on cost.")
    return 0


def cmd_connectivity(cfg, args) -> int:
    from ragas_eval.providers import connectivity_smoke
    out = connectivity_smoke(cfg)
    print(json.dumps({k: v for k, v in out.items() if k != "raw_samples"}, indent=2))
    print("\nraw response blocks:")
    for s in out["raw_samples"]:
        print(f"  block_types={s['block_types']} reasoning_block={s['has_reasoning_block']} "
              f"stop={s['stop_reason']}")
    print(f"\nthinking active on this call: {out['thinking_active']}")
    return 0


# ---------------------------------------------------------------------------
def _finish(cfg, run_dir, samples, cache, parity, dhash, run_meta, routing,
            stats, limiter, ragas_usage, judge_usage, est, rm, jm, stability=None,
            kept=None):
    """`kept` (from cmd_report) carries the run's own cost/pacing/snapshot, so a rebuild
    from cache never overwrites measured spend and settings with the rebuild's zeros."""
    kept = kept or {}
    per_sample = agg.build_per_sample(samples, cache, rm + jm)
    agg.write_per_sample(per_sample, run_dir / "per_sample.jsonl")
    pacing = kept.get("pacing") or (limiter.snapshot() if limiter else {})
    cost = kept.get("cost") or C.build_cost_report(cfg, run_meta["run_id"], est,
                                                  ragas_usage, judge_usage)
    snapshot = kept.get("snapshot") or C.config_snapshot(
        cfg, run_meta["run_id"], dhash, [s.id for s in samples], rm, jm, parity,
        C.library_versions())
    paths = write_reports(run_dir, per_sample, cfg, run_meta=run_meta, routing=routing,
                          stats=stats, pacing=pacing, cost=cost, snapshot=snapshot,
                          stability=stability)
    dash = write_dashboard(run_dir, per_sample, cfg, run_meta=run_meta, cost=cost,
                           snapshot=snapshot, pacing=pacing, routing=routing,
                           stability=stability)
    xlsx = write_excel_report(run_dir, per_sample, samples, cfg, run_meta=run_meta,
                              cost=cost, snapshot=snapshot, pacing=pacing, routing=routing,
                              stability=stability)
    for p in (run_dir / "per_sample.jsonl", paths["aggregates_csv"], paths["summary_json"],
              paths["report_md"], paths.get("cost_json"), paths.get("config_snapshot"),
              dash, xlsx):
        if p:
            log(f"wrote {p}")
    return per_sample, paths, cost


def cmd_run(cfg, args, smoke=False, ragas_only=False, judge_only=False) -> int:
    samples, parity, routing = load_and_check(cfg, args.ids)
    rm, jm = _split_metrics(args.metrics)
    if ragas_only:
        jm = []
    if judge_only:
        rm = []
    dhash = ds.dataset_hash(cfg.unified_eval_json)
    run_dir = _run_dir(cfg, smoke=smoke)
    started = _now_iso()
    mode = "smoke" if smoke else ("ragas-only" if ragas_only else
                                  ("judge-only" if judge_only else
                                   ("resume" if args.resume else "all")))

    log(f"START --{mode}  run_dir={run_dir}")
    log(f"parity gate PASSED (unified sha, KB union fingerprint, judge-prompts sha, "
        f"golden_v2 lineage)")
    log(f"samples={len(samples)} ({', '.join(s.id for s in samples) if len(samples) <= 12 else 'all 41'})")
    _print_routing(routing)
    log(f"metrics: ragas={rm or '—'} judges={jm or '—'}  dataset_sha={dhash}")
    log(f"judge={cfg.bedrock_model_id} thinking={cfg.thinking_mode}@{cfg.thinking_effort} "
        f"max_tokens={cfg.ragas_max_tokens:,} (temperature unset — thinking implies 1)")
    for i in cfg.issues():
        log(f"config note: {i}")

    est = C.estimate(cfg, samples, rm, jm)
    log(f"cost estimate ${est['total_usd']:.2f} (RAGAS ${est['ragas']['usd']:.2f} + "
        f"judge ${est['judge']['usd']:.2f}) — informational only, never a gate")

    from ragas_eval.providers import read_usage, reset_usage, set_limiter, enable_raw_sampling
    limiter = AdaptiveLimiter(cfg, log=lambda m: log(m))
    set_limiter(limiter)
    enable_raw_sampling(8 if smoke else 3)

    cache = _cache_for(cfg, dhash)
    if cache.corrupt_lines:
        log(f"cache: skipped {cache.corrupt_lines} truncated journal line(s) from a prior "
            f"crash — every complete cell before them was recovered")
    log(f"cache: {len(cache)} cell(s) already on disk at {cache.path}")

    watchdog.start(cfg.stall_timeout, log=log)
    t0 = time.monotonic()
    stats = {"ragas": None, "judge": None}

    # --- phase 1: RAGAS ---
    ragas_usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0,
                   "reasoning_blocks": 0, "truncations": 0}
    # judge-only rows (the abstains) never enter the RAGAS phase — routing, not chance:
    # they carry a refusal response and near-miss chunks, so the empty-input partition
    # alone would not exclude them.
    ragas_rows = [s for s in samples if s.route != "judge_only"]
    if rm and ragas_rows:
        log(f"phase 1/2 — RAGAS ({len(rm)} metrics over {len(ragas_rows)} routed rows; "
            f"{len(samples) - len(ragas_rows)} judge-only row(s) excluded). "
            f"Empty-retrieval rows are partitioned to NA on the context metrics.")
        reset_usage()
        runner = R.CellRunner(cfg, ragas_rows, cache, log=lambda m: print(m, flush=True))
        stats["ragas"] = runner.run(rm)
        ragas_usage = read_usage()
        log(f"RAGAS: {ragas_usage['calls']:,} calls, in {ragas_usage['input_tokens']:,} tok, "
            f"out {ragas_usage['output_tokens']:,} tok, "
            f"truncations {ragas_usage['truncations']}")

    # --- phase 2: judges ---
    judge_usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0,
                   "reasoning_blocks": 0, "truncations": 0}
    if jm:
        log(f"phase 2/2 — LLM judges ({len(jm)}: {', '.join(jm)}) over ALL routed rows, "
            f"abstains included.")
        reset_usage()
        kb_names = ds.kb_document_names(cfg)

        def _progress(done, total, rec):
            log(f"judge {done}/{total}: {rec['id']} {rec['metric']} -> "
                f"score={rec.get('score')} na={rec.get('na_reason') or '-'}")

        _, n_judge = J.run_judge(cfg, samples, cache, kb_names, metrics=jm,
                                 progress=_progress)
        judge_usage = read_usage()
        stats["judge"] = {"computed": n_judge,
                          "cached": len(samples) * len(jm) - n_judge}
        log(f"judges: computed {n_judge} cell(s); {judge_usage['calls']:,} calls, "
            f"truncations {judge_usage['truncations']}")

    elapsed = time.monotonic() - t0
    stats["elapsed_s"] = round(elapsed, 1)
    log(f"scoring finished in {elapsed / 60:.1f} min")
    pac = limiter.snapshot()
    log(f"pacing: {pac['throttles']} throttle(s), {pac['transient_errors']} transient, "
        f"{pac['total_wait_s']:.0f}s total wait, max concurrency "
        f"{pac['max_concurrency_reached']}")

    # --- stability repeats (smoke gate) ---
    stability = None
    if smoke and args.repeats > 1:
        log(f"stability check: {len(samples)} sample(s) x metrics x {args.repeats} repeats "
            f"(uncached)")
        reps = {}
        if rm and ragas_rows:
            runner = R.CellRunner(cfg, ragas_rows, cache, log=lambda m: print(m, flush=True))
            reps.update(runner.run_repeats(rm, args.repeats))
        if jm:
            kb_names = ds.kb_document_names(cfg)
            reps.update(J.run_repeats(cfg, samples, kb_names, metrics=jm,
                                      repeats=args.repeats,
                                      log=lambda m: print(m, flush=True)))
        stability = _stability_rows(reps, cfg, args.repeats)
        (run_dir / "stability.json").write_text(json.dumps(stability, indent=2),
                                                encoding="utf-8")
        log(f"stability: {stability['n_exceeding']} cell(s) exceed the "
            f"{cfg.stability_tolerance} tolerance")

    watchdog.stop()
    run_meta = {"run_id": run_dir.name, "mode": mode, "started": started,
                "finished": _now_iso()}
    per_sample, paths, cost = _finish(cfg, run_dir, samples, cache, parity, dhash,
                                      run_meta, routing, stats, limiter,
                                      ragas_usage, judge_usage, est, rm, jm, stability)
    cache.close()

    o = paths["composite"]["overall"]
    log(f"COMPOSITE {o['composite_mean']} over {o['n_scored']}/{o['n_total']} samples "
        f"(read the provenance split in report.md before quoting this)")
    log(f"cost measured ${cost['actual']['total_usd']:.2f} "
        f"(estimate ${cost['estimate']['total_usd']:.2f}) — reported, never gated")
    log(f"DONE --{mode}")
    return 0


def _stability_rows(reps: dict, cfg, repeats: int) -> dict:
    rows = []
    for (sid, metric), recs in sorted(reps.items()):
        scores = [r.get("score") for r in recs]
        got = [s for s in scores if s is not None]
        mean = round(statistics.fmean(got), 4) if got else None
        spread = round(max(got) - min(got), 4) if len(got) > 1 else (0.0 if got else None)
        exceeds = bool(spread is not None and spread > cfg.stability_tolerance)
        rows.append({"id": sid, "metric": metric, "scores": scores, "mean": mean,
                     "spread": spread, "exceeds": exceeds,
                     "na": [r.get("na_reason") for r in recs if r.get("score") is None]})
    rows.sort(key=lambda r: (-(r["spread"] or 0), r["id"], r["metric"]))
    return {"repeats": repeats, "tolerance": cfg.stability_tolerance,
            "n_exceeding": sum(1 for r in rows if r["exceeds"]), "rows": rows}


def cmd_report(cfg, args) -> int:
    run_dir = Path(args.report)
    if not run_dir.exists():
        raise SystemExit(f"[error] no such run directory: {run_dir}")
    samples, parity, routing = load_and_check(cfg, args.ids)
    rm, jm = _split_metrics(args.metrics)
    dhash = ds.dataset_hash(cfg.unified_eval_json)
    cache = _cache_for(cfg, dhash)
    stab_path = run_dir / "stability.json"
    stability = json.loads(stab_path.read_text()) if stab_path.exists() else None
    est = C.estimate(cfg, samples, rm, jm)
    run_meta = {"run_id": run_dir.name, "mode": "report", "finished": _now_iso()}
    zero = {"input_tokens": 0, "output_tokens": 0, "calls": 0,
            "reasoning_blocks": 0, "truncations": 0}
    kept = {}
    for key, name in (("cost", "cost.json"), ("snapshot", "config_snapshot.json")):
        p = run_dir / name
        if p.exists():
            kept[key] = json.loads(p.read_text(encoding="utf-8"))
    summ = run_dir / "summary.json"
    if summ.exists():
        kept["pacing"] = json.loads(summ.read_text(encoding="utf-8")).get("pacing") or None
    _finish(cfg, run_dir, samples, cache, parity, dhash, run_meta, routing,
            {"rebuilt_from_cache": True}, None, zero, zero, est, rm, jm, stability, kept=kept)
    cache.close()
    return 0


def cmd_clear_na(cfg, args) -> int:
    """Drop retryable NA cells so the next run recomputes them (A's --clear-na; terminal
    'skipped:' NAs — the empty-retrieval partition — are never dropped)."""
    samples, _, _ = load_and_check(cfg, args.ids, skip_parity=True)
    rm, jm = _split_metrics(args.metrics)
    dhash = ds.dataset_hash(cfg.unified_eval_json)
    cache = _cache_for(cfg, dhash)
    n = 0
    for s in samples:
        for m in rm + jm:
            rec = cache.get(s.id, m)
            if rec and rec.get("score") is None and R.is_retryable_na(rec.get("na_reason")):
                cache.drop(s.id, m)
                print(f"  dropped {s.id} / {m}  ({rec.get('failure_class')})")
                n += 1
    cache.compact()
    cache.close()
    print(f"cleared {n} NA cell(s) — re-run with --resume to recompute only those")
    return 0


def cmd_clear_cells(cfg, args) -> int:
    """Drop cells unconditionally — scored ones included — so the next run recomputes them.

    Needed because the cache key covers `sample_id | metric | model_id | dataset_hash |
    variant` and NOT the thinking effort or the judge prompt mode (ISSUE-P10). Changing
    `CTX_PRECISION_JUDGE` or the effort therefore leaves the old cells looking like valid
    hits, and `--resume` will happily return them: the column silently mixes regimes. The
    only correct move is to drop the affected column first, which is what this does.

    `cache.py` has documented this recovery path since the cell store was written, but the
    flag was never implemented — found while writing RCA_CONTEXT_PRECISION.md (2026-08-20).

    Scope it with --metrics (and --ids); with neither, it refuses rather than nuking the
    whole cache by accident.
    """
    samples, _, _ = load_and_check(cfg, args.ids, skip_parity=True)
    rm, jm = _split_metrics(args.metrics)
    metrics = rm + jm
    if not args.metrics and not args.ids:
        print("[clear-cells] refusing to drop EVERY cell. Scope it with --metrics and/or "
              "--ids, e.g. --clear-cells --metrics context_precision")
        return 1
    dhash = ds.dataset_hash(cfg.unified_eval_json)
    cache = _cache_for(cfg, dhash)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup = cache.path.with_suffix(f".jsonl.bak_{stamp}")
    if cache.path.exists():
        backup.write_bytes(cache.path.read_bytes())
        print(f"[clear-cells] cache backed up -> {backup.name}")
    n = 0
    for s in samples:
        for m in metrics:
            rec = cache.get(s.id, m)
            if rec is None:
                continue
            mode = rec.get("judge_ctx_precision") if m == "context_precision" else None
            cache.drop(s.id, m)
            print(f"  dropped {s.id} / {m}  (score={rec.get('score')}"
                  + (f", prompt={mode}" if mode else "")
                  + f", effort={rec.get('judge_effort')})")
            n += 1
    cache.compact()
    cache.close()
    print(f"cleared {n} cell(s) across {len(metrics)} metric(s) — re-run with --resume to "
          f"recompute exactly those")
    return 0


def cmd_compare(args) -> int:
    from ragas_eval.compare import compare_runs, format_compare
    try:
        diff = compare_runs(args.compare[0], args.compare[1])
    except FileNotFoundError as e:
        raise SystemExit(f"[compare] {e}")
    text = format_compare(diff)
    print(text)
    out = Path(args.compare[1]) / "compare.md"
    out.write_text(text + "\n", encoding="utf-8")
    print(f"\n[compare] wrote {out}")
    return 0 if diff["comparable"] else 0


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="MeitY PQA RAGAS + LLM-judge evaluation framework")
    ap.add_argument("--validate", action="store_true",
                    help="schema + routing + parity gate (no API)")
    ap.add_argument("--estimate-cost", action="store_true",
                    help="per-phase token/$ estimate (no API)")
    ap.add_argument("--connectivity", action="store_true",
                    help="one small call: creds, model id, and whether thinking is live")
    ap.add_argument("--smoke", action="store_true",
                    help="gated smoke run into runs/smoke/ (paid; use --ids + --repeats)")
    ap.add_argument("--all", action="store_true", help="full run (paid)")
    ap.add_argument("--resume", action="store_true",
                    help="same as --all; cached cells are free, only gaps cost anything")
    ap.add_argument("--ragas-only", action="store_true", help="RAGAS phase only (paid)")
    ap.add_argument("--judge-only", action="store_true", help="judge phase only (paid)")
    ap.add_argument("--report", metavar="RUN_DIR", help="rebuild reports from the cache (no API)")
    ap.add_argument("--clear-cells", action="store_true",
                    help="drop cells outright (scored included) so --resume recomputes them; "
                         "REQUIRED after changing thinking effort or a judge prompt mode, "
                         "which the cache key does not cover (ISSUE-P10). Scope with "
                         "--metrics/--ids.")
    ap.add_argument("--clear-na", action="store_true",
                    help="drop retryable NA cells so --resume recomputes them (no API)")
    ap.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"),
                    help="diff two runs' summary.json (no API)")
    ap.add_argument("--dashboard", metavar="RUN_DIR",
                    help="rebuild dashboard.html for an existing run (no API)")
    ap.add_argument("--excel", metavar="RUN_DIR",
                    help="rebuild evaluation_report.xlsx for an existing run (no API)")
    ap.add_argument("--ids", default=None, help="comma-separated sample ids to subset")
    ap.add_argument("--metrics", default=None, help="comma-separated metric names to subset")
    ap.add_argument("--repeats", type=int, default=3,
                    help="stability repeats during --smoke (default 3)")
    ap.add_argument("--log", default=None, help="tee stdout+stderr to this file, flushed live")
    args = ap.parse_args()

    if args.log:
        _setup_log(args.log)
    args.ids = [i.strip() for i in args.ids.split(",")] if args.ids else None
    args.metrics = [m.strip() for m in args.metrics.split(",")] if args.metrics else None

    cfg = load_config()
    if args.validate:
        return cmd_validate(cfg, args)
    if args.estimate_cost:
        return cmd_estimate(cfg, args)
    if args.connectivity:
        return cmd_connectivity(cfg, args)
    if args.clear_cells:
        return cmd_clear_cells(cfg, args)
    if args.clear_na:
        return cmd_clear_na(cfg, args)
    if args.compare:
        return cmd_compare(args)
    if args.dashboard:
        out = build_dashboard(args.dashboard, cfg)
        print(f"[dashboard] wrote {out}")
        return 0
    if args.excel:
        d = Path(args.excel)
        per_path = d / "per_sample.jsonl"
        if not per_path.exists():
            raise SystemExit(f"[excel] no per_sample.jsonl in {d}")
        samples, _, routing = load_and_check(cfg, None, skip_parity=True)
        per_sample = agg.load_per_sample(per_path)
        snap_p = d / "config_snapshot.json"
        snapshot = json.loads(snap_p.read_text()) if snap_p.exists() else None
        cost_p = d / "cost.json"
        cost = json.loads(cost_p.read_text()) if cost_p.exists() else None
        out = write_excel_report(d, per_sample, samples, cfg, cost=cost, snapshot=snapshot,
                                 routing=routing)
        print(f"[excel] wrote {out}")
        return 0
    if args.report:
        return cmd_report(cfg, args)
    if args.smoke:
        return cmd_run(cfg, args, smoke=True)
    if args.ragas_only:
        return cmd_run(cfg, args, ragas_only=True)
    if args.judge_only:
        return cmd_run(cfg, args, judge_only=True)
    if args.all or args.resume:
        return cmd_run(cfg, args)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
