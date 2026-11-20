# Known issues

> **TEMPLATE NOTE:** carried over from the previous project (`deepeval/Ragas_MoC_ParlQA`)
> because the **Inherited** entries below describe real bugs in the upstream `ragas`
> library itself — not Commerce project data — and the patch files that fix them
> (`ragas_eval/_ragas_*_patch.py`, `pacing.py`, `watchdog.py`, `costs.py`) are part of this
> copy too, so their rationale is still accurate for you. The **ISSUE-M1** and **ISSUE-P11**
> entries below are different: they're Commerce-project-specific calibration findings
> (tuned against that project's own corpus and judge-gate results) and reference two RCA
> docs that were deliberately NOT copied into this template, since they're that project's
> own forensics, not yours. Don't inherit `AC_FACTUALITY_WEIGHT`/`AC_RECALL_WEIGHT`/
> `CTX_PRECISION_JUDGE` overrides from `.env` on faith — if you want non-default values for
> those, run your own calibration against your own corpus first, the same way this project
> did.

Kept from day one (build discipline inherited from the reference projects this one was
itself built from). Two kinds of entry: **ISSUE-M\*** found while building/running the
previous project specifically, and **Inherited** — upstream ragas 0.4.3 bugs already
diagnosed elsewhere, whose patches are ported here so nobody rediscovers them.

---

## Inherited (ported, cited at each fix site)

| Issue | What breaks | Ported as |
|---|---|---|
| packaging | ragas 0.4.3 hard-imports `ChatVertexAI` from a deleted langchain-community submodule — stock ragas will not import | `ragas_eval/_compat.py` |
| **ISSUE-1/6 (B)** | judge responses are a LIST of content blocks under thinking; `str()` mangles the embedded JSON; first-`{`-to-last-`}` slicing dies on a brace in prose | `judge_runner._content_to_text` + `_extract_json`; providers.py also collapses content structurally so the repr can never reach a parser |
| **ISSUE-2 (B)** | transient scoring failures died as NA without retry | per-cell retry pass in `ragas_runner.CellRunner.run` (terminal `skipped:` NAs never retried) |
| **ISSUE-8 (B)** | Claude returns `fix_output_format`'s repair unwrapped; the repair round fails `StringIO` and discards a good classification | `_ragas_parser_patch.py` (degenerate-guarded, logged rewrap) |
| **ISSUE-9 (B)** | `answer_correctness` omits empty TP/FP/FN categories; all-required schema rejects it. Defaulting `FN=[]` was tried upstream and REVERTED — it fabricates recall=1.0 | `_ragas_prompt_patch.py` (prompt hardening, no fabrication) |
| **ISSUE-9b (B)** | when hardening isn't enough, a two-call index-based fallback computes a REAL FN; degenerate results stay honest NA | `_ragas_answer_correctness_patch.py` |
| **ISSUE-9c (B) / P10 (A)** | on Bedrock `max_tokens` caps TOTAL output INCLUDING reasoning; long NLI lists truncate mid-JSON, and at effort=max the judge can think the entire budget away and emit no text | `RAGAS_MAX_TOKENS=64000` + `providers.TruncatedOutputError` (a truncated output is refused, never scored) + default effort **medium** |
| **ISSUE-4 (B) → pacing (A)** | botocore adaptive retry as the only pacing turns runs into hours of back-off | `pacing.py` (token bucket + AIMD + fleet cooldown + full jitter) |
| **ISSUE-P4 (A)** | `ragas.evaluate()` swallows per-cell exceptions into bare NaN and returns nothing until the last cell | per-cell `CellRunner` + failure classes + fsync-per-cell `cache.py` |
| **ISSUE-P7/P12 (A)** | a flat per-cell timeout is the wrong shape for Context Precision (cost ∝ chunk count; constant only valid at measured concurrency) | `CellRunner.timeout_for` (`max(RAGAS_TIMEOUT, PER_CHUNK_TIMEOUT × n_chunks)`); MoC rows carry ≤5 chunks so it rarely binds, ported anyway |
| **ISSUE-P8/P13 (A)** | a worker wedged in a native call disables every asyncio deadline; a suspend blinds the monotonic-clock watchdog | `watchdog.py` (os._exit on stall; suspend caveat documented, not fixed upstream either) |
| **ISSUE-P10 (A)** | cache key excludes thinking effort/prompt mode — changing either silently mixes regimes | effort + prompt modes stamped on every cell and in `config_snapshot.json`; `--clear-na`/manual clearing documented |
| **ISSUE-P11 (A)** | context-judge calibration: precision hardening passed A's gate ON A'S CORPUS; recall hardening FAILED (denominator inflation) | recall stays `stock`. **Precision: the MoC gate has now been run (2026-08-20) and `stock` is confirmed BROKEN on this corpus** — 12 of 44 rows scored exactly 0.00 because the stock prompt requires one chunk to support the WHOLE compound reference, while Context Recall on the same chunks read 1.00 (note: 'no row reaches 1.00' is NOT evidence — the AP denominator carries a +1e-10 epsilon, so an all-ones row scores 0.99999999998 and exact 1.0 is unreachable by construction). Hardened cures it and still rejects question-preamble chunks. See `RCA_CONTEXT_PRECISION.md`. **ADOPTED 2026-08-20**: `CTX_PRECISION_JUDGE=hardened` is now persisted in `Ragas_MoC_ParlQA/.env`, so it applies without a shell prefix (`config.py` loads `.env` at import, before `import ragas_eval` applies the patch). An explicit shell prefix still overrides it. Recall remains `stock` |
| **ISSUE-M1 (MoC)** | Answer Correctness measures scope match as much as correctness: `beta=1` weights claim precision equally with recall, so a response 3-6x longer than the reference is punished as hard as a wrong one (15 rows have Completeness>=0.90 with AC<0.70; 15/15 sampled false positives are verbatim in the corpus, and 2 are the bot's own source citations). Separately, normative prose in our own references ("must not be conflated", "a complete answer reports both") is converted into required claims: 7 such rows average F1 0.449 vs 0.625 | quantified 2026-08-20 in `RCA_ANSWER_CORRECTNESS.md`; no metric change made. Read AC beside Completeness. **Normative prose removed from 6 references 2026-08-20** (gate V10 now warns on it). **Weighting overridden 2026-08-20 by owner instruction**: `AC_FACTUALITY_WEIGHT=0.5` (was 0.75) and `AC_RECALL_WEIGHT=0.6` -> beta=1.224745 (was 1.0, equal), both in `.env`, stamped per cell and in config_snapshot.json. Cache key excludes them -> `--clear-cells --metrics answer_correctness` after any change. Re-weighting reduces the penalty but does not fix the cause |
| **ISSUE-P14 (A)** | one scalar out-tokens-per-call constant was 18× wrong for the highest-volume metric and inverted the direction of cost alarm | `costs.py` uses per-metric output constants and a per-phase (RAGAS vs judge) split |

## Decisions of record (this framework)

- **Hallucination judge dropped** (2026-08-13, S10 spec): Faithfulness covers
  grounding-to-retrieval; Answer Correctness catches wrong figures vs reference.
  `forbidden_facts` is unconsumed — noted here and in README.
- **Sources Validation KB floor**: fabrication = cited document absent from the
  441-document union Extracted_KB (mechanical check in `attribution.py`, authoritative
  over the judge). The old never-check-Extracted_KB warning is withdrawn (S10 amendment).
  The sub-score convention is `score = 0 if fabrication else (recall+precision)/2`;
  recall/precision/fabrication are kept on every cell.
- **Composite**: equal weights over the nine metrics; a sample's composite is the mean of
  the components that scored (partial flagged, strict `composite_full` alongside) — A's
  fix for B's 7-of-20 all-or-nothing coverage collapse.
- **Parity gate**: never compares the on-disk KB against the historical 221-doc
  fingerprint (`50cf039…`).

## ISSUE-M1 (OPEN — data property, not a bug) — 5 rows have empty retrieved_contexts

`GD-0083/0111/0133/0134/0135` were captured with zero extracts (the chatbot's verbatim-gate
behaviour; see `pilot25_capture_check.json` in the artifacts). Their Faithfulness /
Context Precision / Context Recall cells are NA "empty retrieved_contexts" by design and
land in the NA manifest. If retrieval is re-captured for these rows the carve-out
disappears automatically (`--validate` would print 39/0/2) — but the unified file's sha
changes, so the parity manifest must be re-amended first or the gate will (correctly)
refuse to run.

## ISSUE-M2 (OPEN — upstream dependency risk) — the union KB fingerprint recipe is locale-sensitive

The recorded `089fcbd1…` was minted by `cd Extracted_KB && find . | sort | …` under the
operator's locale; `LC_ALL=C sort` yields a different value. The gate therefore recomputes
via the exact same shell pipeline (subprocess), not a Python re-implementation. If this
repo ever moves to a machine with a different default collation, re-mint the fingerprint
(and record the locale) rather than "fixing" the gate.

## ISSUE-M3 (OPEN — calibration observation from the acceptance smoke, 2026-08-13)

Two findings from `runs/smoke/run_20260813_153444` (GD-0011, GD-V1-0002, GD-0165 × 3 repeats):

1. **Abstain citations: rubric vs amendment tension.** The frozen Sources-Validation rubric
   says "for abstain records … any citation is a fabrication" (→ automatic 0), but the S10
   amendment makes the deterministic KB-existence list authoritative for the fabrication
   flag — and GD-0165's two extraneous citations DO exist in the union KB, so the judge
   scored 0.5 (recall 1.0 trivially, precision 0) instead of 0. Reasonable behaviour, but
   it is a deviation from a literal reading of the frozen rubric. Decide before the full
   run whether abstain-with-citations should hard-zero; if so, add one sentence to the
   amendment text in `judge_runner._SOURCES_AMENDMENT` (a code change, not a prompt-file
   edit, so the parity gate is unaffected).

2. **Stability spread on context_precision.** 3 of 27 repeated cells exceeded the 0.05
   tolerance; the largest was GD-V1-0002 context_precision (0.806 / 0.417 / 0.806, spread
   0.39) — one repeat flipped ~2 of 5 chunk verdicts. Thinking mode forces temperature=1,
   so this is expected judge nondeterminism at n_chunks=5 granularity (A's ISSUE-P11
   forensics), not a harness bug; answer_correctness was near-deterministic (spread ≤0.006).
   For decisions that hinge on context_precision, average repeats or treat single-run
   values as ±0.2.
