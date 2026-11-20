# How we score the MeitY PQA chatbot with RAGAS

This document explains what we measure, how each score is produced, what the scoring
strategy does well and badly, and the checks that tell us a run can be trusted.

## 1. What is being evaluated

- **Chatbot:** MeitY Parliament AI Platform, `https://pqa.myscheme.in/meityy`.
- **Dataset:** `meity-pq-SHORT-2part-v1`, 66 real parliamentary questions
  (`golden_dataset.json`; see `golden_dataset_README.md`).
  - **Part 1, ingested (44 rows):** the answer documents are in the chatbot's knowledge
    base. This measures answer quality.
  - **Part 2, not ingested (22 rows, Rajya Sabha Session 271):** the answer documents
    are *not* in the knowledge base. This measures behaviour when the document is missing.
  - Question styles: Natural 17, Partial 15, Follow-up 14 (7 pairs), Verbatim 12,
    Multi-source 8.
- **Knowledge base:** the 2,250 documents in `documents/` (every file there is ingested).

**Part 1 and Part 2 are always reported separately and never averaged together.** Part 2
is expected to score low on several metrics by design.

## 2. Pipeline

```
golden_dataset.json            your questions + scoring key (never modified)
        │  capture_new_chatbot.py   — asks the live chatbot each question
        ▼
data/artifacts/golden_dataset_new_chatbot_captured.json   + response, retrieved passages, citations
        │  tools/build_unified_eval.py
        ▼
data/artifacts/golden_dataset_final_unified_eval.json     the scoring input
        │  cli.py --all             — 6 RAGAS metrics + 3 LLM judges
        ▼
runs/run_vN/<timestamp>/        report.md, dashboard.html, evaluation_report.xlsx,
                                per_sample.jsonl, config_snapshot.json, cost.json
```

At load time the scorer also does two things in memory (the files are never rewritten):

- **Source names are matched.** Expected sources, written as QA received them (e.g.
  `…/Unstarred Question_2878_050826.docx`), are mapped to the chatbot's citation form
  (`lok_sabha/lok_sabha_18/Session_VIII/AU2878_rN86Hc.md`) through each session's
  `metadata.csv`. See `ragas_eval/kb_index.py`.
- **Follow-ups get their context.** Each `-F2` row is linked to its `-F1` through
  `provenance.follow_up_of`, because it was asked in the same chat.

## 3. The nine metrics

All scores are on a 0–1 scale. The judge LLM is Claude Sonnet 5 on AWS Bedrock
(`global.anthropic.claude-sonnet-5`, extended thinking on, effort medium). Embeddings use
Amazon Titan Text Embeddings v2.

### RAGAS (6)

| Metric | Question it answers | Compares | How |
|---|---|---|---|
| **Answer Correctness** | Is the answer factually right? | response vs reference | Claims from both are matched into an F-score, then blended with semantic similarity. Weights: 50% claims / 50% similarity, with recall given 60% of the F-score. |
| **Answer Relevancy** | Does the answer address the question? | response vs question | The LLM writes questions the answer would fit, and those are embedded and compared with the real question. |
| **Semantic Similarity** | Does it mean the same as the reference? | response vs reference | Embedding cosine similarity only, no LLM call. |
| **Faithfulness** | Is every claim supported by what was retrieved? | response vs retrieved passages | The answer is broken into claims and each is checked against the passages. **Main measure for Part 2.** |
| **Context Precision** | Were the retrieved passages relevant, and ranked first? | passages vs reference | One LLM verdict per passage, weighted by rank. Uses the **hardened** prompt (see §5). |
| **Context Recall** | Did retrieval find everything the reference needs? | reference vs passages | Each reference statement is attributed to a passage, or not. |

### LLM judges (3), prompts in `data/artifacts/judge_prompts_v1.md`

| Judge | Question it answers | Sees |
|---|---|---|
| **Completeness** | Are the required facts present? Each `must_include_facts` item is marked PRESENT, PARTIAL or ABSENT; core facts count fully, supporting facts at 0.25. | question, required facts, response. Never the reference, so it can't just copy it. |
| **Response Quality** | Is it a proper official reply? Mean of register, structure, behaviour (e.g. flags `unsupported_aspects` on partial rows) and figures (units, Indian digit grouping). | question, expected behaviour, unsupported aspects, response |
| **Sources Validation** | Are the right documents cited? Mean of citation recall and precision; a citation to a document outside the knowledge base scores 0. | expected vs cited sources, a computed match summary, retrieved passages |

The completeness and response quality judges also see the earlier question and answer for
follow-up (`-F2`) rows. For RAGAS, the follow-up question is prefixed with its earlier
question.

### The composite score

Each row's composite is the **equal-weight mean of its 9 metrics**. If a metric is NA, the
composite uses the rest and is flagged as partial. Reports never lead with one overall
number: the Part 1 / Part 2 split comes first.

## 4. Rules that keep the numbers honest

- **NA is never 0.** A metric that can't be computed (empty retrieval, timeout, unparseable
  judge output) is recorded as NA with a reason, listed in the NA manifest in `report.md`,
  and excluded from means. Every mean shows its denominator (`n_scored / n`). Segments with
  fewer than 5 scored rows are flagged as low-n (`*`).
- **One cell at a time, cached.** Each (row, metric) pair is scored and saved separately, so
  a crash or a stop loses nothing, and `--resume` doesn't pay twice for finished cells.
- **Truncated LLM output is refused**, never scored.
- **Every run records its settings** in `config_snapshot.json`: models, thinking mode,
  context precision prompt, and Answer Correctness weights.
- **Five known bugs in `ragas` 0.4.3 are patched** (see `docs/KNOWN_ISSUES.md`), along with
  rate limiting, timeouts and a stall watchdog.

## 5. Is the scoring strategy good?

**Overall, yes, for finding where the chatbot fails and why.** It's less suited to producing
one headline number.

### Strengths

- **It covers the whole chain.** Retrieval (context precision and recall), grounding
  (faithfulness), answer quality (correctness, completeness, relevancy) and citations
  (sources validation). A low answer score can be traced to "retrieval missed the document"
  or "retrieved it but answered badly".
- **The judges cover what RAGAS can't:** required facts, official register, flagging of
  unsupported aspects, and correct citations.
- **The inputs are real.** Captured answers match what users get in the UI. Retrieved
  passages are full text, not 500-character previews. Citations are matched to the real
  knowledge base.
- **Calibrated where it mattered.** On 2026-11-20 the stock context precision prompt gave a
  false 0.00 to a two-document row (`GD-MTY-MULTI-01-N`). The hardened prompt scored it
  0.50, matching an independent word-match check and context recall, and agreed with that
  check on all 6 rows tested. Evidence is in `runs_calib/`.

### Weaknesses, and how to read around them

| Weakness | Effect | What to do |
|---|---|---|
| **The equal-weight composite mixes different things.** Retrieval, grounding and answer metrics are averaged together, and semantic similarity is counted twice (on its own, and as half of Answer Correctness). | The composite is a rough ranking aid, not a quality score. | Read the per-metric table by part. Optionally, report four group scores instead: Retrieval (CP, CR), Grounding (Faithfulness), Answer (AC, Completeness, Relevancy, RQ), Citations (SV). |
| **Correct behaviour on Part 2 is penalised.** For example *"I could not find an answer to that question."* scores low on Answer Correctness, Relevancy and Completeness, although the dataset README counts it as acceptable. | Part 2 answer metrics look worse than the behaviour deserves. | Use report section 9: faithfulness for Part 1 vs Part 2, rows flagged for unsupported claims, and refusal count. Don't apply a pass mark to Part 2 answer metrics. |
| **One model family scores everything.** Claude Sonnet 5 is the judge for both RAGAS and the LLM judges, with no second opinion. | Possible systematic bias. | Human spot-check (§6, check 6). |
| **Semantic similarity saturates.** Titan embeddings put most on-topic answers in a narrow high band. | It barely separates good answers from mediocre ones. | Give more weight to Completeness and Answer Correctness when comparing. |
| **The sample is small.** 44 Part 1 rows, and some segments have only 5–15. | A difference under about 0.05 between runs or segments may be noise. | Compare only beyond the stability tolerance (0.05), and check the low-n flags. |
| **Answer Correctness weights came from the previous project**, not from MeitY data. | Reasonable, but not calibrated. | Hand-grade about 10 answers and tune `AC_FACTUALITY_WEIGHT` / `AC_RECALL_WEIGHT` to match. |

## 6. How to know a run is trustworthy

**Before running (free except the last check):**

1. `.venv/Scripts/python.exe -m pytest tests/ -q`: all pass.
2. `cli.py --validate`: passes with 0 warnings; shows 44 Part 1 + 22 Part 2 and
   66 / 0 / 0 routing.
3. `cli.py --estimate-cost`: about $19.60.
4. `cli.py --connectivity`: one paid call; confirms credentials, the model and
   `thinking active: True`.

**After the run, in `report.md`:**

5. **Coverage:** close to 66/66 scored per metric. Read every NA reason in the NA manifest;
   truncations should be 0.
6. **Human spot-check (the real test).** Read the 5 worst rows (section 8) and 5 random rows.
   For each, ask whether you agree with the scores. If the judges disagree with you on 2 or
   more of 10, investigate before quoting any number.
7. **Expected patterns.** If one of these doesn't hold, suspect the pipeline before the
   chatbot:
   - Part 1 beats Part 2 on Answer Correctness, Context Recall and Completeness.
   - Faithfulness on Part 2 is about the same as on Part 1 (README §4).
   - Verbatim questions score highest on Context Recall, because they are copied from the
     reply.
   - Refusals (e.g. `GD-MTY-MULTI-03-N` in Part 1) score low on Completeness and Answer
     Correctness.
   - No Rajya Sabha Session 271 document is cited (report section 9).
8. **Metrics that contradict each other on the same row:**
   - Context Precision 0 but Context Recall high: the precision judge is likely wrong. Check
     with `tools/containment_check.py --ids <id>`.
   - Completeness high but Answer Correctness low: check the reference for that row.
   - Faithfulness low on a row whose answer reads correctly: the answer may draw on facts
     that weren't retrieved. This is a real grounding issue, not a scoring error.
9. **Stability:** `cli.py --smoke --ids <5 ids> --repeats 3` (a few dollars). The spread per
   metric should stay within 0.05. If it doesn't, treat smaller differences as noise.

**Before comparing runs over time:** freeze the dataset, judge prompts and settings, so the
parity gate in `--validate` can confirm two runs scored the same inputs the same way.

## 7. Commands

```bash
.venv/Scripts/python.exe cli.py --validate        # free
.venv/Scripts/python.exe cli.py --estimate-cost   # free
.venv/Scripts/python.exe cli.py --connectivity    # 1 small paid call
.venv/Scripts/python.exe cli.py --all             # full paid run (~$19.60)
.venv/Scripts/python.exe cli.py --resume          # continue a stopped run, cached cells free
.venv/Scripts/python.exe cli.py --report RUN_DIR  # rebuild reports from cache, free
.venv/Scripts/python.exe tools/containment_check.py --ids <id>   # non-LLM precision check
```

Settings live in `.env`: `CTX_PRECISION_JUDGE=hardened` (calibrated 2026-11-20),
`AC_FACTUALITY_WEIGHT` and `AC_RECALL_WEIGHT` (defaults 0.5 / 0.6), `BEDROCK_MODEL_ID`,
`BEDROCK_THINKING_EFFORT`.
