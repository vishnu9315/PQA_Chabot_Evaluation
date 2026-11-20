# Short golden dataset (2 parts) — how to use

**File:** `golden_dataset_short_2part.json` (`meity-pq-SHORT-2part-v1`, built 20 Nov 2026) — **66 rows**
**Review copy:** `golden_dataset_short_2part_review.xlsx` (same rows, for reading only; the pipeline reads the JSON)

A small set for a fast evaluation run. It is in the same format as `golden_dataset_final by vishnu.json`
(`meity-pq-2019-2026-GOLDEN-v1`), so it should load in the existing pipeline without changes.

## 1. What is the same as the approved golden file

- Top-level keys: `name`, `built`, `note`, `n_rows`, `rows`.
- Every row has the same 20 keys in the same order, with the same value types.
- `expected_sources` entries have the same fields (`document`, `house`, `qno`, `parl_type`, `session`,
  `answer_date`, `pages`, `necessity`). Every source is `required`.
- Runtime fields (`response`, `retrieved_contexts`, `cited_sources`, `cited_sources_raw`) are empty.
- `default_mix` and `ragas_eligible` are `true` on all 66 rows: run all of them.

## 2. What is different

| Thing | Detail |
|---|---|
| Two parts | Rows 1–44 = **Part 1, ingested**. Rows 45–66 = **Part 2, not ingested**. |
| Part marker | `provenance.part` = `1-ingested` or `2-not-ingested`; `provenance.kb_status` = `ingested` or `not_ingested`. No new row key was added. |
| New question kinds | `provenance.query_style` can also be `Multi-source` or `Follow-up` (besides `Verbatim`, `Natural`, `Partial`). `question_origin` keeps the existing values only. |
| Extra provenance keys | `part`, `kb_status`, `monsoon_question_id` (Monsoon rows), `source_question_ids` (multi-source rows), `turn` and `follow_up_of` (follow-up rows). |
| Reference text | `reference` and `must_include_facts` contain no commentary about the reply (for example "the reply does not state …" or "the reply neither confirms nor denies …"). What the official reply leaves out is listed only in `unsupported_aspects`. Rows taken from the approved file are otherwise unchanged. |
| New sessions | `session` values `ls18-viii` (Lok Sabha 18 Session VIII) and `rs-271` (Rajya Sabha 271), both Monsoon 2026. |

### What is in each part

| | Verbatim | Natural | Partial | Multi-source | Follow-up | Total |
|---|---|---|---|---|---|---|
| Part 1 — ingested | 7 | 10 | 9 | 8 | 10 (5 pairs) | **44** |
| Part 2 — not ingested | 5 | 7 | 6 | 0 | 4 (2 pairs) | **22** |

Part 1 single-document rows: 16 from the approved golden file (LS17 I–VIII, LS18 II–VII, RS 249–270) and 10 from
Lok Sabha 18 Session VIII. Part 2: Rajya Sabha Session 271 only.

## 3. How to run

1. Point the pipeline at `golden_dataset_short_2part.json` and run all 66 rows **in file order**.
2. Handle the follow-up pairs as described in 3.2.
3. **Report Part 1 and Part 2 separately** (group by `provenance.part`). Do not average them into one number:
   Part 2 is expected to score low on several metrics by design (section 4).

Split in Python:

```python
import json
rows = json.load(open("golden_dataset_short_2part.json"))["rows"]
part1 = [r for r in rows if r["provenance"]["part"] == "1-ingested"]
part2 = [r for r in rows if r["provenance"]["part"] == "2-not-ingested"]
```

### 3.1 Multi-source rows (rows 27–34, ids `GD-MTY-MULTI-01-N` … `-08-N`)

- Each needs **two documents** (`MULTI-04` needs three). All are listed in `expected_sources` and all are `required`.
- A full answer has the figures from every document; an answer built from only one document is incomplete.
- Source recall for these rows = how many of the listed documents were cited.
- `provenance.source_question_ids` gives the original questions each row was built from.

### 3.2 Follow-up pairs (ids ending `-F1` and `-F2`)

| Rows | Pair |
|---|---|
| 35–36 | `GD-MTY-RS-249-U642-F1` / `-F2` |
| 37–38 | `GD-MTY-LS17-2-S132-F1` / `-F2` |
| 39–40 | `GD-MTY-RS-250-U2750-F1` / `-F2` |
| 41–42 | `GD-MTY-LS18-8-U2878-F1` / `-F2` |
| 43–44 | `GD-MTY-LS18-8-U1776-F1` / `-F2` |
| 63–64 | `GD-MTY-RS-271-U713-F1` / `-F2` (Part 2) |
| 65–66 | `GD-MTY-RS-271-U715-F1` / `-F2` (Part 2) |

- `-F1` asks about one part of a question. `-F2` asks the next part and depends on `-F1`
  (for example: "What security measures are in place to protect them from such hacking?").
- **Ask `-F2` in the same chat session, right after its `-F1`.** `provenance.follow_up_of` on the `-F2` row
  holds the id of its `-F1`.
- If the pipeline opens a new chat for every row, the seven `-F2` rows are asked without context and their
  scores mean nothing. In that case either keep the session open for each pair, or leave the seven `-F2` rows
  out of the averages and say so in the report. The `-F1` rows are normal stand-alone questions.

### 3.3 File names in `expected_sources`

- Rows from the approved file keep their document paths exactly.
- Lok Sabha 18 Session VIII and Rajya Sabha 271 rows use the file names as QA received them
  (`…/Unstarred Question_2878_050826.docx`, `…/Starred Question_212_070826_RS.pdf`).
  **If the ingested Session VIII files have different names, add them to the citation-name mapping**, otherwise
  correct citations will be counted as misses on those rows (10 single rows, 4 follow-up rows and
  `MULTI-01`, `-02`, `-04`, `-07`).

## 4. Part 2 (not ingested) — what to look for

Part 2 asks 22 questions whose answer documents (Rajya Sabha 271) are **not** in the knowledge base. The
`reference` is the official answer, so it shows what a correct answer would be, not what the assistant can know.
The point of Part 2 is to see **how the assistant behaves when the document is missing**.

### 4.1 First check: is Rajya Sabha 271 really not ingested?

The developer's message said "Till Lok Sabha-18 Session 8 has been ingested" and did not mention Rajya Sabha 271.
Part 2 assumes it is absent. **If the assistant cites `…_RS.pdf` files of Session 271 and they exist in the KB,
Rajya Sabha 271 has been ingested**; then Part 2 is just more ingested questions and must be reported that way.

### 4.2 How to read each metric on Part 2

| Metric | What to expect | What it tells you |
|---|---|---|
| Source recall / precision | **0 by design.** `expected_sources` names the official PDF only so reviewers can open it. | Nothing about quality. Do not report it as a failure. |
| Fabricated citation | Must stay at none. | A cited document that is not in the KB is a real failure. |
| Completeness (core facts) | Low to medium. | How much of the official answer could be rebuilt from older sessions. Not a pass/fail measure here. |
| Answer correctness / similarity | Low to medium, for the same reason. | Same as above. |
| Context recall / precision | Low: the retrieved text cannot contain the new figures. | Confirms the documents are missing. |
| **Faithfulness / hallucination** | **Should stay as high as on Part 1.** | **The main measure for Part 2.** |

### 4.3 What a good answer and a bad answer look like

**Acceptable:**

- The assistant says it does not have this reply or the latest figures, and offers what earlier replies say.
- It answers from an older reply and makes the date clear ("as of March 2026 …").
- It gives the right current figure from a **Lok Sabha Session VIII** document on the same subject. Several
  subjects were answered in both Houses in the same weeks (for example IndiaAI compute: 15 service providers,
  93.18 lakh GPU hours). That is a correct answer with a real source, not a hallucination.

**Not acceptable — flag each of these:**

1. **Invented figures, dates or names**: anything not in the retrieved text. One invented figure is a failure
   for that row.
2. **Old figures presented as current**: an earlier session's number given as today's position with no date.
3. **A citation that does not support the claim**, or a citation to a document that is not in the KB.
4. **Filling gaps**: for `answer_partial` rows, `unsupported_aspects` lists what even the official reply does
   not give. If the assistant supplies those details, it made them up.
5. **Claiming to quote the Rajya Sabha 271 reply** (question number, date, Minister's words) when it cannot have it.

### 4.4 What to put in the report for Part 2

- Faithfulness / hallucination score for Part 1 and Part 2 side by side.
- Number of Part 2 rows (out of 22) with at least one invented or undated-old figure, with the row ids.
- Number of Part 2 rows where the assistant said it lacked the information.
- Completeness for Part 2, labelled "from earlier sessions only", never compared against a pass mark.
- Whether any Rajya Sabha 271 document was cited (the check in 4.1).

## 5. Known limits

- The 8 multi-source rows and the 14 follow-up rows were written on 20 Nov 2026 and have not had a second
  review. Their figures were checked against the source text by script.
- No evaluation has been run on this file yet.
- Follow-up `-F2` scores are only valid when the pair runs in one chat (3.2).

Questions on the dataset: Akhilesh Chauhan (NeGD QA). To rebuild or change the selection:
`qa/golden/assistant/short-eval/build_short_golden.py`.
