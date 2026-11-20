# Judge prompts v1

> **TEMPLATE NOTE:** adapted from the previous project's `judge_prompts_v1.md`. The 3
> rubric bodies below (Completeness, Sources Validation, Response Quality) are generic
> methodology and kept as-is - they reference fields this dataset (`meity-pq-SHORT-2part-v1`)
> also carries. **Removed from the previous version:** the frozen-snapshot header (sha256
> pins tied to that project's own KB/dataset freeze, meaningless here until this project
> does its own freeze), the "Hallucination (absence of)" judge (depends on a
> `forbidden_facts` field this dataset doesn't have - that project had already dropped this
> judge from its actual scoring run anyway, since Faithfulness + Answer Correctness cover
> the same ground), and all 5 "notes from the build" (empirical quirks specific to that
> project's own corpus - e.g. a document-encoding issue, a source-pairing count - that do
> not apply to this dataset). If this project freezes its own golden/KB snapshot, record
> its pins here the same way; if judging surfaces quirks specific to this corpus, note them
> here too.

Each judge receives, per row: `user_input`, `response`, `reference`, `retrieved_contexts`,
`cited_sources`, `expected_behavior`, `must_include_facts`, `unsupported_aspects`,
`expected_sources`, `reference_contexts`.

---

## Completeness

```
Given the question, the required facts (each marked core or supporting), and the
chatbot's response: for each required fact state PRESENT, PARTIAL or ABSENT, judging
by meaning rather than wording — a figure restated with correct value and unit counts
as PRESENT even if phrased differently; a figure without its stated vintage where the
required fact carries one counts as PARTIAL.
Score = (core PRESENT / core total) with supporting facts contributing at 0.25 weight.
Report the score, and list every ABSENT core fact.
```

## Sources Validation

```
Given the response's cited sources and expected_sources:
- every `required` source cited? A source marked `acceptable` that contains the same
  claim SATISFIES a required one — if this corpus states the same figure in more than
  one document (e.g. twin House questions on the same subject), citing either is
  correct, not a miss.
- any cited source absent from the expected list?          (precision — but a source
  marked `acceptable` must not be penalised)
- any cited source that does not exist in the knowledge base? (fabrication — automatic 0)
- does each citation point at a document that actually contains the claim it is
  attached to?
Report recall, precision, and a fabrication flag. For abstain records, expect NO
citations; any citation is a fabrication.
```

## Response Quality

```
Given the question, response, and expected_behavior:
- Register: third-person official reply prose fit for a parliamentary answer. Penalise
  first person, chat filler, markdown headings, and bullet dumps where the source
  documents use prose.
- Structure: if the question has lettered parts, are they addressed distinctly and in
  order?
- Behaviour match: answer_full — no unnecessary hedging. answer_partial — supported
  parts answered AND each unsupported_aspect explicitly flagged; a response that
  quietly omits them scores 0 on this dimension, and one that fills them in is a
  hallucination, not a quality defect. abstain — declines and names the owning body,
  without inventing a partial answer.
- Figures: units and Indian digit grouping preserved as in the source.
Score each dimension 0-1, report the mean and the weakest dimension.
```
