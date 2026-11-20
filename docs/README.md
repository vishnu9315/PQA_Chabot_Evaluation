# docs

Background and forensics for the framework. None of this is read at runtime — the framework
runs without it — but it records *why* several non-obvious choices were made, and the code
cites these files by name.

| file | what it is |
|---|---|
| `RAGAS_SCORING.md` | how this project scores the chatbot: the 9 metrics, the scoring strategy's strengths and weaknesses, and the checks that make a run trustworthy. Start here. |
| `KNOWN_ISSUES.md` | the upstream-ragas issues this package patches around, by issue id (see its own TEMPLATE NOTE for which entries are reusable vs. Commerce-project-specific). `ragas_eval/__init__.py`, `_ragas_prompt_patch.py` and `_ragas_ctx_recall_patch.py` cite it at the site of each fix. |

**Not copied into this template** (deliberately — they're the previous project's own
calibration forensics, not generic): `RCA_ANSWER_CORRECTNESS.md` (why that project weighted
Answer Correctness as it did) and `RCA_CONTEXT_PRECISION.md` (that project's
context-precision investigation). `config.py`'s `AC_FACTUALITY_WEIGHT`/`AC_RECALL_WEIGHT`
and `CTX_PRECISION_JUDGE`/`CTX_RECALL_JUDGE` still default to that project's values; if you
want different ones, run your own calibration and write your own equivalent doc here.
