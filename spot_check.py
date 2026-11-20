"""Statement-level verdicts for ONE record — scores must be explainable, not just trusted.

Re-runs, against a single sample:
  * Context Recall's own classification prompt — which reference statements were
    attributed to the retrieved contexts, and why;
  * Answer Correctness's decomposition + TP/FP/FN classification — which claims were
    counted where, and the resulting 0.75·F1 + 0.25·similarity mix;
  * one LLM-judge verdict (default: completeness) with its per-fact PRESENT/PARTIAL/ABSENT
    breakdown.

    .venv/bin/python3 spot_check.py GD-0011 [--judge completeness|response_quality|sources_validation] [--out FILE]

Uses the same judge, thinking config and pacing as a real run. PAID (a handful of calls).
Read-only w.r.t. the dataset. Ported pattern from project A's spot_check.py.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import load_config                          # noqa: E402
import ragas_eval                                       # noqa: E402,F401  (applies patches)
from ragas_eval import dataset as ds                    # noqa: E402
from ragas_eval import judge_runner as J                # noqa: E402
from ragas_eval.attribution import build_hint           # noqa: E402
from ragas_eval.pacing import AdaptiveLimiter           # noqa: E402
from ragas_eval.providers import get_embeddings, get_llm, set_limiter  # noqa: E402


def _wrap(s, width=104, indent=" " * 8):
    words, lines, cur = str(s).split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return ("\n" + indent).join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sample_id")
    ap.add_argument("--judge", default="completeness", choices=J.JUDGE_METRICS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config()
    samples = ds.load_samples()
    s = next((x for x in samples if x.id == args.sample_id), None)
    if s is None:
        raise SystemExit(f"unknown id {args.sample_id}")

    set_limiter(AdaptiveLimiter(cfg, log=lambda m: print(f"    [pace] {m}")))
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    llm = LangchainLLMWrapper(get_llm(cfg), bypass_temperature=True)
    emb = LangchainEmbeddingsWrapper(get_embeddings(cfg))

    out = {"id": s.id, "question": s.user_input, "route": s.route,
           "n_retrieved_contexts": len(s.retrieved_contexts)}
    print("=" * 110)
    print(f"SPOT CHECK — {s.id}  [{s.provenance}, {s.expected_behavior}, route={s.route}]")
    print("=" * 110)
    print(f"Question: {_wrap(s.user_input)}")

    # ---------------- Context Recall ----------------
    if s.retrieved_contexts:
        print("\n" + "-" * 110)
        print("CONTEXT RECALL — per reference statement: attributable to the retrieved set?")
        print("-" * 110)
        _cr = importlib.import_module("ragas.metrics._context_recall")
        from ragas.metrics import LLMContextRecall
        m = LLMContextRecall(llm=llm)
        cls = asyncio.run(m.context_recall_prompt.generate(
            data=_cr.QCA(question=s.user_input, context="\n".join(s.retrieved_contexts),
                         answer=s.reference),
            llm=llm, callbacks=None))
        items = getattr(cls, "classifications", cls)
        rows = []
        for i, c in enumerate(items, 1):
            att = bool(getattr(c, "attributed", 0))
            rows.append({"statement": getattr(c, "statement", ""), "attributed": att,
                         "reason": getattr(c, "reason", "")})
            print(f"\n  [{i}] {'ATTRIBUTED' if att else 'NOT ATTRIBUTED'}")
            print(f"      claim : {_wrap(getattr(c, 'statement', ''))}")
            print(f"      reason: {_wrap(getattr(c, 'reason', ''))}")
        n_att = sum(1 for r in rows if r["attributed"])
        print(f"\n  => context_recall = {n_att}/{len(rows)} = {n_att / max(1, len(rows)):.4f}")
        out["context_recall"] = {"score": n_att / max(1, len(rows)),
                                 "classifications": rows}
    else:
        print("\n(context metrics skipped — this row's retrieved_contexts is empty)")

    # ---------------- Answer Correctness ----------------
    if s.route != "judge_only":
        print("\n" + "-" * 110)
        print("ANSWER CORRECTNESS — claim TP/FP/FN, then 0.75·F1 + 0.25·similarity")
        print("-" * 110)
        from ragas.metrics import AnswerCorrectness, AnswerSimilarity
        _ac = importlib.import_module("ragas.metrics._answer_correctness")
        ac = AnswerCorrectness(llm=llm, embeddings=emb, weights=[0.75, 0.25],
                               answer_similarity=AnswerSimilarity(embeddings=emb))

        async def decompose(text):
            res = await ac.statement_generator_prompt.generate(
                data=_ac.StatementGeneratorInput(question=s.user_input, answer=text),
                llm=llm, callbacks=None)
            return list(res.statements)

        answer_stmts = asyncio.run(decompose(s.response))
        gt_stmts = asyncio.run(decompose(s.reference))
        print(f"\n  answer decomposed into {len(answer_stmts)} claims; "
              f"ground truth into {len(gt_stmts)} claims")
        cls = asyncio.run(ac.correctness_prompt.generate(
            data=_ac.QuestionAnswerGroundTruth(question=s.user_input, answer=answer_stmts,
                                               ground_truth=gt_stmts),
            llm=llm, callbacks=None))
        tp, fp, fn = list(cls.TP), list(cls.FP), list(cls.FN)
        for label, group, gloss in (
                ("TP", tp, "in the answer AND supported by ground truth"),
                ("FP", fp, "in the answer but NOT supported by ground truth"),
                ("FN", fn, "in the ground truth but MISSING from the answer")):
            print(f"\n  {label} — {len(group)} ({gloss})")
            for x in group[:6]:
                print(f"    · {_wrap(getattr(x, 'statement', x), indent=' ' * 6)}")
            if len(group) > 6:
                print(f"    … and {len(group) - 6} more")

        def fbeta(tp_n, fp_n, fn_n, beta=1.0):
            p = tp_n / (tp_n + fp_n + 1e-10)
            r = tp_n / (tp_n + fn_n + 1e-10)
            if p == 0 and r == 0:
                return 0.0
            return ((1 + beta ** 2) * p * r) / ((beta ** 2 * p) + r + 1e-10)

        f1 = fbeta(len(tp), len(fp), len(fn))
        from ragas import SingleTurnSample
        sim = asyncio.run(AnswerSimilarity(embeddings=emb).single_turn_ascore(
            SingleTurnSample(response=s.response, reference=s.reference)))
        score = 0.75 * f1 + 0.25 * float(sim)
        print(f"\n  => factual F1 = fbeta({len(tp)}, {len(fp)}, {len(fn)}) = {f1:.4f}")
        print(f"  => semantic similarity = {float(sim):.4f}")
        print(f"  => answer_correctness = 0.75·{f1:.4f} + 0.25·{float(sim):.4f} = {score:.4f}")
        out["answer_correctness"] = {"TP": len(tp), "FP": len(fp), "FN": len(fn),
                                     "factual_f1": round(f1, 4),
                                     "semantic_similarity": round(float(sim), 4),
                                     "score": round(score, 4)}

    # ---------------- one judge ----------------
    print("\n" + "-" * 110)
    print(f"JUDGE — {J.JUDGE_LABEL[args.judge]} (frozen judge_prompts v1)")
    print("-" * 110)
    kb_names = ds.kb_document_names(cfg)
    rubrics = J.load_rubrics(cfg)
    rec = J._judge_one(get_llm(cfg), s.judge_item(), args.judge, rubrics[args.judge],
                       build_hint(s, kb_names), cfg.judge_retry_rounds)
    print(f"\n  score = {rec.get('score')}   na = {rec.get('na_reason') or '-'}")
    print(f"  justification: {_wrap(rec.get('justification', ''))}")
    print(f"  detail: {json.dumps(rec.get('detail', {}), ensure_ascii=False, indent=2)[:1500]}")
    out["judge"] = {k: rec.get(k) for k in ("metric", "score", "na_reason",
                                            "justification", "detail")}

    if args.out:
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
