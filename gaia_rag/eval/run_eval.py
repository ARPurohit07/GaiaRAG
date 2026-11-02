"""Run the gold set through the pipeline and report metrics.

    python -m gaia_rag.eval.run_eval
    python -m gaia_rag.eval.run_eval --only rrlyrae_reliability
    python -m gaia_rag.eval.run_eval --kind out_of_domain

Results are written to data/processed/eval/ as JSON alongside the
printed report, so two runs can be diffed after a prompt or threshold
change. Every case records the full evidence set and the verdicts, not
just the score, because a faithfulness number without the failing claim
attached is not actionable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from gaia_rag.config import CONFIG  # noqa: E402
from gaia_rag.eval.metrics import CaseResult, Summary, score_case  # noqa: E402
from gaia_rag.generate.answer import GaiaRAG  # noqa: E402
from gaia_rag.generate.prompts import (  # noqa: E402
    RELEVANCE_SYSTEM,
    relevance_user_prompt,
)
from gaia_rag.providers import get_llm  # noqa: E402

GOLDSET = Path(__file__).resolve().parent / "goldset" / "gaia_gold.jsonl"


def load_goldset(path: Path = GOLDSET) -> list[dict]:
    cases = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def score_relevancy(judge, question: str, answer: str) -> float:
    if not answer:
        return 0.0
    try:
        raw = judge.complete_json(
            RELEVANCE_SYSTEM, relevance_user_prompt(question, answer), temperature=0.0
        )
        return max(0.0, min(1.0, float(raw.get("score", 0.0))))
    except (RuntimeError, ValueError, TypeError):
        return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the Gaia RAG.")
    parser.add_argument("--only", help="run a single case by id")
    parser.add_argument("--kind", help="run only cases of this kind")
    parser.add_argument("--goldset", type=Path, default=GOLDSET,
                        help="JSONL question set (default: 50-case safety set)")
    parser.add_argument("--resume", type=Path,
                        help="resume a previously interrupted progress JSONL")
    parser.add_argument(
        "--no-relevancy",
        action="store_true",
        help="skip the LLM relevancy judge (faster)",
    )
    args = parser.parse_args()

    cases = load_goldset(args.goldset)
    if args.only:
        cases = [c for c in cases if c["id"] == args.only]
    if args.kind:
        cases = [c for c in cases if c.get("kind") == args.kind]
    if not cases:
        print("no matching cases")
        return 1

    CONFIG.ensure_dirs()
    rag = GaiaRAG()
    judge = None if args.no_relevancy else get_llm("judge")

    progress_path = args.resume or (
        CONFIG.paths.eval_out / f"eval-progress-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    )
    records = (
        [json.loads(line) for line in progress_path.read_text(encoding="utf-8").splitlines()]
        if args.resume else []
    )
    if [r["case"]["id"] for r in records] != [c["id"] for c in cases[:len(records)]]:
        raise ValueError("resume file does not match this question set")
    results = [CaseResult(**r["result"]) for r in records]
    print(f"progress file: {progress_path} ({len(records)} completed)")
    started = time.time()
    try:
        for i, case in enumerate(cases, 1):
            if i <= len(records):
                continue
            t0 = time.time()
            print(f"[{i}/{len(cases)}] {case['id']} ... ", end="", flush=True)
            answer = rag.ask(case["question"])
            relevancy = (
                score_relevancy(judge, case["question"], answer.answer)
                if judge
                else 0.0
            )
            result = score_case(case, answer, relevancy)
            results.append(result)
            elapsed = time.time() - t0
            print(f"{'PASS' if result.passed else 'FAIL'}  ({elapsed:.0f}s)")
            if not result.passed:
                for f in result.failures:
                    print(f"        - {f}")

            records.append(
                {
                    "case": case,
                    "result": asdict(result),
                    # `passed` and `answered_wrong` are properties, so
                    # asdict() drops them. Store them explicitly: the
                    # ablation harness reuses these verdicts as its
                    # full-system reference instead of re-running every
                    # case, which halves a sweep.
                    "passed": result.passed,
                    "answered_wrong": result.answered_wrong,
                    "outcome": answer.outcome.value,
                    "answer": answer.answer,
                    "claims": answer.claims,
                    "citations": answer.citations,
                    "sql": answer.bundle.sql if answer.bundle else "",
                    "spec": (
                        asdict(answer.bundle.spec)
                        if answer.bundle and answer.bundle.spec
                        else None
                    ),
                    "retrieved": [
                        {"id": e.evidence_id, "kind": e.kind, "text": e.text}
                        for e in answer.evidence
                    ],
                    "notes": answer.bundle.notes if answer.bundle else [],
                    "numeric_checks": (
                        [asdict(c) for c in answer.numeric.checks]
                        if answer.numeric
                        else []
                    ),
                    "claim_verdicts": (
                        [asdict(v) for v in answer.claim_report.verdicts]
                        if answer.claim_report
                        else []
                    ),
                    "subject_issues": (
                        [asdict(issue) for issue in answer.subject_report.issues]
                        if answer.subject_report
                        else []
                    ),
                    "seconds": round(elapsed, 1),
                }
            )
            with progress_path.open("a", encoding="utf-8") as progress:
                progress.write(json.dumps(records[-1], default=str) + "\n")
    finally:
        rag.close()

    summary = Summary(results)
    report = summary.render()
    print()
    print(report)
    print(f"total time: {time.time() - started:.0f}s")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = CONFIG.paths.eval_out / f"eval-{stamp}.json"
    out.write_text(
        json.dumps(
            {
                "summary": summary.as_dict(),
                "config": {
                    "gen_model": CONFIG.models.gen_model,
                    "judge_model": CONFIG.models.judge_model,
                    "embed_model": CONFIG.models.embed_model,
                    "min_claim_support": CONFIG.verification.min_claim_support,
                    "numeric_rel_tol": CONFIG.verification.numeric_rel_tol,
                    "top_k_docs": CONFIG.retrieval.top_k_docs,
                },
                "cases": records,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    (CONFIG.paths.eval_out / f"eval-{stamp}.txt").write_text(report, encoding="utf-8")
    print(f"written: {out}")

    return 0 if all(r.passed for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
