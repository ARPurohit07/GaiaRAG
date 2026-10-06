"""Score this pipeline's answers with RAGAS, as an outside opinion.

    .venv-ragas/Scripts/python.exe -m gaia_rag.eval.ragas_eval --limit 12
    .venv-ragas/Scripts/python.exe -m gaia_rag.eval.ragas_eval --all

Runs in its own virtualenv (`.venv-ragas`). RAGAS pulls langchain and a
large dependency tree, and the project's own pipeline deliberately has
almost no dependencies, so the two are kept apart.

Why bother, given this project already measures faithfulness:

* It is **someone else's implementation and prompts**. That removes the
  "you wrote both the system and the metric" objection, which is the
  fair criticism of any self-reported faithfulness number.

What it does *not* do, and should not be presented as doing:

* It is **not an independent method**. RAGAS faithfulness decomposes an
  answer into statements and LLM-judges entailment against the context --
  structurally the same approach as this project's claim judge. The
  deterministic numeric check has no RAGAS equivalent, and that is the
  check doing the real work on catalogue data.
* It **cannot see this system's main property**. RAGAS has no notion of a
  correct refusal; faithfulness over "I can't answer this from the
  catalogue" is degenerate. Abstention cases are therefore excluded, so
  RAGAS says nothing about the behaviour that most of the gold set
  exists to verify.
* With a small local judge the scores are **indicative, not precise**.
  RAGAS is known to be sensitive to judge quality.

Answers are read from a completed `eval-*.json` rather than regenerated,
so RAGAS scores exactly the outputs this project's own metrics scored --
the comparison is on identical text, not on a fresh sample.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

EVAL_DIR = ROOT / "data" / "processed" / "eval"

# RAGAS calls OpenAI by default if nothing is configured; make a missing
# configuration fail loudly rather than reach for a hosted endpoint.
os.environ.setdefault("OPENAI_API_KEY", "not-used-local-only")


def newest_complete_run(path: Path | None = None) -> Path:
    if path is not None:
        return path
    best: tuple[int, Path] | None = None
    for candidate in sorted(EVAL_DIR.glob("eval-*.json")):
        try:
            n = json.loads(candidate.read_text(encoding="utf-8"))["summary"]["cases"]
        except (OSError, ValueError, KeyError):
            continue
        if best is None or n >= best[0]:
            best = (n, candidate)
    if best is None:
        raise FileNotFoundError(
            f"no eval-*.json in {EVAL_DIR}; run gaia_rag.eval.run_eval first"
        )
    return best[1]


def build_reference(case: dict) -> str:
    """A ground-truth string for the metrics that need one.

    Built from the gold case's own `note`, which records how the expected
    value was obtained (almost always a SQL query against the catalogue),
    plus the required values. These are verified facts, not a model's
    idea of a good answer.
    """
    parts = []
    if case.get("must_contain_numbers"):
        values = ", ".join(str(v) for v in case["must_contain_numbers"])
        parts.append(f"The correct value(s): {values}.")
    for phrase in case.get("must_mention") or []:
        parts.append(f"The answer must mention {phrase}.")
    if case.get("note"):
        parts.append(case["note"])
    if not case.get("should_answer", True):
        parts.append(
            "There is no answer in the corpus; the correct response is to "
            "decline rather than provide a value."
        )
    return " ".join(parts) or "No reference available."


def stratified(cases: list[dict], limit: int | None) -> list[dict]:
    """Spread a subset across case kinds rather than taking a prefix.

    The gold set is ordered curated-then-generated and grouped by subset,
    so the first N cases would be almost all one kind.
    """
    if limit is None or limit >= len(cases):
        return cases
    by_kind: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        by_kind[case.get("kind", "unspecified")].append(case)
    picked: list[dict] = []
    while len(picked) < limit:
        progressed = False
        for kind in sorted(by_kind):
            if by_kind[kind] and len(picked) < limit:
                picked.append(by_kind[kind].pop(0))
                progressed = True
        if not progressed:
            break
    return picked


def main() -> int:
    parser = argparse.ArgumentParser(description="Score answers with RAGAS.")
    parser.add_argument("--run", default=None, help="eval-*.json to score")
    parser.add_argument(
        "--limit",
        type=int,
        default=12,
        help="stratified subset size (RAGAS costs ~12-15 LLM calls per case)",
    )
    parser.add_argument("--all", action="store_true", help="score every case")
    parser.add_argument("--judge", default="qwen3:8b")
    parser.add_argument("--embed", default="nomic-embed-text")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="faithfulness + context recall only (~3 judge calls/sample)",
    )
    parser.add_argument(
        "--context-precision",
        action="store_true",
        help="add context precision (~7 extra judge calls/sample)",
    )
    args = parser.parse_args()

    from langchain_ollama import OllamaEmbeddings
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    from ragas.run_config import RunConfig

    from gaia_rag.eval.ragas_llm import OllamaRagasLLM

    run_path = newest_complete_run(Path(args.run) if args.run else None)
    data = json.loads(run_path.read_text(encoding="utf-8"))
    print(f"scoring answers from: {run_path.name}")

    # Only cases where an answer was actually served. A refusal has no
    # statements to decompose, so RAGAS faithfulness over it is noise.
    usable = []
    skipped_abstained = 0
    for record in data["cases"]:
        contexts = [e["text"] for e in record.get("retrieved", []) if e.get("text")]
        if not record.get("answer") or not contexts:
            skipped_abstained += 1
            continue
        if not record["result"].get("served", False):
            skipped_abstained += 1
            continue
        usable.append((record, contexts))
    print(
        f"  {len(usable)} answered cases usable, "
        f"{skipped_abstained} skipped (abstained/withheld: RAGAS cannot score a "
        "correct refusal)"
    )

    chosen = stratified(
        [r for r, _c in usable], None if args.all else args.limit
    )
    chosen_ids = {c["case"]["id"] for c in chosen}
    selected = [(r, c) for r, c in usable if r["case"]["id"] in chosen_ids]

    samples = [
        SingleTurnSample(
            user_input=record["case"]["question"],
            response=record["answer"],
            retrieved_contexts=contexts,
            reference=build_reference(record["case"]),
        )
        for record, contexts in selected
    ]
    print(f"  evaluating {len(samples)} sample(s)\n")
    for record, _c in selected:
        print(f"    - [{record['case']['kind']}] {record['case']['id']}")

    # A judge that talks to Ollama directly, because the langchain-ollama
    # version RAGAS resolves cannot disable a reasoning model's thinking:
    # 131s per call with it on against 3.4s with it off on this machine,
    # across several calls per sample per metric. It also means RAGAS
    # judges through the same provider and settings as the pipeline it is
    # scoring, so a disagreement cannot be put down to configuration.
    judge = OllamaRagasLLM(model=args.judge)
    embedder = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=args.embed))

    # Context precision costs one call per retrieved context (~7 here) --
    # more than every other metric combined -- and it re-measures a
    # figure this project already reports and understands: precision is
    # low by design, because both retrievers run for every question. So
    # it is opt-in rather than default.
    metrics = [Faithfulness(llm=judge), LLMContextRecall(llm=judge)]
    if not args.fast:
        metrics.append(ResponseRelevancy(llm=judge, embeddings=embedder))
    if args.context_precision:
        metrics.append(LLMContextPrecisionWithReference(llm=judge))

    print(f"\n  judge: {args.judge} (thinking disabled)")
    print(f"  embeddings: {args.embed}")
    print(f"  metrics: {', '.join(type(m).__name__ for m in metrics)}")
    print("  running RAGAS (this is the slow part)...\n")
    # RAGAS defaults to 16 workers. Ollama serves one model serially, so
    # extra concurrency buys nothing and only thrashes the context of an
    # 11GB model; a small pool keeps the event loop healthy and the
    # per-call timeout meaningful.
    run_config = RunConfig(max_workers=4, timeout=600, max_retries=3)
    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=metrics,
        llm=judge,
        embeddings=embedder,
        run_config=run_config,
        show_progress=True,
    )

    print("\n" + "=" * 66)
    print("RAGAS RESULTS")
    print("=" * 66)
    scores = {}
    for key, value in result._repr_dict.items() if hasattr(result, "_repr_dict") else []:
        scores[key] = value
    if not scores:
        try:
            scores = dict(result)
        except TypeError:
            scores = {}
    for key, value in scores.items():
        try:
            print(f"  {key:34s} {float(value):.3f}")
        except (TypeError, ValueError):
            print(f"  {key:34s} {value}")
    print("=" * 66)
    print(
        "Subset of answered cases only; abstention cases are excluded and\n"
        "RAGAS cannot assess them. Scores are indicative: the judge is a\n"
        "local 8B model."
    )

    out = EVAL_DIR / f"ragas-{run_path.stem.replace('eval-', '')}.json"
    payload = {
        "source_run": run_path.name,
        "judge": args.judge,
        "embeddings": args.embed,
        "samples": len(samples),
        "metrics": [type(m).__name__ for m in metrics],
        "judge_calls": judge.calls,
        "skipped_abstained": skipped_abstained,
        "scores": {k: str(v) for k, v in scores.items()},
        "case_ids": sorted(chosen_ids),
    }
    try:
        df = result.to_pandas()
        payload["per_case"] = json.loads(df.to_json(orient="records"))
    except Exception:
        pass
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
