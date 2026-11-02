"""Verify that a complete RAGAS run clears a metric threshold.

Checks per-case completeness before averaging, so missing or NaN scores
cannot make an apparently strong mean from a smaller hidden sample.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

METRICS = (
    "faithfulness",
    "context_recall",
    "answer_relevancy",
    "llm_context_precision_with_reference",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--questions", type=int, default=50)
    parser.add_argument("--min-score", type=float, default=0.7)
    args = parser.parse_args()

    data = json.loads(args.run.read_text(encoding="utf-8"))
    rows = data.get("per_case", [])
    ids = data.get("case_ids", [])
    if data.get("samples") != args.questions or len(rows) != args.questions:
        raise ValueError("RAGAS did not score the requested number of questions")
    if len(ids) != args.questions or len(set(ids)) != args.questions:
        raise ValueError("RAGAS question ids are missing or duplicated")
    if data.get("skipped_abstained") != 0:
        raise ValueError("RAGAS skipped one or more benchmark questions")

    failed = []
    for metric in METRICS:
        values = [row.get(metric) for row in rows]
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            failed.append(f"{metric}: missing or non-finite per-case value")
            continue
        average = sum(values) / len(values)
        print(f"{metric:38s} {average:.3f} ({len(values)}/{args.questions} valid)")
        if average <= args.min_score:
            failed.append(f"{metric}: {average:.3f} <= {args.min_score:.3f}")
    if failed:
        raise ValueError("; ".join(failed))
    print(f"PASS: all {len(METRICS)} RAGAS metrics exceed {args.min_score:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
