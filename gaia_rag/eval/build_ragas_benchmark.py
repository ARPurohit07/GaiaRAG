"""Combine the 42 answerable safety cases and eight independently verified
supplemental cases into a 50-question RAGAS benchmark.

The eight refusal cases remain in the original 50-question safety run.
RAGAS cannot meaningfully judge faithfulness or context precision for a
correct refusal, so the separate benchmark contains only answerable cases.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from .ragas_eval import EVAL_DIR, REFERENCES


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--safety-run", type=Path, required=True)
    parser.add_argument("--extra-run", type=Path, required=True)
    args = parser.parse_args()

    safety = json.loads(args.safety_run.read_text(encoding="utf-8"))
    extra = json.loads(args.extra_run.read_text(encoding="utf-8"))
    if safety["summary"]["cases"] != 50 or extra["summary"]["cases"] != 8:
        raise ValueError("expected a complete 50-case safety run and 8-case extra run")

    records = [r for r in safety["cases"] if r["case"]["should_answer"]]
    records.extend(extra["cases"])
    ids = [r["case"]["id"] for r in records]
    if len(ids) != 50 or len(set(ids)) != 50:
        raise ValueError(f"benchmark must have 50 distinct answerable questions; got {len(ids)}")
    unserved = [r["case"]["id"] for r in records if not r["result"]["served"]]
    if unserved:
        raise ValueError(f"answerable benchmark cases were not served: {unserved}")
    missing = [r["case"]["id"] for r in records if not (
        r["case"].get("reference") or r["case"]["id"] in REFERENCES
    )]
    if missing:
        raise ValueError(f"missing factual reference answers: {missing}")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = EVAL_DIR / f"benchmark-50-{stamp}.json"
    out.write_text(json.dumps({
        "summary": {"cases": 50, "served": 50},
        "source_runs": [args.safety_run.name, args.extra_run.name],
        "cases": records,
    }, indent=2), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
