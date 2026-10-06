"""Evaluation metrics.

The familiar RAG metrics (faithfulness, context precision/recall, answer
relevancy) are here, but the two that carry the most weight on this
corpus are the ones that need no model:

* `numeric_exactness` -- do the numbers in the answer trace to evidence,
  arithmetically. On catalogue data this is the difference between a
  useful system and a confident one.
* `abstention_correctness` -- did the system answer exactly when it
  should have. Reported separately from faithfulness because a system
  that abstains on everything scores a perfect 1.0 faithfulness while
  being worthless, and one number hiding that is worse than no number.

`answered_wrong_rate` is the headline safety figure: the share of
questions where an answer was served that should not have been, or was
served with a wrong value. That is the number to drive to zero, and it is
not an average of the others.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..generate.answer import RAGAnswer
from ..verify.numeric import _matches, extract_numbers


@dataclass
class CaseResult:
    case_id: str
    kind: str
    source: str
    question: str
    should_answer: bool
    served: bool
    outcome: str
    answer: str

    faithfulness: float = 0.0
    numeric_exactness: float = 1.0
    subject_consistency: float = 1.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    answer_relevancy: float = 0.0

    abstention_correct: bool = True
    required_numbers_present: bool = True
    forbidden_numbers_absent: bool = True
    required_mentions_present: bool = True
    forbidden_mentions_absent: bool = True

    failures: list[str] = field(default_factory=list)

    @property
    def answered_wrong(self) -> bool:
        """Served an answer that should not have been served, or served a
        wrong one. The single figure that matters for trust."""
        if self.served and not self.should_answer:
            return True
        if not self.served:
            return False
        return not (
            self.required_numbers_present
            and self.forbidden_numbers_absent
            and self.required_mentions_present
            and self.forbidden_mentions_absent
        )

    @property
    def passed(self) -> bool:
        return self.abstention_correct and not self.answered_wrong


def _contains_number(text: str, target: float) -> bool:
    return any(_matches(value, target) for value, _ in extract_numbers(text))


def score_case(case: dict, answer: RAGAnswer, relevancy: float) -> CaseResult:
    should_answer = bool(case.get("should_answer", True))
    served = answer.served
    text = answer.answer or ""
    lowered = text.lower()

    result = CaseResult(
        case_id=case["id"],
        kind=case.get("kind", "unspecified"),
        source=case.get("source", "curated"),
        question=case["question"],
        should_answer=should_answer,
        served=served,
        outcome=answer.outcome.value,
        answer=text,
        answer_relevancy=relevancy,
    )

    if answer.claim_report:
        result.faithfulness = answer.claim_report.score
    if answer.numeric:
        result.numeric_exactness = answer.numeric.score
    if answer.subject_report:
        result.subject_consistency = 1.0 if answer.subject_report.passed else 0.0

    # Context precision: of what we put in the prompt, how much was used.
    retrieved = {e.evidence_id for e in answer.evidence}
    cited = set(answer.citations) & retrieved
    result.context_precision = len(cited) / len(retrieved) if retrieved else 0.0

    # Context recall: did retrieval surface the evidence the case needs?
    expected = case.get("expected_evidence") or []
    if expected:
        found = sum(
            1 for pattern in expected if any(pattern in rid for rid in retrieved)
        )
        result.context_recall = found / len(expected)
    else:
        # Cases that should abstain name no expected evidence, so recall
        # is vacuously satisfied. Scoring it as 0 would drag the mean down
        # for behaving correctly.
        result.context_recall = 1.0

    result.abstention_correct = served == should_answer
    if not result.abstention_correct:
        result.failures.append(
            f"should have {'answered' if should_answer else 'abstained'} "
            f"but outcome was {answer.outcome.value}"
        )

    # Content checks apply only when an answer was actually served.
    if served:
        for target in case.get("must_contain_numbers") or []:
            if not _contains_number(text, float(target)):
                result.required_numbers_present = False
                result.failures.append(f"missing required value {target}")
        for target in case.get("forbidden_numbers") or []:
            if _contains_number(text, float(target)):
                result.forbidden_numbers_absent = False
                result.failures.append(
                    f"states forbidden value {target} (the known wrong answer)"
                )
        for phrase in case.get("must_mention") or []:
            if phrase.lower() not in lowered:
                result.required_mentions_present = False
                result.failures.append(f"missing required mention {phrase!r}")
        # Groups of acceptable phrasings: at least one per group must
        # appear. This tests substance without pinning the answer to one
        # vocabulary -- "StarHorse" and "spectroscopic and photometric
        # data" convey the same thing, and only one of them is a proper
        # noun the model has no reason to prefer.
        for group in case.get("must_mention_any") or []:
            if not any(str(alt).lower() in lowered for alt in group):
                result.required_mentions_present = False
                result.failures.append(
                    f"missing all of {list(group)!r} (need at least one)"
                )
        for phrase in case.get("must_not_mention") or []:
            if phrase.lower() in lowered:
                result.forbidden_mentions_absent = False
                result.failures.append(f"contains forbidden mention {phrase!r}")

    return result


@dataclass
class Summary:
    results: list[CaseResult]

    def _mean(self, attr: str, only_served: bool = False) -> float:
        rows = [r for r in self.results if not only_served or r.served]
        if not rows:
            return 0.0
        return sum(getattr(r, attr) for r in rows) / len(rows)

    def _subset(self, source: str) -> dict:
        rows = [r for r in self.results if r.source == source]
        if not rows:
            return {"cases": 0}
        n = len(rows)
        return {
            "cases": n,
            "pass_rate": sum(r.passed for r in rows) / n,
            "answered_wrong_rate": sum(r.answered_wrong for r in rows) / n,
        }

    def as_dict(self) -> dict:
        served = [r for r in self.results if r.served]
        n = len(self.results) or 1
        return {
            # Reported separately because they measure different things.
            # Generated cases test plumbing -- query planning, numeric
            # grounding, units -- and twenty variations of "farthest in
            # subset" are not twenty independent tests. The curated cases
            # carry the faithfulness signal, so averaging the two would
            # let easy generated cases mask a trap failure.
            "curated": self._subset("curated"),
            "generated": self._subset("generated"),
            "cases": len(self.results),
            "served": len(served),
            "abstained": len(self.results) - len(served),
            "answered_wrong_rate": sum(r.answered_wrong for r in self.results) / n,
            "pass_rate": sum(r.passed for r in self.results) / n,
            "abstention_correctness": sum(r.abstention_correct for r in self.results)
            / n,
            "faithfulness_served": self._mean("faithfulness", only_served=True),
            "numeric_exactness_served": self._mean(
                "numeric_exactness", only_served=True
            ),
            "subject_consistency_served": self._mean(
                "subject_consistency", only_served=True
            ),
            "context_precision": self._mean("context_precision"),
            "context_recall": self._mean("context_recall"),
            "answer_relevancy": self._mean("answer_relevancy"),
        }

    def render(self) -> str:
        d = self.as_dict()
        lines = [
            "=" * 66,
            "GAIA RAG EVALUATION",
            "=" * 66,
            f"cases                     : {d['cases']}",
            f"  served                  : {d['served']}",
            f"  abstained               : {d['abstained']}",
            "",
            f"answered-wrong rate       : {d['answered_wrong_rate']:.2%}   <- drive to 0",
            f"pass rate                 : {d['pass_rate']:.2%}",
            f"abstention correctness    : {d['abstention_correctness']:.2%}",
            "",
            "by population (curated carries the faithfulness signal)",
            f"  curated   : {d['curated']['cases']:>2} cases, "
            f"pass {d['curated'].get('pass_rate', 0):.2%}, "
            f"wrong {d['curated'].get('answered_wrong_rate', 0):.2%}",
            f"  generated : {d['generated']['cases']:>2} cases, "
            f"pass {d['generated'].get('pass_rate', 0):.2%}, "
            f"wrong {d['generated'].get('answered_wrong_rate', 0):.2%}",
            "",
            "quality of served answers",
            f"  faithfulness (claims)   : {d['faithfulness_served']:.3f}",
            f"  numeric exactness       : {d['numeric_exactness_served']:.3f}",
            f"  subject consistency     : {d['subject_consistency_served']:.3f}",
            "",
            "retrieval",
            f"  context precision       : {d['context_precision']:.3f}",
            f"  context recall          : {d['context_recall']:.3f}",
            "",
            f"answer relevancy          : {d['answer_relevancy']:.3f}",
            "=" * 66,
        ]

        failures = [r for r in self.results if not r.passed]
        if failures:
            lines.append(f"FAILURES ({len(failures)})")
            for r in failures:
                lines.append(f"  [{r.kind}] {r.case_id}: {r.outcome}")
                for f in r.failures:
                    lines.append(f"      - {f}")
        else:
            lines.append("all cases passed")
        lines.append("=" * 66)

        by_kind: dict[str, list] = {}
        for r in self.results:
            by_kind.setdefault(r.kind, []).append(r)
        lines.append("by kind")
        for kind, rows in sorted(by_kind.items()):
            passed = sum(r.passed for r in rows)
            lines.append(f"  {kind:26s} {passed}/{len(rows)}")
        lines.append("")
        lines.append("per-case detail")
        for r in self.results:
            mark = "PASS" if r.passed else "FAIL"
            lines.append(
                f"  {mark}  {r.case_id:30s} {r.outcome:24s} "
                f"faith={r.faithfulness:.2f} num={r.numeric_exactness:.2f} "
                f"recall={r.context_recall:.2f}"
            )
        return "\n".join(lines)
