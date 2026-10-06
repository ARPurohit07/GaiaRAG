"""Claim-level entailment judging.

Each claim is judged against only the evidence it cites, not the whole
retrieved set. Judging against everything lets a claim pass because
*something* in the context supports it, which is how citation drift goes
unnoticed: the answer attributes an RR Lyrae caveat to the Cepheid
document and nobody notices because both were retrieved.

The judge is an LLM and therefore the softest component here. It runs
after `verify.numeric`, which catches wrong numbers arithmetically, so
the judge is left with what it is actually decent at: deciding whether a
qualitative statement is entailed by a passage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..generate.prompts import JUDGE_SYSTEM, judge_user_prompt
from ..retrieve.evidence import Evidence

VERDICTS = ("supported", "partial", "unsupported", "contradicted")

#: Credit per verdict when computing the aggregate faithfulness score.
VERDICT_WEIGHT = {
    "supported": 1.0,
    "partial": 0.5,
    "unsupported": 0.0,
    "contradicted": 0.0,
}


@dataclass
class ClaimVerdict:
    claim: str
    cited_ids: list[str]
    verdict: str
    reason: str
    offending_span: str | None = None
    missing_citations: list[str] = field(default_factory=list)

    @property
    def weight(self) -> float:
        return VERDICT_WEIGHT.get(self.verdict, 0.0)


@dataclass
class ClaimReport:
    verdicts: list[ClaimVerdict] = field(default_factory=list)

    @property
    def score(self) -> float:
        """Mean grounding credit across claims.

        An answer with no claims scores 0.0, not 1.0: a generator that
        emits prose with an empty claim list has evaded the check rather
        than passed it.
        """
        if not self.verdicts:
            return 0.0
        return sum(v.weight for v in self.verdicts) / len(self.verdicts)

    @property
    def contradicted(self) -> list[ClaimVerdict]:
        return [v for v in self.verdicts if v.verdict == "contradicted"]

    @property
    def unsupported(self) -> list[ClaimVerdict]:
        return [v for v in self.verdicts if v.verdict in ("unsupported", "contradicted")]

    def summary(self) -> str:
        if not self.verdicts:
            return "no claims were emitted"
        counts: dict[str, int] = {}
        for v in self.verdicts:
            counts[v.verdict] = counts.get(v.verdict, 0) + 1
        return ", ".join(f"{n} {k}" for k, n in sorted(counts.items()))


class ClaimJudge:
    def __init__(self, llm=None):
        from ..providers import get_llm

        self.llm = llm or get_llm("judge")

    def judge_one(
        self, claim: str, cited: list[Evidence], cited_ids: list[str]
    ) -> ClaimVerdict:
        missing = [cid for cid in cited_ids if cid not in {e.evidence_id for e in cited}]
        if not cited:
            # Citing nothing, or citing only ids that do not exist, is an
            # unsupported claim by construction -- no need to ask a model.
            return ClaimVerdict(
                claim=claim,
                cited_ids=cited_ids,
                verdict="unsupported",
                reason=(
                    "claim cites no valid evidence id"
                    if not cited_ids
                    else f"cited ids not in the retrieved set: {', '.join(cited_ids)}"
                ),
                missing_citations=missing,
            )

        block = "\n".join(e.render() for e in cited)
        try:
            raw = self.llm.complete_json(
                JUDGE_SYSTEM, judge_user_prompt(claim, block), temperature=0.0
            )
        except (RuntimeError, ValueError) as exc:
            return ClaimVerdict(
                claim=claim,
                cited_ids=cited_ids,
                verdict="unsupported",
                reason=f"judge failed to return a verdict ({exc})",
                missing_citations=missing,
            )

        verdict = str(raw.get("verdict", "")).strip().lower()
        if verdict not in VERDICTS:
            verdict = "unsupported"
        return ClaimVerdict(
            claim=claim,
            cited_ids=cited_ids,
            verdict=verdict,
            reason=str(raw.get("reason", ""))[:400],
            offending_span=(raw.get("offending_span") or None),
            missing_citations=missing,
        )

    def judge(self, claims: list[dict], evidence: list[Evidence]) -> ClaimReport:
        index = {e.evidence_id: e for e in evidence}
        report = ClaimReport()
        for claim in claims:
            text = str(claim.get("text", "")).strip()
            if not text:
                continue
            cited_ids = [str(c) for c in (claim.get("evidence_ids") or [])]
            cited = [index[cid] for cid in cited_ids if cid in index]
            report.verdicts.append(self.judge_one(text, cited, cited_ids))
        return report
