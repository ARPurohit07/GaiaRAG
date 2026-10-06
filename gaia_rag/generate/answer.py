"""The end-to-end pipeline: retrieve, generate, verify, decide.

Verification runs on every answer, not only during evaluation. A
faithfulness number computed offline tells you how the system behaved on
a gold set; running the same checks inline is what stops a specific bad
answer reaching a specific user. The eval harness in `gaia_rag.eval`
reuses exactly these objects, so the measured behaviour and the served
behaviour cannot diverge.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..retrieve.evidence import Evidence
from ..retrieve.router import RetrievalBundle, Router
from ..verify.claims import ClaimJudge, ClaimReport
from ..verify.guard import GuardDecision, Outcome, decide
from ..verify.numeric import NumericReport, verify_numbers
from ..verify.subject import SubjectReport, check_claims
from .prompts import ANSWER_SYSTEM, answer_user_prompt


@dataclass
class RAGAnswer:
    question: str
    outcome: Outcome
    answer: str
    claims: list[dict] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    numeric: NumericReport | None = None
    claim_report: ClaimReport | None = None
    subject_report: SubjectReport | None = None
    decision: GuardDecision | None = None
    bundle: RetrievalBundle | None = None
    generator_error: str | None = None

    @property
    def served(self) -> bool:
        return self.decision.served if self.decision else False

    @property
    def citations(self) -> list[str]:
        seen: list[str] = []
        for claim in self.claims:
            for cid in claim.get("evidence_ids") or []:
                if cid not in seen:
                    seen.append(str(cid))
        return seen

    def report(self, show_evidence: bool = False) -> str:
        lines = [self.answer, ""]
        lines.append(f"outcome        : {self.outcome.value}")
        if self.numeric:
            lines.append(f"numeric grounding: {self.numeric.summary()}")
        if self.claim_report:
            lines.append(
                f"claim support  : {self.claim_report.score:.2f} "
                f"({self.claim_report.summary()})"
            )
        if self.subject_report:
            lines.append(f"subject check  : {self.subject_report.summary()}")
        if self.bundle:
            lines.append(
                f"evidence       : {len(self.bundle.rows)} catalogue row(s), "
                f"{len(self.bundle.docs)} doc chunk(s)"
            )
            if self.bundle.sql:
                lines.append(f"sql            : {self.bundle.sql}")
            for note in self.bundle.notes:
                lines.append(f"note           : {note}")
        if self.citations:
            lines.append(f"cited          : {', '.join(self.citations)}")
        if self.decision and self.decision.reasons:
            lines.append("flags          :")
            lines.extend(f"  - {r}" for r in self.decision.reasons)
        if show_evidence and self.bundle:
            lines.append("\n--- evidence ---")
            lines.append(self.bundle.render())
        return "\n".join(lines)


class GaiaRAG:
    def __init__(self, router: Router | None = None, llm=None, judge=None):
        from ..providers import get_llm

        self.router = router or Router()
        self.llm = llm or get_llm("generator")
        self.judge = judge or ClaimJudge()

    def ask(self, question: str) -> RAGAnswer:
        bundle = self.router.retrieve(question)

        if bundle.is_empty:
            decision = decide(
                model_said_sufficient=False,
                model_answer="",
                model_missing=None,
                has_evidence=False,
                numeric=NumericReport(),
                claims=ClaimReport(),
                retrieval_notes=bundle.notes,
            )
            return RAGAnswer(
                question=question,
                outcome=decision.outcome,
                answer=decision.answer,
                evidence=[],
                decision=decision,
                bundle=bundle,
            )

        try:
            raw = self.llm.complete_json(
                ANSWER_SYSTEM,
                answer_user_prompt(question, bundle.render(), bundle.notes),
                temperature=0.0,
            )
        except (RuntimeError, ValueError) as exc:
            # A generator that returned nothing usable must not be
            # reported as an abstention: abstention is a judgement about
            # the evidence, this is an outage.
            return RAGAnswer(
                question=question,
                outcome=Outcome.WITHHELD_UNGROUNDED,
                answer=f"The generator did not return a usable response: {exc}",
                evidence=bundle.evidence,
                bundle=bundle,
                generator_error=str(exc),
            )

        sufficient = bool(raw.get("sufficient", False))
        answer_text = str(raw.get("answer") or "").strip()
        claims = [c for c in (raw.get("claims") or []) if isinstance(c, dict)]
        missing = raw.get("missing") or None

        # Verify the prose the user would actually see, plus the claims,
        # so a number smuggled into one but not the other is still caught.
        verifiable_text = "\n".join(
            [answer_text] + [str(c.get("text", "")) for c in claims]
        )
        numeric = verify_numbers(verifiable_text, bundle.evidence, question)
        claim_report = (
            self.judge.judge(claims, bundle.evidence) if sufficient else ClaimReport()
        )
        subject_report = check_claims(claims, bundle.evidence)

        decision = decide(
            model_said_sufficient=sufficient and bool(answer_text),
            model_answer=answer_text,
            model_missing=str(missing) if missing else None,
            has_evidence=True,
            numeric=numeric,
            claims=claim_report,
            subject=subject_report,
            retrieval_notes=bundle.notes,
        )

        return RAGAnswer(
            question=question,
            outcome=decision.outcome,
            answer=decision.answer,
            claims=claims,
            evidence=bundle.evidence,
            numeric=numeric,
            claim_report=claim_report,
            subject_report=subject_report,
            decision=decision,
            bundle=bundle,
        )

    def close(self) -> None:
        self.router.close()
