"""The abstain policy: what to do when verification fails.

Detecting an ungrounded answer is only half the job. If the system
detects it and serves it anyway, the detection was decoration. The guard
turns a failed verification into a withheld answer plus a specific
explanation of what failed.

The gates are asymmetric on purpose. A single ungrounded number blocks
the answer outright, because a wrong distance is the whole failure mode
this corpus invites and there is no safe way to serve it. Weak
qualitative support is scored and thresholded instead, because prose
support is a matter of degree and a hard gate there would reject almost
every genuinely useful answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..config import CONFIG
from .claims import ClaimReport
from .numeric import NumericReport
from .subject import SubjectReport


class Outcome(str, Enum):
    ANSWERED = "answered"
    ANSWERED_WITH_WARNINGS = "answered_with_warnings"
    ABSTAINED_NO_EVIDENCE = "abstained_no_evidence"
    ABSTAINED_BY_MODEL = "abstained_by_model"
    WITHHELD_UNGROUNDED = "withheld_ungrounded"


@dataclass
class GuardDecision:
    outcome: Outcome
    answer: str
    reasons: list[str] = field(default_factory=list)
    numeric_score: float = 1.0
    claim_score: float = 0.0
    subject_ok: bool = True

    @property
    def served(self) -> bool:
        return self.outcome in (Outcome.ANSWERED, Outcome.ANSWERED_WITH_WARNINGS)

    @property
    def abstained(self) -> bool:
        return not self.served


def _withheld_message(reasons: list[str]) -> str:
    return (
        "I can't answer this from the catalogue without overstating what the "
        "evidence supports.\n\nWhat failed verification:\n"
        + "\n".join(f"  - {r}" for r in reasons)
        + "\n\nThe retrieved evidence is shown above; nothing was substituted "
        "from outside it."
    )


def decide(
    *,
    model_said_sufficient: bool,
    model_answer: str,
    model_missing: str | None,
    has_evidence: bool,
    numeric: NumericReport,
    claims: ClaimReport,
    subject: SubjectReport | None = None,
    retrieval_notes: list[str] | None = None,
) -> GuardDecision:
    retrieval_notes = retrieval_notes or []
    subject = subject or SubjectReport()
    numeric_score = numeric.score
    claim_score = claims.score

    # 1. Nothing was retrieved. There is nothing to be faithful to.
    if not has_evidence:
        return GuardDecision(
            outcome=Outcome.ABSTAINED_NO_EVIDENCE,
            answer=(
                "Nothing in the Gaia catalogue or its method documents matched "
                "this question, so I have no evidence to answer from."
                + ("\n\n" + "\n".join(retrieval_notes) if retrieval_notes else "")
            ),
            reasons=["no evidence retrieved"] + retrieval_notes,
            numeric_score=1.0,
            claim_score=0.0,
        )

    # 2. The model itself declined. Respect it -- a generator that says
    #    "the evidence doesn't cover this" is working correctly, and
    #    overriding it would train the pipeline to ignore its own signal.
    if not model_said_sufficient:
        return GuardDecision(
            outcome=Outcome.ABSTAINED_BY_MODEL,
            answer=(
                "The retrieved evidence doesn't support an answer to this."
                + (f"\n\nMissing: {model_missing}" if model_missing else "")
            ),
            reasons=["generator reported insufficient evidence"]
            + ([model_missing] if model_missing else []),
            numeric_score=numeric_score,
            claim_score=claim_score,
        )

    reasons: list[str] = []

    # 3. Hard gate: an ungrounded number.
    if CONFIG.verification.require_numeric_grounding and not numeric.passed:
        reasons.append(f"numeric grounding: {numeric.summary()}")
        return GuardDecision(
            outcome=Outcome.WITHHELD_UNGROUNDED,
            answer=_withheld_message(reasons),
            reasons=reasons,
            numeric_score=numeric_score,
            claim_score=claim_score,
        )

    # 4. Hard gate: a claim cites evidence about a different subset.
    #    This is the citation-drift case, and it is a hard gate because
    #    the claim-level judge cannot see it: the cited document really
    #    does contain the number, just for another survey.
    if not subject.passed:
        reasons.extend(issue.describe() for issue in subject.issues)
        return GuardDecision(
            outcome=Outcome.WITHHELD_UNGROUNDED,
            answer=_withheld_message(reasons),
            reasons=reasons,
            numeric_score=numeric_score,
            claim_score=claim_score,
            subject_ok=False,
        )

    # 5. Hard gate: the evidence contradicts a claim. Worse than an
    #    unsupported claim, because the corpus actively says otherwise.
    if claims.contradicted:
        reasons.extend(
            f"contradicted claim: {v.claim} ({v.reason})" for v in claims.contradicted
        )
        return GuardDecision(
            outcome=Outcome.WITHHELD_UNGROUNDED,
            answer=_withheld_message(reasons),
            reasons=reasons,
            numeric_score=numeric_score,
            claim_score=claim_score,
        )

    # 6. Soft gate: aggregate qualitative support.
    if claim_score < CONFIG.verification.min_claim_support:
        reasons.append(
            f"claim support {claim_score:.2f} below the "
            f"{CONFIG.verification.min_claim_support:.2f} threshold "
            f"({claims.summary()})"
        )
        return GuardDecision(
            outcome=Outcome.WITHHELD_UNGROUNDED,
            answer=_withheld_message(reasons),
            reasons=reasons,
            numeric_score=numeric_score,
            claim_score=claim_score,
        )

    # 7. Served. Surface anything short of clean so the caller can see it.
    warnings = [
        f"unsupported claim: {v.claim}" for v in claims.unsupported
    ]
    warnings.extend(
        f"citation reaches another subset (no number taken from it): "
        f"{issue.describe()}"
        for issue in subject.warnings
    )
    if numeric.unmatched:
        warnings.append(f"numeric: {numeric.summary()}")
    if warnings:
        return GuardDecision(
            outcome=Outcome.ANSWERED_WITH_WARNINGS,
            answer=model_answer,
            reasons=warnings,
            numeric_score=numeric_score,
            claim_score=claim_score,
        )
    return GuardDecision(
        outcome=Outcome.ANSWERED,
        answer=model_answer,
        reasons=[],
        numeric_score=numeric_score,
        claim_score=claim_score,
    )
