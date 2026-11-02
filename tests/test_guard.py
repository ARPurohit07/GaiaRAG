"""Tests for the verification stack, with the models stubbed out.

The real pipeline takes a minute or more per question, which is too slow
to assert against. These tests drive the verifiers and the guard directly
with synthetic evidence, so the gating logic is covered deterministically
and in milliseconds.

The cases are built around the actual observed failure: an answer that
reported the GCNS 1% parallax precision as the RR Lyrae uncertainty.

Run: python tests/test_guard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gaia_rag.retrieve.evidence import Evidence, NumericFact
from gaia_rag.units import distance_renderings
from gaia_rag.verify.claims import ClaimJudge, ClaimReport
from gaia_rag.verify.guard import Outcome, decide
from gaia_rag.verify.numeric import verify_numbers
from gaia_rag.verify.subject import check_claims


class StubLLM:
    """Returns a fixed verdict, so judging is deterministic."""

    def __init__(self, verdict: str = "supported"):
        self.name = "stub"
        self.verdict = verdict
        self.calls = 0

    def complete_json(self, system, user, *, temperature=0.0):
        self.calls += 1
        return {"verdict": self.verdict, "reason": "stub", "offending_span": None}

    def complete(self, system, user, *, temperature=0.0):
        return ""


class BatchStubLLM(StubLLM):
    def complete_json(self, system, user, *, temperature=0.0):
        self.calls += 1
        assert "Judge each item independently" in system
        assert len(__import__("json").loads(user)) == 2
        return {"verdicts": [
            {"verdict": "supported", "reason": "cited", "offending_span": None},
            {"verdict": "unsupported", "reason": "not cited", "offending_span": "claim"},
        ]}


def rrlyrae_doc() -> Evidence:
    return Evidence(
        evidence_id="doc:caveats_gaia_rrlyrae#0",
        kind="doc",
        text=(
            "Distance reliability for Gaia DR3 RR Lyrae. dist_pc: derived, "
            "typical relative uncertainty 18%. Good to roughly 15-20%."
        ),
        citation="Muraveva et al. 2018",
        metadata={"doc_id": "caveats_gaia_rrlyrae", "kind": "caveat"},
    )


def gcns_doc() -> Evidence:
    return Evidence(
        evidence_id="doc:caveats_gaia_gcns#0",
        kind="doc",
        text=(
            "Distance reliability for the Gaia Catalogue of Nearby Stars. "
            "dist_pc: measured, typical relative uncertainty 1%."
        ),
        citation="Smart et al. 2021",
        metadata={"doc_id": "caveats_gaia_gcns", "kind": "caveat"},
    )


def cepheid_row() -> Evidence:
    dist_pc = 24952.0
    return Evidence(
        evidence_id="row:gaia_cepheid:4054000049998570624",
        kind="row",
        text=f"Gaia Cepheid 4054000049998570624. Distance {dist_pc} pc.",
        citation="Galaxy web catalogue",
        numeric_facts=[
            NumericFact(
                field_name="dist_pc",
                value=dist_pc,
                unit="pc",
                renderings=distance_renderings(dist_pc),
                status="derived",
                rel_uncertainty=0.1,
            )
        ],
        metadata={"subset": "gaia_cepheid", "is_gaia": True},
    )


# --- numeric gate -----------------------------------------------------


def test_correct_unit_conversion_is_grounded():
    """24952 pc may legitimately be stated as 81383 ly or 24.95 kpc."""
    evidence = [cepheid_row()]
    for text in (
        "The distance is 24952 pc.",
        "The distance is 24.95 kpc.",
        "The distance is 81,383 ly.",
    ):
        report = verify_numbers(text, evidence)
        assert report.passed, (text, report.summary())


def test_wrong_number_is_caught():
    report = verify_numbers("The distance is 24952 ly.", [cepheid_row()])
    assert not report.passed
    assert report.unmatched[0].value == 24952.0
    # It is wrong as light-years but the pc value exists, so the check
    # must rely on the unit, not just the magnitude.
    assert "ungrounded" in report.summary()


def test_written_out_unit_is_still_checked():
    report = verify_numbers("The distance is 24952 light-years.", [cepheid_row()])
    assert not report.passed, report.summary()


def test_invented_number_is_caught():
    report = verify_numbers("The distance is 4,321 pc.", [cepheid_row()])
    assert not report.passed


def test_scientific_notation_is_checked_as_one_value():
    evidence = [Evidence(
        evidence_id="row:cepheid:max_lum", kind="row",
        text="Luminosity 32,910 times solar.", citation="catalogue",
        numeric_facts=[NumericFact("lum_lsun", 32910.0, "Lsun")],
    )]
    good = verify_numbers("Luminosity is 3.29 × 10^4 times solar.", evidence)
    bad = verify_numbers("Luminosity is 3.29 × 10^5 times solar.", evidence)
    assert good.passed, good.summary()
    assert not bad.passed
    assert len(bad.unmatched) == 1


def test_uncertainty_range_is_not_a_negative_number():
    report = verify_numbers("The uncertainty is 20%.", [rrlyrae_doc()])
    assert report.passed, report.summary()


def test_claims_are_judged_in_one_batch_with_separate_verdicts():
    llm = BatchStubLLM()
    report = ClaimJudge(llm).judge([
        {"text": "RR Lyrae uncertainty is 18%.", "evidence_ids": [rrlyrae_doc().evidence_id]},
        {"text": "RR Lyrae uncertainty is 1%.", "evidence_ids": [rrlyrae_doc().evidence_id]},
    ], [rrlyrae_doc()])
    assert llm.calls == 1
    assert [v.verdict for v in report.verdicts] == ["supported", "unsupported"]


def test_percentage_from_wrong_document_is_caught_numerically():
    """1% is absent from the RR Lyrae evidence alone."""
    report = verify_numbers(
        "RR Lyrae distances are reliable to 1%.", [rrlyrae_doc()]
    )
    assert not report.passed, report.summary()


def test_question_numbers_are_exempt():
    report = verify_numbers(
        "There are Cepheids within 3 kpc.",
        [rrlyrae_doc()],
        question="How many Cepheids lie within 3 kpc?",
    )
    assert report.passed


# --- subject gate -----------------------------------------------------


def test_citation_drift_is_caught():
    """The observed failure: an RR Lyrae claim citing the GCNS document."""
    evidence = [rrlyrae_doc(), gcns_doc()]
    claims = [
        {
            "text": "RR Lyrae distances are reliable to 1%.",
            "evidence_ids": ["doc:caveats_gaia_gcns#0"],
        }
    ]
    report = check_claims(claims, evidence)
    assert not report.passed
    issue = report.issues[0]
    assert issue.claim_subsets == {"gaia_rrlyrae"}
    assert "gaia_gcns" in issue.cited_subsets


def test_matching_subject_passes():
    claims = [
        {
            "text": "RR Lyrae distances are good to roughly 15-20%.",
            "evidence_ids": ["doc:caveats_gaia_rrlyrae#0"],
        }
    ]
    assert check_claims(claims, [rrlyrae_doc(), gcns_doc()]).passed


def hyg_row() -> Evidence:
    dist_pc = 7.679
    return Evidence(
        evidence_id="row:hyg:Gl 721",
        kind="row",
        text=(
            "Vega (catalogue subset: hyg). Distance 7.68 pc (25.0 ly), "
            "catalogued, typical relative uncertainty 5%. NOTE: this row is "
            "NOT from Gaia; do not attribute it to Gaia."
        ),
        citation="HYG Database v4.1",
        numeric_facts=[
            NumericFact(
                field_name="dist_pc",
                value=dist_pc,
                unit="pc",
                renderings=distance_renderings(dist_pc),
                status="catalogued",
                rel_uncertainty=0.05,
            )
        ],
        metadata={"subset": "hyg", "is_gaia": False},
    )


def test_superfluous_cross_subset_citation_is_a_warning_not_a_failure():
    """A correct answer must not be withheld for an extra citation.

    "Vega's distance is from the HYG Database, not a Gaia measurement"
    cited the right HYG row AND a GCNS document. Every number in the
    claim is available from the HYG row, so the GCNS citation is merely
    superfluous -- treating it as drift withheld a correct and useful
    answer.
    """
    claims = [
        {
            "text": (
                "The distance to Vega is 7.68 parsecs and comes from the HYG "
                "Database, not from Gaia."
            ),
            "evidence_ids": ["row:hyg:Gl 721", "doc:caveats_gaia_gcns#0"],
        }
    ]
    report = check_claims(claims, [hyg_row(), gcns_doc()])
    assert report.passed, [i.describe() for i in report.issues]
    assert report.warnings, "the extra citation should still be surfaced"


def test_number_taken_from_the_wrong_subset_is_a_hard_failure():
    """The distinguishing case: the number is only in the wrong doc.

    Here the claim cites the correct RR Lyrae document too, so an
    any-citation-matches rule would pass it -- but 10% appears only in
    the Cepheid document, which is where it was actually taken from.
    """
    cepheid = Evidence(
        evidence_id="doc:caveats_gaia_cepheid#0",
        kind="doc",
        text=(
            "Distance reliability for Gaia DR3 classical Cepheids. dist_pc: "
            "derived, typical relative uncertainty 10%."
        ),
        citation="Ripepi et al. 2022",
        metadata={"doc_id": "caveats_gaia_cepheid", "kind": "caveat"},
    )
    claims = [
        {
            "text": (
                "The RR Lyrae period-luminosity distances have a typical "
                "relative uncertainty of 10%."
            ),
            "evidence_ids": [
                "doc:caveats_gaia_rrlyrae#0",
                "doc:caveats_gaia_cepheid#0",
            ],
        }
    ]
    report = check_claims(claims, [rrlyrae_doc(), cepheid])
    assert not report.passed, "10% is only available from the Cepheid document"
    assert "gaia_cepheid" in report.issues[0].cited_subsets


def test_claim_without_a_subject_is_not_flagged():
    claims = [{"text": "The distance is derived.", "evidence_ids": ["doc:caveats_gaia_gcns#0"]}]
    assert check_claims(claims, [gcns_doc()]).passed


# --- guard ------------------------------------------------------------


def _decide(text, claims, evidence, verdict="supported"):
    numeric = verify_numbers(text, evidence)
    claim_report = ClaimJudge(StubLLM(verdict)).judge(claims, evidence)
    subject = check_claims(claims, evidence)
    return decide(
        model_said_sufficient=True,
        model_answer=text,
        model_missing=None,
        has_evidence=True,
        numeric=numeric,
        claims=claim_report,
        subject=subject,
    )


def test_guard_withholds_the_observed_wrong_answer():
    """Even with the judge saying 'supported', this must not be served."""
    text = "RR Lyrae distances are reliable to 1%."
    claims = [
        {
            "text": text,
            "evidence_ids": [
                "doc:caveats_gaia_rrlyrae#0",
                "doc:caveats_gaia_gcns#0",
            ],
        }
    ]
    decision = _decide(text, claims, [rrlyrae_doc(), gcns_doc()])
    assert decision.outcome is Outcome.WITHHELD_UNGROUNDED
    assert not decision.served


def test_guard_serves_a_clean_answer():
    text = "RR Lyrae distances are good to roughly 15-20%."
    claims = [{"text": text, "evidence_ids": ["doc:caveats_gaia_rrlyrae#0"]}]
    decision = _decide(text, claims, [rrlyrae_doc()])
    assert decision.outcome is Outcome.ANSWERED, decision.reasons
    assert decision.served


def test_guard_withholds_on_contradiction():
    text = "RR Lyrae distances are good to roughly 15-20%."
    claims = [{"text": text, "evidence_ids": ["doc:caveats_gaia_rrlyrae#0"]}]
    decision = _decide(text, claims, [rrlyrae_doc()], verdict="contradicted")
    assert decision.outcome is Outcome.WITHHELD_UNGROUNDED


def test_guard_withholds_on_low_claim_support():
    text = "RR Lyrae distances are good to roughly 15-20%."
    claims = [{"text": text, "evidence_ids": ["doc:caveats_gaia_rrlyrae#0"]}]
    decision = _decide(text, claims, [rrlyrae_doc()], verdict="unsupported")
    assert decision.outcome is Outcome.WITHHELD_UNGROUNDED


def test_guard_abstains_without_evidence():
    decision = decide(
        model_said_sufficient=False,
        model_answer="",
        model_missing=None,
        has_evidence=False,
        numeric=verify_numbers("", []),
        claims=ClaimReport(),
    )
    assert decision.outcome is Outcome.ABSTAINED_NO_EVIDENCE
    assert not decision.served


def test_guard_respects_model_abstention():
    decision = decide(
        model_said_sufficient=False,
        model_answer="",
        model_missing="no mass column in the catalogue",
        has_evidence=True,
        numeric=verify_numbers("", [cepheid_row()]),
        claims=ClaimReport(),
    )
    assert decision.outcome is Outcome.ABSTAINED_BY_MODEL
    assert "mass column" in decision.answer


def test_claim_citing_unknown_id_is_unsupported_without_a_judge_call():
    llm = StubLLM("supported")
    report = ClaimJudge(llm).judge(
        [{"text": "Something.", "evidence_ids": ["doc:does_not_exist#9"]}],
        [rrlyrae_doc()],
    )
    assert report.verdicts[0].verdict == "unsupported"
    assert llm.calls == 0, "must not consult the judge for a dangling citation"


def test_empty_claim_list_scores_zero():
    assert ClaimReport().score == 0.0


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failed += 1
                print(f"FAIL {name}: {exc}")
    print(f"\n{failed} failure(s)")
    raise SystemExit(1 if failed else 0)
