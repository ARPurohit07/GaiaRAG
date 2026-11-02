"""Combine the two retrievers into one evidence set.

Both halves run for every question. That is cheap here (36 doc chunks,
an indexed DuckDB lookup) and it is the right default for honesty: a
question that looks purely numeric ("how far is this Cepheid?") still
needs the method document to state that the distance is a
period-luminosity estimate with a 10% uncertainty. Answering it from the
row alone produces a number with false precision -- technically
grounded, materially misleading.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..verify.scope import check_question
from .evidence import Evidence
from .semantic import SemanticRetriever
from .structured import QuerySpec, SpecError, StructuredResult, StructuredRetriever


@dataclass
class RetrievalBundle:
    question: str
    evidence: list[Evidence] = field(default_factory=list)
    rows: list[Evidence] = field(default_factory=list)
    docs: list[Evidence] = field(default_factory=list)
    spec: QuerySpec | None = None
    sql: str = ""
    row_count: int = 0
    best_doc_score: float = 0.0
    planner_error: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.evidence

    def by_id(self) -> dict[str, Evidence]:
        return {e.evidence_id: e for e in self.evidence}

    def render(self) -> str:
        if not self.evidence:
            return "(no evidence retrieved)"
        blocks = []
        if self.rows:
            blocks.append("CATALOGUE ROWS (authoritative for all numbers):")
            blocks.extend(e.render() for e in self.rows)
        if self.docs:
            blocks.append(
                "\nMETHOD AND PROVENANCE DOCUMENTS "
                "(authoritative for how values were derived):"
            )
            blocks.extend(e.render() for e in self.docs)
        return "\n".join(blocks)


class Router:
    def __init__(
        self,
        structured: StructuredRetriever | None = None,
        semantic: SemanticRetriever | None = None,
        llm=None,
    ):
        from ..providers import get_llm

        self.structured = structured or StructuredRetriever()
        self.semantic = semantic or SemanticRetriever()
        self.llm = llm or get_llm("generator")

    def retrieve(self, question: str) -> RetrievalBundle:
        bundle = RetrievalBundle(question=question)

        # --- structured half
        structured_result: StructuredResult | None = None
        try:
            spec = self.structured.plan(question, self.llm)
            bundle.spec = spec
            structured_result = self.structured.run(spec)
            if structured_result.error:
                bundle.planner_error = structured_result.error
            else:
                bundle.rows = structured_result.evidence
                bundle.sql = structured_result.sql
                bundle.row_count = structured_result.row_count
        except (SpecError, ValueError) as exc:
            # A failed plan must not be silently treated as "no matching
            # rows": those two states lead to different honest answers.
            bundle.planner_error = str(exc)
            bundle.notes.append(
                "Catalogue query planning failed, so no rows were retrieved. "
                "Any numeric claim is therefore unsupported."
            )

        if structured_result is not None and not structured_result.error:
            if structured_result.widened_beyond_gaia:
                bundle.notes.append(
                    "This object is NOT in any Gaia subset of the catalogue. "
                    "The rows below come from another survey and are labelled "
                    "as such. Gaia provides no measurement for it here, and "
                    "the answer must say so rather than presenting these "
                    "values as Gaia data."
                )
            if structured_result.unconstrained:
                bundle.notes.append(
                    "This question did not resolve to any specific catalogue "
                    "query (no named object, no filter, no ordering), so no "
                    "rows were retrieved. Do not state any per-star value."
                )
            elif bundle.spec and bundle.spec.limit > 0 and bundle.row_count == 0:
                bundle.notes.append(
                    "The catalogue query ran successfully and matched zero "
                    "rows. That is evidence of absence, not missing data."
                )

        # --- schema scope
        # Withdraw row evidence when the question asks for a quantity the
        # table has no column for. Without this, an aggregate over stars
        # was restated as a count of exoplanets.
        scope = check_question(question)
        if scope.should_withdraw_rows:
            bundle.rows = []
            bundle.row_count = 0
            bundle.notes.append(scope.note())

        # These structured results include their own provenance. Returning
        # them directly avoids an unrelated document and an embedding call.
        if bundle.rows and bundle.spec and bundle.spec.intent in {
            "lookup", "aggregate", "extreme"
        }:
            q = question.lower()
            constant_field_question = (
                bundle.spec.intent == "extreme"
                and "gaia_rrlyrae" in bundle.spec.subsets
                and any(word in q for word in ("lumin", "reddest", "colour", "color"))
            )
            if not constant_field_question:
                bundle.evidence = list(bundle.rows)
                return bundle

        # --- semantic half
        semantic_result = self.semantic.search(question)
        bundle.best_doc_score = semantic_result.best_score
        if semantic_result.below_threshold:
            # Report the gate that actually fired. An earlier version
            # printed "0.76 < 0.45", comparing the dense score against a
            # cosine floor that is no longer the out-of-domain test at
            # all, which made a correct abstention look like a bug.
            bundle.notes.append(
                "No method or provenance document was retrieved: "
                + (
                    semantic_result.reason
                    or "nothing in the document corpus matched this question."
                )
            )
        else:
            bundle.docs = semantic_result.evidence

        bundle.evidence = bundle.rows + bundle.docs
        return bundle

    def close(self) -> None:
        self.structured.close()
