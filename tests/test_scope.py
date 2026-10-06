"""Tests for schema scope, planner reconciliation and lookup widening.

Each covers a failure the gold set caught after the first full eval run,
where the pipeline served a confident wrong answer or fell silent when it
had the evidence to answer.

Needs the built catalogue (`python scripts/build_all.py`), but no model:
`plan()` is driven with a stub so the reconciliation logic is tested on
its own.

Run: python tests/test_scope.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gaia_rag.retrieve.structured import QuerySpec, StructuredRetriever
from gaia_rag.verify.scope import check_question


class SpecLLM:
    """Returns a fixed planner spec, standing in for the model."""

    def __init__(self, spec: dict):
        self.name = "stub"
        self.spec = spec

    def complete_json(self, system, user, *, temperature=0.0):
        return dict(self.spec)

    def complete(self, system, user, *, temperature=0.0):
        return ""


# --- schema scope -----------------------------------------------------


def test_absent_concepts_withdraw_rows():
    """The exoplanet case: a star count was served as an exoplanet count."""
    report = check_question(
        "How many confirmed exoplanets orbit stars in the Gaia Cepheid sample?"
    )
    assert report.should_withdraw_rows
    assert "exoplanets" in report.absent


def test_absent_fields_are_recognised():
    for question, expected in (
        ("What is the stellar mass of Gaia Cepheid 405400004999857062?", "stellar mass"),
        ("What is the parallax of GCNS 21?", "parallax"),
        ("What is the pulsation period of that Cepheid?", "pulsation period"),
        ("What is the effective temperature of this star?", "effective temperature"),
    ):
        report = check_question(question)
        assert report.should_withdraw_rows, question
        assert expected in report.absent, (question, report.absent)


def test_present_fields_are_not_blocked():
    for question in (
        "What is the most distant Cepheid?",
        "How many Cepheids lie within 3 kpc?",
        "How far away is the nearest GCNS star in light-years?",
        "What is the BP-RP colour of that star?",
    ):
        assert not check_question(question).should_withdraw_rows, question


def test_explanatory_questions_are_not_blocked():
    """A documented caveat is answerable even when the column is absent.

    "How does metallicity affect RR Lyrae distances?" is answered by the
    method documents; blocking it would be over-correction.
    """
    for question in (
        "How does metallicity affect RR Lyrae distances?",
        "Why does period matter for Cepheid distances?",
        "Does extinction bias these distances?",
    ):
        assert not check_question(question).should_withdraw_rows, question


# --- planner reconciliation ------------------------------------------


def _retriever() -> StructuredRetriever:
    return StructuredRetriever()


def test_invented_subset_is_dropped_for_superlatives():
    """For an extreme query the scope is the answer, so the model may not
    narrow it unprompted.

    The planner added `subset = 'gaia_dr3_luminous'` to "what is the most
    distant Gaia star?", turning the answer into the most distant
    *luminous* star: 5,000 pc instead of 29,999.2 pc. Both numbers look
    perfectly reasonable, which is why narrowing is not the safe
    direction people assume.
    """
    retriever = _retriever()
    try:
        llm = SpecLLM(
            {
                "intent": "extreme",
                "filters": [
                    {"field": "subset", "op": "=", "value": "gaia_dr3_luminous"}
                ],
                "limit": 1,
            }
        )
        spec = retriever.plan(
            "What is the most distant Gaia star in the catalogue?", llm
        )
        assert spec.subsets == []
        assert all(f["field"] != "subset" for f in spec.filters), spec.filters
        result = retriever.run(spec)
        facts = {f.field_name: f.value for f in result.evidence[0].numeric_facts}
        assert abs(facts["dist_pc"] - 29999.2) < 1.0, facts["dist_pc"]
    finally:
        retriever.close()


def test_invented_subset_is_dropped_for_aggregates():
    retriever = _retriever()
    try:
        llm = SpecLLM(
            {
                "intent": "aggregate",
                "aggregate": {"fn": "count"},
                "filters": [
                    {"field": "subset", "op": "=", "value": "gaia_gcns"}
                ],
            }
        )
        spec = retriever.plan("How many stars are in the catalogue?", llm)
        assert spec.subsets == []
        result = retriever.run(spec)
        assert result.aggregate_value == 120000, result.aggregate_value
    finally:
        retriever.close()


def test_planner_subsets_survive_for_row_queries():
    """For lookup/filter the planner's scope is kept when detection is silent.

    Here the scope only selects which rows are shown, so discarding a
    plausible filter loses information for no safety gain -- and an empty
    named lookup is already recovered by the widening in `run()`.
    Superlatives and aggregates are the opposite case, covered above:
    there the scope IS the answer, so an unprompted narrowing is a wrong
    answer rather than a narrower one.
    """
    retriever = _retriever()
    try:
        llm = SpecLLM(
            {
                "intent": "filter",
                "subsets": ["gaia_cepheid"],
                "filters": [{"field": "dist_pc", "op": "<", "value": 5000}],
                "limit": 5,
            }
        )
        # A question with no recognisable subset term.
        spec = retriever.plan("Which of those are nearby?", llm)
        assert spec.subsets == ["gaia_cepheid"], spec.subsets
    finally:
        retriever.close()


def test_question_subset_still_wins_for_aggregates():
    """Clearing applies only when the question names nothing.

    The 120,000-vs-2,187 failure came from detection missing the plural
    in "Cepheids", not from the clearing rule itself.
    """
    retriever = _retriever()
    try:
        llm = SpecLLM({"intent": "aggregate", "aggregate": {"fn": "count"}})
        spec = retriever.plan(
            "How many classical Cepheids are in the catalogue?", llm
        )
        assert spec.subsets == ["gaia_cepheid"]
        assert retriever.run(spec).aggregate_value == 2187
    finally:
        retriever.close()


def test_extreme_orders_by_the_field_the_question_names():
    """A superlative's ordering is its definition.

    Defaulting every `extreme` query to distance made "largest BP-RP
    colour among the Cepheids" retrieve the ten most DISTANT Cepheids and
    report the largest colour among those ten (0.880 for a true 1.05) --
    a wrong answer that passed every gate, since 0.880 really was in the
    evidence.
    """
    from gaia_rag.retrieve.extremes import infer_extreme_order

    cases = {
        "What is the largest BP-RP colour index among the Cepheids?": (
            "color_bp_rp",
            "desc",
        ),
        "Which RR Lyrae has the bluest colour?": ("color_bp_rp", "asc"),
        "Which Cepheid is the most luminous?": ("lum_lsun", "desc"),
        "Which GCNS star is faintest?": ("lum_lsun", "asc"),
        "What is the most distant Cepheid?": ("dist_pc", "desc"),
        "Which Cepheid is closest to the Sun?": ("dist_pc", "asc"),
    }
    for question, (field, direction) in cases.items():
        got = infer_extreme_order(question)
        assert got == {"field": field, "dir": direction}, (question, got)
    # A non-superlative question implies no ordering.
    assert infer_extreme_order("How many Cepheids are there?") is None


def test_reddest_cepheid_returns_the_true_maximum():
    retriever = _retriever()
    try:
        llm = SpecLLM({"intent": "extreme", "limit": 10})
        spec = retriever.plan(
            "What is the largest BP-RP colour index among the classical "
            "Cepheids?",
            llm,
        )
        assert spec.order_by == {"field": "color_bp_rp", "dir": "desc"}
        result = retriever.run(spec)
        facts = {f.field_name: f.value for f in result.evidence[0].numeric_facts}
        assert abs(facts["color_bp_rp"] - 1.05) < 0.001, facts["color_bp_rp"]
    finally:
        retriever.close()


def test_bare_luminous_adjective_is_not_the_luminous_subset():
    """"most luminous RR Lyrae" must not also name the luminous sample.

    A bare "luminous" keyword fixed one bug and created another: an RR
    Lyrae question was scoped to RR Lyrae *and* the luminous subset.
    """
    from gaia_rag.verify.subject import detect_subsets

    assert detect_subsets("Which RR Lyrae star is the most luminous?") == {
        "gaia_rrlyrae"
    }
    assert detect_subsets("Which Cepheid is the most luminous?") == {
        "gaia_cepheid"
    }
    # Still detected when it genuinely names the subset.
    assert detect_subsets("How many luminous Gaia DR3 stars are there?") == {
        "gaia_dr3_luminous"
    }
    assert detect_subsets("the luminous sample") == {"gaia_dr3_luminous"}


def test_luminous_phrasing_is_detected():
    """"luminous Gaia DR3 stars" must resolve to the luminous subset.

    The keyword list held the contiguous phrase "luminous star", so the
    natural phrasing with two words in between matched nothing, scope
    widened to all of Gaia, and five cases answered with all-Gaia totals
    (120,000 for 45,000; 29,999 pc for 5,000 pc).
    """
    from gaia_rag.verify.subject import detect_subsets

    assert detect_subsets("How many luminous Gaia DR3 stars are there?") == {
        "gaia_dr3_luminous"
    }
    assert detect_subsets("the most distant luminous Gaia DR3 star") == {
        "gaia_dr3_luminous"
    }
    # Must not collide with the luminosity field, a different token.
    assert detect_subsets("What are the caveats on luminosity values?") == set()


def test_generated_question_phrasings_are_all_detectable():
    """Build-time guard against the generator and detector disagreeing.

    Nothing otherwise ties QUESTION_NAMES to SUBJECT_KEYWORDS, and a
    mismatch does not raise -- it silently widens scope and returns a
    confident wrong number.
    """
    from gaia_rag.eval.goldset.build_goldset import _check_phrasings

    _check_phrasings()  # raises if any phrasing is undetectable


def test_phrasing_guard_detects_a_mismatch():
    import gaia_rag.verify.subject as subj
    from gaia_rag.eval.goldset.build_goldset import _check_phrasings

    original = subj.SUBJECT_KEYWORDS["gaia_dr3_luminous"]
    subj.SUBJECT_KEYWORDS["gaia_dr3_luminous"] = ("luminous star",)
    try:
        _check_phrasings()
    except RuntimeError:
        return
    finally:
        subj.SUBJECT_KEYWORDS["gaia_dr3_luminous"] = original
    raise AssertionError("guard should have rejected the contiguous-only keyword")


def test_plural_subset_terms_are_detected():
    """"Cepheids" must map to gaia_cepheid, not to nothing."""
    from gaia_rag.verify.subject import detect_subsets

    assert detect_subsets("How many classical Cepheids are there?") == {
        "gaia_cepheid"
    }
    assert detect_subsets("RR Lyrae stars") == {"gaia_rrlyrae"}
    assert detect_subsets("supergiants in the disk") == {"gaia_dr3_luminous"}


def test_aggregate_keeps_the_question_subset():
    """End to end on the regression: the count must scope to Cepheids."""
    retriever = _retriever()
    try:
        llm = SpecLLM({"intent": "aggregate", "aggregate": {"fn": "count"}})
        spec = retriever.plan("How many classical Cepheids are in the catalogue?", llm)
        assert spec.subsets == ["gaia_cepheid"]
        result = retriever.run(spec)
        assert result.aggregate_value == 2187, result.aggregate_value
    finally:
        retriever.close()


def test_question_named_subset_is_applied():
    """A subset the question actually names survives reconciliation."""
    retriever = _retriever()
    try:
        llm = SpecLLM({"intent": "filter", "limit": 5})
        spec = retriever.plan("How reliable are RR Lyrae distances?", llm)
        assert spec.subsets == ["gaia_rrlyrae"]
    finally:
        retriever.close()


def test_planner_subset_is_overridden_by_the_question():
    retriever = _retriever()
    try:
        llm = SpecLLM({"intent": "filter", "subsets": ["gaia_gcns"], "limit": 5})
        spec = retriever.plan("How were Cepheid distances derived?", llm)
        assert spec.subsets == ["gaia_cepheid"]
    finally:
        retriever.close()


# --- lookup widening -------------------------------------------------


def test_named_lookup_widens_beyond_gaia():
    """Betelgeuse is HYG-only; the lookup must find it and flag it."""
    retriever = _retriever()
    try:
        spec = QuerySpec.from_dict(
            {"intent": "lookup", "name": "Betelgeuse", "gaia_only": True}
        )
        result = retriever.run(spec)
        assert result.row_count == 1, result.row_count
        assert result.widened_beyond_gaia
        evidence = result.evidence[0]
        assert evidence.metadata["subset"] == "hyg"
        assert evidence.metadata["is_gaia"] is False
        assert "NOT from Gaia" in evidence.text
    finally:
        retriever.close()


def test_gaia_object_does_not_widen():
    retriever = _retriever()
    try:
        spec = QuerySpec.from_dict(
            {"intent": "lookup", "name": "Gaia Cepheid 4054000049998570624"}
        )
        result = retriever.run(spec)
        assert result.row_count >= 1
        assert not result.widened_beyond_gaia
    finally:
        retriever.close()


def test_unconstrained_plan_returns_no_rows():
    """An arbitrary LIMIT n sample is not evidence."""
    retriever = _retriever()
    try:
        result = retriever.run(QuerySpec.from_dict({"intent": "filter", "limit": 10}))
        assert result.row_count == 0
        assert result.unconstrained
    finally:
        retriever.close()


# --- superlative grounding -------------------------------------------


def test_ordered_query_records_its_ranking():
    """A lone row cannot support "closest" unless the evidence says so."""
    retriever = _retriever()
    try:
        spec = QuerySpec.from_dict(
            {
                "intent": "filter",
                "subsets": ["gaia_cepheid"],
                "order_by": {"field": "dist_pc", "dir": "asc"},
                "limit": 1,
            }
        )
        result = retriever.run(spec)
        assert result.row_count == 1
        text = result.evidence[0].text
        assert "lowest dist_pc" in text, text
        assert "nearest" in text, text
        assert "2,187" in text, text
    finally:
        retriever.close()


def test_degenerate_field_is_flagged_instead_of_ranked():
    """A field with one distinct value has no maximum.

    All 67,813 RR Lyrae rows carry luminosity 45 and colour 0.35, because
    the distance method assumed a fixed absolute magnitude and intrinsic
    colour. Asked which is most luminous, the pipeline named one -- the
    caveat document was retrieved and ignored -- so the fact is now
    stated in the row's own evidence, which is much harder to overlook.
    """
    retriever = _retriever()
    try:
        for field, question in (
            ("lum_lsun", "Which RR Lyrae star is the most luminous?"),
            ("color_bp_rp", "Which RR Lyrae has the reddest colour?"),
        ):
            spec = QuerySpec.from_dict(
                {
                    "intent": "extreme",
                    "subsets": ["gaia_rrlyrae"],
                    "order_by": {"field": field, "dir": "desc"},
                    "limit": 1,
                }
            )
            assert retriever._distinct_count(spec, field) == 1, field
            text = retriever.run(spec).evidence[0].text
            assert "share the SAME" in text, (field, text[-200:])
            assert "meaningless" in text, field

        # A field that genuinely varies still gets a ranking statement.
        spec = QuerySpec.from_dict(
            {
                "intent": "extreme",
                "subsets": ["gaia_cepheid"],
                "order_by": {"field": "lum_lsun", "dir": "desc"},
                "limit": 1,
            }
        )
        assert retriever._distinct_count(spec, "lum_lsun") > 1
        text = retriever.run(spec).evidence[0].text
        assert "has the highest lum_lsun" in text, text[-200:]
        assert "share the SAME" not in text
    finally:
        retriever.close()


def test_aggregate_states_what_it_counts():
    """A bare "count = 2187" was restated as 2187 exoplanets."""
    retriever = _retriever()
    try:
        spec = QuerySpec.from_dict(
            {
                "intent": "aggregate",
                "subsets": ["gaia_cepheid"],
                "aggregate": {"fn": "count"},
            }
        )
        result = retriever.run(spec)
        text = result.evidence[0].text
        assert "one catalogue row is one star" in text, text
        assert "not a count of any other kind of object" in text, text
    finally:
        retriever.close()


# --- extreme-intent normalisation ------------------------------------


def test_extreme_intent_gets_an_explicit_ordering():
    """`intent: extreme` must become a real order_by on the spec.

    Leaving the ordering inside compile_spec broke two things at once:
    the unconstrained guard saw no ordering and dropped every row, and
    the superlative annotation was skipped so "this is the most distant"
    came back unsupported. Every "farthest" question failed while every
    "nearest" one passed, because "closest" makes the planner emit an
    explicit ascending order_by.
    """
    spec = QuerySpec.from_dict({"intent": "extreme", "subsets": ["gaia_cepheid"]})
    assert spec.order_by == {"field": "dist_pc", "dir": "desc"}


def test_bare_extreme_query_returns_rows():
    """An extreme query with no filters must not be treated as unconstrained."""
    retriever = _retriever()
    try:
        result = retriever.run(QuerySpec.from_dict({"intent": "extreme", "limit": 1}))
        assert not result.unconstrained
        assert result.row_count == 1, result.row_count
        text = result.evidence[0].text
        assert "highest dist_pc" in text, text
        assert "farthest" in text, text
    finally:
        retriever.close()


def test_extreme_respects_an_explicit_ascending_order():
    """An explicit order_by must survive normalisation."""
    spec = QuerySpec.from_dict(
        {"intent": "extreme", "order_by": {"field": "dist_pc", "dir": "asc"}}
    )
    assert spec.order_by["dir"] == "asc"


def test_farthest_cepheid_is_retrievable_end_to_end():
    retriever = _retriever()
    try:
        spec = QuerySpec.from_dict(
            {"intent": "extreme", "subsets": ["gaia_cepheid"], "limit": 1}
        )
        result = retriever.run(spec)
        assert result.row_count == 1
        facts = {f.field_name: f.value for f in result.evidence[0].numeric_facts}
        assert abs(facts["dist_pc"] - 24952.0) < 1.0, facts["dist_pc"]
    finally:
        retriever.close()


# --- multi-subject retrieval coverage -------------------------------


def test_multi_subject_question_covers_every_subset():
    """A comparison must retrieve a document for each subset it names.

    Without this, "GCNS parallax vs RR Lyrae period-luminosity" returned
    the GCNS and *Cepheid* caveats and no RR Lyrae caveat, because
    "period-luminosity" is Cepheid vocabulary. The generator then
    reported the Cepheid uncertainty (10%) as the RR Lyrae figure (18%),
    and the judge called it supported, since the cited document does say
    10%. A missing document is worse than an irrelevant one.
    """
    from gaia_rag.retrieve.semantic import SemanticRetriever

    r = SemanticRetriever()
    ids = [
        e.evidence_id
        for e in r.search(
            "Which are more precise, the GCNS parallax distances or the "
            "RR Lyrae period-luminosity distances?"
        ).evidence
    ]
    assert any("gaia_gcns" in i for i in ids), ids
    assert any("gaia_rrlyrae" in i for i in ids), ids


def test_single_subject_retrieval_is_unchanged():
    """Coverage logic must not perturb the ordinary single-subject path."""
    from gaia_rag.retrieve.semantic import SemanticRetriever

    r = SemanticRetriever()
    for question, want in (
        ("How reliable are the RR Lyrae distances?", "rrlyrae"),
        ("How were Cepheid distances derived?", "cepheid"),
        ("Why is APOGEE not a Gaia distance?", "apogee"),
    ):
        ids = [e.evidence_id for e in r.search(question).evidence]
        assert any(want in i for i in ids), (question, ids)


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
