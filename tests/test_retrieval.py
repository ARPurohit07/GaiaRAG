"""Tests for the parts that must not regress silently.

These cover the specific bugs found while building the pipeline. Each one
is a case where the system produced a confident, well-formed, wrong
answer -- the failures that are invisible without a test.

Run: python tests/test_retrieval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gaia_rag.ingest.catalog import GAIA_SUBSETS, classify
from gaia_rag.retrieve.lexical import BM25, stem, tokenize
from gaia_rag.retrieve.structured import QuerySpec, SpecError, compile_spec
from gaia_rag.verify.numeric import _scan
from gaia_rag.verify.subject import detect_subsets


# --- label classification --------------------------------------------


def test_specific_labels_beat_generic_gaia_prefix():
    """'Gaia Cepheid 123' must not fall into the plain 'Gaia ' bucket."""
    assert classify("Gaia Cepheid 2021462302494358912") == (
        "gaia_cepheid",
        "2021462302494358912",
    )
    assert classify("Gaia RR Lyrae 414081061185873792") == (
        "gaia_rrlyrae",
        "414081061185873792",
    )
    assert classify("Gaia 202631544805348736") == (
        "gaia_dr3_luminous",
        "202631544805348736",
    )
    assert classify("GCNS 21") == ("gaia_gcns", "21")
    assert classify("APOGEE 2M02575144+0034518") == (
        "apogee",
        "2M02575144+0034518",
    )


def test_non_gaia_labels_are_not_gaia():
    for label in ("Gl 35", "", "Wo 9722"):
        subset, _ = classify(label)
        assert subset not in GAIA_SUBSETS, label


# --- numeric verification --------------------------------------------


def test_percentage_is_data_not_prose():
    """'reliable to 1%' must be checkable.

    This slipped through as a small integer and was the entire substance
    of a wrong answer (1% is GCNS precision; RR Lyrae is ~18%).
    """
    value, literal, unit = _scan("reliable to 1%")[0]
    assert value == 1.0
    assert unit == "%"


def test_units_are_detected():
    cases = {
        "548 pc": "pc",
        "1.3 kpc": "kpc",
        "97.29 ly": "ly",
        "0.64 mag": "mag",
        "12.5 degrees": "degrees",
    }
    for text, expected in cases.items():
        _value, _literal, unit = _scan(text)[0]
        assert unit == expected, (text, unit)


def test_bare_small_integers_have_no_unit():
    for text in ("4 stars", "2 subsets", "the 5 brightest"):
        scanned = _scan(text)
        assert scanned and scanned[0][2] is None, text


def test_unit_regex_is_not_corrupted():
    """Guards against a literal backspace sneaking into the pattern.

    An earlier version ended with \\x08 instead of a word boundary, which
    made unit detection match nothing at all while looking correct.
    """
    from gaia_rag.verify.numeric import _UNIT_SUFFIX

    assert "\x08" not in _UNIT_SUFFIX.pattern


# --- lexical retrieval ------------------------------------------------


def test_plural_stemming_matches_singular_query():
    """'Cepheid' must match a document saying 'Cepheids'."""
    assert stem("cepheids") == stem("cepheid") == "cepheid"
    assert stem("distances") == stem("distance") == "distance"
    assert stem("galaxies") == "galaxy"


def test_stemmer_preserves_domain_terms():
    # "mas" (milliarcseconds) must not be produced from "mass", and
    # "radius" must survive intact.
    assert stem("mass") == "mass"
    assert stem("radius") == "radius"
    assert stem("lyrae") == "lyrae"
    assert stem("analysis") == "analysis"


def test_out_of_domain_query_has_no_specific_terms():
    corpus = [
        "Gaia DR3 RR Lyrae distances from period-luminosity relations",
        "Gaia Catalogue of Nearby Stars parallax distances within 100 pc",
        "APOGEE red giants with StarHorse distances about 25 percent",
    ]
    bm25 = BM25(corpus)
    assert bm25.specific_matched_terms("What is the capital of France?") == []
    # "about" alone must not qualify as topical overlap.
    assert bm25.specific_matched_terms("Tell me about quantum chromodynamics") == []
    assert bm25.specific_matched_terms("How reliable are RR Lyrae distances?")


# --- subject consistency ---------------------------------------------


def test_subject_detection():
    assert detect_subsets("How reliable are RR Lyrae distances?") == {"gaia_rrlyrae"}
    assert detect_subsets("How were Cepheid distances derived?") == {"gaia_cepheid"}
    assert detect_subsets("Are APOGEE distances from Gaia?") == {"apogee"}
    assert detect_subsets("How far is that star?") == set()


def test_subject_keywords_do_not_collide():
    """No keyword may be a substring of another subset's keyword."""
    from gaia_rag.verify.subject import SUBJECT_KEYWORDS

    flat = [
        (subset, kw)
        for subset, kws in SUBJECT_KEYWORDS.items()
        for kw in kws
    ]
    for subset_a, kw_a in flat:
        for subset_b, kw_b in flat:
            if subset_a == subset_b or kw_a == kw_b:
                continue
            assert kw_a not in kw_b, f"{kw_a!r} is inside {kw_b!r}"


# --- query spec safety -----------------------------------------------


def test_spec_rejects_unknown_fields():
    for bad in (
        {"filters": [{"field": "password", "op": "=", "value": 1}]},
        {"filters": [{"field": "dist_pc", "op": "DROP", "value": 1}]},
        {"order_by": {"field": "; DELETE FROM stars", "dir": "asc"}},
        {"aggregate": {"fn": "exec", "field": "dist_pc"}},
        {"subsets": ["gaia_everything"]},
    ):
        try:
            QuerySpec.from_dict(bad)
        except SpecError:
            continue
        raise AssertionError(f"spec should have been rejected: {bad}")


def test_spec_values_are_parameterised():
    """User-controlled values must never be inlined into SQL."""
    spec = QuerySpec.from_dict(
        {
            "intent": "lookup",
            "name": "'; DROP TABLE stars; --",
            "filters": [{"field": "dist_pc", "op": "<", "value": 500}],
        }
    )
    sql, params = compile_spec(spec)
    assert "DROP" not in sql
    assert "'; DROP TABLE stars; --" in params
    assert sql.count("?") == len(params)


def test_limit_is_capped():
    spec = QuerySpec.from_dict({"limit": 100000})
    from gaia_rag.config import CONFIG

    assert spec.limit <= CONFIG.retrieval.max_rows


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
