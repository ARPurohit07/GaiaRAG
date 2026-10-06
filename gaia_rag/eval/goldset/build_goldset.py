"""Build the gold set: curated traps plus cases generated from the data.

    python -m gaia_rag.eval.goldset.build_goldset

Two populations, kept distinct because they measure different things and
averaging them together would mislead:

* **curated** -- hand-written adversarial cases. Each encodes one
  specific way a RAG lies: an uncertainty borrowed from another survey, a
  non-Gaia row presented as Gaia, a question about a column that does not
  exist, a field that looks per-star but is a constant. These carry the
  faithfulness signal and cannot be generated, because generating one
  would require already knowing the failure.
* **generated** -- expected values computed by SQL here, at build time,
  so they are correct by construction and regenerate whenever the
  catalogue changes. These test plumbing: query planning, numeric
  grounding, unit handling.

Two rules keep the generated half honest:

1. A shape is only generated where its answer is **well defined**. Gaia
   rows carry no constellation, spectral type or proper name, and RR
   Lyrae colour and luminosity are single constants across all 67,813
   rows (artefacts of the assumed absolute magnitude), so ranking RR
   Lyrae by them has no answer. Generating those would have produced
   cases with no correct result.
2. Thresholds come from the data (medians, actual extremes) rather than
   round numbers I guessed, so no case is accidentally empty or trivial.

Every case records in `note` how its expected value was obtained.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

from ...config import CONFIG

OUT = Path(__file__).resolve().parent / "gaia_gold.jsonl"

#: Human phrasing per Gaia subset, as a question would name it. These
#: must be detectable by `verify.subject.detect_subsets`, which
#: `_check_phrasings` enforces at build time.
QUESTION_NAMES = {
    "gaia_cepheid": "classical Cepheids",
    "gaia_rrlyrae": "RR Lyrae stars",
    "gaia_gcns": "GCNS nearby stars",
    "gaia_dr3_luminous": "luminous Gaia DR3 stars",
}

#: Subsets whose colour and luminosity vary per star. RR Lyrae is absent
#: deliberately: both fields are constants there, so "most luminous" and
#: "reddest" have no answer.
VARYING_PHOTOMETRY = ("gaia_cepheid", "gaia_gcns", "gaia_dr3_luminous")

#: HYG-only named stars, for Gaia-attribution traps. None of these has a
#: Gaia row, so the correct answer always denies Gaia attribution.
HYG_NAMED = (
    "Betelgeuse",
    "Vega",
    "Sirius",
    "Proxima Centauri",
    "Barnard's Star",
    "Wolf 359",
)

#: (label, question) for fields the `stars` table does not have.
ABSENT_FIELD_QUESTIONS = (
    ("parallax", "What is the parallax of {obj} in milliarcseconds?"),
    ("temperature", "What is the effective temperature of {obj} in kelvin?"),
    ("period", "What is the pulsation period of {obj} in days?"),
    ("mass", "What is the stellar mass of {obj} in solar masses?"),
    ("radius", "What is the radius of {obj} in solar radii?"),
    ("age", "What is the age of {obj} in billions of years?"),
    ("proper_motion", "What is the proper motion of {obj} in mas/yr?"),
    ("radial_velocity", "What is the radial velocity of {obj} in km/s?"),
)

OOD_QUESTIONS = (
    ("capital", "What is the capital of France?"),
    ("recipe", "How do I bake sourdough bread at home?"),
    ("programming", "How do I write a Python decorator that caches results?"),
    ("medical", "What dose of ibuprofen should I take for a headache?"),
    ("legal", "Can my landlord raise my rent without notice?"),
    ("sports", "Who won the 2018 football World Cup?"),
)


def _case(**kw) -> dict:
    base = {
        "id": "",
        "kind": "",
        "source": "curated",
        "question": "",
        "should_answer": True,
        "expected_evidence": [],
        "must_contain_numbers": [],
        "forbidden_numbers": [],
        "must_mention": [],
        "must_mention_any": [],
        "must_not_mention": [],
        "note": "",
    }
    base.update(kw)
    return base


def _check_phrasings() -> None:
    """Assert each generated question names a subset the detector finds.

    Scope for aggregates and superlatives is recovered from the
    question's own words, so a phrasing the detector cannot parse makes
    the query silently widen to all of Gaia -- a confidently wrong
    number, not an error.

    That happened: QUESTION_NAMES said "luminous Gaia DR3 stars" while
    the keyword list held the contiguous phrase "luminous star", so every
    question about that subset went undetected and five cases failed with
    all-Gaia totals. The mismatch was between two files nothing tied
    together, which is exactly what a build-time check is for: it costs
    milliseconds, where finding it through an eval cost an hour.
    """
    from ...verify.subject import detect_subsets

    problems = []
    for subset, name in QUESTION_NAMES.items():
        for template in (
            f"How many {name} are in the catalogue?",
            f"What is the most distant of the {name} in the catalogue?",
        ):
            found = detect_subsets(template)
            if found != {subset}:
                problems.append(
                    f"  {subset}: phrasing {name!r} detects as "
                    f"{sorted(found) or 'nothing'}"
                )
    if problems:
        raise RuntimeError(
            "generated questions are not detectable as their own subset, so "
            "their scope would silently widen:\n"
            + "\n".join(problems)
            + "\n\nFix SUBJECT_KEYWORDS in gaia_rag/verify/subject.py or "
            "QUESTION_NAMES here so the two agree."
        )


# --- generated ---------------------------------------------------------


def generated(con: duckdb.DuckDBPyConnection) -> list[dict]:
    _check_phrasings()
    cases: list[dict] = []

    for subset, name in QUESTION_NAMES.items():
        stats = con.execute(
            """
            SELECT count(*), min(dist_pc), max(dist_pc), avg(dist_pc),
                   median(dist_pc), min(dist_ly), max(dist_ly),
                   max(dist_kpc)
            FROM stars WHERE subset = ?
            """,
            [subset],
        ).fetchone()
        total = int(stats[0])
        min_pc, max_pc = float(stats[1]), float(stats[2])
        avg_pc, median_pc = float(stats[3]), float(stats[4])
        max_ly, max_kpc = float(stats[6]), float(stats[7])

        far = con.execute(
            "SELECT source_id FROM stars WHERE subset = ? "
            "ORDER BY dist_pc DESC LIMIT 1",
            [subset],
        ).fetchone()[0]
        near = con.execute(
            "SELECT source_id FROM stars WHERE subset = ? "
            "ORDER BY dist_pc ASC LIMIT 1",
            [subset],
        ).fetchone()[0]

        # Thresholds from the data, so counts are never empty or trivial.
        within_kpc = round(median_pc / 1000.0, 1) or 0.1
        beyond_kpc = within_kpc
        n_within = int(
            con.execute(
                "SELECT count(*) FROM stars WHERE subset = ? AND dist_kpc < ?",
                [subset, within_kpc],
            ).fetchone()[0]
        )
        n_beyond = int(
            con.execute(
                "SELECT count(*) FROM stars WHERE subset = ? AND dist_kpc > ?",
                [subset, beyond_kpc],
            ).fetchone()[0]
        )

        shapes = [
            (
                "farthest",
                "numeric",
                f"What is the most distant of the {name} in the catalogue, "
                "and how far away is it in parsecs?",
                [round(max_pc, 1)],
                [f"row:{subset}:{far}"],
                f"SELECT max(dist_pc) WHERE subset='{subset}' -> "
                f"{max_pc:.1f} pc (source_id {far}).",
            ),
            (
                "nearest",
                "numeric",
                f"Which of the {name} is closest to the Sun, and at what "
                "distance in parsecs?",
                [round(min_pc, 2)],
                [f"row:{subset}:{near}"],
                f"SELECT min(dist_pc) WHERE subset='{subset}' -> "
                f"{min_pc:.2f} pc (source_id {near}).",
            ),
            (
                "count",
                "aggregate",
                f"How many {name} are in the catalogue?",
                [float(total)],
                ["agg:count"],
                f"SELECT count(*) WHERE subset='{subset}' -> {total}.",
            ),
            (
                "count_within",
                "aggregate",
                f"How many {name} lie within {within_kpc} kpc of the Sun?",
                [float(n_within)],
                ["agg:count"],
                f"count(*) WHERE subset='{subset}' AND dist_kpc < "
                f"{within_kpc} -> {n_within}. Threshold is the subset median, "
                "so the count is non-trivial by construction.",
            ),
            (
                "count_beyond",
                "aggregate",
                f"How many {name} lie more than {beyond_kpc} kpc from the Sun?",
                [float(n_beyond)],
                ["agg:count"],
                f"count(*) WHERE subset='{subset}' AND dist_kpc > "
                f"{beyond_kpc} -> {n_beyond}.",
            ),
            (
                "farthest_ly",
                "unit_conversion",
                f"How far away is the most distant of the {name}, in "
                "light-years?",
                [round(max_ly, 0)],
                [f"row:{subset}:{far}"],
                f"{max_pc:.1f} pc = {max_ly:.0f} ly. Reporting the parsec "
                "figure as light-years is the failure; stating both is "
                "correct, so the parsec value is not forbidden.",
            ),
            (
                "farthest_kpc",
                "unit_conversion",
                f"In kiloparsecs, how far away is the most distant of the "
                f"{name}?",
                [round(max_kpc, 2)],
                [f"row:{subset}:{far}"],
                f"{max_pc:.1f} pc = {max_kpc:.2f} kpc.",
            ),
            (
                "avg_distance",
                "aggregate",
                f"What is the average distance of the {name} in parsecs?",
                [round(avg_pc, 2)],
                ["agg:avg"],
                f"SELECT avg(dist_pc) WHERE subset='{subset}' -> "
                f"{avg_pc:.2f} pc.",
            ),
            (
                "median_distance",
                "aggregate",
                f"What is the median distance of the {name} in parsecs?",
                [round(median_pc, 2)],
                ["agg:median"],
                f"SELECT median(dist_pc) WHERE subset='{subset}' -> "
                f"{median_pc:.2f} pc.",
            ),
        ]

        if subset in VARYING_PHOTOMETRY:
            lum = con.execute(
                "SELECT max(lum_lsun) FROM stars WHERE subset = ?", [subset]
            ).fetchone()[0]
            colour = con.execute(
                "SELECT max(color_bp_rp) FROM stars WHERE subset = ?", [subset]
            ).fetchone()[0]
            shapes.append(
                (
                    "most_luminous",
                    "numeric",
                    f"What is the highest luminosity among the {name}, in "
                    "solar luminosities?",
                    [round(float(lum), 1)],
                    ["row:"],
                    f"SELECT max(lum_lsun) WHERE subset='{subset}' -> "
                    f"{float(lum):.1f} Lsun. Generated only for subsets whose "
                    "luminosity actually varies per star.",
                )
            )
            shapes.append(
                (
                    "reddest",
                    "numeric",
                    f"What is the largest BP-RP colour index among the "
                    f"{name}?",
                    [round(float(colour), 3)],
                    ["row:"],
                    f"SELECT max(color_bp_rp) WHERE subset='{subset}' -> "
                    f"{float(colour):.3f}.",
                )
            )

        for suffix, kind, question, numbers, evidence, note in shapes:
            cases.append(
                _case(
                    id=f"gen_{subset}_{suffix}",
                    kind=kind,
                    source="generated",
                    question=question,
                    expected_evidence=evidence,
                    must_contain_numbers=numbers,
                    note=note,
                )
            )
    return cases


# --- curated -----------------------------------------------------------


def curated(con: duckdb.DuckDBPyConnection) -> list[dict]:
    def one(sql: str, params: list | None = None):
        return con.execute(sql, params or []).fetchone()

    cepheid_far_id = one(
        "SELECT source_id FROM stars WHERE subset='gaia_cepheid' "
        "ORDER BY dist_pc DESC LIMIT 1"
    )[0]
    cepheid_min_pc = float(
        one("SELECT min(dist_pc) FROM stars WHERE subset='gaia_cepheid'")[0]
    )
    gcns = one(
        "SELECT source_id, dist_pc, dist_ly, max(dist_pc) OVER () FROM stars "
        "WHERE subset='gaia_gcns' ORDER BY dist_pc ASC LIMIT 1"
    )
    gcns_id, gcns_min_pc, gcns_min_ly, gcns_max_pc = (
        gcns[0],
        float(gcns[1]),
        float(gcns[2]),
        float(gcns[3]),
    )
    gaia_far = one(
        "SELECT subset, source_id, dist_pc FROM stars WHERE is_gaia "
        "ORDER BY dist_pc DESC LIMIT 1"
    )
    gaia_near = one(
        "SELECT subset, source_id, dist_pc FROM stars WHERE is_gaia "
        "ORDER BY dist_pc ASC LIMIT 1"
    )
    rrl_lum = float(
        one("SELECT max(lum_lsun) FROM stars WHERE subset='gaia_rrlyrae'")[0]
    )

    cases: list[dict] = []

    # --- reliability figures, one per subset (the citation-drift family)
    reliability = [
        (
            "rrlyrae",
            "RR Lyrae",
            "How reliable are the RR Lyrae distances in this catalogue?",
            ["15-20"],
            [1.0, 10.0],
            ["metallicity", "absolute magnitude", "reddening", "standard candle"],
            "RR Lyrae distances are good to 15-20%. 1% is the GCNS parallax "
            "precision and 10% is the Cepheid figure; either here is the "
            "citation-drift failure this family exists to catch.",
        ),
        (
            "cepheid",
            "Cepheid",
            "How reliable are the Cepheid distances in this catalogue?",
            [],
            [1.0, 18.0],
            ["10", "period-luminosity", "wesenheit"],
            "Cepheid distances carry a 10% typical relative uncertainty. "
            "1% (GCNS) or 18% (RR Lyrae) would be borrowed from elsewhere.",
        ),
        (
            "gcns",
            "GCNS",
            "How precise are the GCNS parallax distances?",
            [],
            [18.0],
            ["1", "parallax", "under 1"],
            "GCNS parallax distances are precise to well under 1%.",
        ),
        (
            "apogee",
            "APOGEE",
            "How uncertain are the APOGEE StarHorse distances?",
            [],
            [1.0],
            ["25", "starhorse", "spectroscopic", "photometr"],
            "APOGEE StarHorse distances were selected to under 25% error.",
        ),
    ]
    for key, label, question, must, forbidden, any_of, note in reliability:
        cases.append(
            _case(
                id=f"reliability_{key}",
                kind="trap_wrong_uncertainty",
                question=question,
                expected_evidence=[f"doc:caveats_{'gaia_' + key if key in ('rrlyrae','cepheid','gcns') else key}"],
                must_mention=must,
                forbidden_numbers=forbidden,
                must_mention_any=[any_of],
                note=note,
            )
        )

    # --- Gaia attribution traps over HYG-only named stars
    for name in HYG_NAMED:
        slug = name.lower().replace(" ", "_").replace("'", "")
        cases.append(
            _case(
                id=f"attribution_{slug}",
                kind="trap_attribution",
                question=f"According to Gaia, how far away is {name}?",
                expected_evidence=[f"row:hyg:{name}"],
                must_mention_any=[["not", "no gaia", "hyg", "hipparcos"]],
                note=(
                    f"{name} appears only in the HYG subset, which is "
                    "Hipparcos/Yale/Gliese data. Gaia gives no distance for it "
                    "here, and the HYG value must not be presented as a Gaia "
                    "measurement."
                ),
            )
        )

    cases.append(
        _case(
            id="apogee_not_gaia",
            kind="trap_attribution",
            question="Are the APOGEE distances in this catalogue Gaia measurements?",
            expected_evidence=["doc:caveats_apogee"],
            must_mention_any=[
                ["not gaia", "not a gaia", "not gaia-only", "not purely gaia"],
                ["starhorse", "spectra", "spectroscopic", "photometr", "infrared"],
            ],
            note=(
                "StarHorse combines APOGEE spectra with Gaia EDR3 parallaxes "
                "and infrared photometry; not Gaia-only. Must deny Gaia "
                "attribution AND say what the distance rests on."
            ),
        )
    )
    cases.append(
        _case(
            id="hyg_not_gaia",
            kind="trap_attribution",
            question="Is the HYG part of this catalogue based on Gaia astrometry?",
            expected_evidence=["doc:caveats_hyg"],
            must_mention_any=[["not", "hipparcos", "no"]],
            note="HYG is Hipparcos/Yale/Gliese and must never be called Gaia.",
        )
    )

    # --- the degenerate-field trap, found while designing the shapes
    cases.append(
        _case(
            id="rrlyrae_luminosity_is_constant",
            kind="degenerate_field",
            question="Which RR Lyrae star in the catalogue is the most luminous?",
            should_answer=True,
            expected_evidence=["doc:caveats_field_lum_lsun"],
            must_mention_any=[
                ["same", "constant", "identical", "cannot", "not a per-star", "all"]
            ],
            note=(
                f"Every one of the 67,813 RR Lyrae rows carries the SAME "
                f"luminosity ({rrl_lum:g} Lsun), implied by the assumed "
                "absolute magnitude M_G = 0.64 rather than measured. There is "
                "no most-luminous RR Lyrae, so ranking them is meaningless and "
                "the honest answer says so instead of naming one."
            ),
        )
    )
    cases.append(
        _case(
            id="rrlyrae_colour_is_constant",
            kind="degenerate_field",
            question="Which RR Lyrae star has the reddest BP-RP colour?",
            should_answer=True,
            expected_evidence=["doc:caveats_field_color_bp_rp"],
            must_mention_any=[
                ["same", "constant", "identical", "cannot", "not a per-star", "all"]
            ],
            note=(
                "Every RR Lyrae row carries BP-RP = 0.35, the assumed "
                "intrinsic colour used for the reddening estimate. Ranking "
                "them by colour has no answer."
            ),
        )
    )

    # --- absent columns: the object exists, the field does not
    for field, template in ABSENT_FIELD_QUESTIONS:
        obj = f"Gaia Cepheid {cepheid_far_id}"
        cases.append(
            _case(
                id=f"absent_{field}",
                kind="unanswerable_in_corpus",
                question=template.format(obj=obj),
                should_answer=False,
                note=(
                    f"The stars table has no {field} column for any subset. "
                    "Harder than out-of-domain: the object is real and the "
                    "field sounds like it should be there."
                ),
            )
        )
    cases.append(
        _case(
            id="absent_exoplanets",
            kind="unanswerable_in_corpus",
            question=(
                "How many confirmed exoplanets orbit stars in the Gaia Cepheid "
                "sample?"
            ),
            should_answer=False,
            note=(
                "No exoplanet data is ingested and no cross-match exists. This "
                "case caught the worst observed failure: a count of 2,187 "
                "Cepheid STARS served as '2187 confirmed exoplanets', passing "
                "every value- and subset-level check because both the value "
                "and the subset were correct."
            ),
        )
    )
    cases.append(
        _case(
            id="absent_metallicity_value",
            kind="unanswerable_in_corpus",
            question=f"What is the [Fe/H] of Gaia Cepheid {cepheid_far_id}?",
            should_answer=False,
            note=(
                "No metallicity column. Note the contrast with "
                "`metallicity_caveat`, which asks how metallicity affects RR "
                "Lyrae distances and IS answerable from the documents."
            ),
        )
    )
    cases.append(
        _case(
            id="absent_unnamed_object",
            kind="unanswerable_in_corpus",
            question="How far away is the star Zubeneschamali according to Gaia?",
            should_answer=False,
            note=(
                "Not present in any subset under that name. Must say it is not "
                "in the catalogue rather than offer a nearby row."
            ),
        )
    )

    # --- out of domain
    for key, question in OOD_QUESTIONS:
        cases.append(
            _case(
                id=f"ood_{key}",
                kind="out_of_domain",
                question=question,
                should_answer=False,
                note=(
                    "Out of domain; must abstain rather than answer from "
                    "parametric knowledge."
                    + (
                        " Adversarial: the corpus is built from Python source, "
                        "so code vocabulary appears in it."
                        if key == "programming"
                        else ""
                    )
                ),
            )
        )

    # --- evidence of absence: the query runs and matches nothing
    empties = [
        (
            "cepheids_near",
            "Are there any classical Cepheids within 100 parsecs of the Sun?",
            f"Verified zero: the nearest Cepheid is at {cepheid_min_pc:.1f} pc.",
        ),
        (
            "gcns_far",
            "Does the GCNS subset contain any star beyond 500 parsecs?",
            f"Verified zero: GCNS spans {gcns_min_pc:.2f}-{gcns_max_pc:.1f} pc.",
        ),
        (
            "cepheids_beyond_30kpc",
            "Are there classical Cepheids more than 30 kiloparsecs away?",
            "Verified zero: the most distant Cepheid is at 24.95 kpc.",
        ),
        (
            "gcns_within_10pc",
            "Does the GCNS subset include any star closer than 10 parsecs?",
            f"Verified zero: the nearest GCNS row is {gcns_min_pc:.2f} pc, "
            "because only the 5,000 brightest were kept.",
        ),
    ]
    for key, question, note in empties:
        cases.append(
            _case(
                id=f"empty_{key}",
                kind="empty_result",
                question=question,
                must_mention_any=[["no", "not", "none", "zero"]],
                note=(
                    note
                    + " The query runs and matches nothing, which is evidence "
                    "of absence -- the system should say so, not abstain. The "
                    "negation check accepts any of no/not/none/zero: an earlier "
                    "version required the literal 'not any', which a correct "
                    "answer ('does not contain any star') does not contain."
                ),
            )
        )

    # --- legitimate multi-subject claims: must NOT trip subject drift
    comparisons = [
        (
            "gcns_vs_rrlyrae",
            "Which are more precise, the GCNS parallax distances or the "
            "RR Lyrae period-luminosity distances?",
            ["doc:caveats_gaia_gcns", "doc:caveats_gaia_rrlyrae"],
            [["gcns", "parallax"], ["rr lyrae", "15-20", "18"]],
            "GCNS <1% vs RR Lyrae 15-20%.",
        ),
        (
            "cepheid_vs_rrlyrae",
            "Are Cepheid distances or RR Lyrae distances more reliable here?",
            ["doc:caveats_gaia_cepheid", "doc:caveats_gaia_rrlyrae"],
            [["cepheid", "10"], ["rr lyrae", "15-20", "18"]],
            "Cepheid 10% vs RR Lyrae 15-20%.",
        ),
        (
            "gcns_vs_apogee",
            "Which distances are better determined, GCNS or APOGEE?",
            ["doc:caveats_gaia_gcns", "doc:caveats_apogee"],
            [["gcns", "parallax"], ["apogee", "25", "starhorse"]],
            "GCNS <1% vs APOGEE 25%.",
        ),
    ]
    for key, question, evidence, any_groups, note in comparisons:
        cases.append(
            _case(
                id=f"compare_{key}",
                kind="comparison",
                question=question,
                expected_evidence=evidence,
                must_mention_any=any_groups,
                note=(
                    note
                    + " A claim naming two subsets is legitimate, so the "
                    "subject-consistency check must not flag it. This guards "
                    "against that check over-firing, which once withheld a "
                    "correct answer."
                ),
            )
        )

    # --- method and derivation, one per ingestion path
    methods = [
        (
            "cepheid",
            "How were the Cepheid distances in this catalogue derived?",
            ["doc:caveats_gaia_cepheid"],
            [["period-luminosity", "wesenheit", "period luminosity"]],
            ["parallax inversion"],
            "Wesenheit period-luminosity relation, Ripepi et al. 2022. "
            "Explicitly NOT a parallax distance.",
        ),
        (
            "rrlyrae",
            "How were the RR Lyrae distances determined?",
            ["doc:caveats_gaia_rrlyrae"],
            [["absolute magnitude", "0.64", "reddening", "colour", "color"]],
            [],
            "Fixed M_G = 0.64 with a colour-based reddening estimate "
            "(Muraveva et al. 2018).",
        ),
        (
            "gcns",
            "How were the GCNS distances obtained?",
            ["doc:caveats_gaia_gcns"],
            [["parallax"]],
            [],
            "Gaia EDR3 parallax inversion within ~100 pc.",
        ),
        (
            "luminous",
            "How were distances found for the luminous Gaia DR3 stars?",
            ["doc:caveats_gaia_dr3_luminous"],
            [["bailer-jones", "geometric", "r_med_geo"]],
            [],
            "Bailer-Jones et al. 2021 geometric distances from EDR3 "
            "parallaxes.",
        ),
        (
            "apogee",
            "How were the APOGEE distances computed?",
            ["doc:caveats_apogee"],
            [["starhorse", "spectra", "spectroscopic", "infrared"]],
            [],
            "StarHorse (Queiroz et al. 2023).",
        ),
    ]
    for key, question, evidence, any_groups, not_mention, note in methods:
        cases.append(
            _case(
                id=f"method_{key}",
                kind="method",
                question=question,
                expected_evidence=evidence,
                must_mention_any=any_groups,
                must_not_mention=not_mention,
                note=note,
            )
        )

    # --- caveats and completeness
    caveats = [
        (
            "gcns_completeness",
            "Is the GCNS nearby-star sample in this catalogue complete?",
            ["doc:caveats_gaia_gcns"],
            [["magnitude-limited", "not volume-complete", "brightest"]],
            "Only the 5,000 brightest GCNS stars were kept.",
        ),
        (
            "gcns_boundary",
            "Do all GCNS stars lie within 100 parsecs?",
            ["doc:caveats_gaia_gcns"],
            [["nominal", "beyond", "some", "250", "slightly"]],
            "About 250 of the 5,000 sit slightly beyond 100 pc; selection "
            "was on parallax, which carries uncertainty.",
        ),
        (
            "extinction",
            "Are the distances in this catalogue corrected for interstellar "
            "extinction?",
            [],
            [["no", "not corrected", "without", "none", "wesenheit"]],
            "Largely not, and it varies by subset: no correction for the "
            "luminous sample, the Cepheid Wesenheit index is extinction-free "
            "by construction, RR Lyrae use a colour-based estimate.",
        ),
        (
            "luminosity_quality",
            "How accurate are the luminosity values in this catalogue?",
            ["doc:caveats_field_lum_lsun"],
            [["order-of-magnitude", "order of magnitude", "crude", "derived"]],
            "Derived from distance with a crude bolometric correction and no "
            "extinction correction; order-of-magnitude only.",
        ),
        (
            "colour_dereddening",
            "Are the BP-RP colours corrected for reddening?",
            ["doc:caveats_field_color_bp_rp"],
            [["not", "no"]],
            "Not dereddened, so dust-reddened stars look cooler than they are.",
        ),
        (
            "metallicity_caveat",
            "Does metallicity affect the reliability of the RR Lyrae distances?",
            ["doc:caveats_gaia_rrlyrae"],
            [["metallicity", "ignores", "dependence"]],
            "The method assumes a single absolute magnitude and ignores the "
            "known metallicity dependence. Answerable from the documents even "
            "though no metallicity COLUMN exists -- contrast with "
            "`absent_metallicity_value`.",
        ),
    ]
    for key, question, evidence, any_groups, note in caveats:
        cases.append(
            _case(
                id=f"caveat_{key}",
                kind="method",
                question=question,
                expected_evidence=evidence,
                must_mention_any=any_groups,
                note=note,
            )
        )

    # --- selection criteria, recoverable only from the extracted ADQL
    selections = [
        (
            "parallax_cut",
            "What parallax_over_error cut was applied when selecting the "
            "luminous Gaia DR3 stars?",
            ["doc:selection_gaia_far"],
            [5.0],
            [],
            "add_gaia_far.py filters parallax_over_error > 5.",
        ),
        (
            "cepheid_table",
            "Which Gaia DR3 archive table were the Cepheids selected from?",
            ["doc:selection_gaia_variables"],
            [],
            [["vari_cepheid"]],
            "gaiadr3.vari_cepheid, with type_best_classification = 'DCEP'.",
        ),
        (
            "cepheid_classification",
            "What variability classification was required for the Cepheids?",
            ["doc:selection_gaia_variables"],
            [],
            [["dcep", "type_best_classification"]],
            "type_best_classification = 'DCEP', period > 1.5 d.",
        ),
        (
            "distance_range",
            "What distance range was the luminous Gaia DR3 sample restricted "
            "to?",
            ["doc:selection_gaia_far"],
            [1000.0, 5000.0],
            [],
            "r_med_geo BETWEEN 1000 AND 5000 pc.",
        ),
        (
            "gcns_source",
            "Which VizieR catalogue was the GCNS data downloaded from?",
            ["doc:selection_gcns"],
            [],
            [["j/a+a/649/a6", "649", "vizier"]],
            "VizieR J/A+A/649/A6 (Smart et al. 2021).",
        ),
    ]
    for key, question, evidence, numbers, any_groups, note in selections:
        cases.append(
            _case(
                id=f"selection_{key}",
                kind="selection",
                question=question,
                expected_evidence=evidence,
                must_contain_numbers=numbers,
                must_mention_any=any_groups,
                note=note
                + " Recoverable only from the ADQL extracted from the tool "
                "source, not from the prose docstrings.",
            )
        )

    # --- cross-subset superlatives: cannot come from one filtered query
    cases.append(
        _case(
            id="farthest_gaia_overall",
            kind="numeric",
            question=(
                "What is the most distant Gaia star in the catalogue, and how "
                "far away is it in parsecs?"
            ),
            expected_evidence=[f"row:{gaia_far[0]}:{gaia_far[1]}"],
            must_contain_numbers=[round(float(gaia_far[2]), 1)],
            note=(
                f"Across all Gaia subsets: {float(gaia_far[2]):.1f} pc, a "
                f"{gaia_far[0]} row. The planner once narrowed this to the "
                "luminous subset unprompted and answered 5,000 pc -- a "
                "plausible-looking number for a different question."
            ),
        )
    )
    cases.append(
        _case(
            id="nearest_gaia_overall",
            kind="numeric",
            question=(
                "Which Gaia star in the catalogue is closest to the Sun, in "
                "parsecs?"
            ),
            expected_evidence=[f"row:{gaia_near[0]}:{gaia_near[1]}"],
            must_contain_numbers=[round(float(gaia_near[2]), 2)],
            note=(
                f"Across all Gaia subsets: {float(gaia_near[2]):.2f} pc, a "
                f"{gaia_near[0]} row."
            ),
        )
    )

    # --- ambiguous scope: the answer must state what it counted
    gaia_total = int(one("SELECT count(*) FROM stars WHERE is_gaia")[0])
    cases.append(
        _case(
            id="ambiguous_scope_count",
            kind="scope",
            question="How many stars are in the catalogue?",
            expected_evidence=["agg:count"],
            must_contain_numbers=[float(gaia_total)],
            note=(
                f"Names no subset, so the scope is all Gaia rows: "
                f"{gaia_total}. This is the user-facing side of the rule that "
                "an aggregate's scope comes from the question and not from the "
                "model -- the planner must not narrow it unprompted, and the "
                "answer should make the scope it used visible."
            ),
        )
    )

    # --- unit conversion
    cases.append(
        _case(
            id="gcns_nearest_in_ly",
            kind="unit_conversion",
            question="How far away is the nearest GCNS star in light-years?",
            expected_evidence=[f"row:gaia_gcns:{gcns_id}"],
            must_contain_numbers=[round(gcns_min_ly, 2)],
            note=(
                f"{gcns_min_pc:.2f} pc = {gcns_min_ly:.2f} ly. Requiring the "
                "light-year figure is the check; the parsec value is not "
                "forbidden, since quoting both is correct."
            ),
        )
    )

    return cases


#: Default size of the written gold set. A full sweep costs roughly
#: 85s/case for this pipeline and several minutes per case for RAGAS, so
#: the set is capped and trimmed by `trim_to` rather than grown without
#: limit. Raise it with --target when a longer run is affordable.
DEFAULT_TARGET = 50


def trim_to(cases: list[dict], target: int) -> list[dict]:
    """Downsample while keeping every kind, curated cases first.

    Truncating the list would be worse than useless: it is ordered
    curated-then-generated and grouped by subset, so a prefix would be
    almost entirely one kind and the trimmed set would silently stop
    testing whole failure modes.

    Round-robining over (source, kind) groups keeps at least one case of
    every kind, and visits curated groups first so the adversarial cases
    survive a tight budget ahead of the generated plumbing checks.
    """
    if target >= len(cases):
        return cases

    groups: dict[tuple[str, str], list[dict]] = {}
    for case in cases:
        groups.setdefault(
            (case.get("source", "curated"), case.get("kind", "unspecified")), []
        ).append(case)

    order = sorted(groups, key=lambda k: (k[0] != "curated", k[1]))
    kept: list[dict] = []
    while len(kept) < target:
        progressed = False
        for key in order:
            if groups[key] and len(kept) < target:
                kept.append(groups[key].pop(0))
                progressed = True
        if not progressed:
            break
    # Restore the original ordering so runs stay comparable.
    keep_ids = {c["id"] for c in kept}
    return [c for c in cases if c["id"] in keep_ids]


def build(target: int = DEFAULT_TARGET) -> dict:
    con = duckdb.connect(str(CONFIG.paths.duckdb), read_only=True)
    try:
        cases = curated(con) + generated(con)
    finally:
        con.close()

    full_total = len(cases)
    cases = trim_to(cases, target)

    ids = [c["id"] for c in cases]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise RuntimeError(f"duplicate case ids: {sorted(duplicates)}")

    OUT.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n",
        encoding="utf-8",
    )

    from collections import Counter

    return {
        "total": len(cases),
        "available": full_total,
        "by_source": dict(Counter(c["source"] for c in cases)),
        "by_kind": dict(Counter(c["kind"] for c in cases)),
        "should_answer": sum(c["should_answer"] for c in cases),
        "should_abstain": sum(not c["should_answer"] for c in cases),
        "path": str(OUT),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the gold set.")
    parser.add_argument(
        "--target",
        type=int,
        default=DEFAULT_TARGET,
        help=f"cases to keep (default {DEFAULT_TARGET}; all shapes are "
        "generated, then downsampled keeping every kind)",
    )
    stats = build(parser.parse_args().target)
    print(f"cases          : {stats['total']} of {stats['available']} available")
    print(f"  curated      : {stats['by_source'].get('curated', 0)}")
    print(f"  generated    : {stats['by_source'].get('generated', 0)}")
    print(f"should answer  : {stats['should_answer']}")
    print(f"should abstain : {stats['should_abstain']}")
    print("by kind:")
    for kind, n in sorted(stats["by_kind"].items(), key=lambda kv: -kv[1]):
        print(f"  {kind:26s} {n}")
    print(f"written: {stats['path']}")
