"""Schema scope: refuse to put a number on a concept the data lacks.

This exists because of the worst failure the gold set caught. Asked "how
many confirmed exoplanets orbit stars in the Gaia Cepheid sample?", the
pipeline answered "there are 2187 confirmed exoplanets" -- which is the
*number of Cepheid stars*, relabelled. No exoplanet data is ingested at
all.

Every existing check passed, and each was right to:

* numeric grounding: 2187 really is in the evidence;
* subject consistency: the claim is about Cepheids, the evidence is about
  Cepheids;
* the entailment judge: the evidence does state count = 2187.

The defect is not in any of them. It is that `count(*)` over a table of
stars answers "how many stars", and nothing stopped that number being
attached to a different noun. Relabelling an aggregate is invisible to
every check that reasons about values and subsets, because the value and
the subset are both correct.

So the guard is on the question, not the answer: if a question asks for a
quantity the schema has no column for, the row evidence is withdrawn
before generation, leaving nothing to misattribute.

Note what is deliberately *not* blocked: asking how metallicity affects
RR Lyrae distances is a legitimate question about a documented caveat.
Only numeric row evidence is withdrawn, so questions answerable from the
method documents still work.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: Concepts a user may reasonably ask about that the `stars` table has no
#: column for. Each maps to the phrasing used when declining.
#:
#: parallax, period and apparent magnitude are listed because they are
#: genuinely absent: Galaxy's CSV already collapsed them into distance
#: and luminosity. They are the most plausible wrong-confidence questions
#: precisely because they sound like they should be there.
ABSENT_CONCEPTS: dict[str, tuple[str, ...]] = {
    "exoplanets": ("exoplanet", "exoplanets", "planet", "planets", "planetary system"),
    "stellar mass": ("mass", "masses", "solar masses"),
    "stellar radius": ("radius", "radii"),
    "effective temperature": (
        "temperature",
        "temperatures",
        "teff",
        "t_eff",
        "effective temperature",
    ),
    "stellar age": ("age", "ages"),
    "metallicity values": ("[fe/h]", "fe/h"),
    "proper motion": ("proper motion", "pmra", "pmdec"),
    "radial velocity": ("radial velocity", "radial velocities"),
    "parallax": ("parallax", "parallaxes"),
    "pulsation period": ("period", "periods", "pulsation period"),
    "apparent magnitude": (
        "apparent magnitude",
        "g magnitude",
        "phot_g_mean_mag",
        "brightness",
    ),
}

#: Words signalling that the question wants a value or a count, as
#: opposed to an explanation. A question about how a concept *affects*
#: something is answerable from the documents even when the column is
#: absent.
_QUANTITY_CUES = (
    "how many",
    "how much",
    "what is the",
    "what are the",
    "value",
    "count",
    "average",
    "mean",
    "median",
    "total",
    "list",
    "which",
    "highest",
    "lowest",
    "largest",
    "smallest",
)


def _mentions(text: str, phrase: str) -> bool:
    return bool(
        re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text)
    )


@dataclass
class ScopeReport:
    absent: list[str] = field(default_factory=list)
    asks_for_quantity: bool = False

    @property
    def should_withdraw_rows(self) -> bool:
        return bool(self.absent) and self.asks_for_quantity

    def note(self) -> str:
        names = ", ".join(self.absent)
        return (
            f"The catalogue has no column for: {names}. No row or aggregate "
            "evidence was supplied, because a count or statistic over this "
            "table counts STARS and would be wrong if restated as any other "
            "quantity. Say that this data is not available rather than "
            "offering a number."
        )


def check_question(question: str) -> ScopeReport:
    lowered = (question or "").lower()
    report = ScopeReport()
    report.asks_for_quantity = any(cue in lowered for cue in _QUANTITY_CUES)
    for label, phrases in ABSENT_CONCEPTS.items():
        if any(_mentions(lowered, phrase) for phrase in phrases):
            report.absent.append(label)
    return report
