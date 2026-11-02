"""Deterministic numeric grounding.

This is the most useful check in the system and it uses no model at all.

An LLM judge reads "Betelgeuse lies 168 pc away" against evidence saying
"548 pc" and often calls it supported, because the sentences are
semantically near-identical -- the judge is scoring paraphrase, and a
swapped number is an excellent paraphrase. Embedding similarity has the
same blind spot. So numbers are checked arithmetically instead: every
numeric literal in the answer must correspond to a value that actually
appears in the evidence, under some legitimate unit rendering.

Three refinements keep this from being either uselessly strict or
quietly permissive:

* Numbers echoed from the question are exempt. If someone asks "which
  Cepheids are within 3 kpc", the 3 in the answer is theirs, not a
  claim.
* Small integers are tracked but not penalised, because they are nearly
  always prose ("the two subsets", "all five fields") rather than data.
* A number carrying a unit is always data, however small. That one is
  not hypothetical: "RR Lyrae distances are reliable to 1%" passed this
  check because the small-integer exemption swallowed the 1 -- and that
  single figure was the whole substance of the answer, and it was wrong.
  1% is the GCNS parallax precision; RR Lyrae distances are good to
  about 18%.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..config import CONFIG
from ..retrieve.evidence import Evidence

# Matches 1,234.5 / 1234.5 / 1.2e3 / -0.64 / 548
_NUMBER = re.compile(
    r"(?<!\d)[-+]?\d+(?:\.\d+)?\s*[×x]\s*10\s*(?:\^|\*\*)\s*[-+]?\d+"
    r"|(?<!\d)[-+]?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?"
    r"|\d+\.\d+(?:[eE][-+]?\d+)?"
    r"|\d+(?:[eE][-+]?\d+)?)"
)

#: Units that may directly follow a number.
#:
#: The word-boundary anchors apply only to the alphabetic alternatives.
#: "%" is punctuation, and requiring a boundary after it never matches:
#: the next character is usually a space, and two non-word characters
#: have no boundary between them. An earlier version anchored the whole
#: group and silently matched nothing at all.
_UNIT_SUFFIX = re.compile(
    r"\s*(?:%|percent\b|parsecs?\b|kiloparsecs?\b|light[- ]years?\b"
    r"|pc\b|kpc\b|ly\b|kly\b|au\b|mas\b|mag\b|deg\b"
    r"|degrees?\b|l_?sun\b|solar\b|arcsec\b)",
    re.IGNORECASE,
)

#: Integers at or below this are treated as prose, not data.
SMALL_INT_CEILING = 12

#: Publication years and similar bibliographic numbers. These appear in
#: citations throughout the method docs, and matching them against
#: catalogue values would be meaningless either way.
_YEAR_RANGE = (1850, 2100)


def _scan(text: str) -> list[tuple[float, str, str | None]]:
    """Return (value, literal, unit) for every number; unit if adjacent."""
    out: list[tuple[float, str, str | None]] = []
    text = text or ""
    for match in _NUMBER.finditer(text):
        literal = match.group(0)
        try:
            if re.search(r"[×x]", literal):
                coefficient, exponent = re.split(
                    r"\s*[×x]\s*10\s*(?:\^|\*\*)\s*", literal
                )
                value = float(coefficient) * 10 ** int(exponent)
            else:
                value = float(literal.replace(",", ""))
        except ValueError:
            continue
        unit_match = _UNIT_SUFFIX.match(text, match.end())
        unit = unit_match.group(0).strip() if unit_match else None
        out.append((value, literal, unit))
    return out


def extract_numbers(text: str) -> list[tuple[float, str]]:
    """Return (value, literal) for every number in `text`."""
    return [(value, literal) for value, literal, _unit in _scan(text)]


def _is_year(value: float) -> bool:
    return float(value).is_integer() and _YEAR_RANGE[0] <= value <= _YEAR_RANGE[1]


def _is_small_int(value: float) -> bool:
    return float(value).is_integer() and abs(value) <= SMALL_INT_CEILING


@dataclass
class NumberCheck:
    literal: str
    value: float
    matched: bool
    matched_to: str | None = None  # "<evidence_id>:<field> (<unit>)"
    category: str = "data"  # data | from_question | small_int | year
    unit: str | None = None
    closest: float | None = None
    closest_source: str | None = None

    @property
    def counts_against(self) -> bool:
        return self.category == "data" and not self.matched

    def describe(self) -> str:
        shown = f"{self.literal}{self.unit or ''}"
        if self.closest is not None:
            return (
                f"{shown} (closest in evidence: {self.closest:,.4g} "
                f"from {self.closest_source})"
            )
        return f"{shown} (nothing comparable in evidence)"


@dataclass
class NumericReport:
    checks: list[NumberCheck] = field(default_factory=list)
    allowed_count: int = 0

    @property
    def data_checks(self) -> list[NumberCheck]:
        return [c for c in self.checks if c.category == "data"]

    @property
    def unmatched(self) -> list[NumberCheck]:
        return [c for c in self.checks if c.counts_against]

    @property
    def score(self) -> float:
        """Fraction of data-bearing numbers that trace to evidence.

        An answer with no numbers scores 1.0: there is nothing to get
        wrong, and penalising it would push the system toward padding
        answers with numbers.
        """
        data = self.data_checks
        if not data:
            return 1.0
        return sum(1 for c in data if c.matched) / len(data)

    @property
    def passed(self) -> bool:
        return not self.unmatched

    def summary(self) -> str:
        if not self.data_checks:
            return "no data-bearing numbers in the answer"
        if self.passed:
            return f"all {len(self.data_checks)} numbers trace to evidence"
        bad = ", ".join(c.describe() for c in self.unmatched)
        return f"{len(self.unmatched)} ungrounded number(s): {bad}"


#: Spellings that mean the same unit, so a correct restatement is not
#: flagged. Anything not listed compares by its own lowercased name.
_UNIT_ALIASES = {
    "parsec": "pc",
    "parsecs": "pc",
    "kiloparsec": "kpc",
    "kiloparsecs": "kpc",
    "light-year": "ly",
    "light-years": "ly",
    "lightyear": "ly",
    "lightyears": "ly",
    "light year": "ly",
    "light years": "ly",
    "percent": "%",
    "degree": "deg",
    "degrees": "deg",
    "lsun": "l_sun",
    "solar": "l_sun",
}


def _norm_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    key = unit.strip().lower()
    return _UNIT_ALIASES.get(key, key)


def _allowed_values(evidence: list[Evidence]) -> list[tuple[float, str, str | None]]:
    """Every number the answer may state, with its origin and its unit.

    Two sources: the typed numeric facts attached to catalogue rows
    (including all their unit renderings), and every numeric literal
    appearing in retrieved document text -- because a relation like
    "M_G = 0.64" quoted from a method document is properly grounded.

    The unit travels with the value because magnitude alone cannot
    distinguish a correct restatement from the most likely error in this
    domain. 24,952 pc is also 81,383 ly; an answer saying "24,952 ly" is
    wrong by a factor of 3.26 while stating a number that really does
    appear in the evidence.
    """
    allowed: list[tuple[float, str, str | None]] = []
    for ev in evidence:
        for fact in ev.numeric_facts:
            for unit, value in fact.renderings.items():
                allowed.append(
                    (
                        float(value),
                        f"{ev.evidence_id}:{fact.field_name} ({unit})",
                        _norm_unit(unit),
                    )
                )
        for value, _literal, unit in _scan(ev.text):
            allowed.append((value, f"{ev.evidence_id}:text", _norm_unit(unit)))
    return allowed


def _matches(answer_value: float, evidence_value: float) -> bool:
    rel_tol = CONFIG.verification.numeric_rel_tol
    abs_tol = CONFIG.verification.numeric_abs_tol
    if abs(answer_value - evidence_value) <= abs_tol:
        return True
    scale = max(abs(answer_value), abs(evidence_value))
    if scale == 0:
        return True
    return abs(answer_value - evidence_value) / scale <= rel_tol


def verify_numbers(
    answer_text: str, evidence: list[Evidence], question: str = ""
) -> NumericReport:
    allowed = _allowed_values(evidence)
    question_values = {v for v, _ in extract_numbers(question)}

    report = NumericReport(allowed_count=len(allowed))
    for value, literal, unit in _scan(answer_text):
        if any(_matches(value, qv) for qv in question_values):
            category = "from_question"
        elif unit is not None:
            # Carries a unit, so it is a measurement no matter how small.
            category = "data"
        elif _is_small_int(value):
            category = "small_int"
        elif _is_year(value):
            category = "year"
        else:
            category = "data"

        answer_unit = _norm_unit(unit)
        matched_to = None
        closest = None
        closest_source = None
        best_gap = float("inf")
        for ev_value, source, ev_unit in allowed:
            # A unit on the answer's number must agree with the unit of
            # the evidence value it claims to restate. An evidence number
            # with no unit attached (a bare figure in prose) is accepted
            # either way, since there is nothing to disagree with.
            unit_ok = (
                answer_unit is None or ev_unit is None or answer_unit == ev_unit
            )
            if unit_ok and _matches(value, ev_value):
                matched_to = source
                break
            gap = abs(value - ev_value)
            if gap < best_gap:
                best_gap, closest, closest_source = gap, ev_value, source

        report.checks.append(
            NumberCheck(
                literal=literal,
                value=value,
                matched=matched_to is not None,
                matched_to=matched_to,
                category=category,
                unit=unit,
                closest=closest if matched_to is None else None,
                closest_source=closest_source if matched_to is None else None,
            )
        )
    return report
