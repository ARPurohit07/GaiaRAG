"""Subject consistency: does a claim cite evidence about its own subject?

This catches citation drift, which an entailment judge reliably misses.
The observed failure: a claim asserting "RR Lyrae distances are reliable
to 1%" cited both the RR Lyrae caveat document and the GCNS caveat
document. The GCNS document does say 1%, so the judge found the number
in the cited evidence and returned "supported" -- and the answer was
wrong by a factor of eighteen.

No amount of judge prompting fixes that robustly, because the judge is
being asked to notice that the right number came from the wrong row. But
the subject is recoverable from surface text: "RR Lyrae" names a subset,
the cited document declares a subset, and the two can simply be compared.

The same keyword map feeds query planning, so a question naming RR Lyrae
filters the catalogue to RR Lyrae rows rather than relying on the model
to remember the mapping -- which, measured, it did not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .numeric import _matches as _close
from .numeric import extract_numbers

if TYPE_CHECKING:
    from ..retrieve.evidence import Evidence

#: Surface terms that identify a catalogue subset. Order within a subset
#: does not matter; across subsets, more specific phrases must not be
#: substrings of a less specific one (checked by tests).
SUBJECT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "gaia_cepheid": ("cepheid", "dcep", "classical cepheid"),
    "gaia_rrlyrae": ("rr lyrae", "rrlyrae", "rr-lyrae"),
    "gaia_gcns": ("gcns", "catalogue of nearby stars", "catalog of nearby stars"),
    "gaia_dr3_luminous": (
        # "luminous" must be qualified, not bare.
        #
        # It began as the contiguous phrase "luminous star", which missed
        # the natural "luminous Gaia DR3 stars" and silently widened five
        # questions to all of Gaia. Replacing it with a bare "luminous"
        # fixed that but over-matched: "which RR Lyrae is the most
        # luminous?" then named this subset too, scoping an RR Lyrae
        # question to RR Lyrae *and* the luminous sample.
        #
        # Requiring a following dataset word keeps the subset reading and
        # drops the plain adjective. `\s+\w*\s*` allows the intervening
        # "Gaia DR3".
        r"luminous\s+(?:\w+\s+){0,2}?(?:gaia|dr3|star|stars|sample|subset)",
        r"gaia\s+dr3\s+luminous",
        "ob star",
        "bright giant",
        "supergiant",
        "bailer-jones",
    ),
    "apogee": ("apogee", "starhorse", "red giant"),
    "hyg": ("hyg", "hipparcos", "gliese", "yale bright star"),
}


def detect_subsets(text: str) -> set[str]:
    """Subsets named by surface terms in `text`.

    Plurals are matched explicitly. Omitting that was a real and costly
    bug: "How many classical Cepheids are in the catalogue?" matched no
    keyword at all, because the list holds "cepheid" and the question
    says "Cepheids". Detection came back empty, the caller took that to
    mean the question named no subset, and the count silently widened
    from 2,187 Cepheids to all 120,000 Gaia stars -- an answer that was
    fully grounded, confidently worded, and wrong by a factor of 55.

    This is the same plural-matching gap already fixed in the BM25
    tokeniser; it simply had a second home here.
    """
    lowered = (text or "").lower()
    found: set[str] = set()
    for subset, keywords in SUBJECT_KEYWORDS.items():
        for keyword in keywords:
            # A keyword containing regex metacharacters is treated as a
            # pattern; a plain phrase is escaped and matched literally.
            # Either way it is bounded, so "apogee" cannot fire inside a
            # longer token, and a regular plural suffix is allowed.
            if re.search(r"[\\(?]", keyword):
                pattern = rf"(?<![a-z0-9])(?:{keyword})(?:e?s)?(?![a-z0-9])"
            else:
                pattern = rf"(?<![a-z0-9]){re.escape(keyword)}(?:e?s)?(?![a-z0-9])"
            if re.search(pattern, lowered):
                found.add(subset)
                break
    return found


def evidence_subsets(evidence) -> set[str] | None:
    """Subsets an evidence item is about, or None if it is general.

    Rows declare their subset in metadata. Document chunks declare theirs
    through the doc id, which is derived from the subset at ingest time.

    None means "not specific to any subset" -- shared documents such as
    the field-level caveats or the overall provenance notes. Those can
    never be a subject mismatch, and None distinguishes them from an item
    that is specifically about some other survey.
    """
    found: set[str] = set()
    subset = (evidence.metadata or {}).get("subset")
    if subset:
        found.add(subset)
    doc_id = (evidence.metadata or {}).get("doc_id") or ""
    for candidate in SUBJECT_KEYWORDS:
        if candidate in doc_id:
            found.add(candidate)
    return found or None


@dataclass
class SubjectIssue:
    claim: str
    claim_subsets: set[str] = field(default_factory=set)
    cited_subsets: set[str] = field(default_factory=set)
    offending_ids: list[str] = field(default_factory=list)

    def describe(self) -> str:
        about = ", ".join(sorted(self.claim_subsets))
        cited = ", ".join(sorted(self.cited_subsets))
        where = f" via {', '.join(self.offending_ids)}" if self.offending_ids else ""
        return (
            f"claim is about {about} but cites evidence about {cited}{where}: "
            f"{self.claim}"
        )


@dataclass
class SubjectReport:
    #: Hard failures: a NUMBER in the claim is only available from
    #: evidence about a different subset.
    issues: list[SubjectIssue] = field(default_factory=list)
    #: Soft failures: a citation to another subset that contributes no
    #: number. Surfaced, not gated.
    warnings: list[SubjectIssue] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.issues

    def summary(self) -> str:
        parts = []
        if self.issues:
            parts.append(
                f"{len(self.issues)} claim(s) take a number from another subset"
            )
        if self.warnings:
            parts.append(
                f"{len(self.warnings)} claim(s) cite another subset without "
                "using its numbers"
            )
        return (
            ", ".join(parts)
            if parts
            else "every claim's numbers come from its own subset"
        )


def check_claims(claims: list[dict], evidence: list["Evidence"]) -> SubjectReport:
    index = {e.evidence_id: e for e in evidence}
    report = SubjectReport()
    for claim in claims:
        text = str(claim.get("text", "")).strip()
        if not text:
            continue
        claim_subsets = detect_subsets(text)
        if not claim_subsets:
            # The claim names no subset, so there is no subject to drift.
            continue
        cited_ids = [str(cid) for cid in (claim.get("evidence_ids") or [])]
        cited = [(cid, index[cid]) for cid in cited_ids if cid in index]
        if not cited:
            continue

        # Which citations are about some *other* subset.
        offending: list[str] = []
        wrong_subsets: set[str] = set()
        matching: list = []
        for cid, ev in cited:
            subsets = evidence_subsets(ev)
            if subsets is None:
                matching.append(ev)  # general document, applies everywhere
            elif subsets & claim_subsets:
                matching.append(ev)
            else:
                offending.append(cid)
                wrong_subsets |= subsets

        if not offending:
            continue

        # A wrong-subject citation alone is not proof of drift, and
        # treating it as such withheld a correct answer: "Vega's distance
        # is from the HYG Database, not a Gaia measurement" cited the
        # right HYG row *and* a GCNS document, and the GCNS citation was
        # merely superfluous.
        #
        # But requiring only that *some* citation match is too weak -- the
        # original failure cited the correct RR Lyrae document AND the
        # GCNS one, and took its number from the GCNS one.
        #
        # What separates them is where the numbers come from. So the hard
        # gate is numeric: every number in the claim must be obtainable
        # from evidence about the claim's own subject. Prose-only drift is
        # reported as a warning, because attribution in prose is a matter
        # of degree and the claim judge already scores it.
        claim_numbers = [v for v, _literal in extract_numbers(text)]
        available = []
        for ev in matching:
            available.extend(v for v, _literal in extract_numbers(ev.text))
            for fact in ev.numeric_facts:
                available.extend(float(x) for x in fact.renderings.values())

        unsourced = [
            n
            for n in claim_numbers
            if not any(_close(n, a) for a in available)
        ]
        issue = SubjectIssue(
            claim=text,
            claim_subsets=claim_subsets,
            cited_subsets=wrong_subsets,
            offending_ids=offending,
        )
        if unsourced:
            report.issues.append(issue)
        else:
            report.warnings.append(issue)
    return report
