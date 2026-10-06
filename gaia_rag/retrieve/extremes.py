"""Work out which field a superlative question is about.

Without this, every `extreme` query was ordered by distance whatever it
asked about, and the two consequences were not symmetrical:

* "largest BP-RP colour index among the classical Cepheids" retrieved
  the ten most **distant** Cepheids and reported the largest colour among
  those ten -- 0.880 instead of the true 1.05. That answer passed every
  verification gate, because 0.880 really does appear in the evidence.
  The evidence was simply the wrong rows.
* "most luminous RR Lyrae" was ordered by distance too, but there the
  claim "X is the most luminous" was correctly judged unsupported, so the
  answer was withheld rather than served.

One silent error and one honest refusal from a single careless default.
A query that returns plausible rows for a question it does not answer is
exactly the failure the QuerySpec design exists to prevent, and a
default ordering reintroduced it.
"""

from __future__ import annotations

import re

#: (pattern, field, direction), most specific first; first match wins.
EXTREME_PATTERNS: tuple[tuple[str, str, str], ...] = (
    (r"\bblu(?:est|er)\b|\bsmallest\s+(?:bp-?rp|colou?r)", "color_bp_rp", "asc"),
    (
        r"\bredd(?:est|er)\b"
        r"|\b(?:largest|highest|greatest|max(?:imum)?)\s+(?:bp-?rp|colou?r)"
        r"|\bbp-?rp\b",
        "color_bp_rp",
        "desc",
    ),
    (r"\b(?:least|lowest)\s+lumin|\bfaintest\b|\bdimmest\b", "lum_lsun", "asc"),
    (
        r"\bmost\s+lumin"
        r"|\b(?:highest|greatest|largest|max(?:imum)?)\s+lumin"
        r"|\bbrightest\b",
        "lum_lsun",
        "desc",
    ),
    (r"\b(?:closest|nearest|least\s+distant)\b", "dist_pc", "asc"),
    (
        r"\b(?:farthest|furthest|most\s+distant|greatest\s+distance)\b",
        "dist_pc",
        "desc",
    ),
)

#: Used when a question gives no usable signal. Distance is the right
#: fallback because it is the field every subset actually varies in.
DEFAULT_ORDER = {"field": "dist_pc", "dir": "desc"}


def infer_extreme_order(question: str) -> dict | None:
    """The ordering a superlative question implies, or None if unclear."""
    lowered = (question or "").lower()
    for pattern, field, direction in EXTREME_PATTERNS:
        if re.search(pattern, lowered):
            return {"field": field, "dir": direction}
    return None
