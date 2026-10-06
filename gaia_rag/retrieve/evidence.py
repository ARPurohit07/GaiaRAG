"""The evidence model shared by both retrievers.

Every piece of retrieved context becomes an `Evidence` with a stable id,
and that id is what the generator must cite and what the verifier checks
against. Mixing "a chunk of prose" and "a database row" into one type is
what lets a single faithfulness check cover both halves of a hybrid RAG.

`NumericFact` is the part that makes numeric grounding possible. A row
does not just carry the number 1300.4; it carries the knowledge that
this is a distance in parsecs, that stating it as "4,240 ly" or
"1.3 kpc" is the same fact, and that it was derived by a method with a
10% uncertainty. Without that, a verifier either rejects correct unit
conversions or accepts invented numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..units import distance_renderings, format_distance

EvidenceKind = Literal["row", "doc"]


@dataclass
class NumericFact:
    field_name: str
    value: float
    unit: str
    renderings: dict[str, float] = field(default_factory=dict)
    status: str = "unknown"  # measured | derived | catalogued
    rel_uncertainty: float | None = None

    def __post_init__(self) -> None:
        if not self.renderings:
            self.renderings = {self.unit: self.value}


@dataclass
class Evidence:
    evidence_id: str
    kind: EvidenceKind
    text: str
    citation: str
    score: float = 0.0
    numeric_facts: list[NumericFact] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def render(self) -> str:
        """How this evidence appears in the generator's prompt."""
        return f"[{self.evidence_id}] {self.text}\n    source: {self.citation}"


# --- row -> evidence --------------------------------------------------

#: Catalogue field -> (unit, human label). Only fields that exist in the
#: table are listed; nothing is synthesised.
ROW_NUMERIC_FIELDS = {
    "dist_pc": ("pc", "distance"),
    "ra_deg": ("deg", "right ascension"),
    "dec_deg": ("deg", "declination"),
    "color_bp_rp": ("mag", "BP-RP colour index"),
    "lum_lsun": ("Lsun", "luminosity"),
}


def row_to_evidence(
    row: dict,
    provenance: dict[tuple[str, str], dict] | None = None,
) -> Evidence:
    """Turn a `stars` row into evidence, attaching per-field provenance."""
    provenance = provenance or {}
    subset = row["subset"]
    name = row.get("proper_name") or row.get("label") or f"row {row.get('row_id')}"

    facts: list[NumericFact] = []
    for field_name, (unit, _label) in ROW_NUMERIC_FIELDS.items():
        value = row.get(field_name)
        if value is None:
            continue
        prov = provenance.get((subset, field_name), {})
        renderings = (
            distance_renderings(float(value))
            if field_name == "dist_pc"
            else {unit: float(value)}
        )
        facts.append(
            NumericFact(
                field_name=field_name,
                value=float(value),
                unit=unit,
                renderings=renderings,
                status=prov.get("status", "unknown"),
                rel_uncertainty=prov.get("rel_uncertainty"),
            )
        )

    parts = [f"{name} (catalogue subset: {subset})."]
    if row.get("spectral_type"):
        parts.append(f"Spectral type {row['spectral_type']}.")
    if row.get("constellation"):
        parts.append(f"Constellation {row['constellation']}.")
    if row.get("dist_pc") is not None:
        dist_prov = provenance.get((subset, "dist_pc"), {})
        sentence = f"Distance {format_distance(float(row['dist_pc']))}"
        if dist_prov.get("status"):
            sentence += f", {dist_prov['status']}"
        if dist_prov.get("rel_uncertainty") is not None:
            sentence += (
                f", typical relative uncertainty "
                f"{dist_prov['rel_uncertainty'] * 100:.0f}%"
            )
        parts.append(sentence + ".")
        if dist_prov.get("method"):
            parts.append(f"Distance method: {dist_prov['method']}")
    if row.get("ra_deg") is not None and row.get("dec_deg") is not None:
        parts.append(
            f"Position RA {float(row['ra_deg']):.5f} deg, "
            f"Dec {float(row['dec_deg']):.5f} deg (ICRS)."
        )
    if row.get("color_bp_rp") is not None:
        parts.append(f"BP-RP colour index {float(row['color_bp_rp']):.3f}.")
    if row.get("lum_lsun") is not None:
        parts.append(
            # Plain decimal, not `:,.3g`. That format rendered 32910 as
            # "3.29e+04", the model faithfully restated it as
            # "3.29 x 10^4", and the numeric verifier -- which reads
            # digits, not notation -- could not match it to the required
            # 32910 and reported a correct answer as ungrounded. Evidence
            # should state values in the form an answer can quote and a
            # checker can read, and three significant figures also threw
            # away precision for no benefit.
            f"Luminosity {float(row['lum_lsun']):,.1f} times solar "
            "(derived, order-of-magnitude)."
        )
    if not row.get("is_gaia", False):
        parts.append(
            "NOTE: this row is NOT from Gaia; do not attribute it to Gaia."
        )

    return Evidence(
        evidence_id=row["evidence_id"],
        kind="row",
        text=" ".join(parts),
        citation=(
            f"Galaxy web catalogue row, subset {subset}"
            + (f", source_id {row['source_id']}" if row.get("source_id") else "")
        ),
        numeric_facts=facts,
        metadata={
            "subset": subset,
            "is_gaia": bool(row.get("is_gaia", False)),
            "proper_name": row.get("proper_name"),
            "label": row.get("label"),
        },
    )


def chunk_to_evidence(chunk, score: float) -> Evidence:
    return Evidence(
        evidence_id=chunk.chunk_id,
        kind="doc",
        text=f"{chunk.title}: {chunk.text}",
        citation=f"{chunk.citation} (via {chunk.source_path})",
        score=score,
        metadata={"doc_id": chunk.doc_id, "kind": chunk.kind},
    )
