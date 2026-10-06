"""Unit handling for astronomical quantities.

The catalogue stores distances in parsecs; people ask in light-years,
and papers quote kiloparsecs. A model that silently swaps them produces
an answer that is wrong by 3.26x while looking perfectly confident, and
no text-similarity metric will catch it. So conversions live here, are
tested, and the numeric verifier uses this same table to decide whether
a number in an answer is a legitimate restatement of the evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

LY_PER_PC = 3.2615637769
AU_PER_PC = 206264.806


def pc_to_ly(pc: float) -> float:
    return pc * LY_PER_PC


def ly_to_pc(ly: float) -> float:
    return ly / LY_PER_PC


def pc_to_kpc(pc: float) -> float:
    return pc / 1000.0


def parallax_mas_to_pc(parallax_mas: float) -> float:
    """Distance from parallax. Only valid for a well-measured parallax.

    Inverting a noisy parallax is biased, which is exactly why the Gaia
    rows in this corpus use Bailer-Jones geometric distances instead.
    """
    if parallax_mas <= 0:
        raise ValueError("non-positive parallax has no distance by inversion")
    return 1000.0 / parallax_mas


def absolute_magnitude(apparent_mag: float, dist_pc: float) -> float:
    """M = m - 5 log10(d/10), no extinction correction."""
    if dist_pc <= 0:
        raise ValueError("distance must be positive")
    import math

    return apparent_mag - 5.0 * math.log10(dist_pc / 10.0)


@dataclass(frozen=True)
class Quantity:
    value: float
    unit: str


#: Every alternative rendering of a stored value that a correct answer is
#: allowed to state. The verifier accepts a number in the answer if it
#: matches the stored value under any of these, and rejects it otherwise.
DISTANCE_ALIASES = {
    "pc": lambda v: v,
    "ly": pc_to_ly,
    "kpc": pc_to_kpc,
    "kly": lambda v: pc_to_ly(v) / 1000.0,
    "au": lambda v: v * AU_PER_PC,
}


def distance_renderings(dist_pc: float) -> dict[str, float]:
    """All legitimate ways to state one stored distance."""
    return {unit: fn(dist_pc) for unit, fn in DISTANCE_ALIASES.items()}


def format_distance(dist_pc: float) -> str:
    """Human phrasing used when rendering evidence into a prompt."""
    ly = pc_to_ly(dist_pc)
    if dist_pc >= 1000:
        return f"{dist_pc:,.0f} pc ({pc_to_kpc(dist_pc):.2f} kpc; {ly:,.0f} ly)"
    return f"{dist_pc:,.2f} pc ({ly:,.1f} ly)"
