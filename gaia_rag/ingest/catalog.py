"""Build the structured half of the corpus: Gaia rows in DuckDB.

Source is the Galaxy project's web catalogue, which is real Gaia data
already resolved to distances by Galaxy's own tooling. Rows carry a
label in the `gl` column saying which Gaia product they came from, and
that label is the only reliable way to know how a given distance was
derived -- a GCNS parallax distance and an RR Lyrae period-luminosity
distance differ by an order of magnitude in uncertainty.

Two things here exist specifically to keep answers honest:

1. Rows are split by provenance (`subset`), and non-Gaia rows (HYG,
   APOGEE) are kept in the same table but labelled, so the system can
   never quote a Hipparcos distance as a Gaia measurement.
2. `field_provenance` records, per subset and per field, whether a value
   is measured or derived, by what method, with what uncertainty, and
   with what caveat. Retrieval always carries this alongside the number.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb

from ..config import CONFIG

# --- row classification -----------------------------------------------

SUBSET_PREFIXES = {
    "gaia_gcns": "GCNS ",
    "gaia_cepheid": "Gaia Cepheid ",
    "gaia_rrlyrae": "Gaia RR Lyrae ",
    "apogee": "APOGEE ",
}

GAIA_SUBSETS = ("gaia_gcns", "gaia_cepheid", "gaia_rrlyrae", "gaia_dr3_luminous")
ALL_SUBSETS = GAIA_SUBSETS + ("apogee", "hyg")

#: HYG's sentinel for "parallax unusable", not a real distance.
HYG_UNKNOWN_DISTANCE_PC = 100000.0

#: Human-readable names for each subset.
#:
#: These exist because lexical retrieval needs the words people actually
#: use. A document that identifies its subject only as `gaia_rrlyrae`
#: tokenises to "gaia" + "rrlyrae" and never matches a question asking
#: about "RR Lyrae" -- which is exactly how the RR Lyrae caveat document
#: became unreachable from the one question it was written to answer.
SUBSET_LABELS = {
    "gaia_gcns": "Gaia Catalogue of Nearby Stars (GCNS) -- nearby stars within ~100 pc",
    "gaia_cepheid": "Gaia DR3 classical Cepheids (DCEP) -- pulsating variable standard candles",
    "gaia_rrlyrae": "Gaia DR3 RR Lyrae -- old pulsating variables of the bulge, inner Galaxy and halo",
    "gaia_dr3_luminous": "Gaia DR3 luminous stars (OB stars, bright giants and supergiants)",
    "apogee": "APOGEE DR17 red giants with StarHorse distances (infrared spectroscopic survey)",
    "hyg": "HYG database stars (Hipparcos, Yale Bright Star and Gliese catalogues)",
}


def classify(label: str) -> tuple[str, str | None]:
    """Map a catalogue label to (subset, source identifier).

    Order matters: "Gaia Cepheid 123" must not fall through to the plain
    "Gaia " luminous-star bucket, so the specific prefixes are tried
    before the generic one.
    """
    label = (label or "").strip()
    if not label:
        return "hyg", None
    for subset, prefix in SUBSET_PREFIXES.items():
        if label.startswith(prefix):
            return subset, label[len(prefix):].strip()
    if label.startswith("Gaia "):
        return "gaia_dr3_luminous", label[len("Gaia "):].strip()
    # Gliese / other HYG designations.
    return "hyg", label


# --- per-field provenance ---------------------------------------------


@dataclass(frozen=True)
class FieldProvenance:
    subset: str
    field: str
    status: str  # measured | derived | catalogued
    method: str
    rel_uncertainty: float | None
    caveat: str
    source_doc: str


DISTANCE_PROVENANCE = [
    FieldProvenance(
        "gaia_gcns",
        "dist_pc",
        "measured",
        "Gaia EDR3 parallax inversion within 100 pc, from the Gaia "
        "Catalogue of Nearby Stars (Smart et al. 2021, A&A 649, A6).",
        0.01,
        "Reliable: at these distances Gaia parallaxes are precise to well "
        "under 1%. Two selection effects matter: only the 5,000 brightest "
        "GCNS stars were kept, so the subset is magnitude-limited and NOT "
        "volume-complete (it under-represents faint red dwarfs, which are "
        "the most common nearby stars); and the 100 pc boundary is nominal "
        "-- about 250 of the 5,000 rows sit slightly beyond it, because "
        "selection was on parallax, which carries uncertainty.",
        "gcns",
    ),
    FieldProvenance(
        "gaia_dr3_luminous",
        "dist_pc",
        "derived",
        "Bailer-Jones et al. 2021 geometric distance (r_med_geo) from Gaia "
        "EDR3 parallaxes, selected with parallax_over_error > 5.",
        0.2,
        "Bayesian geometric distance, not raw parallax inversion. "
        "Restricted to 1-5 kpc and absolute G < -1, so the subset contains "
        "only intrinsically luminous stars (OB stars, bright giants) and is "
        "not a fair sample of the disk. No extinction correction.",
        "gaia_far",
    ),
    FieldProvenance(
        "gaia_cepheid",
        "dist_pc",
        "derived",
        "Wesenheit period-luminosity relation W = G - 1.90(BP-RP), "
        "M_W = -5.988 - 3.176(log P - 1) (Ripepi et al. 2022, A&A 674, A17), "
        "applied to Gaia DR3 vari_cepheid fundamental-mode periods.",
        0.1,
        "Standard-candle distance, independent of parallax, usable to "
        "~25 kpc. The Wesenheit index is extinction-free by construction. "
        "Only DCEP (classical Cepheid) classifications with period > 1.5 d "
        "are included.",
        "gaia_variables",
    ),
    FieldProvenance(
        "gaia_rrlyrae",
        "dist_pc",
        "derived",
        "Fixed absolute magnitude M_G = 0.64 (Muraveva et al. 2018) with "
        "reddening estimated from colour, assuming intrinsic BP-RP = 0.65 "
        "and A_G = 1.9 E(BP-RP), applied to Gaia DR3 vari_rrlyrae.",
        0.18,
        "Good to roughly 15-20%. Assumes a single absolute magnitude for "
        "all RR Lyrae, ignoring the known metallicity dependence of the "
        "M_G-[Fe/H] relation, and the reddening is a colour-based estimate "
        "rather than a measurement.",
        "gaia_variables",
    ),
    FieldProvenance(
        "apogee",
        "dist_pc",
        "derived",
        "StarHorse distances (Queiroz et al. 2023, A&A 673, A155) combining "
        "APOGEE DR17 spectra with Gaia EDR3 parallaxes and 2MASS/WISE "
        "infrared photometry.",
        0.25,
        "NOT a Gaia-only distance: spectroscopic and photometric data "
        "dominate beyond a few kpc. Selected to distance error < 25%.",
        "apogee",
    ),
    FieldProvenance(
        "hyg",
        "dist_pc",
        "catalogued",
        "HYG Database v4.1 (astronexus), compiled from Hipparcos, the Yale "
        "Bright Star Catalogue and the Gliese catalogue.",
        0.05,
        "NOT Gaia data. Hipparcos astrometry is far less precise than "
        "Gaia's, and these distances must never be attributed to Gaia.",
        "hyg",
    ),
]


def _provenance_rows() -> list[FieldProvenance]:
    """Provenance facts taken from Galaxy's SOURCES.txt and tool scripts.

    These are assertions about how the numbers in the catalogue were
    produced. They are the ground truth the RAG cites when asked how
    reliable a value is, so they are stated explicitly rather than
    inferred at query time.
    """
    rows = list(DISTANCE_PROVENANCE)
    for subset in ALL_SUBSETS:
        is_gaia = subset in GAIA_SUBSETS
        photometry_doc = "gcns" if is_gaia else "hyg"
        rows.append(
            FieldProvenance(
                subset,
                "color_bp_rp",
                "measured" if subset != "hyg" else "catalogued",
                (
                    "Gaia DR3 BP-RP colour index"
                    if subset != "hyg"
                    else "HYG colour index, mixed photometric systems"
                )
                + (
                    " For RR Lyrae this is a CONSTANT 0.35 for every row: "
                    "the assumed intrinsic colour used to estimate "
                    "reddening, not a measured colour."
                    if subset == "gaia_rrlyrae"
                    else ""
                ),
                0.02,
                (
                    "NOT a per-star value: every RR Lyrae row carries the "
                    "same colour, so RR Lyrae cannot be ranked or compared "
                    "by colour."
                    if subset == "gaia_rrlyrae"
                    else "Not corrected for interstellar reddening, so "
                    "dust-reddened stars appear cooler than they really are."
                ),
                photometry_doc,
            )
        )
        # For RR Lyrae both colour and luminosity are single constants
        # across all 67,813 rows (BP-RP = 0.35, L = 45 Lsun), because the
        # distance method assumed a fixed absolute magnitude and a fixed
        # intrinsic colour. They are placeholders implied by the method,
        # not per-star measurements, and any question that ranks or
        # compares RR Lyrae by them has no real answer. Saying so is the
        # difference between a system that declines to rank them and one
        # that invents an ordering over 67,813 identical values.
        degenerate = subset == "gaia_rrlyrae"
        rows.append(
            FieldProvenance(
                subset,
                "lum_lsun",
                "derived",
                "Luminosity in solar units, derived from the distance and "
                "apparent magnitude with a crude bolometric correction."
                + (
                    " For RR Lyrae this is a CONSTANT 45 solar luminosities "
                    "for every row, implied by the assumed absolute "
                    "magnitude M_G = 0.64, not measured per star."
                    if degenerate
                    else ""
                ),
                0.3,
                (
                    "NOT a per-star value: every RR Lyrae row carries the "
                    "same luminosity, so RR Lyrae cannot be ranked or "
                    "compared by it and no single RR Lyrae is 'the most "
                    "luminous'."
                    if degenerate
                    else "Depends on the distance above, so it inherits that "
                    "uncertainty, and no extinction correction is applied. "
                    "Treat as an order-of-magnitude figure only."
                ),
                "gaia_variables" if degenerate else "gaia_far",
            )
        )
        for field, axis in (("ra_deg", "right ascension"), ("dec_deg", "declination")):
            rows.append(
                FieldProvenance(
                    subset,
                    field,
                    "measured" if is_gaia else "catalogued",
                    f"ICRS {axis} in degrees"
                    + (" from Gaia DR3 astrometry" if is_gaia else ""),
                    1e-6,
                    "Positions are the most reliable quantity in the catalogue.",
                    photometry_doc,
                )
            )
    return rows


# --- build ------------------------------------------------------------

SCHEMA = """
CREATE TABLE stars (
    row_id          BIGINT,
    evidence_id     VARCHAR NOT NULL,
    subset          VARCHAR NOT NULL,
    is_gaia         BOOLEAN NOT NULL,
    label           VARCHAR,
    source_id       VARCHAR,
    proper_name     VARCHAR,
    constellation   VARCHAR,
    spectral_type   VARCHAR,
    ra_deg          DOUBLE,
    dec_deg         DOUBLE,
    dist_pc         DOUBLE,
    dist_ly         DOUBLE,
    dist_kpc        DOUBLE,
    color_bp_rp     DOUBLE,
    lum_lsun        DOUBLE
);

CREATE TABLE field_provenance (
    subset           VARCHAR NOT NULL,
    field            VARCHAR NOT NULL,
    status           VARCHAR NOT NULL,
    method           VARCHAR NOT NULL,
    rel_uncertainty  DOUBLE,
    caveat           VARCHAR NOT NULL,
    source_doc       VARCHAR NOT NULL
);
"""

#: The row transform, done entirely in SQL.
#:
#: The classification here must stay equivalent to `classify()` above --
#: tests/test_catalog.py asserts they agree row for row. The duplication
#: is deliberate: SQL loads 300k rows in about a second where a Python
#: loop with executemany takes minutes, but `classify()` stays as the
#: readable reference and the thing unit tests can exercise directly.
#:
#: Prefix lengths used by substr(): 'GCNS ' and 'Gaia ' are 5 chars,
#: 'Gaia Cepheid ' is 13, 'Gaia RR Lyrae ' is 14, 'APOGEE ' is 7.
_TRANSFORM = """
INSERT INTO stars
WITH base AS (
    SELECT
        TRY_CAST(id AS BIGINT)               AS row_id,
        trim(coalesce(gl, ''))               AS label,
        nullif(trim(coalesce(proper, '')), '') AS proper_name,
        nullif(trim(coalesce(con, '')), '')    AS constellation,
        nullif(trim(coalesce(spect, '')), '')  AS spectral_type,
        TRY_CAST(ra AS DOUBLE)               AS ra_deg,
        TRY_CAST(dec AS DOUBLE)              AS dec_deg,
        TRY_CAST(dist AS DOUBLE)             AS dist_pc,
        TRY_CAST(ci AS DOUBLE)               AS color_bp_rp,
        TRY_CAST(lum AS DOUBLE)              AS lum_lsun
    -- The dialect is pinned rather than sniffed. DuckDB's sniffer treats
    -- ' as a candidate quote character, and the catalogue contains
    -- "Van Maanen's Star", whose lone apostrophe makes every candidate
    -- dialect look malformed -- the sniff fails outright on the real
    -- file. strict_mode=false keeps the apostrophe as literal text.
    FROM read_csv(
        ?,
        header = true,
        delim = ',',
        all_varchar = true,
        strict_mode = false
    )
),
classified AS (
    SELECT *, CASE
        WHEN label = ''                     THEN 'hyg'
        WHEN label LIKE 'GCNS %'            THEN 'gaia_gcns'
        WHEN label LIKE 'Gaia Cepheid %'    THEN 'gaia_cepheid'
        WHEN label LIKE 'Gaia RR Lyrae %'   THEN 'gaia_rrlyrae'
        WHEN label LIKE 'APOGEE %'          THEN 'apogee'
        WHEN label LIKE 'Gaia %'            THEN 'gaia_dr3_luminous'
        ELSE 'hyg'
    END AS subset
    FROM base
),
ided AS (
    SELECT *, CASE subset
        WHEN 'gaia_gcns'         THEN nullif(trim(substr(label, 6)), '')
        WHEN 'gaia_cepheid'      THEN nullif(trim(substr(label, 14)), '')
        WHEN 'gaia_rrlyrae'      THEN nullif(trim(substr(label, 15)), '')
        WHEN 'apogee'            THEN nullif(trim(substr(label, 8)), '')
        WHEN 'gaia_dr3_luminous' THEN nullif(trim(substr(label, 6)), '')
        ELSE nullif(label, '')
    END AS source_id
    FROM classified
)
SELECT
    coalesce(row_id, 0),
    'row:' || subset || ':' ||
        coalesce(source_id, proper_name, CAST(coalesce(row_id, 0) AS VARCHAR)),
    subset,
    subset IN ('gaia_gcns', 'gaia_cepheid', 'gaia_rrlyrae', 'gaia_dr3_luminous'),
    nullif(label, ''),
    source_id,
    proper_name,
    constellation,
    spectral_type,
    ra_deg,
    dec_deg,
    dist_pc,
    dist_pc * ?,
    dist_pc / 1000.0,
    color_bp_rp,
    lum_lsun
FROM ided
-- A row with no usable distance cannot support a numeric claim, so it is
-- dropped rather than stored with a null that might later be rendered as
-- a number. This also drops Sol, which sits at distance 0.
WHERE dist_pc IS NOT NULL AND dist_pc > 0
  -- HYG encodes "no usable parallax" as exactly 100000 pc. There are 64
  -- such rows and the next largest real HYG distance is 990 pc, so the
  -- gap is unambiguous. Left in, they would be returned as the most
  -- distant stars in the catalogue by any "farthest" query, and the
  -- answer would cite a real row and pass every grounding check while
  -- being pure fiction. A sentinel is the most dangerous kind of bad
  -- data for a RAG: it is well-formed.
  AND dist_pc < """ + str(HYG_UNKNOWN_DISTANCE_PC) + """
"""


def build(csv_path: Path | None = None, db_path: Path | None = None) -> dict:
    from ..units import LY_PER_PC

    csv_path = csv_path or CONFIG.paths.galaxy_catalogue
    db_path = db_path or CONFIG.paths.duckdb
    if not csv_path.exists():
        raise FileNotFoundError(
            f"catalogue not found at {csv_path}. Set GAIA_RAG_GALAXY_CSV."
        )
    CONFIG.ensure_dirs()
    if db_path.exists():
        db_path.unlink()

    con = duckdb.connect(str(db_path))
    con.execute(SCHEMA)
    con.execute(_TRANSFORM, [str(csv_path), LY_PER_PC])

    total_csv = con.execute(
        "SELECT count(*) FROM read_csv(?, header = true, delim = ',', "
        "all_varchar = true, strict_mode = false)",
        [str(csv_path)],
    ).fetchone()[0]
    counts = dict(
        con.execute(
            "SELECT subset, count(*) FROM stars GROUP BY subset"
        ).fetchall()
    )
    skipped = int(total_csv) - sum(counts.values())

    con.executemany(
        "INSERT INTO field_provenance VALUES (?,?,?,?,?,?,?)",
        [
            (
                p.subset,
                p.field,
                p.status,
                p.method,
                p.rel_uncertainty,
                p.caveat,
                p.source_doc,
            )
            for p in _provenance_rows()
        ],
    )

    con.execute("CREATE INDEX idx_subset ON stars(subset)")
    con.execute("CREATE INDEX idx_proper ON stars(proper_name)")
    con.execute("CREATE INDEX idx_source ON stars(source_id)")
    con.execute("CREATE INDEX idx_dist ON stars(dist_pc)")
    total = con.execute("SELECT count(*) FROM stars").fetchone()[0]
    con.close()

    return {"total": total, "skipped_no_distance": skipped, "by_subset": counts}


if __name__ == "__main__":
    stats = build()
    print(f"rows ingested        : {stats['total']:,}")
    print(f"dropped (no distance): {stats['skipped_no_distance']:,}")
    for subset, n in sorted(stats["by_subset"].items(), key=lambda kv: -kv[1]):
        tag = "GAIA" if subset in GAIA_SUBSETS else "    "
        print(f"  {tag} {subset:22s} {n:>8,}")
