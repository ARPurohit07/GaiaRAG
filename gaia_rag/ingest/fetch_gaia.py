"""Optional: fetch fresh Gaia DR3 rows from the ESA archive.

    python -m gaia_rag.ingest.fetch_gaia --subset cepheid --limit 2000
    python -m gaia_rag.ingest.fetch_gaia --subset gcns --limit 5000

The rest of the pipeline builds from Galaxy's local catalogue, which is
real Gaia data but already collapsed to distance and luminosity. This
module exists to recover the source quantities those were derived from
-- parallax, `parallax_over_error`, `phot_g_mean_mag`, `bp_rp`,
pulsation period -- which make a strictly better corpus: the system can
then report a parallax *and* its signal-to-noise rather than trusting a
precomputed distance, and the provenance stops being an assertion from a
docstring and becomes a retrieved value.

This is not run by `build_all.py` and the pipeline does not require it.
The ESA TAP endpoint was unreachable from the machine this was developed
on (every query timed out, including trivially cheap ones), so treat a
timeout here as an environment condition rather than a bug and fall back
to the local catalogue.

Output goes to `data/raw/gaia_<subset>.csv` and is deliberately left as
raw archive columns. Nothing downstream reads it yet; wiring it into
`catalog.build()` means extending the `stars` schema with the source
columns, which is a schema change and so a deliberate one.

ADQL queries follow the selections documented in Galaxy's own tools, so
the rows are comparable to what is already in the catalogue.
"""

from __future__ import annotations

import argparse
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from ..config import CONFIG

TAP_SYNC = "https://gea.esac.esa.int/tap-server/tap/sync"
TAP_ASYNC = "https://gea.esac.esa.int/tap-server/tap/async"

#: Each query selects the measured quantities, not a derived distance,
#: so the provenance chain can be reconstructed from the data itself.
QUERIES: dict[str, str] = {
    "gcns": """
SELECT TOP {limit}
    g.source_id, g.ra, g.dec, g.parallax, g.parallax_over_error,
    g.phot_g_mean_mag, g.bp_rp, g.pmra, g.pmdec
FROM gaiadr3.gaia_source AS g
WHERE g.parallax > 10 AND g.parallax_over_error > 10
  AND g.phot_g_mean_mag IS NOT NULL AND g.bp_rp IS NOT NULL
ORDER BY g.phot_g_mean_mag ASC
""",
    "cepheid": """
SELECT TOP {limit}
    v.source_id, g.ra, g.dec, g.phot_g_mean_mag, g.bp_rp,
    v.pf AS period_days, v.type_best_classification,
    g.parallax, g.parallax_over_error
FROM gaiadr3.vari_cepheid AS v
JOIN gaiadr3.gaia_source AS g USING (source_id)
WHERE v.type_best_classification = 'DCEP' AND v.pf > 1.5
  AND g.bp_rp IS NOT NULL AND g.phot_g_mean_mag IS NOT NULL
""",
    "rrlyrae": """
SELECT TOP {limit}
    v.source_id, g.ra, g.dec, g.phot_g_mean_mag, g.bp_rp,
    v.pf AS period_days, v.best_classification,
    g.parallax, g.parallax_over_error
FROM gaiadr3.vari_rrlyrae AS v
JOIN gaiadr3.gaia_source AS g USING (source_id)
WHERE g.bp_rp IS NOT NULL AND g.phot_g_mean_mag IS NOT NULL
""",
    "luminous": """
SELECT TOP {limit}
    g.source_id, g.ra, g.dec, g.phot_g_mean_mag, g.bp_rp,
    g.parallax, g.parallax_over_error, d.r_med_geo AS dist_pc,
    d.r_lo_geo, d.r_hi_geo
FROM gaiadr3.gaia_source AS g
JOIN external.gaiaedr3_distance AS d USING (source_id)
WHERE g.random_index < 900000000 AND g.parallax_over_error > 5
  AND d.r_med_geo BETWEEN 1000 AND 5000
  AND g.phot_g_mean_mag - 5 * LOG10(d.r_med_geo / 10.0) < -1.0
""",
}


def _submit_async(adql: str, timeout: int) -> str:
    """Submit an async TAP job and return its result URL."""
    data = urllib.parse.urlencode(
        {
            "REQUEST": "doQuery",
            "LANG": "ADQL",
            "FORMAT": "csv",
            "PHASE": "RUN",
            "QUERY": adql,
        }
    ).encode()
    request = urllib.request.Request(TAP_ASYNC, data=data)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        job_url = response.geturl().split("?")[0]
    return job_url


def _await_job(job_url: str, timeout: int, poll_s: float = 3.0) -> str:
    """Poll a TAP job until it completes; return the result CSV."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with urllib.request.urlopen(f"{job_url}/phase", timeout=60) as response:
            phase = response.read().decode().strip()
        if phase == "COMPLETED":
            with urllib.request.urlopen(
                f"{job_url}/results/result", timeout=timeout
            ) as response:
                return response.read().decode("utf-8", "replace")
        if phase in ("ERROR", "ABORTED"):
            with urllib.request.urlopen(f"{job_url}/error", timeout=60) as response:
                raise RuntimeError(f"TAP job {phase}: {response.read().decode()[:500]}")
        time.sleep(poll_s)
    raise TimeoutError(f"TAP job did not finish within {timeout}s: {job_url}")


def fetch(subset: str, limit: int, timeout: int | None = None) -> Path:
    if subset not in QUERIES:
        raise ValueError(
            f"unknown subset {subset!r}; choose from {', '.join(QUERIES)}"
        )
    timeout = timeout or CONFIG.models.timeout_s
    adql = QUERIES[subset].format(limit=limit).strip()
    CONFIG.ensure_dirs()
    out = CONFIG.paths.raw / f"gaia_{subset}.csv"

    print(f"submitting ADQL for {subset!r} (limit {limit:,})...")
    try:
        job_url = _submit_async(adql, timeout)
        print(f"  job: {job_url}")
        csv_text = _await_job(job_url, timeout)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(
            f"could not reach the Gaia archive ({exc}). This endpoint was "
            "unreachable from the development machine too; the pipeline does "
            "not need it -- build from the local Galaxy catalogue instead."
        ) from exc

    out.write_text(csv_text, encoding="utf-8")
    rows = max(0, csv_text.count("\n") - 1)
    print(f"  wrote {rows:,} rows -> {out}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch raw Gaia DR3 rows from the ESA TAP archive."
    )
    parser.add_argument(
        "--subset", required=True, choices=sorted(QUERIES), help="which query to run"
    )
    parser.add_argument("--limit", type=int, default=2000, help="row cap")
    parser.add_argument(
        "--timeout", type=int, default=None, help="seconds before giving up"
    )
    args = parser.parse_args()
    try:
        fetch(args.subset, args.limit, args.timeout)
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
