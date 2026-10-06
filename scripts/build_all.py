"""Build the whole corpus from scratch.

    python scripts/build_all.py

Runs the three ingest stages in order: catalogue rows into DuckDB,
method/provenance documents onto disk, then the hybrid document index.
Order matters -- the document builder reads the provenance table that the
catalogue stage defines, and the index reads the documents.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gaia_rag.config import CONFIG  # noqa: E402
from gaia_rag.ingest import catalog, docs, index  # noqa: E402
from gaia_rag.ingest.catalog import GAIA_SUBSETS  # noqa: E402


def main() -> int:
    CONFIG.ensure_dirs()

    if not CONFIG.paths.galaxy_catalogue.exists():
        print(
            f"ERROR: source catalogue not found:\n  "
            f"{CONFIG.paths.galaxy_catalogue}\n\n"
            "Set GAIA_RAG_GALAXY_CSV to the Galaxy project's "
            "web/data/hygdata_web.csv",
            file=sys.stderr,
        )
        return 1

    t0 = time.time()
    print("[1/3] catalogue rows -> DuckDB")
    stats = catalog.build()
    gaia_rows = sum(n for s, n in stats["by_subset"].items() if s in GAIA_SUBSETS)
    print(f"      {stats['total']:,} rows ({gaia_rows:,} from Gaia)")
    print(f"      dropped without a usable distance: {stats['skipped_no_distance']}")
    for subset, n in sorted(stats["by_subset"].items(), key=lambda kv: -kv[1]):
        tag = "gaia" if subset in GAIA_SUBSETS else "    "
        print(f"        {tag} {subset:22s} {n:>8,}")

    print("[2/3] method and provenance documents")
    doc_stats = docs.build()
    print(
        f"      {doc_stats['documents']} documents, "
        f"{doc_stats['chars']:,} characters"
    )

    print("[3/3] hybrid document index")
    idx_stats = index.build()
    print(
        f"      {idx_stats['chunks']} chunks, dim {idx_stats['dim']}, "
        f"model {idx_stats['model']}"
    )

    print(f"\ndone in {time.time() - t0:.1f}s")
    print(f"  catalogue : {CONFIG.paths.duckdb}")
    print(f"  index     : {CONFIG.paths.doc_index}")
    print("\nNext:")
    print('  python scripts/ask.py "How were Cepheid distances derived?"')
    print("  python -m gaia_rag.eval.run_eval")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
