"""Build the unstructured half of the corpus: method and provenance docs.

Everything here is extracted from text that exists on disk -- Galaxy's
SOURCES.txt and the docstrings of its ingestion tools -- rather than
written from memory about the cited papers. That matters: if the corpus
itself contained paraphrases I invented, then a "faithful" answer
grounded in that corpus would still be wrong, and the faithfulness
metric would happily report success. Grounding is only worth measuring
if the ground is real.

Each document gets a stable id used in citations (`doc:<id>#<n>`) and
carries the path it was extracted from, so any claim can be traced back
to a file and line range by a human.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import CONFIG


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    source_path: str
    citation: str
    kind: str  # "provenance" | "method" | "caveat"


#: Which tool script documents which subset, and how to cite it.
TOOL_DOCS = {
    "gcns": (
        "add_gcns.py",
        "Gaia Catalogue of Nearby Stars ingestion",
        "Gaia Collaboration, Smart et al. 2021, A&A 649, A6 (VizieR J/A+A/649/A6)",
    ),
    "gaia_far": (
        "add_gaia_far.py",
        "Luminous Gaia DR3 stars with Bailer-Jones distances",
        "Bailer-Jones et al. 2021; Gaia DR3 (external.gaiaedr3_distance)",
    ),
    "gaia_variables": (
        "add_gaia_variables.py",
        "Gaia DR3 Cepheids and RR Lyrae as standard candles",
        "Ripepi et al. 2022, A&A 674, A17; Muraveva et al. 2018",
    ),
    "apogee": (
        "add_apogee_ir.py",
        "APOGEE DR17 red giants with StarHorse distances",
        "Queiroz et al. 2023, A&A 673, A155 (VizieR J/A+A/673/A155/apogee17)",
    ),
    "hyg": (
        "trim_hyg.py",
        "HYG database trimming for the web catalogue",
        "HYG Database v4.1, David Nash (astronexus), CC BY-SA 4.0",
    ),
    "openngc": (
        "build_openngc.py",
        "OpenNGC objects with Gaia-based parallaxes",
        "OpenNGC, Mattia Verga, CC BY-SA 4.0",
    ),
    "planetary_nebulae": (
        "build_planetary_nebulae.py",
        "Planetary nebula distances",
        "Gonzalez-Santamaria et al. 2019/2020, A&A 630, A150; Acker et al. 1992, VizieR V/84",
    ),
}


def _parse(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return None


def _module_docstring(path: Path) -> str | None:
    """Pull a module docstring without importing (the tools need network)."""
    tree = _parse(path)
    return ast.get_docstring(tree) if tree else None


#: Constants whose values are not provenance (endpoints, paths, limits
#: that say nothing about selection) and would only add noise.
_BORING_CONSTANTS = {"TAP", "TAP_ASYNC", "TAP_SYNC", "OUT", "CACHE", "HEADERS"}


_SQL_TABLE = re.compile(r"(?:FROM|JOIN)\s+\"?([A-Za-z_][\w./+]*)\"?", re.IGNORECASE)
_SQL_IDENT = re.compile(r"\b(?:[a-z]\.)?([a-z_][a-z0-9_]{2,})\b")
_SQL_NOISE = frozenset(
    """select from join where and or not null using as is between order by asc
    desc top log10 count avg min max sum table1c recno""".split()
)


def _render_string(node: ast.AST, consts: dict[str, object]) -> str | None:
    """Render a string literal, including an f-string, to its text.

    f-strings matter here: `add_gaia_far.py` builds its ADQL with one, so
    the node is a `JoinedStr` rather than a `Constant`. An earlier
    version checked only for `Constant` and silently produced a selection
    document with no query in it -- for precisely the subset whose
    `parallax_over_error > 5` cut a question was most likely to ask
    about. Interpolated names are substituted with their real values
    where known, so the recorded query is the one that actually ran.
    """
    if isinstance(node, ast.Constant):
        return node.value.strip() if isinstance(node.value, str) else None
    if not isinstance(node, ast.JoinedStr):
        return None
    out: list[str] = []
    for part in node.values:
        if isinstance(part, ast.Constant) and isinstance(part.value, str):
            out.append(part.value)
        elif isinstance(part, ast.FormattedValue):
            expr = part.value
            if isinstance(expr, ast.Name) and expr.id in consts:
                out.append(str(consts[expr.id]))
            else:
                try:
                    out.append(f"{{{ast.unparse(expr)}}}")
                except Exception:
                    out.append("{...}")
    return "".join(out).strip() or None


def _describe_query(sql: str) -> str:
    """One prose sentence naming a query's tables and filtered columns.

    Gives lexical and dense retrieval something to match. The columns are
    taken from the WHERE clause, which is where the selection actually
    happens -- those are the names a question about quality cuts or
    classification filters will use.
    """
    tables = []
    for match in _SQL_TABLE.finditer(sql):
        name = match.group(1)
        if name not in tables:
            tables.append(name)

    where = sql.upper().find("WHERE")
    columns: list[str] = []
    if where != -1:
        for match in _SQL_IDENT.finditer(sql[where + 5 :].lower()):
            name = match.group(1)
            if name in _SQL_NOISE or name in columns:
                continue
            columns.append(name)

    parts = []
    if tables:
        parts.append(
            "Reads the catalogue table(s) " + ", ".join(tables) + "."
        )
    if columns:
        parts.append(
            "Selection filters on the columns: " + ", ".join(columns[:12]) + "."
        )
    return " ".join(parts)


def _selection_criteria(path: Path) -> str | None:
    """Extract the actual query text and selection constants from a tool.

    The docstrings say what a tool does in prose; the code says exactly
    what it selected. Only docstrings were ingested at first, which left
    the corpus thin -- 17.8k characters total, with one gold case finding
    just half its expected documents. The ADQL in these files names the
    archive tables, the quality cuts (`parallax_over_error > 5`), the
    classification filters (`type_best_classification = 'DCEP'`) and the
    VizieR catalogue identifiers. That is the most verifiable provenance
    available anywhere in the project, and it was sitting unread on disk.
    """
    tree = _parse(path)
    if tree is None:
        return None

    # Numeric constants first, so f-string queries can be rendered with
    # their real values substituted in.
    consts: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    consts[target.id] = node.value.value

    queries: list[tuple[str, str]] = []
    constants: list[tuple[str, str]] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if not targets:
            continue
        name = targets[0]
        if name in _BORING_CONSTANTS or name.startswith("_"):
            continue
        value = node.value
        text = _render_string(value, consts)
        if text and len(text) > 60 and "SELECT" in text.upper():
            queries.append((name, text))
        elif isinstance(value, ast.Constant) and isinstance(
            value.value, (int, float)
        ):
            constants.append((name, repr(value.value)))

    if not queries and not constants:
        return None

    lines: list[str] = []
    if queries:
        # A prose gloss before the SQL. Without it these documents were
        # retrieved for 1 of 3 selection questions: a raw ADQL block
        # neither embeds like prose nor shares vocabulary with "what
        # quality cut was applied", so there was nothing for either
        # retriever to match on. Naming the tables and the filtered
        # columns in a sentence gives both a handle.
        for name, text in queries:
            gloss = _describe_query(text)
            if gloss:
                lines.append(gloss)
                lines.append("")
        lines.append("Exact catalogue query used to select these rows:")
        for name, text in queries:
            lines.append(f"{name}:")
            lines.append(text)
            lines.append("")
    if constants:
        lines.append("Selection constants defined in the ingestion script:")
        for name, text in constants:
            lines.append(f"- {name} = {text}")
    return "\n".join(lines).strip()


def _split_sources(text: str) -> list[tuple[str, str]]:
    """Split SOURCES.txt into (heading, body) blocks.

    The file is organised as blank-line-separated blocks whose first line
    names a file or a catalogue, so that first line is the heading.
    """
    blocks: list[tuple[str, str]] = []
    for raw_block in re.split(r"\n\s*\n", text):
        block = raw_block.strip("\n").rstrip()
        if not block:
            continue
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        # Skip the title underline decoration ("====").
        if set(lines[0].strip()) <= {"=", "-"}:
            continue
        heading = lines[0].strip()
        if len(lines) == 1:
            continue
        body = "\n".join(ln.strip() for ln in lines[1:])
        blocks.append((heading, body))
    return blocks


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:48] or "block"


def collect() -> list[Document]:
    docs: list[Document] = []
    paths = CONFIG.paths

    # 1. Method docs from the ingestion tools' own docstrings. These are
    #    the richest and most specific text available: they state the
    #    actual selection cuts, the relations used, and the caveats.
    for doc_id, (filename, title, citation) in TOOL_DOCS.items():
        path = paths.galaxy_tools / filename
        if not path.exists():
            continue
        docstring = _module_docstring(path)
        if not docstring:
            continue
        docs.append(
            Document(
                doc_id=f"method_{doc_id}",
                title=title,
                text=docstring.strip(),
                source_path=str(path),
                citation=citation,
                kind="method",
            )
        )

        # 1b. The query and cuts the script actually ran, as a separate
        #     document so a question about selection criteria retrieves
        #     the criteria rather than the prose summary of them.
        criteria = _selection_criteria(path)
        if criteria:
            docs.append(
                Document(
                    doc_id=f"selection_{doc_id}",
                    title=f"Selection criteria and query: {title}",
                    text=criteria,
                    source_path=str(path),
                    citation=f"{citation} -- selection as implemented",
                    kind="selection",
                )
            )

    # 2. Provenance blocks from SOURCES.txt.
    if paths.galaxy_sources.exists():
        text = paths.galaxy_sources.read_text(encoding="utf-8", errors="replace")
        for heading, body in _split_sources(text):
            docs.append(
                Document(
                    doc_id=f"sources_{_slug(heading)}",
                    title=f"Data source: {heading}",
                    text=f"{heading}\n{body}",
                    source_path=str(paths.galaxy_sources),
                    citation="Galaxy project data provenance (web/data/SOURCES.txt)",
                    kind="provenance",
                )
            )

    # 3. Field-level caveats, generated from the same provenance table the
    #    structured retriever uses, so the two halves of the corpus cannot
    #    drift apart and contradict each other.
    #    Distance provenance differs per subset and is the thing most
    #    questions turn on, so each subset gets its own document. The
    #    other fields are documented once per distinct variant instead of
    #    once per subset: emitting them per subset produced six chunks of
    #    byte-identical boilerplate about colour and luminosity, which
    #    then outranked the distinctive documents and broke retrieval for
    #    the questions they were written to answer.
    from .catalog import SUBSET_LABELS, _provenance_rows

    all_provs = _provenance_rows()

    def describe(prov) -> str:
        pct = (
            f"{prov.rel_uncertainty * 100:.0f}%"
            if prov.rel_uncertainty is not None
            else "unquantified"
        )
        return (
            f"- {prov.field}: {prov.status}, typical relative uncertainty "
            f"{pct}.\n  Method: {prov.method}\n  Caveat: {prov.caveat}"
        )

    for prov in all_provs:
        if prov.field != "dist_pc":
            continue
        label = SUBSET_LABELS.get(prov.subset, prov.subset)
        docs.append(
            Document(
                doc_id=f"caveats_{prov.subset}",
                title=f"Distance reliability and caveats: {label}",
                text="\n".join(
                    [
                        f"How distances were determined for {label}, and how far "
                        "they can be trusted.",
                        f"Internal catalogue subset name: {prov.subset}.",
                        "",
                        describe(prov),
                    ]
                ),
                source_path="gaia_rag/ingest/catalog.py::DISTANCE_PROVENANCE",
                citation="Derived from Galaxy SOURCES.txt and tool docstrings",
                kind="caveat",
            )
        )

    variants: dict[tuple, list[str]] = {}
    for prov in all_provs:
        if prov.field == "dist_pc":
            continue
        key = (
            prov.field,
            prov.status,
            prov.method,
            prov.rel_uncertainty,
            prov.caveat,
        )
        variants.setdefault(key, []).append(prov.subset)

    by_field: dict[str, list[tuple[tuple, list[str]]]] = {}
    for key, subsets in variants.items():
        by_field.setdefault(key[0], []).append((key, subsets))

    for field_name, entries in by_field.items():
        # A variant that applies to exactly one subset gets its own
        # document, named after that subset.
        #
        # Folding them into the shared field document turned it back into
        # a hub chunk: adding the RR Lyrae constant-luminosity facts to
        # the general lum_lsun document made it the top hit for "how were
        # Cepheid distances derived?", on generic words like "derived"
        # and "distance", pushing the actual Cepheid document out of the
        # results entirely. Subset-specific facts belong in
        # subset-specific documents -- the same lesson as the earlier
        # de-duplication, relearned.
        shared = [(k, s) for k, s in entries if len(s) > 1]
        for key, subsets in [(k, s) for k, s in entries if len(s) == 1]:
            subset = subsets[0]
            label = SUBSET_LABELS.get(subset, subset)
            prov = next(
                p
                for p in all_provs
                if (p.field, p.status, p.method, p.rel_uncertainty, p.caveat) == key
            )
            docs.append(
                Document(
                    doc_id=f"caveats_field_{field_name}_{subset}",
                    title=f"Reliability of {field_name} for {label}",
                    text="\n".join(
                        [
                            f"The {field_name} field as it applies "
                            f"specifically to {label}.",
                            "",
                            describe(prov),
                        ]
                    ),
                    source_path="gaia_rag/ingest/catalog.py::_provenance_rows",
                    citation="Derived from Galaxy SOURCES.txt and tool docstrings",
                    kind="caveat",
                )
            )

        if not shared:
            continue
        lines = [
            f"Reliability of the {field_name} field across the catalogue.",
            "",
        ]
        for key, subsets in shared:
            # No subset identifiers at all, in any form.
            #
            # Human labels were tried first and put "RR Lyrae" and
            # "classical Cepheids" into every field document. Switching
            # to internal slugs looked safer but is not: the lexical
            # tokeniser splits on underscores, so "gaia_cepheid" becomes
            # "gaia" + "cepheid" and the shared luminosity document still
            # matched all three terms of "how were Cepheid distances
            # derived?" -- beating the actual Cepheid document.
            #
            # Nothing is lost by omitting them: a variant that applies to
            # only one subset now has its own document above, so the
            # exceptions are where a reader would look for them.
            all_subsets = set(SUBSET_LABELS) == set(subsets)
            labels = (
                "all catalogue subsets"
                if all_subsets
                else "all catalogue subsets except those with a "
                "subset-specific note of their own"
            )
            prov = next(
                p
                for p in all_provs
                if (p.field, p.status, p.method, p.rel_uncertainty, p.caveat) == key
            )
            lines.append(f"Applies to: {labels}")
            lines.append(describe(prov))
            lines.append("")
        docs.append(
            Document(
                doc_id=f"caveats_field_{field_name}",
                title=f"Reliability and caveats: {field_name} field",
                text="\n".join(lines).strip(),
                source_path="gaia_rag/ingest/catalog.py::_provenance_rows",
                citation="Derived from Galaxy SOURCES.txt and tool docstrings",
                kind="caveat",
            )
        )

    return docs


def build() -> dict:
    CONFIG.ensure_dirs()
    docs = collect()
    if not docs:
        raise RuntimeError(
            "no documents collected -- check GAIA_RAG_GALAXY_SOURCES and "
            "GAIA_RAG_GALAXY_TOOLS point at the Galaxy project."
        )
    out_dir = CONFIG.paths.docs
    for stale in out_dir.glob("*.json"):
        stale.unlink()
    for doc in docs:
        (out_dir / f"{doc.doc_id}.json").write_text(
            json.dumps(asdict(doc), indent=2), encoding="utf-8"
        )
    by_kind: dict[str, int] = {}
    for doc in docs:
        by_kind[doc.kind] = by_kind.get(doc.kind, 0) + 1
    return {
        "documents": len(docs),
        "by_kind": by_kind,
        "chars": sum(len(d.text) for d in docs),
        "out_dir": str(out_dir),
    }


def load() -> list[Document]:
    docs = []
    for path in sorted(CONFIG.paths.docs.glob("*.json")):
        docs.append(Document(**json.loads(path.read_text(encoding="utf-8"))))
    return docs


if __name__ == "__main__":
    stats = build()
    print(f"documents : {stats['documents']}")
    print(f"characters: {stats['chars']:,}")
    for kind, n in sorted(stats["by_kind"].items()):
        print(f"  {kind:12s} {n}")
    print(f"written to: {stats['out_dir']}")
