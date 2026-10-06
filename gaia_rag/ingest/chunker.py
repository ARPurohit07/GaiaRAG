"""Chunk documents for retrieval.

Chunks are the citation unit, so they are split on paragraph and bullet
boundaries rather than at a fixed character offset. A caveat cut in half
mid-sentence is worse than useless: it retrieves, it looks like support,
and it says the opposite of what the full sentence says.

Every chunk keeps its parent document's title and citation inline, so a
chunk pulled out of context still states what it is evidence about.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..config import CONFIG
from .docs import Document

# Rough token estimate: good enough for sizing, and avoids a tokenizer
# dependency in the ingest path.
_CHARS_PER_TOKEN = 4

#: nomic-embed-text task prefixes. These must stay paired: documents
#: embedded with one prefix are only comparable to queries embedded with
#: the other, so `ingest.index` records the scheme and the retriever
#: refuses a mismatch.
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


def query_text(question: str) -> str:
    """Embed a question the way the index expects."""
    return f"{QUERY_PREFIX}{question}"


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


@dataclass
class Chunk:
    chunk_id: str  # "doc:<doc_id>#<n>" -- the citation handle
    doc_id: str
    title: str
    text: str
    citation: str
    source_path: str
    kind: str
    ordinal: int

    @property
    def embedding_text(self) -> str:
        """What actually gets embedded.

        Two things matter here. The title is prepended because chunk
        bodies often describe a method without naming the dataset, so
        without it a question about RR Lyrae reliability has nothing to
        match on. And the text carries nomic-embed-text's documented
        `search_document:` task prefix -- the model is trained with
        asymmetric query/document prefixes, and omitting them compresses
        the whole similarity range (an off-topic question still scored
        0.53 against this corpus without them).
        """
        return f"{DOC_PREFIX}{self.title}\n\n{self.text}"


def _split_units(text: str) -> list[str]:
    """Break text into the smallest units we are willing to split on."""
    units: list[str] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        block = block.strip()
        if not block:
            continue
        # Keep bullet lists together per bullet.
        if re.match(r"^\s*[-*]\s", block, re.MULTILINE):
            current: list[str] = []
            for line in block.splitlines():
                if re.match(r"^\s*[-*]\s", line) and current:
                    units.append("\n".join(current).strip())
                    current = [line]
                else:
                    current.append(line)
            if current:
                units.append("\n".join(current).strip())
        else:
            units.append(block)
    return [u for u in units if u]


def chunk_document(doc: Document) -> list[Chunk]:
    max_tokens = CONFIG.retrieval.chunk_tokens
    overlap_tokens = CONFIG.retrieval.chunk_overlap

    units = _split_units(doc.text)
    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_tokens = 0

    def flush() -> None:
        nonlocal buffer, buffer_tokens
        if not buffer:
            return
        text = "\n\n".join(buffer).strip()
        ordinal = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=f"doc:{doc.doc_id}#{ordinal}",
                doc_id=doc.doc_id,
                title=doc.title,
                text=text,
                citation=doc.citation,
                source_path=doc.source_path,
                kind=doc.kind,
                ordinal=ordinal,
            )
        )
        # Carry the tail of this chunk into the next one so a fact split
        # across the boundary is still retrievable from one side.
        if overlap_tokens > 0:
            tail: list[str] = []
            tail_tokens = 0
            for unit in reversed(buffer):
                unit_tokens = _approx_tokens(unit)
                if tail_tokens + unit_tokens > overlap_tokens:
                    break
                tail.insert(0, unit)
                tail_tokens += unit_tokens
            buffer = tail
            buffer_tokens = tail_tokens
        else:
            buffer = []
            buffer_tokens = 0

    for unit in units:
        unit_tokens = _approx_tokens(unit)
        if unit_tokens > max_tokens:
            # Oversized single paragraph: emit it whole rather than
            # cutting a sentence. Retrieval tolerates a long chunk;
            # truncation corrupts meaning.
            flush()
            ordinal = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"doc:{doc.doc_id}#{ordinal}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    text=unit,
                    citation=doc.citation,
                    source_path=doc.source_path,
                    kind=doc.kind,
                    ordinal=ordinal,
                )
            )
            buffer, buffer_tokens = [], 0
            continue
        if buffer_tokens + unit_tokens > max_tokens:
            flush()
        buffer.append(unit)
        buffer_tokens += unit_tokens

    flush()
    return chunks


def chunk_all(docs: list[Document]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc))
    return chunks
