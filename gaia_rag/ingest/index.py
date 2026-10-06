"""Embed chunks and persist the vector index.

The corpus is small (tens of chunks), so an exact numpy dot-product
search beats an approximate index: it is faster at this scale, has no
index-build nondeterminism, and makes eval runs bit-reproducible. The
store is a plain .npz plus a .jsonl of chunk records, both diffable.

Swap in FAISS here if the corpus grows past ~100k chunks; the retriever
only depends on `search()`.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..config import CONFIG
from ..providers import get_embedder
from .chunker import Chunk, chunk_all
from .docs import load as load_docs


def build() -> dict:
    CONFIG.ensure_dirs()
    docs = load_docs()
    if not docs:
        raise RuntimeError("no documents on disk -- run gaia_rag.ingest.docs first")

    chunks = chunk_all(docs)
    embedder = get_embedder()
    vectors = embedder.embed([c.embedding_text for c in chunks])

    np.savez_compressed(
        CONFIG.paths.doc_index,
        vectors=vectors,
        chunk_ids=np.array([c.chunk_id for c in chunks], dtype=object),
        model=np.array(embedder.name),
    )
    with CONFIG.paths.doc_store.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(asdict(chunk)) + "\n")

    return {
        "chunks": len(chunks),
        "dim": int(vectors.shape[1]) if len(chunks) else 0,
        "model": embedder.name,
        "mean_chars": int(np.mean([len(c.text) for c in chunks])) if chunks else 0,
        "index": str(CONFIG.paths.doc_index),
    }


class DocIndex:
    """Loaded vector index over document chunks."""

    def __init__(self, index_path: Path | None = None, store_path: Path | None = None):
        index_path = index_path or CONFIG.paths.doc_index
        store_path = store_path or CONFIG.paths.doc_store
        if not index_path.exists() or not store_path.exists():
            raise FileNotFoundError(
                "document index missing -- run `python -m gaia_rag.ingest.index`"
            )
        payload = np.load(index_path, allow_pickle=True)
        self.vectors: np.ndarray = payload["vectors"]
        self.model = str(payload["model"])
        self.chunks: list[Chunk] = []
        with store_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                self.chunks.append(Chunk(**json.loads(line)))
        if len(self.chunks) != self.vectors.shape[0]:
            raise RuntimeError(
                f"index/store mismatch: {self.vectors.shape[0]} vectors vs "
                f"{len(self.chunks)} chunks. Rebuild the index."
            )

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        """Cosine similarity search. Vectors are pre-normalised."""
        if not self.chunks:
            return []
        sims = self.vectors @ query_vector.astype(np.float32)
        k = min(k, len(self.chunks))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(self.chunks[i], float(sims[i])) for i in top]


if __name__ == "__main__":
    stats = build()
    print(f"chunks     : {stats['chunks']}")
    print(f"dimension  : {stats['dim']}")
    print(f"embed model: {stats['model']}")
    print(f"mean chars : {stats['mean_chars']}")
    print(f"index      : {stats['index']}")
