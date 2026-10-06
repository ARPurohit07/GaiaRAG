"""Hybrid retrieval over method and provenance documents.

This half of the corpus answers "how was this derived?" and "how much
should I trust it?" rather than "what is the value?".

Dense and lexical scores are fused by reciprocal rank fusion. Neither
works alone on this corpus: dense retrieval blurs rare terms like "RR
Lyrae" into generic Gaia-distance similarity, and BM25 alone misses
paraphrases that share no vocabulary with the documents.

The abstention gate is lexical rather than a cosine threshold, and that
is a measured choice. On this corpus an off-topic question scores ~0.53
cosine against the nearest chunk while a genuine question scores ~0.72;
no threshold in that gap survives a corpus change. Lexical overlap
separates them cleanly instead: a question about sourdough shares no
content term with any chunk, so there is nothing to retrieve and we say
so rather than serving the nearest Gaia document.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import CONFIG
from ..ingest.chunker import query_text
from ..ingest.index import DocIndex
from .evidence import Evidence, chunk_to_evidence
from .lexical import BM25, reciprocal_rank_fusion


@dataclass
class SemanticResult:
    evidence: list[Evidence] = field(default_factory=list)
    best_score: float = 0.0
    best_dense: float = 0.0
    best_lexical: float = 0.0
    matched_terms: list[str] = field(default_factory=list)
    below_threshold: bool = False
    reason: str = ""


class SemanticRetriever:
    def __init__(self, index: DocIndex | None = None, embedder=None):
        from ..providers import get_embedder

        self.index = index or DocIndex()
        self.embedder = embedder or get_embedder()
        if self.embedder.name != self.index.model:
            raise RuntimeError(
                f"index was built with {self.index.model!r} but the configured "
                f"embedder is {self.embedder.name!r}. Rebuild the index or set "
                "GAIA_RAG_EMBED_MODEL to match -- mixing embedding spaces "
                "silently destroys retrieval quality."
            )
        # BM25 is built over the raw chunk text, without the embedding
        # task prefix, which would otherwise add a term to every document.
        self.bm25 = BM25(
            [f"{c.title}\n\n{c.text}" for c in self.index.chunks]
        )

    def _ensure_subset_coverage(
        self, question: str, kept_ids: list[int], dense, k: int
    ) -> list[int]:
        """Guarantee a document for every subset the question names.

        Ranking alone does not do this, and the gap produces the worst
        class of error. Asked to compare GCNS parallax distances with RR
        Lyrae period-luminosity distances, retrieval returned the GCNS
        caveat and the *Cepheid* caveat but not the RR Lyrae one --
        "period-luminosity" is Cepheid vocabulary, so it dominated the
        lexical score. The generator then stated the Cepheid uncertainty
        (10%) as the RR Lyrae figure (18%), and the entailment judge
        called it supported, because the document it cited really does
        say 10%.

        A missing document is worse than an irrelevant one: the model
        fills the gap from whatever neighbouring subset was retrieved.
        Since the named subsets are already recovered deterministically,
        coverage is guaranteed rather than hoped for.
        """
        from ..verify.subject import detect_subsets

        wanted = detect_subsets(question)
        if len(wanted) < 2:
            # With one subject there is nothing to confuse it with, and
            # ranking already favours the right document.
            return kept_ids

        chunks = self.index.chunks
        covered = {
            subset
            for subset in wanted
            for i in kept_ids
            if subset in chunks[i].doc_id
        }
        for subset in sorted(wanted - covered):
            candidates = [
                i
                for i in range(len(chunks))
                if subset in chunks[i].doc_id and i not in kept_ids
            ]
            if not candidates:
                continue
            best = max(candidates, key=lambda i: float(dense[i]))
            if len(kept_ids) < k:
                kept_ids.append(best)
            else:
                # Replace the weakest slot rather than growing the prompt.
                kept_ids[-1] = best
        return kept_ids

    def search(self, question: str, k: int | None = None) -> SemanticResult:
        k = k or CONFIG.retrieval.top_k_docs
        chunks = self.index.chunks
        if not chunks:
            return SemanticResult(below_threshold=True, reason="empty document index")

        matched = self.bm25.specific_matched_terms(question)
        lexical = self.bm25.score(question)
        query_vector = self.embedder.embed([query_text(question)])[0]
        dense = self.index.vectors @ query_vector.astype(np.float32)

        best_dense = float(dense.max())
        best_lexical = float(lexical.max())

        # Out-of-domain gate. No query term occurs anywhere in the corpus
        # => the corpus has nothing to say, whatever the cosine says.
        if not matched or best_lexical <= 0.0:
            return SemanticResult(
                best_score=best_dense,
                best_dense=best_dense,
                best_lexical=best_lexical,
                matched_terms=matched,
                below_threshold=True,
                reason=(
                    "no distinctive term in the question occurs in the document "
                    f"corpus (nearest chunk by embedding only: {best_dense:.2f})"
                ),
            )

        pool = min(max(k * 3, 10), len(chunks))
        dense_rank = list(np.argsort(-dense)[:pool])
        lexical_rank = list(np.argsort(-lexical)[:pool])
        fused = reciprocal_rank_fusion([dense_rank, lexical_rank])

        # Keep only chunks that at least one retriever ranked highly,
        # then truncate. Filtering after truncation dropped correct
        # documents that sat just outside the fused top-k while being a
        # clear top hit for one retriever.
        top_dense = set(dense_rank[:k])
        top_lexical = set(lexical_rank[:k])
        ordered = sorted(fused.items(), key=lambda kv: -kv[1])
        candidates = [
            (i, score)
            for i, score in ordered
            if i in top_dense or i in top_lexical
        ]
        # Each retriever's single best hit is always included. Fusion is a
        # heuristic over two noisy rankings; an unambiguous top-1 from
        # either side is a stronger signal than its fused position, and
        # guaranteeing it makes retrieval failures impossible to blame on
        # the fusion constant.
        guaranteed = [i for i in (lexical_rank[0], dense_rank[0])]
        kept_ids: list[int] = []
        for i in guaranteed:
            if i not in kept_ids:
                kept_ids.append(i)
        for i, _score in candidates:
            if len(kept_ids) >= k:
                break
            if i not in kept_ids:
                kept_ids.append(i)
        kept_ids = self._ensure_subset_coverage(question, kept_ids, dense, k)
        kept = [(i, fused.get(i, 0.0)) for i in kept_ids[:k]]

        evidence = [
            chunk_to_evidence(chunks[i], float(dense[i])) for i, _score in kept
        ]
        return SemanticResult(
            evidence=evidence,
            best_score=float(kept[0][1]),
            best_dense=best_dense,
            best_lexical=best_lexical,
            matched_terms=matched,
            below_threshold=False,
        )
