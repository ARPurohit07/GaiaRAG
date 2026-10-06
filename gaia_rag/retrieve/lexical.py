"""BM25 lexical scoring over document chunks.

Dense retrieval alone measurably fails on this corpus. The documents are
few, short, and dominated by shared boilerplate ("Gaia", "distance",
"parallax", "catalogue"), so a 768-dimensional embedding of "how
reliable are RR Lyrae distances" lands nearest whichever chunk is most
generically about Gaia distances -- not the one chunk that names RR
Lyrae. Measured on this corpus, dense-only retrieval missed the correct
document for both "RR Lyrae reliability" and "Cepheid method".

What distinguishes those questions is a rare exact term, which is
precisely what an inverse-document-frequency weighting is good at. BM25
is also what makes the abstention gate trustworthy: an out-of-domain
question ("capital of France") shares no content term with any chunk and
scores a flat zero, whereas its cosine similarity sits at 0.53 -- close
enough to in-domain questions to be useless as a threshold.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")

#: Deliberately minimal. Domain words must survive, and an aggressive
#: list would strip terms like "no" that flip a caveat's meaning.
STOPWORDS = frozenset(
    """a about an and are as at be been but by can do does did for from
    get give has have how i if in into is it its just like me more most my
    of on or our so such tell than that the their then there these this
    those to us very was we were what when where which who why will with
    you your""".split()
)

#: A matched term is only treated as evidence of topical overlap if it
#: appears in at most this fraction of chunks. Boilerplate that occurs
#: everywhere ("distance", "catalogue") says nothing about whether the
#: corpus covers a question.
SPECIFIC_TERM_MAX_DF = 0.5


def stem(token: str) -> str:
    """Strip regular English plurals, and nothing else.

    Without this, "How were Cepheid distances derived?" failed to match
    the document titled "classical Cepheids": BM25 treats `cepheid` and
    `cepheids` as unrelated terms, and the one high-information term in
    the question was silently dropped. A full Porter stemmer would also
    work, but over-stems domain vocabulary ("variables" -> "variabl",
    "parallax" -> "parallax" but "Lyrae" -> "Lyra"), so the rules here
    stay deliberately narrow and reversible.

    The guards matter: "radius" must not become "radiu", and "mass" must
    not become "mas" -- which is a real and different unit here
    (milliarcseconds).
    """
    if len(token) <= 3 or not token.endswith("s"):
        return token
    if token.endswith(("ss", "us", "is", "as", "os")):
        return token
    if token.endswith("ies"):
        return token[:-3] + "y"
    return token[:-1]


def tokenize(text: str) -> list[str]:
    return [
        stem(token)
        for token in _TOKEN.findall((text or "").lower())
        if token not in STOPWORDS and len(token) > 1
    ]


class BM25:
    """Okapi BM25. Small corpus, so everything stays dense in memory."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(d) for d in documents]
        self.doc_len = np.array([len(t) for t in self.doc_tokens], dtype=np.float32)
        self.avg_len = float(self.doc_len.mean()) if len(self.doc_len) else 0.0
        self.n_docs = len(documents)

        self.term_freqs: list[Counter] = [Counter(t) for t in self.doc_tokens]
        df = Counter()
        for tokens in self.doc_tokens:
            df.update(set(tokens))
        self.doc_freq: dict[str, int] = dict(df)
        # Standard BM25 idf with the +0.5 smoothing, floored at a small
        # positive value so a term present in every document contributes
        # nothing rather than a negative score.
        self.idf = {
            term: max(
                math.log((self.n_docs - count + 0.5) / (count + 0.5) + 1.0), 1e-6
            )
            for term, count in df.items()
        }

    def score(self, query: str) -> np.ndarray:
        scores = np.zeros(self.n_docs, dtype=np.float32)
        if not self.n_docs:
            return scores
        for term in tokenize(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, freqs in enumerate(self.term_freqs):
                tf = freqs.get(term, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (
                    1.0 - self.b + self.b * self.doc_len[i] / (self.avg_len or 1.0)
                )
                scores[i] += idf * (tf * (self.k1 + 1.0)) / denom
        return scores

    def matched_terms(self, query: str) -> list[str]:
        """Query terms that appear anywhere in the corpus."""
        return [t for t in dict.fromkeys(tokenize(query)) if t in self.idf]

    def specific_matched_terms(self, query: str) -> list[str]:
        """Matched terms distinctive enough to indicate topical overlap.

        The abstention gate uses this rather than `matched_terms`. Bare
        presence is too weak a signal: "Tell me about quantum
        chromodynamics" matched this corpus on the word "about" alone and
        was served five Gaia documents. Requiring a term that occurs in
        only part of the corpus rejects it.
        """
        ceiling = max(1, int(self.n_docs * SPECIFIC_TERM_MAX_DF))
        return [
            t
            for t in self.matched_terms(query)
            if self.doc_freq.get(t, 0) <= ceiling
        ]


#: RRF smoothing constant. The usual default is 60, tuned for TREC-scale
#: runs of thousands of documents. On a 34-chunk corpus that value is far
#: larger than the ranks themselves, so 1/(k+rank) is nearly flat and
#: fusion stops discriminating: a chunk ranked 5th by both retrievers
#: outscores one ranked 1st by BM25 and absent from the dense top-k,
#: purely for appearing twice. That demoted the correct Cepheid document
#: out of the results while BM25 had it at rank 1. A smaller constant
#: keeps rank-1 genuinely ahead of rank-5.
RRF_K = 10


def reciprocal_rank_fusion(
    rankings: list[list[int]], k: int = RRF_K
) -> dict[int, float]:
    """Fuse ranked id lists by RRF: sum of 1/(k + rank).

    Chosen over a weighted sum of scores because BM25 and cosine live on
    incomparable scales, and any normalisation of them needs per-corpus
    tuning that would silently rot as documents change. RRF only uses
    rank order, so it has no scale parameter to get wrong.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return fused
