"""A naive chunk-and-embed RAG, for comparison on the same gold set.

    python -m gaia_rag.eval.baseline            # full gold set
    python -m gaia_rag.eval.baseline --limit 10

Without this, every claim about the main pipeline's design is an
anecdote. "Numbers do not embed" and "abstention has to be structural"
are assertions until something measures what happens without them.

The baseline is deliberately **steel-manned**, because beating a
strawman would prove nothing:

* It gets the *same* generator model, the same `k`, and the same
  embedding model.
* Its index contains the same document chunks *plus* the catalogue rows
  rendered as text -- including the full provenance sentences the main
  pipeline builds. A real naive implementation would probably have
  thinner row text than this.
* Its index is **seeded with every row any gold case needs**, so a
  failure is never caused by the row being absent. Coverage is handed to
  it; only the method differs.

What it does *not* have is the part under test: structured SQL
retrieval, claim-level citation, the numeric/subject/scope verifiers, and
any route to abstention. It answers every question from its top-k
chunks, which is what the overwhelming majority of RAG systems do.

The comparison therefore isolates method rather than access or model
quality.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from gaia_rag.config import CONFIG  # noqa: E402
from gaia_rag.ingest.index import DocIndex  # noqa: E402
from gaia_rag.providers import get_embedder, get_llm  # noqa: E402
from gaia_rag.retrieve.evidence import Evidence, row_to_evidence  # noqa: E402
from gaia_rag.verify.claims import ClaimReport  # noqa: E402
from gaia_rag.verify.numeric import NumericReport, verify_numbers  # noqa: E402
from gaia_rag.verify.subject import SubjectReport  # noqa: E402

NAIVE_SYSTEM = """You are a helpful assistant answering questions about a Gaia \
star catalogue. Use the provided context to answer the question as helpfully and \
specifically as you can. Give concrete numbers where possible."""

#: How many random catalogue rows to put in the baseline index, on top of
#: every row the gold set needs. Enough to act as realistic distractors
#: without making the embedding pass take all day.
SAMPLE_ROWS = 2000


@dataclass
class _Outcome:
    value: str


@dataclass
class NaiveAnswer:
    """Shaped like RAGAnswer so the same scorer can read it."""

    question: str
    answer: str
    evidence: list[Evidence] = field(default_factory=list)
    outcome: _Outcome = field(default_factory=lambda: _Outcome("answered"))
    claims: list[dict] = field(default_factory=list)
    numeric: NumericReport | None = None
    claim_report: ClaimReport | None = None
    subject_report: SubjectReport | None = None
    bundle: None = None

    @property
    def served(self) -> bool:
        # A naive RAG has no abstention path: it always answers.
        return bool(self.answer.strip())

    @property
    def citations(self) -> list[str]:
        return []


class NaiveRAG:
    def __init__(self, gold_cases: list[dict], seed: int = 0):
        self.embedder = get_embedder()
        self.llm = get_llm("generator")
        self.k = CONFIG.retrieval.top_k_docs

        doc_index = DocIndex()
        if self.embedder.name != doc_index.model:
            raise RuntimeError("embedder/index mismatch; rebuild the index")

        texts: list[str] = []
        self.items: list[Evidence] = []

        # Document chunks, reusing the embeddings already on disk.
        doc_vectors = doc_index.vectors
        for chunk in doc_index.chunks:
            self.items.append(
                Evidence(
                    evidence_id=chunk.chunk_id,
                    kind="doc",
                    text=f"{chunk.title}: {chunk.text}",
                    citation=chunk.citation,
                    metadata={"doc_id": chunk.doc_id},
                )
            )

        # Catalogue rows as text: the gold-referenced ones plus a sample.
        rows = self._load_rows(gold_cases, seed)
        for ev in rows:
            self.items.append(ev)
            texts.append(ev.text)

        row_vectors = (
            self.embedder.embed(texts)
            if texts
            else np.zeros((0, doc_vectors.shape[1]), dtype=np.float32)
        )
        self.vectors = np.vstack([doc_vectors, row_vectors])
        if self.vectors.shape[0] != len(self.items):
            raise RuntimeError("baseline index/item mismatch")

    def _load_rows(self, gold_cases: list[dict], seed: int) -> list[Evidence]:
        con = duckdb.connect(str(CONFIG.paths.duckdb), read_only=True)
        try:
            provenance = {
                (r[0], r[1]): {
                    "status": r[2],
                    "method": r[3],
                    "rel_uncertainty": r[4],
                    "caveat": r[5],
                }
                for r in con.execute(
                    "SELECT subset, field, status, method, rel_uncertainty, "
                    "caveat FROM field_provenance"
                ).fetchall()
            }
            cols = (
                "row_id, evidence_id, subset, is_gaia, label, source_id, "
                "proper_name, constellation, spectral_type, ra_deg, dec_deg, "
                "dist_pc, dist_ly, dist_kpc, color_bp_rp, lum_lsun"
            )

            wanted = {
                pattern
                for case in gold_cases
                for pattern in (case.get("expected_evidence") or [])
                if pattern.startswith("row:")
            }
            seen: set[str] = set()
            out: list[Evidence] = []

            # Every row the gold set needs, so coverage is never the cause
            # of a baseline failure.
            for pattern in sorted(wanted):
                cur = con.execute(
                    f"SELECT {cols} FROM stars WHERE evidence_id = ?", [pattern]
                )
                names = [d[0] for d in cur.description]
                for raw in cur.fetchall():
                    row = dict(zip(names, raw))
                    if row["evidence_id"] in seen:
                        continue
                    seen.add(row["evidence_id"])
                    out.append(row_to_evidence(row, provenance))

            # Plus a deterministic pseudo-random sample as distractors.
            # Ordering by a hash is reproducible across runs; DuckDB's
            # USING SAMPLE does not accept REPEATABLE in this position.
            cur = con.execute(
                f"SELECT {cols} FROM stars "
                f"ORDER BY hash(row_id + {seed}) LIMIT {SAMPLE_ROWS}"
            )
            names = [d[0] for d in cur.description]
            for raw in cur.fetchall():
                row = dict(zip(names, raw))
                if row["evidence_id"] in seen:
                    continue
                seen.add(row["evidence_id"])
                out.append(row_to_evidence(row, provenance))
            return out
        finally:
            con.close()

    def ask(self, question: str) -> NaiveAnswer:
        qv = self.embedder.embed([question])[0].astype(np.float32)
        sims = self.vectors @ qv
        k = min(self.k, len(self.items))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        picked = [self.items[i] for i in top]

        context = "\n\n".join(f"- {e.text}" for e in picked)
        user = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        try:
            text = self.llm.complete(NAIVE_SYSTEM, user, temperature=0.0)
        except (RuntimeError, ValueError) as exc:
            text = f"(generation failed: {exc})"
        # Scored with the same deterministic checker as the main
        # pipeline: do the numbers it states trace to the context it was
        # given? Leaving this at the default 1.0 would flatter the
        # baseline, and this comparison is the whole point.
        return NaiveAnswer(
            question=question,
            answer=text,
            evidence=picked,
            numeric=verify_numbers(text, picked, question),
        )


def main() -> int:
    from gaia_rag.eval.metrics import Summary, score_case
    from gaia_rag.eval.run_eval import load_goldset

    parser = argparse.ArgumentParser(description="Run the naive RAG baseline.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--kind", default=None)
    args = parser.parse_args()

    cases = load_goldset()
    if args.kind:
        cases = [c for c in cases if c.get("kind") == args.kind]
    if args.limit:
        cases = cases[: args.limit]

    CONFIG.ensure_dirs()
    print(f"building baseline index ({SAMPLE_ROWS} sampled rows + gold rows)...")
    rag = NaiveRAG(cases)
    print(f"  {len(rag.items)} items indexed\n")

    results = []
    records = []
    started = time.time()
    for i, case in enumerate(cases, 1):
        t0 = time.time()
        print(f"[{i}/{len(cases)}] {case['id']} ... ", end="", flush=True)
        answer = rag.ask(case["question"])
        result = score_case(case, answer, 0.0)
        results.append(result)
        print(f"{'PASS' if result.passed else 'FAIL'}  ({time.time() - t0:.0f}s)")
        for f in result.failures:
            print(f"        - {f}")
        records.append(
            {
                "case": case,
                "answer": answer.answer,
                "retrieved": [e.evidence_id for e in answer.evidence],
                "failures": result.failures,
            }
        )

    summary = Summary(results)
    print()
    print(summary.render())
    print(f"total time: {time.time() - started:.0f}s")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = CONFIG.paths.eval_out / f"baseline-{stamp}.json"
    out.write_text(
        json.dumps(
            {
                "system": "naive dense-only RAG, no verification, no abstention",
                "sample_rows": SAMPLE_ROWS,
                "summary": summary.as_dict(),
                "cases": records,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
