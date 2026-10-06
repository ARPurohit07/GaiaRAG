"""Central configuration. Everything a run depends on is resolved here.

Paths default to this repo; model choices default to the local Ollama
install. Override any value with an environment variable (GAIA_RAG_*)
so eval runs stay reproducible without editing code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def _env(name: str, default: str) -> str:
    return os.environ.get(f"GAIA_RAG_{name}", default)


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(f"GAIA_RAG_{name}", default))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(f"GAIA_RAG_{name}", default))


@dataclass(frozen=True)
class Paths:
    root: Path = ROOT
    raw: Path = DATA / "raw"
    processed: Path = DATA / "processed"
    docs: Path = DATA / "docs"
    duckdb: Path = DATA / "processed" / "gaia.duckdb"
    doc_index: Path = DATA / "processed" / "doc_index.npz"
    doc_store: Path = DATA / "processed" / "doc_store.jsonl"
    eval_out: Path = DATA / "processed" / "eval"

    # Source catalogue produced by the Galaxy project. Real Gaia data,
    # already distance-resolved. Override if Galaxy lives elsewhere.
    galaxy_catalogue: Path = Path(
        _env(
            "GALAXY_CSV",
            r"C:\Users\rohan\OneDrive\Desktop\Galaxy\web\data\hygdata_web.csv",
        )
    )
    galaxy_sources: Path = Path(
        _env(
            "GALAXY_SOURCES",
            r"C:\Users\rohan\OneDrive\Desktop\Galaxy\web\data\SOURCES.txt",
        )
    )
    galaxy_tools: Path = Path(
        _env(
            "GALAXY_TOOLS",
            r"C:\Users\rohan\OneDrive\Desktop\Galaxy\web\tools",
        )
    )


@dataclass(frozen=True)
class Models:
    backend: str = _env("BACKEND", "ollama")
    host: str = _env("OLLAMA_HOST", "http://127.0.0.1:11434")
    embed_model: str = _env("EMBED_MODEL", "nomic-embed-text")
    gen_model: str = _env("GEN_MODEL", "qwen3:8b")
    # The judge is deliberately separable from the generator: scoring your
    # own output with the same weights inflates faithfulness.
    judge_model: str = _env("JUDGE_MODEL", "qwen3:8b")
    embed_dim: int = _env_int("EMBED_DIM", 768)
    # Reasoning models are both slower and worse at the constrained-JSON
    # calls this pipeline makes. See OllamaLLM for the measurement.
    disable_thinking: bool = _env("DISABLE_THINKING", "1") == "1"
    gen_temperature: float = _env_float("GEN_TEMP", 0.0)
    judge_temperature: float = _env_float("JUDGE_TEMP", 0.0)
    timeout_s: int = _env_int("TIMEOUT_S", 600)


@dataclass(frozen=True)
class Retrieval:
    top_k_docs: int = _env_int("TOP_K_DOCS", 6)
    max_rows: int = _env_int("MAX_ROWS", 40)
    chunk_tokens: int = _env_int("CHUNK_TOKENS", 220)
    chunk_overlap: int = _env_int("CHUNK_OVERLAP", 40)
    # Below this best-document similarity, semantic evidence is treated as
    # absent rather than weak -- the abstain path, not a guess.
    min_doc_similarity: float = _env_float("MIN_DOC_SIM", 0.45)


@dataclass(frozen=True)
class Verification:
    # Relative tolerance when matching a number in the answer to a number
    # in the evidence. Catalogue values get rounded in prose ("1,300 pc"
    # for 1300.4), so an exact match is too strict.
    numeric_rel_tol: float = _env_float("NUM_REL_TOL", 0.02)
    numeric_abs_tol: float = _env_float("NUM_ABS_TOL", 0.01)
    # An answer failing either gate is downgraded to an abstention.
    #
    # 0.65, not 0.67: two supported claims out of three is 0.6667, which
    # a 0.67 threshold rejects by a rounding margin. That knife edge
    # withheld an otherwise sound answer. A gate should not turn on the
    # third decimal place of a three-claim answer.
    min_claim_support: float = _env_float("MIN_CLAIM_SUPPORT", 0.65)
    require_numeric_grounding: bool = _env("REQUIRE_NUM_GROUND", "1") == "1"


@dataclass(frozen=True)
class Config:
    paths: Paths = field(default_factory=Paths)
    models: Models = field(default_factory=Models)
    retrieval: Retrieval = field(default_factory=Retrieval)
    verification: Verification = field(default_factory=Verification)

    def ensure_dirs(self) -> None:
        for p in (
            self.paths.raw,
            self.paths.processed,
            self.paths.docs,
            self.paths.eval_out,
        ):
            p.mkdir(parents=True, exist_ok=True)


CONFIG = Config()
