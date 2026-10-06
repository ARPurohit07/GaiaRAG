"""Thin provider interfaces.

The pipeline only ever talks to these two protocols, so swapping Ollama
for a hosted model is a config change, not a rewrite. Keeping the judge
behind the same interface lets an eval run use a stronger judge than the
generator without touching the metrics code.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return L2-normalised embeddings, shape (len(texts), dim)."""
        ...


class LLM(Protocol):
    name: str

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        ...

    def complete_json(
        self, system: str, user: str, *, temperature: float = 0.0
    ) -> dict:
        """Complete and parse a JSON object, tolerating fenced output."""
        ...
