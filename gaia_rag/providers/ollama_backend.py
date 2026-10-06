"""Ollama-backed Embedder and LLM.

Uses the HTTP API directly rather than the ollama package so the only
hard dependency is the standard library plus numpy. Embeddings are
L2-normalised on the way out, which makes cosine similarity a dot
product everywhere downstream.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Sequence

import numpy as np

from ..config import CONFIG

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
# qwen3 and friends emit chain-of-thought in <think> blocks; strip it so
# JSON parsing and claim extraction see only the answer.
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def _post(url: str, payload: dict, timeout: int) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # surface Ollama's own message
        detail = exc.read().decode("utf-8", "replace")[:400]
        raise RuntimeError(f"ollama HTTP {exc.code} at {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"cannot reach ollama at {url} ({exc.reason}). Is `ollama serve` running?"
        ) from exc


def strip_reasoning(text: str) -> str:
    return _THINK.sub("", text).strip()


def extract_json(text: str) -> dict:
    """Best-effort JSON object extraction from a model response."""
    text = strip_reasoning(text)
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1)
    text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Fall back to the outermost brace-balanced span.
        start = text.find("{")
        if start == -1:
            raise ValueError(f"no JSON object in response: {text[:200]!r}")
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    parsed = json.loads(text[start : i + 1])
                    break
        else:
            raise ValueError(f"unbalanced JSON in response: {text[:200]!r}")
    if not isinstance(parsed, dict):
        raise ValueError(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed


class OllamaEmbedder:
    def __init__(self, model: str | None = None, host: str | None = None):
        self.name = model or CONFIG.models.embed_model
        self.host = (host or CONFIG.models.host).rstrip("/")
        self.timeout = CONFIG.models.timeout_s

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, CONFIG.models.embed_dim), dtype=np.float32)
        vectors = []
        for text in texts:
            data = _post(
                f"{self.host}/api/embeddings",
                {"model": self.name, "prompt": text},
                self.timeout,
            )
            vec = data.get("embedding")
            if not vec:
                raise RuntimeError(f"empty embedding from {self.name}")
            vectors.append(vec)
        arr = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


class OllamaLLM:
    """Chat completions, with extended thinking disabled by default.

    Disabling it is not a quality trade-off here; it is strictly better
    on both axes. Measured on this pipeline's planner call with qwen3:8b,
    one request took 131s with thinking on and 3.4s with it off -- and
    the thinking run returned worse output, inventing catalogue subset
    names ("Baade-Wesselink method accuracy") where the direct run
    correctly returned an empty list. Constrained JSON generation fights
    with a reasoning preamble: the model spends its budget reasoning in
    prose it is then not permitted to emit.

    A model that rejects the `think` field is retried without it, so this
    stays safe across the non-reasoning models in the same registry.
    """

    def __init__(self, model: str | None = None, host: str | None = None):
        self.name = model or CONFIG.models.gen_model
        self.host = (host or CONFIG.models.host).rstrip("/")
        self.timeout = CONFIG.models.timeout_s
        self._supports_think = True

    def _chat(self, payload: dict) -> dict:
        url = f"{self.host}/api/chat"
        if self._supports_think and CONFIG.models.disable_thinking:
            try:
                return _post(url, {**payload, "think": False}, self.timeout)
            except RuntimeError as exc:
                if "HTTP 400" not in str(exc):
                    raise
                # This model has no thinking mode to disable.
                self._supports_think = False
        return _post(url, payload, self.timeout)

    def _payload(self, system: str, user: str, temperature: float) -> dict:
        return {
            "model": self.name,
            "stream": False,
            "options": {"temperature": temperature},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        data = self._chat(self._payload(system, user, temperature))
        return strip_reasoning(data.get("message", {}).get("content", ""))

    def complete_json(
        self, system: str, user: str, *, temperature: float = 0.0
    ) -> dict:
        payload = self._payload(system, user, temperature)
        payload["format"] = "json"
        data = self._chat(payload)
        return extract_json(data.get("message", {}).get("content", ""))


def get_embedder() -> OllamaEmbedder:
    if CONFIG.models.backend != "ollama":
        raise NotImplementedError(
            f"backend {CONFIG.models.backend!r} not wired; implement the "
            "Embedder protocol in gaia_rag/providers/ and return it here."
        )
    return OllamaEmbedder()


def get_llm(role: str = "generator") -> OllamaLLM:
    if CONFIG.models.backend != "ollama":
        raise NotImplementedError(
            f"backend {CONFIG.models.backend!r} not wired; implement the "
            "LLM protocol in gaia_rag/providers/ and return it here."
        )
    model = (
        CONFIG.models.judge_model if role == "judge" else CONFIG.models.gen_model
    )
    return OllamaLLM(model)
