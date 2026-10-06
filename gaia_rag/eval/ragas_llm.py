"""A RAGAS LLM backed directly by Ollama, with thinking disabled.

This exists for one measured reason. The `langchain-ollama` version that
RAGAS 0.2.x resolves cannot disable a reasoning model's thinking, and on
this machine one qwen3:8b call takes about 131 seconds with thinking on
against 3.4 seconds with it off. RAGAS issues several calls per sample
per metric, so the difference is between a run of under an hour and one
of several hours.

Going straight to Ollama's HTTP API also means RAGAS judges through the
*same* provider and the same settings as the pipeline it is scoring, so
a disagreement between RAGAS's numbers and this project's own cannot be
explained away by different inference configuration.

Runs inside `.venv-ragas`, which is why it re-implements the few lines of
HTTP rather than importing the project's provider: the RAGAS venv does
not have the project on its path by default, and keeping this file
self-contained avoids a cross-environment import.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import typing as t
import urllib.error
import urllib.request

from langchain_core.outputs import Generation, LLMResult
from ragas.llms.base import BaseRagasLLM

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


class OllamaRagasLLM(BaseRagasLLM):
    """Minimal synchronous Ollama chat client in RAGAS's LLM interface."""

    def __init__(
        self,
        model: str = "qwen3:8b",
        # A judge call takes 8-20s here with thinking disabled, so a few
        # minutes is already generous. The first attempt at this run used
        # 900s and hung for 40 minutes on a single dead connection,
        # accumulating 12 CPU-seconds and never erroring -- almost
        # certainly Ollama idling out mid-run, which it has done
        # repeatedly. A tight timeout plus one retry turns that into a
        # brief pause instead of a silent stall.
        host: str = "http://127.0.0.1:11434",
        timeout: int = 240,
        disable_thinking: bool = True,
        retries: int = 1,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.disable_thinking = disable_thinking
        self.retries = retries
        self._supports_think = True
        self.calls = 0
        self.retried = 0

    # --- RAGAS interface ------------------------------------------------

    def generate_text(
        self,
        prompt,
        n: int = 1,
        temperature: float | None = None,
        stop: t.Optional[t.List[str]] = None,
        callbacks=None,
    ) -> LLMResult:
        text = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
        # Temperature 0 for reproducibility: RAGAS defaults to 1e-8 for
        # the judge, and a scored run should be repeatable.
        out = self._chat(text, 0.0 if temperature is None else temperature, stop)
        return LLMResult(generations=[[Generation(text=out)] * max(1, n)])

    async def agenerate_text(
        self,
        prompt,
        n: int = 1,
        temperature: float | None = None,
        stop: t.Optional[t.List[str]] = None,
        callbacks=None,
    ) -> LLMResult:
        """Run the blocking HTTP call in a thread.

        This must not block, and getting it wrong cost two full runs.
        RAGAS's executor schedules every metric call as a coroutine on a
        single event loop (`asyncio.as_completed`, 16 workers, nest_asyncio,
        and no thread offloading of its own). Calling urllib directly
        inside the coroutine blocked that loop, and because RAGAS's
        `timeout=180` is an asyncio timeout living on the same loop, it
        could never fire either. The result was a process that hung
        indefinitely while using no CPU -- which looks exactly like a
        slow run until you measure CPU and find it flat.

        `to_thread` keeps the loop responsive, so RAGAS's own timeout and
        retry logic work as designed.
        """
        text = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
        out = await asyncio.to_thread(
            self._chat, text, 0.0 if temperature is None else temperature, stop
        )
        return LLMResult(generations=[[Generation(text=out)] * max(1, n)])

    def is_finished(self, response: LLMResult) -> bool:
        return True

    # --- transport ------------------------------------------------------

    def _chat(self, user: str, temperature: float, stop) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": temperature},
            "messages": [{"role": "user", "content": user}],
        }
        if stop:
            payload["options"]["stop"] = list(stop)

        last: Exception | None = None
        for attempt in range(self.retries + 1):
            if attempt:
                self.retried += 1
                # A fresh request also re-warms a model Ollama unloaded.
                time.sleep(2.0)
            try:
                if self._supports_think and self.disable_thinking:
                    try:
                        return self._post({**payload, "think": False})
                    except RuntimeError as exc:
                        if "HTTP 400" not in str(exc):
                            raise
                        self._supports_think = False
                return self._post(payload)
            except (RuntimeError, TimeoutError, OSError) as exc:
                last = exc
        raise RuntimeError(f"ollama call failed after retries: {last}")

    def _post(self, payload: dict) -> str:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"ollama HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"cannot reach ollama: {exc.reason}") from exc
        self.calls += 1
        content = data.get("message", {}).get("content", "")
        return _THINK.sub("", content).strip()
