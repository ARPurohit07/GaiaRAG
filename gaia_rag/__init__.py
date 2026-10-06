"""GaiaRAG: grounded question answering over Gaia catalogue data.

    from gaia_rag import GaiaRAG

    rag = GaiaRAG()
    answer = rag.ask("How were the Cepheid distances derived?")
    print(answer.outcome, answer.answer, answer.citations)

`GaiaRAG` is resolved lazily so that `import gaia_rag` does not require a
built index or a running Ollama: the ingest modules and the tests import
from this package too, and they must work before anything is built.
"""

from __future__ import annotations

__all__ = ["CONFIG", "GaiaRAG", "__version__"]

__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "GaiaRAG":
        from .generate.answer import GaiaRAG

        return GaiaRAG
    if name == "CONFIG":
        from .config import CONFIG

        return CONFIG
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
