"""Ask the Gaia RAG a question.

    python scripts/ask.py "How far away is the most distant Cepheid?"
    python scripts/ask.py --evidence "How reliable are RR Lyrae distances?"
    python scripts/ask.py            # interactive
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gaia_rag.generate.answer import GaiaRAG  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Query the Gaia RAG.")
    parser.add_argument("question", nargs="*", help="the question to ask")
    parser.add_argument(
        "--evidence", action="store_true", help="print the retrieved evidence"
    )
    args = parser.parse_args()

    rag = GaiaRAG()
    try:
        if args.question:
            answer = rag.ask(" ".join(args.question))
            print(answer.report(show_evidence=args.evidence))
            return 0 if answer.served else 2

        print("Gaia RAG. Blank line or Ctrl-C to exit.\n")
        while True:
            try:
                question = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not question:
                return 0
            print()
            print(rag.ask(question).report(show_evidence=args.evidence))
            print()
    finally:
        rag.close()


if __name__ == "__main__":
    raise SystemExit(main())
