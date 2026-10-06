"""Run every test module.

    python scripts/run_tests.py

The suites are plain asserts with no pytest dependency, so this just
imports each module and calls its test functions. test_scope.py needs the
built catalogue; the others need nothing.
"""

from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODULES = [
    "test_units",
    "test_retrieval",
    "test_guard",
    "test_scope",
]


def load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "tests" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    total = failed = 0
    for name in MODULES:
        print(f"\n--- {name} ---")
        try:
            module = load(name)
        except Exception as exc:  # a module that will not import is a failure
            print(f"  ERROR importing: {exc}")
            failed += 1
            continue
        for attr in sorted(vars(module)):
            if not attr.startswith("test_"):
                continue
            fn = getattr(module, attr)
            if not callable(fn):
                continue
            total += 1
            try:
                fn()
                print(f"  PASS {attr}")
            except Exception:
                failed += 1
                print(f"  FAIL {attr}")
                traceback.print_exc(limit=2)

    print(f"\n{total - failed}/{total} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
