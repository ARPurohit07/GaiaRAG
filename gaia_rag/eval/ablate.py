"""Ablations: measure what each component is actually worth.

    python -m gaia_rag.eval.ablate                  # every ablation
    python -m gaia_rag.eval.ablate --only no_numeric_gate
    python -m gaia_rag.eval.ablate --list

Every design claim in this project -- that numbers need an arithmetic
check, that lexical retrieval is load-bearing, that abstention has to be
structural -- is an anecdote until something measures the system without
it. This disables one component at a time and re-runs the gold cases
that component exists to handle.

Each ablation runs only the case kinds it could plausibly affect, and the
full system is re-run on that same subset for comparison, so the delta is
measured on identical cases rather than against a global average. That
keeps a full sweep to minutes rather than hours.

Components are disabled by monkeypatching from here rather than by
threading flags through the pipeline. An eval harness is the right place
for that: production code stays free of switches that exist only to be
turned off, and an ablation cannot accidentally ship.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from gaia_rag.config import CONFIG  # noqa: E402
from gaia_rag.eval.metrics import Summary, score_case  # noqa: E402
from gaia_rag.eval.run_eval import load_goldset  # noqa: E402
from gaia_rag.generate.answer import GaiaRAG  # noqa: E402


# --- ablation implementations -----------------------------------------
#
# Each returns a list of (module, attribute, original) so it can be undone.


def _patch(target, attr: str, value):
    original = getattr(target, attr)
    setattr(target, attr, value)
    return (target, attr, original)


def ablate_dense_only():
    """Remove BM25 from ranking, keeping the out-of-domain gate."""
    import numpy as np

    from gaia_rag.retrieve import lexical

    def zero_score(self, query):
        return np.zeros(self.n_docs, dtype=np.float32)

    return [_patch(lexical.BM25, "score", zero_score)]


def ablate_no_ood_gate():
    """Never treat a question as out of domain."""
    from gaia_rag.retrieve import lexical

    return [
        _patch(lexical.BM25, "specific_matched_terms", lambda self, q: ["_forced"])
    ]


def ablate_no_numeric_gate():
    """Stop requiring that numbers trace to evidence."""
    from gaia_rag.verify import guard

    relaxed = dataclasses.replace(
        CONFIG,
        verification=dataclasses.replace(
            CONFIG.verification, require_numeric_grounding=False
        ),
    )
    return [_patch(guard, "CONFIG", relaxed)]


def ablate_no_subject_gate():
    """Stop checking that a claim's numbers come from its own subset."""
    from gaia_rag.generate import answer as answer_mod
    from gaia_rag.verify.subject import SubjectReport

    return [_patch(answer_mod, "check_claims", lambda claims, evidence: SubjectReport())]


def ablate_no_scope_guard():
    """Stop withdrawing rows for out-of-schema quantities."""
    from gaia_rag.retrieve import router
    from gaia_rag.verify.scope import ScopeReport

    return [_patch(router, "check_question", lambda question: ScopeReport())]


def ablate_no_abstention():
    """Serve whatever the generator produced, always."""
    from gaia_rag.generate import answer as answer_mod
    from gaia_rag.verify.guard import GuardDecision, Outcome

    def always_answer(
        *,
        model_said_sufficient,
        model_answer,
        model_missing,
        has_evidence,
        numeric,
        claims,
        subject=None,
        retrieval_notes=None,
    ):
        return GuardDecision(
            outcome=Outcome.ANSWERED,
            answer=model_answer or "(no answer produced)",
            reasons=[],
            numeric_score=numeric.score,
            claim_score=claims.score,
        )

    return [_patch(answer_mod, "decide", always_answer)]


def ablate_no_unconstrained_guard():
    """Allow an arbitrary LIMIT n sample to be used as evidence."""
    from gaia_rag.retrieve import structured

    original = structured.StructuredRetriever.run

    def run_without_guard(self, spec):
        # Give the spec a harmless ordering so the unconstrained branch
        # does not trigger, mimicking a pipeline that never had the check.
        if (
            spec.intent != "aggregate"
            and not spec.name
            and not spec.filters
            and not spec.order_by
        ):
            spec.order_by = {"field": "row_id", "dir": "asc"}
        return original(self, spec)

    return [_patch(structured.StructuredRetriever, "run", run_without_guard)]


#: name -> (factory, case kinds it could affect, one-line rationale)
ABLATIONS: dict[str, tuple] = {
    "dense_only": (
        ablate_dense_only,
        ("method", "selection", "comparison", "trap_wrong_uncertainty"),
        "BM25 removed from ranking; dense retrieval alone",
    ),
    "no_ood_gate": (
        ablate_no_ood_gate,
        ("out_of_domain",),
        "out-of-domain questions are never rejected on lexical grounds",
    ),
    "no_numeric_gate": (
        ablate_no_numeric_gate,
        ("numeric", "unit_conversion", "aggregate"),
        "numbers no longer have to trace to evidence",
    ),
    "no_subject_gate": (
        ablate_no_subject_gate,
        ("trap_wrong_uncertainty", "trap_attribution", "comparison"),
        "claims may take numbers from another subset's evidence",
    ),
    "no_scope_guard": (
        ablate_no_scope_guard,
        ("unanswerable_in_corpus",),
        "row evidence is kept for out-of-schema quantities",
    ),
    "no_abstention": (
        ablate_no_abstention,
        ("out_of_domain", "unanswerable_in_corpus", "trap_attribution"),
        "every generated answer is served, whatever verification said",
    ),
    "no_unconstrained_guard": (
        ablate_no_unconstrained_guard,
        ("method", "trap_wrong_uncertainty"),
        "an arbitrary row sample may be used as evidence",
    ),
}


def _recompute_verdicts(record: dict) -> tuple[bool, bool]:
    """Full-system pass / answered-wrong from a stored eval record.

    Newer runs store both explicitly; older files predate that, so they
    are derived from the fields that were saved.
    """
    if "passed" in record and "answered_wrong" in record:
        return bool(record["passed"]), bool(record["answered_wrong"])
    r = record["result"]
    if r["served"] and not r["should_answer"]:
        wrong = True
    elif not r["served"]:
        wrong = False
    else:
        wrong = not (
            r["required_numbers_present"]
            and r["forbidden_numbers_absent"]
            and r["required_mentions_present"]
            and r["forbidden_mentions_absent"]
        )
    return (bool(r["abstention_correct"]) and not wrong), wrong


def load_full_reference(path: Path | None = None) -> dict[str, tuple[bool, bool]]:
    """Per-case full-system verdicts from the newest complete eval run.

    Reusing these instead of re-running the unablated system on every
    subset halves a full sweep. It is only valid because the ablations
    run the same cases against the same build -- if the pipeline changes,
    the reference run has to be redone, which `--full` makes explicit.
    """
    if path is None:
        # Pick the newest *complete* run. Targeted `--only` invocations
        # also write eval files, and the newest one is often a single
        # case, which would leave almost nothing to reuse.
        best: tuple[int, Path] | None = None
        for candidate in sorted(CONFIG.paths.eval_out.glob("eval-*.json")):
            try:
                n = json.loads(candidate.read_text(encoding="utf-8"))["summary"][
                    "cases"
                ]
            except (OSError, ValueError, KeyError):
                continue
            # >= so that, among equally complete runs, the newest wins.
            if best is None or n >= best[0]:
                best = (n, candidate)
        if best is None:
            return {}
        path = best[1]
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for record in data.get("cases", []):
        passed, wrong = _recompute_verdicts(record)
        out[record["case"]["id"]] = (passed, wrong)
    print(f"  full-system reference: {path.name} ({len(out)} cases)")

    # A reference run older than the code it is meant to describe makes
    # every delta meaningless, and the failure is silent -- the numbers
    # still print. Easy to do: the newest complete run on disk is often
    # the one from before the fix that prompted the ablation.
    # Only pipeline modules count. The eval package itself does not
    # affect answers, and including it made the guard cry wolf: writing
    # the RAGAS harness tripped it on an ablation whose reference run was
    # in fact current. A warning that fires when nothing is wrong gets
    # ignored when something is.
    package = Path(__file__).resolve().parent.parent
    newest_source = max(
        (
            p.stat().st_mtime
            for p in package.rglob("*.py")
            if "eval" not in p.relative_to(package).parts
        ),
        default=0.0,
    )
    if newest_source > path.stat().st_mtime:
        print(
            "  WARNING: pipeline source is NEWER than this reference run.\n"
            "           The full-system column describes an older build, so\n"
            "           the deltas below are not trustworthy. Re-run\n"
            "           `python -m gaia_rag.eval.run_eval` first, or pass\n"
            "           --full with a run from the current build."
        )
    return out


def _run(cases: list[dict], label: str) -> Summary:
    rag = GaiaRAG()
    results = []
    try:
        for i, case in enumerate(cases, 1):
            print(f"    [{i}/{len(cases)}] {case['id']} ... ", end="", flush=True)
            answer = rag.ask(case["question"])
            result = score_case(case, answer, 0.0)
            results.append(result)
            print("PASS" if result.passed else "FAIL")
    finally:
        rag.close()
    return Summary(results)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run component ablations.")
    parser.add_argument("--only", help="run a single ablation by name")
    parser.add_argument("--list", action="store_true", help="list ablations")
    parser.add_argument(
        "--full",
        default=None,
        help="eval-*.json whose full-system verdicts to reuse "
        "(default: newest); must be from the current build",
    )
    args = parser.parse_args()

    if args.list:
        for name, (_f, kinds, why) in ABLATIONS.items():
            print(f"{name:24s} {why}")
            print(f"{'':24s} kinds: {', '.join(kinds)}")
        return 0

    names = [args.only] if args.only else list(ABLATIONS)
    for name in names:
        if name not in ABLATIONS:
            print(f"unknown ablation {name!r}; --list to see them")
            return 1

    all_cases = load_goldset()
    CONFIG.ensure_dirs()
    reference = load_full_reference(Path(args.full) if args.full else None)
    rows = []

    for name in names:
        factory, kinds, why = ABLATIONS[name]
        cases = [c for c in all_cases if c.get("kind") in kinds]
        if not cases:
            print(f"-- {name}: no matching cases, skipped")
            continue

        print(f"\n=== {name} ===")
        print(f"    {why}")
        print(f"    {len(cases)} case(s) of kind: {', '.join(sorted(kinds))}")

        missing = [c["id"] for c in cases if c["id"] not in reference]
        if missing:
            print(f"  full system (not in reference: {len(missing)} case(s)):")
            baseline = _run([c for c in cases if c["id"] in set(missing)], "full")
            full_pass = sum(r.passed for r in baseline.results)
            full_wrong = sum(r.answered_wrong for r in baseline.results)
            n_full = len(baseline.results)
            for c in cases:
                if c["id"] in reference:
                    p, w = reference[c["id"]]
                    full_pass += p
                    full_wrong += w
                    n_full += 1
        else:
            print("  full system: reused from reference run")
            verdicts = [reference[c["id"]] for c in cases]
            full_pass = sum(p for p, _w in verdicts)
            full_wrong = sum(w for _p, w in verdicts)
            n_full = len(verdicts)

        print(f"  ablated ({name}):")
        patches = factory()
        try:
            ablated = _run(cases, name)
        finally:
            for target, attr, original in patches:
                setattr(target, attr, original)

        a = ablated.as_dict()
        b_pass = full_pass / n_full if n_full else 0.0
        b_wrong = full_wrong / n_full if n_full else 0.0
        rows.append(
            {
                "ablation": name,
                "rationale": why,
                "cases": len(cases),
                "full_pass": b_pass,
                "ablated_pass": a["pass_rate"],
                "full_wrong": b_wrong,
                "ablated_wrong": a["answered_wrong_rate"],
            }
        )
        print(
            f"  -> pass {b_pass:.0%} -> {a['pass_rate']:.0%}   "
            f"answered-wrong {b_wrong:.0%} -> {a['answered_wrong_rate']:.0%}"
        )

    print("\n" + "=" * 78)
    print("ABLATION SUMMARY (full system -> component removed)")
    print("=" * 78)
    print(f"{'component removed':24s} {'n':>3s}  {'pass':>13s}  {'answered-wrong':>16s}")
    for r in rows:
        print(
            f"{r['ablation']:24s} {r['cases']:>3d}  "
            f"{r['full_pass']:>5.0%} -> {r['ablated_pass']:>5.0%}  "
            f"{r['full_wrong']:>7.0%} -> {r['ablated_wrong']:>6.0%}"
        )
    print("=" * 78)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = CONFIG.paths.eval_out / f"ablation-{stamp}.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
