"""Score this pipeline's answers with RAGAS, as an outside opinion.

    .venv-ragas/Scripts/python.exe -m gaia_rag.eval.ragas_eval --limit 12
    .venv-ragas/Scripts/python.exe -m gaia_rag.eval.ragas_eval --all

Runs in its own virtualenv (`.venv-ragas`). RAGAS pulls langchain and a
large dependency tree, and the project's own pipeline deliberately has
almost no dependencies, so the two are kept apart.

Why bother, given this project already measures faithfulness:

* It is **someone else's implementation and prompts**. That removes the
  "you wrote both the system and the metric" objection, which is the
  fair criticism of any self-reported faithfulness number.

What it does *not* do, and should not be presented as doing:

* It is **not an independent method**. RAGAS faithfulness decomposes an
  answer into statements and LLM-judges entailment against the context --
  structurally the same approach as this project's claim judge. The
  deterministic numeric check has no RAGAS equivalent, and that is the
  check doing the real work on catalogue data.
* It **cannot see this system's main property**. RAGAS has no notion of a
  correct refusal; faithfulness over "I can't answer this from the
  catalogue" is degenerate. Abstention cases are therefore excluded, so
  RAGAS says nothing about the behaviour that most of the gold set
  exists to verify.
* With a small local judge the scores are **indicative, not precise**.
  RAGAS is known to be sensitive to judge quality.

Answers are read from a completed `eval-*.json` rather than regenerated,
so RAGAS scores exactly the outputs this project's own metrics scored --
the comparison is on identical text, not on a fresh sample.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

EVAL_DIR = ROOT / "data" / "processed" / "eval"

# Independent reference answers, written from the verified gold cases and
# catalogue queries. The gold set's `note` is a test-maintainer note: many
# notes deliberately name WRONG values, SQL, and regression history. Using
# those notes as RAGAS ground truth makes context recall measure whether the
# retriever found the distractors, not whether it found the answer.
REFERENCES = {
    "reliability_rrlyrae": "RR Lyrae distances are derived from an assumed absolute magnitude with colour-based reddening and are accurate to roughly 15-20%.",
    "reliability_cepheid": "Cepheid distances use a Wesenheit period-luminosity relation and have about 10% typical relative uncertainty.",
    "reliability_gcns": "GCNS distances use Gaia EDR3 parallax inversion and are precise to well under 1% nearby.",
    "reliability_apogee": "APOGEE StarHorse distances combine spectroscopy, photometry and Gaia parallaxes and were selected to have under 25% error.",
    "attribution_betelgeuse": "Betelgeuse has no Gaia distance in this catalogue; its listed distance comes from the HYG catalogue.",
    "attribution_vega": "Vega has no Gaia distance in this catalogue; its listed distance comes from the HYG catalogue.",
    "attribution_sirius": "Sirius has no Gaia distance in this catalogue; its listed distance comes from the HYG catalogue.",
    "attribution_proxima_centauri": "Proxima Centauri has no Gaia distance in this catalogue; its listed distance comes from the HYG catalogue.",
    "rrlyrae_luminosity_is_constant": "All RR Lyrae rows have the same assumed luminosity of 45 solar luminosities, so no individual star is most luminous.",
    "rrlyrae_colour_is_constant": "All RR Lyrae rows have the same assumed BP-RP colour of 0.35, so no individual star is reddest.",
    "absent_parallax": "The catalogue has no parallax column, so this star's parallax cannot be given from it.",
    "absent_temperature": "The catalogue has no temperature column, so this star's temperature cannot be given from it.",
    "absent_period": "The catalogue has no period column, so this star's period cannot be given from it.",
    "absent_mass": "The catalogue has no mass column, so this star's mass cannot be given from it.",
    "ood_capital": "The Gaia star catalogue does not answer this unrelated geography question.",
    "ood_recipe": "The Gaia star catalogue does not answer this unrelated cooking question.",
    "ood_programming": "The Gaia star catalogue does not answer this unrelated programming question.",
    "ood_medical": "The Gaia star catalogue does not answer this unrelated medical question.",
    "empty_cepheids_near": "There are no classical Cepheids within 100 parsecs in this catalogue.",
    "empty_gcns_far": "There are no GCNS stars beyond 500 parsecs in this catalogue.",
    "empty_cepheids_beyond_30kpc": "There are no Cepheids beyond 30 kiloparsecs in this catalogue.",
    "empty_gcns_within_10pc": "There are no GCNS stars within 10 parsecs in this catalogue.",
    "compare_gcns_vs_rrlyrae": "GCNS parallax distances are precise to under 1%, while RR Lyrae standard-candle distances are accurate to roughly 15-20%.",
    "compare_cepheid_vs_rrlyrae": "Cepheid period-luminosity distances have about 10% uncertainty, while RR Lyrae distances are accurate to roughly 15-20%.",
    "compare_gcns_vs_apogee": "GCNS parallax distances are precise to under 1%, while APOGEE StarHorse distances were selected to under 25% error.",
    "method_cepheid": "Cepheid distances use the Wesenheit period-luminosity relation of Ripepi et al. 2022, not parallax.",
    "method_rrlyrae": "RR Lyrae distances use a fixed absolute magnitude M_G = 0.64 and a colour-based reddening estimate following Muraveva et al. 2018.",
    "method_gcns": "GCNS distances come from Gaia EDR3 parallax inversion for nearby stars within about 100 parsecs.",
    "method_luminous": "The luminous Gaia DR3 sample uses Bailer-Jones et al. 2021 geometric distances from EDR3 parallaxes.",
    "method_apogee": "APOGEE distances are computed by StarHorse using spectroscopy, photometry and Gaia parallaxes.",
    "selection_parallax_cut": "The luminous Gaia DR3 sample requires parallax_over_error greater than 5.",
    "selection_cepheid_table": "The Cepheids were selected from gaiadr3.vari_cepheid.",
    "selection_cepheid_classification": "The Cepheids require type_best_classification DCEP and a period greater than 1.5 days.",
    "selection_distance_range": "The luminous Gaia DR3 sample was restricted to Bailer-Jones geometric distances from 1000 to 5000 parsecs.",
    "farthest_gaia_overall": "The most distant Gaia star in the catalogue is an RR Lyrae star at 29999.2 parsecs.",
    "nearest_gaia_overall": "The nearest Gaia star in the catalogue is a GCNS star at 29.83 parsecs.",
    "ambiguous_scope_count": "The catalogue has 120000 Gaia stars across all Gaia subsets.",
    "gcns_nearest_in_ly": "The nearest GCNS star is at 29.83 parsecs, or 97.29 light-years.",
    "gen_gaia_cepheid_farthest": "The farthest Gaia Cepheid is at 24952.0 parsecs.",
    "gen_gaia_cepheid_nearest": "The nearest Gaia Cepheid is at 263.90 parsecs.",
    "gen_gaia_cepheid_count": "There are 2187 Gaia Cepheids in the catalogue.",
    "gen_gaia_cepheid_count_within": "There are 1092 Gaia Cepheids within 6.8 kiloparsecs.",
    "gen_gaia_cepheid_count_beyond": "There are 1095 Gaia Cepheids beyond 6.8 kiloparsecs.",
    "gen_gaia_cepheid_farthest_ly": "The farthest Gaia Cepheid is at 81383 light-years.",
    "gen_gaia_cepheid_farthest_kpc": "The farthest Gaia Cepheid is at 24.95 kiloparsecs.",
    "gen_gaia_cepheid_avg_distance": "The average Gaia Cepheid distance is 8443.64 parsecs.",
    "gen_gaia_cepheid_most_luminous": "The most luminous Gaia Cepheid has a luminosity of 32910.0 solar luminosities.",
    "gen_gaia_cepheid_reddest": "The reddest Gaia Cepheid has a BP-RP colour of 1.050.",
    "gen_gaia_rrlyrae_farthest_ly": "The farthest Gaia RR Lyrae star is at 97844 light-years.",
    "gen_gaia_rrlyrae_farthest_kpc": "The farthest Gaia RR Lyrae star is at 30.00 kiloparsecs.",
}

# RAGAS calls OpenAI by default if nothing is configured; make a missing
# configuration fail loudly rather than reach for a hosted endpoint.
os.environ.setdefault("OPENAI_API_KEY", "not-used-local-only")


def newest_complete_run(path: Path | None = None) -> Path:
    if path is not None:
        return path
    best: tuple[int, Path] | None = None
    for candidate in sorted(EVAL_DIR.glob("eval-*.json")):
        try:
            n = json.loads(candidate.read_text(encoding="utf-8"))["summary"]["cases"]
        except (OSError, ValueError, KeyError):
            continue
        if best is None or n >= best[0]:
            best = (n, candidate)
    if best is None:
        raise FileNotFoundError(
            f"no eval-*.json in {EVAL_DIR}; run gaia_rag.eval.run_eval first"
        )
    return best[1]


def build_reference(case: dict) -> str:
    """A ground-truth string for the metrics that need one.

    Built from the gold case's own `note`, which records how the expected
    value was obtained (almost always a SQL query against the catalogue),
    plus the required values. These are verified facts, not a model's
    idea of a good answer.
    """
    if case.get("reference"):
        return str(case["reference"])
    if case["id"] in REFERENCES:
        return REFERENCES[case["id"]]
    parts = []
    if case.get("must_contain_numbers"):
        values = ", ".join(str(v) for v in case["must_contain_numbers"])
        parts.append(f"The correct value(s): {values}.")
    for phrase in case.get("must_mention") or []:
        parts.append(f"The answer must mention {phrase}.")
    if case.get("note"):
        parts.append(case["note"])
    if not case.get("should_answer", True):
        parts.append(
            "There is no answer in the corpus; the correct response is to "
            "decline rather than provide a value."
        )
    return " ".join(parts) or "No reference available."


def stratified(cases: list[dict], limit: int | None) -> list[dict]:
    """Spread a subset across case kinds rather than taking a prefix.

    The gold set is ordered curated-then-generated and grouped by subset,
    so the first N cases would be almost all one kind.
    """
    if limit is None or limit >= len(cases):
        return cases
    by_kind: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        # Eval records wrap the gold case under `case`. Looking for
        # `kind` on the outer record silently selected a plain prefix.
        gold_case = case.get("case", case)
        by_kind[gold_case.get("kind", "unspecified")].append(case)
    picked: list[dict] = []
    while len(picked) < limit:
        progressed = False
        for kind in sorted(by_kind):
            if by_kind[kind] and len(picked) < limit:
                picked.append(by_kind[kind].pop(0))
                progressed = True
        if not progressed:
            break
    return picked


def main() -> int:
    parser = argparse.ArgumentParser(description="Score answers with RAGAS.")
    parser.add_argument("--run", default=None, help="eval-*.json to score")
    parser.add_argument(
        "--limit",
        type=int,
        default=12,
        help="stratified subset size (RAGAS costs ~12-15 LLM calls per case)",
    )
    parser.add_argument("--all", action="store_true", help="score every case")
    parser.add_argument("--judge", default="qwen3:8b")
    parser.add_argument("--embed", default="nomic-embed-text")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="faithfulness + context recall only (~3 judge calls/sample)",
    )
    parser.add_argument(
        "--context-precision",
        action="store_true",
        help="add context precision (~7 extra judge calls/sample)",
    )
    args = parser.parse_args()

    from langchain_ollama import OllamaEmbeddings
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    from ragas.run_config import RunConfig

    from gaia_rag.eval.ragas_llm import OllamaRagasLLM

    run_path = newest_complete_run(Path(args.run) if args.run else None)
    data = json.loads(run_path.read_text(encoding="utf-8"))
    print(f"scoring answers from: {run_path.name}")

    # Only cases where an answer was actually served. A refusal has no
    # statements to decompose, so RAGAS faithfulness over it is noise.
    usable = []
    skipped_abstained = 0
    for record in data["cases"]:
        contexts = [e["text"] for e in record.get("retrieved", []) if e.get("text")]
        if not record.get("answer") or not contexts:
            skipped_abstained += 1
            continue
        if not record["result"].get("served", False):
            skipped_abstained += 1
            continue
        usable.append((record, contexts))
    print(
        f"  {len(usable)} answered cases usable, "
        f"{skipped_abstained} skipped (abstained/withheld: RAGAS cannot score a "
        "correct refusal)"
    )

    chosen = stratified(
        [r for r, _c in usable], None if args.all else args.limit
    )
    chosen_ids = {c["case"]["id"] for c in chosen}
    contexts_by_id = {r["case"]["id"]: c for r, c in usable}
    selected = [(r, contexts_by_id[r["case"]["id"]]) for r in chosen]

    samples = [
        SingleTurnSample(
            user_input=record["case"]["question"],
            response=record["answer"],
            retrieved_contexts=contexts,
            reference=build_reference(record["case"]),
        )
        for record, contexts in selected
    ]
    print(f"  evaluating {len(samples)} sample(s)\n")
    for record, _c in selected:
        print(f"    - [{record['case']['kind']}] {record['case']['id']}")

    # A judge that talks to Ollama directly, because the langchain-ollama
    # version RAGAS resolves cannot disable a reasoning model's thinking:
    # 131s per call with it on against 3.4s with it off on this machine,
    # across several calls per sample per metric. It also means RAGAS
    # judges through the same provider and settings as the pipeline it is
    # scoring, so a disagreement cannot be put down to configuration.
    judge = OllamaRagasLLM(model=args.judge)
    embedder = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=args.embed))

    # Context precision costs one call per retrieved context (~7 here) --
    # more than every other metric combined -- and it re-measures a
    # figure this project already reports and understands: precision is
    # low by design, because both retrievers run for every question. So
    # it is opt-in rather than default.
    metrics = [Faithfulness(llm=judge), LLMContextRecall(llm=judge)]
    if not args.fast:
        metrics.append(ResponseRelevancy(llm=judge, embeddings=embedder))
    if args.context_precision:
        metrics.append(LLMContextPrecisionWithReference(llm=judge))

    print(f"\n  judge: {args.judge} (thinking disabled)")
    print(f"  embeddings: {args.embed}")
    print(f"  metrics: {', '.join(type(m).__name__ for m in metrics)}")
    print("  running RAGAS (this is the slow part)...\n")
    # RAGAS defaults to 16 workers. Ollama serves one model serially, so
    # extra concurrency buys nothing and only thrashes the context of an
    # 11GB model; a small pool keeps the event loop healthy and the
    # per-call timeout meaningful.
    run_config = RunConfig(max_workers=1, timeout=600, max_retries=2)
    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=metrics,
        llm=judge,
        embeddings=embedder,
        run_config=run_config,
        show_progress=True,
    )

    print("\n" + "=" * 66)
    print("RAGAS RESULTS")
    print("=" * 66)
    scores = {}
    for key, value in result._repr_dict.items() if hasattr(result, "_repr_dict") else []:
        scores[key] = value
    if not scores:
        try:
            scores = dict(result)
        except TypeError:
            scores = {}
    for key, value in scores.items():
        try:
            print(f"  {key:34s} {float(value):.3f}")
        except (TypeError, ValueError):
            print(f"  {key:34s} {value}")
    print("=" * 66)
    print(
        "Subset of answered cases only; abstention cases are excluded and\n"
        "RAGAS cannot assess them. Scores are indicative: the judge is a\n"
        "local 8B model."
    )

    out = EVAL_DIR / f"ragas-{run_path.stem.replace('eval-', '')}.json"
    payload = {
        "source_run": run_path.name,
        "judge": args.judge,
        "embeddings": args.embed,
        "samples": len(samples),
        "metrics": [type(m).__name__ for m in metrics],
        "judge_calls": judge.calls,
        "skipped_abstained": skipped_abstained,
        "scores": {k: str(v) for k, v in scores.items()},
        "case_ids": sorted(chosen_ids),
    }
    try:
        df = result.to_pandas()
        payload["per_case"] = json.loads(df.to_json(orient="records"))
    except Exception:
        pass
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
