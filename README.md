# GaiaRAG

A retrieval-augmented question-answering system over the Gaia star
catalogue data produced by the companion Galaxy project, built so that
every number in an answer can be traced to a catalogue row and every
methodological claim to a source document — and so that answers which
cannot be traced are **withheld rather than served**.

Runs entirely locally on Ollama. No API keys, no network at query time.

The catalogue and source documents are not included in this repository.
To build the index, provide `hygdata_web.csv`, `SOURCES.txt`, and the
Galaxy ingestion tools. By default the build looks for a `Galaxy`
directory next to this repository. Set `GAIA_RAG_GALAXY_CSV`,
`GAIA_RAG_GALAXY_SOURCES`, and `GAIA_RAG_GALAXY_TOOLS` to use other
locations. The published evaluation figures use the local catalogue
described below; a different input catalogue will produce different
answers and scores.

```bash
pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull qwen3:8b

python scripts/build_all.py
python scripts/ask.py "How were the Cepheid distances derived?"
python -m gaia_rag.eval.run_eval
python scripts/run_tests.py
```

## Why this is not an ordinary RAG

A conventional chunk-and-embed RAG fails badly on catalogue data, in a
way that is easy to miss because the failures look like successes.

**Numbers do not embed.** "548 pc" and "168 pc" are nearly identical in
embedding space and read as paraphrases to an LLM judge. A swapped
distance is the single most likely error here and the least likely to be
caught by semantic scoring. So no number in an answer is ever produced by
the language model: numbers come from SQL over DuckDB, and a
**deterministic arithmetic check** confirms every numeric literal in the
answer traces to a retrieved value under some legitimate unit rendering.

**Parametric knowledge contaminates grounding.** The generator knows a
lot of real astronomy. "Betelgeuse is about 548 ly away" is a true
sentence and a *grounding failure* if no retrieved row says so. Truth
and faithfulness come apart here, and only the latter is measurable, so
the prompt, the claim schema and the metrics all treat recalled facts as
failures.

**Provenance is part of the answer.** A distance means something
different depending on how it was obtained. A GCNS parallax distance is
good to under 1%; an RR Lyrae period-luminosity distance to about 18%.
The same bare number carries both, so every value is retrieved together
with its derivation method, its uncertainty and its caveats, and the
system is required to state them.

## Architecture

```
question
   |
   +-- planner (LLM) --> QuerySpec (validated JSON, not SQL)
   |                        + deterministic subset detection
   |                        |
   |                     DuckDB ----------> catalogue rows + per-field provenance
   |
   +-- hybrid doc retrieval (BM25 + dense, RRF) --> method / caveat chunks
                                                    or an explicit abstention
   |
   v
generator (LLM) --> {sufficient, answer, claims[{text, evidence_ids}]}
   |
   +-- numeric verifier   (deterministic, arithmetic)
   +-- subject verifier   (deterministic, citation drift)
   +-- claim judge        (LLM entailment, per claim, against cited evidence only)
   |
   v
guard --> answered | answered_with_warnings | abstained | withheld_ungrounded
```

### The structured half

The model never writes SQL. It emits a `QuerySpec` — a small JSON object
naming fields, operators and bounds — which is validated against a
whitelist and compiled to parameterised SQL in
[`retrieve/structured.py`](gaia_rag/retrieve/structured.py).

This is not only about injection. Free-form text-to-SQL fails in a way
that is very hard to detect downstream: a query referencing the wrong
column or quietly dropping a filter still returns plausible rows, and an
answer grounded in those rows passes every faithfulness check while
being wrong. A spec that *cannot express* an invalid query cannot fail
that way.

Two further rules:

- **An unconstrained query returns nothing.** With no named object, no
  filter and no ordering, `LIMIT 10` is an arbitrary sample that answers
  nothing but looks authoritative in a prompt.
- **Subsets are detected lexically, not inferred.** Asked how reliable
  RR Lyrae distances were, the planner ignored the mapping stated in its
  prompt and retrieved ten arbitrary GCNS stars. Surface terms don't need
  inferring, so they aren't left to inference.

### The document half

36 documents / 44 chunks, extracted from text that exists on disk —
Galaxy's `SOURCES.txt`, the docstrings of its ingestion tools, and the
ADQL those tools actually run — rather
than written from memory about the cited papers. If the corpus itself
contained paraphrases invented by a model, a "faithful" answer grounded
in it would still be wrong and the metric would report success.
Grounding is only worth measuring if the ground is real.

Retrieval fuses BM25 and dense cosine by reciprocal rank fusion. Both
are needed; see the measurements below.

### Verification

| Check | Kind | Gate |
|---|---|---|
| Schema scope | deterministic | **pre-generation** — row evidence withdrawn for out-of-schema quantities |
| Numeric grounding | deterministic | **hard** — one ungrounded number withholds the answer |
| Subject consistency | deterministic | **hard** — a claim citing another subset's evidence withholds |
| Claim entailment | LLM judge | soft — aggregate score thresholded at 0.65 |
| Contradiction | LLM judge | **hard** |
| Retrieval emptiness | deterministic | abstain |

The gates are asymmetric deliberately. A wrong distance is the failure
mode this corpus invites and there is no safe way to serve it. Weak
*qualitative* support is a matter of degree, so it is scored and
thresholded rather than hard-gated, which would reject almost every
useful answer.

Detection alone would be decoration: [`verify/guard.py`](gaia_rag/verify/guard.py)
turns a failed check into a withheld answer plus a specific statement of
what failed.

## What the build actually found

Every item below was a real defect caught while building this, and each
produced a confident, well-formed, wrong output. They are the reason the
test suite and the gold set exist in the shape they do.

**A sentinel value that would become the "most distant star."** HYG
encodes "no usable parallax" as exactly 100,000 pc. 64 rows carry it, and
the next largest real HYG distance is 990 pc. Left in, any "farthest
star" query returns them, citing a real row, passing every grounding
check, while being pure fiction. A well-formed sentinel is the most
dangerous kind of bad data for a RAG.

**A wrong uncertainty that passed every gate.** Asked how reliable RR
Lyrae distances are, the system answered "reliable with a typical
relative uncertainty of 1%" — the GCNS figure, eighteen times too
precise. It passed because:
1. the planner retrieved GCNS rows instead of RR Lyrae rows;
2. the claim cited *both* caveat documents, and the GCNS one really does
   say 1%, so the entailment judge found support and returned
   "supported";
3. the numeric verifier classified `1` in "1%" as a small integer and
   never checked it.

Fixing it needed all three: lexical subset detection, the deterministic
subject-consistency check, and unit-aware number extraction. No amount of
judge prompting would have caught (2) — the judge was being asked to
notice that the right number came from the wrong row.

**Dense retrieval missing the one document that answers the question.**
With embeddings alone, "How reliable are RR Lyrae distances?" and "How
were Cepheid distances derived?" both missed their target document,
landing instead on whichever chunk was most generically about Gaia
distances. Three separate causes:

- *No plural stemming.* The documents say "Cepheid**s**"; the question
  says "Cepheid". BM25 treated them as unrelated terms and dropped the
  only high-information term in the question.
- *Documents identified by slug.* A caveat document naming its subject
  only as `gaia_rrlyrae` tokenises to `gaia` + `rrlyrae` and can never
  match "RR Lyrae".
- *RRF's default constant.* `k=60` is tuned for TREC-scale runs. On 34
  chunks it is far larger than the ranks themselves, so fusion stops
  discriminating and a chunk ranked 5th by both retrievers outscored one
  ranked **1st** by BM25. The correct Cepheid document was at BM25 rank 1
  and was being discarded by the fusion step.

**Self-inflicted near-duplicate chunks.** Generating per-field caveats
per subset produced six chunks of byte-identical boilerplate about colour
and luminosity, which then outranked the distinctive documents. Related:
listing every subset's human label under "Applies to" put "RR Lyrae" and
"Cepheid" into all five field documents, so those matched every
subset-specific question.

**Cosine similarity is useless as an abstention threshold here.** Nomic
scores an off-topic question ("capital of France") at 0.53 against this
corpus and a genuine one at 0.72. No threshold in that gap survives a
corpus change. The gate is lexical instead: a question sharing no
*distinctive* term with the corpus is out of domain. Requiring merely
*any* shared term was too weak — "Tell me about quantum chromodynamics"
matched on the word "about" and was served five Gaia documents.

**Extended thinking made the model slower *and* worse.** One planner call
with `qwen3:8b`: **131s** with thinking on, **3.4s** with it off — and
the thinking run invented catalogue subset names
("Baade-Wesselink method accuracy") where the direct run correctly
returned an empty list. Constrained JSON generation fights with a
reasoning preamble: the model spends its budget reasoning in prose it is
then not permitted to emit. Disabled by default.

### Found by the first full eval run

The five gold-set failures on the first complete run were all real, and
none of them were caught by anything cheaper than the eval.

**A star count served as an exoplanet count.** "How many confirmed
exoplanets orbit stars in the Gaia Cepheid sample?" returned *"there are
2187 confirmed exoplanets"* — the number of Cepheid **stars**, relabelled.
No exoplanet data is ingested at all.

Every check passed, and each was right to. 2187 really is in the
evidence; the claim is about Cepheids and so is the evidence; and the
evidence does say `count = 2187`. The defect was in none of them:
`count(*)` over a table of stars answers "how many stars", and nothing
stopped that number being attached to a different noun. Relabelling an
aggregate is invisible to every check that reasons about *values* and
*subsets*, because both were correct. The fix
([`verify/scope.py`](gaia_rag/verify/scope.py)) guards the **question**
instead: if it asks for a quantity the schema has no column for, row
evidence is withdrawn before generation, so there is nothing to
misattribute. Aggregate evidence now also states what it counts ("one
catalogue row is one star"). Questions like "how does metallicity affect
RR Lyrae distances?" are deliberately *not* blocked — they are answerable
from the method documents.

**The planner inventing filters, in both directions.** Asked what
distance Gaia gives for Betelgeuse, it produced `subset = 'gaia_gcns'`
out of nothing, which excluded the single row that answers the question.
This is the mirror image of the RR Lyrae failure, where it omitted a
subset the question named. Subset constraints are now honoured only when
the question actually names that subset.

**Abstaining when the data had an answer.** Betelgeuse is in the
catalogue, just not from Gaia. A named lookup that finds nothing in the
Gaia subsets now widens to the whole catalogue — dropping subset filters
too, not just the `gaia_only` flag — so the system can give the genuinely
useful reply: *"Gaia does not provide a distance for Betelgeuse in this
catalogue. The 152.67 pc (497.9 ly) value is from the HYG Database v4.1
and is not a Gaia measurement."* A prompt rule now states that a
well-evidenced negative is an answer: "the data says no" and "I don't
know" are different, and collapsing them loses real information.

**A correct judge verdict withholding a correct answer.** Asked for the
*closest* Cepheid, the pipeline retrieved the right row and was withheld,
because the judge called "X is closest to the Sun" unsupported — and it
was right: a single row cannot establish a minimum. The ordering was a
fact about the query that never reached the evidence. Rows from an
ordered query now carry their own ranking ("lowest `dist_pc` of the 2,187
matching rows"). The fix was to make the evidence honest, not to lower
the threshold.

**Substance in the claims, not in the answer.** Two cases answered
correctly but tersely — "The APOGEE distances are not Gaia measurements."
— while the detail sat in the `claims` list the user never sees. The
prompt now requires `answer` to stand alone. Separately, two gold
assertions demanded specific proper nouns ("StarHorse", "metallicity")
where equivalent phrasings are just as correct, so those became
any-of groups: substance is still required, exact vocabulary is not.
Worth flagging plainly, since relaxing a test to match the output is
exactly the move that can be rationalised — the uncertainty figure itself
stayed a verbatim requirement.

**The fix that was worse than the bug.** Correcting the invented-filter
problem above, I made the planner's subset choice conditional on the
question naming that subset — and cleared it otherwise. That turned "How
many classical Cepheids are in the catalogue?" into **120,000** instead
of **2,187**: wrong by a factor of 55, fully grounded, confidently
worded, and passing every gate. Two separate mistakes compounded:

- *The same bug in a second place.* `detect_subsets` matched "cepheid"
  but not "Cepheid**s**". I had already fixed plural matching in the BM25
  tokeniser and never thought to look for it in the keyword matcher. One
  root cause, two homes, and fixing it once felt like fixing it.
- *A dangerous failure direction.* On detection failure the code
  *discarded* a correct filter, widening the scope.

My first conclusion from this was that **widening is more dangerous than
narrowing**, on the reasoning that a too-narrow filter returns an
obviously odd result while a too-broad one returns a plausible number
for a different question. A later gold case showed that to be wrong.
Asked for "the most distant Gaia star", the planner silently added
`subset = 'gaia_dr3_luminous'` and the answer became **5,000 pc instead
of 29,999.2 pc** — a narrowing, and just as plausible-looking as the
widening had been.

The real factor is the query's **shape**, not the direction of the
error:

- For `extreme` and `aggregate`, **the scope is the answer**. Any
  unprompted scope change corrupts it either way, so the scope comes
  from the question alone and the model may not narrow it.
- For `lookup` and `filter`, the scope only selects which rows are
  shown. The planner's choice is kept, and an empty named lookup is
  recovered by the widening in `run()`.

Both halves of that rule are now regression-tested, in both directions.

The general lesson is in the gold set now as a regression test: this
defect was introduced *by* a verification improvement, was invisible to
all three verifiers, and was caught only because an aggregate case with a
known answer already existed.

**A correct abstention that looked like a bug.** The abstention message
read "No document chunk cleared the similarity floor (0.76 < 0.45)" —
comparing the dense score against a cosine floor that was no longer the
out-of-domain test at all. The behaviour was right; the explanation was
nonsense. Diagnostics drift out of sync with logic quietly.

**A superlative ordered by the wrong field.** Every `extreme` query
defaulted to ordering by distance whatever it asked about, so "largest
BP-RP colour among the Cepheids" retrieved the ten most *distant*
Cepheids and reported the largest colour **among those ten** -- 0.880
against a true 1.05. It passed every verification gate, because 0.880
really is in the evidence; the evidence was simply the wrong rows. That
is precisely the "a wrong query returns plausible rows" failure the
QuerySpec design exists to prevent, reintroduced by a careless default.
The same defect also made "most luminous RR Lyrae" order by distance,
where the claim was correctly judged unsupported and withheld -- one
silent error and one honest refusal from a single cause.

**A field that looks per-star but is a constant.** All 67,813 RR Lyrae
rows carry luminosity 45 and colour 0.35, because the distance method
*assumed* a fixed absolute magnitude and intrinsic colour. "Which RR
Lyrae is most luminous?" therefore has **no answer**, and the honest
reply is that every row shares the value -- not to name one of 67,813
ties. The pipeline named one, and the caveat document saying otherwise
was retrieved and ignored. Detection is now data-driven
(`count(DISTINCT field) == 1`), stated in the row's own evidence, and
generalises to any field that collapses under a narrow filter. It was
only noticed because the data was inspected before generating cases
against it, rather than trusting the schema.

**Absence that could not be cited.** A query matching zero rows is a
verified fact, but it lived in a retrieval *note* -- which reaches the
prompt yet carries no evidence id. So "the query matched zero rows,
therefore there are none" cited a document that does not say that, the
judge correctly called it unsupported, and two evidence-of-absence cases
were withheld for answers that were right. Absence is only answerable if
it is evidence, so it now gets its own `query:zero_rows` item with the
scope, the conditions and the count.

**A deadlock in the RAGAS integration, invisible on the progress bar.**
RAGAS schedules every metric call as a coroutine on one event loop
(`as_completed`, 16 workers, `nest_asyncio`, no thread offloading). A
custom judge that did a *blocking* HTTP call inside that coroutine
blocked the loop -- and because RAGAS's own timeout is an asyncio timeout
on the same loop, it could never fire either. Two runs hung indefinitely
at 0/117 while using no CPU. The tell was a **flat CPU line**: a tqdm bar
at 0% cannot distinguish "waiting on a slow model" from "deadlocked", but
CPU delta over a 30-second window answers it immediately. Fixed with
`asyncio.to_thread` and a worker cap of 4, since Ollama serialises
requests for one model anyway.

**Ingest speed and a CSV that defeats DuckDB's sniffer.** Row-by-row
`executemany` over 300k rows took minutes; the same transform in SQL
takes 2.5s. And `read_csv_auto` fails outright on this file — the
sniffer treats `'` as a candidate quote character and the catalogue
contains `Van Maanen's Star`, whose lone apostrophe makes every candidate
dialect look malformed. The dialect is pinned explicitly, which is better
for reproducibility anyway.

## Data

Ingested from Galaxy's `web/data/hygdata_web.csv` — real Gaia data,
already distance-resolved by Galaxy's tooling. 299,935 rows, of which
**120,000 are Gaia**:

| subset | rows | distance method | typical uncertainty |
|---|---|---|---|
| `gaia_rrlyrae` | 67,813 | M_G = 0.64 + colour-based reddening (Muraveva+ 2018) | ~18% |
| `gaia_dr3_luminous` | 45,000 | Bailer-Jones+ 2021 geometric (1–5 kpc) | ~20% |
| `gaia_gcns` | 5,000 | EDR3 parallax inversion, <100 pc (Smart+ 2021) | <1% |
| `gaia_cepheid` | 2,187 | Wesenheit period–luminosity (Ripepi+ 2022) | ~10% |
| `apogee` | 137,415 | StarHorse, APOGEE DR17 + EDR3 (Queiroz+ 2023) | ~25% |
| `hyg` | 42,520 | Hipparcos / Yale / Gliese | ~5% |

`apogee` and `hyg` are **not Gaia** and are flagged `is_gaia = FALSE`, so
the system cannot present a Hipparcos distance as a Gaia measurement.
This matters more than it sounds: Betelgeuse and Vega are present *only*
in the HYG subset, so "what distance does Gaia give for Betelgeuse?" has
no Gaia answer, and the gold set tests exactly that.

The Gaia archive TAP fetcher in `ingest/` is optional enrichment for
pulling fresh rows with full source fields (parallax, `phot_g_mean_mag`,
periods). The ESA TAP endpoint was unreachable from the build machine, so
the pipeline is built against the local catalogue and works offline.

## Evaluation

```bash
python -m gaia_rag.eval.run_eval                     # full gold set
python -m gaia_rag.eval.run_eval --kind out_of_domain
python -m gaia_rag.eval.run_eval --only rrlyrae_reliability
```

The gold set (`gaia_rag/eval/goldset/gaia_gold.jsonl`) holds **50 cases**, downsampled from 100 generated shapes, in two
populations reported separately:

- **38 curated** — hand-written adversarial cases, each encoding one
  specific way a RAG lies. These carry the faithfulness signal and
  cannot be generated, since generating one would mean already knowing
  the failure.
- **12 generated** — expected values computed by SQL at build time, so
  they are correct by construction and regenerate when the catalogue
  changes (`python -m gaia_rag.eval.goldset.build_goldset`). These test
  plumbing: query planning, numeric grounding, units.

The split matters: twenty variations of "farthest in subset" are not
twenty independent tests of faithfulness, and averaging them in would
let easy generated cases mask a trap failure. Every case records in a
`note` how its expected value was obtained. Case kinds:

- `numeric` / `aggregate` — a specific value must appear
- `selection` — the exact ADQL cut or archive table must be named
- `empty_result` — the query runs and matches nothing; saying "none"
  is the answer, not an abstention
- `comparison` — names two subsets legitimately, so it guards the
  subject-consistency check against over-firing
- `method` — a derivation must be described correctly
- `unit_conversion` — 29.83 pc is 97.29 ly; reporting the parsec figure
  as light-years is the failure
- `trap_wrong_uncertainty` — the RR Lyrae case above, with `1.0` as a
  **forbidden** number
- `trap_attribution` — must not credit Gaia for HYG or APOGEE data
- `out_of_domain` — must abstain
- `unanswerable_in_corpus` — the *object* exists but the *field* does
  not (stellar mass), which is harder than out-of-domain

Metrics are in [`eval/metrics.py`](gaia_rag/eval/metrics.py). The
headline is **answered-wrong rate**: the share of questions where an
answer was served that should not have been, or was served with a wrong
value. It is reported separately and is not an average of the others,
because a system that abstains on everything scores a perfect
faithfulness while being worthless, and one number hiding that is worse
than no number at all. Abstention correctness is reported alongside it
for the same reason.

### Results

50-case safety gold set, `qwen3:8b` generator and judge, and
`nomic-embed-text` embeddings (run `eval-20261102-191655`):

| metric | value |
|---|---|
| cases | 50 (42 served, 8 correct refusals) |
| pass rate | **100%** |
| answered-wrong rate | **0%** |
| abstention correctness | 100% |
| faithfulness, claims (served) | 0.994 |
| numeric exactness (served) | 1.000 |
| subject consistency (served) | 1.000 |
| context recall | 0.940 |
| context utilisation | 0.810 |
| answer relevancy | 0.910 |

The full run includes the previously failing empty-result, comparison,
and scientific-notation cases. All four pass. The numeric verifier now
reads scientific notation, uncertainty ranges and written-out units such
as “light-years.” A separate eight-case answerable supplement also
passes 8/8; it exposed and led to fixes for an invented threshold in a
nearest-HYG query and for confusing distance dispersion with measurement
uncertainty.

### Context utilisation, and why it is not "precision"

This metric counts how much of the retrieved context the answer actually
**cited** (`cited / retrieved`). It is deliberately *not* called context
precision: precision in the IR sense asks what fraction of retrieved
context is **relevant**, and an item can be perfectly relevant and simply
uncited. The old name invited the reading that a low score meant most
retrieved context was junk -- it does not -- and it also led to this
figure being described as comparable to RAGAS's `context_precision`,
which judges relevance and ranking. They measure different things.

The current retriever returns a directly relevant document for each
named subset, or each of two named subsets in a comparison. Structured
lookup, count and extreme evidence already includes provenance, so it
does not add unrelated document chunks to those answers. A unique
extreme keeps only its top row; ties remain visible. These changes raised
context utilisation to 0.810 without reducing the gold set pass rate.

### Ablations: what each component is actually worth

`python -m gaia_rag.eval.ablate` disables one component at a time and
re-runs only the gold cases that component exists to handle, reusing the
full-system verdicts from a reference run.

The table below is historical, measured before the current retrieval and
evidence changes. Re-run the ablations before using its numbers to compare
current components.

| component removed | n | pass | answered-wrong |
|---|---|---|---|
| **BM25 removed (dense only)** | 7 | 100% -> **0%** | 0% -> 0% |
| **no scope guard** | 5 | 100% -> **20%** | 0% -> **80%** |
| **no abstention** | 11 | 100% -> **27%** | 0% -> **73%** |
| **no unconstrained guard** | 4 | 100% -> **50%** | 0% -> **25%** |
| no out-of-domain gate | 3 | 100% -> 100% | 0% -> 0% |
| no numeric gate | 22 | 100% -> 100% | 0% -> 0% |
| no subject gate | 5 | 100% -> 100% | 0% -> 0% |

Four components are decisively load-bearing. Removing BM25 breaks *all
seven* document cases, which settles the hybrid-retrieval question with
evidence rather than anecdote. One detail is worth noticing: dense-only
drops pass to 0% while answered-wrong stays at 0% -- retrieval collapsed
and the verifiers converted that into refusals rather than lies.
Degrading retrieval makes this system quieter, not wronger.

**Three ablations showed no effect, and that includes the numeric gate --
the component this project argues is most important.** The explanation is
mundane but it matters: these gates only fire when the model errs, so on
a run where every answer is already correct, removing the safety net
changes nothing. It measures "the net was not needed today", not "the net
is useless" -- the 1% and 10% citation-drift bugs are real history where
it did fire. But the ablation as constructed does **not** support the
claim, and saying otherwise would be dishonest. Measuring it properly
needs cases built to provoke ungrounded numbers, not cases the pipeline
already handles.

### External check: RAGAS

The safety set has 42 answerable cases and eight refusal cases. RAGAS's
faithfulness and context metrics do not measure correct refusal. The
separate RAGAS benchmark contains those 42 answerable cases plus eight
additional answerable questions verified from the catalogue and
provenance, giving **50 distinct served answers**. The original refusal
cases remain in the 50-question safety set.

Run the safety set, the supplement, then build and score the benchmark:

Install the optional RAGAS dependencies in a separate environment first:

```bash
python -m venv .venv-ragas
.venv-ragas/Scripts/python.exe -m pip install -r requirements-ragas.txt
```

On Windows with Bash, `bash scripts/run_eval_then_ragas.sh` runs all five
steps. The equivalent commands are:

```bash
python -m gaia_rag.eval.run_eval
python -m gaia_rag.eval.run_eval --goldset gaia_rag/eval/goldset/ragas_extra.jsonl
python -m gaia_rag.eval.build_ragas_benchmark --safety-run data/processed/eval/<safety-run>.json --extra-run data/processed/eval/<extra-run>.json
.venv-ragas/Scripts/python.exe -m gaia_rag.eval.ragas_eval --run data/processed/eval/<benchmark-50-run>.json --all --context-precision --judge qwen3:8b
python -m gaia_rag.eval.verify_ragas --run data/processed/eval/<ragas-benchmark-50-run>.json
```

Full 50-question RAGAS run `ragas-benchmark-50-20261102-192234.json`,
using the local `qwen3:8b` judge and `nomic-embed-text` embeddings:

| RAGAS metric | score |
|---|---:|
| faithfulness | **0.862** |
| context recall | **0.980** |
| answer relevancy | **0.888** |
| context precision with reference | **1.000** |

All four averages exceed 0.7. Each metric has **50/50 finite per-case
values**; no question was skipped, and the run made 361 judge calls.
The [public evaluation report](reports/evaluation-2026-11-02.json) records
the aggregate and per-case scores without copying source data or answers.
These are averages, not a claim that each individual answer exceeds 0.7:
ten answers scored at or below 0.7 on RAGAS faithfulness. The 1.000
context-precision score is also helped by the retriever returning one
focused context for most questions; it does not imply that all of the
catalogue's potential evidence is equally useful.

The benchmark builder checks that all 50 cases have distinct ids,
references and served answers. RAGAS scores the stored answer text and
retrieved contexts, so the external metrics inspect exactly what the
project served. Its references are factual answers written from the
gold assertions and catalogue queries; test-maintainer notes containing
known wrong values are not used as references.

RAGAS supplies an independent implementation and prompt set, though
its faithfulness metric also uses LLM statement extraction and
entailment. The deterministic numeric and subject checks remain separate.

Each run writes full JSON to `data/processed/eval/` — evidence sets,
claim verdicts, numeric checks, the generated SQL and the spec — so two
runs can be diffed after a threshold or prompt change. A faithfulness
score without the failing claim attached is not actionable.

## Configuration

Every knob is an env var (`GAIA_RAG_*`); see
[`config.py`](gaia_rag/config.py).

| variable | default | effect |
|---|---|---|
| `GAIA_RAG_GEN_MODEL` | `qwen3:8b` | generator |
| `GAIA_RAG_JUDGE_MODEL` | `qwen3:8b` | entailment judge |
| `GAIA_RAG_EMBED_MODEL` | `nomic-embed-text` | embeddings |
| `GAIA_RAG_DISABLE_THINKING` | `1` | see the 131s → 3.4s finding |
| `GAIA_RAG_NUM_REL_TOL` | `0.02` | numeric match tolerance |
| `GAIA_RAG_MIN_CLAIM_SUPPORT` | `0.65` | soft faithfulness gate |
| `GAIA_RAG_TOP_K_DOCS` | `2` | maximum method chunks; named subjects use one each |
| `GAIA_RAG_CONTEXT_TOKENS` | `8192` | Ollama context window |
| `GAIA_RAG_MAX_OUTPUT_TOKENS` | `512` | Ollama answer and judge output cap |
| `GAIA_RAG_REQUIRE_NUM_GROUND` | `1` | hard numeric gate |
| `GAIA_RAG_GALAXY_CSV` | Galaxy's path | source catalogue |

The judge is configured separately from the generator on purpose:
scoring your own output with the same weights inflates faithfulness. For
a stronger judge, point `GAIA_RAG_JUDGE_MODEL` at a larger model — the
metrics code is unchanged by it.

## Known limitations

- **The judge is the weakest component.** It runs after the
  deterministic checks precisely so it is only asked to do what it is
  reasonably good at, but an 8B local judge is still the least reliable
  gate. Nothing numeric depends on it.
- **Latency.** 50–310s per question on CPU, depending on how many
  claims an answer makes: cost is dominated by one judge call per
  claim, not by context length. Fine for evaluation, far too slow for
  interactive use.
- **Derived fields only.** Galaxy's CSV has already collapsed parallax,
  apparent magnitude and period into distance and luminosity, so the
  system cannot answer questions about raw Gaia measurements. The TAP
  fetcher exists for this; it needs network access.
- **No cross-catalogue reasoning.** Exoplanets, clusters, HII regions and
  nebulae are in Galaxy's data but not ingested here. The gold set
  asserts the system *abstains* on them rather than guessing.
- **`lum_lsun` is order-of-magnitude.** Derived from distance with a
  crude bolometric correction and no extinction correction; it inherits
  the distance uncertainty. Flagged as such in provenance, but it is weak
  evidence for anything.

## Layout

```
gaia_rag/
  config.py              all configuration, env-overridable
  units.py               pc / ly / kpc / parallax conversions (tested)
  providers/             LLM + Embedder protocols, Ollama backend
  ingest/
    catalog.py           CSV -> DuckDB, provenance table, sentinel handling
    docs.py              method/caveat corpus from on-disk text
    chunker.py           boundary-aware chunking, nomic task prefixes
    index.py             embed + persist, exact search
    fetch_gaia.py        optional: raw rows from the ESA TAP archive
  retrieve/
    extremes.py          which field a superlative is about
    evidence.py          Evidence + NumericFact: the citation unit
    structured.py        QuerySpec -> validated parameterised SQL
    lexical.py           BM25, stemming, RRF
    semantic.py          hybrid fusion + out-of-domain gate
    router.py            both halves into one evidence bundle
  generate/
    prompts.py           grounded-answer and judge prompts
    answer.py            the pipeline
  verify/
    numeric.py           deterministic arithmetic grounding
    subject.py           citation-drift detection
    scope.py             refuses numbers for out-of-schema concepts
    claims.py            per-claim entailment judging
    guard.py             the abstain policy
  eval/
    ablate.py            per-component ablations
    ragas_eval.py        external scoring via RAGAS
    ragas_llm.py         RAGAS judge backed directly by Ollama
    goldset/             50 cases (38 curated, 12 SQL-generated)
      build_goldset.py   regenerates the generated half from the data
    metrics.py           metrics incl. answered-wrong rate
    run_eval.py          runner
scripts/
  build_all.py           full corpus build
  run_eval_then_ragas.sh chained eval + RAGAS run
  ask.py                 CLI
  run_tests.py           all suites, no pytest needed
tests/
  test_units.py          conversions
  test_retrieval.py      classification, lexical retrieval, spec safety
  test_guard.py          the verification stack, models stubbed
  test_scope.py          schema scope, planner reconciliation, widening
```
