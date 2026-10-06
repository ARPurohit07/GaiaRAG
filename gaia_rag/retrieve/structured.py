"""Structured retrieval over the Gaia catalogue.

Numbers never come from the language model. The model's only job is to
produce a *query spec* -- a small JSON object naming fields, operators
and bounds -- which is validated against a whitelist and compiled into
parameterised SQL here. DuckDB produces the numbers.

This is deliberately not text-to-SQL. Free-form SQL from a model fails
in a way that is very hard to catch downstream: a query that references
the wrong column or quietly drops a filter still returns plausible rows,
and an answer grounded in those rows passes every faithfulness check
while being wrong. A spec that cannot express an invalid query cannot
fail that way.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import duckdb

from ..config import CONFIG
from ..verify.subject import detect_subsets
from .extremes import DEFAULT_ORDER, infer_extreme_order
from .evidence import Evidence, NumericFact, row_to_evidence

# --- whitelists -------------------------------------------------------

SELECTABLE = (
    "row_id",
    "evidence_id",
    "subset",
    "is_gaia",
    "label",
    "source_id",
    "proper_name",
    "constellation",
    "spectral_type",
    "ra_deg",
    "dec_deg",
    "dist_pc",
    "dist_ly",
    "dist_kpc",
    "color_bp_rp",
    "lum_lsun",
)

FILTERABLE = {
    "dist_pc": "DOUBLE",
    "dist_ly": "DOUBLE",
    "dist_kpc": "DOUBLE",
    "ra_deg": "DOUBLE",
    "dec_deg": "DOUBLE",
    "color_bp_rp": "DOUBLE",
    "lum_lsun": "DOUBLE",
    "subset": "VARCHAR",
    "constellation": "VARCHAR",
    "spectral_type": "VARCHAR",
    "is_gaia": "BOOLEAN",
}

OPERATORS = {"<", "<=", ">", ">=", "=", "!=", "between"}

#: Rows kept for an `extreme` query. The extreme row is the answer;
#: two spares make a tie visible without padding the prompt.
EXTREME_ROW_CAP = 3

AGGREGATES = {"count", "avg", "min", "max", "sum", "median", "stddev"}

VALID_SUBSETS = (
    "gaia_gcns",
    "gaia_cepheid",
    "gaia_rrlyrae",
    "gaia_dr3_luminous",
    "apogee",
    "hyg",
)


class SpecError(ValueError):
    """The model produced a spec that cannot be compiled safely."""


# --- spec -------------------------------------------------------------


@dataclass
class QuerySpec:
    intent: str = "filter"  # lookup | filter | extreme | aggregate
    name: str | None = None
    subsets: list[str] = field(default_factory=list)
    filters: list[dict] = field(default_factory=list)
    order_by: dict | None = None
    limit: int = 10
    aggregate: dict | None = None
    gaia_only: bool = True

    @classmethod
    def from_dict(cls, raw: dict) -> "QuerySpec":
        if not isinstance(raw, dict):
            raise SpecError(f"spec must be an object, got {type(raw).__name__}")
        spec = cls(
            intent=str(raw.get("intent") or "filter").lower(),
            name=(str(raw["name"]).strip() or None) if raw.get("name") else None,
            subsets=[str(s) for s in (raw.get("subsets") or [])],
            filters=list(raw.get("filters") or []),
            order_by=raw.get("order_by") or None,
            limit=int(raw.get("limit") or 10),
            aggregate=raw.get("aggregate") or None,
            gaia_only=bool(raw.get("gaia_only", True)),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if self.intent not in {"lookup", "filter", "extreme", "aggregate"}:
            raise SpecError(f"unknown intent {self.intent!r}")

        # Normalise "extreme" into an explicit ordering over the field
        # the question is actually about -- see `infer_extreme_order`.
        #
        # The ordering used to be applied inside compile_spec alone, which
        # left `order_by` as None on the spec and broke two unrelated
        # things downstream: the unconstrained-query guard saw no name, no
        # filter and no ordering and dropped every row, and the
        # superlative annotation that grounds "this is the most distant"
        # was skipped, so the claim came back unsupported and the answer
        # was withheld.
        #
        # The symptom was oddly specific -- every "farthest" question
        # failed while every "nearest" one passed -- because a question
        # phrased "closest to the Sun" makes the planner emit an explicit
        # ascending order_by, taking the working path. One representation
        # for one concept avoids the whole class.
        if self.intent == "extreme" and not self.order_by:
            self.order_by = dict(DEFAULT_ORDER)
        for subset in self.subsets:
            if subset not in VALID_SUBSETS:
                raise SpecError(
                    f"unknown subset {subset!r}; valid: {', '.join(VALID_SUBSETS)}"
                )
        for flt in self.filters:
            if not isinstance(flt, dict):
                raise SpecError(f"filter must be an object, got {flt!r}")
            fld = flt.get("field")
            op = str(flt.get("op", "")).lower()
            if fld not in FILTERABLE:
                raise SpecError(
                    f"field {fld!r} is not filterable; valid: "
                    f"{', '.join(sorted(FILTERABLE))}"
                )
            if op not in OPERATORS:
                raise SpecError(f"operator {op!r} not allowed")
            if op == "between":
                lo, hi = flt.get("low"), flt.get("high")
                if lo is None or hi is None:
                    raise SpecError("between requires 'low' and 'high'")
            elif flt.get("value") is None:
                raise SpecError(f"filter on {fld!r} requires 'value'")
        if self.order_by:
            fld = self.order_by.get("field")
            if fld not in FILTERABLE:
                raise SpecError(f"cannot order by {fld!r}")
            if str(self.order_by.get("dir", "asc")).lower() not in {"asc", "desc"}:
                raise SpecError("order direction must be asc or desc")
        if self.aggregate:
            fn = str(self.aggregate.get("fn", "")).lower()
            if fn not in AGGREGATES:
                raise SpecError(f"aggregate {fn!r} not allowed")
            if fn != "count" and self.aggregate.get("field") not in FILTERABLE:
                raise SpecError(f"cannot aggregate {self.aggregate.get('field')!r}")
        # A hard cap: an answer cannot faithfully summarise 10,000 rows,
        # and a huge result set just dilutes the prompt.
        self.limit = max(1, min(int(self.limit), CONFIG.retrieval.max_rows))

        # A superlative needs the extreme row, not ten of them. Planners
        # routinely emit `limit: 10` for "the most distant X", and since
        # the top row already carries a ranking statement ("the highest
        # dist_pc of the 2,187 matching rows"), rows 2-10 are never cited
        # and contribute nothing. Measured over a 50-case run, the
        # superlative kinds retrieved 13-16 items and cited 1.2 -- the
        # worst citation coverage of any kind, and they were also the
        # slowest cases, because every extra row lengthens the prompt.
        #
        # Three rather than one, so a tie or near-tie at the top is
        # visible to the answer rather than hidden behind a LIMIT.
        if self.intent == "extreme":
            self.limit = min(self.limit, EXTREME_ROW_CAP)


# --- compilation ------------------------------------------------------


def compile_spec(spec: QuerySpec) -> tuple[str, list[Any]]:
    """Compile a validated spec into parameterised SQL."""
    spec.validate()
    where: list[str] = []
    params: list[Any] = []

    if spec.gaia_only and not spec.subsets:
        where.append("is_gaia = TRUE")
    if spec.subsets:
        where.append(f"subset IN ({','.join('?' * len(spec.subsets))})")
        params.extend(spec.subsets)

    if spec.name:
        # Match a human name, a catalogue label or a raw source_id.
        where.append(
            "(lower(proper_name) = lower(?) OR lower(label) = lower(?) "
            "OR source_id = ? OR lower(proper_name) LIKE lower(?) "
            "OR lower(label) LIKE lower(?))"
        )
        params.extend(
            [spec.name, spec.name, spec.name, f"%{spec.name}%", f"%{spec.name}%"]
        )

    for flt in spec.filters:
        fld = flt["field"]
        op = str(flt["op"]).lower()
        if op == "between":
            where.append(f"{fld} BETWEEN ? AND ?")
            params.extend([flt["low"], flt["high"]])
        else:
            where.append(f"{fld} {op} ?")
            params.append(flt["value"])

    clause = f" WHERE {' AND '.join(where)}" if where else ""

    if spec.intent == "aggregate" and spec.aggregate:
        fn = str(spec.aggregate["fn"]).lower()
        if fn == "count":
            select = "count(*) AS value"
        else:
            select = f"{fn}({spec.aggregate['field']}) AS value"
        return (
            f"SELECT {select}, count(*) AS n_rows FROM stars{clause}",
            params,
        )

    # No `extreme` special case here: validate() has already turned it
    # into an explicit order_by, so this is the single place ordering is
    # read from and every consumer sees the same spec.
    order = ""
    if spec.order_by:
        direction = str(spec.order_by.get("dir", "asc")).upper()
        order = f" ORDER BY {spec.order_by['field']} {direction}"

    cols = ", ".join(SELECTABLE)
    return (f"SELECT {cols} FROM stars{clause}{order} LIMIT {spec.limit}", params)


# --- retriever --------------------------------------------------------

PLANNER_SYSTEM = """You translate an astronomy question into a JSON query spec \
for a Gaia star catalogue. You never answer the question and you never invent \
numbers.

Return ONLY a JSON object with these optional keys:
  intent    : "lookup" (a named object), "filter" (rows matching conditions),
              "extreme" (most/least by a field), or "aggregate" (a statistic)
  name      : the object name or source_id, if the question names one
  subsets   : any of ["gaia_gcns","gaia_cepheid","gaia_rrlyrae",
              "gaia_dr3_luminous","apogee","hyg"]
  filters   : [{"field":..,"op":"<|<=|>|>=|=|!=|between","value":..}]
              or {"field":..,"op":"between","low":..,"high":..}
  order_by  : {"field":..,"dir":"asc|desc"}
  limit     : integer, at most 40
  aggregate : {"fn":"count|avg|min|max|sum|median|stddev","field":..}
  gaia_only : true unless the question is explicitly about non-Gaia data

Filterable fields: dist_pc, dist_ly, dist_kpc, ra_deg, dec_deg, color_bp_rp, \
lum_lsun, subset, constellation, spectral_type, is_gaia.

Distances are stored in parsecs. If the question asks in light-years, filter \
on dist_ly; in kiloparsecs, on dist_kpc. Map object types to subsets: \
Cepheids -> gaia_cepheid, RR Lyrae -> gaia_rrlyrae, nearby stars -> \
gaia_gcns, luminous/distant OB stars -> gaia_dr3_luminous.

If the question needs no catalogue rows at all (it asks about methods, \
reliability or provenance), return {"intent":"filter","limit":0}."""


@dataclass
class StructuredResult:
    spec: QuerySpec
    sql: str
    params: list
    evidence: list[Evidence]
    aggregate_value: float | None = None
    row_count: int = 0
    error: str | None = None
    #: True when the plan had no name, filter or ordering, so no rows
    #: were fetched on purpose. Distinct from "the query matched nothing".
    unconstrained: bool = False
    #: True when a named lookup found nothing in the Gaia subsets and the
    #: search was widened to the whole catalogue. The rows that come back
    #: are then NOT Gaia data, which the answer has to say.
    widened_beyond_gaia: bool = False


class StructuredRetriever:
    def __init__(self, db_path: Path | None = None):
        db_path = db_path or CONFIG.paths.duckdb
        if not db_path.exists():
            raise FileNotFoundError(
                f"catalogue db missing at {db_path} -- run "
                "`python -m gaia_rag.ingest.catalog`"
            )
        self.con = duckdb.connect(str(db_path), read_only=True)
        self._provenance = self._load_provenance()

    def _load_provenance(self) -> dict[tuple[str, str], dict]:
        rows = self.con.execute(
            "SELECT subset, field, status, method, rel_uncertainty, caveat, "
            "source_doc FROM field_provenance"
        ).fetchall()
        return {
            (r[0], r[1]): {
                "status": r[2],
                "method": r[3],
                "rel_uncertainty": r[4],
                "caveat": r[5],
                "source_doc": r[6],
            }
            for r in rows
        }

    def plan(self, question: str, llm) -> QuerySpec:
        raw = llm.complete_json(PLANNER_SYSTEM, question, temperature=0.0)
        spec = QuerySpec.from_dict(raw)

        # Reconcile the model's subset choice with the question's own
        # words. Two observed failures, in opposite directions:
        #
        #  - Asked how reliable RR Lyrae distances are, the planner
        #    ignored the mapping stated in its prompt, filtered nothing,
        #    and returned ten arbitrary GCNS stars.
        #  - Asked what distance Gaia gives for Betelgeuse, it invented
        #    `subset = 'gaia_gcns'` out of nothing, which excluded the one
        #    row that answers the question.
        #
        # The question's own words always win when it names a subset.
        #
        # When it names none, what to do depends on the query's shape,
        # not on which direction the error goes. An earlier version
        # reasoned that narrowing is safer than widening, because a
        # too-narrow filter yields an obviously odd result while a
        # too-broad one yields a plausible number for a different
        # question. That is wrong for superlatives and aggregates, where
        # the SCOPE IS THE ANSWER:
        #
        #  - Clearing a correct filter answered "how many classical
        #    Cepheids?" with 120,000 instead of 2,187.
        #  - Keeping an invented one answered "the most distant Gaia
        #    star" with 5,000 pc instead of 29,999.2 pc, because the
        #    planner silently added `subset = 'gaia_dr3_luminous'`.
        #
        # Both are plausible-looking and both are badly wrong, so neither
        # direction is inherently safe. For `extreme` and `aggregate` the
        # scope therefore comes from the question alone and the model may
        # not narrow it unprompted. For `lookup` and `filter` the scope
        # merely selects which rows are shown, and an empty named lookup
        # is already recovered by the widening in `run()`.
        detected = detect_subsets(question)
        if detected:
            spec.subsets = sorted(detected)
            spec.filters = [
                f
                for f in spec.filters
                if f.get("field") != "subset"
                or str(f.get("value")) in detected
            ]
        elif spec.intent in ("extreme", "aggregate"):
            spec.subsets = []
            spec.filters = [
                f for f in spec.filters if f.get("field") != "subset"
            ]

        # Order by the field the question actually names. A
        # superlative's ordering IS its definition: "the reddest
        # Cepheid" ordered by distance answers a different question,
        # and does so silently. An unambiguous signal in the wording
        # therefore overrides the model's choice, exactly as it does
        # for subsets.
        if spec.intent == "extreme" or spec.order_by:
            inferred = infer_extreme_order(question)
            if inferred is not None:
                spec.order_by = inferred

        spec.validate()
        return spec

    def run(self, spec: QuerySpec) -> StructuredResult:
        if spec.intent != "aggregate" and spec.limit <= 0:
            return StructuredResult(spec, "", [], evidence=[], row_count=0)

        # An unconstrained row query is not evidence. With no name, no
        # filter and no aggregate, `LIMIT n` returns whichever rows the
        # storage layer happens to yield first -- an arbitrary sample that
        # answers nothing but looks authoritative in a prompt, and
        # invites exactly the kind of claim that cites a real row for a
        # fact the row does not establish.
        if (
            spec.intent != "aggregate"
            and not spec.name
            and not spec.filters
            and not spec.order_by
        ):
            return StructuredResult(
                spec,
                "",
                [],
                evidence=[],
                row_count=0,
                error=None,
                unconstrained=True,
            )
        try:
            sql, params = compile_spec(spec)
        except SpecError as exc:
            return StructuredResult(
                spec, "", [], evidence=[], error=f"invalid spec: {exc}"
            )

        cursor = self.con.execute(sql, params)
        columns = [d[0] for d in cursor.description]
        rows = [dict(zip(columns, r)) for r in cursor.fetchall()]

        # A named object absent from the Gaia subsets may still be in the
        # catalogue under another survey, and saying so is far more useful
        # than abstaining. Asked what distance Gaia gives for Betelgeuse,
        # the pipeline found nothing and fell silent -- but the honest
        # answer is that Betelgeuse is present only via HYG/Hipparcos and
        # Gaia gives no distance for it here. Row evidence carries the
        # "NOT from Gaia" marker, so widening the search cannot be
        # mistaken for a Gaia measurement.
        widened = False
        if not rows and spec.name and spec.intent != "aggregate":
            # Drop every subset constraint as well as the Gaia-only flag.
            # The question being answered here is "does this object exist
            # anywhere in the catalogue?", and a leftover subset filter
            # answers a narrower one -- which is how a Betelgeuse lookup
            # carrying an invented `subset = 'gaia_gcns'` filter still
            # found nothing after widening.
            widened_spec = replace(
                spec,
                gaia_only=False,
                subsets=[],
                filters=[f for f in spec.filters if f.get("field") != "subset"],
            )
            wide_sql, wide_params = compile_spec(widened_spec)
            cursor = self.con.execute(wide_sql, wide_params)
            columns = [d[0] for d in cursor.description]
            rows = [dict(zip(columns, r)) for r in cursor.fetchall()]
            if rows:
                widened = True
                sql, params = wide_sql, wide_params

        if spec.intent == "aggregate" and spec.aggregate:
            if not rows:
                return StructuredResult(spec, sql, params, evidence=[], row_count=0)
            value = rows[0].get("value")
            n_rows = int(rows[0].get("n_rows") or 0)
            fn = str(spec.aggregate["fn"]).lower()
            target = spec.aggregate.get("field", "rows")
            scope = ", ".join(spec.subsets) if spec.subsets else (
                "all Gaia subsets" if spec.gaia_only else "the whole catalogue"
            )
            # The counted noun is stated explicitly. Left as a bare
            # "count = 2187", this evidence was restated as "2187
            # confirmed exoplanets" -- the value and the subset were both
            # correct, so no value-level or subset-level check could see
            # the relabelling. One row is one star; saying so is the only
            # defence inside the evidence itself.
            counted = (
                "stars (one catalogue row is one star)"
                if fn == "count"
                else f"the {target} column of stars"
            )
            text = (
                f"Aggregate {fn}({target}) over {scope} with the stated "
                f"filters = {value!r}. This counts {counted}, computed from "
                f"{n_rows:,} matching catalogue rows. It is not a count of "
                "any other kind of object."
            )
            numeric = []
            if isinstance(value, (int, float)):
                from ..units import distance_renderings

                unit = "pc" if target.endswith("_pc") else "count" if fn == "count" else ""
                renderings = (
                    distance_renderings(float(value))
                    if target == "dist_pc"
                    else {unit or "value": float(value)}
                )
                numeric = [
                    NumericFact(
                        field_name=f"{fn}_{target}",
                        value=float(value),
                        unit=unit or "value",
                        renderings=renderings,
                        status="derived",
                    ),
                    NumericFact(
                        field_name="n_rows",
                        value=float(n_rows),
                        unit="count",
                        renderings={"count": float(n_rows)},
                        status="derived",
                    ),
                ]
            ev = Evidence(
                evidence_id=f"agg:{fn}:{target}",
                kind="row",
                text=text,
                citation="DuckDB aggregate over the Galaxy web catalogue",
                numeric_facts=numeric,
                metadata={"sql": sql, "n_rows": n_rows},
            )
            return StructuredResult(
                spec,
                sql,
                params,
                evidence=[ev],
                aggregate_value=value if isinstance(value, (int, float)) else None,
                row_count=n_rows,
            )

        if not rows:
            # A verified empty result is a FACT, and it needs an id so a
            # claim can cite it.
            #
            # Previously this returned no evidence and left the zero-row
            # fact in a retrieval note. Notes reach the prompt but are not
            # citable, so "the query matched zero rows, therefore there
            # are none" cited a document that does not say that, the
            # judge correctly called it unsupported, and two
            # evidence-of-absence cases were withheld -- for answers that
            # were right. Absence is only answerable if it is evidence.
            scope = ", ".join(spec.subsets) if spec.subsets else (
                "all Gaia subsets" if spec.gaia_only else "the whole catalogue"
            )
            described = "; ".join(
                f"{f['field']} {f['op']} "
                + (
                    f"{f.get('low')}..{f.get('high')}"
                    if str(f.get("op")).lower() == "between"
                    else str(f.get("value"))
                )
                for f in spec.filters
            ) or "no additional filters"
            empty = Evidence(
                evidence_id="query:zero_rows",
                kind="row",
                text=(
                    f"The catalogue was queried over {scope} with these "
                    f"conditions: {described}. The query executed "
                    f"successfully and matched EXACTLY ZERO rows. This is a "
                    f"verified negative result: there are no such objects in "
                    f"the catalogue, which is a definite answer, not missing "
                    f"data."
                ),
                citation=f"DuckDB query over the Galaxy web catalogue: {sql}",
                numeric_facts=[
                    NumericFact(
                        field_name="matching_rows",
                        value=0.0,
                        unit="count",
                        renderings={"count": 0.0},
                        status="measured",
                    )
                ],
                metadata={"subset": spec.subsets[0] if spec.subsets else None,
                          "is_gaia": spec.gaia_only, "sql": sql},
            )
            return StructuredResult(
                spec, sql, params, evidence=[empty], row_count=0
            )

        evidence = [row_to_evidence(r, self._provenance) for r in rows]

        # Ground superlatives. A single row cannot establish "this is the
        # closest Cepheid" -- and the entailment judge correctly called
        # that claim unsupported, which withheld an otherwise perfect
        # answer. The ordering is a fact about the query, so it is
        # recorded in the evidence rather than left for the reader to
        # assume.
        if evidence and spec.order_by:
            field_name = spec.order_by["field"]
            ascending = str(spec.order_by.get("dir", "asc")).lower() == "asc"
            total = self._count_matching(spec)
            extreme = "lowest" if ascending else "highest"

            # A field with one distinct value cannot be ranked, and the
            # top row is not a maximum -- it is an arbitrary pick from a
            # tie. RR Lyrae are the real case: all 67,813 rows carry
            # luminosity 45 and colour 0.35, because the distance method
            # assumed a fixed absolute magnitude and intrinsic colour.
            # Asked which RR Lyrae is most luminous, the pipeline named
            # one, and the caveat document saying otherwise was retrieved
            # and ignored. Stating it in the row's own evidence makes it
            # much harder to overlook than a separate document.
            #
            # Checked by querying the data rather than hardcoding the
            # subset, so it also covers any field that becomes degenerate
            # under a narrow filter.
            if self._distinct_count(spec, field_name) == 1:
                shared = evidence[0].numeric_facts
                value = next(
                    (f.value for f in shared if f.field_name == field_name), None
                )
                evidence[0].text += (
                    f" IMPORTANT: all {total:,} rows matching this query share "
                    f"the SAME {field_name}"
                    + (f" ({value:g})" if value is not None else "")
                    + ". There is therefore no single highest or lowest one, "
                    "and this row is an arbitrary member of a tie, not a "
                    f"maximum. Ranking by {field_name} is meaningless here and "
                    "the answer should say so rather than name a star."
                )
            else:
                evidence[0].text += (
                    f" This row has the {extreme} {field_name} of the "
                    f"{total:,} catalogue rows matching this query, so it is "
                    "the "
                    + (
                        "nearest"
                        if ascending and field_name.startswith("dist")
                        else "farthest"
                        if field_name.startswith("dist")
                        else extreme
                    )
                    + " one under these filters."
                )

        return StructuredResult(
            spec,
            sql,
            params,
            evidence=evidence,
            row_count=len(rows),
            widened_beyond_gaia=widened,
        )

    def _distinct_count(self, spec: QuerySpec, field: str) -> int:
        """Distinct values of `field` among the rows a spec matches.

        `field` always comes from an already-validated `order_by`, so it
        is one of the whitelisted column names and safe to interpolate.
        """
        if field not in FILTERABLE:
            return 0
        counting = replace(
            spec, intent="aggregate", aggregate={"fn": "count"}, order_by=None
        )
        sql, params = compile_spec(counting)
        sql = sql.replace("count(*) AS value", f"count(DISTINCT {field}) AS value", 1)
        row = self.con.execute(sql, params).fetchone()
        return int(row[0]) if row else 0

    def _count_matching(self, spec: QuerySpec) -> int:
        """Rows matching a spec's filters, ignoring its ordering and limit."""
        counting = replace(
            spec, intent="aggregate", aggregate={"fn": "count"}, order_by=None
        )
        sql, params = compile_spec(counting)
        row = self.con.execute(sql, params).fetchone()
        return int(row[0]) if row else 0

    def provenance_for(self, subsets: list[str]) -> list[dict]:
        return [
            {"subset": s, "field": f, **v}
            for (s, f), v in self._provenance.items()
            if not subsets or s in subsets
        ]

    def close(self) -> None:
        self.con.close()
