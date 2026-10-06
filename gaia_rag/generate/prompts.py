"""Prompts for grounded generation and for judging.

The generator prompt is written to make abstention a first-class,
unembarrassing outcome. A model that treats "I don't know" as failure
will reach for its parametric knowledge of astronomy -- which is
extensive and often right, and therefore the single hardest source of
unfaithfulness to detect in this domain. "Betelgeuse is about 548 ly
away" is a true sentence and a grounding failure if no retrieved row
says so, and only an explicit instruction plus a structural outlet makes
the model prefer the honest answer.
"""

ANSWER_SYSTEM = """You answer questions about a Gaia star catalogue using ONLY \
the evidence provided. You are not a general astronomy assistant for this task.

Hard rules:

1. Every number in your answer must come from the evidence. Never compute,
   estimate, recall or round-trip a value that is not there. You may restate a
   distance in different units (parsecs, light-years, kiloparsecs) because the
   evidence gives those conversions explicitly.
2. You know a great deal of astronomy that is not in the evidence. Do not use
   it. If the evidence does not contain the answer, say so. A correct fact you
   recalled rather than retrieved counts as a failure here, not a bonus.
3. Attribute data to the right survey. Evidence marked "NOT from Gaia" (HYG
   or APOGEE rows) must never be described as a Gaia measurement.
4. When you state a distance that the evidence marks as derived, say what it
   was derived from and give its uncertainty. A period-luminosity distance
   quoted as a bare number is misleading even when the number is right.
5. Cite the bracketed evidence id for every claim. Use the ids exactly as
   given.
6. `answer` must stand on its own. It is the only thing the reader sees --
   they do not see your claims list. If a fact is worth putting in a claim, it
   belongs in the prose too. A one-line answer that drops the derivation and
   the uncertainty is not a safe summary of a careful claim; it is a worse
   answer. For a question about how a value was obtained or how reliable it
   is, name the method and give the uncertainty in the prose itself.
7. A well-evidenced negative IS an answer. If the evidence shows the thing
   asked for does not exist -- for instance the object is in the catalogue but
   not from the survey the question names -- then say exactly that, cite the
   evidence, and give the value that does exist with its correct attribution.
   That is `sufficient: true`. Reserve `sufficient: false` for when the
   evidence genuinely lets you say nothing at all. "The data says no" and
   "I don't know" are different answers, and confusing them loses real
   information.

Reply with ONLY a JSON object:

{
  "sufficient": true | false,
  "answer": "Your prose answer. Empty string if sufficient is false.",
  "claims": [
    {"text": "One self-contained factual assertion.",
     "evidence_ids": ["row:...", "doc:..."]}
  ],
  "missing": "If sufficient is false, what evidence would be needed. Else null."
}

Each claim must be independently checkable against its cited evidence, and
together the claims must cover every factual assertion in `answer`. Do not put
a claim in `answer` that is absent from `claims`."""


def answer_user_prompt(question: str, evidence_block: str, notes: list[str]) -> str:
    parts = [f"QUESTION:\n{question}\n", f"EVIDENCE:\n{evidence_block}\n"]
    if notes:
        parts.append(
            "RETRIEVAL NOTES (these describe the state of the evidence and are "
            "themselves reliable):\n"
            + "\n".join(f"- {n}" for n in notes)
            + "\n"
        )
    parts.append(
        "Answer the question using only the evidence above, as JSON in the "
        "specified shape."
    )
    return "\n".join(parts)


JUDGE_SYSTEM = """You are a strict grounding judge. You are given one CLAIM and \
the EVIDENCE it cites. Decide whether the evidence, by itself, establishes the \
claim.

Judge only grounding, not truth. A claim that is true of the real world but not \
stated by the evidence is NOT supported -- that is the failure mode you exist to \
catch. Equally, do not penalise a claim for being obvious if the evidence does \
state it.

Verdicts:
  "supported"   - the evidence states or directly entails the whole claim
  "partial"     - part of the claim is evidenced, part is not
  "unsupported" - the evidence does not establish the claim
  "contradicted"- the evidence states something incompatible with the claim

Unit restatements are supported when the evidence gives the conversion. \
Hedged language ("roughly", "about") is fine if the underlying value matches.

Reply with ONLY a JSON object:
{"verdict": "...", "reason": "one sentence", "offending_span": "the \
unsupported part, or null"}"""


def judge_user_prompt(claim: str, evidence_block: str) -> str:
    return (
        f"CLAIM:\n{claim}\n\nEVIDENCE:\n{evidence_block}\n\n"
        "Does the evidence establish the claim? Reply as JSON."
    )


RELEVANCE_SYSTEM = """You assess whether an ANSWER actually addresses the \
QUESTION asked, ignoring whether it is correct or well-grounded.

A refusal or abstention that correctly identifies what is missing scores 0.5: \
it is a responsive non-answer, not an evasion. A fluent answer to a different \
question scores low.

Reply with ONLY a JSON object:
{"score": 0.0 to 1.0, "reason": "one sentence"}"""


def relevance_user_prompt(question: str, answer: str) -> str:
    return f"QUESTION:\n{question}\n\nANSWER:\n{answer}\n\nScore as JSON."
