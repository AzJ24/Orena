"""Scoring + reflective feedback for the FOCUS frame track.

This is the reusable core GEPA optimizes against. `score_and_feedback` turns one
(prediction, reference) pair into:

  * a scalar score in [0, 1] (higher is better -- what GEPA sums/averages), and
  * a natural-language `feedback` string -- the "Actionable Side Information"
    GEPA reflects on. The whole point of GEPA over RL is that this feedback is
    rich, not a bare reward: when the deterministic parser rejects an answer it
    hands back the exact reason ("Number format requires a non-negative integer,
    got '2 clips'"), which is precisely the signal a reflection LM needs to
    rewrite the prompt so the model stops emitting units.

Scoring mirrors `focus.evaluation.Evaluator._evaluate_single` exactly (same
parsers, same compare), so a GEPA gain is a real leaderboard gain and not an
artefact of a looser metric. Judge formats (open_ended / matching /
multiple_choice) need an LLM judge; pass one via `judge`, or leave it None to
have those examples reported as unscorable so the caller can filter them out.
"""

from __future__ import annotations

from dataclasses import dataclass

from focus.data.formats import JUDGE_FORMATS, get_format_class

# Per-format one-line shape rule, injected into feedback on a parse failure so
# the reflection LM sees the exact contract the answer must satisfy.
_SHAPE_HINT = {
    "binary": "exactly 'yes' or 'no' (nothing else, no period).",
    "number": "digits only, e.g. 0 or 1 or 2 (no words, no units).",
    "fo_class": "registered class name(s), comma-separated, or exactly 'none'.",
    "time": "one or more hh:mm:ss timestamps, comma-separated.",
    "percentage": "a number, optionally with '%'.",
    "open_ended": "a short free-text phrase.",
    "matching": "a short phrase matching the expected pattern.",
    "multiple_choice": "exactly one of the offered options, verbatim.",
}


@dataclass
class Scored:
    score: float
    feedback: str
    scorable: bool = True  # False only when a judge is required but absent


def score_and_feedback(
    pred: str,
    ref_format: str,
    ref_answer: str,
    format_kwargs: dict | None = None,
    question: str | None = None,
    judge=None,
    req=None,
    ref=None,
) -> Scored:
    """Score one prediction against its reference and explain why.

    `judge` (a `focus.evaluation.judges.Judge`) is only consulted for judge
    formats; deterministic formats never touch it, so the whole GEPA loop can run
    with no judge and no API key as long as the dataset is filtered to
    deterministic formats.
    """
    fmt = get_format_class(ref_format)(**(format_kwargs or {}))
    hint = _SHAPE_HINT.get(ref_format, "the required form.")

    try:
        gold = fmt.read(ref_answer)
    except ValueError:
        # A reference we cannot parse is scored 0 by the official evaluator too;
        # flag it so the dataset builder can drop it rather than train on noise.
        return Scored(0.0, f"reference answer {ref_answer!r} is itself unparseable", scorable=False)

    try:
        pred_parsed = fmt.read(pred)
    except ValueError as e:
        return Scored(
            0.0,
            f"FORMAT ERROR: {e} The answer must be {hint} "
            f"The correct answer was {ref_answer!r}.",
        )

    if ref_format in JUDGE_FORMATS:
        if judge is None:
            return Scored(0.0, "needs an LLM judge (no judge configured)", scorable=False)
        from focus.evaluation.judges import majority_vote
        correct = bool(majority_vote([judge], req, ref_answer, pred))
        if correct:
            return Scored(1.0, f"Correct (judge): matched {ref_answer!r}.")
        return Scored(
            0.0,
            f"WRONG: the judge ruled {pred!r} does not match the reference "
            f"{ref_answer!r} for the question {question!r}.",
        )

    if fmt.compare(gold, pred_parsed):
        return Scored(1.0, f"Correct: {ref_answer!r}.")
    return Scored(
        0.0,
        f"WRONG VALUE: {pred!r} is a validly-formatted answer but the correct "
        f"answer is {ref_answer!r} (format {ref_format}). The format was fine; "
        f"the content was wrong.",
    )
