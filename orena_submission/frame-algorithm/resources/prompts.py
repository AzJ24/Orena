"""The 'direct' system prompt and answer parser this checkpoint was trained
and evaluated with (orena_sft/prompts.py). Kept byte-identical here so the
submission's prompt matches training exactly.
"""

from __future__ import annotations

import re

from focus.foreign_objects import FO_DEFINITION, FOType

_ANSWER_RULES = """\
Rules for the answer:
- Write the value only. No sentence, no explanation, no units, no trailing
  period, and never repeat the question.
- Asks yes or no -> write exactly: yes   or   no
- Asks how many / for a count -> write digits only, e.g. 0 or 1 or 2.
- Asks which foreign object class(es) -> write class names exactly as spelled
  in the list above, comma-separated (e.g. Clip, Sponge), or exactly: none
  Never answer with a generic description such as "surgical instrument".
- Asks for a time -> write hh:mm:ss.
- Lists options to choose from -> copy exactly one of those options, verbatim.
- Anything else -> a short phrase, at most a few words.

If you are unsure, still commit to your single best answer in the required
form. An empty, hedged, or explanatory answer is scored as wrong."""

_DIRECT_SHAPE = """\
Reply with the answer and nothing else -- no reasoning, no preamble, no
explanation, no restating the question. A single short line."""


def build_system_prompt() -> str:
    class_names = ", ".join(FOType.names())
    objects_block = (
        f"{FO_DEFINITION.strip()}\n\n"
        f"The foreign object classes are exactly: {class_names}."
    )
    return (
        "You are a surgical video analysis assistant. You are shown one frame "
        "from a laparoscopic procedure and asked a single question about it.\n\n"
        f"{objects_block}\n\n"
        f"{_DIRECT_SHAPE}\n\n"
        f"{_ANSWER_RULES}\n"
    )


_ANSWER_RE = re.compile(r"[*_`\s]*ANSWER[*_`\s]*[:\-][^\S\n]*(.*)", re.IGNORECASE)
_REASONING_RE = re.compile(r"^[*_`\s]*REASONING[*_`\s]*[:\-].*$", re.IGNORECASE | re.MULTILINE)
_LEADING_ANSWER_RE = re.compile(r"^answer\b[:\-]?\s*", re.IGNORECASE)
_STRIP_CHARS = " \t\n\"'`*_.:"
_OPEN_ENDED_MAX_LEN = 300


def extract_answer(raw: str) -> str:
    """Reduce a generation to the bare value to be scored (mirrors orena_sft/prompts.py)."""
    text = (raw or "").strip()
    if not text:
        return ""

    matches = list(_ANSWER_RE.finditer(text))
    if matches:
        answer = matches[-1].group(1)
    else:
        without_reasoning = _REASONING_RE.sub("", text)
        lines = [ln for ln in without_reasoning.splitlines() if ln.strip()]
        answer = _LEADING_ANSWER_RE.sub("", lines[-1] if lines else text)

    answer = answer.strip().strip(_STRIP_CHARS).strip()
    if len(answer) > _OPEN_ENDED_MAX_LEN:
        answer = answer[:_OPEN_ENDED_MAX_LEN].rstrip()
    return answer
