"""The PROCEDURE system prompt.

Kept here rather than as a third arm of `../orena_sft/prompts.py` for one reason: that
module is load-bearing for two already-shipped models, and a mid-competition edit to it
risks the FRAME and SEGMENT runs for no gain. The thing §2.5 of the plan actually wanted
to protect against -- a forked FO vocabulary drifting from the registry -- is avoided by
reading `FOType.names()` live, exactly as the shared module does. `extract_answer` is
track-agnostic and is imported, not reimplemented.

What differs from the segment prompt:
  * the input is a procedure observed from its START up to a stated time, and nothing
    after that time is observable -- roughly half the questions are "so far" questions
    and the model must not extrapolate past the window;
  * a clock is burned into every frame, so a timestamp can be read directly instead of
    interpolated between the `<... seconds>` markers;
  * counting guidance, because `number` triples in share here and answers reach 76.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "orena_sft"))

from focus.foreign_objects import FO_DEFINITION, FO_DEFINITIONS_FILE, FOType  # noqa: E402

from prompts import extract_answer  # noqa: E402,F401  (re-exported for the evaluator)

_INTRO = (
    "You are a surgical video analysis assistant. You are shown a laparoscopic "
    "procedure from its very beginning up to a stated point in time, and asked a "
    "single question about it.\n\n"
    "The procedure is given as frames in chronological order, sampled across the whole "
    "observed period. Each frame carries its timestamp twice: burned into the picture "
    "as hh:mm:ss, and written before it as <SECONDS seconds> -- a frame marked "
    "<3742.0 seconds> was taken 3742.0 seconds (01:02:22) after the procedure began. "
    "Both clocks count from the START OF THE PROCEDURE. Read the burned-in clock when "
    "you can; it is exact.\n\n"
    "Consecutive frames can be a minute or more apart, so events happen between them. "
    "You are shown only what occurred up to the stated end time -- nothing later exists "
    "for you to reason about."
)

_DIRECT_SHAPE = (
    "Reply with the answer and nothing else -- no reasoning, no preamble, no\n"
    "explanation, no restating the question. A single short line."
)

_ANSWER_RULES = """\
Rules for the answer:
- Write the value only. No sentence, no explanation, no units, no trailing
  period, and never repeat the question.
- Asks yes or no -> write exactly: yes   or   no
- Asks how many / for a count -> write digits only, e.g. 0 or 3 or 12.
  Count over the WHOLE observed period, not just one frame, unless the question
  names a single time point. Distinct instances of the same class each count once.
- Asks which foreign object class(es) -> write class names exactly as spelled
  in the list above, comma-separated (e.g. Clip, Sponge), or exactly: none
  Never answer with a generic description such as "surgical instrument".
- Asks when something happens -> write hh:mm:ss, read off the clock burned into
  the frames. Those clocks count from the START OF THE PROCEDURE, not from any
  point mentioned in the question. If the event falls between two frames, give
  your best estimate between their timestamps rather than snapping to one.
  For several time points, separate them with commas: 00:09:19, 01:12:44
- Asks for a percentage or a share -> write the number only, e.g. 40
- Lists options to choose from -> copy exactly one of those options, verbatim.
- Anything else -> a short phrase, at most a few words.

If you are unsure, still commit to your single best answer in the required
form. An empty, hedged, or explanatory answer is scored as wrong.\
"""


def build_system_prompt(include_definitions: bool = False, style: str = "direct") -> str:
    """The procedure system prompt.

    `structured` is refused as it is on the segment track: the targets are bare
    answers, so a REASONING line would have to be stripped back off again.
    """
    if style != "direct":
        raise ValueError(f"procedure supports style='direct' only, got {style!r}")

    if include_definitions:
        objects_block = FO_DEFINITIONS_FILE.read_text().strip()
    else:
        objects_block = (f"{FO_DEFINITION.strip()}\n\n"
                         f"The foreign object classes are exactly: {', '.join(FOType.names())}.")

    return f"{_INTRO}\n\n{objects_block}\n\n{_DIRECT_SHAPE}\n\n{_ANSWER_RULES}\n"


def window_line(end_time: float) -> str:
    """Prefix stating what the model is allowed to know about.

    Every window starts at zero and only the end varies, and 46.5% of repeated
    (video, question) pairs change their answer with it -- so this is data, not
    decoration. Ten tokens.
    """
    s = int(end_time)
    return (f"Procedure observed from 00:00:00 to "
            f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}.")
