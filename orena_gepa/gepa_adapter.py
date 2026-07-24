"""GEPAAdapter for the FOCUS frame track.

The single integration point between GEPA and this task. The system under
optimization is: (system prompt) + one surgical frame + one question -> answer,
scored by the official FOCUS parsers. GEPA evolves exactly one component,
`"system_prompt"`, whose seed is `prompts.build_system_prompt(...)`.

Responsibilities implemented (see gepa.core.adapter.GEPAAdapter):
  * evaluate           -- run the VLM under the candidate prompt, score every
                          example with `metric.score_and_feedback`, and (when
                          asked) capture a trajectory carrying the feedback.
  * make_reflective_dataset -- reshape those trajectories into the
                          {Inputs, Generated Outputs, Feedback} records the
                          reflection LM reads to propose a better prompt.

`propose_new_texts` is left to GEPA's default proposer (driven by
`reflection_lm`), so nothing model-specific about prompt rewriting lives here.
"""

from __future__ import annotations

from typing import Any

from gepa.core.adapter import EvaluationBatch, GEPAAdapter

from focus.data.data_models import Request

from metric import score_and_feedback

COMPONENT = "system_prompt"


def _to_request(ex: dict) -> Request:
    """Build a focus Request from a dataset record so the LLM judge (judge
    formats) gets the real question/procedure/timing context it reads."""
    return Request(
        qID=ex["qID"],
        videoID=ex.get("videoID", ""),
        start_time=float(ex.get("start_time", 0.0)),
        end_time=float(ex.get("end_time", 0.0)),
        procedure_type=ex.get("procedure_type", ""),
        question=ex["question"],
    )


class FocusFrameAdapter(GEPAAdapter):
    def __init__(self, runner, judge=None):
        """`runner` exposes .generate(system_prompt, examples) -> [(answer, raw)].
        `judge` is only needed if the dataset includes judge formats."""
        self.runner = runner
        self.judge = judge

    def evaluate(self, batch: list[dict], candidate: dict[str, str],
                 capture_traces: bool = False) -> EvaluationBatch:
        system_prompt = candidate[COMPONENT]
        generations = self.runner.generate(system_prompt, batch)

        outputs, scores, trajectories = [], [], [] if capture_traces else None
        for ex, (answer, raw) in zip(batch, generations):
            res = score_and_feedback(
                pred=answer,
                ref_format=ex["format"],
                ref_answer=ex["answer"],
                format_kwargs=ex.get("format_kwargs"),
                question=ex["question"],
                judge=self.judge,
                req=_to_request(ex) if self.judge is not None else ex,
            )
            outputs.append(answer)
            scores.append(res.score)
            if capture_traces:
                trajectories.append({
                    "data": {"input": self._render_input(ex)},
                    "answer": answer,
                    "full_assistant_response": raw,
                    "feedback": res.feedback,
                    "score": res.score,
                })
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(self, candidate: dict[str, str], eval_batch: EvaluationBatch,
                                components_to_update: list[str]) -> dict[str, list[dict[str, Any]]]:
        assert components_to_update == [COMPONENT], components_to_update
        records = [
            {
                "Inputs": traj["data"]["input"],
                "Generated Outputs": traj["answer"],
                "Feedback": traj["feedback"],
            }
            for traj in (eval_batch.trajectories or [])
        ]
        if not records:
            raise ValueError("no trajectories captured; call evaluate(capture_traces=True)")
        return {COMPONENT: records}

    @staticmethod
    def _render_input(ex: dict) -> str:
        # The image is not text; describe it and pass the question + expected
        # answer format so the reflection LM can reason about failures.
        return (
            f"[surgical frame image]\n"
            f"Question: {ex['question']}\n"
            f"Expected answer format: {ex['format']}"
        )
