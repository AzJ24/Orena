"""Offline end-to-end test of the GEPA wiring -- no GPU, no model, no API key.

Replaces the two expensive pieces with deterministic stubs:
  * a stub task runner (no VLM): returns the gold answer only when the candidate
    prompt contains a magic marker, a wrong answer otherwise;
  * a stub reflection LM (no API): returns the seed prompt with the marker added.

If GEPA is wired correctly it must: run the real `FocusFrameAdapter.evaluate`
and `make_reflective_dataset`, feed the parser feedback to the stub reflector,
accept the improved candidate, and report a val score of 1.0. This exercises
every integration surface between this repo and GEPA except the model itself.

Usage:
    .venv/bin/python orena_gepa/selftest_loop.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import gepa

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evolution_logger import EvolutionTracker, TeeLogger, write_final_summary  # noqa: E402
from gepa_adapter import COMPONENT, FocusFrameAdapter  # noqa: E402

MARKER = "[[ANSWER-YES-NO-ONLY]]"

# Tiny binary dataset; image_path is unused by the stub runner.
_DATA = [
    {"qID": f"q{i}", "question": f"Is object {i} present? yes or no",
     "format": "binary", "answer": "yes" if i % 2 else "no",
     "format_kwargs": {}, "image_path": "n/a"}
    for i in range(8)
]


class StubRunner:
    """Stands in for VLMRunner: compliant only once the prompt has the marker."""
    def generate(self, system_prompt, examples):
        if MARKER in system_prompt:
            return [(ex["answer"], f"<raw>{ex['answer']}") for ex in examples]
        # Non-compliant: emits prose the Binary parser rejects (score 0).
        return [("I think yes.", "<raw>I think yes.") for _ in examples]


def stub_reflection_lm(prompt: str) -> str:
    # GEPA extracts the text inside ``` fences as the new instruction.
    return f"```\nAnswer with exactly 'yes' or 'no'. {MARKER}\n```"


def main():
    adapter = FocusFrameAdapter(StubRunner())
    seed = {COMPONENT: "Look at the frame and answer the question."}

    # Sanity: seed scores 0, feedback is the parser's actual rejection message.
    eb = adapter.evaluate(_DATA, seed, capture_traces=True)
    assert sum(eb.scores) == 0, eb.scores
    print("seed feedback sample:", eb.trajectories[0]["feedback"])
    assert adapter.make_reflective_dataset(seed, eb, [COMPONENT])[COMPONENT]

    out_dir = Path(tempfile.mkdtemp(prefix="gepa_selftest_"))
    tracker = EvolutionTracker(out_dir, component=COMPONENT)
    result = gepa.optimize(
        seed_candidate=seed,
        trainset=_DATA,
        valset=_DATA,
        adapter=adapter,
        reflection_lm=stub_reflection_lm,
        max_metric_calls=40,
        reflection_minibatch_size=4,
        logger=TeeLogger(out_dir / "run_log.txt"),
        callbacks=[tracker],
        display_progress_bar=False,
    )
    tracker.close()
    write_final_summary(result, out_dir, component=COMPONENT)

    best = result.best_candidate[COMPONENT]
    assert MARKER in best, "GEPA did not adopt the improved prompt"

    # Evolution artefacts must exist and record both ends of the run.
    md = (out_dir / "evolution.md").read_text()
    assert "Seed prompt" in md and "SEED → BEST diff" in md, "evolution.md incomplete"
    assert (out_dir / "evolution.jsonl").read_text().strip(), "evolution.jsonl empty"
    assert MARKER in (out_dir / "best_prompt.txt").read_text()
    print(f"\nevolution artefacts written to {out_dir}")
    print("PASS: GEPA ran the adapter, reflected on parser feedback, adopted the "
          "improved prompt (val -> 1.0), and the evolution log captured seed->best.")


if __name__ == "__main__":
    main()
