"""Offline test of the GEPA scoring/feedback core -- no GPU, no model, no API key.

Runs `metric.score_and_feedback` over an already-recorded `predictions.jsonl`
from `orena_sft/eval_base/`, and:

  1. checks score parity against the official `focus` evaluator on the
     deterministic (non-judge) formats -- if these disagree, a GEPA "win" would
     be measuring the wrong thing;
  2. prints a few sample feedback strings, which are exactly the Actionable Side
     Information the reflection LM would use to rewrite the prompt.

Usage:
    .venv/bin/python orena_gepa/selftest_metric.py \\
        orena_sft/eval_base/Qwen3.5-9B_wo_thinking_direct/lapchole/predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from focus.data.formats import JUDGE_FORMATS, get_format_class

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metric import score_and_feedback  # noqa: E402


def official_correct(pred: str, fmt_type: str, ref_answer: str) -> bool | None:
    """Replicate Evaluator._evaluate_single for deterministic formats; None for judge ones."""
    if fmt_type in JUDGE_FORMATS:
        return None
    fmt = get_format_class(fmt_type)()
    try:
        gold = fmt.read(ref_answer)
        pred_parsed = fmt.read(pred)
    except ValueError:
        return False
    return bool(fmt.compare(gold, pred_parsed))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("predictions", type=Path)
    ap.add_argument("--show", type=int, default=6, help="how many sample feedbacks to print")
    args = ap.parse_args()

    with args.predictions.open() as f:
        rows = [json.loads(line) for line in f]

    n_det = mismatches = 0
    correct = 0
    samples: list[str] = []
    for r in rows:
        fmt = r["format"]
        res = score_and_feedback(
            pred=r["pred_answer"], ref_format=fmt, ref_answer=str(r["gt_answer"]),
            question=r["question"],
        )
        off = official_correct(r["pred_answer"], fmt, str(r["gt_answer"]))
        if off is not None:  # deterministic format -> check parity
            n_det += 1
            if bool(res.score) != off:
                mismatches += 1
            if off:
                correct += 1
        if len(samples) < args.show and res.score == 0.0 and res.scorable:
            samples.append(f"[{fmt}] Q: {r['question'][:70]}\n"
                           f"   gt={r['gt_answer']!r}  pred={r['pred_answer']!r}\n"
                           f"   FEEDBACK: {res.feedback}")

    print(f"rows={len(rows)}  deterministic-scored={n_det}  "
          f"metric-vs-official mismatches={mismatches}")
    print(f"deterministic accuracy (this metric) = {correct}/{n_det} = "
          f"{correct / n_det:.3f}" if n_det else "no deterministic rows")
    print("\n--- sample reflective feedback (what GEPA would see on failures) ---")
    for s in samples:
        print(s)

    if mismatches:
        raise SystemExit(f"FAIL: {mismatches} scoring disagreements with the official evaluator")
    print("\nPASS: metric matches the official evaluator on all deterministic formats.")


if __name__ == "__main__":
    main()
