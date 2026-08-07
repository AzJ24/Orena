"""Score an answer.json against the batch's ground truth, and against the budget.

Uses `focus.data.formats` for the closed-form answers, which is the same
deterministic comparison the challenge applies (including `Time`'s 5 s acceptance
threshold — the tolerance the absolute-timestamp reconstruction is judged by).
Open-ended and multiple-choice go to an LLM judge on the platform; here they are
printed for inspection and excluded from the score rather than guessed at.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from focus.data.formats import get_format_class

# Formats focus compares deterministically; the rest need the judge.
DETERMINISTIC = {"binary", "number", "percentage", "fo_class", "time"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=Path, required=True)
    ap.add_argument("--elapsed", type=float, required=True)
    ap.add_argument("--rc", type=int, required=True)
    ap.add_argument("--per-question", type=float, default=15.0)
    ap.add_argument("--output-subdir", default="output")
    args = ap.parse_args()

    interface = args.batch / "interface_1"
    requests = json.loads((interface / "request.json").read_text())
    refs = {r["qID"]: r for r in json.loads((args.batch / "references.json").read_text())}
    answers_path = args.batch / args.output_subdir / "answer.json"

    B = len(requests)
    budget = 120 + B * args.per_question
    print(f"exit code            : {args.rc}")
    print(f"questions in batch   : {B}")
    print(f"WALL CLOCK (total)   : {args.elapsed:.1f}s")
    print(f"BUDGET               : 120 + {B}x{args.per_question:.0f} = {budget:.0f}s")
    print(f"margin               : {budget - args.elapsed:.1f}s "
          f"({args.elapsed / budget * 100:.1f}% used)")
    print(f"forfeit threshold    : {budget * 1.2:.0f}s (>20% over = whole batch lost)")

    if args.rc != 0:
        print("\nFATAL: run exited non-zero")
        return 1
    if not answers_path.exists():
        print(f"\nFATAL: no answer.json at {answers_path}")
        return 1

    answers = {a["qID"]: a for a in json.loads(answers_path.read_text())}
    print(f"\nresponses            : {len(answers)} / {B}")
    if set(answers) != {r["qID"] for r in requests}:
        print("FATAL: qID mismatch between request.json and answer.json")
        return 1

    empty = [q for q, a in answers.items() if not str(a.get("content", "")).strip()]
    print(f"empty answers        : {len(empty)} {empty if empty else ''}")

    latencies = [a.get("latency", 0.0) for a in answers.values()]
    over = [q for q, a in answers.items() if a.get("latency", 0) > args.per_question]
    print(f"per-question latency : mean {sum(latencies) / len(latencies):.2f}s  "
          f"max {max(latencies):.2f}s  over {args.per_question:.0f}s: {len(over)}")

    print("\n--- answers ---")
    n_scored = n_correct = 0
    by_format: dict[str, list[int]] = {}
    for req in requests:
        qid = req["qID"]
        ref, ans = refs[qid], answers[qid]
        pred, gold, fmt = str(ans.get("content", "")), ref["answer"], ref["format"]

        verdict = "  (judge)"
        if fmt in DETERMINISTIC:
            try:
                fobj = get_format_class(fmt)(**ref.get("format_kwargs", {}))
                ok = bool(fobj.compare(fobj.read(pred), fobj.read(gold)))
            except Exception:
                ok = False  # unparseable is scored wrong, as on the platform
            n_scored += 1
            n_correct += ok
            by_format.setdefault(fmt, []).append(int(ok))
            verdict = "  RIGHT " if ok else "  wrong"

        print(f"  {qid} [{fmt:15s}]{verdict}  pred={pred!r:34.34}  gold={gold!r:26.26}"
              f"  {ans.get('latency', 0):5.2f}s")

    if n_scored:
        print(f"\ndeterministic accuracy: {n_correct}/{n_scored} = {n_correct / n_scored:.1%}")
        for fmt in sorted(by_format):
            hits = by_format[fmt]
            print(f"  {fmt:15s} {sum(hits)}/{len(hits)}")
        print("(open_ended / multiple_choice are LLM-judged on the platform and "
              "excluded here)")

    print()
    ok_time = args.elapsed <= budget
    print(f">>> {'PASS' if ok_time else 'FAIL'}: {args.elapsed:.1f}s "
          f"{'within' if ok_time else 'EXCEEDS'} {budget:.0f}s budget <<<")
    return 0 if ok_time else 1


if __name__ == "__main__":
    raise SystemExit(main())
