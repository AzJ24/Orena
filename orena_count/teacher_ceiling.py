"""Stage 2: can a strong teacher enumerate these frames correctly?

Stage 1 showed the base 9B counts its own list correctly (95% self-consistent)
but lists far too few instances -- it sees one clip where there are five. So the
failure is perception, not arithmetic. That leaves one question before building
a trace-distillation pipeline: can ANY model see them?

If the teacher also undercounts, there is no signal to distill and the
enumerate-then-count line is dead. If it enumerates accurately, rejection-sampled
traces would be teaching perception and Stage 3 is worth building.

Samples instance-count questions stratified by ground-truth count, so the answer
is not just "how accurate" but "where does it break" -- a teacher that handles
1-2 objects and collapses at 5+ tells you the ceiling is clutter, not the task.

    .venv/bin/python orena_count/teacher_ceiling.py --n-per-count 40
"""

from __future__ import annotations

import argparse
import base64
import collections
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COUNT_DIR = Path(__file__).resolve().parent
TEMPLATE = "How many different foreign object instances appear in this frame?"


def load_env(path: Path = REPO_ROOT / ".env") -> None:
    """Same convention as orena_gepa/run_gepa.py: KEY=VALUE lines into os.environ."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def parse(text: str) -> tuple[int | None, int]:
    """Return (answer, n_listed) from a teacher reply."""
    n_listed = len(re.findall(r"^\s*\d+\.\s", text, re.M))
    m = re.findall(r"ANSWER:\s*(\d+)", text)
    if not m:
        m = re.findall(r"\b(\d+)\b", text)
    return (int(m[-1]) if m else None), n_listed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="anthropic/claude-opus-5")
    ap.add_argument("--prompt", default=str(COUNT_DIR / "prompts" / "enumerate_v1.txt"))
    ap.add_argument("--split", default=str(REPO_ROOT / "orena_sft/sft_export/combined/train.jsonl"))
    ap.add_argument("--n-per-count", type=int, default=40,
                    help="frames sampled per ground-truth count value (1..max-count)")
    ap.add_argument("--max-count", type=int, default=6)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(COUNT_DIR / "results" / "teacher_ceiling.jsonl"))
    args = ap.parse_args()

    load_env()
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        print("no OPENROUTER_API_KEY / OPENAI_API_KEY (checked shell and .env)", file=sys.stderr)
        return 1

    from openai import OpenAI
    client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")
    system_prompt = Path(args.prompt).read_text()

    rows = [json.loads(l) for l in open(args.split)]
    pool = collections.defaultdict(list)
    for r in rows:
        if r["format"] != "number" or TEMPLATE not in r["messages"][0]["content"][1]["text"]:
            continue
        try:
            gt = int(r["messages"][1]["content"][0]["text"])
        except ValueError:
            continue
        if 1 <= gt <= args.max_count:
            pool[gt].append(r)

    rng = random.Random(args.seed)
    sample = []
    for gt in sorted(pool):
        rng.shuffle(pool[gt])
        sample += [(gt, r) for r in pool[gt][: args.n_per_count]]
    print(f"sampled {len(sample)} frames: "
          + ", ".join(f"gt={k}:{sum(1 for g, _ in sample if g == k)}" for k in sorted(pool)),
          flush=True)

    def ask(item):
        gt, r = item
        path = r["messages"][0]["content"][0]["image"]
        question = r["messages"][0]["content"][1]["text"]
        b64 = base64.b64encode(Path(path).read_bytes()).decode()
        for attempt in range(4):
            try:
                resp = client.chat.completions.create(
                    model=args.model, max_tokens=400, temperature=0,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": [
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                            {"type": "text", "text": question},
                        ]},
                    ],
                )
                text = resp.choices[0].message.content or ""
                pred, n_listed = parse(text)
                return {"qID": r["qID"], "gt": gt, "pred": pred, "n_listed": n_listed,
                        "image": path, "raw": text}
            except Exception as exc:  # transient rate limits / 5xx
                if attempt == 3:
                    return {"qID": r["qID"], "gt": gt, "pred": None, "n_listed": 0,
                            "image": path, "raw": f"ERROR: {exc}"}
        return None

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        out = list(ex.map(ask, sample))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")

    ok = [o for o in out if o["pred"] is not None]
    errs = len(out) - len(ok)
    correct = sum(o["pred"] == o["gt"] for o in ok)
    consistent = sum(o["pred"] == o["n_listed"] for o in ok if o["n_listed"])
    listed = sum(1 for o in ok if o["n_listed"])
    print(f"\nteacher = {args.model}   n={len(out)}   api errors={errs}")
    print(f"  exact-match accuracy : {correct}/{len(ok)} = {correct/max(len(ok),1):.3f}")
    print(f"  produced a list      : {listed}/{len(ok)}")
    print(f"  list length == answer: {consistent}/{max(listed,1)}")
    print(f"\n{'gt':>4}{'n':>5}{'acc':>8}{'mean pred':>11}{'mean listed':>13}")
    for gt in sorted({o["gt"] for o in ok}):
        sub = [o for o in ok if o["gt"] == gt]
        acc = sum(o["pred"] == gt for o in sub) / len(sub)
        mp = sum(o["pred"] for o in sub) / len(sub)
        ml = sum(o["n_listed"] for o in sub) / len(sub)
        print(f"{gt:>4}{len(sub):>5}{acc:>8.3f}{mp:>11.2f}{ml:>13.2f}")
    within1 = sum(abs(o["pred"] - o["gt"]) <= 1 for o in ok) / max(len(ok), 1)
    under = sum(o["pred"] < o["gt"] for o in ok) / max(len(ok), 1)
    print(f"\n  within +/-1: {within1:.3f}   undercounts: {under:.3f}")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
