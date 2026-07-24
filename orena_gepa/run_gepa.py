"""Run GEPA to evolve the FOCUS frame-track system prompt.

Seeds with the hand-written prompt from `orena_sft/prompts.py` and lets GEPA
reflect on parser feedback to rewrite it. The task model is the local VLM; the
reflection model is any litellm-addressable LLM (needs the matching API key) or,
for offline smoke tests, a Python callable passed in code.

Usage (real run, needs a reflection LM + GPU):
    export OPENAI_API_KEY=...
    .venv/bin/python orena_gepa/run_gepa.py \\
        --model-id Qwen/Qwen3.5-9B \\
        --reflection-lm openai/gpt-4.1-mini \\
        --max-metric-calls 150

Build the data first with build_gepa_dataset.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import gepa

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "orena_sft"))
from evolution_logger import EvolutionTracker, TeeLogger, write_final_summary  # noqa: E402
from gepa_adapter import COMPONENT, FocusFrameAdapter  # noqa: E402
from prompts import build_system_prompt  # noqa: E402

GEPA_DIR = Path(__file__).resolve().parent
REPO_ROOT = GEPA_DIR.parent


def load_env(path: Path = REPO_ROOT / ".env") -> None:
    """Load KEY=VALUE lines from the repo-root .env into os.environ (without
    overriding anything already set), so the reflection LM's API key is picked
    up automatically. Avoids a hard dependency on python-dotenv."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))

    # An `sk-or-...` value is an OpenRouter key even if stored as OPENAI_API_KEY;
    # expose it under the name litellm's openrouter/* models actually read.
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if openai_key.startswith("sk-or-"):
        os.environ.setdefault("OPENROUTER_API_KEY", openai_key)


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=GEPA_DIR / "gepa_data")
    ap.add_argument("--out-dir", type=Path, default=None,
                     help="defaults to runs/gepa-<seed-style>[-defs] so the two seed "
                          "arms never share (and overwrite) each other's evolution log")
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--seed-style", choices=["direct", "structured"], default="direct",
                     help="which hand-written prompt to seed GEPA with")
    ap.add_argument("--fo-definitions", action="store_true")
    ap.add_argument("--reflection-lm", default="openrouter/openai/gpt-4.1",
                     help="litellm model id for the reflection LM (needs its API key). "
                          "Default routes gpt-4.1 through OpenRouter (reads OPENROUTER_API_KEY, "
                          "auto-derived from an sk-or- OPENAI_API_KEY in .env).")
    ap.add_argument("--max-metric-calls", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--with-judge", action="store_true",
                     help="load a TransformersJudge for open_ended/matching/multiple_choice examples")
    args = ap.parse_args()

    if args.out_dir is None:
        arm = args.seed_style + ("-defs" if args.fo_definitions else "")
        # Include the reflection model so runs that differ only in the evolution
        # LM (e.g. gpt-5.2 vs opus) land in separate dirs instead of overwriting.
        refl_slug = args.reflection_lm.split("/")[-1]
        args.out_dir = GEPA_DIR / "runs" / f"gepa-{arm}-{refl_slug}"

    load_env()
    _needed_key = {"openrouter/": "OPENROUTER_API_KEY", "openai/": "OPENAI_API_KEY",
                   "anthropic/": "ANTHROPIC_API_KEY"}
    for prefix, var in _needed_key.items():
        if args.reflection_lm.startswith(prefix) and not os.environ.get(var):
            raise SystemExit(f"{var} not found (checked env and repo-root .env) for "
                             f"--reflection-lm {args.reflection_lm}. Set it and retry.")

    trainset = load_jsonl(args.data_dir / "train.jsonl")
    valset = load_jsonl(args.data_dir / "val.jsonl")
    print(f"train={len(trainset)}  val={len(valset)}")

    from vlm_runner import VLMRunner
    runner = VLMRunner(args.model_id, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)

    judge = None
    if args.with_judge:
        from focus.evaluation.judges import TransformersJudge
        judge = TransformersJudge()

    adapter = FocusFrameAdapter(runner, judge=judge)
    seed = {COMPONENT: build_system_prompt(args.fo_definitions, style=args.seed_style)}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tracker = EvolutionTracker(args.out_dir, component=COMPONENT)

    result = gepa.optimize(
        seed_candidate=seed,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=args.reflection_lm,
        max_metric_calls=args.max_metric_calls,
        run_dir=str(args.out_dir),
        logger=TeeLogger(args.out_dir / "run_log.txt"),
        callbacks=[tracker],
        display_progress_bar=True,
    )

    tracker.close()
    write_final_summary(result, args.out_dir, component=COMPONENT)


if __name__ == "__main__":
    main()
