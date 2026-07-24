# orena_gepa — GEPA prompt optimization for the FOCUS frame track

Automatically evolve the system prompt in [`orena_sft/prompts.py`](../orena_sft/prompts.py)
instead of hand-tuning it, using **GEPA** (Genetic-Pareto reflective prompt
evolution, https://gepa-ai.github.io/gepa/).

## What GEPA is, in one paragraph

GEPA optimizes a prompt with an LLM in the loop instead of gradients. It keeps a
Pareto front of candidate prompts, and repeats: **select** a promising candidate
→ **run** it on a minibatch, capturing not just a score but *why* each answer was
right or wrong (the "Actionable Side Information") → **reflect** with an LLM that
reads those failures and **rewrites** the prompt → **accept** if it improves, and
update the front. Because the feedback is natural language ("Number format
requires a non-negative integer, got '2 clips'") rather than a scalar reward, it
converges in tens of rollouts, needs no weight access, and produces
human-readable prompts. It is the natural complement to the SFT arm: same task,
same scorer, but it moves the *prompt* rather than the *weights*.

## Why this task is an unusually good fit

The frame-track journal already found that base VLMs here are **not blind, they
are non-compliant**: most wrong `fo_class` answers are the right idea in the
wrong shape ("surgical instrument" instead of a registered class name), which the
deterministic parsers in `focus.data.formats` reject. That failure mode is
exactly what reflective prompt evolution fixes, and the parsers hand GEPA the
precise reason for every rejection for free. So the seed prompt is our best
hand-written one, and GEPA's job is to close the compliance gap automatically.

## How the pieces map onto GEPA

| GEPA concept       | Here |
|--------------------|------|
| component to evolve | the single `"system_prompt"` string |
| seed candidate      | `prompts.build_system_prompt(style=..., include_definitions=...)` |
| task model          | the local VLM (Qwen3.5 / Gemma), run via `vlm_runner.VLMRunner` |
| metric (score)      | the official FOCUS parser — identical to `Evaluator._evaluate_single` |
| feedback (ASI)      | the parser's own `ValueError` message + the correct answer (`metric.py`) |
| reflection LM       | any litellm model id (needs its API key), or a callable |

## Files

- `metric.py` — `score_and_feedback`: one (pred, reference) → score in [0,1] + a
  natural-language feedback string. The reusable, model-free core.
- `build_gepa_dataset.py` — samples small stratified `train.jsonl` / `val.jsonl`
  from the FOCUS frame track, **preserving `format_kwargs`** so scoring is exact.
- `vlm_runner.py` — loads the VLM once; `generate(system_prompt, examples)`.
- `gepa_adapter.py` — `FocusFrameAdapter`: `evaluate` (rollout + score + capture
  feedback) and `make_reflective_dataset` (feedback → reflection records).
- `run_gepa.py` — wires seed + adapter + reflection LM into `gepa.optimize`.
- `evolution_logger.py` — follows the prompt across the run: a tee'd `run_log.txt`,
  a per-step `evolution.jsonl`, and a human-readable `evolution.md` (seed prompt,
  each accepted prompt with a unified diff vs its parent, and a final SEED→BEST
  section). Also drops `seed_prompt.txt` / `best_prompt.txt`.
- `selftest_metric.py` — **offline test**: metric-vs-official parity on a recorded
  `predictions.jsonl` (no GPU/keys).
- `selftest_loop.py` — **offline test**: full GEPA loop with stub task + reflection.

## Tested (offline, no GPU / no API key)

```bash
# 1. scorer parity + a look at the feedback GEPA would reflect on
.venv/bin/python orena_gepa/selftest_metric.py \
    orena_sft/eval_base/Qwen3.5-9B_wo_thinking_direct/lapchole/predictions.jsonl
# -> deterministic-scored=1864  metric-vs-official mismatches=0  PASS

# 2. end-to-end wiring: GEPA reflects on parser feedback and adopts a better prompt
.venv/bin/python orena_gepa/selftest_loop.py
# -> val score 0 -> 1.0  PASS
```

## Running it for real

```bash
# 0. deps: `gepa` is installed. The reflection LM key is read from the repo-root
#    .env automatically -- run_gepa.py maps an sk-or-... OPENAI_API_KEY to
#    OPENROUTER_API_KEY, so the default openrouter/openai/gpt-4.1 just works.

# 1. build a small optimization set (few rollouts is the point)
.venv/bin/python orena_gepa/build_gepa_dataset.py --datasets heico lapchole \
    --n-train 40 --n-val 80

# 2. optimize (task model on GPU, seeded from the hand-written 'direct' prompt)
.venv/bin/python orena_gepa/run_gepa.py \
    --model-id Qwen/Qwen3.5-9B \
    --seed-style direct \
    --reflection-lm openrouter/openai/gpt-4.1 \
    --max-metric-calls 150
# best prompt -> orena_gepa/runs/gepa/best_prompt.txt
```

The reflection LM is addressed via litellm. Defaults to `openrouter/openai/gpt-4.1`;
use any litellm id (`openai/...` + `OPENAI_API_KEY`, `anthropic/...` +
`ANTHROPIC_API_KEY`, etc.). `run_gepa.py` refuses to start if the matching key
is missing.

### Following the evolution

Every run writes, into `--out-dir` (default `orena_gepa/runs/gepa/`):

- `run_log.txt` — GEPA's full per-iteration trace (each proposed prompt included).
- `evolution.md` — readable timeline: **seed prompt → each accepted prompt with a
  diff vs its parent → final SEED vs BEST + diff and a lineage table**.
- `evolution.jsonl` — the same, machine-readable (incl. the reflection LM's raw output).
- `seed_prompt.txt` / `best_prompt.txt`.

Live, the console prints `[evolution] iter N: ACCEPTED candidate K …` / rejections
as they happen, and a final `EVOLUTION: seed val=… -> best val=… (Δ …)` line.

**Note on scores:** the per-accept line shows the *minibatch* acceptance score;
the authoritative held-out val scores are in the final summary / lineage table.

**Reflection model:** only the reflection LM hits an API — default
`openrouter/openai/gpt-4.1` (`--reflection-lm`). The task model answering the
surgical questions is the local VLM; the API is used solely to rewrite prompts.

Then evaluate the winning prompt on the held-out test set the same way the other
arms are evaluated — drop `best_prompt.txt` in as the system prompt in
`orena_sft/base_model_eval.py` (or point a small loader at it) and compare
against the hand-written `direct` / `structured` baselines.

## Notes & scope

- By default only **deterministically-scorable** formats (binary, number,
  fo_class, time, percentage) are sampled, so the whole loop runs with no LLM
  judge and no API key beyond the reflection LM. Add `--include-judge-formats` to
  `build_gepa_dataset.py` and `--with-judge` to `run_gepa.py` to also optimize
  open-ended / matching / multiple-choice (loads a `TransformersJudge`).
- Scores are calibrated to the official metric, so a GEPA gain on `val` is a real
  leaderboard gain, not an artefact of a looser scorer.
- GEPA optimizes the prompt only; it composes with, and is a cheap baseline for,
  the SFT arm ("is fine-tuning needed, or just a better prompt?").
