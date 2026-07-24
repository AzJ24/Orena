"""Evaluate the BASE (un-fine-tuned) Qwen3.5-9B on the FOCUS frame track.

Self-standing: it shares nothing with `evaluate_qwen_frame.py` (the fine-tuned
eval) beyond the `frame_path()` helper, so the baseline can never be perturbed
by a change made for the fine-tuned path. It is the mirror image of
`gemma_eval_base.py`, deliberately built on the canonical documented snippet --
one `apply_chat_template(tokenize=True, ...)` call that also handles the image:

    processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-9B")
    model = AutoModelForMultimodalLM.from_pretrained("Qwen/Qwen3.5-9B")
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=40)
    print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

Qwen specifics, each verified against the real processor rather than assumed
from the Gemma port:

  * **Thinking.** `enable_thinking=False` renders an empty CLOSED block
    (`<think>\\n\\n</think>\\n\\n`), so the model starts directly on the answer;
    `enable_thinking=True` leaves `<think>\\n` OPEN and the model reasons first.
    Same semantics as Gemma, different tokens.
  * **`parse_response()` EXISTS BUT RAISES.** `hasattr(processor,
    "parse_response")` is True, yet calling it raises "This tokenizer does not
    have a `response_schema`" -- unlike Gemma, Qwen ships no schema. Hence the
    hand-rolled `strip_thinking()` below. Attribute presence is not capability.
  * **`padding=True` is REQUIRED for batching** (passed via `processor_kwargs`,
    the form the library asks for). Without it, a batch whose prompts tokenize
    to different lengths raises "Unable to convert output 'input_ids'". Two
    prompts of *identical* length slip through, so a small smoke test can pass
    and the full run still crash.
  * Left-padding (`tokenizer.padding_side = "left"`) makes `input_len` a single
    shared column, so new tokens slice off cleanly for the whole batch.

Usage:
    .venv/bin/python orena_sft/base_model_eval.py --datasets lapchole --limit 50
    .venv/bin/python orena_sft/base_model_eval.py --datasets lapchole \\
        --prompt-style structured --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor

from focus import DatasetSplit, FocusConfig, FocusDataset, Track, set_config
from focus.config import DATASET_BASE_FPS
from focus.data.data_models import Response
from focus.evaluation import Evaluator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_frame_sft_dataset import DEFAULT_ROOT_DIR, frame_path  # noqa: E402
from prompts import build_system_prompt, extract_answer  # noqa: E402

SFT_DIR = Path(__file__).resolve().parent

THINK_CLOSE = "</think>"


def strip_thinking(raw: str) -> tuple[str, str]:
    """Split a raw generation into (thinking, content).

    Qwen's `parse_response()` raises for lack of a response schema, so this does
    the one thing that method would: everything after the final `</think>` is
    the answer. With `enable_thinking=False` the prompt already contains the
    closed block, so the generation has no tags at all and this is a no-op.
    """
    if THINK_CLOSE in raw:
        thinking, _, content = raw.rpartition(THINK_CLOSE)
        return thinking.replace("<think>", "").strip(), content.strip()
    return "", raw.strip()


@torch.no_grad()
def condition_question(question: str, procedure: str | None, retriever=None) -> str:
    """Prepend the stated procedure to the question, or return it untouched.

    `procedure_type` is on every Request (and, per the track spec, on every
    official test case). Injecting it here -- in the user turn, not the system
    prompt -- means the model is TOLD which procedure it is looking at, so it
    can select the right anatomical frame instead of collapsing onto a single
    learned prior. Kept out of the system prompt so it composes with any
    --prompt-style (plain included) and leaves prompts.py untouched.
    """
    if not procedure:
        return question
    lines = [f"Procedure type: {procedure}."]
    if retriever is not None:
        a = retriever.facts(procedure)
        if a:
            lines.append(f"Relevant anatomy: {a}")
    return "\n".join(lines) + "\n" + question


def generate_answers(processor, model, image_paths: list[str], questions: list[str],
                     max_new_tokens: int, enable_thinking: bool = False,
                     system_prompt: str | None = None,
                     procedures: list[str] | None = None, retriever=None):
    """N (image, question) pairs -> N (answer, thinking, raw) + amortized seconds.

    The documented single-sample flow, handed a LIST of conversations: the
    processor batches every tensor and left-pads, so `--batch-size 1` is exactly
    the canonical snippet and larger batches are the same call amortized.

    `system_prompt` (from `prompts.py`) is orthogonal to `enable_thinking`: its
    chain of thought is a plain REASONING line in the visible output, not the
    native `<think>` channel, so the two combine freely. When set, the returned
    answer is the extracted ANSWER value rather than the whole content.

    `procedures` (when given, one per example) prepends the procedure type to
    each question via `condition_question`; None leaves questions unchanged.

    Time is amortized (batch wall-clock / batch size) -- a batched generate()
    cannot attribute latency to individual sequences.
    """
    if procedures is None:
        procedures = [None] * len(questions)
    system_turn = (
        [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]
        if system_prompt else []
    )
    convs = [
        system_turn + [{"role": "user", "content": [
            {"type": "image", "image": Image.open(p).convert("RGB")},
            {"type": "text", "text": condition_question(q, proc, retriever)},
        ]}]
        for p, q, proc in zip(image_paths, questions, procedures)
    ]

    inputs = processor.apply_chat_template(
        convs,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        processor_kwargs={"padding": True},
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    start = time.monotonic()
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    elapsed = time.monotonic() - start

    results = []
    for row in outputs:
        raw = processor.decode(row[input_len:], skip_special_tokens=True)
        thinking, content = strip_thinking(raw)
        results.append((
            extract_answer(content) if system_prompt else content,
            thinking,
            raw,
        ))
    return results, elapsed / len(convs)


def evaluate_dataset(processor, model, dataset: str, cfg: FocusConfig, args) -> None:
    base_fps = float(DATASET_BASE_FPS[dataset])
    ds = FocusDataset(dataset, DatasetSplit.TEST, Track.FRAME)
    n = len(ds) if args.limit is None else min(args.limit, len(ds))

    # Gather valid (req, ref, path) upfront so frames missing on disk are
    # dropped before they can unbalance a batch.
    items = []
    for i in range(n):
        req, ref = ds[i]
        p = frame_path(cfg, dataset, base_fps, req.videoID, req.start_time)
        if p.exists():
            items.append((req, ref, p))

    out_dir = Path(args.output_dir) / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "predictions.jsonl"
    n_skipped = n - len(items)
    print(f"\n[{dataset}] evaluating {len(items)}/{len(ds)} test examples "
          f"(batch size {args.batch_size}"
          + (f", {n_skipped} skipped: no frame on disk" if n_skipped else "") + ")...", flush=True)

    requests, references, responses = [], [], []
    with predictions_path.open("w") as pred_f:
        for start in range(0, len(items), args.batch_size):
            batch = items[start:start + args.batch_size]

            results, elapsed = generate_answers(
                processor, model,
                [str(p) for _, _, p in batch],
                [req.question for req, _, _ in batch],
                args.max_new_tokens,
                enable_thinking=args.enable_thinking,
                system_prompt=args.system_prompt,
                procedures=([req.procedure_type for req, _, _ in batch]
                            if (args.condition_procedure or args.inject_knowledge == "rag") else None),
                retriever=args.retriever,
            )

            for (req, ref, p), (answer, thinking, raw) in zip(batch, results):
                requests.append(req)
                references.append(ref)
                responses.append(Response(qID=req.qID, content=answer, latency=elapsed))

                pred_f.write(json.dumps({
                    "qID": req.qID,
                    "videoID": req.videoID,
                    "image_path": str(p),
                    "question": req.question,
                    "primary_capability": ref.primary.name,
                    "format": ref._format,
                    "gt_answer": ref.answer,
                    "pred_answer": answer,      # what gets scored
                    "thinking": thinking,       # empty unless --enable-thinking
                    "raw_response": raw,        # unparsed, for debugging
                    "generate_time": elapsed,   # amortized per sample within the batch
                }) + "\n")
            pred_f.flush()

            done = min(start + args.batch_size, len(items))
            if done % max(args.batch_size * 5, 25) < args.batch_size or done == len(items):
                print(f"  [{dataset}] {done}/{len(items)} done", flush=True)

    print(f"[{dataset}] wrote raw generations to {predictions_path}")

    evaluator = Evaluator(num_workers=args.judge_workers, judge_kwargs={"device": args.judge_device})
    _, summary_df = evaluator.run(requests, references, responses, output_dir=out_dir)
    print(f"\n[{dataset}] summary:")
    print(summary_df.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--datasets", nargs="+", default=["lapchole"], choices=["heico", "lapchole"])
    ap.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT_DIR)
    ap.add_argument("--output-dir", default=None,
                     help="defaults to <script-dir>/eval_base/<model-id-basename><mode suffix>")
    ap.add_argument("--enable-thinking", action="store_true",
                     help="leave Qwen's <think> block OPEN so the model reasons before answering. "
                          "Off by default (empty closed block = answer directly). Much slower and "
                          "needs a big --max-new-tokens; the reasoning is stripped before scoring.")
    ap.add_argument("--prompt-style", choices=["plain", "structured", "direct"], default="plain",
                     help="'plain' (default) sends the bare question. 'structured' prepends the "
                          "system prompt from prompts.py (FO definition + class vocabulary, a "
                          "REASONING line, and an ANSWER line parsed back out for exact matching). "
                          "'direct' is that same prompt minus the reasoning line -- format control "
                          "only. structured-vs-direct isolates the value of thinking out loud; "
                          "'direct --enable-thinking' reasons natively and formats via the prompt.")
    ap.add_argument("--fo-definitions", action="store_true",
                     help="with --prompt-style structured, include the full per-class descriptions "
                          "(~700 extra prompt tokens per question) instead of the class names alone.")
    ap.add_argument("--system-prompt-file", type=Path, default=None,
                     help="use a system prompt read verbatim from this file (e.g. a GEPA-evolved "
                          "best_prompt.txt) instead of build_system_prompt(). Overrides --prompt-style; "
                          "output is still parsed with extract_answer for exact matching.")
    ap.add_argument("--max-new-tokens", type=int, default=None,
                     help="default: 128 without thinking, 1024 with --enable-thinking "
                          "(the <think> block must fit AND close before the answer is emitted)")
    ap.add_argument("--batch-size", type=int, default=32,
                     help="(image, question) pairs generated together. 1 = exactly the canonical "
                          "single-sample snippet; higher is the same call amortized.")
    ap.add_argument("--limit", type=int, default=None,
                     help="cap test examples per dataset. NOTE: takes the FIRST n rows in dataset "
                          "order, i.e. one or two videos -- fine for checking output shape, not a "
                          "representative accuracy sample.")
    ap.add_argument("--condition-procedure", action="store_true",
                     help="prepend 'Procedure type: <name>.' to each question (from "
                          "req.procedure_type, present on every test case). Tells the model which "
                          "procedure it is looking at so it can pick the right anatomical frame "
                          "instead of collapsing onto one prior. Composes with any --prompt-style.")
    ap.add_argument("--inject-knowledge", choices=["none", "rag"], default="none",
                     help="'rag' retrieves a procedure-specific anatomy block from --rag-index "
                          "and injects it (keyed on procedure_type). Implies procedure conditioning.")
    ap.add_argument("--rag-index", type=Path,
                     default=Path(__file__).resolve().parent.parent / "orena_rag" / "rag_index",
                     help="dir with passages.jsonl + embeddings.npy for --inject-knowledge rag.")
    ap.add_argument("--judge-device", default="cuda")
    ap.add_argument("--judge-workers", type=int, default=1)
    args = ap.parse_args()

    if args.max_new_tokens is None:
        args.max_new_tokens = 2048 if args.enable_thinking else 128

    prompted = args.prompt_style != "plain" or args.system_prompt_file is not None
    if args.fo_definitions and args.prompt_style == "plain":
        ap.error("--fo-definitions has no effect with --prompt-style plain.")
    if args.system_prompt_file is not None:
        # An evolved prompt already governs answer shape; extract_answer still
        # runs on the output (it is a no-op on a bare answer, and recovers the
        # value if the prompt kept an ANSWER: line).
        args.system_prompt = args.system_prompt_file.read_text()
    elif args.prompt_style != "plain":
        args.system_prompt = build_system_prompt(args.fo_definitions, style=args.prompt_style)
    else:
        args.system_prompt = None

    args.retriever = None
    if args.inject_knowledge == "rag":
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from orena_rag.retriever import AnatomyRetriever
        args.retriever = AnatomyRetriever(args.rag_index)
        print(f"RAG retriever loaded from {args.rag_index}")

    # One folder per arm so runs never overwrite each other.
    if args.output_dir is None:
        suffix = "_thinking" if args.enable_thinking else "_wo_thinking"
        if args.system_prompt_file is not None:
            # name the arm after the prompt file's parent dir (the GEPA run name)
            suffix += "_gepa_" + args.system_prompt_file.resolve().parent.name
        elif prompted:
            suffix += f"_{args.prompt_style}" + ("_defs" if args.fo_definitions else "")
        if args.condition_procedure:
            suffix += "_conditioned"
        if args.inject_knowledge == "rag":
            suffix += "_rag"
        args.output_dir = str(SFT_DIR / "eval_base" / (args.model_id.split("/")[-1] + suffix))

    prompt_desc = (f"file:{args.system_prompt_file}" if args.system_prompt_file is not None
                   else args.prompt_style + (' +definitions' if args.fo_definitions else ''))
    print(f"Thinking mode: {'ON' if args.enable_thinking else 'OFF'} | "
          f"prompt: {prompt_desc}"
          f"{' | procedure-conditioned' if args.condition_procedure else ''} | "
          f"max_new_tokens={args.max_new_tokens} | output -> {args.output_dir}")
    if args.system_prompt:
        bar = "=" * 78
        print(f"{bar}\nSYSTEM PROMPT (verbatim, in full)\n{bar}\n"
              f"{args.system_prompt}{bar}", flush=True)

    cfg = FocusConfig(root_dir=args.root_dir)
    set_config(cfg)

    print(f"Loading base model {args.model_id!r} (no fine-tuning)...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    # Left-pad so every sequence's first generated token lands at the same
    # column, making input_len a single shared prompt boundary.
    processor.tokenizer.padding_side = "left"
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_id, dtype="auto", device_map="auto",
    )
    model.eval()

    for dataset in args.datasets:
        evaluate_dataset(processor, model, dataset, cfg, args)

    print(f"\nDone. Per-dataset results.csv / summary.csv written under {args.output_dir}/")


if __name__ == "__main__":
    main()
