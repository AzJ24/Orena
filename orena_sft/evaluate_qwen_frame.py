"""Evaluates a fine-tuned Qwen3.5 checkpoint on the FOCUS frame-track TEST
split, using the FOCUS library's own :class:`focus.evaluation.Evaluator` --
deterministic parsing/comparison for closed-form formats (binary, number,
percentage, fo_class, time), LLM-as-judge for open-ended ones (open_ended,
matching, multiple_choice).

Unlike training, this loads Request/Reference objects directly from
`focus.FocusDataset` (not from `test.jsonl`) since the Evaluator needs the
library's typed dataclasses, not raw dicts. Frame paths are resolved with the
same `frame_path()` helper `build_frame_sft_dataset.py` uses, so this stays
consistent with how the model was trained.

Generation is batched (`--batch-size`) with left-padding -- the canonical
pattern for batched decoder generation, so every sequence's first generated
token lands at the same right-edge position. Timing is amortized per sample
(batch wall-clock / batch size), since a batched `generate()` call can't
separate one sequence's latency from another's.

Writes, per dataset, under `<output-dir>/<dataset>/`:
  - `predictions.jsonl` -- every raw (question, gt_answer, pred_answer) triple,
    for manual inspection, written incrementally as each batch finishes.
  - `results.csv` / `summary.csv` -- from Evaluator.run(): per-question
    correctness and the hierarchical accuracy summary.

Requires `trl` and `peft` (see sft_train_qwen_frame.py).

Usage:
    .venv/bin/python orena_sft/evaluate_qwen_frame.py \\
        --checkpoint-dir ./orena_sft/checkpoints/Qwen3.5-9B-lora-2026-07-16_16-41-57 \\
        --datasets heico lapchole --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from focus import DatasetSplit, FocusConfig, FocusDataset, Track, set_config
from focus.data.data_models import Response
from focus.evaluation import Evaluator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_frame_sft_dataset import DEFAULT_ROOT_DIR, frame_path  # noqa: E402
from prompts import build_system_prompt, extract_answer  # noqa: E402

from focus.config import DATASET_BASE_FPS  # noqa: E402


def build_question(question: str, procedure_type: str, condition: bool,
                   knowledge_mode: str, kb: dict | None, retriever=None) -> str:
    """Prepend procedure conditioning and/or injected knowledge to a question.

    Conditioning surfaces procedure_type (given at inference); knowledge injection
    adds procedure-specific anatomy (targets situs) and/or foreign-object priors
    (targets fo_class), keyed on procedure_type. 'rag' pulls the anatomy block from
    the retriever (built RAG index) instead of the static kb. All are additive
    prompt context; the question is unchanged when nothing is enabled."""
    if not condition and knowledge_mode == "none":
        return question
    lines = [f"Procedure type: {procedure_type}."]
    if knowledge_mode == "rag" and retriever is not None:
        a = retriever.facts(procedure_type)
        if a:
            lines.append(f"Relevant anatomy: {a}")
    else:
        entry = (kb or {}).get(procedure_type, {})
        if knowledge_mode in ("anatomy", "both") and entry.get("anatomy"):
            lines.append(f"Relevant anatomy: {entry['anatomy']}")
        if knowledge_mode in ("fo_priors", "both") and entry.get("fo_priors"):
            lines.append(f"Foreign objects that may be present: {entry['fo_priors']}")
    return "\n".join(lines) + "\n" + question

SFT_DIR = Path(__file__).resolve().parent


def load_model(checkpoint_dir: str, base_model_id: str, merge_lora: bool):
    # Load the processor from base_model_id, not checkpoint_dir: the image
    # processor/chat template never change during LoRA fine-tuning, and
    # intermediate Trainer checkpoints (checkpoint-N/, as opposed to the
    # final save_pretrained() at the end of training) only ever contain the
    # bare tokenizer, not the full multimodal processor -- loading from
    # checkpoint_dir there silently degrades AutoProcessor to a tokenizer.
    processor = AutoProcessor.from_pretrained(base_model_id)
    # Left-pad for generation: batched decoding requires every sequence's
    # "next token to predict" position to align at the right edge, so shorter
    # prompts are padded on the left rather than the right (the opposite of
    # the training collate_fn, which right-pads because there loss masking,
    # not decoding alignment, is what matters).
    processor.tokenizer.padding_side = "left"
    is_lora = (Path(checkpoint_dir) / "adapter_config.json").exists()

    if is_lora:
        from peft import PeftModel

        print(f"Loading base model {base_model_id!r}, then LoRA adapter from {checkpoint_dir!r}...")
        base_model = Qwen3_5ForConditionalGeneration.from_pretrained(
            base_model_id, dtype=torch.bfloat16, device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, checkpoint_dir)
        if merge_lora:
            model = model.merge_and_unload()
    else:
        print(f"Loading full fine-tuned model from {checkpoint_dir!r}...")
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            checkpoint_dir, dtype=torch.bfloat16, device_map="auto",
        )

    model.eval()
    return processor, model


@torch.no_grad()
def generate_batch(
    processor, model, image_paths: list[str], questions: list[str], max_new_tokens: int,
    system_prompt: str | None = None,
) -> tuple[list[str], list[str], float, float]:
    """Generate answers for a batch of (image, question) pairs.

    Returns (answers, raw_texts, generate_time, total_time). With a
    `system_prompt` the model's output is post-processed by `extract_answer`
    (pulling the ANSWER line out of a structured reply, stripping the trailing
    period that would fail Binary/Number parsing, clamping to the 300-char
    OpenEnded limit); `raw_texts` always keeps the untouched generation. With
    no system prompt this is the original plain path and answers == raw_texts.

    MUST match the --prompt-style used at training time: a model fine-tuned
    with a system prompt and evaluated without it (or vice versa) is measuring
    a train/inference mismatch, not the model.

    Both times are *amortized per sample* -- the batch wall-clock time divided
    by the batch size. A batched generate() call runs all sequences together,
    so there is no separable per-sample latency; the amortized figure is the
    meaningful per-example throughput number. generate_time covers only
    model.generate(); total_time also covers image loading + tokenization.

    Relies on left-padding (set on the tokenizer in load_model): with all
    prompts padded to the same length on the left, the prompt boundary is a
    single column (inputs["input_ids"].shape[1]) shared by the whole batch,
    so new tokens slice off cleanly for every row at once.
    """
    total_start = time.monotonic()

    images = [Image.open(p).convert("RGB") for p in image_paths]
    system_turn = (
        [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]
        if system_prompt else []
    )
    texts = [
        processor.apply_chat_template(
            system_turn + [{"role": "user", "content": [
                {"type": "image", "image": p},
                {"type": "text", "text": q},
            ]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        for p, q in zip(image_paths, questions)
    ]
    inputs = processor(
        text=texts, images=images, return_tensors="pt", padding=True,
    ).to(model.device)

    generate_start = time.monotonic()
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generate_time = time.monotonic() - generate_start

    new_tokens = generated[:, inputs["input_ids"].shape[1]:]
    raw_texts = [a.strip() for a in processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)]
    answers = [extract_answer(t) for t in raw_texts] if system_prompt else list(raw_texts)

    total_time = time.monotonic() - total_start
    bs = len(image_paths)
    return answers, raw_texts, generate_time / bs, total_time / bs


def evaluate_dataset(processor, model, dataset: str, cfg: FocusConfig, args) -> None:
    base_fps = float(DATASET_BASE_FPS[dataset])
    ds = FocusDataset(dataset, DatasetSplit.TEST, Track.FRAME)

    n = len(ds) if args.limit is None else min(args.limit, len(ds))

    # Gather valid (req, ref, path) triples upfront -- cheap metadata only, no
    # images opened yet -- so batching is a simple slice over this list and
    # frames missing on disk are dropped before they can unbalance a batch.
    items = []
    for i in range(n):
        req, ref = ds[i]
        p = frame_path(cfg, dataset, base_fps, req.videoID, req.start_time)
        if p.exists():
            items.append((req, ref, p))

    n_skipped = n - len(items)
    print(f"\n[{dataset}] evaluating {len(items)}/{len(ds)} test examples "
          f"(batch size {args.batch_size}"
          + (f", {n_skipped} skipped: no frame on disk" if n_skipped else "") + ")...")

    out_dir = Path(args.output_dir) / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "predictions.jsonl"

    requests, references, responses = [], [], []

    with predictions_path.open("w") as pred_f:
        for start in range(0, len(items), args.batch_size):
            batch = items[start:start + args.batch_size]
            image_paths = [str(p) for _, _, p in batch]
            # Time the knowledge injection (RAG retrieval / kb lookup), amortized
            # per sample; cached per procedure_type so it is ~0 after first use.
            _t = time.monotonic()
            questions = [
                build_question(req.question, req.procedure_type,
                               args.condition_procedure, args.inject_knowledge,
                               args.knowledge_base, args.retriever)
                for req, _, _ in batch
            ]
            retrieve_time = (time.monotonic() - _t) / len(batch)

            answers, raw_texts, generate_time, total_time = generate_batch(
                processor, model, image_paths, questions, args.max_new_tokens,
                system_prompt=args.system_prompt,
            )

            for (req, ref, p), answer, raw in zip(batch, answers, raw_texts):
                requests.append(req)
                references.append(ref)
                responses.append(Response(qID=req.qID, content=answer, latency=generate_time))

                pred_f.write(json.dumps({
                    "qID": req.qID,
                    "videoID": req.videoID,
                    "image_path": str(p),
                    "question": req.question,
                    "primary_capability": ref.primary.name,
                    "format": ref._format,
                    "gt_answer": ref.answer,
                    "pred_answer": answer,
                    "raw_response": raw,
                    "generate_time": generate_time,   # model.generate(), amortized per sample
                    "retrieve_time": retrieve_time,   # knowledge injection (RAG/kb), amortized per sample
                    "total_time": total_time,         # generate + image load + tokenize, amortized
                    "full_time": total_time + retrieve_time,  # retrieval + generation (the real per-question cost)
                }) + "\n")
            pred_f.flush()

            done = min(start + args.batch_size, len(items))
            print(f"  [{dataset}] {done}/{len(items)} done")

    print(f"[{dataset}] wrote raw generations to {predictions_path}")

    evaluator = Evaluator(num_workers=args.judge_workers, judge_kwargs={"device": args.judge_device})
    results_df, summary_df = evaluator.run(requests, references, responses, output_dir=out_dir)

    print(f"\n[{dataset}] summary:")
    print(summary_df.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-dir", required=True,
                     help="trained model dir (LoRA adapter or full model, from sft_train_qwen_frame.py)")
    ap.add_argument("--base-model-id", default="Qwen/Qwen3.5-9B",
                     help="base model to load before applying the LoRA adapter (ignored for a full checkpoint)")
    ap.add_argument("--no-merge-lora", action="store_true",
                     help="keep the LoRA adapter separate instead of merging into the base weights")
    ap.add_argument("--datasets", nargs="+", default=["heico"], choices=["heico", "lapchole"])
    ap.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT_DIR)
    ap.add_argument("--output-dir", default=None,
                     help="defaults to <checkpoint-dir>/eval")
    ap.add_argument("--prompt-style", choices=["plain", "direct", "structured"], default="plain",
                     help="MUST match the --prompt-style the checkpoint was trained with. 'plain' "
                          "(default) = no system prompt, as all earlier runs. 'direct' = format-"
                          "control prompt. 'structured' = same plus a REASONING line (usable here "
                          "even though it cannot be trained on, since the ANSWER line is parsed "
                          "back out).")
    ap.add_argument("--condition-procedure", action="store_true",
                     help="prepend 'Procedure type: <name>.' to each question, surfacing "
                          "procedure_type (given at inference) so the model conditions on the "
                          "domain instead of collapsing to its training prior.")
    ap.add_argument("--inject-knowledge", choices=["none", "anatomy", "fo_priors", "both", "rag"],
                     default="none",
                     help="inject procedure-specific knowledge keyed on procedure_type: 'anatomy'/"
                          "'fo_priors'/'both' from the static --knowledge-file (targets situs/fo_class); "
                          "'rag' retrieves the anatomy block from the built RAG index at --rag-index. "
                          "Implies procedure conditioning.")
    ap.add_argument("--knowledge-file", type=Path,
                     default=Path(__file__).resolve().parent.parent / "orena_rag" / "procedure_knowledge.json",
                     help="JSON: {procedure_type: {anatomy, fo_priors}} used by --inject-knowledge.")
    ap.add_argument("--rag-index", type=Path,
                     default=Path(__file__).resolve().parent.parent / "orena_rag" / "rag_index",
                     help="dir with passages.jsonl + embeddings.npy for --inject-knowledge rag.")
    ap.add_argument("--fo-definitions", action="store_true",
                     help="with a non-plain --prompt-style, include the full per-class descriptions "
                          "instead of the class names alone.")
    ap.add_argument("--max-new-tokens", type=int, default=None,
                     help="default: 32 for plain/direct (terse answers), 128 for structured "
                          "(the REASONING line needs room). NOTE: raising this on a model that "
                          "answers in prose LOWERS the score -- open_ended/multiple_choice are "
                          "auto-failed above 300 characters by OpenEnded.verify.")
    ap.add_argument("--batch-size", type=int, default=32,
                     help="number of (image, question) pairs generated together per forward pass")
    ap.add_argument("--limit", type=int, default=None,
                     help="cap the number of test examples per dataset, for a quick sanity check")
    ap.add_argument("--judge-device", default="cuda",
                     help="device for the default LLM judge (TransformersJudge). Defaults to 'cuda': "
                          "the 4B judge in fp32 (~16GB) fits alongside the main model on one ~98GB GPU "
                          "and is vastly faster than CPU (which judges one question at a time, the eval "
                          "bottleneck). Use 'cpu' only if GPU memory is tight. Requires accelerate for cuda.")
    ap.add_argument("--judge-workers", type=int, default=1)
    args = ap.parse_args()

    if args.fo_definitions and args.prompt_style == "plain":
        ap.error("--fo-definitions has no effect with --prompt-style plain.")
    args.system_prompt = (
        build_system_prompt(args.fo_definitions, style=args.prompt_style)
        if args.prompt_style != "plain" else None
    )
    if args.max_new_tokens is None:
        args.max_new_tokens = 128 if args.prompt_style == "structured" else 32

    # One sub-folder per prompt arm so evaluating the same checkpoint under
    # different prompts never overwrites a previous result.
    style_tag = "" if args.prompt_style == "plain" else f"_{args.prompt_style}"
    if style_tag and args.fo_definitions:
        style_tag += "_defs"
    if args.condition_procedure and args.inject_knowledge == "none":
        style_tag += "_conditioned"
    if args.inject_knowledge != "none":
        style_tag += f"_kb-{args.inject_knowledge}"

    # Load the knowledge source once (keyed on procedure_type): static kb, or the
    # RAG retriever for --inject-knowledge rag.
    args.knowledge_base, args.retriever = None, None
    if args.inject_knowledge == "rag":
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from orena_rag.retriever import AnatomyRetriever
        args.retriever = AnatomyRetriever(args.rag_index)
        print(f"RAG retriever loaded from {args.rag_index}")
    elif args.inject_knowledge != "none":
        args.knowledge_base = json.loads(Path(args.knowledge_file).read_text())
    args.output_dir = args.output_dir or str(Path(args.checkpoint_dir) / f"eval{style_tag}")

    print(f"Prompt style: {args.prompt_style}{' +definitions' if args.fo_definitions else ''} | "
          f"max_new_tokens={args.max_new_tokens} | output -> {args.output_dir}")
    if args.system_prompt:
        bar = "=" * 78
        print(f"{bar}\nSYSTEM PROMPT (verbatim, in full)\n{bar}\n"
              f"{args.system_prompt}{bar}", flush=True)

    cfg = FocusConfig(root_dir=args.root_dir)
    set_config(cfg)

    processor, model = load_model(args.checkpoint_dir, args.base_model_id, not args.no_merge_lora)

    for dataset in args.datasets:
        evaluate_dataset(processor, model, dataset, cfg, args)

    print(f"\nDone. Per-dataset results.csv / summary.csv written under {args.output_dir}/")


if __name__ == "__main__":
    main()
