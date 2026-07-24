"""Evaluates a fine-tuned Gemma 4 checkpoint on the FOCUS frame-track TEST
split, using the FOCUS library's own :class:`focus.evaluation.Evaluator` --
deterministic parsing/comparison for closed-form formats (binary, number,
percentage, fo_class, time), LLM-as-judge for open-ended ones (open_ended,
matching, multiple_choice).

Mirror of `evaluate_qwen_frame.py` for Gemma 4. Everything that differs was
verified empirically against google/gemma-4-31B-it (not assumed):

  1. Model class : Gemma4ForConditionalGeneration (vs Qwen3_5ForConditionalGeneration).
  2. Generation prompt : Google's documented thinking-OFF call
     (`add_generation_prompt=True, enable_thinking=False`), which renders
     `…<|turn>model\\n<|channel>thought\\n<channel|>` -- an EMPTY, CLOSED thought
     block meaning "thinking done, answer now". This is the identical call
     `sft_train_gemma_frame.build_prompt()` uses for training targets, so eval
     and training cannot drift. The answer is extracted with
     `processor.parse_response()` -> `content`.
  3. Batched images : Gemma's processor requires a NESTED per-text image list
     (`[[img0], [img1], ...]`), not Qwen's flat `[img0, img1, ...]` -- passing a
     flat list raises "inconsistently sized batches of images and text".

Generation is batched (`--batch-size`) with left-padding. Timing is amortized
per sample (batch wall-clock / batch size).

Writes, per dataset, under `<output-dir>/<dataset>/`:
  - `predictions.jsonl` -- every raw (question, gt_answer, pred_answer) triple.
  - `results.csv` / `summary.csv` -- from Evaluator.run().

Usage:
    .venv/bin/python orena_sft/evaluate_gemma_frame.py \\
        --checkpoint-dir ./orena_sft/checkpoints/gemma-4-31b-heico-only \\
        --datasets lapchole
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Gemma4ForConditionalGeneration

from focus import DatasetSplit, FocusConfig, FocusDataset, Track, set_config
from focus.data.data_models import Response
from focus.evaluation import Evaluator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_frame_sft_dataset import DEFAULT_ROOT_DIR, frame_path  # noqa: E402

from focus.config import DATASET_BASE_FPS  # noqa: E402

SFT_DIR = Path(__file__).resolve().parent


def load_model(checkpoint_dir: str, base_model_id: str, merge_lora: bool):
    # Load the processor from base_model_id, not checkpoint_dir: the image
    # processor/chat template never change during LoRA fine-tuning, and
    # intermediate Trainer checkpoints only ever contain the bare tokenizer,
    # not the full multimodal processor.
    processor = AutoProcessor.from_pretrained(base_model_id)
    # Left-pad for batched generation (see evaluate_qwen_frame.py for rationale).
    processor.tokenizer.padding_side = "left"
    is_lora = (Path(checkpoint_dir) / "adapter_config.json").exists()

    if is_lora:
        from peft import PeftModel

        print(f"Loading base model {base_model_id!r}, then LoRA adapter from {checkpoint_dir!r}...")
        base_model = Gemma4ForConditionalGeneration.from_pretrained(
            base_model_id, dtype=torch.bfloat16, device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, checkpoint_dir)
        if merge_lora:
            model = model.merge_and_unload()
    else:
        print(f"Loading full fine-tuned model from {checkpoint_dir!r}...")
        model = Gemma4ForConditionalGeneration.from_pretrained(
            checkpoint_dir, dtype=torch.bfloat16, device_map="auto",
        )

    model.eval()
    return processor, model


@torch.no_grad()
def generate_batch(
    processor, model, image_paths: list[str], questions: list[str], max_new_tokens: int
) -> tuple[list[str], float, float]:
    """Generate answers for a batch of (image, question) pairs.

    Returns (answers, generate_time, total_time) with the two times amortized
    per sample (batch wall-clock / batch size). Relies on left-padding so the
    prompt boundary is a single shared column across the batch.
    """
    total_start = time.monotonic()

    images = [Image.open(p).convert("RGB") for p in image_paths]
    # Google's documented thinking-OFF prompt: renders
    # `…<|turn>model\n<|channel>thought\n<channel|>` (an EMPTY, CLOSED thought
    # block). This is the exact call sft_train_gemma_frame.build_prompt() uses
    # to build training targets, so eval and training cannot drift.
    texts = [
        processor.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "image", "image": p},
                {"type": "text", "text": q},
            ]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        for p, q in zip(image_paths, questions)
    ]
    # Gemma's processor requires a NESTED per-text image list, not a flat one.
    inputs = processor(
        text=texts, images=[[img] for img in images], return_tensors="pt", padding=True,
    ).to(model.device)

    generate_start = time.monotonic()
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generate_time = time.monotonic() - generate_start

    new_tokens = generated[:, inputs["input_ids"].shape[1]:]
    # skip_special_tokens=False so parse_response() can see the channel markers;
    # it returns {"thinking": ..., "content": ...} and strips the <turn|>
    # terminator, so no hand-rolled prefix/suffix stripping is needed.
    answers = []
    for row in new_tokens:
        raw = processor.decode(row, skip_special_tokens=False)
        parsed = processor.parse_response(raw)
        answers.append((parsed.get("content") or "").strip())

    total_time = time.monotonic() - total_start
    bs = len(image_paths)
    return answers, generate_time / bs, total_time / bs


def evaluate_dataset(processor, model, dataset: str, cfg: FocusConfig, args) -> None:
    base_fps = float(DATASET_BASE_FPS[dataset])
    ds = FocusDataset(dataset, DatasetSplit.TEST, Track.FRAME)

    n = len(ds) if args.limit is None else min(args.limit, len(ds))

    # Gather valid (req, ref, path) triples upfront so frames missing on disk
    # are dropped before they can unbalance a batch.
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
            questions = [req.question for req, _, _ in batch]

            answers, generate_time, total_time = generate_batch(
                processor, model, image_paths, questions, args.max_new_tokens
            )

            for (req, ref, p), answer in zip(batch, answers):
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
                    "generate_time": generate_time,  # amortized per sample within the batch
                    "total_time": total_time,        # amortized per sample within the batch
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
                     help="trained model dir (LoRA adapter or full model, from sft_train_gemma_frame.py)")
    ap.add_argument("--base-model-id", default="google/gemma-4-31B-it",
                     help="base model to load before applying the LoRA adapter (ignored for a full checkpoint)")
    ap.add_argument("--no-merge-lora", action="store_true",
                     help="keep the LoRA adapter separate instead of merging into the base weights")
    ap.add_argument("--datasets", nargs="+", default=["heico"], choices=["heico", "lapchole"])
    ap.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT_DIR)
    ap.add_argument("--output-dir", default=None,
                     help="defaults to <checkpoint-dir>/eval")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=32,
                     help="number of (image, question) pairs generated together per forward pass")
    ap.add_argument("--limit", type=int, default=None,
                     help="cap the number of test examples per dataset, for a quick sanity check")
    ap.add_argument("--judge-device", default="cuda",
                     help="device for the default LLM judge (TransformersJudge). Defaults to 'cuda'; "
                          "use 'cpu' only if GPU memory is tight. Requires accelerate for cuda.")
    ap.add_argument("--judge-workers", type=int, default=1)
    args = ap.parse_args()

    args.output_dir = args.output_dir or str(Path(args.checkpoint_dir) / "eval")

    cfg = FocusConfig(root_dir=args.root_dir)
    set_config(cfg)

    processor, model = load_model(args.checkpoint_dir, args.base_model_id, not args.no_merge_lora)

    for dataset in args.datasets:
        evaluate_dataset(processor, model, dataset, cfg, args)

    print(f"\nDone. Per-dataset results.csv / summary.csv written under {args.output_dir}/")


if __name__ == "__main__":
    main()
