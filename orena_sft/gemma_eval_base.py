"""Evaluate the BASE (un-fine-tuned) google/gemma-4-31B-it on the FOCUS frame track.

Deliberately kept simple and aligned with Google's official usage guidance for
gemma-4-31B-it -- one `apply_chat_template(tokenize=True, ...)` call that also
handles the image, `enable_thinking=False`, decode with
`skip_special_tokens=False`, then `processor.parse_response(...)`:

    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt",
        add_generation_prompt=True, enable_thinking=False,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    outputs = model.generate(**inputs, max_new_tokens=...)
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    processor.parse_response(response)

Why this matters (learned the hard way):
  * `enable_thinking=False` renders `<|turn>model\\n<|channel>thought\\n<channel|>`
    -- an EMPTY, CLOSED thought block (analogous to Qwen's `<think></think>`),
    i.e. thinking OFF. Omitting it entirely leaves the model with no "thinking
    is done" signal, and base Gemma then emits its own `thought\\n...` prefix,
    which wrecks the deterministic answer parsers.
  * `parse_response()` splits the raw output into `{"thinking": ..., "content": ...}`,
    so the scored answer is `content` -- no hand-rolled prefix stripping.

Generation is one sample at a time, matching Google's snippet: simple and
obviously correct for a baseline check. Use `--limit` to keep runs short.

`--prompt-style structured` swaps the bare question for the system prompt in
`prompts.py`: the FO definition, the registered class vocabulary, a REASONING
line (chain of thought in plain text, independent of `--enable-thinking`), and
an ANSWER line that is parsed back out so the deterministic formats still get a
bare value. It exists to answer "is SFT needed here, or just a better prompt?"
-- SFT bought ~+25 points mostly by teaching output format, at the cost of
catastrophic forgetting; if prompting recovers the format gap, that compliance
comes without the forgetting. Each arm writes to its own `eval_base/` folder.

Usage (h200 needed for 31B):
    .venv/bin/python orena_sft/gemma_eval_base.py --datasets lapchole --limit 50
    .venv/bin/python orena_sft/gemma_eval_base.py --datasets lapchole --limit 50 \\
        --prompt-style structured --batch-size 16
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
from focus.data.data_models import Response
from focus.evaluation import Evaluator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_frame_sft_dataset import DEFAULT_ROOT_DIR, frame_path  # noqa: E402
from prompts import build_system_prompt, extract_answer  # noqa: E402

from focus.config import DATASET_BASE_FPS  # noqa: E402

SFT_DIR = Path(__file__).resolve().parent


@torch.no_grad()
def generate_answers(processor, model, image_paths: list[str], questions: list[str],
                     max_new_tokens: int, enable_thinking: bool = False,
                     system_prompt: str | None = None):
    """N (image, question) pairs -> N (answer, thinking, raw) + amortized seconds.

    Same flow Google documents, just handed a LIST of conversations instead of
    one: `apply_chat_template` batches all five tensors (input_ids,
    attention_mask, mm_token_type_ids, pixel_values, image_position_ids) and
    left-pads automatically, so `input_len` is a single shared column and new
    tokens slice off cleanly for every row. With --batch-size 1 this is
    byte-for-byte Google's snippet.

    `enable_thinking` is the documented switch:
      * False -> prompt pre-fills an EMPTY closed thought block
                 (`<|channel>thought\\n<channel|>`), model answers directly.
      * True  -> `<|think|>` is injected into the system prompt; the model emits
                 `<|channel>thought\\n[reasoning]<channel|>` then the answer, and
                 needs a much larger max_new_tokens or the block never closes
                 and `content` comes back empty.
    `parse_response()` splits reasoning from the answer either way.

    `system_prompt` (the structured prompt from `prompts.py`) is orthogonal to
    `enable_thinking`: its chain of thought is a plain REASONING line in the
    visible output, not the native thought channel, so the two can be combined
    or used separately. When set, the returned answer is the extracted ANSWER
    value rather than the whole `content`. Verified by rendering: adding a
    system turn with enable_thinking=False still ends the prompt in the empty
    CLOSED `<|channel>thought\\n<channel|>` block and injects no `<|think|>`,
    i.e. it does not silently flip thinking back on.

    Time is amortized (batch wall-clock / batch size) -- a batched generate()
    can't attribute latency to individual sequences.
    """
    system_turn = (
        [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]
        if system_prompt else []
    )
    convs = [
        system_turn + [{"role": "user", "content": [
            {"type": "image", "image": Image.open(p).convert("RGB")},
            {"type": "text", "text": q},
        ]}]
        for p, q in zip(image_paths, questions)
    ]

    inputs = processor.apply_chat_template(
        convs,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    start = time.monotonic()
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    elapsed = time.monotonic() - start

    results = []
    for row in outputs:
        # skip_special_tokens=False so parse_response can see the channel markers.
        raw = processor.decode(row[input_len:], skip_special_tokens=False)
        parsed = processor.parse_response(raw)
        content = (parsed.get("content") or "").strip()
        results.append((
            extract_answer(content) if system_prompt else content,
            (parsed.get("thinking") or "").strip(),
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
          + (f", {n_skipped} skipped: no frame on disk" if n_skipped else "") + ")...")

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
                    "pred_answer": answer,      # parse_response -> content
                    "thinking": thinking,       # parse_response -> thinking (empty when off)
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
    ap.add_argument("--model-id", default="google/gemma-4-31B-it")
    ap.add_argument("--datasets", nargs="+", default=["lapchole"], choices=["heico", "lapchole"])
    ap.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT_DIR)
    ap.add_argument("--output-dir", default=None,
                     help="defaults to <script-dir>/eval_base/<model-id-basename>")
    ap.add_argument("--enable-thinking", action="store_true",
                     help="turn Gemma's thinking mode ON (injects <|think|> into the system "
                          "prompt). Off by default. parse_response() separates reasoning from "
                          "the answer either way, so scoring is unaffected -- but thinking is "
                          "MUCH slower and needs a big --max-new-tokens.")
    ap.add_argument("--prompt-style", choices=["plain", "structured", "direct"], default="plain",
                     help="'plain' (default) sends the bare question, as every earlier baseline "
                          "did. 'structured' prepends the system prompt from prompts.py: the FO "
                          "definition + the registered class vocabulary, a REASONING line, and an "
                          "ANSWER line parsed back out for exact matching. Format-agnostic and "
                          "free of anatomy/procedure vocabulary, so it stays OOD-safe.")
    ap.add_argument("--fo-definitions", action="store_true",
                     help="with --prompt-style structured, include the full per-class descriptions "
                          "from FO_DEFINITIONS_FILE (~700 extra prompt tokens per question) instead "
                          "of the class names alone. Separate arm: recognition help vs. latency.")
    ap.add_argument("--max-new-tokens", type=int, default=None,
                     help="default: 128 without thinking, 1024 with --enable-thinking "
                          "(reasoning must fit AND close before the answer is emitted)")
    ap.add_argument("--batch-size", type=int, default=1,
                     help="how many (image, question) pairs to generate together. Default 1 = "
                          "exactly Google's documented per-sample flow. Raise it (e.g. 16/32) for "
                          "a large speedup, especially with --enable-thinking.")
    ap.add_argument("--limit", type=int, default=None,
                     help="cap test examples per dataset (recommended for a fast baseline check)")
    ap.add_argument("--judge-device", default="cuda")
    ap.add_argument("--judge-workers", type=int, default=1)
    args = ap.parse_args()

    if args.max_new_tokens is None:
        args.max_new_tokens = 1024 if args.enable_thinking else 128

    prompted = args.prompt_style != "plain"
    if args.fo_definitions and not prompted:
        ap.error("--fo-definitions has no effect with --prompt-style plain.")
    args.system_prompt = (
        build_system_prompt(args.fo_definitions, style=args.prompt_style) if prompted else None
    )

    # Keep every arm's results in its own folder so runs never overwrite each
    # other (the checkpoint-collision lesson, applied to eval output).
    if args.output_dir is None:
        suffix = "_thinking" if args.enable_thinking else "_wo_thinking"
        if prompted:
            suffix += f"_{args.prompt_style}" + ("_defs" if args.fo_definitions else "")
        args.output_dir = str(SFT_DIR / "eval_base" / (args.model_id.split("/")[-1] + suffix))

    print(f"Thinking mode: {'ON' if args.enable_thinking else 'OFF'} | "
          f"prompt: {args.prompt_style}{' +definitions' if args.fo_definitions else ''} | "
          f"max_new_tokens={args.max_new_tokens} | output -> {args.output_dir}")
    if args.system_prompt:
        bar = "=" * 78
        print(f"{bar}\nSYSTEM PROMPT (verbatim, in full)\n{bar}\n"
              f"{args.system_prompt}{bar}", flush=True)

    cfg = FocusConfig(root_dir=args.root_dir)
    set_config(cfg)

    print(f"Loading base model {args.model_id!r} (no fine-tuning)...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    # Left-pad so that with --batch-size > 1 every sequence's first generated
    # token lands at the same column, making `input_len` a single shared prompt
    # boundary. Harmless at batch size 1 (nothing to pad).
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
