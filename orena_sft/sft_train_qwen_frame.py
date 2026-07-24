"""SFT training script for Qwen3.5 on the FOCUS frame track.

Consumes the `train.jsonl` / `eval.jsonl` produced by
`build_frame_sft_dataset.py` (chat-format records, each with one image
referenced by path). Fine-tunes with LoRA by default via TRL's SFTTrainer.

Requires `trl` and `peft`, which aren't in pyproject.toml yet:
    uv add trl peft

Usage:
    .venv/bin/python orena_sft/sft_train_qwen_frame.py \\
        --output-dir ./checkpoints/qwen3.5-9b-frame-sft
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from datetime import datetime
from pathlib import Path

import torch
import wandb
from datasets import load_dataset
from PIL import Image
from transformers import (
    AutoProcessor,
    EarlyStoppingCallback,
    Qwen3_5ForConditionalGeneration,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import build_system_prompt  # noqa: E402


ASSISTANT_MARKER = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def with_system(messages: list[dict], system_prompt: str | None) -> list[dict]:
    """Prepend the system turn, or return `messages` untouched when there is no
    prompt (`--prompt-style plain`, the historical behaviour).

    Used at all three places the chat is rendered -- the collator, the eval
    sample callback, and the startup preview -- so training and every
    generation path can never disagree about what the model is being shown.
    """
    if not system_prompt:
        return messages
    return [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}] + messages


def build_collate_fn(processor, system_prompt: str | None = None):
    """Builds full-sequence inputs and masks out everything up to and
    including the assistant marker from the loss, keeping only the answer
    text as a training target.

    The Qwen3.5 chat template always renders an already-answered assistant
    turn as `<|im_start|>assistant\\n<think>\\n{reasoning}\\n</think>\\n\\n{answer}`
    -- with an empty (but still present) think block whenever no
    reasoning_content is supplied, which is always true for our QA data.
    That differs from what `add_generation_prompt=True` renders for a *fresh*
    turn (thinking left open, `<think>\\n`, since enable_thinking defaults to
    True there), so re-deriving the prompt length from a separate
    add_generation_prompt call would diverge from the actual full-text
    boundary. Instead we find the literal marker inside `full_text` itself
    and slice -- guaranteed to be an exact string prefix, so the tokenized
    prefix is guaranteed to match the corresponding prefix of the full
    tokenization.

    A system prompt changes nothing about this: verified against the real
    processor that the marker still occurs exactly once, the prompt tokens are
    still an exact prefix of the full tokenization, and the unmasked target is
    byte-identical (6 tokens for a `fo_class` answer). The system turn sits
    before the marker, so it is masked out of the loss like the rest of the
    prompt -- it only grows the sequence (552 -> 952 tokens)."""

    def collate_fn(examples: list[dict]) -> dict[str, torch.Tensor]:
        input_ids_list, labels_list = [], []
        mm_token_type_ids_list = []
        pixel_values_list, grid_thw_list = [], []

        for ex in examples:
            messages = ex["messages"]
            image_path = messages[0]["content"][0]["image"]
            image = Image.open(image_path).convert("RGB")

            full_text = processor.apply_chat_template(
                with_system(messages, system_prompt), tokenize=False
            )
            marker_idx = full_text.rindex(ASSISTANT_MARKER) + len(ASSISTANT_MARKER)
            prompt_text = full_text[:marker_idx]

            full = processor(text=[full_text], images=[image], return_tensors="pt")
            prompt = processor(text=[prompt_text], images=[image], return_tensors="pt")
            prompt_len = prompt["input_ids"].shape[1]

            input_ids = full["input_ids"][0]
            labels = input_ids.clone()
            labels[:prompt_len] = -100

            input_ids_list.append(input_ids)
            labels_list.append(labels)
            mm_token_type_ids_list.append(full["mm_token_type_ids"][0])
            pixel_values_list.append(full["pixel_values"])
            grid_thw_list.append(full["image_grid_thw"])

        pad_id = processor.tokenizer.pad_token_id
        max_len = max(x.shape[0] for x in input_ids_list)

        batch_input_ids = torch.full((len(examples), max_len), pad_id, dtype=torch.long)
        batch_attention_mask = torch.zeros((len(examples), max_len), dtype=torch.long)
        batch_labels = torch.full((len(examples), max_len), -100, dtype=torch.long)
        # 0 = text token, 1 = image token (per Qwen3.5's M-RoPE convention);
        # padding is typed 0 like any other non-image filler token.
        batch_mm_token_type_ids = torch.zeros((len(examples), max_len), dtype=torch.long)

        for i, (ids, lbl, tt) in enumerate(zip(input_ids_list, labels_list, mm_token_type_ids_list)):
            n = ids.shape[0]
            batch_input_ids[i, :n] = ids
            batch_attention_mask[i, :n] = 1
            batch_labels[i, :n] = lbl
            batch_mm_token_type_ids[i, :n] = tt

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels,
            "mm_token_type_ids": batch_mm_token_type_ids,
            "pixel_values": torch.cat(pixel_values_list, dim=0),
            "image_grid_thw": torch.cat(grid_thw_list, dim=0),
        }

    return collate_fn


class SampleGenerationCallback(TrainerCallback):
    """After every eval, generates a real answer for one fixed eval example
    and logs (image, question, gt_answer, pred_answer) to a local JSONL file
    and to wandb -- so training progress can be eyeballed qualitatively,
    not just read off the eval_loss curve."""

    def __init__(self, processor, sample: dict, output_dir: str, max_new_tokens: int = 32,
                 system_prompt: str | None = None):
        self.processor = processor
        self.sample = sample
        self.output_path = Path(output_dir) / "eval_samples.jsonl"
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        messages = self.sample["messages"]
        image_path = messages[0]["content"][0]["image"]
        question = messages[0]["content"][1]["text"]
        gt_answer = messages[1]["content"][0]["text"]
        image = Image.open(image_path).convert("RGB")

        prompt_text = self.processor.apply_chat_template(
            with_system(messages[:1], self.system_prompt),
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = self.processor(text=[prompt_text], images=[image], return_tensors="pt").to(model.device)

        was_training = model.training
        model.eval()
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        model.train(was_training)

        new_tokens = generated[0][inputs["input_ids"].shape[1]:]
        pred_answer = self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        record = {
            "step": state.global_step,
            "qID": self.sample.get("qID"),
            "videoID": self.sample.get("videoID"),
            "image_path": image_path,
            "question": question,
            "gt_answer": gt_answer,
            "pred_answer": pred_answer,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

        if wandb.run is not None:
            wandb.log({
                "eval_sample/question": question,
                "eval_sample/gt_answer": gt_answer,
                "eval_sample/pred_answer": pred_answer,
                "eval_sample/image": wandb.Image(image_path),
            }, step=state.global_step)


SFT_DIR = Path(__file__).resolve().parent


def preview_example(processor, model, sample: dict, max_new_tokens: int = 64,
                    system_prompt: str | None = None) -> None:
    """Print ONE concrete example at startup so the prompt structure and the
    model's actual output are visible before training: the full rendered chat
    (system / user / model turns, with `<|image_pad|>` marking image tokens),
    the ground-truth answer (the training target), the generation prompt fed at
    inference, and a live greedy generation from the current weights."""
    messages = sample["messages"]
    image_path = messages[0]["content"][0]["image"]
    gt_answer = messages[1]["content"][0]["text"]
    image = Image.open(image_path).convert("RGB")

    full_text = processor.apply_chat_template(with_system(messages, system_prompt), tokenize=False)
    prompt_text = processor.apply_chat_template(
        with_system(messages[:1], system_prompt),
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )

    inputs = processor(text=[prompt_text], images=[image], return_tensors="pt").to(model.device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    model.train(was_training)
    pred = processor.tokenizer.decode(
        generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

    bar = "=" * 80
    prompt_note = "no system prompt" if not system_prompt else "with system prompt"
    print(f"\n{bar}\nEXAMPLE PREVIEW (before training)\n{bar}")
    print(f"--- FULL RENDERED CHAT (what training sees; {prompt_note}) ---")
    print(full_text)
    print("--- GROUND-TRUTH ANSWER (training target) ---")
    print(repr(gt_answer))
    print("--- GENERATION PROMPT (fed to the model at inference) ---")
    print(repr(prompt_text))
    print("--- MODEL'S ACTUAL GENERATION (current / untrained-adapter weights) ---")
    print(repr(pred))
    print(f"{bar}\n", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-file", default=str(SFT_DIR / "sft_export" / "combined" / "train.jsonl"))
    ap.add_argument("--eval-file", default=str(SFT_DIR / "sft_export" / "combined" / "eval.jsonl"))
    ap.add_argument("--output-dir", default=None,
                     help="defaults to <script-dir>/checkpoints/<run_name> if not set")
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1,
                     help="cap total optimizer steps (overrides --epochs when >0); "
                          "use for a quick calibration/smoke run, e.g. --max-steps 20")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--no-lora", action="store_true", help="full fine-tune instead of LoRA")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--eval-steps", type=int, default=200)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--early-stopping-patience", type=int, default=None,
                     help="stop training if eval_loss doesn't improve for this many consecutive "
                          "evals (each eval happens every --eval-steps); disabled if not set. "
                          "Requires --save-steps to be a multiple of --eval-steps.")
    ap.add_argument("--prompt-style", choices=["plain", "direct", "structured"], default="plain",
                     help="system prompt prepended to EVERY training example. 'plain' (default) "
                          "sends none -- identical to all previous runs. 'direct' prepends the "
                          "format-control prompt from prompts.py (FO class vocabulary + answer "
                          "shape rules), so the model learns to READ the label space instead of "
                          "memorising it. Evaluate with the SAME --prompt-style or the numbers are "
                          "a train/inference mismatch.")
    ap.add_argument("--fo-definitions", action="store_true",
                     help="with a non-plain --prompt-style, include the full per-class descriptions "
                          "instead of the class names alone.")
    ap.add_argument("--system-prompt-file", type=Path, default=None,
                     help="train with a system prompt read verbatim from this file (e.g. a "
                          "GEPA-evolved best_prompt.txt) instead of build_system_prompt(). Overrides "
                          "--prompt-style / --fo-definitions. The prompt MUST expect a bare answer "
                          "(the targets are bare answers); evaluate the checkpoint with the SAME file.")
    ap.add_argument("--condition-procedure", action="store_true",
                     help="prepend 'Procedure type: <name>.' to each training question (from the "
                          "record's procedure_type). Train and eval must match, so evaluate the "
                          "checkpoint with the same --condition-procedure.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wandb-project", default="orena-frame-sft")
    ap.add_argument("--run-name", default=None,
                     help="wandb run name; auto-generated (model + lora/full + timestamp) if not set")
    args = ap.parse_args()

    if args.early_stopping_patience is not None and args.save_steps % args.eval_steps != 0:
        ap.error("--save-steps must be a multiple of --eval-steps for early stopping "
                  "(load_best_model_at_end needs a checkpoint at every eval point).")

    # 'structured' asks for a REASONING line then an ANSWER line, but every
    # training target in the JSONL is a bare answer ("Clip, Silicone loop").
    # Training on that pair would teach the model to IGNORE the prompt it is
    # given -- the opposite of the intent. Supporting it needs reasoning traces
    # in the targets (rejection-sampled from a teacher), which is a different
    # project; refuse loudly rather than train something quietly incoherent.
    if args.system_prompt_file is None and args.prompt_style == "structured":
        ap.error("--prompt-style structured cannot be trained on: the targets in the JSONL are "
                  "bare answers with no REASONING line, so the model would learn to disobey the "
                  "prompt. Use 'direct' (bare answer, matches the targets) or 'plain'.")
    if args.fo_definitions and args.prompt_style == "plain" and args.system_prompt_file is None:
        ap.error("--fo-definitions has no effect with --prompt-style plain.")

    if args.system_prompt_file is not None:
        # A verbatim evolved prompt (bare-answer style) replaces build_system_prompt.
        system_prompt = args.system_prompt_file.read_text()
    elif args.prompt_style != "plain":
        system_prompt = build_system_prompt(args.fo_definitions, style=args.prompt_style)
    else:
        system_prompt = None

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    style_tag = ("-promptfile" if args.system_prompt_file is not None
                 else "" if args.prompt_style == "plain" else f"-{args.prompt_style}")
    run_name = args.run_name or (
        f"{args.model_id.split('/')[-1]}-{'lora' if not args.no_lora else 'full'}"
        f"{style_tag}-{timestamp}"
    )
    args.output_dir = args.output_dir or str(SFT_DIR / "checkpoints" / run_name)

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_id, dtype=torch.bfloat16, device_map="auto",
    )

    if not args.no_lora:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        ))
        model.print_trainable_parameters()

    dataset = load_dataset("json", data_files={
        "train": args.train_file,
        "eval": args.eval_file,
    })

    if args.condition_procedure:
        # Prepend "Procedure type: <name>." to each user question, once at load
        # time, so the collator / eval callback / preview all render it. Eval must
        # use the same --condition-procedure.
        def _condition(ex):
            proc = ex.get("procedure_type")
            if proc:
                for turn in ex["messages"]:
                    if turn["role"] == "user":
                        for c in turn["content"]:
                            if c.get("type") == "text":
                                c["text"] = f"Procedure type: {proc}.\n{c['text']}"
                                break
                        break
            return ex
        dataset = dataset.map(_condition)

    effective_batch_size = args.batch_size * args.grad_accum
    # Trainer's WandbCallback reuses an already-active run instead of
    # starting its own, so initializing explicitly here (with a real project
    # name, timestamped run name, and a full config snapshot) is what
    # actually shows up on the dashboard -- report_to="wandb" alone only
    # gets you the default "huggingface" project with an auto-generated name.
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "model_id": args.model_id,
            "lora": not args.no_lora,
            "prompt_style": args.prompt_style,
            "fo_definitions": args.fo_definitions,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "train_file": args.train_file,
            "eval_file": args.eval_file,
            "n_train_examples": len(dataset["train"]),
            "n_eval_examples": len(dataset["eval"]),
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "early_stopping_patience": args.early_stopping_patience,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "effective_batch_size": effective_batch_size,
            "learning_rate": args.lr,
            "seed": args.seed,
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "started_at": timestamp,
        },
    )

    config = SFTConfig(
        output_dir=args.output_dir,
        run_name=run_name,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=20,
        report_to="wandb",
        seed=args.seed,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        load_best_model_at_end=args.early_stopping_patience is not None,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    callbacks = [SampleGenerationCallback(processor, dataset["eval"][0], args.output_dir,
                                          system_prompt=system_prompt)]
    if args.early_stopping_patience is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        data_collator=build_collate_fn(processor, system_prompt),
        # The full processor, not just processor.tokenizer: Trainer auto-saves
        # whatever processing_class is at every intermediate checkpoint
        # (checkpoint-N/), so passing only the tokenizer left those without
        # a preprocessor_config.json -- AutoProcessor.from_pretrained() on
        # such a checkpoint later silently degrades to a bare tokenizer.
        processing_class=processor,
        callbacks=callbacks,
    )

    # Show one concrete example (prompt structure + a live generation) before
    # training kicks off, so the output type is visible up front.
    preview_example(processor, model, dataset["eval"][0], system_prompt=system_prompt)

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
