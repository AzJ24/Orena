"""SFT training script for Gemma 4 (google/gemma-4-31B-it) on the FOCUS frame track.

Mirror of `sft_train_qwen_frame.py` for Gemma 4. Consumes the same
`train.jsonl` / `eval.jsonl` from `build_frame_sft_dataset.py`, fine-tunes with
LoRA via TRL's SFTTrainer.

Everything that differs from the Qwen script comes from Gemma 4's model/
processor/chat-template, ALL verified empirically against
google/gemma-4-31B-it before writing (not assumed):

  1. Model class   : Gemma4ForConditionalGeneration (processor: Gemma4Processor
                     via AutoProcessor).
  2. Prompt / loss masking : training text is built as
     `build_prompt(user_turn) + answer + TURN_END`, where `build_prompt()` is
     Google's documented thinking-OFF inference prompt
     (`add_generation_prompt=True, enable_thinking=False`), rendering
     `…<|turn>model\\n<|channel>thought\\n<channel|>` -- an EMPTY, CLOSED thought
     block. Because the masked prefix IS the inference prompt, train and
     inference cannot drift. Verified: the prompt tokenization is an exact token
     prefix of the full tokenization, so only the answer carries loss.
  3. Image tensors : the processor returns `pixel_values` (n_images, 2520, 768)
     and `image_position_ids` (n_images, 2520, 2) -- fixed soft-token size per
     image, concatenated along dim 0 across the batch. There is NO
     `image_grid_thw` (that was Qwen-specific).
  4. mm_token_type_ids : Gemma 4 ALSO returns this (0 = text, 1 = image), and
     Gemma4ForConditionalGeneration.forward requires it for multimodal position
     handling -- same as Qwen, propagated and padded with 0.
  5. Thinking switch : `enable_thinking=False` is Google's documented way to
     turn thinking OFF -- it pre-fills the EMPTY closed block
     `<|channel>thought\\n<channel|>` so the model starts on the answer.
     `enable_thinking=True` instead injects `<|think|>` into a SYSTEM turn and
     the model emits real reasoning. (Earlier versions of this script hand-built
     a bare `<|turn>model\\n` prompt; that omitted the "thinking is done" signal
     and base Gemma then generated its own `thought\\n…` prefix.)

Requires `trl` and `peft`:
    uv add trl peft

Usage:
    .venv/bin/python orena_sft/sft_train_gemma_frame.py \\
        --output-dir ./checkpoints/gemma-4-31b-frame-sft
"""

from __future__ import annotations

import argparse
import json
import os
import socket
from datetime import datetime
from pathlib import Path

import torch
import wandb
from datasets import load_dataset
from PIL import Image
from transformers import (
    AutoProcessor,
    EarlyStoppingCallback,
    Gemma4ForConditionalGeneration,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer


# Terminator that closes a model turn, per Gemma 4's rendered chat format.
TURN_END = "<turn|>\n"


def build_prompt(processor, user_messages: list[dict]) -> str:
    """The inference prompt exactly as Google documents it for thinking-OFF.

    `enable_thinking=False` renders `…<|turn>model\\n<|channel>thought\\n<channel|>`
    -- an EMPTY, CLOSED thought block (`<|channel>` opens, `<channel|>` closes),
    i.e. "thinking is done, answer now". This is the same call used to build
    training targets (see `build_collate_fn`), so train and inference are
    consistent by construction rather than by coincidence."""
    return processor.apply_chat_template(
        user_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def build_collate_fn(processor):
    """Builds full-sequence inputs and masks the prompt out of the loss, keeping
    only the answer as the target.

    The training text is constructed as `build_prompt(user_turn) + answer +
    TURN_END`, i.e. the *literal inference prompt* followed by the gold answer.
    That makes the masked prefix byte-identical to what the model is fed at
    generation time (verified: the prompt tokenization is an exact token prefix
    of the full tokenization), so there is no train/inference format drift --
    and unlike the previous approach, the targets carry Gemma's documented empty
    thought block."""

    def collate_fn(examples: list[dict]) -> dict[str, torch.Tensor]:
        input_ids_list, labels_list = [], []
        mm_token_type_ids_list = []
        pixel_values_list, image_position_ids_list = [], []

        for ex in examples:
            messages = ex["messages"]
            image_path = messages[0]["content"][0]["image"]
            image = Image.open(image_path).convert("RGB")
            answer = messages[1]["content"][0]["text"]

            prompt_text = build_prompt(processor, messages[:1])
            full_text = prompt_text + answer + TURN_END

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
            image_position_ids_list.append(full["image_position_ids"])

        pad_id = processor.tokenizer.pad_token_id
        max_len = max(x.shape[0] for x in input_ids_list)

        batch_input_ids = torch.full((len(examples), max_len), pad_id, dtype=torch.long)
        batch_attention_mask = torch.zeros((len(examples), max_len), dtype=torch.long)
        batch_labels = torch.full((len(examples), max_len), -100, dtype=torch.long)
        # 0 = text token, 1 = image token (Gemma 4 convention, same as Qwen);
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
            # Fixed soft-token size per image -> concatenate along dim 0.
            "pixel_values": torch.cat(pixel_values_list, dim=0),
            "image_position_ids": torch.cat(image_position_ids_list, dim=0),
        }

    return collate_fn


class SampleGenerationCallback(TrainerCallback):
    """After every eval, generates a real answer for one fixed eval example and
    logs (image, question, gt_answer, pred_answer) to a local JSONL file and to
    wandb -- qualitative progress, not just the eval_loss curve."""

    def __init__(self, processor, sample: dict, output_dir: str, max_new_tokens: int = 32):
        self.processor = processor
        self.sample = sample
        self.output_path = Path(output_dir) / "eval_samples.jsonl"
        self.max_new_tokens = max_new_tokens

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        messages = self.sample["messages"]
        image_path = messages[0]["content"][0]["image"]
        question = messages[0]["content"][1]["text"]
        gt_answer = messages[1]["content"][0]["text"]
        image = Image.open(image_path).convert("RGB")

        prompt_text = build_prompt(self.processor, messages[:1])
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


def preview_example(processor, model, sample: dict, max_new_tokens: int = 64) -> None:
    """Print ONE concrete example at startup so the prompt structure and the
    model's actual output are visible before training: the full rendered chat
    (system / user / model turns, with `<|image|>` marking where image tokens
    go), the ground-truth answer (the training target), the generation prompt
    fed at inference, and a live greedy generation from the current weights."""
    messages = sample["messages"]
    image_path = messages[0]["content"][0]["image"]
    gt_answer = messages[1]["content"][0]["text"]
    image = Image.open(image_path).convert("RGB")

    full_text = processor.apply_chat_template(messages, tokenize=False)
    prompt_text = build_prompt(processor, messages[:1])

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
    print(f"\n{bar}\nEXAMPLE PREVIEW (before training)\n{bar}")
    print("--- FULL RENDERED CHAT (what training sees; note: no system prompt is used) ---")
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
    ap.add_argument("--model-id", default="google/gemma-4-31B-it")
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
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wandb-project", default="orena-frame-sft")
    ap.add_argument("--run-name", default=None,
                     help="wandb run name; auto-generated (model + lora/full + timestamp) if not set")
    args = ap.parse_args()

    if args.early_stopping_patience is not None and args.save_steps % args.eval_steps != 0:
        ap.error("--save-steps must be a multiple of --eval-steps for early stopping "
                  "(load_best_model_at_end needs a checkpoint at every eval point).")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = args.run_name or (
        f"{args.model_id.split('/')[-1]}-{'lora' if not args.no_lora else 'full'}-{timestamp}"
    )
    args.output_dir = args.output_dir or str(SFT_DIR / "checkpoints" / run_name)

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        args.model_id, dtype=torch.bfloat16, device_map="auto",
    )

    if not args.no_lora:
        from peft import LoraConfig, get_peft_model

        # Scope LoRA to the LANGUAGE MODEL only, via a regex on the full module
        # path. Gemma 4's vision tower ALSO has q_proj/k_proj/.../down_proj, but
        # those are `Gemma4ClippableLinear` wrappers, not plain nn.Linear, which
        # PEFT cannot adapt (raises "Target module ... is not supported"). The
        # `language_model.` prefix selects exactly the 410 plain-Linear text
        # projections and no vision modules -- verified against the model. This
        # also matches the Qwen run, whose vision tower used different names and
        # so was effectively text-only LoRA too.
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
            target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
            task_type="CAUSAL_LM",
        ))
        model.print_trainable_parameters()

    dataset = load_dataset("json", data_files={
        "train": args.train_file,
        "eval": args.eval_file,
    })

    effective_batch_size = args.batch_size * args.grad_accum
    # Explicit wandb.init (before SFTTrainer) so the run lands in a real project
    # with a timestamped name + full config snapshot; Trainer's WandbCallback
    # reuses the active run.
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "model_id": args.model_id,
            "lora": not args.no_lora,
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
        save_total_limit=2,
        report_to="wandb",
        seed=args.seed,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        load_best_model_at_end=args.early_stopping_patience is not None,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    callbacks = [SampleGenerationCallback(processor, dataset["eval"][0], args.output_dir)]
    if args.early_stopping_patience is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        data_collator=build_collate_fn(processor),
        # Full processor (not just tokenizer) so intermediate checkpoints get a
        # loadable preprocessor_config.json.
        processing_class=processor,
        callbacks=callbacks,
    )

    # Show one concrete example (prompt structure + a live generation) before
    # training kicks off, so the output type is visible up front.
    preview_example(processor, model, dataset["eval"][0])

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
