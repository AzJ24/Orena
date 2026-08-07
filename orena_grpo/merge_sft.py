"""Merge an SFT LoRA adapter into the base Qwen weights and save a full model.
Needed for evaluating warm-start GRPO checkpoints: the GRPO adapter was trained
on top of base+SFT, so eval must load base+SFT as its base before applying it.

By default the merged model lands in `<repo>/models/<checkpoint-name>`, i.e. the
same name as the adapter's checkpoint folder, so merged models sit together and
stay traceable to the run that produced them. Override with --out-dir.
"""
import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model-id", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--sft-adapter-dir", required=True)
    ap.add_argument("--out-dir", default=None,
                     help="defaults to <repo>/models/<checkpoint-name> (the adapter dir's basename)")
    ap.add_argument("--models-dir", type=Path, default=MODELS_DIR,
                     help="where merged models are collected when --out-dir is not given")
    args = ap.parse_args()

    if args.out_dir is None:
        args.out_dir = str(args.models_dir / Path(args.sft_adapter_dir).name)

    print(f"Loading base {args.base_model_id} on CPU...")
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.base_model_id, dtype=torch.bfloat16, device_map="cpu",
    )
    print(f"Applying + merging SFT adapter {args.sft_adapter_dir}...")
    merged = PeftModel.from_pretrained(base, args.sft_adapter_dir).merge_and_unload()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.out_dir)
    AutoProcessor.from_pretrained(args.base_model_id).save_pretrained(args.out_dir)
    print(f"Saved merged base+SFT to {args.out_dir}")


if __name__ == "__main__":
    main()
