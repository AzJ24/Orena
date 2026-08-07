"""Fold a LoRA adapter into the base weights to produce a standalone model.

Merging is not just packaging: at inference it removes the adapter's extra matmul
from every targeted module, and it lets the submission container load one model
instead of base + adapter. The result is a plain Qwen3_5ForConditionalGeneration
that needs no peft at runtime.

The merge is verified rather than assumed -- a targeted projection is captured
before and after `merge_and_unload()`, and an unadapted tensor is checked to be
untouched. A silent no-op merge would otherwise ship the base model.

    python segment_track/merge_lora.py --adapter-dir <run> --out-dir <run>-merged
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

SEG_DIR = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter-dir", required=True)
    ap.add_argument("--base-model-id", default=None,
                    help="defaults to base_model_name_or_path from adapter_config.json")
    ap.add_argument("--out-dir", default=None, help="defaults to <adapter-dir>-merged")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    adapter = Path(args.adapter_dir)
    cfg = json.loads((adapter / "adapter_config.json").read_text())
    base_id = args.base_model_id or cfg["base_model_name_or_path"]
    out = Path(args.out_dir or f"{adapter}-merged")

    print(f"adapter : {adapter}")
    print(f"base    : {base_id}")
    print(f"out     : {out}")
    print(f"lora    : r={cfg.get('r')} alpha={cfg.get('lora_alpha')} "
          f"targets={cfg.get('target_modules')}")

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        base_id, dtype=torch.bfloat16, device_map={"": args.device})

    # One adapted projection and one that LoRA never touches, for the check below.
    probe_name = next(n for n, _ in model.named_parameters() if n.endswith("layers.0.mlp.gate_proj.weight"))
    control_name = next(n for n, _ in model.named_parameters() if "embed_tokens" in n)
    probe_before = dict(model.named_parameters())[probe_name].detach().clone()
    control_before = dict(model.named_parameters())[control_name].detach().clone()

    model = PeftModel.from_pretrained(model, str(adapter))
    model = model.merge_and_unload()
    print("merged.")

    params = dict(model.named_parameters())
    probe_after = params[probe_name].detach()
    control_after = params[control_name].detach()
    d_probe = (probe_after.float() - probe_before.float()).abs().max().item()
    d_control = (control_after.float() - control_before.float()).abs().max().item()
    print(f"  adapted   {probe_name}: max|delta| = {d_probe:.3e}")
    print(f"  unadapted {control_name}: max|delta| = {d_control:.3e}")
    if d_probe == 0.0:
        raise SystemExit("MERGE WAS A NO-OP -- adapted weights are unchanged; refusing to save.")
    if d_control != 0.0:
        raise SystemExit("Unadapted weights changed -- something other than the merge modified them.")

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), safe_serialization=True)
    AutoProcessor.from_pretrained(
        str(adapter) if (adapter / "tokenizer_config.json").exists() else base_id
    ).save_pretrained(str(out))

    # Keep provenance next to the weights; the merged dir loses adapter_config.json.
    shutil.copy2(adapter / "adapter_config.json", out / "source_adapter_config.json")
    (out / "MERGE_INFO.json").write_text(json.dumps(
        {"adapter_dir": str(adapter), "base_model": base_id,
         "lora_r": cfg.get("r"), "lora_alpha": cfg.get("lora_alpha")}, indent=2))

    shards = sorted(out.glob("*.safetensors"))
    size = sum(f.stat().st_size for f in shards) / 1e9
    print(f"\nwrote {len(shards)} shards, {size:.1f} GB -> {out}")
    print("load it with Qwen3_5ForConditionalGeneration.from_pretrained(<out>) -- no peft needed.")


if __name__ == "__main__":
    main()
