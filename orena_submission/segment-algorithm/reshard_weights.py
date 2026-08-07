"""Re-shards the merged checkpoint into layer-sized safetensors shards.

The merged 27B checkpoint ships as two shards, the larger being 49.83 GB. A COPY
instruction produces exactly one image layer, and ECR rejects any layer over 50 GB
-- so copying that shard would clear the limit by 0.35%, with a failed 55 GB upload
as the cost of being wrong. Re-sharding to `--max-shard-gb` leaves every layer an
order of magnitude clear of the ceiling.

Tensors are streamed one at a time through `safetensors.safe_open`, so this needs a
few GB of RAM rather than the 55 GB the model would take to load.

Usage:
    venv3.12/bin/python orena_submission/segment-algorithm/reshard_weights.py \
        --src segment_track/checkpoints/segment-27b-alldata-n80-650-20260730-merged \
        --dst orena_submission/segment-algorithm/resources/segment_model
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

# Everything a from_pretrained() needs besides the weights themselves.
SIDECAR_FILES = (
    "config.json", "generation_config.json", "chat_template.jinja",
    "processor_config.json", "tokenizer_config.json", "tokenizer.json",
    "preprocessor_config.json", "video_preprocessor_config.json",
    "special_tokens_map.json", "vocab.json", "merges.txt",
)


def tensor_sources(src: Path) -> dict[str, Path]:
    """`{tensor name: file holding it}` for sharded or single-file checkpoints."""
    index = src / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        return {name: src / file for name, file in weight_map.items()}
    single = src / "model.safetensors"
    if not single.exists():
        raise SystemExit(f"no safetensors weights found in {src}")
    with safe_open(single, framework="pt") as f:
        return {name: single for name in f.keys()}


def plan_shards(sources: dict[str, Path], max_bytes: int) -> list[list[str]]:
    """Group tensors into shards, each at most `max_bytes` (a lone larger tensor
    still gets its own shard -- a tensor cannot be split)."""
    sizes: dict[str, int] = {}
    for file in sorted(set(sources.values())):
        with safe_open(file, framework="pt") as f:
            for name in f.keys():
                if sources.get(name) == file:
                    slice_ = f.get_slice(name)
                    n = 1
                    for d in slice_.get_shape():
                        n *= d
                    # bf16/fp16 = 2 bytes; ask the slice rather than assume.
                    sizes[name] = n * _dtype_bytes(slice_.get_dtype())

    shards: list[list[str]] = [[]]
    current = 0
    # Keep the checkpoint's own ordering so related tensors stay together.
    for name in sources:
        size = sizes[name]
        if shards[-1] and current + size > max_bytes:
            shards.append([])
            current = 0
        shards[-1].append(name)
        current += size
    return [s for s in shards if s]


def _dtype_bytes(dtype: str) -> int:
    return {
        "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
        "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
        "I32": 4, "U32": 4, "F32": 4,
        "I64": 8, "U64": 8, "F64": 8,
    }[dtype]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--max-shard-gb", type=float, default=8.0,
                    help="decimal GB per shard; well under the 50 GB layer limit")
    args = ap.parse_args()

    src, dst = args.src.resolve(), args.dst.resolve()
    if not src.is_dir():
        raise SystemExit(f"source not a directory: {src}")
    dst.mkdir(parents=True, exist_ok=True)

    sources = tensor_sources(src)
    shards = plan_shards(sources, int(args.max_shard_gb * 1e9))
    total = len(shards)
    print(f"{len(sources)} tensors -> {total} shard(s) of at most {args.max_shard_gb} GB")

    weight_map: dict[str, str] = {}
    total_bytes = 0
    for i, names in enumerate(shards, start=1):
        out_name = f"model-{i:05d}-of-{total:05d}.safetensors"
        tensors = {}
        # Reopen per source file; safe_open holds an mmap, so this stays cheap.
        for name in names:
            with safe_open(sources[name], framework="pt") as f:
                tensors[name] = f.get_tensor(name)
        save_file(tensors, dst / out_name, metadata={"format": "pt"})
        del tensors
        written = (dst / out_name).stat().st_size
        total_bytes += written
        weight_map.update({name: out_name for name in names})
        print(f"  [{i}/{total}] {out_name}  {written / 1e9:.2f} GB  ({len(names)} tensors)")

    (dst / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total_bytes}, "weight_map": weight_map}, indent=2))

    for name in SIDECAR_FILES:
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
            print(f"  copied {name}")

    largest = max(p.stat().st_size for p in dst.glob("*.safetensors"))
    print(f"\ntotal {total_bytes / 1e9:.2f} GB, largest shard {largest / 1e9:.2f} GB "
          f"({'OK' if largest < 50e9 else 'OVER'} vs the 50 GB layer limit)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
