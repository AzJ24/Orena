import json

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoProcessor, Qwen3_5VisionConfig, Qwen3_5VisionModel

from .base import BaseEncoder

_VISUAL_PREFIX = "model.visual."


def _load_visual_state_dict(repo: str) -> dict:
    """Pull ONLY the vision-tower weights out of the full checkpoint shards.

    Qwen3.5 is a multimodal LLM; the vision encoder lives under `model.visual.`.
    We download the shards but keep only those ~300M-param tensors, so the 4B
    language model is never instantiated.
    """
    index = json.load(open(hf_hub_download(repo, "model.safetensors.index.json")))
    shards = sorted({s for k, s in index["weight_map"].items() if k.startswith(_VISUAL_PREFIX)})
    state = {}
    for shard in shards:
        sd = load_file(hf_hub_download(repo, shard))
        for k, v in sd.items():
            if k.startswith(_VISUAL_PREFIX):
                state[k[len(_VISUAL_PREFIX):]] = v
    return state


class QwenEncoder(BaseEncoder):
    """Vision tower of Qwen3.5 (a natively-multimodal LLM), used as an image encoder.

    The vision tower is a tubelet ViT-L; its image processor handles temporal
    patching internally, so a still image needs no manual frame replication. It
    emits a variable-length sequence of merged patch tokens (dim 2560, the same
    representation that feeds the LLM) with no CLS token — we mean-pool them into
    one fixed embedding per image.
    """

    name = "qwen"
    model_id = "Qwen/Qwen3.5-4B"

    def _load(self) -> None:
        cfg = Qwen3_5VisionConfig.from_pretrained(self.model_id)
        model = Qwen3_5VisionModel(cfg)
        missing, _ = model.load_state_dict(_load_visual_state_dict(self.model_id), strict=False)
        if missing:
            raise RuntimeError(f"qwen: missing vision weights: {missing[:5]}...")

        self.processor = AutoProcessor.from_pretrained(self.model_id).image_processor
        self.merge_sq = cfg.spatial_merge_size ** 2
        self.model = model.to(self.device, dtype=torch.float16)
        self.embed_dim = cfg.out_hidden_size  # 2560

    def preprocess(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        grid = inputs["image_grid_thw"]
        # merged tokens per image = (t * h * w) / spatial_merge_size**2
        tokens_per_image = [(int(t) * int(h) * int(w)) // self.merge_sq for t, h, w in grid.tolist()]
        return {
            "pixel_values": inputs["pixel_values"].to(self.device, dtype=torch.float16),
            "grid_thw": grid.to(self.device),
            "tokens_per_image": tokens_per_image,
        }

    def forward_features(self, batch):
        out = self.model(hidden_states=batch["pixel_values"], grid_thw=batch["grid_thw"])
        merged = out.pooler_output  # [sum(tokens_per_image), 2560], images concatenated
        per_image = torch.split(merged, batch["tokens_per_image"], dim=0)
        return torch.stack([t.mean(0) for t in per_image])  # mean-pool -> [B, 2560]
