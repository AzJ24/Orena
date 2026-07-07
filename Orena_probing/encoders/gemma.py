import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoProcessor, Gemma4VisionConfig, Gemma4VisionModel

from .base import BaseEncoder

_VISION_PREFIX = "model.vision_tower."


def _load_vision_state_dict(repo: str) -> dict:
    """Pull ONLY the SigLIP-style vision-tower weights out of the full checkpoint.

    Gemma 4 E2B is a multimodal LLM in a single model.safetensors; the vision
    tower lives under `model.vision_tower.`. We memory-map the file and keep only
    those ~90M-param tensors, so the language model is never instantiated.
    """
    path = hf_hub_download(repo, "model.safetensors")
    state = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            if k.startswith(_VISION_PREFIX):
                state[k[len(_VISION_PREFIX):]] = f.get_tensor(k)
    return state


class GemmaEncoder(BaseEncoder):
    """Vision tower of Gemma 4 E2B (a multimodal LLM), used as an image encoder.

    The vision tower is a SigLIP-style ViT-B (768-dim, 16 layers, patch-16). Its
    processor uses Pan & Scan tiling and emits a variable number of merged "soft
    tokens" per image (no CLS token), which we mean-pool into one 768-dim
    embedding per image.
    """

    name = "gemma"
    model_id = "google/gemma-4-E2B-it"

    def _load(self) -> None:
        cfg = Gemma4VisionConfig.from_pretrained(self.model_id)
        model = Gemma4VisionModel(cfg)
        missing, _ = model.load_state_dict(_load_vision_state_dict(self.model_id), strict=False)
        if missing:
            raise RuntimeError(f"gemma: missing vision weights: {missing[:5]}...")

        self.processor = AutoProcessor.from_pretrained(self.model_id).image_processor
        self.model = model.to(self.device, dtype=torch.bfloat16)
        self.embed_dim = cfg.hidden_size  # 768

    def preprocess(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        return {
            "pixel_values": inputs["pixel_values"].to(self.device, dtype=torch.bfloat16),
            "pixel_position_ids": inputs["image_position_ids"].to(self.device),
            "tokens_per_image": inputs["num_soft_tokens_per_image"].tolist(),
        }

    def forward_features(self, batch):
        out = self.model(
            pixel_values=batch["pixel_values"],
            pixel_position_ids=batch["pixel_position_ids"],
        )
        tokens = out.last_hidden_state  # [sum(tokens_per_image), 768], images concatenated
        per_image = torch.split(tokens, batch["tokens_per_image"], dim=0)
        return torch.stack([t.float().mean(0) for t in per_image])  # mean-pool -> [B, 768]
