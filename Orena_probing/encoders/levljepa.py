from pathlib import Path

import timm
import torch
import torchvision.transforms.functional as TVF
from safetensors.torch import load_file
from torchvision.transforms import InterpolationMode

from .base import BaseEncoder

# LeVLJEPA's vision tower is a plain timm ViT-B/16 trained with ImageNet-style
# normalization (see the model card's transforms), NOT CLIP stats.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
IMG_SIZE = 224

# Local checkpoint shipped under models/LeVLJEPA/. Falls back to the HF hub if absent.
_LOCAL_DIR = Path(__file__).resolve().parents[1] / "models" / "LeVLJEPA"
_HF_REPO = "lukaskuhndkfz/LeVLJEPA-ViT-B-DataComp-200k"


def _vision_weights_path() -> str:
    local = _LOCAL_DIR / "vision_encoder.safetensors"
    if local.exists():
        return str(local)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(_HF_REPO, "vision_encoder.safetensors")


class LeVLJepaEncoder(BaseEncoder):
    """LeVLJEPA vision encoder: timm vit_base_patch16_224, DataComp-large pretrained.

    We probe the 768-dim ViT backbone feature (the model's pooled/[CLS] output),
    NOT the 256-dim cross-modal projection head — the projector is trained purely
    for image-text alignment and discards representation detail useful for probing.
    """

    name = "levljepa"

    def _load(self) -> None:
        model = timm.create_model(
            "vit_base_patch16_224", pretrained=False, num_classes=0, dynamic_img_size=True
        )

        weights = load_file(_vision_weights_path())
        # The safetensors bundle prefixes the backbone with "encoder."; the rest
        # ("pre_proj.", "projector.") is the SSL objective head we don't need.
        enc_state = {
            k[len("encoder.") :]: v for k, v in weights.items() if k.startswith("encoder.")
        }
        missing, _ = model.load_state_dict(enc_state, strict=False)
        if missing:
            raise RuntimeError(f"levljepa: missing encoder weights: {missing[:5]}...")

        self.model = model.to(self.device)
        self.embed_dim = model.num_features  # 768

    def preprocess(self, images):
        frames = []
        for im in images:
            x = TVF.pil_to_tensor(im).float().div(255.0)  # (3, H, W) in [0, 1]
            x = TVF.resize(
                x, [IMG_SIZE, IMG_SIZE], interpolation=InterpolationMode.BICUBIC, antialias=True
            )
            x = (x - IMAGENET_MEAN) / IMAGENET_STD
            frames.append(x)
        return torch.stack(frames, dim=0).to(self.device)  # (B, 3, 224, 224)

    def forward_features(self, batch):
        return self.model(batch)  # (B, 768) pooled feature
