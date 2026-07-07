import importlib.util

import torch
import torchvision.transforms.functional as TVF
from huggingface_hub import hf_hub_download
from torchvision.transforms import InterpolationMode

from .base import BaseEncoder

# CLIP normalization stats — the model card stresses these, NOT ImageNet stats.
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
IMG_SIZE = 224


def _load_module_py(repo: str):
    """Import the repo's own module.py (defines vit_large / VisionTransformer)."""
    path = hf_hub_download(repo, "module.py")
    spec = importlib.util.spec_from_file_location("levjepa_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class LevJepaEncoder(BaseEncoder):
    """LeJEPA ViT-L/16 video encoder, used on still images via 2-frame replication.

    This is a 3D (tubelet) ViT: its patch embedding consumes frames two at a
    time, so a single image is faked into a T=2 clip by replicating it once.
    We take the post-LayerNorm [CLS] token (1024-dim) as the image embedding.
    """

    name = "levjepa"
    model_id = "Machine-Learning-Oncology/levjepa-vitl-k710"

    def _load(self) -> None:
        vit_models = _load_module_py(self.model_id)
        ckpt_path = hf_hub_download(self.model_id, "epoch-0438.ckpt")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        geom = ckpt["hyper_parameters"]["model"]  # vit_large, img_size 224, num_frames 16, tubelet 2
        model = getattr(vit_models, geom["name"])(
            img_size=int(geom["img_size"]),
            patch_size=int(geom["patch_size"]),
            num_frames=int(geom["num_frames"]),
            tubelet_size=int(geom["tubelet_size"]),
        )

        # Keep only the encoder weights; drop projector/sigreg (SSL-objective only).
        enc_state = {
            k[len("encoder.") :]: v for k, v in ckpt["state_dict"].items() if k.startswith("encoder.")
        }
        missing, unexpected = model.load_state_dict(enc_state, strict=False)
        if missing:
            raise RuntimeError(f"levjepa: missing encoder weights: {missing[:5]}...")

        self.model = model.to(self.device)
        self.embed_dim = int(geom_embed_dim(geom))

    def preprocess(self, images):
        # Each PIL image -> (3, 224, 224) float, CLIP-normalized, then replicated
        # to T=2 and stacked into the (B, C, T, H, W) layout the 3D ViT expects.
        frames = []
        for im in images:
            x = TVF.pil_to_tensor(im).float().div(255.0)  # (3, H, W) in [0, 1]
            x = TVF.resize(x, [IMG_SIZE, IMG_SIZE], interpolation=InterpolationMode.BICUBIC, antialias=True)
            x = (x - CLIP_MEAN) / CLIP_STD
            x = x.unsqueeze(1).repeat(1, 2, 1, 1)  # (3, T=2, 224, 224)
            frames.append(x)
        batch = torch.stack(frames, dim=0)  # (B, 3, 2, 224, 224)
        return batch.to(self.device)

    def forward_features(self, batch):
        tokens = self.model(batch)  # (B, 1 + N_patches, 1024)
        return tokens[:, 0]  # [CLS] token


def geom_embed_dim(geom: dict) -> int:
    # vit_large is fixed at embed_dim 1024; keep a small map in case other sizes appear.
    return {"vit_base": 768, "vit_large": 1024, "vit_huge": 1280}.get(geom["name"], 1024)
