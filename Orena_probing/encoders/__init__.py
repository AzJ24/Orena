from .base import BaseEncoder
from .biomedClip import BiomedClipEncoder
from .clip import ClipEncoder
from .dino import DinoEncoder
from .dinov3 import Dinov3Encoder
from .gemma import GemmaEncoder
from .levjepa import LevJepaEncoder
from .levljepa import LeVLJepaEncoder
from .qwen import QwenEncoder

ENCODER_REGISTRY = {
    "clip": ClipEncoder,
    "dino": DinoEncoder,
    "biomedclip": BiomedClipEncoder,
    "dinov3": Dinov3Encoder,
    "levjepa": LevJepaEncoder,
    "levljepa": LeVLJepaEncoder,
    "qwen": QwenEncoder,
    "gemma": GemmaEncoder,
}


def build_encoder(name: str, device: str) -> BaseEncoder:
    if name not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown encoder '{name}'. Options: {list(ENCODER_REGISTRY)}")
    return ENCODER_REGISTRY[name](device=device)
