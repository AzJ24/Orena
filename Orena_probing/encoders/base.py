from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
from PIL import Image


class BaseEncoder(ABC):
    """A frozen pretrained vision encoder used purely for feature extraction.

    Subclasses load a pretrained model in `_load()`, expose `embed_dim`, and
    implement `preprocess()` / `forward_features()`. No gradients ever flow
    through the encoder (Step 1 of the linear-probing methodology).
    """

    name: str

    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.embed_dim: int = -1
        self._load()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    @abstractmethod
    def _load(self) -> None:
        """Set self.model (and any processor/transform) and self.embed_dim."""

    @abstractmethod
    def preprocess(self, images: list[Image.Image]):
        """Turn a list of PIL images into whatever forward_features expects."""

    @abstractmethod
    def forward_features(self, batch) -> torch.Tensor:
        """Run the frozen encoder on a preprocessed batch, return [B, embed_dim]."""

    @torch.no_grad()
    def encode(self, images: list[Image.Image], batch_size: int = 64) -> np.ndarray:
        """image -> pretrained encoder -> fixed embedding vector, batched."""
        all_feats = []
        for i in range(0, len(images), batch_size):
            chunk = [im.convert("RGB") for im in images[i : i + batch_size]]
            batch = self.preprocess(chunk)
            feats = self.forward_features(batch)
            all_feats.append(feats.float().cpu().numpy())
        return np.concatenate(all_feats, axis=0)
