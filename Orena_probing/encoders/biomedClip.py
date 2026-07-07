import open_clip
import torch

from .base import BaseEncoder


class BiomedClipEncoder(BaseEncoder):
    name = "biomedclip"
    model_id = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

    def _load(self) -> None:
        model, _, preprocess = open_clip.create_model_and_transforms(self.model_id)
        self.model = model.to(self.device)
        self.transform = preprocess
        with torch.no_grad():
            dummy = torch.zeros(1, 3, *self.model.visual.image_size).to(self.device)
            self.embed_dim = self.model.encode_image(dummy).shape[-1]

    def preprocess(self, images):
        batch = torch.stack([self.transform(im) for im in images])
        return batch.to(self.device)

    def forward_features(self, batch):
        return self.model.encode_image(batch)
