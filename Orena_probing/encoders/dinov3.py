from transformers import AutoImageProcessor, AutoModel

from .base import BaseEncoder


class Dinov3Encoder(BaseEncoder):
    """Gated on HF — request access at the model_id URL below before this will load."""

    name = "dinov3"
    model_id = "facebook/dinov3-vitb16-pretrain-lvd1689m"

    def _load(self) -> None:
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device)
        self.embed_dim = self.model.config.hidden_size

    def preprocess(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items()}

    def forward_features(self, batch):
        # CLS token of the last hidden state, same probing feature as DINOv2.
        out = self.model(**batch)
        return out.last_hidden_state[:, 0, :]
