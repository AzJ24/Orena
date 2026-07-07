from transformers import AutoImageProcessor, AutoModel

from .base import BaseEncoder


class DinoEncoder(BaseEncoder):
    name = "dino"
    model_id = "facebook/dinov2-base"

    def _load(self) -> None:
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device)
        self.embed_dim = self.model.config.hidden_size

    def preprocess(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items()}

    def forward_features(self, batch):
        # CLS token of the last hidden state, the standard DINOv2 probing feature.
        out = self.model(**batch)
        return out.last_hidden_state[:, 0, :]
