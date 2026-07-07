from transformers import CLIPModel, CLIPProcessor

from .base import BaseEncoder


class ClipEncoder(BaseEncoder):
    name = "clip"
    model_id = "openai/clip-vit-base-patch32"

    def _load(self) -> None:
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        self.model = CLIPModel.from_pretrained(self.model_id).to(self.device)
        self.embed_dim = self.model.config.projection_dim

    def preprocess(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items()}

    def forward_features(self, batch):
        out = self.model.get_image_features(**batch)
        # transformers>=5 wraps this in a ModelOutput; pull out the plain tensor.
        return out[0] if isinstance(out, tuple) else getattr(out, "pooler_output", out)
