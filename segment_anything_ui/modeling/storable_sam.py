from safetensors import safe_open
from segment_anything.modeling import Sam
import torch.nn as nn


class ModifiedImageEncoder(nn.Module):

    def __init__(self, image_encoder, saved_path: str | None = None):
        super().__init__()
        self.image_encoder = image_encoder
        if saved_path is not None:
            self.embeddings = safe_open(saved_path)
        else:
            self.embeddings = None

    def forward(self, x):
        return self.image_encoder(x) if self.embeddings is None else self.embeddings


class StorableSam(Sam):

    def transform(self, saved_path):
        self.image_encoder = ModifiedImageEncoder(self.image_encoder, saved_path)

    def precompute(self, image):
        return self.image_encoder(image)
