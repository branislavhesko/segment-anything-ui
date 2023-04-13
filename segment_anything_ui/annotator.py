import dataclasses
import cv2
import numpy as np
from PySide6.QtWidgets import QWidget
from segment_anything import SamPredictor
import torch


@dataclasses.dataclass()
class Annotator:
    sam: torch.nn.Module | None = None
    embedding: torch.Tensor | None = None
    image: np.ndarray | None = None
    points: list[np.ndarray] = dataclasses.field(default_factory=list)
    mask: list[np.ndarray] = dataclasses.field(default_factory=list)
    labels: list[str] = dataclasses.field(default_factory=list)
    predictor: SamPredictor | None = None
    last_mask: np.ndarray | None = None
    parent: QWidget | None = None

    def set_image(self, image: np.ndarray):
        self.image = image
        return self

    def make_embedding(self):
        if self.sam is None:
            return
        self.predictor = SamPredictor(self.sam)
        self.predictor.set_image(self.image)

    # TODO: add box support
    def make_prediction(self):
        if self.predictor is None:
            # TODO: testing code
            mask = np.zeros_like(self.image)[..., 0]
            mask[50:100, 50:100] = 1.
            self.last_mask = mask
            return
        masks, scores, logits = self.predictor.predict(
            point_coords=self.points,
            point_labels=self.labels,
            multimask_output=False
        )
        mask = masks[0]
        self.last_mask = mask

    def visualize_last_mask(self):
        last_mask = np.zeros_like(self.image)
        last_mask[:, :, 1] = self.last_mask
        self.parent.update(cv2.addWeighted(self.image.copy(), 0.5, last_mask, 0.5, 0))

    def visualize_mask(self):
        visualization = np.zeros_like(self.image)
        visualization = cv2.applyColorMap(visualization, cv2.COLORMAP_JET)
        return visualization

    def merge_image_visualization(self):
        return cv2.addWeighted(self.image, 0.5, self.visualize_mask(), 0.5, 0)