import dataclasses
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PySide6.QtWidgets import QWidget
from segment_anything import SamPredictor
import torch


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


@dataclasses.dataclass()
class Annotator:
    sam: torch.nn.Module | None = None
    embedding: torch.Tensor | None = None
    image: np.ndarray | None = None
    mask: list[np.ndarray] = dataclasses.field(default_factory=list)
    predictor: SamPredictor | None = None
    visualization: np.ndarray | None = None
    last_mask: np.ndarray | None = None
    parent: QWidget | None = None

    def __post_init__(self):
        self.MAX_MASKS = 10
        self.cmap = get_cmap(self.MAX_MASKS)

    def set_image(self, image: np.ndarray):
        self.image = image
        return self

    def make_embedding(self):
        if self.sam is None:
            return
        self.predictor = SamPredictor(self.sam)
        self.predictor.set_image(self.image)

    # TODO: add box support
    def make_prediction(self, annotation: dict):
        if self.predictor is None:
            # TODO: testing code
            mask = np.zeros_like(self.image)[..., 0]
            mask[50:100, 50:100] = 255
            self.last_mask = mask
            return
        masks, scores, logits = self.predictor.predict(
            point_coords=annotation["points"],
            point_labels=annotation["labels"],
            box=annotation["bounding_boxes"],
            multimask_output=False
        )
        mask = masks[0]
        self.last_mask = mask * 255

    def visualize_last_mask(self):
        last_mask = np.zeros_like(self.image)
        last_mask[:, :, 1] = self.last_mask
        self.parent.update(cv2.addWeighted(self.image.copy() if self.visualization is None else self.visualization.copy(), 0.5, last_mask, 0.5, 0))

    def visualize_mask(self):
        mask_argmax = self.make_instance_mask()
        visualization = np.zeros_like(self.image)
        for i in range(1, np.amax(mask_argmax) + 1):
            color = self.cmap(i)
            visualization[mask_argmax == i, :] = np.array(color[:3]) * 255
        return visualization

    def make_instance_mask(self):
        background = np.zeros_like(self.mask[0]) + 1
        mask_argmax = np.argmax(np.concatenate([np.expand_dims(background, 0), np.array(self.mask)], axis=0), axis=0).astype(np.uint8)
        return mask_argmax

    def merge_image_visualization(self):
        self.visualization = cv2.addWeighted(self.image, 0.5, self.visualize_mask(), 0.5, 0)
        return self.visualization

    def remove_last_mask(self):
        self.mask.pop()

    def save_mask(self):
        self.mask.append(self.last_mask)
        if len(self.mask) >= self.MAX_MASKS:
            self.MAX_MASKS += 10
            self.cmap = get_cmap(self.MAX_MASKS)

    def clear(self):
        self.last_mask = None
        self.mask = []
