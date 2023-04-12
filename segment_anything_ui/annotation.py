import dataclasses
import numpy as np
from segment_anything import SamPredictor
import torch


@dataclasses.dataclass()
class Annotation:
    image: np.ndarray
    embedding: torch.Tensor
    points: list[np.ndarray]
    mask: list[np.ndarray]
    labels: list[str]
    predictor: SamPredictor | None = None

    def __post_init__(self):
        self.make_embedding()

    def make_visualization(self):
        # do something with self.image, self.mask, self.labels
        pass

    def make_embedding(self, sam):
        self.predictor = SamPredictor(sam)
        self.predictor.set_image(self.image)

    # TODO: add box support
    def make_prediction(self):
        masks, scores, logits = self.predictor.predict(
            point_coords=self.points,
            point_labels=self.labels,
            multimask_output=False
        )
        return masks, scores, logits