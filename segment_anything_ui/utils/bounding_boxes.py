import dataclasses
import numpy as np


@dataclasses.dataclass
class BoundingBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    label: str
    mask_uid: str = ""
    
    def to_dict(self):
        return {
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
            "label": self.label,
            "mask_uid": self.mask_uid
        }
    
    @property
    def center(self):
        return np.array([(self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2])

    def distance_to(self, point: np.ndarray):
        return np.linalg.norm(self.center - point)

    def contains(self, point: np.ndarray):
        return self.x_min <= point[0] <= self.x_max and self.y_min <= point[1] <= self.y_max


def get_mask_bounding_box(mask, label: str):
    where = np.where(mask)
    x_min = np.min(where[1])
    y_min = np.min(where[0])
    x_max = np.max(where[1])
    y_max = np.max(where[0])
    return BoundingBox(
        x_min / mask.shape[1], 
        y_min / mask.shape[0], 
        x_max / mask.shape[1], 
        y_max / mask.shape[0], 
        label
    )


def get_bounding_boxes(masks, labels):
    bounding_boxes = []
    for mask, label in zip(masks, labels):
        bounding_box = get_mask_bounding_box(mask, label)
        bounding_boxes.append(bounding_box)
    return bounding_boxes
