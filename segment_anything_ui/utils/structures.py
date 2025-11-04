import dataclasses
from enum import Enum
import uuid

import cv2
import numpy as np
from PySide6.QtCore import QPoint
from PySide6.QtGui import QPolygon


class AnnotationMode(Enum):
    MASK = "mask"
    BOUNDING_BOX = "bounding_box"


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


@dataclasses.dataclass()
class Annotation:
    bounding_box: BoundingBox | None = None
    mask_uid: str | None = None
    mask_id: int | None = None
    label: str | None = None
            
            
class Annotations:
    """
    Annotations is a class that contains a list of masks and their corresponding bounding boxes.
    """
    DEFAULT_LABEL = "default"
    
    def __init__(self) -> None:
        self.annotations: list[Annotation] = []
        self.current_annotation_id: int = -1
        self.masks = []

    def append(self, mask, label: str | None = None):
        label = self.DEFAULT_LABEL if label is None else label
        annotation = Annotation(
            bounding_box=get_mask_bounding_box(mask, label),
            mask_uid=str(uuid.uuid4()),
            mask_id=len(self.masks),
            label=label
        )
        self.masks.append(mask)
        self.annotations.append(annotation)
        return annotation
    
    def get_mask_by_uid(self, mask_uid: str):
        mask_uuids = [annotation.mask_uid for annotation in self.annotations]
        mask_id = mask_uuids.index(mask_uid)
        return self.masks[mask_id]
    
    def get_label_by_id(self, mask_id: int):
        return self.annotations[mask_id].label
    
    def get_mask_by_id(self, mask_id: int):
        return self.masks[mask_id]
    
    def set_current_mask(self, mask: np.ndarray, label: str | None = None):
        mask_id = self.current_annotation_id
        if label is not None:
            self.annotations[mask_id].label = label
        self.masks[mask_id] = mask
        
    def get_current_mask(self):
        mask_id = self.current_annotation_id
        return self.masks[mask_id]
    
    def get_current_annotation(self):
        return self.annotations[self.current_annotation_id]
    
    def get_bounding_boxes(self):
        return [annotation.bounding_box for annotation in self.annotations]
    
    def remove_by_uid(self, mask_uid: str):
        mask_uids = [annotation.mask_uid for annotation in self.annotations]
        mask_id = mask_uids.index(mask_uid)
        self.remove_by_id(mask_id)
        return mask_uid
    
    def remove_by_id(self, mask_id: int):
        self.masks.pop(mask_id)
        self.annotations.pop(mask_id)
        return mask_id
    
    @classmethod
    def from_masks(cls, masks, labels: list[str] | None = None):
        annotations = cls()
        if labels is None:
            labels = [cls.DEFAULT_LABEL] * len(masks)
        for mask, label in zip(masks, labels):
            annotations.append(mask, label)
        return annotations

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, mask_uid: str):
        return self.get_mask_by_uid(mask_uid)
    
    def __setitem__(self, mask_uid: str, value):
        self.annotations[mask_uid] = value
        
    def __iter__(self):
        return iter(zip(self.masks, self.annotations))

    def __next__(self):
        if self.current_annotation_id >= len(self.annotations):
            raise StopIteration
        return self.annotations[self.current_annotation_id]
    

@dataclasses.dataclass
class DrawnBoundingBox:
    xstart: float | int
    ystart: float | int
    xend: float | int = -1.
    yend: float | int = -1.

    def to_numpy(self):
        return np.array([self.xstart, self.ystart, self.xend, self.yend])

    def scale(self, sx, sy):
        return DrawnBoundingBox(
            xstart=self.xstart * sx,
            ystart=self.ystart * sy,
            xend=self.xend * sx,
            yend=self.yend * sy
        )

    def to_int(self):
        return DrawnBoundingBox(
            xstart=int(self.xstart),
            ystart=int(self.ystart),
            xend=int(self.xend),
            yend=int(self.yend)
        )

@dataclasses.dataclass
class Polygon:
    points: list = dataclasses.field(default_factory=list)

    def to_numpy(self):
        return np.array(self.points).reshape(-1, 2)

    def to_mask(self, num_rows, num_cols):
        mask = np.zeros((num_rows, num_cols))
        mask = cv2.fillPoly(mask, pts=[self.to_numpy(), ], color=255)
        return mask

    def is_plotable(self):
        return len(self.points) > 3

    def to_qpolygon(self):
        return QPolygon([
            QPoint(x, y) for x, y in self.points
        ])
