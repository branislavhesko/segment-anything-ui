import dataclasses
from typing import Callable
import uuid

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit
from segment_anything import SamPredictor
from segment_anything.build_sam import Sam
from segment_anything_ui.model_builder import (
    get_predictor, get_mask_generator, SamPredictor)
from segment_anything_ui.utils import bounding_boxes
try:
    from segment_anything_ui.model_builder import EfficientViTSamPredictor, EfficientViTSam
except (ImportError, ModuleNotFoundError):
    class EfficientViTSamPredictor:
        pass

    class EfficientViTSam:
        pass

from skimage.measure import regionprops
import torch

from segment_anything_ui.utils.shapes import BoundingBox
from segment_anything_ui.utils.bounding_boxes import get_bounding_boxes, get_mask_bounding_box


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    try:
        return plt.cm.get_cmap(name, n)
    except:
        return plt.get_cmap(name, n)


def crop_image(
        image,
        box: BoundingBox | None = None,
        image_shape: tuple[int, int] | None = None
):
    if image_shape is None:
        image_shape = image.shape[:2][::-1]
    if box is None:
        return cv2.resize(image, image_shape)

    if len(image.shape) == 2:
        return cv2.resize(image[box.ystart:box.yend, box.xstart:box.xend], image_shape)
    return cv2.resize(image[box.ystart:box.yend, box.xstart:box.xend, :], image_shape)


def insert_image(image, box: BoundingBox | None = None):
    new_image = np.zeros_like(image)
    if box is None:
        new_image = image
    else:
        if len(image.shape) == 2:
            new_image[box.ystart:box.yend, box.xstart:box.xend] = cv2.resize(
                image.astype(np.uint8), (int(box.xend) - int(box.xstart), int(box.yend) - int(box.ystart)))
        else:
            new_image[box.ystart:box.yend, box.xstart:box.xend, :] = cv2.resize(
                image.astype(np.uint8), (int(box.xend) - int(box.xstart), int(box.yend) - int(box.ystart)))
    return new_image


def find_closest_bounding_box(bounding_boxes: list[BoundingBox], point: np.ndarray):
    closest_bounding_box = None
    closest_bounding_box_id = -1
    min_distance = float('inf')
    for idx, bounding_box in enumerate(bounding_boxes):
        distance = bounding_box.distance_to(point)
        if distance < min_distance and bounding_box.contains(point):
            min_distance = distance
            closest_bounding_box = bounding_box
            closest_bounding_box_id = idx
    return closest_bounding_box, closest_bounding_box_id


@dataclasses.dataclass()
class AutomaticMaskGeneratorSettings:
    points_per_side: int = 32
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.95
    stability_score_offset: float = 1.0
    box_nms_thresh: float = 0.7
    crop_n_layers: int = 0
    crop_nms_thresh: float = 0.7


class LabelValueParam(QWidget):
    def __init__(self, label_text, default_value, value_type_converter: Callable = lambda x: x, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(self, text=label_text, alignment=Qt.AlignCenter)
        self.value = QLineEdit(self, text=default_value, alignment=Qt.AlignCenter)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.value)
        self.converter = value_type_converter

    def get_value(self):
        return self.converter(self.value.text())


class CustomForm(QWidget):

    def __init__(self, parent: QWidget, automatic_mask_generator_settings: AutomaticMaskGeneratorSettings) -> None:
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.widgets = []

        for field in dataclasses.fields(automatic_mask_generator_settings):
            widget = LabelValueParam(field.name, str(field.default), field.type)
            self.widgets.append(widget)
            self.layout.addWidget(widget)

    def get_values(self):
        return AutomaticMaskGeneratorSettings(**{widget.label.text(): widget.get_value() for widget in self.widgets})


@dataclasses.dataclass()
class Annotation:
    bounding_box: BoundingBox | None = None
    mask_uid: str | None = None
    mask_id: int | None = None
    label: str | None = None

    def __post_init__(self):
        if self.mask is not None:
            self.bounding_box = get_mask_bounding_box(self.mask)
            
            
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
    
    
class BoundingBoxAnnotation:
    def __init__(self) -> None:
        self.bounding_boxes: list[BoundingBox] = []
        self.bounding_box_id: int = -1

    def append(self, bounding_box: BoundingBox):
        self.bounding_boxes.append(bounding_box)
    
    def get_bounding_box(self, bounding_box_id: int):
        return self.bounding_boxes[bounding_box_id]
    
    def get_current_bounding_box(self):
        return self.bounding_boxes[self.bounding_box_id]
    
    def set_current_bounding_box(self, bounding_box: BoundingBox):
        self.bounding_boxes[self.bounding_box_id] = bounding_box

    def remove(self, mask_uid: str):
        bounding_box_id = next((idx for idx, bounding_box in enumerate(self.bounding_boxes) if bounding_box.mask_uid == mask_uid), None)
        if bounding_box_id is None:
            return
        bounding_box = self.bounding_boxes.pop(bounding_box_id)
        if self.bounding_box_id >= bounding_box_id:
            self.bounding_box_id -= 1
        return bounding_box
    
    def remove_by_id(self, bounding_box_id: int):
        mask_uid = self.bounding_boxes[bounding_box_id].mask_uid
        self.remove(mask_uid)
        return mask_uid

    def __len__(self):
        return len(self.bounding_boxes)

    
class MasksAnnotation:
    DEFAULT_LABEL = "default"

    def __init__(self) -> None:
        self.masks = []
        self.label_map = {}
        self.masks_uids: list[str] = []
        self.mask_id: int = -1

    def add_mask(self, mask, label: str | None = None):
        self.masks.append(mask)
        self.masks_uids.append(str(uuid.uuid4()))
        self.label_map[len(self.masks)] = self.DEFAULT_LABEL if label is None else label
        return self.masks_uids[-1]

    def add_label(self, mask_id: int, label: str):
        self.label_map[mask_id] = label

    def get_mask(self, mask_id: int):
        return self.masks[mask_id]

    def get_label(self, mask_id: int):
        return self.label_map[mask_id]

    def get_current_mask(self):
        return self.masks[self.mask_id]

    def set_current_mask(self, mask, label: str = None):
        self.masks[self.mask_id] = mask
        self.label_map[self.mask_id] = self.DEFAULT_LABEL if label is None else label

    def __getitem__(self, mask_id: int):
        return self.get_mask(mask_id)

    def __setitem__(self, mask_id: int, value):
        self.masks[mask_id] = value

    def __len__(self):
        return len(self.masks)

    def __iter__(self):
        return iter(zip(self.masks, self.label_map.values()))

    def __next__(self):
        if self.mask_id >= len(self.masks):
            raise StopIteration
        return self.masks[self.mask_id]

    def append(self, mask, label: str | None = None):
        return self.add_mask(mask, label)
    
    def pop_by_uuid(self, mask_uid: str):
        mask_id = next((idx for idx, m_uid in enumerate(self.masks_uids) if m_uid == mask_uid), None)
        if mask_id is None:
            return
        return self.pop(mask_id)

    def pop(self, mask_id: int = -1):
        _ = self.masks.pop(mask_id)
        mask_uid = self.masks_uids.pop(mask_id)
        self.label_map.pop(mask_id + 1)
        new_label_map = {}
        for index, value in enumerate(self.label_map.values()):
            new_label_map[index + 1] = value
        self.label_map = new_label_map
        return mask_uid

    @classmethod
    def from_masks(cls, masks, labels: list[str] | None = None):
        annotation = cls()
        if labels is None:
            labels = [None] * len(masks)
        for mask, label in zip(masks, labels):
            annotation.append(mask, label)
        return annotation


@dataclasses.dataclass()
class Annotator:
    sam: Sam | EfficientViTSam | None = None
    embedding: torch.Tensor | None = None
    image: np.ndarray | None = None
    annotations: Annotations = dataclasses.field(default_factory=Annotations)
    predictor: SamPredictor | EfficientViTSamPredictor | None = None
    visualization: np.ndarray | None = None
    last_mask: np.ndarray | None = None
    partial_mask: np.ndarray | None = None
    merged_mask: np.ndarray | None = None
    parent: QWidget | None = None
    cmap: plt.cm = None
    original_image: np.ndarray | None = None
    zoomed_bounding_box: BoundingBox | None = None

    def __post_init__(self):
        self.MAX_MASKS = 10
        self.cmap = get_cmap(self.MAX_MASKS)

    def set_image(self, image: np.ndarray):
        self.image = image
        return self

    def make_embedding(self):
        if self.sam is None:
            return
        self.predictor = get_predictor(self.sam)
        self.predictor.set_image(crop_image(self.image, self.zoomed_bounding_box))

    def predict_all(self, settings: AutomaticMaskGeneratorSettings):
        generator = get_mask_generator(
            sam=self.sam,
            **dataclasses.asdict(settings)
        )
        masks = generator.generate(self.image)
        masks = [(m["segmentation"] * 255).astype(np.uint8) for m in masks]
        label = self.parent.annotation_layout.label_picker.currentItem().text()
        self.annotations = Annotations.from_masks(masks, [label, ] * len(masks))
        self.cmap = get_cmap(len(self.annotations))

    def make_prediction(self, annotation: dict):
        masks, scores, logits = self.predictor.predict(
            point_coords=annotation["points"],
            point_labels=annotation["labels"],
            box=annotation["bounding_boxes"],
            multimask_output=False
        )
        mask = masks[0]
        self.last_mask = insert_image(mask, self.zoomed_bounding_box) * 255

    def pick_partial_mask(self):
        if self.partial_mask is None:
            self.partial_mask = self.last_mask.copy()
        else:
            self.partial_mask = np.maximum(self.last_mask, self.partial_mask)
        self.last_mask = None

    def move_current_mask_to_background(self):
        self.annotations.set_current_mask(self.annotations.get_current_mask() * 0.5)

    def merge_masks(self):
        new_mask = np.bitwise_or(self.last_mask, self.merged_mask)
        self.annotations.set_current_mask(new_mask, self.parent.annotation_layout.label_picker.currentItem().text())
        self.merged_mask = None

    def visualize_last_mask(self, label: str | None = None):
        last_mask = np.zeros_like(self.image)
        last_mask[:, :, 1] = self.last_mask
        if self.partial_mask is not None:
            last_mask[:, :, 0] = self.partial_mask
        if self.merged_mask is not None:
            last_mask[:, :, 2] = self.merged_mask
        if label is not None:
            props = regionprops(self.last_mask)[0]
            cv2.putText(
                last_mask,
                label,
                (int(props.centroid[1]), int(props.centroid[0])),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                [255, 255, 255],
                2
            )
        if self.is_show_bounding_boxes:
            last_mask_bounding_boxes = get_mask_bounding_box(last_mask[:, :, 1], label)
            cv2.rectangle(
                last_mask,
                (int(last_mask_bounding_boxes.x_min * self.image.shape[1]), int(last_mask_bounding_boxes.y_min * self.image.shape[0])),
                (int(last_mask_bounding_boxes.x_max * self.image.shape[1]), int(last_mask_bounding_boxes.y_max * self.image.shape[0])),
                (0, 255, 0),
                2
            )
            cv2.putText(
                last_mask,
                label,
                (int(last_mask_bounding_boxes.x_min * self.image.shape[1]), int(last_mask_bounding_boxes.y_min * self.image.shape[0])),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                [255, 255, 255],
                2
            )
        visualization = cv2.addWeighted(self.image.copy() if self.visualization is None else self.visualization.copy(),
                                        0.8, last_mask, 0.5, 0)
        self.parent.update(crop_image(visualization, self.zoomed_bounding_box))

    def _visualize_mask(self) -> tuple:
        mask_argmax = self.make_instance_mask()
        visualization = np.zeros_like(self.image)
        border = np.zeros(self.image.shape[:2], dtype=np.uint8)
        for i in range(1, np.amax(mask_argmax) + 1):
            color = self.cmap(i)
            single_mask = np.zeros_like(mask_argmax)
            single_mask[mask_argmax == i] = 1
            visualization[mask_argmax == i, :] = np.array(color[:3]) * 255
            border += single_mask - cv2.erode(
                single_mask, np.ones((3, 3), np.uint8), iterations=1)
            label = self.annotations.get_label_by_id(i)
            single_mask_center = np.mean(np.where(single_mask == 1), axis=1)
            if np.isnan(single_mask_center).any():
                continue
            if self.parent.settings.is_show_text():
                cv2.putText(
                    visualization,
                    label,
                    (int(single_mask_center[1]), int(single_mask_center[0])),
                    cv2.FONT_HERSHEY_PLAIN,
                    0.5,
                    [255, 255, 255],
                    1
                )
            if self.is_show_bounding_boxes:
                bounding_boxes = self.get_bounding_boxes()
                for idx, bounding_box in enumerate(bounding_boxes):
                    cv2.rectangle(
                        visualization,
                        (int(bounding_box.x_min * self.image.shape[1]), int(bounding_box.y_min * self.image.shape[0])),
                        (int(bounding_box.x_max * self.image.shape[1]), int(bounding_box.y_max * self.image.shape[0])),
                        (0, 0, 255) if idx != self.bounding_boxes.bounding_box_id else (0, 255, 0),
                        2
                    )
                    cv2.putText(
                        visualization,
                        bounding_box.label,
                        (int(bounding_box.x_min * self.image.shape[1]), int(bounding_box.y_min * self.image.shape[0])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        [255, 255, 255],
                        2
                    )
        border = (border == 0).astype(np.uint8)
        return visualization, border

    def has_annotations(self):
        return len(self.annotations) > 0

    def make_instance_mask(self):
        background = np.zeros_like(self.annotations.masks[0]) + 1
        mask_argmax = np.argmax(np.concatenate([np.expand_dims(background, 0), np.array(self.annotations.masks)], axis=0), axis=0).astype(np.uint8)
        return mask_argmax
    
    def get_bounding_boxes(self):
        return self.annotations.get_bounding_boxes()

    def merge_image_visualization(self):
        image = self.image.copy()
        if not len(self.annotations):
            return crop_image(image, self.zoomed_bounding_box)
        visualization, border = self._visualize_mask()
        self.visualization = cv2.addWeighted(image, 0.8, visualization, 0.7, 0) * border[:, :, np.newaxis]
        return crop_image(self.visualization, self.zoomed_bounding_box)

    def remove_last_mask(self):
        mask_id = len(self.annotations)
        self.annotations.remove_by_id(mask_id)

    def make_labels(self):
        return self.annotations.get_labels()

    def save_mask(self, label: str = Annotations.DEFAULT_LABEL):
        if self.partial_mask is not None:
            last_mask = self.partial_mask
            self.partial_mask = None
        else:
            last_mask = self.last_mask
        self.annotations.append(last_mask, label=label)
        if len(self.annotations) >= self.MAX_MASKS:
            self.MAX_MASKS += 10
            self.cmap = get_cmap(self.MAX_MASKS)
        
    @property
    def is_show_bounding_boxes(self):
        return self.parent.settings.is_show_bounding_boxes()

    def clear_last_masks(self):
        self.last_mask = None
        self.partial_mask = None
        self.visualization = None

    def clear(self):
        self.last_mask = None
        self.visualization = None
        self.masks = MasksAnnotation()
        self.bounding_boxes = BoundingBoxAnnotation()
        self.partial_mask = None
