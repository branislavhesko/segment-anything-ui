import json
import os
from typing import Callable
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QListWidget

from segment_anything_ui.draw_label import PaintType
from segment_anything_ui.annotator import AutomaticMaskGeneratorSettings, CustomForm, MasksAnnotation


class AnnotationLayout(QWidget):

    def __init__(self, parent, config) -> None:
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        labels = self._load_labels(config)
        self.layout.setAlignment(Qt.AlignTop)
        self.add_point = QPushButton("Add Point")
        self.add_box = QPushButton("Add Box")
        self.annotate_all = QPushButton("Annotate All")
        self.manual_polygon = QPushButton("Manual Polygon")
        self.cancel_annotation = QPushButton("Cancel Annotation")
        self.save_annotation = QPushButton("Save Annotation")
        self.pick_mask = QPushButton("Pick Mask")
        self.label_picker = QListWidget()
        self.label_picker.addItems(labels)
        self.label_picker.setCurrentRow(0)
        self.move_current_mask_background = QPushButton("Move Current Mask to Front")
        self.remove_hidden_masks = QPushButton("Remove Hidden Masks")
        self.remove_hidden_masks_label = QLabel("Remove Hidden Masks with hidden area less than")
        self.remove_hidden_masks_line = QLineEdit("0.5")
        self.save_annotation.setShortcut("N")

        self.annotation_settings = CustomForm(self, AutomaticMaskGeneratorSettings())
        for w in [
                self.add_point,
                self.add_box,
                self.annotate_all,
                self.pick_mask,
                self.move_current_mask_background,
                self.cancel_annotation,
                self.save_annotation,
                self.manual_polygon,
                self.label_picker,
                self.annotation_settings,
                self.remove_hidden_masks,
                self.remove_hidden_masks_label,
                self.remove_hidden_masks_line
            ]:
            self.layout.addWidget(w)
        self.add_point.clicked.connect(self.on_add_point)
        self.add_box.clicked.connect(self.on_add_box)
        self.annotate_all.clicked.connect(self.on_annotate_all)
        self.cancel_annotation.clicked.connect(self.on_cancel_annotation)
        self.save_annotation.clicked.connect(self.on_save_annotation)
        self.pick_mask.clicked.connect(self.on_pick_mask)
        self.manual_polygon.clicked.connect(self.on_manual_polygon)
        self.remove_hidden_masks.clicked.connect(self.on_remove_hidden_masks)
        self.move_current_mask_background.clicked.connect(self.on_move_current_mask_background_fn)

    @staticmethod
    def _load_labels(config):
        if not os.path.exists(config.label_file):
            return ["default"]
        with open(config.label_file, "r") as f:
            labels = json.load(f)
        MasksAnnotation.DEFAULT_LABEL = list(labels.keys())[0] if len(labels) > 0 else "default"
        return labels

    def on_move_current_mask_background_fn(self):
        self.parent().annotator.move_current_mask_to_background()
        self.parent().update(self.parent().annotator.merge_image_visualization())

    def on_remove_hidden_masks(self):
        annotations = self.parent().annotator.masks
        argmax_mask = self.parent().annotator.make_instance_mask()
        limit_ratio = float(self.remove_hidden_masks_line.text())
        new_masks = []
        new_labels = []
        for idx, (mask, label) in enumerate(annotations):
            num_pixels = np.sum(mask > 0)
            num_visible = np.sum(argmax_mask == (idx + 1))
            ratio = num_visible / num_pixels

            if ratio > limit_ratio:
                new_masks.append(mask)
                new_labels.append(label)

        print("Removed ", len(annotations) - len(new_masks), " masks.")
        self.parent().annotator.masks = MasksAnnotation.from_masks(new_masks, new_labels)
        self.parent().update(self.parent().annotator.merge_image_visualization())

    def on_pick_mask(self):
        self.parent().image_label.change_paint_type(PaintType.MASK_PICKER)

    def on_manual_polygon(self):
        # Sets emphasis on the button
        self.manual_polygon.setProperty("active", True)
        self.parent().image_label.change_paint_type(PaintType.POLYGON)

    def on_add_point(self):
        self.parent().image_label.change_paint_type(PaintType.POINT)

    def on_add_box(self):
        self.parent().image_label.change_paint_type(PaintType.BOX)

    def on_annotate_all(self):
        self.parent().annotator.predict_all(self.annotation_settings.get_values())
        self.parent().update(self.parent().annotator.merge_image_visualization())

    def on_cancel_annotation(self):
        self.parent().image_label.clear()
        self.parent().update(self.parent().annotator.merge_image_visualization())

    def on_save_annotation(self):
        self.parent().annotator.save_mask()
        self.parent().update(self.parent().annotator.merge_image_visualization())
        self.parent().image_label.clear()
