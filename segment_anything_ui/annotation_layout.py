import enum
import json
import os
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QListWidget, QMessageBox

from segment_anything_ui.draw_label import PaintType
from segment_anything_ui.annotator import AutomaticMaskGeneratorSettings, CustomForm
from segment_anything_ui.utils.structures import Annotations, BoundingBox


class MergeState(enum.Enum):
    PICKING = enum.auto()
    MERGING = enum.auto()


class AnnotationLayout(QWidget):

    def __init__(self, parent, config) -> None:
        super().__init__(parent)
        self.config = config
        self.zoom_flag = False
        self.merge_state = MergeState.PICKING
        self.layout = QVBoxLayout(self)
        labels = self._load_labels(config)
        self.layout.setAlignment(Qt.AlignTop)
        self.add_point = QPushButton(f"Add Point [ {config.key_mapping.ADD_POINT.name} ]")
        self.add_box = QPushButton(f"Add Box [ {config.key_mapping.ADD_BOX.name} ]")
        self.annotate_all = QPushButton(f"Annotate All [ {config.key_mapping.ANNOTATE_ALL.name} ]")
        self.manual_polygon = QPushButton(f"Manual Polygon [ {config.key_mapping.MANUAL_POLYGON.name} ]")
        self.manual_bounding_box = QPushButton(f"Manual Bounding Box [ {config.key_mapping.MANUAL_BOUNDING_BOX.name} ]")
        self.cancel_annotation = QPushButton(f"Cancel Annotation [ {config.key_mapping.CANCEL_ANNOTATION.name} ]")
        self.save_annotation = QPushButton(f"Save Annotation [ {config.key_mapping.SAVE_ANNOTATION.name} ]")
        self.pick_mask = QPushButton(f"Pick Mask [ {config.key_mapping.PICK_MASK.name} ]")
        self.pick_bounding_box = QPushButton(f"Pick Bounding Box [ {config.key_mapping.PICK_BOUNDING_BOX.name} ]")
        self.merge_masks = QPushButton(f"Merge Masks [ {config.key_mapping.MERGE_MASK.name} ]")
        self.delete_mask = QPushButton(f"Delete Mask [ {config.key_mapping.DELETE_MASK.name} ]")
        self.partial_annotation = QPushButton(f"Partial Annotation [ {config.key_mapping.PARTIAL_ANNOTATION.name} ]")
        self.zoom_rectangle = QPushButton(f"Zoom Rectangle [ {config.key_mapping.ZOOM_RECTANGLE.name} ]")
        self.label_picker = QListWidget()
        self.label_picker.addItems(labels)
        self.label_picker.setCurrentRow(0)
        self.move_current_mask_background = QPushButton("Move Current Mask to Front")
        self.remove_hidden_masks = QPushButton("Remove Hidden Masks")
        self.remove_hidden_masks_label = QLabel("Remove Hidden Masks with hidden area less than")
        self.remove_hidden_masks_line = QLineEdit("0.5")
        self.save_annotation.setShortcut(config.key_mapping.SAVE_ANNOTATION.key)
        self.add_point.setShortcut(config.key_mapping.ADD_POINT.key)
        self.add_box.setShortcut(config.key_mapping.ADD_BOX.key)
        self.annotate_all.setShortcut(config.key_mapping.ANNOTATE_ALL.key)
        self.cancel_annotation.setShortcut(config.key_mapping.CANCEL_ANNOTATION.key)
        self.pick_mask.setShortcut(config.key_mapping.PICK_MASK.key)
        self.pick_bounding_box.setShortcut(config.key_mapping.PICK_BOUNDING_BOX.key)
        self.partial_annotation.setShortcut(config.key_mapping.PARTIAL_ANNOTATION.key)
        self.delete_mask.setShortcut(config.key_mapping.DELETE_MASK.key)
        self.zoom_rectangle.setShortcut(config.key_mapping.ZOOM_RECTANGLE.key)
        self.manual_bounding_box.setShortcut(config.key_mapping.MANUAL_BOUNDING_BOX.key)

        self.annotation_settings = CustomForm(self, AutomaticMaskGeneratorSettings())
        for w in [
                self.add_point,
                self.add_box,
                self.annotate_all,
                self.pick_mask,
                self.pick_bounding_box,
                self.merge_masks,
                self.move_current_mask_background,
                self.cancel_annotation,
                self.delete_mask,
                self.partial_annotation,
                self.save_annotation,
                self.manual_polygon,
                self.manual_bounding_box,
                self.label_picker,
                self.zoom_rectangle,
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
        self.pick_bounding_box.clicked.connect(self.on_pick_bounding_box)
        self.manual_polygon.clicked.connect(self.on_manual_polygon)
        self.remove_hidden_masks.clicked.connect(self.on_remove_hidden_masks)
        self.move_current_mask_background.clicked.connect(self.on_move_current_mask_background_fn)
        self.merge_masks.clicked.connect(self.on_merge_masks)
        self.partial_annotation.clicked.connect(self.on_partial_annotation)
        self.delete_mask.clicked.connect(self.on_delete_mask)
        self.zoom_rectangle.clicked.connect(self.on_zoom_rectangle)
        self.manual_bounding_box.clicked.connect(self.on_manual_bounding_box)

    def on_delete_mask(self):
        if self.parent().image_label.paint_type == PaintType.MASK_PICKER:
            self.parent().info_label.setText("Deleting mask!")
            self.parent().annotator.annotations.remove_by_id(self.parent().annotator.annotations.current_annotation_id)
            self.parent().annotator.annotations.current_annotation_id = -1
            self.parent().annotator.last_mask = None
            self.parent().update(self.parent().annotator.merge_image_visualization())
        elif self.parent().image_label.paint_type == PaintType.BOX_PICKER:
            self.parent().info_label.setText("Deleting bounding box!")
            self.parent().annotator.annotations.remove_by_id(self.parent().annotator.annotations.current_annotation_id)
            self.parent().annotator.last_mask = None
            self.parent().annotator.annotations.current_annotation_id = -1
            self.parent().update(self.parent().annotator.merge_image_visualization())
        else:
            QMessageBox.warning(self, "Error", "Please pick a mask or bounding box to delete!")

    def on_partial_annotation(self):
        self.parent().info_label.setText("Partial annotation!")
        self.parent().annotator.pick_partial_mask()
        self.parent().image_label.clear()

    @staticmethod
    def _load_labels(config):
        if not os.path.exists(config.label_file):
            return ["default"]
        with open(config.label_file, "r") as f:
            labels = json.load(f)
        Annotations.DEFAULT_LABEL = list(labels.keys())[0] if len(labels) > 0 else "default"
        return labels

    def on_merge_masks(self):
        self.parent().image_label.change_paint_type(PaintType.MASK_PICKER)
        if self.merge_state == MergeState.PICKING:
            self.parent().info_label.setText("Pick a mask to merge with!")
            self.merge_state = MergeState.MERGING
            self.parent().annotator.merged_mask = self.parent().annotator.last_mask.copy()
        elif self.merge_state == MergeState.MERGING:
            self.parent().info_label.setText("Merging masks!")
            self.parent().annotator.merge_masks()
            self.merge_state = MergeState.PICKING

    def on_move_current_mask_background_fn(self):
        self.parent().info_label.setText("Moving current mask to background!")
        self.parent().annotator.move_current_mask_to_background()
        self.parent().update(self.parent().annotator.merge_image_visualization())

    def on_remove_hidden_masks(self):
        self.parent().info_label.setText("Removing hidden masks!")
        annotations = self.parent().annotator.annotations
        argmax_mask = self.parent().annotator.make_instance_mask()
        limit_ratio = float(self.remove_hidden_masks_line.text())
        new_masks = []
        new_labels = []
        for idx, (mask, annotation) in enumerate(annotations):
            num_pixels = np.sum(mask > 0)
            num_visible = np.sum(argmax_mask == (idx + 1))
            ratio = num_visible / num_pixels

            if ratio > limit_ratio:
                new_masks.append(mask)
                new_labels.append(annotation.label)

        print("Removed ", len(annotations) - len(new_masks), " masks.")
        self.parent().annotator.annotations = Annotations.from_masks(new_masks, new_labels)
        self.parent().update(self.parent().annotator.merge_image_visualization())

    def on_pick_mask(self):
        self.parent().info_label.setText("Pick a mask to do modifications!")
        self.parent().image_label.change_paint_type(PaintType.MASK_PICKER)
        
    def on_pick_bounding_box(self):
        self.parent().info_label.setText("Pick a bounding box to do modifications!")
        self.parent().image_label.change_paint_type(PaintType.BOX_PICKER)

    def on_manual_polygon(self):
        # Sets emphasis on the button
        self.manual_polygon.setProperty("active", True)
        self.parent().image_label.change_paint_type(PaintType.POLYGON)
            
    def on_manual_bounding_box(self):
        self.parent().info_label.setText("Manual bounding box annotation!")
        self.parent().image_label.change_paint_type(PaintType.BOX_MANUAL)

    def on_add_point(self):
        self.parent().info_label.setText("Adding point annotation!")
        self.parent().image_label.change_paint_type(PaintType.POINT)

    def on_add_box(self):
        self.parent().info_label.setText("Adding box annotation!")
        self.parent().image_label.change_paint_type(PaintType.BOX)

    def on_zoom_rectangle(self):
        if self.zoom_flag:
            self.parent().info_label.setText("Zooming rectangle OFF!")
            self.parent().image_label.change_paint_type(PaintType.POINT)
            self.parent().annotator.zoomed_bounding_box = None
            self.parent().annotator.make_embedding()
            self.parent().update(self.parent().annotator.merge_image_visualization())
            self.zoom_flag = False
        else:
            self.parent().info_label.setText("Pick Mask to zoom!")
            self.zoom_rectangle.setText(f"Zoom Rectangle [ {self.config.key_mapping.ZOOM_RECTANGLE.name} ]")
            self.parent().image_label.change_paint_type(PaintType.ZOOM_PICKER)
            self.zoom_flag = True

    def on_annotate_all(self):
        self.parent().info_label.setText("Annotating all. This make take some time.")
        self.parent().annotator.predict_all(self.annotation_settings.get_values())
        self.parent().update(self.parent().annotator.merge_image_visualization())
        self.parent().info_label.setText("Annotate all finished.")

    def on_cancel_annotation(self):
        self.parent().image_label.clear()
        self.parent().annotator.clear_last_masks()
        self.parent().update(self.parent().annotator.merge_image_visualization())

    def on_save_annotation(self):
        if self.parent().image_label.paint_type == PaintType.BOX_MANUAL:
            label = self.label_picker.currentItem().text()
            bounding_box = BoundingBox.from_drawn_bounding_box(
                self.parent().image_label.bounding_box,
                label,
                self.config.window_size[0],
                self.config.window_size[1]
            )
            self.parent().annotator.annotations.append_bounding_box(bounding_box)
        else:
            if self.parent().image_label.paint_type == PaintType.POLYGON:
                self.parent().annotator.last_mask = self.parent().image_label.polygon.to_mask(
                    self.config.window_size[0], self.config.window_size[1])
            self.parent().annotator.save_mask(label=self.label_picker.currentItem().text())

        self.parent().update(self.parent().annotator.merge_image_visualization())
        self.parent().image_label.clear()
         