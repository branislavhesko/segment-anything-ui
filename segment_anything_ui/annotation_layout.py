from typing import Callable
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton

from segment_anything_ui.draw_label import PaintType
from segment_anything_ui.annotator import AutomaticMaskGeneratorSettings, CustomForm


class AnnotationLayout(QWidget):

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignTop)
        self.add_point = QPushButton("Add Point")
        self.add_box = QPushButton("Add Box")
        self.annotate_all = QPushButton("Annotate All")
        self.manual_polygon = QPushButton("Manual Polygon")
        self.cancel_annotation = QPushButton("Cancel Annotation")
        self.save_annotation = QPushButton("Save Annotation")
        self.pick_mask = QPushButton("Pick Mask")
        self.save_annotation.setShortcut("N")
        self.annotation_settings = CustomForm(self, AutomaticMaskGeneratorSettings())
        self.layout.addWidget(self.add_point)
        self.layout.addWidget(self.add_box)
        self.layout.addWidget(self.annotate_all)
        self.layout.addWidget(self.pick_mask)
        self.layout.addWidget(self.cancel_annotation)
        self.layout.addWidget(self.save_annotation)
        self.layout.addWidget(self.manual_polygon)
        self.layout.addWidget(self.annotation_settings)
        self.add_point.clicked.connect(self.on_add_point)
        self.add_box.clicked.connect(self.on_add_box)
        self.annotate_all.clicked.connect(self.on_annotate_all)
        self.cancel_annotation.clicked.connect(self.on_cancel_annotation)
        self.save_annotation.clicked.connect(self.on_save_annotation)
        self.pick_mask.clicked.connect(self.on_pick_mask)
        self.manual_polygon.clicked.connect(self.on_manual_polygon)

    def on_pick_mask(self):
        self.parent().image_label.change_paint_type(PaintType.MASK_PICKER)

    def on_manual_polygon(self):
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
