import logging
import sys

import cv2
import numpy as np
import torch
from PySide6.QtWidgets import (QApplication, QGridLayout, QLabel,
                               QMessageBox, QWidget)
from PySide6.QtCore import Qt

from segment_anything_ui.annotator import Annotator
from segment_anything_ui.annotation_layout import AnnotationLayout
from segment_anything_ui.config import Config
from segment_anything_ui.draw_label import DrawLabel
from segment_anything_ui.image_pixmap import ImagePixmap
from segment_anything_ui.model_builder import build_model
from segment_anything_ui.settings_layout import SettingsLayout


class SegmentAnythingUI(QWidget):

    def __init__(self, config) -> None:
        super().__init__()
        self.config: Config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.setWindowTitle("Segment Anything UI")
        self.setWindowState(Qt.WindowState.WindowMaximized)
        # self.setGeometry(100, 100, 800, 600)
        self.layout = QGridLayout(self)
        self.image_label = DrawLabel(self)
        self.settings = SettingsLayout(self, config=self.config)
        self.info_label = QLabel("Information about running process.")
        self.sam = self.init_sam()
        self.annotator = Annotator(sam=self.sam, parent=self)
        self.annotation_layout = AnnotationLayout(self, config=self.config)
        self.layout.addWidget(self.annotation_layout, 0, 0, 1, 1, Qt.AlignCenter)
        self.layout.addWidget(self.image_label, 0, 1, 1, 1, Qt.AlignCenter)
        self.layout.addWidget(self.settings, 0, 3, 1, 1, Qt.AlignCenter)
        self.layout.addWidget(self.info_label, 1, 1, Qt.AlignBottom)

        self.set_image(np.zeros((self.config.window_size[1], self.config.window_size[0], 3), dtype=np.uint8))
        self.show()

    def set_image(self, image: np.ndarray):
        self.annotator.set_image(image).make_embedding()
        self.annotator.clear()
        self.update(image)

    def update(self, image: np.ndarray):
        image = cv2.resize(image, self.config.window_size)
        pixmap = ImagePixmap()
        pixmap.set_image(image)
        print("Updating image")
        self.image_label.setPixmap(pixmap)

    def init_sam(self):
        try:
            checkpoint_path = str(self.settings.checkpoint_path.text())
            sam = build_model(self.config.get_sam_model_name(), checkpoint_path, self.device)
        except Exception as e:
            logging.getLogger().exception(f"Error loading model: {e}")
            QMessageBox.critical(self, "Error", "Could not load model")
            return None
        return sam

    def get_mask(self):
        return self.annotator.make_instance_mask()

    def get_labels(self):
        return self.annotator.make_labels()


if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = SegmentAnythingUI(Config())
    sys.exit(app.exec())