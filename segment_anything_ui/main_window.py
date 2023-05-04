import sys

import numpy as np
import torch
from PySide6.QtWidgets import (QApplication, QGridLayout, QLabel,
                               QMessageBox, QWidget)
from segment_anything import sam_model_registry

from segment_anything_ui.annotator import Annotator
from segment_anything_ui.annotation_layout import AnnotationLayout
from segment_anything_ui.config import Config
from segment_anything_ui.draw_label import DrawLabel
from segment_anything_ui.image_pixmap import ImagePixmap
from segment_anything_ui.settings_layout import SettingsLayout


class SegmentAnythingUI(QWidget):

    def __init__(self, config) -> None:
        super().__init__()
        self.config: Config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.setWindowTitle("Segment Anything UI")
        # self.setGeometry(100, 100, 800, 600)
        self.layout = QGridLayout(self)
        self.image_label = DrawLabel(self)
        self.settings = SettingsLayout(self)
        self.info_label = QLabel("Information about running process.")
        self.sam = self.init_sam()
        self.annotator = Annotator(sam=self.sam, parent=self)
        self.annotation_layout = AnnotationLayout(self, config=self.config)
        self.layout.addWidget(self.annotation_layout, 0, 0)
        self.layout.addWidget(self.image_label, 0, 1)
        self.layout.addWidget(self.settings, 0, 2)
        self.layout.addWidget(self.info_label, 1, 1)

        self.set_image(np.zeros((self.config.window_size, self.config.window_size, 3), dtype=np.uint8))
        self.show()

    def set_image(self, image: np.ndarray):
        self.annotator.set_image(image).make_embedding()
        self.annotator.clear()
        self.update(image)

    def update(self, image: np.ndarray):
        pixmap = ImagePixmap()
        pixmap.set_image(image)
        print("Updating image")
        self.image_label.setPixmap(pixmap)

    def init_sam(self):
        try:
            sam = sam_model_registry[self.config.get_model_name()](checkpoint=str(self.settings.checkpoint_path.text()))
            sam.to(device=self.device)
        except Exception as e:
            print(e)
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