import sys

import numpy as np
from PySide6 import QtCore, QtWidgets
from PySide6.QtWidgets import QApplication, QGridLayout, QMainWindow, QFileDialog, QMessageBox, QWidget
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch

from segment_anything_ui.annotation_layout import AnnotationLayout
from segment_anything_ui.image_pixmap import ImagePixmap
from segment_anything_ui.settings_layout import SettingsLayout


class SegmentAnythingUI(QWidget):

    def __init__(self) -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.setWindowTitle("Segment Anything UI")
        self.setGeometry(100, 100, 800, 600)
        self.layout = QGridLayout(self)
        self.image_label = QtWidgets.QLabel(self)
        self.settings = SettingsLayout(self)
        self.annotation_layout = AnnotationLayout(self)
        self.layout.addWidget(self.annotation_layout, 0, 0)
        self.layout.addWidget(self.image_label, 0, 1)
        self.layout.addWidget(self.settings, 0, 2)
        self.update(np.zeros((512, 512, 3), dtype=np.uint8))
        self.show()
        self.sam = self.init_sam()

    def update(self, image: np.ndarray):
        pixmap = ImagePixmap()
        pixmap.set_image(image)
        print("Updating image")
        self.image_label.setPixmap(pixmap)

    def init_sam(self):
        sam = sam_model_registry["vit_h"](checkpoint=str(self.settings.checkpoint_path.text()))
        sam.to(device=self.device)
        return sam



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SegmentAnythingUI()
    sys.exit(app.exec())